use std::collections::{BTreeSet, HashSet};
use std::io::{self, BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::backup::create_database_backup;
use crate::code_assist::{
    CodeAssistReportInput, CodeEvalCaseInput, apply_code_memory_suggestions,
    build_code_assist_report, build_code_pattern_report, build_code_review_plan_report,
    code_index_guard_from_freshness, deterministic_reason_report, evaluate_code_cases,
};
use crate::code_index::{check_code_index_freshness, index_project};
use crate::code_reason::{CodeReasonTask, reason_about_search, reason_about_symbol};
use crate::compaction::{CompactionProposal, propose_compaction};
use crate::config::{Config, model_name_matches};
use crate::context_pack::{
    ProjectContextFormat as TaskProjectContextFormat, build_memory_fragments,
    code_context_summaries, code_memory_summaries, context_memory_summaries, format_task_context,
    fragment_memory_ids, merge_core_and_task_memories,
};
use crate::context_plan::{ContextPlanRequest, estimate_context_tokens, plan_context_access};
use crate::devsystem::{AgentRunReport, DevsystemRequest, IndexRunSummary, build_devsystem_report};
use crate::embedding::{embed_indexed_code_symbols, embed_memory, embed_missing};
use crate::extract::{MemoryCandidate, extract_memory_candidates, prepare_extraction_input};
use crate::graph_extract::{apply_graph_extraction, extract_memory_graph};
use crate::lsif_index::{generate_and_import_rust_analyzer_lsif, import_rust_analyzer_lsif};
use crate::maintenance::{MaintenanceOptions, run_maintenance};
use crate::mcp_format::{
    CodeExploreFormat, format_code_assist_text, format_code_explore, format_code_files,
    format_code_memories, format_code_outline, format_code_patterns_text,
    format_code_reason_report, format_code_results, format_code_symbol, format_memories,
    format_memory, format_relations, memory_eval_text,
};
use crate::project::{resolve_project_id, resolve_project_id_from_path};
use crate::retrieval::RetrievalMode as SearchMode;
use crate::retrieval::{
    ContextRetrievalOutput, ContextRetrievalRequest, RetrievalDiagnostics, RetrievalMode,
    format_retrieval_diagnostics, run_context_retrieval,
};
use crate::search::{
    CodeSearchRequest, MemorySearchRequest, code_model_for_role, ollama_from_config,
    search_code_tuple as search_code, search_memories_tuple as search_memories,
};
use crate::semantic_ops::{SemanticOperationRequest, run_semantic_operation};
use crate::store::{
    CORE_MEMORY_KINDS, CodeMemory, CodeMemorySearchOptions, CodeRelation, CodeSearchResult,
    CodeSimilarityPairOptions, DEFAULT_MEMORY_SCOPE, DEFAULT_MEMORY_TIER, EvalRunRecord,
    ListOptions, MEMORY_SCOPES, Memory, MemoryGraph, MemoryStatus, NewCodeMemory, NewMemory,
    NewTaskSession, ProjectExport, ProjectProfileUpdate, RememberOutcome, RetrievalEventRecord,
    SearchOptions, StatusFilter, Store, TaskSessionUpdate,
};
use crate::validation::{ValidationAction, validate_memories};

const PROTOCOL_VERSION: &str = "2025-06-18";
const HTTP_HEADER_LIMIT_BYTES: usize = 64 * 1024;
const HTTP_BODY_LIMIT_BYTES: usize = 5 * 1024 * 1024;
const HTTP_IO_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct McpSmokeReport {
    pub project_id: String,
    pub database_marker: PathBuf,
    pub tools_count: usize,
    pub remembered_id: String,
    pub search_hits: usize,
    pub context_hits: usize,
    pub cross_project_hits: usize,
    pub total_memories: u64,
    pub active_memories: u64,
}

#[derive(Debug, Clone)]
pub struct McpHttpSmokeReport {
    pub address: String,
    pub initialize_status: u16,
    pub sse_status: u16,
    pub forbidden_origin_status: u16,
    pub wrong_method_status: u16,
    pub oversized_body_status: u16,
    pub server_info_name: String,
}

pub async fn run_stdio(config: Config) -> Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();

    for line in stdin.lock().lines() {
        let line = line.context("failed to read MCP stdin")?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_message(
                    &mut stdout,
                    json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": {
                            "code": -32700,
                            "message": format!("Parse error: {error}")
                        }
                    }),
                )?;
                continue;
            }
        };

        if let Some(response) = handle_message(&config, request).await {
            write_message(&mut stdout, response)?;
        }
    }

    Ok(())
}

pub async fn run_http(config: Config, host: &str, port: u16) -> Result<()> {
    if !matches!(host, "127.0.0.1" | "localhost" | "::1") {
        bail!("dukememory MCP HTTP endpoint must bind to localhost");
    }
    let listener = TcpListener::bind((host, port))
        .with_context(|| format!("failed to bind MCP HTTP endpoint on {host}:{port}"))?;
    eprintln!("dukememory MCP HTTP listening on http://{host}:{port}");
    for stream in listener.incoming() {
        let stream = stream.context("failed to accept MCP HTTP connection")?;
        let config = config.clone();
        thread::spawn(move || {
            if let Err(error) = handle_http_stream(config, stream) {
                eprintln!("dukememory MCP HTTP connection failed: {error:#}");
            }
        });
    }
    Ok(())
}

fn handle_http_stream(config: Config, mut stream: TcpStream) -> Result<()> {
    stream
        .set_read_timeout(Some(HTTP_IO_TIMEOUT))
        .context("failed to set MCP HTTP read timeout")?;
    stream
        .set_write_timeout(Some(HTTP_IO_TIMEOUT))
        .context("failed to set MCP HTTP write timeout")?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build MCP HTTP request runtime")?;
    if let Err(error) = runtime.block_on(handle_http_connection(&config, &mut stream)) {
        let body = json!({
            "jsonrpc": "2.0",
            "id": Value::Null,
            "error": {
                "code": -32603,
                "message": error.to_string()
            }
        });
        let _ = write_http_json(&mut stream, 500, &body);
    }
    Ok(())
}

async fn handle_http_connection(config: &Config, stream: &mut TcpStream) -> Result<()> {
    let mut buffer = Vec::new();
    let mut temp = [0_u8; 4096];
    loop {
        let read = stream
            .read(&mut temp)
            .context("failed to read MCP HTTP request")?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&temp[..read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if buffer.len() > HTTP_HEADER_LIMIT_BYTES {
            return write_http_text(stream, 431, "request headers too large");
        }
    }
    let Some(header_end) = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
    else {
        return write_http_text(stream, 400, "missing HTTP header terminator");
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = headers.lines();
    let Some(request_line) = lines.next() else {
        return write_http_text(stream, 400, "missing HTTP request line");
    };
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or("");
    let path = request_parts.next().unwrap_or("/");
    let mut content_length = 0_usize;
    let mut origin = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => match value.trim().parse::<usize>() {
                Ok(length) => content_length = length,
                Err(_) => return write_http_text(stream, 400, "invalid content-length"),
            },
            "origin" => origin = Some(value.trim().to_string()),
            _ => {}
        }
    }
    if let Some(origin) = origin
        && !origin.starts_with("http://127.0.0.1")
        && !origin.starts_with("http://localhost")
        && !origin.starts_with("http://[::1]")
    {
        return write_http_text(stream, 403, "forbidden origin");
    }
    if method == "GET" && path == "/mcp" {
        return write_http_sse(stream);
    }
    if path != "/mcp" {
        return write_http_text(stream, 404, "not found");
    }
    if method != "POST" {
        return write_http_text(stream, 405, "method not allowed");
    }
    if content_length > HTTP_BODY_LIMIT_BYTES {
        return write_http_text(stream, 413, "request body too large");
    }
    let mut body = buffer[header_end..].to_vec();
    while body.len() < content_length {
        let read = stream
            .read(&mut temp)
            .context("failed to read MCP HTTP body")?;
        if read == 0 {
            return write_http_text(stream, 400, "incomplete HTTP body");
        }
        body.extend_from_slice(&temp[..read]);
    }
    let body = &body[..content_length.min(body.len())];
    let request: Value = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => return write_http_text(stream, 400, &format!("invalid JSON: {error}")),
    };
    match handle_message(config, request).await {
        Some(response) => write_http_json(stream, 200, &response),
        None => write_http_text(stream, 202, ""),
    }
}

fn write_http_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    write_http_headers(stream, status, "application/json", body.len())?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn write_http_text(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    write_http_headers(stream, status, "text/plain; charset=utf-8", body.len())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_http_sse(stream: &mut TcpStream) -> Result<()> {
    let body = "event: endpoint\ndata: /mcp\n\n";
    write_http_headers(stream, 200, "text/event-stream", body.len())?;
    write!(stream, "Cache-Control: no-cache\r\n")?;
    stream.write_all(b"\r\n")?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_http_headers(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    content_length: usize,
) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {content_length}\r\nConnection: close\r\n",
        http_status_reason(status)
    )?;
    if content_type != "text/event-stream" {
        stream.write_all(b"\r\n")?;
    }
    Ok(())
}
fn http_status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

pub async fn run_smoke(config: &Config) -> Result<McpSmokeReport> {
    let token = uuid::Uuid::now_v7().to_string().replace('-', "");
    let project_id = format!("mcp-smoke-{token}");
    let other_project_id = format!("mcp-smoke-other-{token}");
    let root = std::env::temp_dir().join(format!("dukememory-mcp-smoke-{token}"));
    std::fs::create_dir_all(&root)
        .with_context(|| format!("failed to create smoke directory {}", root.display()))?;

    let mut smoke_config = config.clone();
    smoke_config.database_marker = root.join("schema.marker");

    let mut id = 1_u64;
    let initialize = smoke_request(&smoke_config, &mut id, "initialize", json!({})).await?;
    if initialize["serverInfo"]["name"] != "dukememory" {
        bail!("MCP initialize returned unexpected serverInfo: {initialize}");
    }

    let tools = smoke_request(&smoke_config, &mut id, "tools/list", json!({})).await?;
    let tool_names = tools["tools"]
        .as_array()
        .context("MCP tools/list response is missing tools array")?
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect::<Vec<_>>();
    for required in [
        "dukememory_remember",
        "dukememory_search",
        "dukememory_context",
        "dukememory_status",
    ] {
        if !tool_names.contains(&required) {
            bail!("MCP tools/list did not include required tool `{required}`");
        }
    }
    if tool_names
        .iter()
        .any(|name| !name.starts_with("dukememory_"))
    {
        bail!("MCP tools/list exposed a non-dukememory tool: {tool_names:?}");
    }

    let smoke_query = "mcp smoke scopecheck";
    let body = "Dukememory MCP smoke scopecheck memory. This must stay scoped to one project.";
    let remembered = smoke_tool(
        &smoke_config,
        &mut id,
        "dukememory_remember",
        json!({
            "project_id": project_id,
            "kind": "smoke",
            "status": "active",
            "source": "mcp_smoke",
            "body": body,
            "embed": false,
            "deduplicate": false
        }),
    )
    .await?;
    if remembered["inserted"].as_bool() != Some(true) {
        bail!("MCP remember did not insert smoke memory: {remembered}");
    }
    let remembered_id = remembered["id"]
        .as_str()
        .context("MCP remember response is missing id")?
        .to_string();

    let search = smoke_tool(
        &smoke_config,
        &mut id,
        "dukememory_search",
        json!({
            "project_id": project_id,
            "query": smoke_query,
            "mode": "keyword",
            "status": "active",
            "limit": 5
        }),
    )
    .await?;
    let search_results = search["results"]
        .as_array()
        .context("MCP search response is missing results array")?;
    let search_hits = search_results.len();
    if !search_results
        .iter()
        .any(|result| result["id"] == remembered_id)
    {
        bail!("MCP search did not return the remembered smoke memory: {search}");
    }

    let cross_project = smoke_tool(
        &smoke_config,
        &mut id,
        "dukememory_search",
        json!({
            "project_id": other_project_id,
            "query": smoke_query,
            "mode": "keyword",
            "status": "active",
            "limit": 5
        }),
    )
    .await?;
    let cross_project_hits = cross_project["results"]
        .as_array()
        .context("MCP cross-project search response is missing results array")?
        .len();
    if cross_project_hits != 0 {
        bail!("MCP search leaked smoke memory across project ids: {cross_project}");
    }

    let context = smoke_tool(
        &smoke_config,
        &mut id,
        "dukememory_context",
        json!({
            "project_id": project_id,
            "query": smoke_query,
            "mode": "keyword",
            "memory_limit": 5,
            "code_limit": 0
        }),
    )
    .await?;
    let context_results = context["memories"]
        .as_array()
        .context("MCP context response is missing memories array")?;
    let context_hits = context_results.len();
    if !context_results
        .iter()
        .any(|result| result["id"] == remembered_id)
    {
        bail!("MCP context did not include the remembered smoke memory: {context}");
    }

    let status = smoke_tool(
        &smoke_config,
        &mut id,
        "dukememory_status",
        json!({
            "project_id": project_id
        }),
    )
    .await?;
    let total_memories = status["total_memories"]
        .as_u64()
        .context("MCP status response is missing total_memories")?;
    let active_memories = status["active_memories"]
        .as_u64()
        .context("MCP status response is missing active_memories")?;
    if total_memories != 1 || active_memories != 1 {
        bail!("MCP status returned unexpected memory counts: {status}");
    }

    Ok(McpSmokeReport {
        project_id,
        database_marker: smoke_config.database_marker,
        tools_count: tool_names.len(),
        remembered_id,
        search_hits,
        context_hits,
        cross_project_hits,
        total_memories,
        active_memories,
    })
}

pub async fn run_http_smoke(config: &Config) -> Result<McpHttpSmokeReport> {
    let token = uuid::Uuid::now_v7().to_string().replace('-', "");
    let root = std::env::temp_dir().join(format!("dukememory-mcp-http-smoke-{token}"));
    std::fs::create_dir_all(&root)
        .with_context(|| format!("failed to create HTTP smoke directory {}", root.display()))?;

    let mut smoke_config = config.clone();
    smoke_config.database_marker = root.join("schema.marker");
    let listener =
        TcpListener::bind(("127.0.0.1", 0)).context("failed to bind HTTP smoke listener")?;
    let addr = listener
        .local_addr()
        .context("failed to read HTTP smoke listener address")?;
    let address = format!("http://{addr}");
    let server_config = smoke_config.clone();
    let server = thread::spawn(move || -> Result<()> {
        for _ in 0..5 {
            let (stream, _) = listener
                .accept()
                .context("failed to accept HTTP smoke request")?;
            handle_http_stream(server_config.clone(), stream)?;
        }
        Ok(())
    });

    let initialize_body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    })
    .to_string();
    let initialize = http_smoke_roundtrip(
        addr,
        &format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nOrigin: http://127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            initialize_body.len(),
            initialize_body
        ),
    )?;
    let initialize_json: Value = serde_json::from_str(&initialize.body)
        .context("HTTP smoke initialize response was not JSON")?;
    let server_info_name = initialize_json["result"]["serverInfo"]["name"]
        .as_str()
        .unwrap_or("")
        .to_string();
    if initialize.status != 200 || server_info_name != "dukememory" {
        bail!("HTTP MCP initialize smoke failed: {initialize:?}");
    }

    let sse = http_smoke_roundtrip(
        addr,
        &format!("GET /mcp HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"),
    )?;
    if sse.status != 200 || !sse.body.contains("data: /mcp") {
        bail!("HTTP MCP SSE endpoint smoke failed: {sse:?}");
    }

    let forbidden = http_smoke_roundtrip(
        addr,
        &format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nOrigin: https://example.com\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ),
    )?;
    let wrong_method = http_smoke_roundtrip(
        addr,
        &format!(
            "PUT /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ),
    )?;
    let oversized = http_smoke_roundtrip(
        addr,
        &format!(
            "POST /mcp HTTP/1.1\r\nHost: {addr}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            HTTP_BODY_LIMIT_BYTES + 1
        ),
    )?;
    if forbidden.status != 403 || wrong_method.status != 405 || oversized.status != 413 {
        bail!(
            "HTTP MCP negative smoke failed: forbidden={}, wrong_method={}, oversized={}",
            forbidden.status,
            wrong_method.status,
            oversized.status
        );
    }

    server
        .join()
        .map_err(|_| anyhow!("HTTP smoke server thread panicked"))??;

    Ok(McpHttpSmokeReport {
        address,
        initialize_status: initialize.status,
        sse_status: sse.status,
        forbidden_origin_status: forbidden.status,
        wrong_method_status: wrong_method.status,
        oversized_body_status: oversized.status,
        server_info_name,
    })
}

#[derive(Debug)]
struct HttpSmokeResponse {
    status: u16,
    body: String,
}

fn http_smoke_roundtrip(addr: std::net::SocketAddr, request: &str) -> Result<HttpSmokeResponse> {
    let mut stream = TcpStream::connect(addr).context("failed to connect to HTTP smoke server")?;
    stream
        .write_all(request.as_bytes())
        .context("failed to write HTTP smoke request")?;
    stream
        .shutdown(std::net::Shutdown::Write)
        .context("failed to finish HTTP smoke request")?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .context("failed to read HTTP smoke response")?;
    parse_http_smoke_response(&response)
}

fn parse_http_smoke_response(raw: &str) -> Result<HttpSmokeResponse> {
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((raw, ""));
    let status_line = head.lines().next().unwrap_or("");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .context("HTTP smoke response is missing status code")?
        .parse::<u16>()
        .context("HTTP smoke response status code was invalid")?;
    Ok(HttpSmokeResponse {
        status,
        body: body.to_string(),
    })
}

async fn smoke_request(
    config: &Config,
    id: &mut u64,
    method: &str,
    params: Value,
) -> Result<Value> {
    let request_id = *id;
    *id += 1;
    let response = handle_message(
        config,
        json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }),
    )
    .await
    .with_context(|| format!("MCP {method} did not return a response"))?;
    if let Some(error) = response.get("error") {
        bail!("MCP {method} failed: {error}");
    }
    Ok(response["result"].clone())
}

async fn smoke_tool(config: &Config, id: &mut u64, name: &str, arguments: Value) -> Result<Value> {
    let result = smoke_request(
        config,
        id,
        "tools/call",
        json!({
            "name": name,
            "arguments": arguments
        }),
    )
    .await?;
    result
        .get("structuredContent")
        .cloned()
        .with_context(|| format!("MCP tool `{name}` response is missing structuredContent"))
}

async fn handle_message(config: &Config, request: Value) -> Option<Value> {
    let id = request.get("id").cloned()?;
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let params = request.get("params").cloned().unwrap_or_else(|| json!({}));

    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
                "logging": {}
            },
            "serverInfo": {
                "name": "dukememory",
                "version": env!("CARGO_PKG_VERSION")
            },
            "instructions": concat!(
                "Use only dukememory_* tools for durable memory. ",
                "Default reads to active memories in the current project. ",
                "For code understanding and navigation, use the project-scoped Dukememory code graph first: ",
                "prefer dukememory_code_explore for structural code questions and code changes; use dukememory_prepare, dukememory_code_search, dukememory_read_symbol, ",
                "dukememory_find_callers, dukememory_find_callees, and dukememory_impact. ",
                "Pass project_path when the current workspace root is known. ",
                "Refresh stale indexes with dukememory_code_index and dukememory_code_lsif_index. ",
                "Automatic writes and extracted memories should normally be pending until promoted. ",
                "Never search other projects unless the user explicitly asks for cross-project memory. ",
                "Never store secrets, credentials, API keys, or transient scratch thoughts."
            )
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => call_tool(config, params).await,
        "resources/list" => Ok(json!({ "resources": resources_list() })),
        "resources/templates/list" => Ok(json!({ "resourceTemplates": resource_templates_list() })),
        "resources/read" => resource_read(config, params),
        "prompts/list" => Ok(json!({ "prompts": prompts_list() })),
        "prompts/get" => prompt_get(params),
        "logging/setLevel" => Ok(json!({})),
        _ => Err(json!({
            "code": -32601,
            "message": format!("Method not found: {method}")
        })),
    };

    Some(match result {
        Ok(result) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result
        }),
        Err(error) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": error
        }),
    })
}

fn resources_list() -> Value {
    json!([
        {
            "uri": "dukememory://ontology",
            "name": "dukememory_ontology",
            "description": "Universal memory kinds, scopes, and domain examples.",
            "mimeType": "application/json"
        },
        {
            "uri": "dukememory://health",
            "name": "dukememory_health",
            "description": "Current storage metrics and database health.",
            "mimeType": "application/json"
        },
        {
            "uri": "dukememory://project/{project_id}/profile",
            "name": "dukememory_project_profile_template",
            "description": "Project profile resource. Replace {project_id} with an explicit project id.",
            "mimeType": "application/json"
        }
    ])
}

fn resource_templates_list() -> Value {
    json!([
        {
            "uriTemplate": "dukememory://project/{project_id}/profile",
            "name": "dukememory_project_profile",
            "description": "Read one explicit project profile by project id.",
            "mimeType": "application/json"
        }
    ])
}

fn prompts_list() -> Value {
    json!([
        {
            "name": "dukememory_agent_before",
            "description": "Start a coding task with project-scoped memory and code context.",
            "arguments": [
                {"name": "query", "description": "Current task or question.", "required": true},
                {"name": "project_path", "description": "Workspace root path.", "required": false}
            ]
        },
        {
            "name": "dukememory_agent_after",
            "description": "Extract durable pending memories after a task.",
            "arguments": [
                {"name": "summary", "description": "Task outcome or transcript summary.", "required": true},
                {"name": "project_path", "description": "Workspace root path.", "required": false}
            ]
        },
        {
            "name": "dukememory_agent_workflow",
            "description": "Run the full prepare, reason, patch, and remember loop for a coding task.",
            "arguments": [
                {"name": "query", "description": "Current coding task.", "required": true},
                {"name": "project_path", "description": "Workspace root path.", "required": false}
            ]
        },
        {
            "name": "dukememory_memory_review",
            "description": "Review pending memories before promotion.",
            "arguments": [
                {"name": "project_path", "description": "Workspace root path.", "required": false}
            ]
        },
        {
            "name": "dukememory_code_risk",
            "description": "Analyze likely code risks for a proposed change.",
            "arguments": [
                {"name": "query", "description": "Proposed change.", "required": true},
                {"name": "project_path", "description": "Workspace root path.", "required": false}
            ]
        }
    ])
}

fn prompt_get(params: Value) -> std::result::Result<Value, Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_params("prompts/get requires params.name"))?;
    let arguments = params
        .get("arguments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let arg = |key: &str| {
        arguments
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let text = match name {
        "dukememory_agent_before" => format!(
            "Call dukememory_prepare with query `{}` and project_path `{}` before editing. Use returned memory/code hits before broad filesystem search; pass debug=true only when full trace is needed.",
            arg("query"),
            arg("project_path")
        ),
        "dukememory_agent_after" => format!(
            "Call dukememory_extract with source `dukememory_agent_after`, input equal to this task summary, and project_path `{}`. Keep automatic writes pending unless validation is explicitly requested.\n\nSummary:\n{}",
            arg("project_path"),
            arg("summary")
        ),
        "dukememory_agent_workflow" => format!(
            "Run this Dukememory agent workflow for task `{}` in project_path `{}`:\n1. PREPARE: call dukememory_prepare with auto_index=true, mode=hybrid, and the task query. Read memory fragments, retrieval diagnostics, indexed code hits, and code_neighborhood before editing.\n2. REASON: start from code graph neighbors, callers, callees, and active memories. Use dukememory_code_explore or dukememory_read_symbol for missing symbols before broad filesystem search.\n3. PATCH: edit only after the relevant symbols/files are identified, then run focused tests and broaden tests when shared behavior changed.\n4. REMEMBER: after verification, call dukememory_agent_after with a concise task summary. Use dukememory_code_memory action=remember for durable symbol/file notes. Keep automatic writes pending unless the user asks to apply policy.",
            arg("query"),
            arg("project_path")
        ),
        "dukememory_memory_review" => format!(
            "Call dukememory_review for project_path `{}`. Promote only stable, reusable project facts; archive duplicates, secrets, transient status, and speculation.",
            arg("project_path")
        ),
        "dukememory_code_risk" => format!(
            "Call dukememory_code_risk with query `{}` and project_path `{}`. Ground findings in indexed symbols, callers/callees, and active memories.",
            arg("query"),
            arg("project_path")
        ),
        other => {
            return Err(invalid_params(&format!(
                "unknown dukememory prompt `{other}`"
            )));
        }
    };
    Ok(json!({
        "description": name,
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": text
                }
            }
        ]
    }))
}

fn resource_read(config: &Config, params: Value) -> std::result::Result<Value, Value> {
    let uri = params
        .get("uri")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_params("resources/read requires params.uri"))?;
    let value = match uri {
        "dukememory://ontology" => ontology_json(),
        "dukememory://health" => {
            let store = Store::open(&config.database_marker).map_err(tool_error)?;
            let health = store.health().map_err(tool_error)?;
            json!({
                "database_url": config.database_url,
                "database_marker": config.database_marker,
                "health": health
            })
        }
        _ => {
            if let Some(project_id) = parse_project_profile_resource_uri(uri) {
                let store = Store::open(&config.database_marker).map_err(tool_error)?;
                let profile = store.project_profile(project_id).map_err(tool_error)?;
                json!({
                    "project": profile
                })
            } else {
                return Err(invalid_params(&format!(
                    "unknown dukememory resource uri `{uri}`"
                )));
            }
        }
    };
    let text = serde_json::to_string_pretty(&value).map_err(|error| tool_error(error.into()))?;
    Ok(json!({
        "contents": [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": text
            }
        ]
    }))
}

fn parse_project_profile_resource_uri(uri: &str) -> Option<&str> {
    uri.strip_prefix("dukememory://project/")
        .and_then(|value| value.strip_suffix("/profile"))
        .filter(|project_id| !project_id.is_empty() && !project_id.contains('/'))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuditMode {
    Off,
    WritesOnly,
    All,
}

impl AuditMode {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Ok(Self::Off),
            "writes_only" | "writes-only" | "writes" | "mutations" => Ok(Self::WritesOnly),
            "all" | "on" | "enabled" => Ok(Self::All),
            other => Err(anyhow!(
                "invalid DUKEMEMORY_AUDIT_MODE `{other}`; use off, writes_only, or all"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::WritesOnly => "writes_only",
            Self::All => "all",
        }
    }
}

fn configured_audit_mode() -> AuditMode {
    std::env::var("DUKEMEMORY_AUDIT_MODE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| AuditMode::parse(&value))
        .transpose()
        .unwrap_or_else(|error| {
            eprintln!("warning: {error}; using DUKEMEMORY_AUDIT_MODE=all");
            Some(AuditMode::All)
        })
        .unwrap_or(AuditMode::All)
}

fn should_audit_tool_call(mode: AuditMode, name: &str) -> bool {
    match mode {
        AuditMode::Off => false,
        AuditMode::All => true,
        AuditMode::WritesOnly => is_mutating_tool(name),
    }
}

fn is_mutating_tool(name: &str) -> bool {
    matches!(
        name,
        "dukememory_extract"
            | "dukememory_agent_after"
            | "dukememory_eval"
            | "dukememory_prepare"
            | "dukememory_agent_before"
            | "dukememory_agent_task"
            | "dukememory_devsystem"
            | "dukememory_task_session"
            | "dukememory_task_eval"
            | "dukememory_test_plan"
            | "dukememory_graph"
            | "dukememory_graph_extract"
            | "dukememory_episode"
            | "dukememory_remember"
            | "dukememory_remember_smart"
            | "dukememory_review_apply"
            | "dukememory_validate_pending"
            | "dukememory_promote"
            | "dukememory_supersede"
            | "dukememory_archive"
            | "dukememory_prune_pending"
            | "dukememory_compact"
            | "dukememory_maintenance"
            | "dukememory_cleanup_schemas"
            | "dukememory_backup"
            | "dukememory_import"
            | "dukememory_code_index"
            | "dukememory_code_lsif_index"
            | "dukememory_embed_missing"
    )
}

fn audit_metrics(events: &[crate::store::AuditEvent]) -> Value {
    fn increment(map: &mut serde_json::Map<String, Value>, key: &str) {
        let next = map.get(key).and_then(Value::as_u64).unwrap_or(0) + 1;
        map.insert(key.to_string(), json!(next));
    }

    let mut by_actor = serde_json::Map::new();
    let mut by_action = serde_json::Map::new();
    let mut by_target_type = serde_json::Map::new();
    for event in events {
        increment(&mut by_actor, &event.actor);
        increment(&mut by_action, &event.action);
        increment(&mut by_target_type, &event.target_type);
    }
    json!({
        "total": events.len(),
        "by_actor": by_actor,
        "by_action": by_action,
        "by_target_type": by_target_type
    })
}

fn audit_mcp_tool_call(
    config: &Config,
    name: &str,
    arguments: &Value,
    response: &Value,
) -> Result<()> {
    let audit_mode = configured_audit_mode();
    if !should_audit_tool_call(audit_mode, name) {
        return Ok(());
    }
    let Ok(project_id) = resolve_project(arguments) else {
        return Ok(());
    };
    let store = Store::open(&config.database_marker)?;
    let structured = response
        .get("structuredContent")
        .cloned()
        .unwrap_or(Value::Null);
    let target_id = structured
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| structured.get("run_id").and_then(Value::as_str));
    store.record_audit_event(
        &project_id,
        "mcp",
        name,
        "tool_call",
        target_id,
        json!({
            "audit_mode": audit_mode.as_str(),
            "arguments": sanitize_audit_value(arguments),
            "result_keys": structured
                .as_object()
                .map(|object| object.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        }),
    )?;
    Ok(())
}

fn sanitize_audit_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sanitized = serde_json::Map::new();
            for (key, value) in object {
                let lower = key.to_ascii_lowercase();
                if lower.contains("password")
                    || lower.contains("secret")
                    || lower.contains("token")
                    || lower.contains("api_key")
                    || lower == "body"
                    || lower == "input"
                    || lower == "text_body"
                {
                    sanitized.insert(key.clone(), Value::String("<redacted>".to_string()));
                } else {
                    sanitized.insert(key.clone(), sanitize_audit_value(value));
                }
            }
            Value::Object(sanitized)
        }
        Value::Array(values) => Value::Array(values.iter().map(sanitize_audit_value).collect()),
        Value::String(text) if text.chars().count() > 500 => {
            Value::String(truncate_text(text, 500))
        }
        other => other.clone(),
    }
}

async fn call_tool(config: &Config, params: Value) -> std::result::Result<Value, Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_params("tools/call requires params.name"))?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = match name {
        "dukememory_extract" => tool_extract(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_models" => tool_models(config).await.map_err(tool_error),
        "dukememory_project_profile" => {
            tool_project_profile(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_task_session" => {
            tool_task_session(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_task_eval" => tool_task_eval(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_test_plan" => tool_test_plan(config, arguments.clone()).map_err(tool_error),
        "dukememory_ontology" => tool_ontology().map_err(tool_error),
        "dukememory_eval" => tool_eval(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_prepare" => tool_prepare(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_agent_before" => tool_prepare(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_agent_task" => tool_agent_task(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_devsystem" => tool_devsystem(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_agent_after" => tool_agent_after(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_context" => tool_context(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_context_plan" => {
            tool_context_plan(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_graph" => tool_graph(config, arguments.clone()).map_err(tool_error),
        "dukememory_episode" => tool_episode(config, arguments.clone()).map_err(tool_error),
        "dukememory_graph_extract" => tool_graph_extract(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_search" => tool_search(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_remember" => tool_remember(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_remember_smart" => tool_remember_smart(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_get" => tool_get(config, arguments.clone()).map_err(tool_error),
        "dukememory_list" => tool_list(config, arguments.clone()).map_err(tool_error),
        "dukememory_review" => tool_review(config, arguments.clone()).map_err(tool_error),
        "dukememory_review_apply" => {
            tool_review_apply(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_audit_log" => tool_audit_log(config, arguments.clone()).map_err(tool_error),
        "dukememory_validate_pending" => tool_validate_pending(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_promote" => tool_promote(config, arguments.clone()).map_err(tool_error),
        "dukememory_supersede" => tool_supersede(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_archive" => tool_archive(config, arguments.clone()).map_err(tool_error),
        "dukememory_prune_pending" => {
            tool_prune_pending(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_compact" => tool_compact(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_maintenance" => tool_maintenance(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_ops_pipeline" => tool_ops_pipeline(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_status" => tool_status(config, arguments.clone()).map_err(tool_error),
        "dukememory_health" => tool_health(config).map_err(tool_error),
        "dukememory_cleanup_schemas" => {
            tool_cleanup_schemas(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_backup" => tool_backup(config, arguments.clone()).map_err(tool_error),
        "dukememory_export" => tool_export(config, arguments.clone()).map_err(tool_error),
        "dukememory_import" => tool_import(config, arguments.clone()).map_err(tool_error),
        "dukememory_code_index" => tool_code_index(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_lsif_index" => {
            tool_code_lsif_index(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_code_status" => tool_code_status(config, arguments.clone()).map_err(tool_error),
        "dukememory_code_files" => tool_code_files(config, arguments.clone()).map_err(tool_error),
        "dukememory_code_outline" => {
            tool_code_outline(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_code_search" => tool_code_search(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_explore" => tool_code_explore(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_memory" => tool_code_memory(config, arguments.clone()).map_err(tool_error),
        "dukememory_code_affected" => {
            tool_code_affected(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_code_patterns" => tool_code_patterns(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_duplicates" => {
            tool_code_duplicates(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_code_assist" => tool_code_assist(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_review_plan" => {
            tool_code_review_plan(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_code_eval" => tool_code_eval(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_read_symbol" => tool_read_symbol(config, arguments.clone()).map_err(tool_error),
        "dukememory_code_brief" => tool_code_brief(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_plan" => tool_code_plan(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_code_risk" => tool_code_risk(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_find_callers" => {
            tool_find_callers(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_find_callees" => {
            tool_find_callees(config, arguments.clone()).map_err(tool_error)
        }
        "dukememory_impact" => tool_impact(config, arguments.clone()).map_err(tool_error),
        "dukememory_embed_missing" => tool_embed_missing(config, arguments.clone())
            .await
            .map_err(tool_error),
        "dukememory_semantic" => tool_semantic(config, arguments.clone(), None)
            .await
            .map_err(tool_error),
        "dukememory_dedupe" => tool_semantic(config, arguments.clone(), Some("dedupe"))
            .await
            .map_err(tool_error),
        "dukememory_related" => tool_semantic(config, arguments.clone(), Some("related"))
            .await
            .map_err(tool_error),
        "dukememory_semantic_review" => tool_semantic(config, arguments.clone(), Some("review"))
            .await
            .map_err(tool_error),
        "dukememory_semantic_route" => tool_semantic(config, arguments.clone(), Some("route"))
            .await
            .map_err(tool_error),
        "dukememory_semantic_clusters" => {
            tool_semantic(config, arguments.clone(), Some("clusters"))
                .await
                .map_err(tool_error)
        }
        "dukememory_semantic_tags" => tool_semantic(config, arguments.clone(), Some("tag"))
            .await
            .map_err(tool_error),
        "dukememory_stale_check" => tool_semantic(config, arguments.clone(), Some("stale"))
            .await
            .map_err(tool_error),
        "dukememory_consistency_check" => {
            tool_semantic(config, arguments.clone(), Some("consistency"))
                .await
                .map_err(tool_error)
        }
        "dukememory_eval_generate" => tool_semantic(config, arguments.clone(), Some("eval_cases"))
            .await
            .map_err(tool_error),
        "dukememory_hard_negatives" => {
            tool_semantic(config, arguments.clone(), Some("hard_negatives"))
                .await
                .map_err(tool_error)
        }
        "dukememory_embedding_health" => tool_semantic(config, arguments.clone(), Some("health"))
            .await
            .map_err(tool_error),
        "dukememory_model_migration" => tool_semantic(config, arguments.clone(), Some("migration"))
            .await
            .map_err(tool_error),
        "dukememory_isolation_check" => {
            tool_semantic(config, arguments.clone(), Some("isolation_check"))
                .await
                .map_err(tool_error)
        }
        "dukememory_memory_hints" => tool_semantic(config, arguments.clone(), Some("hints"))
            .await
            .map_err(tool_error),
        "dukememory_policy_decision" => tool_semantic(config, arguments.clone(), Some("policy"))
            .await
            .map_err(tool_error),
        "dukememory_retrieval_quality" => {
            tool_semantic(config, arguments.clone(), Some("retrieval_quality"))
                .await
                .map_err(tool_error)
        }
        "dukememory_auto_eval" => tool_semantic(config, arguments.clone(), Some("auto_eval"))
            .await
            .map_err(tool_error),
        "dukememory_ab_compare" => tool_semantic(config, arguments.clone(), Some("ab_compare"))
            .await
            .map_err(tool_error),
        "dukememory_lifecycle_review" => {
            tool_semantic(config, arguments.clone(), Some("lifecycle"))
                .await
                .map_err(tool_error)
        }
        "dukememory_code_memory_suggest" => {
            tool_semantic(config, arguments.clone(), Some("code_memory_suggest"))
                .await
                .map_err(tool_error)
        }
        "dukememory_verify_conflicts" => {
            tool_semantic(config, arguments.clone(), Some("verify_conflicts"))
                .await
                .map_err(tool_error)
        }
        "dukememory_topic_map" => tool_semantic(config, arguments.clone(), Some("topic_map"))
            .await
            .map_err(tool_error),
        "dukememory_budget_optimize" => {
            tool_semantic(config, arguments.clone(), Some("budget_optimize"))
                .await
                .map_err(tool_error)
        }
        "dukememory_feedback" => tool_semantic(config, arguments.clone(), Some("feedback"))
            .await
            .map_err(tool_error),
        "dukememory_self_heal" => tool_semantic(config, arguments.clone(), Some("self_heal"))
            .await
            .map_err(tool_error),
        "dukememory_outcome_learn" => {
            tool_semantic(config, arguments.clone(), Some("outcome_learn"))
                .await
                .map_err(tool_error)
        }
        "dukememory_conflict_graph" => {
            tool_semantic(config, arguments.clone(), Some("conflict_graph"))
                .await
                .map_err(tool_error)
        }
        "dukememory_memory_compiler" => {
            tool_semantic(config, arguments.clone(), Some("memory_compiler"))
                .await
                .map_err(tool_error)
        }
        "dukememory_policy_ab" => tool_semantic(config, arguments.clone(), Some("policy_ab"))
            .await
            .map_err(tool_error),
        "dukememory_context_policy" => {
            tool_semantic(config, arguments.clone(), Some("context_policy"))
                .await
                .map_err(tool_error)
        }
        "dukememory_trace" | "dukememory_task_replay" => {
            tool_semantic(config, arguments.clone(), Some("trace"))
                .await
                .map_err(tool_error)
        }
        "dukememory_counterfactual_eval" => {
            tool_semantic(config, arguments.clone(), Some("counterfactual_eval"))
                .await
                .map_err(tool_error)
        }
        "dukememory_code_causality" | "dukememory_memory_impact" => {
            tool_semantic(config, arguments.clone(), Some("causality"))
                .await
                .map_err(tool_error)
        }
        "dukememory_temporal_context" => {
            tool_semantic(config, arguments.clone(), Some("temporal_context"))
                .await
                .map_err(tool_error)
        }
        _ => Err(json!({
            "code": -32602,
            "message": format!("Unknown dukememory tool: {name}")
        })),
    };
    if let Ok(response) = &result {
        let _ = audit_mcp_tool_call(config, name, &arguments, response);
    }
    result
}

async fn tool_semantic(
    config: &Config,
    arguments: Value,
    forced_action: Option<&str>,
) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let action = forced_action
        .map(str::to_string)
        .or_else(|| optional_string(&arguments, "action"))
        .unwrap_or_else(|| "hints".to_string());
    let store = Store::open(&config.database_marker)?;
    let report = run_semantic_operation(
        config,
        &store,
        SemanticOperationRequest {
            project_id: project_id.clone(),
            action: action.clone(),
            query: optional_string(&arguments, "query"),
            body: optional_string(&arguments, "body"),
            memory_id: optional_string(&arguments, "memory_id")
                .or_else(|| optional_string(&arguments, "id")),
            symbol: optional_string(&arguments, "symbol"),
            file_path: optional_string(&arguments, "file_path"),
            project_path: optional_string(&arguments, "project_path"),
            input: optional_string(&arguments, "input"),
            other_project_id: optional_string(&arguments, "other_project_id"),
            expected_ids: optional_string_array(&arguments, "expected_ids")?,
            helpful_ids: optional_string_array(&arguments, "helpful_ids")?,
            unhelpful_ids: optional_string_array(&arguments, "unhelpful_ids")?,
            limit: optional_u64(&arguments, "limit")
                .unwrap_or(20)
                .clamp(1, 500) as usize,
            status: status_filter_arg(&arguments, MemoryStatus::Active)?,
            kind: optional_string(&arguments, "kind"),
            memory_tier: optional_string(&arguments, "memory_tier"),
            mode: RetrievalMode::parse(
                optional_string(&arguments, "mode")
                    .as_deref()
                    .unwrap_or("hybrid"),
            )?,
            min_similarity: arguments
                .get("min_similarity")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            target_memory_model: optional_string(&arguments, "target_memory_model"),
            target_code_model: optional_string(&arguments, "target_code_model"),
            as_of: optional_string(&arguments, "as_of"),
            retrieval_event_id: optional_string(&arguments, "retrieval_event_id"),
            outcome_kind: optional_string(&arguments, "outcome_kind"),
            severity: optional_string(&arguments, "severity"),
            apply: optional_bool(&arguments, "apply").unwrap_or(false),
        },
    )
    .await?;
    Ok(tool_result(
        format!("dukememory semantic `{action}` completed for project `{project_id}`"),
        report,
    ))
}

async fn tool_context(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let memory_limit = optional_u64(&arguments, "memory_limit")
        .unwrap_or(8)
        .clamp(1, 30) as usize;
    let core_memory_limit = optional_u64(&arguments, "core_memory_limit")
        .unwrap_or(5)
        .clamp(0, 10) as usize;
    let code_limit = optional_u64(&arguments, "code_limit")
        .unwrap_or(8)
        .clamp(0, 30) as usize;
    let token_budget = optional_u64(&arguments, "token_budget")
        .unwrap_or(3_000)
        .clamp(1_000, 30_000) as usize;
    let debug = optional_bool(&arguments, "debug").unwrap_or(false);
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let (text, structured) = build_context_payload(
        config,
        &store,
        BuildContextPayloadRequest {
            tool_name: "dukememory_context",
            task_session_id: None,
            project_id: &project_id,
            query: &query,
            memory_limit,
            core_memory_limit,
            code_limit,
            token_budget,
            mode,
            debug,
        },
    )
    .await?;

    Ok(tool_result(text, structured))
}

fn tool_context_plan(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let memory_limit = optional_u64(&arguments, "memory_limit")
        .unwrap_or(8)
        .clamp(1, 30) as usize;
    let core_memory_limit = optional_u64(&arguments, "core_memory_limit")
        .unwrap_or(5)
        .clamp(0, 10) as usize;
    let code_limit = optional_u64(&arguments, "code_limit")
        .unwrap_or(8)
        .clamp(0, 30) as usize;
    let token_budget = optional_u64(&arguments, "token_budget")
        .unwrap_or(3_000)
        .clamp(1_000, 30_000) as usize;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let status = store.status(&project_id)?;
    let code_status = store.code_status(&project_id)?;
    let plan = plan_context_access(ContextPlanRequest {
        query: &query,
        memory_limit,
        core_memory_limit,
        code_limit,
        token_budget,
    });
    let text = format!(
        "Dukememory context plan\nproject: {project_id}\ntask_type: {}\ntoken_budget: {}\nlimits: memory={} core={} code={} graph={} code_memories={}\nsources: memories={} graph={} code={} code_memories={} eval_history={}\nstatus: active_memories={} indexed_symbols={}",
        plan.task_type,
        plan.budget_plan.effective_token_budget,
        plan.memory_limit,
        plan.core_memory_limit,
        plan.code_limit,
        plan.graph_limit,
        plan.code_memory_limit,
        plan.source_plan.memories,
        plan.source_plan.memory_graph,
        plan.source_plan.code_index,
        plan.source_plan.code_memories,
        plan.source_plan.eval_history,
        status.active_memories,
        code_status.symbols
    );
    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "query": query,
            "plan": plan,
            "status": {
                "active_memories": status.active_memories,
                "pending_memories": status.pending_memories,
                "indexed_symbols": code_status.symbols,
                "code_symbol_embeddings": code_status.symbol_embeddings
            }
        }),
    ))
}

fn tool_project_profile(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let name = optional_string(&arguments, "name");
    let root_path = optional_string(&arguments, "root_path");
    let project_type = optional_string(&arguments, "project_type");
    let description = optional_string(&arguments, "description");
    let domains = optional_string_array(&arguments, "domains")?;
    let has_update = name.is_some()
        || root_path.is_some()
        || project_type.is_some()
        || description.is_some()
        || !domains.is_empty();
    let profile = if has_update {
        store.update_project_profile(
            &project_id,
            ProjectProfileUpdate {
                name,
                root_path,
                project_type,
                description,
                domains: if domains.is_empty() {
                    None
                } else {
                    Some(domains)
                },
            },
        )?
    } else {
        store.project_profile(&project_id)?
    };

    Ok(tool_result(
        format!(
            "dukememory project profile `{}` type={} domains={}",
            profile.id,
            profile.project_type,
            if profile.domains.is_empty() {
                "<none>".to_string()
            } else {
                profile.domains.join(",")
            }
        ),
        json!({
            "project": profile,
            "updated": has_update
        }),
    ))
}

fn tool_task_session(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let action = optional_string(&arguments, "action").unwrap_or_else(|| "list".to_string());
    let store = Store::open(&config.database_marker)?;
    match action.as_str() {
        "create" => {
            let query = optional_string(&arguments, "query")
                .ok_or_else(|| anyhow!("dukememory_task_session create requires query"))?;
            let status =
                optional_string(&arguments, "status").unwrap_or_else(|| "planned".to_string());
            let phase =
                optional_string(&arguments, "phase").unwrap_or_else(|| "planned".to_string());
            let progress = optional_u64(&arguments, "progress").unwrap_or(0).min(100) as usize;
            let result = arguments
                .get("result")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let session = store.create_task_session(NewTaskSession {
                project_id: &project_id,
                query: &query,
                status: &status,
                phase: &phase,
                progress,
                result,
            })?;
            Ok(tool_result(
                format!("Created dukememory task session `{}`.", session.id),
                json!({"action": "create", "session": session}),
            ))
        }
        "update" => {
            let id = optional_string(&arguments, "id")
                .ok_or_else(|| anyhow!("dukememory_task_session update requires id"))?;
            let update = TaskSessionUpdate {
                status: optional_string(&arguments, "status"),
                phase: optional_string(&arguments, "phase"),
                progress: optional_u64(&arguments, "progress").map(|value| value.min(100) as usize),
                memory_ids: optional_string_array_field(&arguments, "memory_ids")?,
                code_symbol_ids: optional_string_array_field(&arguments, "code_symbol_ids")?,
                file_paths: optional_string_array_field(&arguments, "file_paths")?,
                test_paths: optional_string_array_field(&arguments, "test_paths")?,
                summary: if arguments.get("summary").is_some() {
                    Some(optional_string(&arguments, "summary"))
                } else {
                    None
                },
                result: arguments.get("result").cloned(),
            };
            let session = store.update_task_session(&project_id, &id, update)?;
            Ok(tool_result(
                format!("Updated dukememory task session `{}`.", session.id),
                json!({"action": "update", "session": session}),
            ))
        }
        "get" => {
            let id = optional_string(&arguments, "id")
                .ok_or_else(|| anyhow!("dukememory_task_session get requires id"))?;
            let session = store.get_task_session(&project_id, &id)?;
            Ok(tool_result(
                format!("Read dukememory task session `{id}`."),
                json!({"action": "get", "session": session}),
            ))
        }
        "list" => {
            let status = optional_string(&arguments, "status")
                .filter(|value| value != "any" && !value.trim().is_empty());
            let limit = optional_u64(&arguments, "limit")
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            let sessions = store.list_task_sessions(&project_id, status.as_deref(), limit)?;
            Ok(tool_result(
                format!(
                    "Listed {} dukememory task sessions in project `{project_id}`.",
                    sessions.len()
                ),
                json!({"action": "list", "sessions": sessions}),
            ))
        }
        other => bail!(
            "invalid dukememory_task_session action `{other}`; use create, update, get, or list"
        ),
    }
}

async fn tool_task_eval(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let action = optional_string(&arguments, "action").unwrap_or_else(|| "build".to_string());
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 50) as usize;
    let store = Store::open(&config.database_marker)?;
    let session_id =
        optional_string(&arguments, "session_id").or_else(|| optional_string(&arguments, "id"));
    let session = session_id
        .as_deref()
        .map(|id| store.get_task_session(&project_id, id))
        .transpose()?
        .flatten();
    let query = optional_string(&arguments, "query")
        .or_else(|| session.as_ref().map(|session| session.query.clone()))
        .ok_or_else(|| anyhow!("dukememory_task_eval requires query or session_id"))?;
    let expected_ids = optional_string_array_field(&arguments, "expected_ids")?
        .or_else(|| session.as_ref().map(|session| session.memory_ids.clone()))
        .unwrap_or_default();
    let forbidden_ids = optional_string_array(&arguments, "forbidden_ids")?;
    let expected_contains = optional_string_array(&arguments, "expected_contains")?;
    let forbidden_contains = optional_string_array(&arguments, "forbidden_contains")?;
    let min_results = optional_u64(&arguments, "min_results").unwrap_or(1);
    let case = json!({
        "name": optional_string(&arguments, "name")
            .unwrap_or_else(|| format!("task-{}", short_task_eval_id(session_id.as_deref()))),
        "project_id": project_id,
        "query": query,
        "expected_ids": expected_ids,
        "forbidden_ids": forbidden_ids,
        "expected_contains": expected_contains,
        "forbidden_contains": forbidden_contains,
        "min_results": min_results
    });

    let hard_negative_report = run_semantic_operation(
        config,
        &store,
        SemanticOperationRequest {
            project_id: project_id.clone(),
            action: "hard_negatives".to_string(),
            query: Some(query.clone()),
            body: None,
            memory_id: None,
            symbol: None,
            file_path: None,
            project_path: optional_string(&arguments, "project_path"),
            input: None,
            other_project_id: None,
            expected_ids: expected_ids.clone(),
            helpful_ids: Vec::new(),
            unhelpful_ids: Vec::new(),
            limit,
            status: StatusFilter::One(MemoryStatus::Active),
            kind: None,
            memory_tier: None,
            mode: RetrievalMode::parse(mode.as_str())?,
            min_similarity: optional_f64(&arguments, "min_similarity").unwrap_or(0.0),
            target_memory_model: None,
            target_code_model: None,
            as_of: optional_string(&arguments, "as_of"),
            retrieval_event_id: None,
            outcome_kind: None,
            severity: None,
            apply: false,
        },
    )
    .await;
    let (hard_negatives, hard_negative_warning) = match hard_negative_report {
        Ok(report) => (report, None),
        Err(error) => (
            json!({
                "action": "hard_negatives",
                "project_id": project_id,
                "candidates": []
            }),
            Some(error.to_string()),
        ),
    };

    let eval = match action.as_str() {
        "build" => None,
        "run" => {
            let value = tool_eval(
                config,
                json!({
                    "project_id": project_id,
                    "cases": [case.clone()],
                    "limit": limit,
                    "mode": mode.as_str(),
                    "suite_name": optional_string(&arguments, "suite_name")
                        .unwrap_or_else(|| "task-session".to_string())
                }),
            )
            .await?;
            Some(value.get("structuredContent").cloned().unwrap_or(value))
        }
        other => bail!("invalid dukememory_task_eval action `{other}`; use build or run"),
    };

    let text = format!(
        "Dukememory task eval {action}\nproject: {project_id}\nquery: {query}\nexpected_ids: {}\nhard_negatives: {}\neval_run: {}",
        expected_ids.len(),
        hard_negatives["candidates"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0),
        eval.is_some()
    );
    Ok(tool_result(
        text,
        json!({
            "action": action,
            "project_id": project_id,
            "session_id": session_id,
            "case": case,
            "hard_negatives": hard_negatives,
            "hard_negative_warning": hard_negative_warning,
            "eval": eval
        }),
    ))
}

fn short_task_eval_id(session_id: Option<&str>) -> String {
    session_id
        .map(|id| {
            id.chars()
                .filter(|ch| ch.is_ascii_hexdigit())
                .take(8)
                .collect()
        })
        .filter(|id: &String| !id.is_empty())
        .unwrap_or_else(|| {
            uuid::Uuid::now_v7()
                .to_string()
                .chars()
                .filter(|ch| ch.is_ascii_hexdigit())
                .take(8)
                .collect()
        })
}

fn tool_test_plan(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let depth = optional_u64(&arguments, "depth").unwrap_or(5).clamp(1, 8) as usize;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(100)
        .clamp(1, 500) as usize;
    let mut files = optional_string_array(&arguments, "files")?;
    if let Some(file) = optional_string(&arguments, "file") {
        files.push(file);
    }
    if let Some(session_id) =
        optional_string(&arguments, "session_id").or_else(|| optional_string(&arguments, "id"))
        && let Some(session) = store.get_task_session(&project_id, &session_id)?
    {
        files.extend(session.file_paths);
    }
    let mut symbols = optional_string_array(&arguments, "symbols")?;
    if let Some(symbol) = optional_string(&arguments, "symbol") {
        symbols.push(symbol);
    }
    for symbol_ref in symbols {
        let symbol = store.resolve_code_symbol_reference(&project_id, &symbol_ref, None, None)?;
        let callers = store.find_callers(&project_id, &symbol.id, depth)?;
        let callees = store.find_callees(&project_id, &symbol.id, depth)?;
        files.extend(impacted_files_for_relations(
            &store,
            &project_id,
            &symbol.id,
            &callers,
            &callees,
        )?);
    }
    files = merge_unique_strings(&files, &[]);
    if files.is_empty() {
        bail!("dukememory_test_plan requires files, file, symbols, symbol, or session_id");
    }
    let affected_tests = store.affected_test_files(&project_id, &files, depth, limit)?;
    let commands = test_commands_for_files(&files, &affected_tests);
    let confidence = if !affected_tests.is_empty() {
        "high"
    } else if files.iter().any(|file| looks_like_source_file(file)) {
        "medium"
    } else {
        "low"
    };
    let text = format!(
        "Dukememory test plan\nproject: {project_id}\nchanged_files: {}\naffected_tests: {}\ncommands: {}",
        files.len(),
        affected_tests.len(),
        commands.len()
    );
    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "files": files,
            "depth": depth,
            "limit": limit,
            "affected_tests": affected_tests,
            "commands": commands,
            "confidence": confidence
        }),
    ))
}

fn looks_like_source_file(file: &str) -> bool {
    file.ends_with(".rs")
        || file.ends_with(".py")
        || file.ends_with(".js")
        || file.ends_with(".jsx")
        || file.ends_with(".ts")
        || file.ends_with(".tsx")
        || file.ends_with(".go")
        || file.ends_with(".java")
        || file.ends_with(".kt")
        || file.ends_with(".swift")
}

fn test_commands_for_files(files: &[String], tests: &[String]) -> Vec<Value> {
    let mut commands = BTreeSet::new();
    for test in tests {
        if test.ends_with(".rs") && test.starts_with("tests/") {
            let name = Path::new(test)
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("integration");
            commands.insert(format!("cargo test --test {name}"));
        } else if test.ends_with(".rs") {
            commands.insert("cargo test".to_string());
        } else if test.ends_with(".py") {
            commands.insert(format!("pytest {test}"));
        } else if test.ends_with(".js")
            || test.ends_with(".jsx")
            || test.ends_with(".ts")
            || test.ends_with(".tsx")
        {
            commands.insert(format!("npm test -- {test}"));
        } else if test.ends_with(".go") {
            commands.insert("go test ./...".to_string());
        }
    }
    if commands.is_empty() {
        if files.iter().any(|file| file.ends_with(".rs")) {
            commands.insert("cargo test".to_string());
        }
        if files.iter().any(|file| file.ends_with(".py")) {
            commands.insert("pytest".to_string());
        }
        if files.iter().any(|file| {
            file.ends_with(".js")
                || file.ends_with(".jsx")
                || file.ends_with(".ts")
                || file.ends_with(".tsx")
        }) {
            commands.insert("npm test".to_string());
        }
        if files.iter().any(|file| file.ends_with(".go")) {
            commands.insert("go test ./...".to_string());
        }
    }
    commands
        .into_iter()
        .map(|command| {
            json!({
                "command": command,
                "reason": "selected from impacted files and affected test paths"
            })
        })
        .collect()
}

async fn tool_agent_task(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let memory_limit = optional_u64(&arguments, "memory_limit")
        .unwrap_or(8)
        .clamp(1, 30) as usize;
    let core_memory_limit = optional_u64(&arguments, "core_memory_limit")
        .unwrap_or(5)
        .clamp(0, 10) as usize;
    let code_limit = optional_u64(&arguments, "code_limit")
        .unwrap_or(8)
        .clamp(0, 30) as usize;
    let token_budget = optional_u64(&arguments, "token_budget")
        .unwrap_or(3_000)
        .clamp(1_000, 30_000) as usize;
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let auto_index = optional_bool(&arguments, "auto_index").unwrap_or(true);
    let full_rebuild = optional_bool(&arguments, "full_rebuild").unwrap_or(false);
    let embed_symbols = optional_bool(&arguments, "embed_symbols").unwrap_or(false);
    let embed_symbol_limit = optional_u64(&arguments, "embed_symbol_limit")
        .unwrap_or(500)
        .clamp(1, 500) as usize;
    let include_code_assist = optional_bool(&arguments, "include_code_assist").unwrap_or(true);
    let include_consistency = optional_bool(&arguments, "include_consistency").unwrap_or(true);
    let code_assist_limit = optional_u64(&arguments, "code_assist_limit")
        .unwrap_or(8)
        .clamp(1, 25) as usize;
    let pattern_similarity = optional_f64(&arguments, "pattern_similarity").unwrap_or(0.72);
    let duplicate_similarity = optional_f64(&arguments, "duplicate_similarity").unwrap_or(0.92);
    let debug = optional_bool(&arguments, "debug").unwrap_or(false);
    let project_path = optional_string(&arguments, "project_path");

    let mut store = Store::open(&config.database_marker)?;
    let (project_id, index) = if auto_index {
        let path = project_path
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or(std::env::current_dir()?);
        let report = index_project(
            &mut store,
            &path,
            optional_string(&arguments, "project_id"),
            full_rebuild,
        )?;
        let embedded_symbols = if embed_symbols {
            embed_indexed_code_symbols(
                config,
                &store,
                &report.project_id,
                &report.indexed_files,
                embed_symbol_limit,
            )
            .await?
        } else {
            0
        };
        let project_id = report.project_id.clone();
        let index = json!({
            "enabled": true,
            "project_id": report.project_id,
            "root_path": report.root_path,
            "full_rebuild": report.full_rebuild,
            "files_seen": report.files_seen,
            "files_indexed": report.files_indexed,
            "files_skipped": report.files_skipped,
            "files_deleted": report.files_deleted,
            "indexed_files": report.indexed_files,
            "symbols_indexed": report.symbols_indexed,
            "relations_indexed": report.relations_indexed,
            "relation_targets_reset": report.relation_targets_reset,
            "calls_resolved": report.calls_resolved,
            "uses_resolved": report.uses_resolved,
            "modules_resolved": report.modules_resolved,
            "timing_ms": report.timing,
            "embed_symbols": embed_symbols,
            "embed_symbol_limit": embed_symbol_limit,
            "embedded_symbols": embedded_symbols
        });
        (project_id, index)
    } else {
        (
            resolve_project(&arguments)?,
            json!({
                "enabled": false,
                "reason": "auto_index disabled"
            }),
        )
    };

    let mut session = store.create_task_session(NewTaskSession {
        project_id: &project_id,
        query: &query,
        status: "running",
        phase: "prepare",
        progress: 10,
        result: json!({
            "index": index,
            "include_code_assist": include_code_assist,
            "include_consistency": include_consistency
        }),
    })?;

    let context_result = build_context_payload(
        config,
        &store,
        BuildContextPayloadRequest {
            tool_name: "dukememory_agent_task",
            task_session_id: Some(&session.id),
            project_id: &project_id,
            query: &query,
            memory_limit,
            core_memory_limit,
            code_limit,
            token_budget,
            mode,
            debug,
        },
    )
    .await;
    let (context_text, context_structured) = match context_result {
        Ok(result) => result,
        Err(error) => {
            mark_task_session_failed(&store, &project_id, &session.id, "context", &error)?;
            return Err(error.context("dukememory_agent_task failed while building context"));
        }
    };

    let mut artifacts = collect_agent_task_artifacts(&context_structured, None);
    let mut session_result =
        agent_task_session_result(&index, &context_structured, None, None, &artifacts);
    session = store.update_task_session(
        &project_id,
        &session.id,
        TaskSessionUpdate {
            status: Some("running".to_string()),
            phase: Some("context".to_string()),
            progress: Some(55),
            memory_ids: Some(artifacts.memory_ids.clone()),
            code_symbol_ids: Some(artifacts.code_symbol_ids.clone()),
            file_paths: Some(artifacts.file_paths.clone()),
            test_paths: Some(artifacts.test_paths.clone()),
            summary: Some(Some(format!(
                "Loaded context for agent task `{}`.",
                truncate_text(&query, 120)
            ))),
            result: Some(session_result.clone()),
        },
    )?;

    let code_assist_structured = if include_code_assist {
        let mut assist_arguments = json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "limit": code_assist_limit,
            "pattern_similarity": pattern_similarity,
            "duplicate_similarity": duplicate_similarity
        });
        if let Some(project_path) = &project_path {
            assist_arguments["project_path"] = json!(project_path);
        }
        let assist_result = tool_code_assist(config, assist_arguments).await;
        match assist_result {
            Ok(value) => value.get("structuredContent").cloned(),
            Err(error) => {
                mark_task_session_failed(&store, &project_id, &session.id, "code_assist", &error)?;
                return Err(
                    error.context("dukememory_agent_task failed while building code assist")
                );
            }
        }
    } else {
        None
    };

    if let Some(code_assist) = &code_assist_structured {
        artifacts.merge(collect_agent_task_artifacts(
            &context_structured,
            Some(code_assist),
        ));
    }
    let consistency_structured = if include_consistency {
        let consistency_result = run_semantic_operation(
            config,
            &store,
            SemanticOperationRequest {
                project_id: project_id.clone(),
                action: "consistency".to_string(),
                query: Some(query.clone()),
                body: None,
                memory_id: None,
                symbol: None,
                file_path: None,
                project_path: project_path.clone(),
                input: None,
                other_project_id: None,
                expected_ids: Vec::new(),
                helpful_ids: Vec::new(),
                unhelpful_ids: Vec::new(),
                limit: code_assist_limit.max(20),
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
                mode: RetrievalMode::parse(mode.as_str())?,
                min_similarity: 0.0,
                target_memory_model: None,
                target_code_model: None,
                as_of: optional_string(&arguments, "as_of"),
                retrieval_event_id: None,
                outcome_kind: None,
                severity: None,
                apply: false,
            },
        )
        .await;
        match consistency_result {
            Ok(value) => Some(value),
            Err(error) => {
                mark_task_session_failed(&store, &project_id, &session.id, "consistency", &error)?;
                return Err(
                    error.context("dukememory_agent_task failed while running consistency check")
                );
            }
        }
    } else {
        None
    };
    session_result = agent_task_session_result(
        &index,
        &context_structured,
        code_assist_structured.as_ref(),
        consistency_structured.as_ref(),
        &artifacts,
    );
    session = store.update_task_session(
        &project_id,
        &session.id,
        TaskSessionUpdate {
            status: Some("completed".to_string()),
            phase: Some("ready".to_string()),
            progress: Some(100),
            memory_ids: Some(artifacts.memory_ids.clone()),
            code_symbol_ids: Some(artifacts.code_symbol_ids.clone()),
            file_paths: Some(artifacts.file_paths.clone()),
            test_paths: Some(artifacts.test_paths.clone()),
            summary: Some(Some(format!(
                "Prepared agent task `{}` with {} memories, {} code symbols, {} files, and {} tests.",
                truncate_text(&query, 120),
                artifacts.memory_ids.len(),
                artifacts.code_symbol_ids.len(),
                artifacts.file_paths.len(),
                artifacts.test_paths.len()
            ))),
            result: Some(session_result),
        },
    )?;

    let text = format!(
        "Dukememory agent task ready\nproject: {project_id}\nsession: {}\nprogress: 100%\ncontext: {}\nartifacts: {} memories, {} code symbols, {} files, {} tests\ncode_assist: {}\nconsistency: {}",
        session.id,
        context_text.lines().next().unwrap_or("context loaded"),
        artifacts.memory_ids.len(),
        artifacts.code_symbol_ids.len(),
        artifacts.file_paths.len(),
        artifacts.test_paths.len(),
        include_code_assist,
        consistency_structured
            .as_ref()
            .and_then(|value| value["status"].as_str())
            .unwrap_or(if include_consistency {
                "unknown"
            } else {
                "skipped"
            })
    );
    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "query": query,
            "progress": 100,
            "phase": "ready",
            "session": session,
            "index": index,
            "context": context_structured,
            "code_assist": code_assist_structured,
            "consistency": consistency_structured,
            "artifacts": agent_task_artifacts_json(&artifacts)
        }),
    ))
}

async fn tool_devsystem(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let project_id = resolve_project(&arguments)?;
    let mut files = optional_string_array(&arguments, "files")?;
    if let Some(file) = optional_string(&arguments, "file") {
        files.push(file);
    }
    if files.is_empty() {
        bail!("dukememory_devsystem requires files or file");
    }
    let project_path = optional_path(&arguments, "project_path")?;
    let write_memory = optional_bool(&arguments, "write_memory").unwrap_or(true);
    let auto_index = optional_bool(&arguments, "auto_index").unwrap_or(true);
    let full_rebuild = optional_bool(&arguments, "full_rebuild").unwrap_or(false);
    let embed_symbols = optional_bool(&arguments, "embed_symbols").unwrap_or(false);
    let embed_symbol_limit = optional_u64(&arguments, "embed_symbol_limit")
        .unwrap_or(200)
        .clamp(1, 10_000) as usize;
    let run_evidence = optional_bool(&arguments, "run_evidence").unwrap_or(false);
    let evidence_timeout_seconds = optional_u64(&arguments, "evidence_timeout_seconds")
        .unwrap_or(120)
        .clamp(1, 3600);
    let max_evidence_commands = optional_u64(&arguments, "max_evidence_commands")
        .unwrap_or(5)
        .clamp(1, 50) as usize;
    let allowed_evidence_commands = optional_string_array(&arguments, "allowed_evidence_commands")?;
    let duplicate_similarity = optional_f64(&arguments, "duplicate_similarity").unwrap_or(0.92);
    let review_limit = optional_u64(&arguments, "review_limit")
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let policy_override = arguments.get("policy").cloned();
    let mut store = Store::open(&config.database_marker)?;
    let index_run = prepare_devsystem_index_run(
        config,
        &mut store,
        DevsystemIndexRunRequest {
            project_id: &project_id,
            project_path: project_path.as_deref(),
            auto_index,
            full_rebuild,
            embed_symbols,
            embed_symbol_limit,
        },
    )
    .await?;
    let report = build_devsystem_report(
        &mut store,
        DevsystemRequest {
            project_id: project_id.clone(),
            query: query.clone(),
            files,
            project_path,
            write_memory,
            auto_index,
            full_rebuild,
            embed_symbols,
            embed_symbol_limit,
            precomputed_index_run: Some(index_run),
            run_evidence,
            evidence_timeout_seconds,
            max_evidence_commands,
            allowed_evidence_commands,
            code_embedding_model: Some(config.code_embed_model().to_string()),
            duplicate_similarity,
            review_limit,
            policy_override,
        },
    )?;
    Ok(tool_result(
        format_devsystem_report(&report),
        json!({
            "project_id": project_id,
            "query": query,
            "report": report
        }),
    ))
}

struct DevsystemIndexRunRequest<'a> {
    project_id: &'a str,
    project_path: Option<&'a Path>,
    auto_index: bool,
    full_rebuild: bool,
    embed_symbols: bool,
    embed_symbol_limit: usize,
}

async fn prepare_devsystem_index_run(
    config: &Config,
    store: &mut Store,
    request: DevsystemIndexRunRequest<'_>,
) -> Result<IndexRunSummary> {
    if !request.auto_index {
        return Ok(IndexRunSummary::disabled(
            request.project_id.to_string(),
            request.project_path.map(Path::to_path_buf),
            "auto_index disabled by request",
            request.full_rebuild,
            request.embed_symbols,
            request.embed_symbol_limit,
        ));
    }
    let Some(root) = request.project_path else {
        return Ok(IndexRunSummary::disabled(
            request.project_id.to_string(),
            None,
            "auto_index requested but project_path was not supplied",
            request.full_rebuild,
            request.embed_symbols,
            request.embed_symbol_limit,
        ));
    };
    let index_report = index_project(
        store,
        root,
        Some(request.project_id.to_string()),
        request.full_rebuild,
    )?;
    let indexed_files = index_report.indexed_files.clone();
    let embedded_symbols = if request.embed_symbols {
        embed_indexed_code_symbols(
            config,
            store,
            &index_report.project_id,
            &indexed_files,
            request.embed_symbol_limit,
        )
        .await?
    } else {
        0
    };
    Ok(IndexRunSummary::from_index_report(
        index_report,
        request.embed_symbols,
        request.embed_symbol_limit,
        embedded_symbols,
    ))
}

fn format_devsystem_report(report: &AgentRunReport) -> String {
    let index_line = report
        .telemetry
        .index_run
        .as_ref()
        .map(|run| {
            if run.enabled {
                format!(
                    "auto_index=enabled files_indexed={} symbols_indexed={} embedded_symbols={}",
                    run.files_indexed, run.symbols_indexed, run.embedded_symbols
                )
            } else {
                format!(
                    "auto_index=disabled reason={}",
                    run.reason.as_deref().unwrap_or("unspecified")
                )
            }
        })
        .unwrap_or_else(|| "auto_index=unknown".to_string());
    let entropy = report
        .file_entropy_reports
        .iter()
        .map(|item| {
            format!(
                "{} score={} verdict={} responsibilities={}",
                item.file_path, item.score, item.verdict, item.responsibility_count
            )
        })
        .collect::<Vec<_>>()
        .join("\n- ");
    let evidence_summary = report
        .quality_evidence_reports
        .iter()
        .map(|evidence| format!("{}={}", evidence.command, evidence.status))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "dukedevsystem advisory report\nsession: {}\nreadiness: {}%\nfinal_verdict: {}\n{}\nquality_gate_status: {} blockers={} decisions={} warnings={}\nstages: {}\nmemory_writes: {} pending candidates, inserted={}, duplicates={}\nquality_evidence: {}\nrecommended_tests: {}\nrecommended_test_commands: {}\nfile_entropy:\n- {}",
        report.task_session_id,
        report.readiness_percent,
        report.final_verdict,
        index_line,
        report.quality_gate_summary.overall_status,
        report.quality_gate_summary.blocker_count,
        report.quality_gate_summary.decision_count,
        report.quality_gate_summary.warning_count,
        report.stage_reports.len(),
        report.memory_writes.ids.len(),
        report.memory_writes.inserted_count,
        report.memory_writes.duplicate_count,
        if evidence_summary.is_empty() {
            "none".to_string()
        } else {
            evidence_summary
        },
        report.recommended_tests.len(),
        report.recommended_test_commands.len(),
        if entropy.is_empty() {
            "none".to_string()
        } else {
            entropy
        }
    )
}

#[derive(Debug, Default, Clone)]
struct AgentTaskArtifacts {
    memory_ids: Vec<String>,
    code_symbol_ids: Vec<String>,
    file_paths: Vec<String>,
    test_paths: Vec<String>,
}

impl AgentTaskArtifacts {
    fn merge(&mut self, other: Self) {
        self.memory_ids = merge_unique_strings(&self.memory_ids, &other.memory_ids);
        self.code_symbol_ids = merge_unique_strings(&self.code_symbol_ids, &other.code_symbol_ids);
        self.file_paths = merge_unique_strings(&self.file_paths, &other.file_paths);
        self.test_paths = merge_unique_strings(&self.test_paths, &other.test_paths);
    }
}

fn collect_agent_task_artifacts(
    context: &Value,
    code_assist: Option<&Value>,
) -> AgentTaskArtifacts {
    let mut memory_ids = BTreeSet::new();
    let mut code_symbol_ids = BTreeSet::new();
    let mut file_paths = BTreeSet::new();
    let mut test_paths = BTreeSet::new();

    collect_string_field_array(
        &mut memory_ids,
        context.get("memory_fragments"),
        "memory_id",
    );
    collect_string_field_array(&mut code_symbol_ids, context.get("code"), "symbol.id");
    collect_string_field_array(&mut file_paths, context.get("code"), "symbol.file_path");
    collect_string_field_array(
        &mut code_symbol_ids,
        context.pointer("/retrieval_audit/code_hits"),
        "symbol_id",
    );
    collect_string_field_array(
        &mut file_paths,
        context.pointer("/retrieval_audit/code_hits"),
        "file_path",
    );
    collect_string_field_array(
        &mut code_symbol_ids,
        context.get("code_neighborhood"),
        "symbol_id",
    );
    collect_string_field_array(
        &mut file_paths,
        context.get("code_neighborhood"),
        "file_path",
    );

    if let Some(code_assist) = code_assist {
        let report = code_assist.get("report").unwrap_or(code_assist);
        collect_string_field_array(&mut code_symbol_ids, report.get("symbols"), "symbol.id");
        collect_string_field_array(&mut file_paths, report.get("symbols"), "symbol.file_path");
        collect_string_array(&mut file_paths, report.get("impacted_files"));
        collect_string_array(&mut test_paths, report.get("affected_tests"));
    }

    AgentTaskArtifacts {
        memory_ids: memory_ids.into_iter().collect(),
        code_symbol_ids: code_symbol_ids.into_iter().collect(),
        file_paths: file_paths.into_iter().collect(),
        test_paths: test_paths.into_iter().collect(),
    }
}

fn collect_string_array(out: &mut BTreeSet<String>, value: Option<&Value>) {
    if let Some(values) = value.and_then(Value::as_array) {
        for value in values {
            if let Some(text) = value.as_str().filter(|text| !text.trim().is_empty()) {
                out.insert(text.to_string());
            }
        }
    }
}

fn collect_string_field_array(out: &mut BTreeSet<String>, value: Option<&Value>, path: &str) {
    if let Some(values) = value.and_then(Value::as_array) {
        for value in values {
            if let Some(text) = nested_string(value, path).filter(|text| !text.trim().is_empty()) {
                out.insert(text.to_string());
            }
        }
    }
}

fn nested_string<'a>(value: &'a Value, path: &str) -> Option<&'a str> {
    let mut current = value;
    for part in path.split('.') {
        current = current.get(part)?;
    }
    current.as_str()
}

fn merge_unique_strings(left: &[String], right: &[String]) -> Vec<String> {
    left.iter()
        .chain(right.iter())
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn agent_task_artifacts_json(artifacts: &AgentTaskArtifacts) -> Value {
    json!({
        "memory_ids": artifacts.memory_ids,
        "code_symbol_ids": artifacts.code_symbol_ids,
        "file_paths": artifacts.file_paths,
        "test_paths": artifacts.test_paths
    })
}

fn agent_task_session_result(
    index: &Value,
    context: &Value,
    code_assist: Option<&Value>,
    consistency: Option<&Value>,
    artifacts: &AgentTaskArtifacts,
) -> Value {
    let code_assist_summary = code_assist.map(|value| {
        let report = value.get("report").unwrap_or(value);
        json!({
            "actual_mode": report.get("actual_mode").cloned().unwrap_or(Value::Null),
            "warning": report.get("warning").cloned().unwrap_or(Value::Null),
            "symbols": report
                .get("symbols")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            "patterns": report
                .get("patterns")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            "duplicate_pairs": report
                .get("duplicate_pairs")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            "impacted_files": report.get("impacted_files").cloned().unwrap_or_else(|| json!([])),
            "affected_tests": report.get("affected_tests").cloned().unwrap_or_else(|| json!([])),
            "test_commands": report.get("test_commands").cloned().unwrap_or_else(|| json!([]))
        })
    });
    json!({
        "index": index,
        "context": {
            "context_plan": context.get("context_plan").cloned().unwrap_or(Value::Null),
            "retrieval": context.get("retrieval").cloned().unwrap_or(Value::Null),
            "retrieval_event": context.get("retrieval_event").cloned().unwrap_or(Value::Null),
            "status": context.get("status").cloned().unwrap_or(Value::Null),
            "memory_actual_mode": context.get("memory_actual_mode").cloned().unwrap_or(Value::Null),
            "code_actual_mode": context.get("code_actual_mode").cloned().unwrap_or(Value::Null),
            "warnings": context.get("warnings").cloned().unwrap_or_else(|| json!([]))
        },
        "code_assist": code_assist_summary.unwrap_or(Value::Null),
        "consistency": consistency
            .map(|value| {
                json!({
                    "readiness_score": value.get("readiness_score").cloned().unwrap_or(Value::Null),
                    "status": value.get("status").cloned().unwrap_or(Value::Null),
                    "findings": value.get("findings").cloned().unwrap_or_else(|| json!([])),
                    "warnings": value.get("warnings").cloned().unwrap_or_else(|| json!([]))
                })
            })
            .unwrap_or(Value::Null),
        "artifacts": agent_task_artifacts_json(artifacts)
    })
}

fn mark_task_session_failed(
    store: &Store,
    project_id: &str,
    session_id: &str,
    phase: &str,
    error: &anyhow::Error,
) -> Result<()> {
    store.update_task_session(
        project_id,
        session_id,
        TaskSessionUpdate {
            status: Some("failed".to_string()),
            phase: Some(phase.to_string()),
            progress: Some(100),
            memory_ids: None,
            code_symbol_ids: None,
            file_paths: None,
            test_paths: None,
            summary: Some(Some(format!("Agent task failed in `{phase}`: {error}"))),
            result: Some(json!({
                "error": error.to_string(),
                "phase": phase
            })),
        },
    )?;
    Ok(())
}

fn tool_ontology() -> Result<Value> {
    Ok(tool_result(
        "dukememory ontology: universal core memory kinds and scopes".to_string(),
        ontology_json(),
    ))
}

fn ontology_json() -> Value {
    json!({
        "core_memory_kinds": CORE_MEMORY_KINDS,
        "memory_scopes": MEMORY_SCOPES,
        "memory_tiers": ["core", "archival", "conversation"],
        "domain_examples": {
            "generic": ["decision", "constraint", "workflow", "setup", "external_service"],
            "game": ["mechanic", "asset", "level", "balance", "lore"],
            "webapp": ["route", "ux_rule", "auth", "billing", "api_contract"],
            "library": ["public_api", "invariant", "benchmark", "compatibility"],
            "research": ["hypothesis", "source", "conclusion", "experiment"],
            "ops": ["environment", "incident", "runbook", "deployment"]
        }
    })
}

async fn tool_models(config: &Config) -> Result<Value> {
    let ollama = ollama_from_config(config);
    let (installed, warning) = match ollama.tags().await {
        Ok(tags) => (
            Some(
                tags.models
                    .into_iter()
                    .map(|model| model.name)
                    .collect::<Vec<_>>(),
            ),
            None,
        ),
        Err(error) => (None, Some(error.to_string())),
    };
    let roles = config
        .model_roles()
        .into_iter()
        .map(|(role, model)| {
            json!({
                "role": role,
                "model": model,
                "available": installed.as_ref().map(|models| {
                    models.iter().any(|installed| model_name_matches(model, installed))
                })
            })
        })
        .collect::<Vec<_>>();
    let mut text = format!(
        "Dukememory models\nollama_base_url: {}",
        config.ollama_base_url
    );
    if let Some(warning) = &warning {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    for role in &roles {
        text.push_str(&format!(
            "\n- {}: {}",
            role["role"].as_str().unwrap_or("unknown"),
            role["model"].as_str().unwrap_or("unknown")
        ));
        if let Some(available) = role["available"].as_bool() {
            text.push_str(if available {
                " (available)"
            } else {
                " (missing)"
            });
        }
    }

    Ok(tool_result(
        text,
        json!({
            "ollama_base_url": config.ollama_base_url.clone(),
            "roles": roles,
            "warning": warning
        }),
    ))
}

async fn tool_prepare(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let memory_limit = optional_u64(&arguments, "memory_limit")
        .unwrap_or(8)
        .clamp(1, 30) as usize;
    let core_memory_limit = optional_u64(&arguments, "core_memory_limit")
        .unwrap_or(5)
        .clamp(0, 10) as usize;
    let code_limit = optional_u64(&arguments, "code_limit")
        .unwrap_or(8)
        .clamp(0, 30) as usize;
    let token_budget = optional_u64(&arguments, "token_budget")
        .unwrap_or(3_000)
        .clamp(1_000, 30_000) as usize;
    let debug = optional_bool(&arguments, "debug").unwrap_or(false);
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let auto_index = optional_bool(&arguments, "auto_index").unwrap_or(true);
    let full_rebuild = optional_bool(&arguments, "full_rebuild").unwrap_or(false);
    let embed_symbols = optional_bool(&arguments, "embed_symbols").unwrap_or(false);
    let embed_symbol_limit = optional_u64(&arguments, "embed_symbol_limit")
        .unwrap_or(500)
        .clamp(1, 500) as usize;
    let validate_pending = optional_bool(&arguments, "validate_pending").unwrap_or(false);
    let apply_policy = optional_bool(&arguments, "apply_policy").unwrap_or(false);
    let validate_limit = optional_u64(&arguments, "validate_limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;

    let mut store = Store::open(&config.database_marker)?;
    let (project_id, index) = if auto_index {
        let path = optional_string(&arguments, "project_path")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_dir()?);
        let report = index_project(
            &mut store,
            &path,
            optional_string(&arguments, "project_id"),
            full_rebuild,
        )?;
        let embedded_symbols = if embed_symbols {
            embed_indexed_code_symbols(
                config,
                &store,
                &report.project_id,
                &report.indexed_files,
                embed_symbol_limit,
            )
            .await?
        } else {
            0
        };
        let project_id = report.project_id.clone();
        let index = json!({
            "enabled": true,
            "project_id": report.project_id,
            "root_path": report.root_path,
            "full_rebuild": report.full_rebuild,
            "files_seen": report.files_seen,
            "files_indexed": report.files_indexed,
            "files_skipped": report.files_skipped,
            "files_deleted": report.files_deleted,
            "indexed_files": report.indexed_files,
            "symbols_indexed": report.symbols_indexed,
            "relations_indexed": report.relations_indexed,
            "relation_targets_reset": report.relation_targets_reset,
            "calls_resolved": report.calls_resolved,
            "uses_resolved": report.uses_resolved,
            "modules_resolved": report.modules_resolved,
            "timing_ms": report.timing,
            "embed_symbols": embed_symbols,
            "embed_symbol_limit": embed_symbol_limit,
            "embedded_symbols": embedded_symbols
        });
        (project_id, index)
    } else {
        (
            resolve_project(&arguments)?,
            json!({
                "enabled": false,
                "reason": "auto_index disabled"
            }),
        )
    };

    let memory_policy = if validate_pending {
        let pending = store.list(
            &project_id,
            ListOptions {
                limit: validate_limit,
                offset: 0,
                status: StatusFilter::One(MemoryStatus::Pending),
                kind: None,
                memory_tier: None,
            },
        )?;
        let ollama = ollama_from_config(config);
        let report =
            validate_memories(&ollama, &config.validate_model, &project_id, &pending).await?;
        if apply_policy {
            for decision in &report.decisions {
                match decision.action {
                    ValidationAction::Promote => {
                        store.promote(&project_id, &decision.id, Some(&decision.reason))?
                    }
                    ValidationAction::Archive => {
                        store.archive(&project_id, &decision.id, Some(&decision.reason))?
                    }
                    ValidationAction::Keep => {}
                }
            }
        }
        json!({
            "enabled": true,
            "apply": apply_policy,
            "limit": validate_limit,
            "model": report.model,
            "decisions": report.decisions
        })
    } else {
        json!({
            "enabled": false
        })
    };

    let (context_text, mut structured) = build_context_payload(
        config,
        &store,
        BuildContextPayloadRequest {
            tool_name: "dukememory_prepare",
            task_session_id: None,
            project_id: &project_id,
            query: &query,
            memory_limit,
            core_memory_limit,
            code_limit,
            token_budget,
            mode,
            debug,
        },
    )
    .await?;

    if let Value::Object(fields) = &mut structured {
        fields.insert("auto_index".to_string(), Value::Bool(auto_index));
        fields.insert("index".to_string(), index.clone());
        fields.insert("memory_policy".to_string(), memory_policy.clone());
    }

    let text = format!(
        "Dukememory prepare\n- auto_index: {}\n- project: {}\n- indexed: {} files, skipped: {}, deleted: {}, resolved: {} calls / {} uses / {} modules, embedded_symbols: {}\n- memory_policy: {}\n\n{}",
        auto_index,
        project_id,
        index["files_indexed"].as_u64().unwrap_or(0),
        index["files_skipped"].as_u64().unwrap_or(0),
        index["files_deleted"].as_u64().unwrap_or(0),
        index["calls_resolved"].as_u64().unwrap_or(0),
        index["uses_resolved"].as_u64().unwrap_or(0),
        index["modules_resolved"].as_u64().unwrap_or(0),
        index["embedded_symbols"].as_u64().unwrap_or(0),
        memory_policy["enabled"].as_bool().unwrap_or(false),
        context_text
    );

    Ok(tool_result(text, structured))
}

struct BuildContextPayloadRequest<'a> {
    tool_name: &'a str,
    task_session_id: Option<&'a str>,
    project_id: &'a str,
    query: &'a str,
    memory_limit: usize,
    core_memory_limit: usize,
    code_limit: usize,
    token_budget: usize,
    mode: SearchMode,
    debug: bool,
}

async fn build_context_payload(
    config: &Config,
    store: &Store,
    request: BuildContextPayloadRequest<'_>,
) -> Result<(String, Value)> {
    let started_at = Instant::now();
    let project_id = request.project_id;
    let query = request.query;
    let context_plan = plan_context_access(ContextPlanRequest {
        query,
        memory_limit: request.memory_limit,
        core_memory_limit: request.core_memory_limit,
        code_limit: request.code_limit,
        token_budget: request.token_budget,
    });
    let memory_limit = context_plan.memory_limit;
    let core_memory_limit = context_plan.core_memory_limit;
    let code_limit = if request.code_limit == 0 {
        0
    } else {
        context_plan.code_limit
    };
    let token_budget = context_plan.budget_plan.effective_token_budget;
    let mode = request.mode;
    let debug = request.debug;
    let retrieval_mode = RetrievalMode::parse(mode.as_str())?;
    let ContextRetrievalOutput {
        memories: task_memories,
        code,
        diagnostics: retrieval_diagnostics,
    } = run_context_retrieval(
        config,
        store,
        ContextRetrievalRequest {
            project_id,
            query,
            memory_limit,
            code_limit: if request.code_limit > 0 && context_plan.source_plan.code_index {
                code_limit
            } else {
                0
            },
            mode: retrieval_mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let memory_actual_mode = retrieval_diagnostics.memory_actual_mode.clone();
    let code_actual_mode = retrieval_diagnostics.code_actual_mode.clone();
    let warnings = retrieval_diagnostics.warnings.clone();
    let core_memories = if core_memory_limit == 0 || !context_plan.source_plan.core_memories {
        Vec::new()
    } else {
        store.core_context_memories_for_query(project_id, query, core_memory_limit)?
    };
    let memories = merge_core_and_task_memories(core_memories, task_memories, memory_limit);
    let status = store.status(project_id)?;
    let code_status = store.code_status(project_id)?;
    let memory_fragments = build_memory_fragments(query, &memories, token_budget);
    let graph_memory_ids = fragment_memory_ids(&memory_fragments);
    let graph_limit = context_plan
        .graph_limit
        .max(memory_fragments.len())
        .max(memory_limit)
        .max(code_limit)
        .min(40);
    let graph = if context_plan.source_plan.memory_graph && graph_limit > 0 {
        store.memory_graph_for_memories(project_id, &graph_memory_ids, graph_limit)?
    } else {
        MemoryGraph {
            entities: Vec::new(),
            facts: Vec::new(),
            edges: Vec::new(),
        }
    };
    let memory_summaries = context_memory_summaries(&memories, &memory_fragments);
    let code_summaries = code_context_summaries(&code);
    let code_neighborhood = if context_plan.source_plan.code_neighborhood {
        code_graph_neighborhood(store, project_id, &code, 3)?
    } else {
        Vec::new()
    };
    let compact_code_neighborhood = compact_code_neighborhood(&code_neighborhood);
    let code_memories = if context_plan.source_plan.code_memories {
        store.code_memories_for_code_results(project_id, &code, context_plan.code_memory_limit)?
    } else {
        Vec::new()
    };
    let used_memory_ids = fragment_memory_ids(&memory_fragments);
    let used_code_memory_ids = code_memories
        .iter()
        .map(|memory| memory.id.clone())
        .collect::<Vec<_>>();
    store.mark_memories_used(project_id, &used_memory_ids)?;
    store.mark_code_memories_used(project_id, &used_code_memory_ids)?;
    let code_memory_summaries = code_memory_summaries(&code_memories);
    let agent_workflow = agent_workflow_steps(
        memory_fragments.len(),
        code.len(),
        code_neighborhood.len(),
        code_memories.len(),
        warnings.len(),
    );
    let mut text = format_task_context(TaskProjectContextFormat {
        project_id,
        query,
        mode: mode.as_str(),
        memories: &memories,
        memory_fragments: &memory_fragments,
        code: &code,
        code_memories: &code_memories,
        graph: &graph,
        total_memories: status.total_memories,
        indexed_symbols: code_status.symbols,
        token_budget,
    });
    text.push_str(&format_context_plan_text(&context_plan));
    if !code_neighborhood.is_empty() {
        text.push_str("\n\n## Code graph neighborhood");
        for entry in &code_neighborhood {
            text.push_str(&format!(
                "\n- {} `{}`: callers={}, callees={}",
                entry["kind"].as_str().unwrap_or("symbol"),
                entry["symbol_name"].as_str().unwrap_or(""),
                entry["callers"].as_array().map(Vec::len).unwrap_or(0),
                entry["callees"].as_array().map(Vec::len).unwrap_or(0)
            ));
        }
    }
    text.push_str(&format_retrieval_diagnostics(&retrieval_diagnostics));
    text.push_str(&format_agent_workflow_steps(&agent_workflow));

    let retrieval_audit = build_retrieval_audit(
        &context_plan,
        &memory_fragments,
        &code,
        &code_memories,
        &graph,
        &retrieval_diagnostics,
    );
    let estimated_tokens = estimate_context_tokens(&[
        text.chars().count(),
        serde_json::to_string(&retrieval_audit)
            .map(|value| value.chars().count())
            .unwrap_or(0),
    ]);
    let latency_ms = started_at.elapsed().as_millis() as u64;
    let plan_value = serde_json::to_value(&context_plan)?;
    let audit_value = retrieval_audit.clone();
    let graph_items = graph
        .entities
        .len()
        .saturating_add(graph.facts.len())
        .saturating_add(graph.edges.len());
    let retrieval_event = match store.record_retrieval_event(RetrievalEventRecord {
        project_id,
        task_session_id: request.task_session_id,
        tool: request.tool_name,
        query,
        task_type: &context_plan.task_type,
        token_budget,
        estimated_tokens,
        latency_ms,
        memory_fragments: memory_fragments.len(),
        code_hits: code.len(),
        graph_items,
        code_memories: code_memories.len(),
        plan: plan_value.clone(),
        audit: audit_value.clone(),
    }) {
        Ok(id) => json!({"recorded": true, "id": id}),
        Err(error) => json!({"recorded": false, "error": error.to_string()}),
    };

    let mut structured = json!({
                "project_id": project_id,
                "query": query,
                "mode": mode.as_str(),
                "task_scoped": true,
                "token_budget": token_budget,
                "estimated_context_tokens": estimated_tokens,
                "latency_ms": latency_ms,
                "context_plan": context_plan,
                "core_memory_limit": core_memory_limit,
                "task_memory_limit": memory_limit,
            "memory_actual_mode": memory_actual_mode,
            "code_actual_mode": code_actual_mode,
            "warnings": warnings,
            "retrieval": retrieval_diagnostics,
            "agent_workflow": agent_workflow,
            "memories": memory_summaries,
            "memory_fragments": compact_memory_fragments(&memory_fragments),
            "code": code_summaries,
            "code_neighborhood": compact_code_neighborhood,
                "code_memories": code_memory_summaries,
                "retrieval_audit": retrieval_audit,
                "retrieval_event": retrieval_event,
                "code_body_included": false,
            "graph_summary": compact_graph_summary(&graph),
            "status": {
                "total_memories": status.total_memories,
                "active_memories": status.active_memories,
                "pending_memories": status.pending_memories,
                "memory_embeddings": status.memory_embeddings,
                "indexed_symbols": code_status.symbols,
                "code_symbol_embeddings": code_status.symbol_embeddings
            }
    });
    if debug && let Value::Object(fields) = &mut structured {
        fields.insert(
            "trace".to_string(),
            build_context_trace(ContextTraceInput {
                project_id,
                query,
                requested_mode: mode.as_str(),
                memory_actual_mode: memory_actual_mode.as_str(),
                code_actual_mode: code_actual_mode.as_str(),
                memories: &memories,
                memory_fragments: &memory_fragments,
                code: &code,
                graph: &graph,
            }),
        );
        fields.insert("graph".to_string(), json!(graph));
        fields.insert(
            "debug_code_neighborhood".to_string(),
            json!(code_neighborhood),
        );
        fields.insert(
            "debug_memory_fragments".to_string(),
            json!(memory_fragments),
        );
    }

    Ok((text, structured))
}

fn annotated_code_relations(relations: &[CodeRelation]) -> Vec<Value> {
    relations
        .iter()
        .map(|relation| {
            let source = match relation.relation_kind.as_str() {
                "ra_call" | "ra_reference" => "rust_analyzer_lsif",
                "calls"
                | "uses"
                | "declares_module"
                | "external_module"
                | "defines_external_module" => "tree_sitter",
                _ => "unknown",
            };
            let confidence = match relation.relation_kind.as_str() {
                "ra_call" | "ra_reference" => "high",
                "calls" if relation.target_symbol_id.is_some() => "medium",
                "uses" | "declares_module" | "external_module" | "defines_external_module" => {
                    "medium"
                }
                "calls" => "low",
                _ => "low",
            };
            let resolution = if relation.target_symbol_id.is_some() {
                "resolved"
            } else {
                "unresolved"
            };
            json!({
                "relation": relation,
                "source": source,
                "confidence": confidence,
                "resolution": resolution
            })
        })
        .collect()
}

fn code_graph_neighborhood(
    store: &Store,
    project_id: &str,
    code: &[CodeSearchResult],
    relation_limit: usize,
) -> Result<Vec<Value>> {
    let mut neighborhood = Vec::new();
    for result in code.iter().take(8) {
        let callers = store.find_callers(project_id, &result.symbol.id, relation_limit)?;
        let callees = store.find_callees(project_id, &result.symbol.id, relation_limit)?;
        let caller_edges = annotated_code_relations(&callers);
        let callee_edges = annotated_code_relations(&callees);
        neighborhood.push(json!({
            "symbol_id": result.symbol.id,
            "symbol_name": result.symbol.name,
            "kind": result.symbol.kind,
            "file_path": result.symbol.file_path,
            "callers": callers,
            "callees": callees,
            "caller_edges": caller_edges,
            "callee_edges": callee_edges
        }));
    }
    Ok(neighborhood)
}

fn agent_workflow_steps(
    memory_fragments: usize,
    code_hits: usize,
    code_neighborhood: usize,
    code_memories: usize,
    warnings: usize,
) -> Vec<Value> {
    vec![
        json!({
            "step": "prepare",
            "tool": "dukememory_prepare",
            "action": "Use the returned task-scoped memory fragments, retrieval diagnostics, indexed code hits, and code graph neighborhood before editing.",
            "ready_when": format!(
                "context loaded with {memory_fragments} memory fragments, {code_hits} code hits, {code_neighborhood} code-neighborhood entries, {code_memories} code memories, and {warnings} retrieval warnings"
            )
        }),
        json!({
            "step": "reason",
            "tool": "dukememory_code_explore",
            "action": "Start from selected symbols, callers, callees, route hints, and active memories. Use code graph navigation before broad filesystem search when code structure is relevant.",
            "ready_when": if code_hits > 0 {
                "relevant symbols and their graph neighbors have been inspected"
            } else {
                "memory graph context has been inspected and a targeted code search is planned if needed"
            }
        }),
        json!({
            "step": "patch",
            "tool": "editor/test runner",
            "action": "Patch only the identified files or symbols, preserve unrelated user changes, and run focused tests first.",
            "ready_when": "the change is implemented and focused verification has passed"
        }),
        json!({
            "step": "remember",
            "tool": "dukememory_agent_after",
            "action": "Extract durable project learnings from the verified task summary. Use dukememory_code_memory action=remember for reusable symbol/file notes.",
            "ready_when": "pending memories or code memories capture only stable, reusable facts"
        }),
    ]
}

fn format_agent_workflow_steps(steps: &[Value]) -> String {
    let mut text = String::from("\nAGENT WORKFLOW\n");
    for (index, step) in steps.iter().enumerate() {
        text.push_str(&format!(
            "- {}. {} (`{}`): {}\n  ready_when: {}\n",
            index + 1,
            step["step"].as_str().unwrap_or("step"),
            step["tool"].as_str().unwrap_or("tool"),
            step["action"].as_str().unwrap_or(""),
            step["ready_when"].as_str().unwrap_or("")
        ));
    }
    text
}

fn format_context_plan_text(plan: &crate::context_plan::ContextPlan) -> String {
    format!(
        "\nCONTEXT ACCESS PLAN\n- task_type: {}\n- token_budget: {} estimated split memory/code/graph/code_memory/response = {}/{}/{}/{}/{}\n- limits: memory={} core={} code={} graph={} code_memories={}\n- sources: memories={} graph={} code_index={} code_neighborhood={} code_memories={} eval_history={}\n",
        plan.task_type,
        plan.budget_plan.effective_token_budget,
        plan.budget_plan.memory_tokens,
        plan.budget_plan.code_tokens,
        plan.budget_plan.graph_tokens,
        plan.budget_plan.code_memory_tokens,
        plan.budget_plan.response_tokens,
        plan.memory_limit,
        plan.core_memory_limit,
        plan.code_limit,
        plan.graph_limit,
        plan.code_memory_limit,
        plan.source_plan.memories,
        plan.source_plan.memory_graph,
        plan.source_plan.code_index,
        plan.source_plan.code_neighborhood,
        plan.source_plan.code_memories,
        plan.source_plan.eval_history
    )
}

fn build_retrieval_audit(
    plan: &crate::context_plan::ContextPlan,
    fragments: &[crate::context_pack::MemoryContextFragment],
    code: &[CodeSearchResult],
    code_memories: &[CodeMemory],
    graph: &MemoryGraph,
    diagnostics: &RetrievalDiagnostics,
) -> Value {
    json!({
        "task_type": plan.task_type,
        "budget": plan.budget_plan,
        "sources": diagnostics.sources,
        "warnings": diagnostics.warnings,
        "memory_fragments": fragments.iter().map(|fragment| {
            json!({
                "fragment_id": fragment.fragment_id,
                "memory_id": fragment.memory_id,
                "source": "memory_fragment",
                "section": fragment.section,
                "score": fragment.fragment_score,
                "reason": fragment.reason,
                "token_cost": estimate_context_tokens(&[fragment.text.chars().count()]),
                "dedupe_key": format!("memory:{}", fragment.memory_id)
            })
        }).collect::<Vec<_>>(),
        "code_hits": code.iter().enumerate().map(|(rank, result)| {
            json!({
                "rank": rank + 1,
                "symbol_id": result.symbol.id,
                "source": "code_index",
                "score": result.score,
                "reason": "selected by requested retrieval mode and context planner code budget",
                "token_cost": estimate_context_tokens(&[
                    result.symbol.signature.chars().count(),
                    result.symbol.body.chars().count().min(1600)
                ]),
                "file_path": result.symbol.file_path,
                "symbol_name": result.symbol.name
            })
        }).collect::<Vec<_>>(),
        "code_memories": code_memories.iter().map(|memory| {
            json!({
                "id": memory.id,
                "source": "code_memory",
                "symbol_id": memory.symbol_id,
                "file_path": memory.file_path,
                "link_status": memory.link_status,
                "quality_score": memory.quality_score,
                "usage_count": memory.usage_count,
                "token_cost": estimate_context_tokens(&[memory.body.chars().count()]),
                "reason": "linked to task-selected code symbol or file"
            })
        }).collect::<Vec<_>>(),
        "graph": {
            "source": "memory_graph",
            "entities": graph.entities.len(),
            "facts": graph.facts.len(),
            "edges": graph.edges.len(),
            "reason": "scoped to selected memory fragments"
        }
    })
}

fn compact_memory_fragments(
    fragments: &[crate::context_pack::MemoryContextFragment],
) -> Vec<Value> {
    fragments
        .iter()
        .map(|fragment| {
            json!({
                "memory_id": fragment.memory_id,
                "fragment_id": fragment.fragment_id,
                "section": fragment.section,
                "rank": fragment.rank,
                "kind": fragment.kind,
                "memory_tier": fragment.memory_tier,
                "source": fragment.source,
                "tags": fragment.tags,
                "importance": fragment.importance,
                "confidence": fragment.confidence,
                "memory_score": fragment.memory_score,
                "fragment_score": fragment.fragment_score,
                "reason": fragment.reason,
                "text_chars": fragment.text.chars().count()
            })
        })
        .collect()
}

fn compact_graph_summary(graph: &MemoryGraph) -> Value {
    json!({
        "entities_count": graph.entities.len(),
        "facts_count": graph.facts.len(),
        "edges_count": graph.edges.len(),
        "entities": graph.entities.iter().take(12).map(|entity| {
            json!({
                "id": entity.id,
                "type": entity.entity_type,
                "name": entity.name
            })
        }).collect::<Vec<_>>(),
        "facts": graph.facts.iter().take(12).map(|fact| {
            json!({
                "id": fact.id,
                "entity_id": fact.entity_id,
                "memory_id": fact.memory_id,
                "predicate": fact.predicate,
                "confidence": fact.confidence
            })
        }).collect::<Vec<_>>(),
        "edges": graph.edges.iter().take(16).map(|edge| {
            json!({
                "id": edge.id,
                "from": edge.from_entity_name,
                "to": edge.to_entity_name,
                "relation_type": edge.relation_type,
                "memory_id": edge.memory_id,
                "confidence": edge.confidence
            })
        }).collect::<Vec<_>>()
    })
}

fn compact_code_neighborhood(neighborhood: &[Value]) -> Vec<Value> {
    neighborhood
        .iter()
        .map(|entry| {
            let callers = entry["callers"]
                .as_array()
                .map(|relations| compact_relation_samples(relations))
                .unwrap_or_default();
            let callees = entry["callees"]
                .as_array()
                .map(|relations| compact_relation_samples(relations))
                .unwrap_or_default();
            json!({
                "symbol_id": entry["symbol_id"],
                "symbol_name": entry["symbol_name"],
                "kind": entry["kind"],
                "file_path": entry["file_path"],
                "caller_count": entry["callers"].as_array().map(Vec::len).unwrap_or(0),
                "callee_count": entry["callees"].as_array().map(Vec::len).unwrap_or(0),
                "callers": callers,
                "callees": callees
            })
        })
        .collect()
}

fn compact_relation_samples(relations: &[Value]) -> Vec<Value> {
    relations
        .iter()
        .take(3)
        .map(|relation| {
            json!({
                "kind": relation["relation_kind"],
                "from_file_path": relation["from_file_path"],
                "from_symbol_id": relation["from_symbol_id"],
                "target_name": relation["target_name"],
                "target_symbol_id": relation["target_symbol_id"]
            })
        })
        .collect()
}

struct ContextTraceInput<'a> {
    project_id: &'a str,
    query: &'a str,
    requested_mode: &'a str,
    memory_actual_mode: &'a str,
    code_actual_mode: &'a str,
    memories: &'a [Memory],
    memory_fragments: &'a [crate::context_pack::MemoryContextFragment],
    code: &'a [CodeSearchResult],
    graph: &'a MemoryGraph,
}

fn build_context_trace(input: ContextTraceInput<'_>) -> Value {
    json!({
        "project_id": input.project_id,
        "query": input.query,
        "requested_mode": input.requested_mode,
        "memory_actual_mode": input.memory_actual_mode,
        "code_actual_mode": input.code_actual_mode,
        "memory_hits": input.memories.iter().enumerate().map(|(rank, memory)| {
            json!({
                "rank": rank + 1,
                "id": memory.id,
                "tier": memory.memory_tier,
                "kind": memory.kind,
                "source": memory.source,
                "score": memory.score,
                "score_breakdown": {
                    "rrf_score": memory.score,
                    "tier_boost": if memory.memory_tier == "core" { 1.0 } else { 0.0 },
                    "importance": memory.importance,
                    "confidence": memory.confidence
                },
                "importance": memory.importance,
                "confidence": memory.confidence
            })
        }).collect::<Vec<_>>(),
        "memory_fragments": input.memory_fragments.iter().enumerate().map(|(rank, fragment)| {
            json!({
                "rank": rank + 1,
                "memory_id": fragment.memory_id,
                "fragment_id": fragment.fragment_id,
                "section": fragment.section,
                "kind": fragment.kind,
                "tier": fragment.memory_tier,
                "reason": fragment.reason,
                "fragment_score": fragment.fragment_score,
                "memory_score": fragment.memory_score,
                "text_chars": fragment.text.chars().count()
            })
        }).collect::<Vec<_>>(),
        "graph_entities": input.graph.entities.iter().enumerate().map(|(rank, entity)| {
            json!({
                "rank": rank + 1,
                "id": entity.id,
                "entity_type": entity.entity_type,
                "name": entity.name
            })
        }).collect::<Vec<_>>(),
        "graph_facts": input.graph.facts.iter().enumerate().map(|(rank, fact)| {
            json!({
                "rank": rank + 1,
                "id": fact.id,
                "entity_id": fact.entity_id,
                "memory_id": fact.memory_id,
                "episode_id": fact.episode_id,
                "predicate": fact.predicate,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "confidence": fact.confidence
            })
        }).collect::<Vec<_>>(),
        "graph_edges": input.graph.edges.iter().enumerate().map(|(rank, edge)| {
            json!({
                "rank": rank + 1,
                "id": edge.id,
                "from_entity_id": edge.from_entity_id,
                "to_entity_id": edge.to_entity_id,
                "memory_id": edge.memory_id,
                "episode_id": edge.episode_id,
                "relation_type": edge.relation_type,
                "valid_from": edge.valid_from,
                "valid_to": edge.valid_to,
                "confidence": edge.confidence
            })
        }).collect::<Vec<_>>(),
        "code_hits": input.code.iter().enumerate().map(|(rank, result)| {
            json!({
                "rank": rank + 1,
                "symbol_id": result.symbol.id,
                "name": result.symbol.name,
                "kind": result.symbol.kind,
                "language": result.symbol.language,
                "file_path": result.symbol.file_path,
                "start_line": result.symbol.start_line,
                "end_line": result.symbol.end_line,
                "score": result.score
            })
        }).collect::<Vec<_>>()
    })
}

async fn tool_extract(config: &Config, arguments: Value) -> Result<Value> {
    let input = required_string(&arguments, "input")?;
    let source_arg =
        optional_string(&arguments, "source").unwrap_or_else(|| "dukememory_extract".to_string());
    let prepared = prepare_extraction_input(&input, &source_arg);
    let max_candidates = optional_u64(&arguments, "max_candidates")
        .unwrap_or(8)
        .clamp(1, 20) as usize;
    let embed = optional_bool(&arguments, "embed").unwrap_or(true);
    let validate = optional_bool(&arguments, "validate").unwrap_or(false);
    let apply_policy = optional_bool(&arguments, "apply_policy").unwrap_or(false);
    let project_id = if optional_string(&arguments, "project_id").is_some()
        || optional_string(&arguments, "project_path").is_some()
    {
        resolve_project(&arguments)?
    } else if let Some(project_path) = prepared.project_path.as_deref() {
        resolve_project_id_from_path(project_path)?
    } else {
        resolve_project(&arguments)?
    };
    let store = Store::open(&config.database_marker)?;
    let episode = store.add_memory_episode(
        &project_id,
        &prepared.source,
        Some("dukememory_extract input"),
        prepared.project_path.as_deref(),
        json!({
            "source": prepared.source,
            "project_path": prepared.project_path,
            "text": truncate_text(&prepared.text, 50_000)
        }),
    )?;
    let ollama = ollama_from_config(config);
    let candidates = extract_memory_candidates(
        &ollama,
        &project_id,
        &prepared.source,
        &prepared.text,
        max_candidates,
    )
    .await?;

    let mut stored = Vec::new();
    let mut inserted_ids = Vec::new();
    for candidate in candidates {
        let outcome = store_candidate(&store, &project_id, &prepared.source, &candidate)?;
        if outcome.inserted {
            inserted_ids.push(outcome.id.clone());
        }
        let embedding = if embed && outcome.inserted {
            match embed_memory(config, &store, &project_id, &outcome.id, &candidate.body).await {
                Ok(dimensions) => json!({
                    "stored": true,
                    "model": config.memory_embed_model(),
                    "dimensions": dimensions
                }),
                Err(error) => json!({
                    "stored": false,
                    "model": config.memory_embed_model(),
                    "error": error.to_string()
                }),
            }
        } else if !outcome.inserted {
            json!({
                "stored": false,
                "model": config.memory_embed_model(),
                "skipped": true,
                "reason": "duplicate"
            })
        } else {
            json!({
                "stored": false,
                "model": config.memory_embed_model(),
                "skipped": true
            })
        };
        stored.push(json!({
            "id": outcome.id,
            "inserted": outcome.inserted,
            "duplicate_of": outcome.duplicate_of,
            "candidate": candidate,
            "embedding": embedding
        }));
    }

    let memory_policy = if validate && !inserted_ids.is_empty() {
        let mut pending = Vec::new();
        for id in &inserted_ids {
            if let Some(memory) = store.get(&project_id, id)?
                && memory.status == MemoryStatus::Pending.as_str()
            {
                pending.push(memory);
            }
        }
        let report =
            validate_memories(&ollama, &config.validate_model, &project_id, &pending).await?;
        if apply_policy {
            for decision in &report.decisions {
                match decision.action {
                    ValidationAction::Promote => {
                        store.promote(&project_id, &decision.id, Some(&decision.reason))?
                    }
                    ValidationAction::Archive => {
                        store.archive(&project_id, &decision.id, Some(&decision.reason))?
                    }
                    ValidationAction::Keep => {}
                }
            }
        }
        json!({
            "enabled": true,
            "apply": apply_policy,
            "model": report.model,
            "decisions": report.decisions
        })
    } else {
        json!({
            "enabled": false
        })
    };

    let inserted_count = stored
        .iter()
        .filter(|value| value["inserted"].as_bool() == Some(true))
        .count();
    let duplicate_count = stored.len().saturating_sub(inserted_count);

    Ok(tool_result(
        format!(
            "Extracted {} dukememory candidates in project `{project_id}`: {inserted_count} stored, {duplicate_count} duplicates skipped.",
            stored.len()
        ),
        json!({
            "project_id": project_id,
            "source": prepared.source,
            "project_path": prepared.project_path,
            "episode": episode,
            "stored": stored,
            "memory_policy": memory_policy
        }),
    ))
}

async fn tool_agent_after(config: &Config, arguments: Value) -> Result<Value> {
    let input = required_string(&arguments, "input")?;
    let mut response = tool_extract(config, arguments.clone()).await?;
    let auto_code_memories = optional_bool(&arguments, "code_memory_suggestions").unwrap_or(true);
    let code_memory_limit = optional_u64(&arguments, "code_memory_limit")
        .unwrap_or(5)
        .clamp(0, 20) as usize;
    if !auto_code_memories || code_memory_limit == 0 {
        response["structuredContent"]["code_memory_suggestions"] = json!({
            "enabled": auto_code_memories,
            "stored": []
        });
        return Ok(response);
    }

    let project_id = response["structuredContent"]["project_id"]
        .as_str()
        .ok_or_else(|| anyhow!("dukememory_agent_after extract response missing project_id"))?
        .to_string();
    let store = Store::open(&config.database_marker)?;
    let query = truncate_text(&input, 1_000);
    let (symbols, actual_mode, warning) = search_code(
        config,
        &store,
        CodeSearchRequest {
            project_id: &project_id,
            query: &query,
            limit: code_memory_limit.clamp(1, 20),
            kind: None,
            file_path: None,
            mode: SearchMode::Hybrid,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let pattern_report = build_code_pattern_report(
        &store,
        &project_id,
        &query,
        symbols,
        config.code_embed_model(),
        code_memory_limit.clamp(1, 20),
        0.72,
    )?;
    let mut stored = Vec::new();
    for suggestion in pattern_report
        .memory_suggestions
        .into_iter()
        .take(code_memory_limit)
    {
        let outcome = store.remember_code_memory(
            &project_id,
            NewCodeMemory {
                symbol_id: suggestion.symbol_id.clone(),
                symbol_kind: None,
                file_path: suggestion.file_path.clone(),
                status: "pending".to_string(),
                kind: suggestion.kind.clone(),
                body: suggestion.body.clone(),
                tags: suggestion.tags.clone(),
                source: Some(suggestion.source.clone()),
                confidence: suggestion.confidence,
            },
            true,
        )?;
        stored.push(json!({
            "outcome": outcome,
            "suggestion": suggestion
        }));
    }

    let inserted = stored
        .iter()
        .filter(|item| item["outcome"]["inserted"].as_bool() == Some(true))
        .count();
    response["structuredContent"]["code_memory_suggestions"] = json!({
        "enabled": true,
        "actual_mode": actual_mode,
        "warning": warning,
        "stored": stored
    });
    if let Some(text) = response["content"][0]["text"].as_str() {
        response["content"][0]["text"] = Value::String(format!(
            "{text}\nGenerated {inserted} pending dukememory code-memory suggestions."
        ));
    }
    Ok(response)
}

async fn tool_search(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 50) as usize;
    let status = status_filter_arg(&arguments, MemoryStatus::Active)?;
    let kind = optional_string(&arguments, "kind");
    let memory_tier = optional_string(&arguments, "memory_tier");
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let (results, actual_mode, warning) = search_memories(
        config,
        &store,
        MemorySearchRequest {
            project_id: &project_id,
            query: &query,
            limit,
            status,
            kind,
            memory_tier,
            mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;

    let text = if results.is_empty() {
        format!("No dukememory memories found in project `{project_id}` for query: {query}")
    } else {
        format_memories(
            &format!(
                "Found {} dukememory memories in project `{project_id}`:",
                results.len()
            ),
            &results,
        )
    };

    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "actual_mode": actual_mode,
            "warning": warning,
            "results": results
        }),
    ))
}

async fn tool_remember(config: &Config, arguments: Value) -> Result<Value> {
    let body = required_string(&arguments, "body")?;
    let kind = optional_string(&arguments, "kind").unwrap_or_else(|| "note".to_string());
    let status = memory_status_arg(&arguments, MemoryStatus::Pending)?;
    let source = optional_string(&arguments, "source");
    let memory_tier = optional_string(&arguments, "memory_tier")
        .unwrap_or_else(|| DEFAULT_MEMORY_TIER.to_string());
    let tags = optional_string_array(&arguments, "tags")?;
    let importance = optional_f64(&arguments, "importance").unwrap_or(0.5);
    let confidence = optional_f64(&arguments, "confidence").unwrap_or(0.7);
    let status_reason = optional_string(&arguments, "reason");
    let embed = optional_bool(&arguments, "embed").unwrap_or(true);
    let deduplicate = optional_bool(&arguments, "deduplicate").unwrap_or(true);
    let allow_sensitive = optional_bool(&arguments, "allow_sensitive").unwrap_or(false);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let memory = NewMemory {
        scope: DEFAULT_MEMORY_SCOPE.to_string(),
        memory_tier,
        kind: kind.clone(),
        body: body.clone(),
        tags,
        source: source.clone(),
        status,
        importance,
        confidence,
        status_reason,
        allow_sensitive,
    };
    let outcome = if deduplicate {
        store.remember_deduplicated(&project_id, memory)?
    } else {
        let id = store.remember(&project_id, memory)?;
        RememberOutcome {
            id,
            inserted: true,
            duplicate_of: None,
        }
    };
    let embedding = if embed && outcome.inserted {
        match embed_memory(config, &store, &project_id, &outcome.id, &body).await {
            Ok(dimensions) => json!({
                "stored": true,
                "model": config.memory_embed_model(),
                "dimensions": dimensions
            }),
            Err(error) => json!({
                "stored": false,
                "model": config.memory_embed_model(),
                "error": error.to_string()
            }),
        }
    } else if !outcome.inserted {
        json!({
            "stored": false,
            "model": config.memory_embed_model(),
            "skipped": true,
            "reason": "duplicate"
        })
    } else {
        json!({
            "stored": false,
            "model": config.memory_embed_model(),
            "skipped": true
        })
    };

    let text = if outcome.inserted {
        format!(
            "Stored dukememory memory `{}` in project `{project_id}` with status `{}`.",
            outcome.id,
            status.as_str()
        )
    } else {
        format!(
            "Skipped duplicate dukememory memory `{}` in project `{project_id}`.",
            outcome.duplicate_of.as_deref().unwrap_or(&outcome.id)
        )
    };

    Ok(tool_result(
        text,
        json!({
            "id": outcome.id,
            "inserted": outcome.inserted,
            "duplicate_of": outcome.duplicate_of,
            "project_id": project_id,
            "kind": kind,
            "status": status.as_str(),
            "source": source,
            "deduplicate": deduplicate,
            "allow_sensitive": allow_sensitive,
            "embedding": embedding
        }),
    ))
}

async fn tool_remember_smart(config: &Config, arguments: Value) -> Result<Value> {
    let body = required_string(&arguments, "body")?;
    let kind = optional_string(&arguments, "kind").unwrap_or_else(|| "note".to_string());
    let status = memory_status_arg(&arguments, MemoryStatus::Pending)?;
    let source = optional_string(&arguments, "source")
        .or_else(|| Some("dukememory_remember_smart".to_string()));
    let memory_tier = optional_string(&arguments, "memory_tier")
        .unwrap_or_else(|| DEFAULT_MEMORY_TIER.to_string());
    let tags = optional_string_array(&arguments, "tags")?;
    let importance = optional_f64(&arguments, "importance").unwrap_or(0.5);
    let confidence = optional_f64(&arguments, "confidence").unwrap_or(0.7);
    let embed = optional_bool(&arguments, "embed").unwrap_or(true);
    let write = optional_bool(&arguments, "write").unwrap_or(true);
    let allow_sensitive = optional_bool(&arguments, "allow_sensitive").unwrap_or(false);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let policy = run_semantic_operation(
        config,
        &store,
        SemanticOperationRequest {
            project_id: project_id.clone(),
            action: "policy".to_string(),
            query: optional_string(&arguments, "query"),
            body: Some(body.clone()),
            memory_id: None,
            symbol: None,
            file_path: None,
            project_path: optional_string(&arguments, "project_path"),
            input: None,
            other_project_id: None,
            expected_ids: Vec::new(),
            helpful_ids: Vec::new(),
            unhelpful_ids: Vec::new(),
            limit: optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 50) as usize,
            status: StatusFilter::One(MemoryStatus::Active),
            kind: Some(kind.clone()),
            memory_tier: Some(memory_tier.clone()),
            mode: RetrievalMode::Hybrid,
            min_similarity: optional_f64(&arguments, "min_similarity").unwrap_or(0.0),
            target_memory_model: None,
            target_code_model: None,
            as_of: optional_string(&arguments, "as_of"),
            retrieval_event_id: None,
            outcome_kind: None,
            severity: None,
            apply: false,
        },
    )
    .await;
    let (policy_report, policy_warning) = match policy {
        Ok(report) => (report, None),
        Err(error) => (
            json!({
                "action": "policy",
                "project_id": project_id,
                "decision": {
                    "action": "insert",
                    "confidence": 0.5,
                    "reason": "semantic policy unavailable; falling back to pending write with hash dedupe"
                },
                "fallback": true
            }),
            Some(error.to_string()),
        ),
    };
    let decision = policy_report["decision"]["action"]
        .as_str()
        .unwrap_or("insert");
    let decision_reason = policy_report["decision"]["reason"]
        .as_str()
        .unwrap_or("policy decision");
    let target_id = policy_report["decision"]["target_id"].as_str();

    let outcome = if !write {
        None
    } else if decision == "skip_duplicate" {
        target_id.map(|id| RememberOutcome {
            id: id.to_string(),
            inserted: false,
            duplicate_of: Some(id.to_string()),
        })
    } else {
        let status_reason = optional_string(&arguments, "reason")
            .unwrap_or_else(|| format!("smart policy `{decision}`: {decision_reason}"));
        let memory = NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: memory_tier.clone(),
            kind: kind.clone(),
            body: body.clone(),
            tags: tags.clone(),
            source: source.clone(),
            status,
            importance,
            confidence,
            status_reason: Some(status_reason),
            allow_sensitive,
        };
        Some(store.remember_deduplicated(&project_id, memory)?)
    };

    let embedding = if embed
        && let Some(outcome) = &outcome
        && outcome.inserted
    {
        match embed_memory(config, &store, &project_id, &outcome.id, &body).await {
            Ok(dimensions) => json!({
                "stored": true,
                "model": config.memory_embed_model(),
                "dimensions": dimensions
            }),
            Err(error) => json!({
                "stored": false,
                "model": config.memory_embed_model(),
                "error": error.to_string()
            }),
        }
    } else {
        json!({
            "stored": false,
            "model": config.memory_embed_model(),
            "skipped": true
        })
    };

    let text = match &outcome {
        Some(outcome) if outcome.inserted => {
            format!(
                "Smart-stored dukememory memory `{}` in project `{project_id}` with status `{}`.",
                outcome.id,
                status.as_str()
            )
        }
        Some(outcome) => format!(
            "Smart-skipped duplicate dukememory memory `{}` in project `{project_id}`.",
            outcome.duplicate_of.as_deref().unwrap_or(&outcome.id)
        ),
        None => format!("Smart memory policy built for project `{project_id}` without writing."),
    };
    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "kind": kind,
            "status": status.as_str(),
            "source": source,
            "write": write,
            "policy": policy_report,
            "policy_warning": policy_warning,
            "outcome": outcome,
            "embedding": embedding,
            "allow_sensitive": allow_sensitive
        }),
    ))
}

fn tool_get(config: &Config, arguments: Value) -> Result<Value> {
    let id = required_string(&arguments, "id")?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let memory = store.get(&project_id, &id)?;

    match memory {
        Some(memory) => Ok(tool_result(
            format_memory("dukememory memory:", &memory),
            json!({
                "project_id": project_id,
                "memory": memory
            }),
        )),
        None => Ok(tool_result(
            format!("Memory `{id}` was not found in project `{project_id}`."),
            json!({
                "project_id": project_id,
                "id": id,
                "memory": Value::Null
            }),
        )),
    }
}

fn tool_list(config: &Config, arguments: Value) -> Result<Value> {
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    let offset = optional_u64(&arguments, "offset").unwrap_or(0) as usize;
    let status = status_filter_arg(&arguments, MemoryStatus::Pending)?;
    let kind = optional_string(&arguments, "kind");
    let memory_tier = optional_string(&arguments, "memory_tier");
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let results = store.list(
        &project_id,
        ListOptions {
            limit,
            offset,
            status,
            kind,
            memory_tier,
        },
    )?;

    let text = if results.is_empty() {
        format!("No dukememory memories matched list filters in project `{project_id}`.")
    } else {
        format_memories(
            &format!(
                "Listed {} dukememory memories in project `{project_id}`:",
                results.len()
            ),
            &results,
        )
    };

    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "results": results,
            "limit": limit,
            "offset": offset
        }),
    ))
}

fn tool_review(config: &Config, arguments: Value) -> Result<Value> {
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let results = store.list(
        &project_id,
        ListOptions {
            limit,
            offset: 0,
            status: StatusFilter::One(MemoryStatus::Pending),
            kind: None,
            memory_tier: None,
        },
    )?;

    Ok(tool_result(
        format_memories(
            &format!(
                "Review {} pending dukememory memories in project `{project_id}`:",
                results.len()
            ),
            &results,
        ),
        json!({
            "project_id": project_id,
            "pending": results
        }),
    ))
}

fn tool_review_apply(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let dry_run = optional_bool(&arguments, "dry_run").unwrap_or(true);
    let decisions = arguments
        .get("decisions")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("dukememory_review_apply requires decisions array"))?;
    let store = Store::open(&config.database_marker)?;
    let mut promoted = 0_usize;
    let mut archived = 0_usize;
    let mut kept = 0_usize;
    let mut applied = Vec::new();
    for decision in decisions {
        let id = decision
            .get("id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| anyhow!("review decision is missing id"))?;
        let action = decision
            .get("action")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| anyhow!("review decision is missing action"))?;
        let reason = decision.get("reason").and_then(Value::as_str);
        match action {
            "promote" => {
                if !dry_run {
                    store.promote(&project_id, id, reason)?;
                    store.record_audit_event(
                        &project_id,
                        "mcp",
                        "review_promote",
                        "memory",
                        Some(id),
                        json!({"reason": reason}),
                    )?;
                }
                promoted += 1;
            }
            "archive" => {
                if !dry_run {
                    store.archive(&project_id, id, reason)?;
                    store.record_audit_event(
                        &project_id,
                        "mcp",
                        "review_archive",
                        "memory",
                        Some(id),
                        json!({"reason": reason}),
                    )?;
                }
                archived += 1;
            }
            "keep" => {
                if !dry_run {
                    store.record_audit_event(
                        &project_id,
                        "mcp",
                        "review_keep",
                        "memory",
                        Some(id),
                        json!({"reason": reason}),
                    )?;
                }
                kept += 1;
            }
            other => bail!("invalid review action `{other}`; use promote, archive, or keep"),
        }
        applied.push(json!({
            "id": id,
            "action": action,
            "reason": reason
        }));
    }
    Ok(tool_result(
        format!(
            "dukememory review_apply: promoted={}, archived={}, kept={}, dry_run={}",
            promoted, archived, kept, dry_run
        ),
        json!({
            "project_id": project_id,
            "dry_run": dry_run,
            "promoted": promoted,
            "archived": archived,
            "kept": kept,
            "decisions": applied
        }),
    ))
}

fn tool_audit_log(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 500) as usize;
    let store = Store::open(&config.database_marker)?;
    let events = store.list_audit_events(&project_id, limit)?;
    Ok(tool_result(
        format!(
            "Listed {} dukememory audit events in project `{project_id}`.",
            events.len()
        ),
        json!({
            "project_id": project_id,
            "audit_mode": configured_audit_mode().as_str(),
            "metrics": audit_metrics(&events),
            "events": events
        }),
    ))
}

async fn tool_validate_pending(config: &Config, arguments: Value) -> Result<Value> {
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    let apply = optional_bool(&arguments, "apply").unwrap_or(false);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let pending = store.list(
        &project_id,
        ListOptions {
            limit,
            offset: 0,
            status: StatusFilter::One(MemoryStatus::Pending),
            kind: None,
            memory_tier: None,
        },
    )?;
    let ollama = ollama_from_config(config);
    let report = validate_memories(&ollama, &config.validate_model, &project_id, &pending).await?;

    if apply {
        for decision in &report.decisions {
            match decision.action {
                ValidationAction::Promote => {
                    store.promote(&project_id, &decision.id, Some(&decision.reason))?
                }
                ValidationAction::Archive => {
                    store.archive(&project_id, &decision.id, Some(&decision.reason))?
                }
                ValidationAction::Keep => {}
            }
        }
    }

    let promoted = report
        .decisions
        .iter()
        .filter(|decision| decision.action == ValidationAction::Promote)
        .count();
    let archived = report
        .decisions
        .iter()
        .filter(|decision| decision.action == ValidationAction::Archive)
        .count();
    let kept = report
        .decisions
        .iter()
        .filter(|decision| decision.action == ValidationAction::Keep)
        .count();

    Ok(tool_result(
        format!(
            "{} validation decisions for project `{project_id}` using `{}`: {promoted} promote, {archived} archive, {kept} keep.",
            if apply { "Applied" } else { "Prepared" },
            report.model
        ),
        json!({
            "project_id": project_id,
            "model": report.model,
            "apply": apply,
            "promote": promoted,
            "archive": archived,
            "keep": kept,
            "decisions": report.decisions
        }),
    ))
}

fn tool_promote(config: &Config, arguments: Value) -> Result<Value> {
    let id = required_string(&arguments, "id")?;
    let reason = optional_string(&arguments, "reason");
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    store.promote(&project_id, &id, reason.as_deref())?;

    Ok(tool_result(
        format!("Promoted dukememory memory `{id}` to active in project `{project_id}`."),
        json!({
            "project_id": project_id,
            "id": id,
            "status": "active"
        }),
    ))
}

async fn tool_supersede(config: &Config, arguments: Value) -> Result<Value> {
    let old_id = required_string(&arguments, "old_id")?;
    let body = required_string(&arguments, "body")?;
    let kind = optional_string(&arguments, "kind").unwrap_or_else(|| "note".to_string());
    let source = optional_string(&arguments, "source");
    let tags = optional_string_array(&arguments, "tags")?;
    let importance = optional_f64(&arguments, "importance").unwrap_or(0.7);
    let confidence = optional_f64(&arguments, "confidence").unwrap_or(0.8);
    let reason = optional_string(&arguments, "reason");
    let embed = optional_bool(&arguments, "embed").unwrap_or(true);
    let allow_sensitive = optional_bool(&arguments, "allow_sensitive").unwrap_or(false);
    let project_id = resolve_project(&arguments)?;
    let mut store = Store::open(&config.database_marker)?;
    let new_id = store.supersede(
        &project_id,
        &old_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: kind.clone(),
            body: body.clone(),
            tags,
            source: source.clone(),
            status: MemoryStatus::Active,
            importance,
            confidence,
            status_reason: reason.clone(),
            allow_sensitive,
        },
        reason.as_deref(),
    )?;
    let embedding = if embed {
        match embed_memory(config, &store, &project_id, &new_id, &body).await {
            Ok(dimensions) => json!({
                "stored": true,
                "model": config.memory_embed_model(),
                "dimensions": dimensions
            }),
            Err(error) => json!({
                "stored": false,
                "model": config.memory_embed_model(),
                "error": error.to_string()
            }),
        }
    } else {
        json!({
            "stored": false,
            "model": config.memory_embed_model(),
            "skipped": true
        })
    };

    Ok(tool_result(
        format!(
            "Superseded dukememory memory `{old_id}` with `{new_id}` in project `{project_id}`."
        ),
        json!({
            "project_id": project_id,
            "old_id": old_id,
            "new_id": new_id,
            "kind": kind,
            "source": source,
            "allow_sensitive": allow_sensitive,
            "embedding": embedding
        }),
    ))
}

fn tool_archive(config: &Config, arguments: Value) -> Result<Value> {
    let id = required_string(&arguments, "id")?;
    let reason = optional_string(&arguments, "reason");
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    store.archive(&project_id, &id, reason.as_deref())?;

    Ok(tool_result(
        format!("Archived dukememory memory `{id}` in project `{project_id}`."),
        json!({
            "project_id": project_id,
            "id": id,
            "status": "archived"
        }),
    ))
}

fn tool_prune_pending(config: &Config, arguments: Value) -> Result<Value> {
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 500) as usize;
    let max_confidence = optional_f64(&arguments, "max_confidence");
    let dry_run = optional_bool(&arguments, "dry_run").unwrap_or(true);
    let reason = optional_string(&arguments, "reason").unwrap_or_else(|| {
        if dry_run {
            "dry run".to_string()
        } else {
            "pruned pending memory".to_string()
        }
    });
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let pruned = store.prune_pending(&project_id, limit, max_confidence, dry_run, Some(&reason))?;

    Ok(tool_result(
        format!(
            "{} {} pending dukememory memories in project `{project_id}`.",
            if dry_run { "Matched" } else { "Archived" },
            pruned.len()
        ),
        json!({
            "project_id": project_id,
            "dry_run": dry_run,
            "matched": pruned,
            "limit": limit,
            "max_confidence": max_confidence,
            "reason": reason
        }),
    ))
}

async fn tool_compact(config: &Config, arguments: Value) -> Result<Value> {
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(40)
        .clamp(2, 500) as usize;
    let min_memories = optional_u64(&arguments, "min_memories")
        .unwrap_or(20)
        .clamp(2, 500) as usize;
    let kind = optional_string(&arguments, "kind");
    let apply = optional_bool(&arguments, "apply").unwrap_or(false);
    let embed = optional_bool(&arguments, "embed").unwrap_or(true);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let memories = store.active_memories_for_compaction(&project_id, limit, kind)?;
    if memories.len() < min_memories {
        return Ok(tool_result(
            format!(
                "Skipped dukememory compaction for project `{project_id}`: {} candidate active memories, min_memories={min_memories}.",
                memories.len()
            ),
            json!({
                "project_id": project_id,
                "status": "skipped",
                "reason": "not enough active memories for compaction",
                "candidate_memories": memories.len(),
                "min_memories": min_memories,
                "apply": apply
            }),
        ));
    }

    let ollama = ollama_from_config(config);
    let proposal =
        propose_compaction(&ollama, config.extract_model(), &project_id, &memories).await?;
    let application = if apply {
        let outcome = apply_compaction(config, &store, &project_id, &proposal, embed).await?;
        json!({
            "applied": true,
            "summary_id": outcome.id,
            "summary_inserted": outcome.inserted,
            "summary_duplicate_of": outcome.duplicate_of,
            "archived_source_memories": proposal.source_ids.len()
        })
    } else {
        json!({
            "applied": false
        })
    };

    Ok(tool_result(
        format!(
            "{} dukememory compaction proposal for project `{project_id}`: {} source memories into `{}`.",
            if apply { "Applied" } else { "Prepared" },
            proposal.source_ids.len(),
            proposal.summary_kind
        ),
        json!({
            "project_id": project_id,
            "apply": apply,
            "proposal": proposal,
            "application": application
        }),
    ))
}

async fn tool_maintenance(config: &Config, arguments: Value) -> Result<Value> {
    let apply = optional_bool(&arguments, "apply").unwrap_or(false);
    let all = optional_bool(&arguments, "all").unwrap_or(false);
    let backup = optional_bool(&arguments, "backup").unwrap_or(false);
    let validate_pending = optional_bool(&arguments, "validate_pending").unwrap_or(false);
    let compact = optional_bool(&arguments, "compact").unwrap_or(false);
    let feedback = optional_bool(&arguments, "feedback").unwrap_or(false);
    let embed_missing = optional_bool(&arguments, "embed_missing").unwrap_or(false);
    let any_step = all || backup || validate_pending || compact || feedback || embed_missing;
    let project_id = resolve_project(&arguments)?;
    let report = run_maintenance(
        config,
        &project_id,
        MaintenanceOptions {
            apply,
            backup: all || backup,
            backup_output: optional_string(&arguments, "backup_output").map(PathBuf::from),
            validate_pending: all || validate_pending || !any_step,
            validate_limit: optional_u64(&arguments, "validate_limit")
                .unwrap_or(20)
                .clamp(1, 500) as usize,
            compact: all || compact || !any_step,
            compact_limit: optional_u64(&arguments, "compact_limit")
                .unwrap_or(40)
                .clamp(2, 500) as usize,
            compact_min_memories: optional_u64(&arguments, "compact_min_memories")
                .unwrap_or(20)
                .clamp(2, 500) as usize,
            feedback: all || feedback,
            feedback_limit: optional_u64(&arguments, "feedback_limit")
                .unwrap_or(100)
                .clamp(1, 500) as usize,
            embed_missing: all || embed_missing,
            embed_limit: optional_u64(&arguments, "embed_limit")
                .unwrap_or(50)
                .clamp(1, 500) as usize,
            embed_scope: optional_string(&arguments, "embed_scope")
                .unwrap_or_else(|| "all".to_string()),
        },
    )
    .await?;

    let mut text = format!(
        "dukememory maintenance for project `{}` apply={}",
        report.project_id, report.apply
    );
    if let Some(backup) = &report.backup {
        text.push_str(&format!(
            "\nbackup: {} ({} bytes)",
            backup.output.display(),
            backup.size_bytes
        ));
    }
    if let Some(validation) = &report.validation {
        text.push_str(&format!(
            "\nvalidation: {} ({} pending, {} promote, {} archive, {} keep)",
            validation.status,
            validation.pending,
            validation.promote,
            validation.archive,
            validation.keep
        ));
    }
    if let Some(compaction) = &report.compaction {
        text.push_str(&format!(
            "\ncompaction: {} ({} candidates, min {})",
            compaction.status, compaction.candidate_memories, compaction.min_memories
        ));
        if let Some(error) = &compaction.error {
            text.push_str(&format!("\ncompaction_error: {error}"));
        }
    }
    if let Some(feedback) = &report.feedback {
        text.push_str(&format!(
            "\nfeedback: {} ({} unapplied, {} applied, helpful_updates={}, unhelpful_updates={})",
            feedback.status,
            feedback.unapplied_events,
            feedback.applied_events,
            feedback.helpful_memories_updated,
            feedback.unhelpful_memories_updated
        ));
        if let Some(error) = &feedback.error {
            text.push_str(&format!("\nfeedback_error: {error}"));
        }
    }
    if let Some(embeddings) = &report.embeddings {
        text.push_str(&format!(
            "\nembeddings: {} missing memories, {} missing code symbols, {} memories embedded, {} code symbols embedded",
            embeddings.memories_missing,
            embeddings.code_symbols_missing,
            embeddings.memories_embedded,
            embeddings.code_symbols_embedded
        ));
    }
    for warning in &report.warnings {
        text.push_str(&format!("\nwarning: {warning}"));
    }

    Ok(tool_result(text, json!(report)))
}

async fn tool_ops_pipeline(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let health = store.health()?;
    let status = store.status(&project_id)?;
    let code_status = store.code_status(&project_id)?;
    let consistency = run_semantic_operation(
        config,
        &store,
        SemanticOperationRequest {
            project_id: project_id.clone(),
            action: "consistency".to_string(),
            query: optional_string(&arguments, "query"),
            body: None,
            memory_id: None,
            symbol: None,
            file_path: None,
            project_path: optional_string(&arguments, "project_path"),
            input: None,
            other_project_id: None,
            expected_ids: Vec::new(),
            helpful_ids: Vec::new(),
            unhelpful_ids: Vec::new(),
            limit: optional_u64(&arguments, "limit")
                .unwrap_or(20)
                .clamp(1, 200) as usize,
            status: StatusFilter::One(MemoryStatus::Active),
            kind: None,
            memory_tier: None,
            mode: RetrievalMode::Hybrid,
            min_similarity: 0.0,
            target_memory_model: None,
            target_code_model: None,
            as_of: optional_string(&arguments, "as_of"),
            retrieval_event_id: None,
            outcome_kind: None,
            severity: None,
            apply: false,
        },
    )
    .await
    .unwrap_or_else(|error| {
        json!({
            "action": "consistency",
            "project_id": project_id,
            "status": "unknown",
            "readiness_score": 0.0,
            "findings": [format!("consistency check failed: {error}")],
            "warnings": [error.to_string()]
        })
    });
    let maintenance = run_maintenance(
        config,
        &project_id,
        MaintenanceOptions {
            apply: false,
            backup: false,
            backup_output: None,
            validate_pending: true,
            validate_limit: optional_u64(&arguments, "validate_limit")
                .unwrap_or(20)
                .clamp(1, 500) as usize,
            compact: true,
            compact_limit: optional_u64(&arguments, "compact_limit")
                .unwrap_or(40)
                .clamp(2, 500) as usize,
            compact_min_memories: optional_u64(&arguments, "compact_min_memories")
                .unwrap_or(20)
                .clamp(2, 500) as usize,
            feedback: false,
            feedback_limit: optional_u64(&arguments, "feedback_limit")
                .unwrap_or(100)
                .clamp(1, 500) as usize,
            embed_missing: optional_bool(&arguments, "embed_missing").unwrap_or(true),
            embed_limit: optional_u64(&arguments, "embed_limit")
                .unwrap_or(50)
                .clamp(1, 500) as usize,
            embed_scope: optional_string(&arguments, "embed_scope")
                .unwrap_or_else(|| "all".to_string()),
        },
    )
    .await?;
    let readiness = consistency["readiness_score"].as_f64().unwrap_or(0.0);
    let text = format!(
        "Dukememory ops pipeline\nproject: {project_id}\nreadiness: {readiness:.2}\nactive_memories: {}\nindexed_symbols: {}\nmaintenance_apply: false",
        status.active_memories, code_status.symbols
    );
    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "readiness_score": readiness,
            "health": health,
            "status": status,
            "code_status": code_status,
            "consistency": consistency,
            "maintenance": maintenance
        }),
    ))
}

fn tool_status(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let status = store.status(&project_id)?;
    let code_status = store.code_status(&project_id)?;
    let schema_version = store.schema_version()?;
    let integrity_check = store.integrity_check()?;

    Ok(tool_result(
        format!(
            "dukememory project `{}` type={}: {} pending, {} active, {} superseded, {} archived, {} total, {} memory embeddings. Code index: {} files, {} symbols, {} relations, {} resolved, {} RA references, {} RA calls, {} symbol embeddings. PostgreSQL: {} schema_version={} integrity_check={}",
            status.project_id,
            status.project_type,
            status.pending_memories,
            status.active_memories,
            status.superseded_memories,
            status.archived_memories,
            status.total_memories,
            status.memory_embeddings,
            code_status.files,
            code_status.symbols,
            code_status.relations,
            code_status.resolved_relations,
            code_status.ra_references,
            code_status.ra_calls,
            code_status.symbol_embeddings,
            config.database_url,
            schema_version,
            integrity_check
        ),
        json!({
            "project_id": status.project_id,
            "project_type": status.project_type,
            "pending_memories": status.pending_memories,
            "active_memories": status.active_memories,
            "superseded_memories": status.superseded_memories,
            "archived_memories": status.archived_memories,
            "total_memories": status.total_memories,
            "memory_embeddings": status.memory_embeddings,
            "code_index": code_status,
            "models": model_roles_json(config),
            "database_url": config.database_url,
            "database_marker": config.database_marker.clone(),
            "schema_version": schema_version,
            "integrity_check": integrity_check
        }),
    ))
}

fn tool_health(config: &Config) -> Result<Value> {
    let store = Store::open(&config.database_marker)?;
    let health = store.health()?;
    Ok(tool_result(
        format!(
            "dukememory health: schema={}, projects={}, memories={}, symbols={}, temp_schemas={}, size={} bytes.",
            health.schema,
            health.projects,
            health.memories,
            health.code_symbols,
            health.temp_schemas,
            health.database_size_bytes
        ),
        json!({
            "database_url": config.database_url,
            "database_marker": config.database_marker,
            "health": health
        }),
    ))
}

fn tool_cleanup_schemas(config: &Config, arguments: Value) -> Result<Value> {
    let apply = optional_bool(&arguments, "apply").unwrap_or(false);
    let store = Store::open(&config.database_marker)?;
    let report = store.cleanup_temp_schemas(!apply)?;
    Ok(tool_result(
        format!(
            "dukememory cleanup schemas: dry_run={}, dropped={}, kept={}.",
            report.dry_run,
            report.dropped.len(),
            report.kept.len()
        ),
        json!(report),
    ))
}

fn tool_graph(config: &Config, arguments: Value) -> Result<Value> {
    let action = required_string(&arguments, "action")?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    match action.as_str() {
        "entity" => {
            let name = required_string(&arguments, "name")?;
            let entity_type = optional_string(&arguments, "entity_type")
                .or_else(|| optional_string(&arguments, "type"))
                .unwrap_or_else(|| "concept".to_string());
            let aliases = strings_from_value(arguments.get("aliases"))?;
            let description = optional_string(&arguments, "description");
            let entity = store.upsert_memory_entity(
                &project_id,
                &entity_type,
                &name,
                aliases,
                description,
            )?;
            Ok(tool_result(
                format!("Upserted dukememory graph entity `{}`.", entity.name),
                json!({ "entity": entity }),
            ))
        }
        "fact" => {
            let entity_id = match (
                optional_string(&arguments, "entity_id"),
                optional_string(&arguments, "entity_name"),
            ) {
                (Some(id), _) => Some(id),
                (None, Some(name)) => {
                    let entity_type = optional_string(&arguments, "entity_type")
                        .unwrap_or_else(|| "concept".to_string());
                    Some(
                        store
                            .upsert_memory_entity(
                                &project_id,
                                &entity_type,
                                &name,
                                Vec::new(),
                                None,
                            )?
                            .id,
                    )
                }
                (None, None) => None,
            };
            let predicate = required_string(&arguments, "predicate")?;
            let value = required_string(&arguments, "value")?;
            let memory_id = optional_string(&arguments, "memory_id");
            let confidence = optional_f64(&arguments, "confidence").unwrap_or(0.7);
            let fact = store.add_memory_fact(
                &project_id,
                entity_id.as_deref(),
                memory_id.as_deref(),
                &predicate,
                &value,
                confidence,
            )?;
            Ok(tool_result(
                format!("Stored dukememory graph fact `{}`.", fact.predicate),
                json!({ "fact": fact }),
            ))
        }
        "edge" => {
            let from_id = match (
                optional_string(&arguments, "from_id"),
                optional_string(&arguments, "from_name"),
            ) {
                (Some(id), _) => id,
                (None, Some(name)) => {
                    let entity_type = optional_string(&arguments, "from_type")
                        .unwrap_or_else(|| "concept".to_string());
                    store
                        .upsert_memory_entity(&project_id, &entity_type, &name, Vec::new(), None)?
                        .id
                }
                (None, None) => bail!("dukememory_graph edge requires from_id or from_name"),
            };
            let to_id = match (
                optional_string(&arguments, "to_id"),
                optional_string(&arguments, "to_name"),
            ) {
                (Some(id), _) => id,
                (None, Some(name)) => {
                    let entity_type = optional_string(&arguments, "to_type")
                        .unwrap_or_else(|| "concept".to_string());
                    store
                        .upsert_memory_entity(&project_id, &entity_type, &name, Vec::new(), None)?
                        .id
                }
                (None, None) => bail!("dukememory_graph edge requires to_id or to_name"),
            };
            let relation_type = required_string(&arguments, "relation_type")?;
            let memory_id = optional_string(&arguments, "memory_id");
            let confidence = optional_f64(&arguments, "confidence").unwrap_or(0.7);
            let edge = store.add_memory_edge(
                &project_id,
                &from_id,
                &to_id,
                &relation_type,
                memory_id.as_deref(),
                confidence,
            )?;
            Ok(tool_result(
                format!(
                    "Stored dukememory graph edge `{}` -> `{}`.",
                    edge.from_entity_name, edge.to_entity_name
                ),
                json!({ "edge": edge }),
            ))
        }
        "search" => {
            let query = optional_string(&arguments, "query").unwrap_or_default();
            let as_of = optional_string(&arguments, "as_of");
            let limit = optional_u64(&arguments, "limit")
                .unwrap_or(20)
                .clamp(1, 500) as usize;
            let graph =
                store.search_memory_graph_at(&project_id, &query, limit, as_of.as_deref())?;
            Ok(tool_result(
                format!(
                    "dukememory graph search: {} entities, {} facts, {} edges.",
                    graph.entities.len(),
                    graph.facts.len(),
                    graph.edges.len()
                ),
                json!({ "project_id": project_id, "query": query, "as_of": as_of, "graph": graph }),
            ))
        }
        "invalidate_fact" => {
            let fact_id = required_string(&arguments, "fact_id")
                .or_else(|_| required_string(&arguments, "id"))?;
            let invalidated_by = optional_string(&arguments, "invalidated_by");
            let valid_to = optional_string(&arguments, "valid_to");
            let fact = store.invalidate_memory_fact(
                &project_id,
                &fact_id,
                invalidated_by.as_deref(),
                valid_to.as_deref(),
            )?;
            Ok(tool_result(
                format!("Invalidated dukememory graph fact `{}`.", fact.id),
                json!({ "fact": fact }),
            ))
        }
        "invalidate_edge" => {
            let edge_id = required_string(&arguments, "edge_id")
                .or_else(|_| required_string(&arguments, "id"))?;
            let invalidated_by = optional_string(&arguments, "invalidated_by");
            let valid_to = optional_string(&arguments, "valid_to");
            let edge = store.invalidate_memory_edge(
                &project_id,
                &edge_id,
                invalidated_by.as_deref(),
                valid_to.as_deref(),
            )?;
            Ok(tool_result(
                format!("Invalidated dukememory graph edge `{}`.", edge.id),
                json!({ "edge": edge }),
            ))
        }
        other => bail!(
            "invalid graph action `{other}`; use entity, fact, edge, search, invalidate_fact, or invalidate_edge"
        ),
    }
}

fn tool_episode(config: &Config, arguments: Value) -> Result<Value> {
    let action = optional_string(&arguments, "action").unwrap_or_else(|| "search".to_string());
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    match action.as_str() {
        "add" => {
            let source = optional_string(&arguments, "source")
                .unwrap_or_else(|| "dukememory_episode".to_string());
            let summary = optional_string(&arguments, "summary");
            let raw_ref = optional_string(&arguments, "raw_ref");
            let raw_payload = arguments
                .get("raw_payload")
                .cloned()
                .or_else(|| arguments.get("payload").cloned())
                .unwrap_or_else(|| json!({}));
            let episode = store.add_memory_episode(
                &project_id,
                &source,
                summary.as_deref(),
                raw_ref.as_deref(),
                raw_payload,
            )?;
            Ok(tool_result(
                format!("Stored dukememory episode `{}`.", episode.id),
                json!({ "project_id": project_id, "episode": episode }),
            ))
        }
        "search" => {
            let query = optional_string(&arguments, "query").unwrap_or_default();
            let limit = optional_u64(&arguments, "limit")
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            let episodes = store.search_memory_episodes(&project_id, &query, limit)?;
            Ok(tool_result(
                format!(
                    "Found {} dukememory episodes in project `{project_id}`.",
                    episodes.len()
                ),
                json!({ "project_id": project_id, "query": query, "episodes": episodes }),
            ))
        }
        other => bail!("unknown dukememory_episode action `{other}`; use add or search"),
    }
}

async fn tool_graph_extract(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    let status = status_filter_arg(&arguments, MemoryStatus::Active)?;
    let kind = optional_string(&arguments, "kind");
    let query = optional_string(&arguments, "query");
    let apply = optional_bool(&arguments, "apply").unwrap_or(false);
    let store = Store::open(&config.database_marker)?;
    let memories = if let Some(query) = query.as_deref().filter(|value| !value.trim().is_empty()) {
        store.search(
            &project_id,
            SearchOptions {
                query: query.to_string(),
                limit,
                status,
                kind,
                memory_tier: None,
            },
        )?
    } else {
        store.list(
            &project_id,
            ListOptions {
                limit,
                offset: 0,
                status,
                kind,
                memory_tier: None,
            },
        )?
    };
    let ollama = ollama_from_config(config);
    let proposals = extract_memory_graph(&ollama, &project_id, &memories).await?;
    let report = apply_graph_extraction(&store, &project_id, proposals, apply)?;
    Ok(tool_result(
        format!(
            "dukememory graph extraction in project `{project_id}`: {} memories, {} entities, {} facts, {} edges, apply={apply}.",
            report.memories, report.proposed_entities, report.proposed_facts, report.proposed_edges
        ),
        json!(report),
    ))
}

fn tool_backup(config: &Config, arguments: Value) -> Result<Value> {
    let output = optional_string(&arguments, "output").map(PathBuf::from);
    let store = Store::open(&config.database_marker)?;
    let report = create_database_backup(&store, &config.database_marker, output)?;

    Ok(tool_result(
        format!(
            "Created dukememory PostgreSQL backup: {} ({} bytes).",
            report.output.display(),
            report.size_bytes
        ),
        json!({
            "source": report.source,
            "output": report.output,
            "size_bytes": report.size_bytes
        }),
    ))
}

fn tool_export(config: &Config, arguments: Value) -> Result<Value> {
    let output = PathBuf::from(required_string(&arguments, "output")?);
    let include_code = optional_bool(&arguments, "include_code").unwrap_or(true);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let export = store.export_project(&project_id, include_code)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create export directory {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&export)?;
    std::fs::write(&output, json)
        .with_context(|| format!("failed to write export {}", output.display()))?;

    Ok(tool_result(
        format!(
            "Exported dukememory project `{}` to {}: {} memories, {} code files, {} symbols, {} relations. Embeddings are not exported.",
            export.project.id,
            output.display(),
            export.memories.len(),
            export.code_files.len(),
            export.code_symbols.len(),
            export.code_relations.len()
        ),
        json!({
            "project_id": export.project.id,
            "output": output,
            "schema_version": export.schema_version,
            "include_code": export.includes_code,
            "includes_embeddings": export.includes_embeddings,
            "memories": export.memories.len(),
            "code_files": export.code_files.len(),
            "code_symbols": export.code_symbols.len(),
            "code_relations": export.code_relations.len()
        }),
    ))
}

fn tool_import(config: &Config, arguments: Value) -> Result<Value> {
    let file = PathBuf::from(required_string(&arguments, "file")?);
    let overwrite = optional_bool(&arguments, "overwrite").unwrap_or(false);
    let text = std::fs::read_to_string(&file)
        .with_context(|| format!("failed to read import file {}", file.display()))?;
    let export: ProjectExport = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse import file {}", file.display()))?;
    let mut store = Store::open(&config.database_marker)?;
    let report = store.import_project(export, overwrite)?;

    Ok(tool_result(
        format!(
            "Imported dukememory project `{}` from {}: {} memories imported ({} skipped), {} code files, {} symbols, {} relations. Run dukememory_embed_missing to rebuild embeddings.",
            report.project_id,
            file.display(),
            report.memories_imported,
            report.memories_skipped,
            report.code_files_imported,
            report.code_symbols_imported,
            report.code_relations_imported
        ),
        json!({
            "file": file,
            "report": report
        }),
    ))
}

async fn tool_code_index(config: &Config, arguments: Value) -> Result<Value> {
    let path = optional_string(&arguments, "project_path")
        .map(Into::into)
        .unwrap_or(std::env::current_dir()?);
    let project_id = optional_string(&arguments, "project_id");
    let embed_symbols = optional_bool(&arguments, "embed_symbols").unwrap_or(false);
    let embed_symbol_limit = optional_u64(&arguments, "embed_symbol_limit")
        .unwrap_or(500)
        .clamp(1, 500) as usize;
    let full_rebuild = optional_bool(&arguments, "full_rebuild").unwrap_or(false);
    let mut store = Store::open(&config.database_marker)?;
    let report = index_project(&mut store, &path, project_id, full_rebuild)?;
    let embedded_symbols = if embed_symbols {
        embed_indexed_code_symbols(
            config,
            &store,
            &report.project_id,
            &report.indexed_files,
            embed_symbol_limit,
        )
        .await?
    } else {
        0
    };

    Ok(tool_result(
        format!(
            "Indexed Rust code for project `{}`: {} indexed, {} skipped, {} deleted, {} symbols, {} relations, resolved {} calls / {} uses / {} modules, {} symbol embeddings.",
            report.project_id,
            report.files_indexed,
            report.files_skipped,
            report.files_deleted,
            report.symbols_indexed,
            report.relations_indexed,
            report.calls_resolved,
            report.uses_resolved,
            report.modules_resolved,
            embedded_symbols
        ),
        json!({
            "project_id": report.project_id,
            "root_path": report.root_path,
            "full_rebuild": report.full_rebuild,
            "files_seen": report.files_seen,
            "files_indexed": report.files_indexed,
            "files_skipped": report.files_skipped,
            "files_deleted": report.files_deleted,
            "indexed_files": report.indexed_files,
            "symbols_indexed": report.symbols_indexed,
            "relations_indexed": report.relations_indexed,
            "relation_targets_reset": report.relation_targets_reset,
            "calls_resolved": report.calls_resolved,
            "uses_resolved": report.uses_resolved,
            "modules_resolved": report.modules_resolved,
            "timing_ms": report.timing,
            "embed_symbols": embed_symbols,
            "embed_symbol_limit": embed_symbol_limit,
            "embedded_symbols": embedded_symbols
        }),
    ))
}

fn tool_code_lsif_index(config: &Config, arguments: Value) -> Result<Value> {
    let path: PathBuf = optional_string(&arguments, "project_path")
        .map(Into::into)
        .unwrap_or(std::env::current_dir()?);
    let project_id = match optional_string(&arguments, "project_id") {
        Some(project_id) => resolve_project_id(Some(project_id))?,
        None => resolve_project_id_from_path(&path)?,
    };
    let store = Store::open(&config.database_marker)?;
    let report = if let Some(input) = optional_string(&arguments, "input") {
        let input = PathBuf::from(input);
        let lsif = std::fs::read_to_string(&input)
            .with_context(|| format!("failed to read LSIF {}", input.display()))?;
        import_rust_analyzer_lsif(
            &store,
            &project_id,
            &path,
            &input.display().to_string(),
            &lsif,
        )?
    } else {
        generate_and_import_rust_analyzer_lsif(&store, &project_id, &path)?
    };

    Ok(tool_result(
        format!(
            "Imported rust-analyzer LSIF for project `{}`: {} documents, {} ranges, {} target symbols, {} RA references imported, {} RA calls imported, {} skipped, {} stale removed.",
            report.project_id,
            report.documents,
            report.ranges,
            report.target_symbols_resolved,
            report.relations_imported,
            report.call_relations_imported,
            report.relations_skipped,
            report.stale_relations_removed
        ),
        json!({
            "project_id": report.project_id,
            "root_path": report.root_path,
            "source": report.source,
            "documents": report.documents,
            "ranges": report.ranges,
            "definitions_seen": report.definitions_seen,
            "reference_ranges_seen": report.reference_ranges_seen,
            "target_symbols_resolved": report.target_symbols_resolved,
            "relations_imported": report.relations_imported,
            "call_relations_imported": report.call_relations_imported,
            "relations_skipped": report.relations_skipped,
            "stale_relations_removed": report.stale_relations_removed
        }),
    ))
}

fn tool_code_status(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let status = store.code_status(&project_id)?;

    Ok(tool_result(
        format!(
            "dukememory code index `{}`: {} files, {} symbols, {} relations, {} resolved ({:.1}%), {} unresolved, {} RA references, {} RA calls, {} symbol embeddings.",
            status.project_id,
            status.files,
            status.symbols,
            status.relations,
            status.resolved_relations,
            status.quality.relation_resolution_rate * 100.0,
            status.quality.unresolved_relations,
            status.ra_references,
            status.ra_calls,
            status.symbol_embeddings
        ),
        json!(status),
    ))
}

async fn tool_code_search(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(10).clamp(1, 50) as usize;
    let kind = optional_string(&arguments, "kind");
    let file_path = optional_string(&arguments, "file_path");
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let (results, actual_mode, warning) = search_code(
        config,
        &store,
        CodeSearchRequest {
            project_id: &project_id,
            query: &query,
            limit,
            kind,
            file_path,
            mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;

    let text = if results.is_empty() {
        format!("No indexed code symbols found in project `{project_id}` for query: {query}")
    } else {
        format_code_results(
            &format!(
                "Found {} indexed code symbols in project `{project_id}`:",
                results.len()
            ),
            &results,
        )
    };

    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "actual_mode": actual_mode,
            "warning": warning,
            "results": results
        }),
    ))
}

async fn tool_code_explore(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 25) as usize;
    let relation_limit = optional_u64(&arguments, "relation_limit")
        .unwrap_or(12)
        .clamp(1, 50) as usize;
    let include_body = optional_bool(&arguments, "include_body").unwrap_or(true);
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let (code, actual_mode, warning) = search_code(
        config,
        &store,
        CodeSearchRequest {
            project_id: &project_id,
            query: &query,
            limit,
            kind: optional_string(&arguments, "kind"),
            file_path: optional_string(&arguments, "file_path"),
            mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let code_memories = store.code_memories_for_code_results(&project_id, &code, limit.max(8))?;
    let routes = store.route_hints(&project_id, &query, limit)?;
    let mut impact = Vec::new();
    for result in &code {
        let callers = store.find_callers(&project_id, &result.symbol.id, relation_limit)?;
        let callees = store.find_callees(&project_id, &result.symbol.id, relation_limit)?;
        impact.push(json!({
            "symbol_id": result.symbol.id,
            "symbol_name": result.symbol.name,
            "file_path": result.symbol.file_path,
            "callers": callers,
            "callees": callees
        }));
    }
    let freshness = match optional_string(&arguments, "project_path") {
        Some(path) => {
            let report =
                check_code_index_freshness(&store, &PathBuf::from(path), Some(project_id.clone()))?;
            Some(json!(report))
        }
        None => None,
    };
    let text = format_code_explore(CodeExploreFormat {
        project_id: &project_id,
        query: &query,
        results: &code,
        code_memories: &code_memories,
        routes: &routes,
        impact: impact.as_slice(),
        freshness: freshness.as_ref(),
        include_body,
    });

    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "actual_mode": actual_mode,
            "warning": warning,
            "symbols": code_context_summaries(&code),
            "code_memories": code_memory_summaries(&code_memories),
            "routes": routes,
            "impact": impact,
            "freshness": freshness,
            "body_included": include_body
        }),
    ))
}

fn tool_code_memory(config: &Config, arguments: Value) -> Result<Value> {
    let action = optional_string(&arguments, "action").unwrap_or_else(|| "search".to_string());
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    match action.as_str() {
        "remember" => {
            let body = required_string(&arguments, "body")?;
            let symbol_id = optional_string(&arguments, "symbol_id")
                .or_else(|| optional_string(&arguments, "symbol"));
            let file_path = optional_string(&arguments, "file_path");
            let symbol_kind = optional_string(&arguments, "symbol_kind");
            let outcome = store.remember_code_memory(
                &project_id,
                NewCodeMemory {
                    symbol_id,
                    symbol_kind,
                    file_path,
                    status: optional_string(&arguments, "status")
                        .unwrap_or_else(|| "pending".to_string()),
                    kind: optional_string(&arguments, "kind").unwrap_or_else(|| "note".to_string()),
                    body,
                    tags: optional_string_array(&arguments, "tags")?,
                    source: optional_string(&arguments, "source"),
                    confidence: optional_f64(&arguments, "confidence").unwrap_or(0.8),
                },
                optional_bool(&arguments, "deduplicate").unwrap_or(true),
            )?;
            Ok(tool_result(
                format!(
                    "Stored dukememory code memory `{}` in project `{project_id}` inserted={}.",
                    outcome.id, outcome.inserted
                ),
                json!({
                    "project_id": project_id,
                    "outcome": outcome
                }),
            ))
        }
        "promote" => {
            let id = required_string(&arguments, "id")?;
            let reason = optional_string(&arguments, "reason");
            store.promote_code_memory(&project_id, &id, reason.as_deref())?;
            Ok(tool_result(
                format!("Promoted dukememory code memory `{id}` in project `{project_id}`."),
                json!({
                    "project_id": project_id,
                    "id": id,
                    "status": "active"
                }),
            ))
        }
        "archive" => {
            let id = required_string(&arguments, "id")?;
            let reason = optional_string(&arguments, "reason");
            store.archive_code_memory(&project_id, &id, reason.as_deref())?;
            Ok(tool_result(
                format!("Archived dukememory code memory `{id}` in project `{project_id}`."),
                json!({
                    "project_id": project_id,
                    "id": id,
                    "status": "archived"
                }),
            ))
        }
        "repair" => {
            let limit = optional_u64(&arguments, "limit")
                .unwrap_or(50)
                .clamp(1, 500) as usize;
            let apply = optional_bool(&arguments, "apply").unwrap_or(false);
            let report = store.repair_code_memory_links(&project_id, limit, apply)?;
            Ok(tool_result(
                format!(
                    "Scanned {} dukememory code memories in project `{project_id}`; repaired={}, ambiguous={}, stale={}, dry_run={}.",
                    report.scanned, report.repaired, report.ambiguous, report.stale, report.dry_run
                ),
                json!({
                    "project_id": project_id,
                    "report": report
                }),
            ))
        }
        "list" | "search" => {
            let limit = optional_u64(&arguments, "limit")
                .unwrap_or(20)
                .clamp(1, 100) as usize;
            let file_path_filter = optional_string(&arguments, "file_path");
            let symbol_kind_filter = optional_string(&arguments, "symbol_kind");
            let symbol_ids = optional_string(&arguments, "symbol_id")
                .or_else(|| optional_string(&arguments, "symbol"))
                .map(|symbol_ref| {
                    store
                        .resolve_code_symbol_reference(
                            &project_id,
                            &symbol_ref,
                            file_path_filter.as_deref(),
                            symbol_kind_filter.as_deref(),
                        )
                        .map(|symbol| symbol.id)
                })
                .transpose()?
                .into_iter()
                .collect::<Vec<_>>();
            let file_paths = file_path_filter.into_iter().collect::<Vec<_>>();
            let results = store.search_code_memories(
                &project_id,
                CodeMemorySearchOptions {
                    query: optional_string(&arguments, "query"),
                    limit,
                    status: optional_string(&arguments, "status")
                        .unwrap_or_else(|| "active".to_string()),
                    kind: optional_string(&arguments, "kind"),
                    symbol_ids,
                    file_paths,
                },
            )?;
            Ok(tool_result(
                format_code_memories(
                    &format!(
                        "Found {} dukememory code memories in project `{project_id}`:",
                        results.len()
                    ),
                    &results,
                ),
                json!({
                    "project_id": project_id,
                    "results": results
                }),
            ))
        }
        other => bail!("invalid dukememory_code_memory action `{other}`"),
    }
}

fn tool_code_affected(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let mut files = optional_string_array(&arguments, "files")?;
    if let Some(file) = optional_string(&arguments, "file") {
        files.push(file);
    }
    if files.is_empty() {
        bail!("dukememory_code_affected requires files or file");
    }
    let depth = optional_u64(&arguments, "depth").unwrap_or(5).clamp(1, 8) as usize;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(100)
        .clamp(1, 500) as usize;
    let tests = store.affected_test_files(&project_id, &files, depth, limit)?;
    Ok(tool_result(
        format!(
            "Affected test files for project `{project_id}` from {} changed files: {}",
            files.len(),
            tests.len()
        ) + &tests
            .iter()
            .map(|file| format!("\n- {file}"))
            .collect::<String>(),
        json!({
            "project_id": project_id,
            "files": files,
            "depth": depth,
            "tests": tests
        }),
    ))
}

async fn tool_code_patterns(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 25) as usize;
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let (symbols, actual_mode, warning) = search_code(
        config,
        &store,
        CodeSearchRequest {
            project_id: &project_id,
            query: &query,
            limit,
            kind: optional_string(&arguments, "kind"),
            file_path: optional_string(&arguments, "file_path"),
            mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let min_similarity = optional_f64(&arguments, "min_similarity").unwrap_or(0.72);
    let report = build_code_pattern_report(
        &store,
        &project_id,
        &query,
        symbols,
        config.code_embed_model(),
        limit,
        min_similarity,
    )?;
    let mut applied = Vec::new();
    let apply_limit = optional_u64(&arguments, "apply_limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    if optional_bool(&arguments, "apply_memory_suggestions").unwrap_or(false) {
        applied.extend(apply_code_memory_suggestions(
            &store,
            &project_id,
            &report.memory_suggestions,
            apply_limit,
        )?);
    }
    if optional_bool(&arguments, "promote_patterns").unwrap_or(false) {
        applied.extend(apply_code_memory_suggestions(
            &store,
            &project_id,
            &report.pattern_promotions,
            apply_limit,
        )?);
    }

    Ok(tool_result(
        format_code_patterns_text(&report, &actual_mode, warning.as_deref()),
        json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "actual_mode": actual_mode,
            "warning": warning,
            "min_similarity": min_similarity,
            "applied": applied,
            "report": report
        }),
    ))
}

fn tool_code_duplicates(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(20)
        .clamp(1, 200) as usize;
    let min_similarity = optional_f64(&arguments, "min_similarity").unwrap_or(0.92);
    let store = Store::open(&config.database_marker)?;
    let pairs = store.code_similarity_pairs(
        &project_id,
        CodeSimilarityPairOptions {
            embedding_model: config.code_embed_model().to_string(),
            limit,
            kind: optional_string(&arguments, "kind"),
            file_path: optional_string(&arguments, "file_path"),
            min_similarity,
        },
    )?;
    let mut text = format!(
        "Found {} near-duplicate indexed code symbol pairs in project `{project_id}` using model `{}`:",
        pairs.len(),
        config.code_embed_model()
    );
    for pair in &pairs {
        text.push_str(&format!(
            "\n- {:.3} `{}` {}:{} <-> `{}` {}:{}",
            pair.similarity,
            pair.left.kind,
            pair.left.file_path,
            pair.left.name,
            pair.right.kind,
            pair.right.file_path,
            pair.right.name
        ));
    }
    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "model": config.code_embed_model(),
            "min_similarity": min_similarity,
            "pairs": pairs
        }),
    ))
}

async fn tool_code_assist(config: &Config, arguments: Value) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 25) as usize;
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let pattern_similarity = optional_f64(&arguments, "pattern_similarity").unwrap_or(0.72);
    let duplicate_similarity = optional_f64(&arguments, "duplicate_similarity").unwrap_or(0.92);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let (symbols, actual_mode, warning) = search_code(
        config,
        &store,
        CodeSearchRequest {
            project_id: &project_id,
            query: &query,
            limit,
            kind: None,
            file_path: None,
            mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let pattern_report = build_code_pattern_report(
        &store,
        &project_id,
        &query,
        symbols,
        config.code_embed_model(),
        limit,
        pattern_similarity,
    )?;
    let duplicates = store.code_similarity_pairs(
        &project_id,
        CodeSimilarityPairOptions {
            embedding_model: config.code_embed_model().to_string(),
            limit,
            kind: None,
            file_path: None,
            min_similarity: duplicate_similarity,
        },
    )?;
    let index_guard = optional_string(&arguments, "project_path")
        .map(|path| {
            check_code_index_freshness(&store, &PathBuf::from(path), Some(project_id.clone()))
                .map(|report| code_index_guard_from_freshness(&report))
        })
        .transpose()?;
    let apply_limit = optional_u64(&arguments, "apply_limit")
        .unwrap_or(20)
        .clamp(1, 100) as usize;
    let mut applied = Vec::new();
    if optional_bool(&arguments, "apply_memory_suggestions").unwrap_or(false) {
        applied.extend(apply_code_memory_suggestions(
            &store,
            &project_id,
            &pattern_report.memory_suggestions,
            apply_limit,
        )?);
    }
    if optional_bool(&arguments, "promote_patterns").unwrap_or(false) {
        applied.extend(apply_code_memory_suggestions(
            &store,
            &project_id,
            &pattern_report.pattern_promotions,
            apply_limit,
        )?);
    }
    let report = build_code_assist_report(CodeAssistReportInput {
        project_id: &project_id,
        query: &query,
        actual_mode: actual_mode.to_string(),
        warning,
        pattern_report,
        duplicate_pairs: duplicates,
        applied_memory_suggestions: applied,
        index_guard,
    });
    Ok(tool_result(
        format_code_assist_text(&report),
        json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "pattern_similarity": pattern_similarity,
            "duplicate_similarity": duplicate_similarity,
            "report": report
        }),
    ))
}

fn tool_code_review_plan(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let mut files = optional_string_array(&arguments, "files")?;
    if let Some(file) = optional_string(&arguments, "file") {
        files.push(file);
    }
    if files.is_empty() {
        bail!("dukememory_code_review_plan requires files or file");
    }
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let duplicate_similarity = optional_f64(&arguments, "duplicate_similarity").unwrap_or(0.92);
    let duplicates = store.code_similarity_pairs(
        &project_id,
        CodeSimilarityPairOptions {
            embedding_model: config.code_embed_model().to_string(),
            limit,
            kind: None,
            file_path: None,
            min_similarity: duplicate_similarity,
        },
    )?;
    let query =
        optional_string(&arguments, "query").unwrap_or_else(|| "changed files review".to_string());
    let report =
        build_code_review_plan_report(&store, &project_id, &query, files, duplicates, limit)?;
    Ok(tool_result(
        format!(
            "Code review plan for project `{project_id}`: {} changed files, {} changed symbols, {} affected tests.",
            report.changed_files.len(),
            report.changed_symbols.len(),
            report.affected_tests.len()
        ),
        json!({
            "project_id": project_id,
            "duplicate_similarity": duplicate_similarity,
            "report": report
        }),
    ))
}

async fn tool_code_eval(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 50) as usize;
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let cases_value = arguments
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("dukememory_code_eval requires cases array"))?;
    let cases = cases_value
        .iter()
        .cloned()
        .map(serde_json::from_value::<CodeEvalCaseInput>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut observed = Vec::new();
    for case in cases {
        let (results, _, _) = search_code(
            config,
            &store,
            CodeSearchRequest {
                project_id: &project_id,
                query: &case.query,
                limit,
                kind: None,
                file_path: None,
                mode,
                allow_hybrid_fallback: true,
            },
        )
        .await?;
        observed.push((case, results));
    }
    let report = evaluate_code_cases(&project_id, mode.as_str(), observed);
    Ok(tool_result(
        format!(
            "dukememory code eval: {} passed, {} failed, {} total.",
            report.passed_cases, report.failed_cases, report.total_cases
        ),
        json!({
            "project_id": project_id,
            "mode": mode.as_str(),
            "report": report
        }),
    ))
}

fn tool_read_symbol(config: &Config, arguments: Value) -> Result<Value> {
    let symbol = required_string(&arguments, "symbol")?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let result = store.get_code_symbol(&project_id, &symbol)?;

    match result {
        Some(symbol) => Ok(tool_result(
            format_code_symbol("Indexed code symbol:", &symbol, true),
            json!({
                "project_id": project_id,
                "symbol": symbol
            }),
        )),
        None => Ok(tool_result(
            format!("Indexed symbol `{symbol}` was not found in project `{project_id}`."),
            json!({
                "project_id": project_id,
                "symbol_ref": symbol,
                "symbol": Value::Null
            }),
        )),
    }
}

fn tool_code_files(config: &Config, arguments: Value) -> Result<Value> {
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let files = store.code_files_for_project(&project_id)?;

    Ok(tool_result(
        format_code_files(
            &format!(
                "Found {} indexed files in project `{project_id}`:",
                files.len()
            ),
            &files,
        ),
        json!({
            "project_id": project_id,
            "files": files
        }),
    ))
}

fn tool_code_outline(config: &Config, arguments: Value) -> Result<Value> {
    let file_path = required_string(&arguments, "file_path")?;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let symbols = store.code_symbols_for_file(&project_id, &file_path)?;

    Ok(tool_result(
        format_code_outline(
            &format!(
                "Outline for `{file_path}` in project `{project_id}` ({} symbols):",
                symbols.len()
            ),
            &symbols,
        ),
        json!({
            "project_id": project_id,
            "file_path": file_path,
            "symbols": symbols
        }),
    ))
}

async fn tool_code_brief(config: &Config, arguments: Value) -> Result<Value> {
    let symbol = required_string(&arguments, "symbol")?;
    let model_role =
        optional_string(&arguments, "model_role").unwrap_or_else(|| "fast_code".to_string());
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let Some(symbol_data) = store.get_code_symbol(&project_id, &symbol)? else {
        return Ok(tool_result(
            format!("Indexed symbol `{symbol}` was not found in project `{project_id}`."),
            json!({
                "project_id": project_id,
                "symbol_ref": symbol,
                "report": Value::Null
            }),
        ));
    };
    let callers = store.find_callers(&project_id, &symbol, 50)?;
    let callees = store.find_callees(&project_id, &symbol, 50)?;
    let model = code_model_for_role(config, &model_role)?;
    let ollama = ollama_from_config(config);
    let report = reason_about_symbol(
        &ollama,
        model,
        &project_id,
        &symbol_data,
        &callers,
        &callees,
    )
    .await?;

    Ok(tool_result(
        format_code_reason_report(&report),
        json!({
            "project_id": project_id,
            "symbol": symbol_data,
            "model_role": model_role,
            "report": report
        }),
    ))
}

async fn tool_code_plan(config: &Config, arguments: Value) -> Result<Value> {
    tool_code_reason_search(config, arguments, CodeReasonTask::Plan, "agent_code").await
}

async fn tool_code_risk(config: &Config, arguments: Value) -> Result<Value> {
    tool_code_reason_search(config, arguments, CodeReasonTask::Risk, "deep_code").await
}

async fn tool_code_reason_search(
    config: &Config,
    arguments: Value,
    task: CodeReasonTask,
    default_model_role: &str,
) -> Result<Value> {
    let query = required_string(&arguments, "query")?;
    let memory_limit = optional_u64(&arguments, "memory_limit")
        .unwrap_or(8)
        .clamp(0, 30) as usize;
    let code_limit = optional_u64(&arguments, "code_limit")
        .unwrap_or(8)
        .clamp(0, 30) as usize;
    let model_role =
        optional_string(&arguments, "model_role").unwrap_or_else(|| default_model_role.to_string());
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("hybrid"),
    )?;
    let deterministic = optional_bool(&arguments, "deterministic").unwrap_or(false);
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    if deterministic {
        let (symbols, actual_mode, warning) = search_code(
            config,
            &store,
            CodeSearchRequest {
                project_id: &project_id,
                query: &query,
                limit: code_limit.max(1),
                kind: None,
                file_path: None,
                mode,
                allow_hybrid_fallback: true,
            },
        )
        .await?;
        let pattern_report = build_code_pattern_report(
            &store,
            &project_id,
            &query,
            symbols,
            config.code_embed_model(),
            code_limit.max(1),
            0.72,
        )?;
        let duplicates = store.code_similarity_pairs(
            &project_id,
            CodeSimilarityPairOptions {
                embedding_model: config.code_embed_model().to_string(),
                limit: code_limit.max(1),
                kind: None,
                file_path: None,
                min_similarity: 0.92,
            },
        )?;
        let assist = build_code_assist_report(CodeAssistReportInput {
            project_id: &project_id,
            query: &query,
            actual_mode: actual_mode.to_string(),
            warning,
            pattern_report,
            duplicate_pairs: duplicates,
            applied_memory_suggestions: Vec::new(),
            index_guard: None,
        });
        let report = deterministic_reason_report(task.as_str(), &query, &assist);
        return Ok(tool_result(
            format!(
                "Deterministic dukememory code {} for project `{project_id}`: {} bullets, {} risks.",
                task.as_str(),
                report.bullets.len(),
                report.risks.len()
            ),
            json!({
                "project_id": project_id,
                "query": query,
                "mode": mode.as_str(),
                "deterministic": true,
                "assist": assist,
                "report": report
            }),
        ));
    }
    let memories = if memory_limit == 0 {
        Vec::new()
    } else {
        let (memories, _, _) = search_memories(
            config,
            &store,
            MemorySearchRequest {
                project_id: &project_id,
                query: &query,
                limit: memory_limit,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
                mode,
                allow_hybrid_fallback: true,
            },
        )
        .await?;
        memories
    };
    let code = if code_limit == 0 {
        Vec::new()
    } else {
        let (code, _, _) = search_code(
            config,
            &store,
            CodeSearchRequest {
                project_id: &project_id,
                query: &query,
                limit: code_limit,
                kind: None,
                file_path: None,
                mode,
                allow_hybrid_fallback: true,
            },
        )
        .await?;
        code
    };
    let model = code_model_for_role(config, &model_role)?;
    let ollama = ollama_from_config(config);
    let report =
        reason_about_search(&ollama, model, task, &project_id, &query, &memories, &code).await?;

    Ok(tool_result(
        format_code_reason_report(&report),
        json!({
            "project_id": project_id,
            "query": query,
            "mode": mode.as_str(),
            "model_role": model_role,
            "memories": memories,
            "code": code,
            "report": report
        }),
    ))
}

fn tool_find_callers(config: &Config, arguments: Value) -> Result<Value> {
    let symbol = required_string(&arguments, "symbol")?;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let callers = store.find_callers(&project_id, &symbol, limit)?;

    Ok(tool_result(
        format_relations(
            &format!(
                "Found {} approximate callers for `{symbol}` in project `{project_id}`:",
                callers.len()
            ),
            &callers,
        ),
        json!({
            "project_id": project_id,
            "symbol": symbol,
            "callers": callers
        }),
    ))
}

fn tool_find_callees(config: &Config, arguments: Value) -> Result<Value> {
    let symbol = required_string(&arguments, "symbol")?;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let callees = store.find_callees(&project_id, &symbol, limit)?;

    Ok(tool_result(
        format_relations(
            &format!(
                "Found {} approximate callees for `{symbol}` in project `{project_id}`:",
                callees.len()
            ),
            &callees,
        ),
        json!({
            "project_id": project_id,
            "symbol": symbol,
            "callees": callees
        }),
    ))
}

fn tool_impact(config: &Config, arguments: Value) -> Result<Value> {
    let symbol = required_string(&arguments, "symbol")?;
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 200) as usize;
    let depth = optional_u64(&arguments, "depth").unwrap_or(5).clamp(1, 8) as usize;
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let callers = store.find_callers(&project_id, &symbol, limit)?;
    let callees = store.find_callees(&project_id, &symbol, limit)?;
    let impacted_files =
        impacted_files_for_relations(&store, &project_id, &symbol, &callers, &callees)?;
    let affected_tests = store.affected_test_files(&project_id, &impacted_files, depth, limit)?;
    let caller_edges = annotated_code_relations(&callers);
    let callee_edges = annotated_code_relations(&callees);

    let mut text = format!(
        "Approximate impact for `{symbol}` in project `{project_id}`:
callers: {}
callees: {}
impacted_files: {}
affected_tests: {}",
        callers.len(),
        callees.len(),
        impacted_files.len(),
        affected_tests.len()
    );
    text.push('\n');
    text.push_str(&format_relations("Callers:", &callers));
    text.push('\n');
    text.push_str(&format_relations("Callees:", &callees));
    if !impacted_files.is_empty() {
        text.push_str("\nImpacted files:");
        for file in &impacted_files {
            text.push_str(&format!("\n- {file}"));
        }
    }
    if !affected_tests.is_empty() {
        text.push_str("\nAffected tests:");
        for file in &affected_tests {
            text.push_str(&format!("\n- {file}"));
        }
    }

    Ok(tool_result(
        text,
        json!({
            "project_id": project_id,
            "symbol": symbol,
            "callers": callers,
            "callees": callees,
            "caller_edges": caller_edges,
            "callee_edges": callee_edges,
            "impacted_files": impacted_files,
            "affected_tests": affected_tests,
            "depth": depth
        }),
    ))
}

fn impacted_files_for_relations(
    store: &Store,
    project_id: &str,
    symbol: &str,
    callers: &[CodeRelation],
    callees: &[CodeRelation],
) -> Result<Vec<String>> {
    let mut files = BTreeSet::new();
    if let Some(symbol) = store.get_code_symbol(project_id, symbol)? {
        files.insert(symbol.file_path);
    }
    for relation in callers.iter().chain(callees.iter()) {
        files.insert(relation.from_file_path.clone());
        if let Some(target_symbol_id) = &relation.target_symbol_id
            && let Some(target_symbol) = store.get_code_symbol(project_id, target_symbol_id)?
        {
            files.insert(target_symbol.file_path);
        }
    }
    Ok(files.into_iter().collect())
}

async fn tool_embed_missing(config: &Config, arguments: Value) -> Result<Value> {
    let limit = optional_u64(&arguments, "limit")
        .unwrap_or(50)
        .clamp(1, 500) as usize;
    let scope = optional_string(&arguments, "scope").unwrap_or_else(|| "all".to_string());
    let project_id = resolve_project(&arguments)?;
    let store = Store::open(&config.database_marker)?;
    let report = embed_missing(config, &store, &project_id, limit, &scope).await?;

    Ok(tool_result(
        format!(
            "Built missing embeddings for project `{project_id}`: {} memories with `{}`, {} code symbols with `{}` ({} existing cached, {} reused from cache, {} generated).",
            report.memories,
            config.memory_embed_model(),
            report.code_symbols,
            config.code_embed_model(),
            report.code_symbols_cached,
            report.code_symbols_reused,
            report.code_symbols_generated
        ),
        json!({
            "project_id": project_id,
            "memory_model": config.memory_embed_model(),
            "code_model": config.code_embed_model(),
            "memories_embedded": report.memories,
            "code_symbols_embedded": report.code_symbols,
            "code_symbols_cached": report.code_symbols_cached,
            "code_symbols_reused": report.code_symbols_reused,
            "code_symbols_generated": report.code_symbols_generated,
            "scope": scope,
            "limit": limit
        }),
    ))
}

async fn tool_eval(config: &Config, arguments: Value) -> Result<Value> {
    let default_project_id = resolve_project(&arguments)?;
    let limit = optional_u64(&arguments, "limit").unwrap_or(8).clamp(1, 50) as usize;
    let mode = SearchMode::parse(
        optional_string(&arguments, "mode")
            .as_deref()
            .unwrap_or("keyword"),
    )?;
    let suite_name = optional_string(&arguments, "suite_name");
    let suite_hash = optional_string(&arguments, "suite_hash");
    let cases = arguments
        .get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("dukememory_eval requires cases array"))?;
    let store = Store::open(&config.database_marker)?;
    let mut reports = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let query = case
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("eval case {} is missing query", index + 1))?;
        let name = case
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("case-{}", index + 1));
        let project_id = case
            .get("project_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| default_project_id.clone());
        let expected_contains = strings_from_value(case.get("expected_contains"))?;
        let forbidden_contains = strings_from_value(case.get("forbidden_contains"))?;
        let expected_ids = strings_from_value(case.get("expected_ids"))?;
        let forbidden_ids = strings_from_value(case.get("forbidden_ids"))?;
        let min_results = case
            .get("min_results")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let max_latency_ms = case.get("max_latency_ms").and_then(Value::as_u64);
        let max_estimated_tokens = case
            .get("max_estimated_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize);
        let started_at = Instant::now();
        let (results, actual_mode, warning) = search_memories(
            config,
            &store,
            MemorySearchRequest {
                project_id: &project_id,
                query,
                limit,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
                mode,
                allow_hybrid_fallback: true,
            },
        )
        .await?;
        let latency_ms = started_at.elapsed().as_millis() as u64;
        let top_ids = results
            .iter()
            .map(|memory| memory.id.clone())
            .collect::<Vec<_>>();
        let top_id_set = top_ids.iter().collect::<HashSet<_>>();
        let matched_expected_ids = expected_ids
            .iter()
            .filter(|id| top_id_set.contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let matched_forbidden_ids = forbidden_ids
            .iter()
            .filter(|id| top_id_set.contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let missing_expected_ids = expected_ids
            .iter()
            .filter(|id| !top_id_set.contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let recall_at_k = if expected_ids.is_empty() {
            None
        } else {
            Some(matched_expected_ids.len() as f64 / expected_ids.len() as f64)
        };
        let precision_at_k = if expected_ids.is_empty() {
            None
        } else if top_ids.is_empty() {
            Some(0.0)
        } else {
            Some(matched_expected_ids.len() as f64 / top_ids.len() as f64)
        };
        let mrr = mean_reciprocal_rank(&top_ids, &expected_ids);
        let ndcg_at_k = ndcg_at_k(&top_ids, &expected_ids);
        let haystack_text = results
            .iter()
            .map(memory_eval_text)
            .collect::<Vec<_>>()
            .join("\n");
        let estimated_tokens = estimate_context_tokens(&[haystack_text.chars().count()]);
        let haystack = haystack_text.to_ascii_lowercase();
        let missing_expected = expected_contains
            .iter()
            .filter(|needle| !haystack.contains(&needle.to_ascii_lowercase()))
            .cloned()
            .collect::<Vec<_>>();
        let forbidden_found = forbidden_contains
            .iter()
            .filter(|needle| haystack.contains(&needle.to_ascii_lowercase()))
            .cloned()
            .collect::<Vec<_>>();
        let min_results_ok = min_results
            .map(|min_results| results.len() >= min_results)
            .unwrap_or(true);
        let latency_ok = max_latency_ms
            .map(|max_latency_ms| latency_ms <= max_latency_ms)
            .unwrap_or(true);
        let token_budget_ok = max_estimated_tokens
            .map(|max_tokens| estimated_tokens <= max_tokens)
            .unwrap_or(true);
        let passed = missing_expected.is_empty()
            && missing_expected_ids.is_empty()
            && matched_forbidden_ids.is_empty()
            && forbidden_found.is_empty()
            && min_results_ok
            && latency_ok
            && token_budget_ok;
        reports.push(json!({
            "name": name,
            "project_id": project_id,
            "query": query,
            "passed": passed,
            "hits": results.len(),
            "actual_mode": actual_mode,
            "warning": warning,
            "latency_ms": latency_ms,
                "max_latency_ms": max_latency_ms,
                "latency_ok": latency_ok,
                "estimated_tokens": estimated_tokens,
                "max_estimated_tokens": max_estimated_tokens,
                "token_budget_ok": token_budget_ok,
            "expected_ids": expected_ids,
            "forbidden_ids": forbidden_ids,
            "matched_expected_ids": matched_expected_ids,
            "missing_expected_ids": missing_expected_ids,
            "matched_forbidden_ids": matched_forbidden_ids,
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "mrr": mrr,
            "ndcg_at_k": ndcg_at_k,
            "missing_expected": missing_expected,
            "forbidden_found": forbidden_found,
            "top_ids": top_ids
        }));
    }
    let passed = reports
        .iter()
        .filter(|case| case["passed"].as_bool().unwrap_or(false))
        .count();
    let failed = reports.len().saturating_sub(passed);
    let previous_run = store.latest_eval_run_summary(
        &default_project_id,
        suite_name.as_deref(),
        suite_hash.as_deref(),
        mode.as_str(),
        None,
    )?;
    let run_id = store.record_eval_run(EvalRunRecord {
        project_id: &default_project_id,
        suite_name: suite_name.as_deref(),
        suite_hash: suite_hash.as_deref(),
        mode: mode.as_str(),
        total_cases: reports.len(),
        passed_cases: passed,
        failed_cases: failed,
        detail: json!({
            "default_project_id": default_project_id,
            "suite_name": &suite_name,
            "suite_hash": &suite_hash,
            "limit": limit,
            "mode": mode.as_str(),
            "cases": &reports
        }),
    })?;
    Ok(tool_result(
        format!(
            "dukememory eval: {} passed, {} failed, {} total. run_id={}",
            passed,
            failed,
            reports.len(),
            run_id
        ),
        json!({
            "run_id": run_id,
            "suite_name": suite_name,
            "suite_hash": suite_hash,
            "previous_run": previous_run,
            "total_cases": reports.len(),
            "passed_cases": passed,
            "failed_cases": failed,
            "cases": reports
        }),
    ))
}

fn mean_reciprocal_rank(top_ids: &[String], expected_ids: &[String]) -> Option<f64> {
    if expected_ids.is_empty() {
        return None;
    }
    let expected = expected_ids.iter().collect::<HashSet<_>>();
    top_ids
        .iter()
        .position(|id| expected.contains(id))
        .map(|index| 1.0 / (index as f64 + 1.0))
        .or(Some(0.0))
}

fn ndcg_at_k(top_ids: &[String], expected_ids: &[String]) -> Option<f64> {
    if expected_ids.is_empty() {
        return None;
    }
    let expected = expected_ids.iter().collect::<HashSet<_>>();
    let dcg = top_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| {
            if expected.contains(id) {
                Some(1.0 / ((index as f64) + 2.0).log2())
            } else {
                None
            }
        })
        .sum::<f64>();
    let ideal_hits = expected_ids.len().min(top_ids.len());
    if ideal_hits == 0 {
        return Some(0.0);
    }
    let idcg = (0..ideal_hits)
        .map(|index| 1.0 / ((index as f64) + 2.0).log2())
        .sum::<f64>();
    Some(if idcg > 0.0 { dcg / idcg } else { 0.0 })
}

fn store_candidate(
    store: &Store,
    project_id: &str,
    source: &str,
    candidate: &MemoryCandidate,
) -> Result<RememberOutcome> {
    store.remember_deduplicated(
        project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: candidate.kind.clone(),
            body: candidate.body.clone(),
            tags: candidate.tags.clone(),
            source: Some(source.to_string()),
            status: MemoryStatus::Pending,
            importance: candidate.importance,
            confidence: candidate.confidence,
            status_reason: candidate.reason.clone(),
            allow_sensitive: false,
        },
    )
}

async fn apply_compaction(
    config: &Config,
    store: &Store,
    project_id: &str,
    proposal: &CompactionProposal,
    embed: bool,
) -> Result<RememberOutcome> {
    let outcome = store.remember_deduplicated(
        project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: proposal.summary_kind.clone(),
            body: proposal.summary_body.clone(),
            tags: proposal.tags.clone(),
            source: Some("dukememory_compact".to_string()),
            status: MemoryStatus::Active,
            importance: proposal.importance,
            confidence: proposal.confidence,
            status_reason: Some(proposal.reason.clone()),
            allow_sensitive: false,
        },
    )?;
    let summary_id = outcome
        .duplicate_of
        .as_deref()
        .unwrap_or(outcome.id.as_str())
        .to_string();
    for source_id in &proposal.source_ids {
        store.archive(
            project_id,
            source_id,
            Some(&format!("compacted into {summary_id}")),
        )?;
    }
    if embed && outcome.inserted {
        embed_memory(
            config,
            store,
            project_id,
            &outcome.id,
            &proposal.summary_body,
        )
        .await?;
    }
    Ok(outcome)
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

fn tools() -> Value {
    Value::Array(vec![
        tool_schema(
            "dukememory_extract",
            "Extract conservative pending memory candidates from a transcript, summary, or hook payload using the configured local LLM.",
            json!({
                "input": {"type": "string", "description": "Transcript, summary, or hook payload text to inspect for durable project memory."},
                "source": {"type": "string", "default": "dukememory_extract"},
                "max_candidates": limit_schema(8, 20),
                "embed": {"type": "boolean", "default": true, "description": "Build embeddings for stored candidates."},
                "validate": {"type": "boolean", "default": false, "description": "Validate newly inserted pending memories with the validation model."},
                "apply_policy": {"type": "boolean", "default": false, "description": "When validate=true, apply promote/archive decisions."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["input"],
        ),
        tool_schema(
            "dukememory_models",
            "Show configured model roles and whether each model is available from Ollama.",
            json!({}),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_project_profile",
            "Show or update the universal memory profile for one project. Use project_type=generic when the domain is unknown.",
            json!({
                "name": {"type": "string"},
                "root_path": {"type": "string"},
                "project_type": {"type": "string", "default": "generic"},
                "description": {"type": "string"},
                "domains": {"type": "array", "items": {"type": "string"}},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_task_session",
            "Create, update, read, or list an agent task session with progress, phase, related memories, touched code symbols, files, tests, and final result.",
            json!({
                "action": {"type": "string", "enum": ["create", "update", "get", "list"], "default": "list"},
                "id": {"type": "string", "description": "Task session id for update/get."},
                "query": {"type": "string", "description": "Original task request. Required for create."},
                "status": {"type": "string", "enum": ["planned", "running", "completed", "failed", "archived", "any"], "default": "any"},
                "phase": {"type": "string", "description": "Current task phase, for example prepare, code, test, or done."},
                "progress": {"type": "integer", "minimum": 0, "maximum": 100},
                "memory_ids": {"type": "array", "items": {"type": "string"}},
                "code_symbol_ids": {"type": "array", "items": {"type": "string"}},
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "test_paths": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
                "result": {"type": "object", "additionalProperties": true},
                "limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_task_eval",
            "Build or run a retrieval regression eval case from a task session, including expected memory ids and hard-negative candidates.",
            json!({
                "action": {"type": "string", "enum": ["build", "run"], "default": "build"},
                "session_id": {"type": "string", "description": "Task session id. Alias: id."},
                "id": {"type": "string"},
                "query": {"type": "string", "description": "Override query when no session_id is supplied."},
                "name": {"type": "string"},
                "expected_ids": {"type": "array", "items": {"type": "string"}},
                "forbidden_ids": {"type": "array", "items": {"type": "string"}},
                "expected_contains": {"type": "array", "items": {"type": "string"}},
                "forbidden_contains": {"type": "array", "items": {"type": "string"}},
                "min_results": {"type": "integer", "minimum": 0, "default": 1},
                "limit": limit_schema(8, 50),
                "mode": search_mode_schema("hybrid"),
                "suite_name": {"type": "string"},
                "min_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_test_plan",
            "Select affected tests and concrete test commands from task session files, changed files, or impacted code symbols.",
            json!({
                "session_id": {"type": "string", "description": "Optional task session id. Alias: id."},
                "id": {"type": "string"},
                "file": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
                "symbol": {"type": "string"},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "depth": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
                "limit": limit_schema(100, 500),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_ontology",
            "Return universal memory kinds, scopes, and optional domain examples. Use this before inventing new memory kinds.",
            json!({}),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_eval",
            "Run retrieval regression cases against one project. Each case can assert expected ids, forbidden ids, and expected/forbidden substrings in top-k memory results.",
            json!({
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "project_id": project_id_schema(),
                            "query": {"type": "string"},
                            "expected_contains": {"type": "array", "items": {"type": "string"}},
                            "forbidden_contains": {"type": "array", "items": {"type": "string"}},
                            "expected_ids": {"type": "array", "items": {"type": "string"}},
                            "forbidden_ids": {"type": "array", "items": {"type": "string"}},
                            "min_results": {"type": "integer", "minimum": 0},
                            "max_latency_ms": {"type": "integer", "minimum": 0},
                            "max_estimated_tokens": {"type": "integer", "minimum": 0}
                        },
                        "required": ["query"]
                    }
                },
                "limit": limit_schema(8, 50),
                "mode": search_mode_schema("keyword"),
                "suite_name": {"type": "string"},
                "suite_hash": {"type": "string"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["cases"],
        ),
        tool_schema(
            "dukememory_prepare",
            "Auto-index one project and then build prompt-ready context for an agent task. Prefer this as the first call before coding.",
            json!({
                "query": {"type": "string"},
                "memory_limit": limit_schema(8, 30),
                "core_memory_limit": {"type": "integer", "minimum": 0, "maximum": 10, "default": 5, "description": "Maximum core/project-rule memories to include before task memory fragments. Does not consume memory_limit."},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "token_budget": {"type": "integer", "minimum": 1000, "maximum": 30000, "default": 3000, "description": "Approximate maximum task context budget. Full project memory is not loaded; selected memory fragments are packed within this budget."},
                "debug": {"type": "boolean", "default": false, "description": "Include full trace, graph, and verbose code-neighborhood objects in structuredContent."},
                "mode": search_mode_schema("hybrid"),
                "auto_index": {"type": "boolean", "default": true, "description": "Run incremental code indexing before retrieval."},
                "full_rebuild": {"type": "boolean", "default": false, "description": "Delete and rebuild the full code index instead of only changed/deleted files."},
                "embed_symbols": {"type": "boolean", "default": false, "description": "Build embeddings for symbols in files indexed during this run."},
                "embed_symbol_limit": limit_schema(500, 500),
                "validate_pending": {"type": "boolean", "default": false, "description": "Validate pending memories before returning context."},
                "apply_policy": {"type": "boolean", "default": false, "description": "When validate_pending=true, apply promote/archive decisions."},
                "validate_limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_agent_before",
            "Alias for dukememory_prepare. Use at the start of an agent task to retrieve project memory and code context.",
            json!({
                "query": {"type": "string"},
                "memory_limit": limit_schema(8, 30),
                "core_memory_limit": {"type": "integer", "minimum": 0, "maximum": 10, "default": 5},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "token_budget": {"type": "integer", "minimum": 1000, "maximum": 30000, "default": 3000},
                "debug": {"type": "boolean", "default": false},
                "mode": search_mode_schema("hybrid"),
                "auto_index": {"type": "boolean", "default": true},
                "full_rebuild": {"type": "boolean", "default": false},
                "embed_symbols": {"type": "boolean", "default": false},
                "embed_symbol_limit": limit_schema(500, 500),
                "validate_pending": {"type": "boolean", "default": false},
                "apply_policy": {"type": "boolean", "default": false},
                "validate_limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_agent_task",
            "Run a complete agent task intake: create a task session, optionally index code, build task-scoped context, collect code-assist signals, and store progress/artifacts.",
            json!({
                "query": {"type": "string"},
                "memory_limit": limit_schema(8, 30),
                "core_memory_limit": {"type": "integer", "minimum": 0, "maximum": 10, "default": 5},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "token_budget": {"type": "integer", "minimum": 1000, "maximum": 30000, "default": 3000},
                "mode": search_mode_schema("hybrid"),
                "auto_index": {"type": "boolean", "default": true},
                "full_rebuild": {"type": "boolean", "default": false},
                "embed_symbols": {"type": "boolean", "default": false},
                "embed_symbol_limit": limit_schema(500, 500),
                "include_code_assist": {"type": "boolean", "default": true},
                "include_consistency": {"type": "boolean", "default": true},
                "code_assist_limit": limit_schema(8, 25),
                "pattern_similarity": {"type": "number", "default": 0.72},
                "duplicate_similarity": {"type": "number", "default": 0.92},
                "debug": {"type": "boolean", "default": false},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_devsystem",
            "Run the dukedevsystem advisory orchestrator: Planner, Memory, Architect, Coder, Test, Critic, Refactor, Memory plus File Entropy Score reports for touched files.",
            json!({
                "query": {"type": "string", "description": "Raw development task or intent."},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Project-relative files to evaluate."},
                "file": {"type": "string", "description": "Single project-relative file to evaluate."},
                "write_memory": {"type": "boolean", "default": true, "description": "Write the final quality observation as pending dukememory memory."},
                "auto_index": {"type": "boolean", "default": true, "description": "Run incremental dukememory_code_index before devsystem analysis."},
                "full_rebuild": {"type": "boolean", "default": false, "description": "Delete and rebuild the full code index before devsystem analysis."},
                "embed_symbols": {"type": "boolean", "default": false, "description": "Build embeddings for symbols in files indexed during the auto-index run."},
                "embed_symbol_limit": limit_schema(500, 10000),
                "run_evidence": {"type": "boolean", "default": false, "description": "Run allowed recommended test commands and attach quality evidence to gates."},
                "evidence_timeout_seconds": {"type": "integer", "default": 120, "minimum": 1, "maximum": 3600},
                "max_evidence_commands": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "allowed_evidence_commands": {"type": "array", "items": {"type": "string"}, "description": "Optional exact command filter for quality evidence execution. When supplied, only matching recommended commands are run."},
                "review_limit": limit_schema(50, 200),
                "duplicate_similarity": {"type": "number", "default": 0.92, "description": "Minimum code embedding similarity for duplicate/refactor-risk candidates."},
                "policy": {
                    "type": "object",
                    "description": "Optional per-run dukedevsystem policy override. Project config may also provide [devsystem] values in .dukememory.toml.",
                    "properties": {
                        "boundary_repair_score_threshold": {"type": "integer", "default": 45},
                        "split_score_threshold": {"type": "integer", "default": 75},
                        "boundary_repair_responsibility_count": {"type": "integer", "default": 3},
                        "split_responsibility_count": {"type": "integer", "default": 7},
                        "line_signal_step": {"type": "integer", "default": 200},
                        "max_line_signal": {"type": "integer", "default": 10},
                        "low_coverage_threshold": {"type": "number", "default": 0.35},
                        "ignored_file_patterns": {"type": "array", "items": {"type": "string"}},
                        "static_metadata_patterns": {"type": "array", "items": {"type": "string"}},
                        "generated_file_patterns": {"type": "array", "items": {"type": "string"}},
                        "required_test_commands": {"type": "array", "items": {"type": "string"}},
                        "responsibility_keywords": {
                            "type": "object",
                            "additionalProperties": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_agent_after",
            "Alias for dukememory_extract. Use after an agent task to extract, validate, and store durable project memories.",
            json!({
                "input": {"type": "string", "description": "Transcript, summary, or task result to inspect for durable memory."},
                "source": {"type": "string", "default": "dukememory_agent_after"},
                "max_candidates": limit_schema(8, 30),
                "embed": {"type": "boolean", "default": true},
                "validate": {"type": "boolean", "default": true},
                "apply_policy": {"type": "boolean", "default": false},
                "code_memory_suggestions": {"type": "boolean", "default": true},
                "code_memory_limit": limit_schema(5, 20),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["input"],
        ),
        tool_schema(
            "dukememory_context",
            "Build prompt-ready project context from active memories and indexed code for one task query.",
            json!({
                "query": {"type": "string"},
                "memory_limit": limit_schema(8, 30),
                "core_memory_limit": {"type": "integer", "minimum": 0, "maximum": 10, "default": 5},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "token_budget": {"type": "integer", "minimum": 1000, "maximum": 30000, "default": 3000},
                "debug": {"type": "boolean", "default": false},
                "mode": search_mode_schema("hybrid"),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_context_plan",
            "Plan minimal project-scoped memory/code/graph access for one task without loading full context.",
            json!({
                "query": {"type": "string"},
                "memory_limit": limit_schema(8, 30),
                "core_memory_limit": {"type": "integer", "minimum": 0, "maximum": 10, "default": 5},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "token_budget": {"type": "integer", "minimum": 1000, "maximum": 30000, "default": 3000},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_graph",
            "Create, search, or invalidate project-scoped memory graph entities, facts, and edges. Actions: entity, fact, edge, search, invalidate_fact, invalidate_edge.",
            json!({
                "action": {"type": "string", "enum": ["entity", "fact", "edge", "search", "invalidate_fact", "invalidate_edge"]},
                "id": {"type": "string"},
                "fact_id": {"type": "string"},
                "edge_id": {"type": "string"},
                "name": {"type": "string", "description": "Entity name for action=entity."},
                "entity_type": {"type": "string", "default": "concept"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "entity_id": {"type": "string"},
                "entity_name": {"type": "string"},
                "predicate": {"type": "string"},
                "value": {"type": "string"},
                "from_id": {"type": "string"},
                "from_name": {"type": "string"},
                "from_type": {"type": "string", "default": "concept"},
                "to_id": {"type": "string"},
                "to_name": {"type": "string"},
                "to_type": {"type": "string", "default": "concept"},
                "relation_type": {"type": "string"},
                "memory_id": {"type": "string"},
                "invalidated_by": {"type": "string"},
                "valid_to": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7},
                "query": {"type": "string"},
                "as_of": {"type": "string", "description": "Optional timestamp for temporal graph search."},
                "limit": limit_schema(20, 500),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["action"],
        ),
        tool_schema(
            "dukememory_episode",
            "Add or search project-scoped raw memory episodes used as auditable ground-truth provenance.",
            json!({
                "action": {"type": "string", "enum": ["add", "search"], "default": "search"},
                "source": {"type": "string", "default": "dukememory_episode"},
                "summary": {"type": "string"},
                "raw_ref": {"type": "string"},
                "raw_payload": {"type": "object"},
                "payload": {"type": "object"},
                "query": {"type": "string"},
                "limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_graph_extract",
            "Extract a project-scoped memory knowledge graph from existing memories using the configured local LLM. Defaults to dry-run; set apply=true to write entities, facts, and edges.",
            json!({
                "limit": limit_schema(20, 100),
                "status": status_schema("active"),
                "kind": {"type": "string"},
                "query": {"type": "string", "description": "Optional memory search query. When omitted, recent memories matching status/kind are used."},
                "apply": {"type": "boolean", "default": false, "description": "When false, return proposed graph items without writing them."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_search",
            "Search durable memory in one project. Defaults to active memories only.",
            json!({
                "query": {"type": "string"},
                "limit": limit_schema(8, 50),
                "status": status_schema("active"),
                "kind": {"type": "string"},
                "memory_tier": {"type": "string", "enum": ["core", "archival", "conversation"]},
                "mode": search_mode_schema("hybrid"),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_remember",
            "Create a durable memory. Defaults to pending, so automatic agent writes can be reviewed before becoming active.",
            json!({
                "body": {"type": "string"},
                "kind": {"type": "string", "default": "note"},
                "status": memory_status_schema("pending"),
                "memory_tier": {"type": "string", "enum": ["core", "archival", "conversation"], "default": "archival"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
                "importance": score_schema(0.5),
                "confidence": score_schema(0.7),
                "reason": {"type": "string"},
                "embed": {"type": "boolean", "default": true, "description": "Build an Ollama embedding immediately. If embedding fails, the memory is still stored."},
                "deduplicate": {"type": "boolean", "default": true, "description": "Skip insert when an active or pending memory with the same normalized body already exists in this project."},
                "allow_sensitive": {"type": "boolean", "default": false, "description": "Explicit manual override for safety-blocked sensitive content. Automatic callers should not set this."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["body"],
        ),
        tool_schema(
            "dukememory_remember_smart",
            "Run semantic memory policy, then write a pending memory with local hash dedupe or skip a policy-detected duplicate.",
            json!({
                "body": {"type": "string"},
                "query": {"type": "string"},
                "kind": {"type": "string", "default": "note"},
                "status": memory_status_schema("pending"),
                "memory_tier": {"type": "string", "enum": ["core", "archival", "conversation"], "default": "archival"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string", "default": "dukememory_remember_smart"},
                "importance": score_schema(0.5),
                "confidence": score_schema(0.7),
                "reason": {"type": "string"},
                "write": {"type": "boolean", "default": true},
                "embed": {"type": "boolean", "default": true},
                "limit": limit_schema(8, 50),
                "min_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
                "allow_sensitive": {"type": "boolean", "default": false, "description": "Explicit manual override for safety-blocked sensitive content. Automatic callers should not set this."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["body"],
        ),
        tool_schema(
            "dukememory_get",
            "Read one memory by id, scoped to one project.",
            json!({
                "id": {"type": "string"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["id"],
        ),
        tool_schema(
            "dukememory_list",
            "List memories for review or inspection. Defaults to pending memories.",
            json!({
                "limit": limit_schema(20, 100),
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "status": status_schema("pending"),
                "kind": {"type": "string"},
                "memory_tier": {"type": "string", "enum": ["core", "archival", "conversation"]},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_review",
            "Review pending memories for promotion or archive decisions.",
            json!({
                "limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_review_apply",
            "Apply pending memory review decisions in batch. Actions: promote, archive, keep. Defaults to dry-run.",
            json!({
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "action": {"type": "string", "enum": ["promote", "archive", "keep"]},
                            "reason": {"type": "string"}
                        },
                        "required": ["id", "action"]
                    }
                },
                "dry_run": {"type": "boolean", "default": true},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["decisions"],
        ),
        tool_schema(
            "dukememory_audit_log",
            "Show recent project audit events for tool calls and review decisions.",
            json!({
                "limit": limit_schema(50, 500),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_validate_pending",
            "Validate pending memories with the configured validation model. Defaults to dry-run; set apply=true to promote/archive.",
            json!({
                "limit": limit_schema(20, 100),
                "apply": {"type": "boolean", "default": false},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_promote",
            "Promote a pending memory to active after it is accepted as durable project knowledge.",
            json!({
                "id": {"type": "string"},
                "reason": {"type": "string"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["id"],
        ),
        tool_schema(
            "dukememory_supersede",
            "Replace an old memory with a new active version while preserving history.",
            json!({
                "old_id": {"type": "string"},
                "body": {"type": "string"},
                "kind": {"type": "string", "default": "note"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
                "importance": score_schema(0.7),
                "confidence": score_schema(0.8),
                "reason": {"type": "string"},
                "embed": {"type": "boolean", "default": true, "description": "Build an Ollama embedding for the replacement memory."},
                "allow_sensitive": {"type": "boolean", "default": false, "description": "Explicit manual override for safety-blocked sensitive content. Automatic callers should not set this."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["old_id", "body"],
        ),
        tool_schema(
            "dukememory_archive",
            "Archive a memory without physically deleting it.",
            json!({
                "id": {"type": "string"},
                "reason": {"type": "string"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["id"],
        ),
        tool_schema(
            "dukememory_prune_pending",
            "Archive pending memories in bulk, optionally limited to low-confidence candidates. Defaults to dry_run.",
            json!({
                "limit": limit_schema(50, 500),
                "max_confidence": score_schema(0.5),
                "dry_run": {"type": "boolean", "default": true},
                "reason": {"type": "string"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_compact",
            "Compact older active memories into one durable project_summary. Defaults to dry-run; set apply=true to archive source memories and store the summary.",
            json!({
                "limit": limit_schema(40, 500),
                "min_memories": limit_schema(20, 500),
                "kind": {"type": "string", "description": "Optional kind filter. When omitted, project_summary memories are excluded from candidates."},
                "apply": {"type": "boolean", "default": false},
                "embed": {"type": "boolean", "default": true, "description": "When apply=true, build an embedding for a newly inserted summary."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_maintenance",
            "Run project memory maintenance. With no step flags, performs dry-run validation and compaction checks. Set apply=true to apply validation, compaction, and embedding rebuild steps.",
            json!({
                "apply": {"type": "boolean", "default": false},
                "all": {"type": "boolean", "default": false, "description": "Enable backup, validation, compaction, and embedding rebuild checks."},
                "backup": {"type": "boolean", "default": false},
                "backup_output": {"type": "string"},
                "validate_pending": {"type": "boolean", "default": false},
                "validate_limit": limit_schema(20, 500),
                "compact": {"type": "boolean", "default": false},
                "compact_limit": limit_schema(40, 500),
                "compact_min_memories": limit_schema(20, 500),
                "feedback": {"type": "boolean", "default": false, "description": "Apply or preview unapplied dukememory_feedback audit events."},
                "feedback_limit": limit_schema(100, 500),
                "embed_missing": {"type": "boolean", "default": false},
                "embed_limit": limit_schema(50, 500),
                "embed_scope": {"type": "string", "enum": ["all", "memories", "memory", "code", "symbols", "code_symbols"], "default": "all"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_ops_pipeline",
            "Build a safe operations pipeline report: health, project status, code status, consistency gate, and maintenance dry-run.",
            json!({
                "query": {"type": "string"},
                "limit": limit_schema(20, 200),
                "validate_limit": limit_schema(20, 500),
                "compact_limit": limit_schema(40, 500),
                "compact_min_memories": limit_schema(20, 500),
                "embed_missing": {"type": "boolean", "default": true},
                "embed_limit": limit_schema(50, 500),
                "embed_scope": {"type": "string", "enum": ["all", "memories", "memory", "code", "symbols", "code_symbols"], "default": "all"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_status",
            "Show memory counts by lifecycle status and confirm project isolation.",
            json!({
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_health",
            "Show PostgreSQL storage metrics, embedding coverage counts, graph counts, database size, and temporary schema count.",
            json!({}),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_cleanup_schemas",
            "List or drop temporary PostgreSQL schemas created by smoke tests and isolated audits. Defaults to dry-run; set apply=true to drop.",
            json!({
                "apply": {"type": "boolean", "default": false}
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_backup",
            "Create a consistent PostgreSQL backup of the whole dukememory database. Output defaults to a timestamped file next to the configured database marker path.",
            json!({
                "output": {"type": "string", "description": "Optional absolute output path for the PostgreSQL custom-format backup. Must not already exist."}
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_export",
            "Export one project to portable JSON without embedding blobs. Use this before moving memory to another machine or database.",
            json!({
                "output": {"type": "string", "description": "Absolute path where the JSON export should be written."},
                "include_code": {"type": "boolean", "default": true, "description": "Include indexed code files, symbols, and relations. Embeddings are never exported."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["output"],
        ),
        tool_schema(
            "dukememory_import",
            "Import a dukememory project JSON export. Defaults to non-destructive mode and skips existing ids.",
            json!({
                "file": {"type": "string", "description": "Absolute path to a JSON export created by dukememory_export."},
                "overwrite": {"type": "boolean", "default": false, "description": "When true, replace only the exported project_id inside the local database."}
            }),
            vec!["file"],
        ),
        tool_schema(
            "dukememory_code_index",
            "Incrementally index Rust code for one project using tree-sitter. Use full_rebuild to replace the previous code index.",
            json!({
                "project_path": project_path_schema(),
                "project_id": project_id_schema(),
                "full_rebuild": {"type": "boolean", "default": false, "description": "Delete and rebuild the full code index for this project instead of only changed/deleted files."},
                "embed_symbols": {"type": "boolean", "default": false, "description": "Also build embeddings for symbols in files indexed during this run."},
                "embed_symbol_limit": limit_schema(500, 500)
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_lsif_index",
            "Import rust-analyzer LSIF definitions/references into the code graph as RA-backed ra_reference relations. Run after dukememory_code_index.",
            json!({
                "project_path": project_path_schema(),
                "project_id": project_id_schema(),
                "input": {"type": "string", "description": "Optional path to an existing rust-analyzer LSIF JSON-lines file. When omitted, dukememory runs rust-analyzer lsif for project_path."}
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_status",
            "Show indexed code counts for one project.",
            json!({
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_files",
            "List all indexed code files in the project.",
            json!({
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_outline",
            "Outline all code symbols in a specific indexed file.",
            json!({
                "file_path": {"type": "string", "description": "Project-relative path to the code file (e.g. src/main.rs)."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["file_path"],
        ),
        tool_schema(
            "dukememory_code_search",
            "Search indexed code symbols with PostgreSQL full-text search and optional vectors. Use after dukememory_code_index.",
            json!({
                "query": {"type": "string"},
                "limit": limit_schema(10, 50),
                "mode": search_mode_schema("hybrid"),
                "kind": {"type": "string", "description": "Optional symbol kind: function, struct, enum, trait, impl, module, const, static, type, macro."},
                "file_path": {"type": "string", "description": "Optional project-relative path filter."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_code_explore",
            "One-call code exploration: returns relevant indexed symbols, related code memories, route hints, impact, and freshness information.",
            json!({
                "query": {"type": "string", "description": "Natural-language question, symbol names, file paths, or a flow to explore."},
                "limit": limit_schema(8, 25),
                "relation_limit": limit_schema(12, 50),
                "mode": search_mode_schema("hybrid"),
                "kind": {"type": "string", "description": "Optional symbol kind filter."},
                "file_path": {"type": "string", "description": "Optional project-relative path filter."},
                "include_body": {"type": "boolean", "default": true, "description": "Include indexed symbol bodies in the text response."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_code_memory",
            "Create, search, list, promote, archive, or repair durable code memories linked to indexed symbols or files.",
            json!({
                "action": {"type": "string", "enum": ["remember", "search", "list", "promote", "archive", "repair"], "default": "search"},
                "query": {"type": "string", "description": "Search query for action=search."},
                "body": {"type": "string", "description": "Code memory body for action=remember."},
                "id": {"type": "string", "description": "Code memory id for action=promote/archive."},
                "symbol_id": {"type": "string", "description": "Indexed code symbol id to link/filter."},
                "symbol": {"type": "string", "description": "Indexed symbol id or exact unique symbol name to link/filter."},
                "symbol_kind": {"type": "string", "description": "Optional indexed symbol kind disambiguator for symbol names, such as function, struct, enum, trait, impl, module."},
                "file_path": {"type": "string", "description": "Project-relative file path to link/filter and disambiguate symbol names."},
                "kind": {"type": "string", "default": "note", "description": "Memory kind such as invariant, risk, usage, test_note, note."},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
                "confidence": score_schema(0.8),
                "status": {"type": "string", "enum": ["pending", "active", "archived", "any"], "default": "active", "description": "For remember, default is pending; for search/list, default is active."},
                "deduplicate": {"type": "boolean", "default": true},
                "reason": {"type": "string", "description": "Archive reason."},
                "apply": {"type": "boolean", "default": false, "description": "For action=repair, apply unique stale symbol relinks instead of dry-run."},
                "limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_affected",
            "Find likely affected test files from changed source files using indexed code relations.",
            json!({
                "files": {"type": "array", "items": {"type": "string"}, "description": "Changed project-relative source files."},
                "file": {"type": "string", "description": "Single changed project-relative source file."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
                "limit": limit_schema(100, 500),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_patterns",
            "Find reusable implementation patterns near a task using hybrid code search and code-symbol embeddings.",
            json!({
                "query": {"type": "string"},
                "limit": limit_schema(8, 25),
                "kind": {"type": "string"},
                "file_path": {"type": "string"},
                "mode": search_mode_schema("hybrid"),
                "min_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.72},
                "apply_memory_suggestions": {"type": "boolean", "default": false},
                "promote_patterns": {"type": "boolean", "default": false},
                "apply_limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_code_duplicates",
            "Find near-duplicate indexed code symbols in one project using stored code embeddings.",
            json!({
                "limit": limit_schema(20, 200),
                "kind": {"type": "string"},
                "file_path": {"type": "string"},
                "min_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.92},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_assist",
            "Build a full embedding-assisted development report: relevant symbols, reusable patterns, duplicate code candidates, affected tests, and pending code-memory suggestions.",
            json!({
                "query": {"type": "string"},
                "limit": limit_schema(8, 25),
                "mode": search_mode_schema("hybrid"),
                "pattern_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.72},
                "duplicate_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.92},
                "apply_memory_suggestions": {"type": "boolean", "default": false},
                "promote_patterns": {"type": "boolean", "default": false},
                "apply_limit": limit_schema(20, 100),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_code_review_plan",
            "Build a review plan from changed files: changed symbols, affected tests, test commands, duplicate candidates, and pending code-memory suggestions.",
            json!({
                "files": {"type": "array", "items": {"type": "string"}},
                "file": {"type": "string"},
                "query": {"type": "string"},
                "limit": limit_schema(50, 200),
                "duplicate_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.92},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_eval",
            "Run code-search eval cases against indexed code symbols. Cases can assert expected symbol ids, symbol names, required text, and forbidden text.",
            json!({
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "query": {"type": "string"},
                            "expected_ids": {"type": "array", "items": {"type": "string"}},
                            "expected_symbols": {"type": "array", "items": {"type": "string"}},
                            "expected_contains": {"type": "array", "items": {"type": "string"}},
                            "forbidden_contains": {"type": "array", "items": {"type": "string"}},
                            "min_results": {"type": "integer", "minimum": 0}
                        },
                        "required": ["name", "query"]
                    }
                },
                "limit": limit_schema(8, 50),
                "mode": search_mode_schema("hybrid"),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["cases"],
        ),
        tool_schema(
            "dukememory_read_symbol",
            "Read one indexed code symbol by symbol id or exact symbol name.",
            json!({
                "symbol": {"type": "string"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["symbol"],
        ),
        tool_schema(
            "dukememory_code_brief",
            "Explain one indexed code symbol using a configured code model.",
            json!({
                "symbol": {"type": "string"},
                "model_role": {"type": "string", "default": "fast_code", "enum": ["fast_code", "deep_code", "agent_code", "experiment"]},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["symbol"],
        ),
        tool_schema(
            "dukememory_code_plan",
            "Build an implementation plan from active memories and indexed code hits.",
            json!({
                "query": {"type": "string"},
                "memory_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "model_role": {"type": "string", "default": "agent_code", "enum": ["fast_code", "deep_code", "agent_code", "experiment"]},
                "mode": search_mode_schema("hybrid"),
                "deterministic": {"type": "boolean", "default": false, "description": "Build a plan from indexed symbols, embedding patterns, duplicate candidates, and affected tests without calling an LLM."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_code_risk",
            "Analyze likely risks and impacted code from active memories and indexed code hits.",
            json!({
                "query": {"type": "string"},
                "memory_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "code_limit": {"type": "integer", "minimum": 0, "maximum": 30, "default": 8},
                "model_role": {"type": "string", "default": "deep_code", "enum": ["fast_code", "deep_code", "agent_code", "experiment"]},
                "mode": search_mode_schema("hybrid"),
                "deterministic": {"type": "boolean", "default": false, "description": "Analyze risk from indexed symbols, embedding patterns, duplicate candidates, and affected tests without calling an LLM."},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_find_callers",
            "Find approximate callers for an indexed Rust symbol.",
            json!({
                "symbol": {"type": "string"},
                "limit": limit_schema(50, 200),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["symbol"],
        ),
        tool_schema(
            "dukememory_find_callees",
            "Find approximate callees for an indexed Rust symbol.",
            json!({
                "symbol": {"type": "string"},
                "limit": limit_schema(50, 200),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["symbol"],
        ),
        tool_schema(
            "dukememory_impact",
            "Show approximate callers and callees for an indexed Rust symbol.",
            json!({
                "symbol": {"type": "string"},
                "limit": limit_schema(50, 200),
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            vec!["symbol"],
        ),
        tool_schema(
            "dukememory_semantic",
            "Run embedding-backed semantic operations. Actions: dedupe, related, review, route, clusters, tag, stale, consistency, eval_cases, hard_negatives, health, migration, isolation_check, hints, policy, retrieval_quality, auto_eval, ab_compare, lifecycle, code_memory_suggest, verify_conflicts, topic_map, budget_optimize, feedback.",
            semantic_tool_properties(Some(json!([
                "dedupe",
                "related",
                "review",
                "route",
                "clusters",
                "tag",
                "stale",
                "consistency",
                "eval_cases",
                "hard_negatives",
                "health",
                "migration",
                "isolation_check",
                "hints",
                "policy",
                "retrieval_quality",
                "auto_eval",
                "ab_compare",
                "lifecycle",
                "code_memory_suggest",
                "verify_conflicts",
                "topic_map",
                "budget_optimize",
                "feedback",
                "context_policy",
                "trace",
                "counterfactual_eval",
                "causality",
                "temporal_context"
            ]))),
            vec!["action"],
        ),
        tool_schema(
            "dukememory_dedupe",
            "Find semantic near-duplicate memories in one project using stored embeddings.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_related",
            "Find semantically related memories and code symbols for a query, memory_id, or symbol.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_semantic_review",
            "Compare a proposed memory body against active memories and suggest duplicate, conflict, or supersede review actions.",
            semantic_tool_properties(None),
            vec!["body"],
        ),
        tool_schema(
            "dukememory_semantic_route",
            "Classify a query into memory/code/graph retrieval routing weights.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_semantic_clusters",
            "Cluster active memories by semantic similarity for compaction planning.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_semantic_tags",
            "Suggest stable tags for memories from lexical evidence and project tag examples.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_stale_check",
            "Find code memories whose symbol links or semantic overlap look stale.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_consistency_check",
            "Run a project consistency gate for stale code memories, memory conflicts, duplicate candidates, pending-memory backlog, and optional code-index freshness.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_eval_generate",
            "Generate retrieval eval cases and hard-negative ids from existing memories.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_hard_negatives",
            "Mine semantically close but non-expected memories for retrieval eval hard negatives.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_embedding_health",
            "Show project embedding coverage by current and stored models.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_model_migration",
            "Plan memory/code embedding model migration by comparing target model coverage.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_isolation_check",
            "Check that a semantic query does not leak identical memory ids across two explicit projects.",
            semantic_tool_properties(None),
            vec!["query", "other_project_id"],
        ),
        tool_schema(
            "dukememory_memory_hints",
            "Return agent memory hints: semantic route, related memories/code, and stale candidates.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_policy_decision",
            "Decide whether a proposed memory should be inserted, skipped as a duplicate, reviewed, or treated as a supersede candidate.",
            semantic_tool_properties(None),
            vec!["body"],
        ),
        tool_schema(
            "dukememory_retrieval_quality",
            "Score retrieval quality for a query using redundancy, diversity, relevance, and optional expected memory ids.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_auto_eval",
            "Create a retrieval eval case from a real agent task, query, or memory proposal.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_ab_compare",
            "Compare current and target embedding-model retrieval results for one query.",
            semantic_tool_properties(None),
            vec!["query", "target_memory_model"],
        ),
        tool_schema(
            "dukememory_lifecycle_review",
            "Build a semantic lifecycle review bundle with dedupe, stale, tag, cluster, and embedding health signals.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_code_memory_suggest",
            "Suggest pending code-memory notes from indexed symbols related to a query or file path.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_verify_conflicts",
            "Use semantic conflict candidates and the configured LLM verifier to review a proposed memory body.",
            semantic_tool_properties(None),
            vec!["body"],
        ),
        tool_schema(
            "dukememory_topic_map",
            "Summarize semantic topic/tag distribution and ontology drift signals for project memories.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_budget_optimize",
            "Compare retrieval limit options and recommend a context budget setting for a query.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_feedback",
            "Record agent outcome feedback for retrieved memory ids as an audit event.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_self_heal",
            "Run the autonomous memory self-healing loop: outcome learning, conflict graph, memory compiler, lifecycle review, and auditable apply.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_outcome_learn",
            "Learn memory quality signals from completed or failed task sessions and optionally apply them to ranking feedback.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_conflict_graph",
            "Find memory and graph-fact contradictions and optionally invalidate weaker temporal graph facts.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_memory_compiler",
            "Compile active project memory by promoting stable rules to core, archiving low-signal duplicates, and creating pending split candidates.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_policy_ab",
            "Run live A/B retrieval policy trials and recommend the strongest memory/code/mode policy for a query.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_context_policy",
            "Learn a context retrieval policy for one task from retrieval events, feedback, and a live quality probe.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_trace",
            "Build an agent flight-recorder trace from a task session id, retrieval event id, or query.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_task_replay",
            "Replay-ready alias for dukememory_trace, returning deterministic retrieval inputs and selected memory/code ids.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_counterfactual_eval",
            "Run leave-one-out retrieval counterfactuals and generate hard-negative eval signals for a query.",
            semantic_tool_properties(None),
            vec!["query"],
        ),
        tool_schema(
            "dukememory_code_causality",
            "Build memory-to-code causal links from selected memories, code symbols, code graph relations, and affected tests.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_memory_impact",
            "Alias for dukememory_code_causality focused on memory impact over code files and tests.",
            semantic_tool_properties(None),
            Vec::new(),
        ),
        tool_schema(
            "dukememory_temporal_context",
            "Read historical memory and temporal graph context for a query at an as_of timestamp.",
            semantic_tool_properties(None),
            vec!["query", "as_of"],
        ),
        tool_schema(
            "dukememory_embed_missing",
            "Build missing Ollama embeddings for stored memories and/or indexed code symbols in one project.",
            json!({
                "limit": limit_schema(50, 500),
                "scope": {"type": "string", "enum": ["all", "memories", "memory", "code", "symbols", "code_symbols"], "default": "all"},
                "project_id": project_id_schema(),
                "project_path": project_path_schema()
            }),
            Vec::new(),
        ),
    ])
}

fn tool_schema(name: &str, description: &str, properties: Value, required: Vec<&str>) -> Value {
    let mut properties = properties;
    if properties.get("project_path").is_some()
        && let Value::Object(fields) = &mut properties
    {
        fields.insert("roots".to_string(), roots_schema());
    }
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    })
}

fn limit_schema(default: u64, maximum: u64) -> Value {
    json!({
        "type": "integer",
        "minimum": 1,
        "maximum": maximum,
        "default": default
    })
}

fn score_schema(default: f64) -> Value {
    json!({
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": default
    })
}

fn status_schema(default: &str) -> Value {
    json!({
        "type": "string",
        "enum": ["pending", "active", "superseded", "archived", "any"],
        "default": default
    })
}

fn search_mode_schema(default: &str) -> Value {
    json!({
        "type": "string",
        "enum": ["keyword", "semantic", "hybrid", "rerank"],
        "default": default
    })
}

fn semantic_tool_properties(action_enum: Option<Value>) -> Value {
    let action = match action_enum {
        Some(values) => json!({
            "type": "string",
            "enum": values,
            "description": "Semantic operation action."
        }),
        None => json!({
            "type": "string",
            "description": "Optional override for generic dukememory_semantic only."
        }),
    };
    json!({
        "action": action,
        "query": {"type": "string", "description": "Natural language query for related, route, eval, hard negatives, isolation, or hints."},
        "input": {"type": "string", "description": "Raw agent task/input text for auto-eval, policy, feedback, or budget analysis."},
        "body": {"type": "string", "description": "Proposed memory body for semantic review/conflict/supersede suggestions."},
        "memory_id": {"type": "string", "description": "Source memory id for related lookup or hard-negative mining."},
        "id": {"type": "string", "description": "Alias for memory_id."},
        "symbol": {"type": "string", "description": "Indexed code symbol id or exact symbol name for related lookup."},
        "file_path": {"type": "string", "description": "Indexed file path filter for code-memory suggestions."},
        "other_project_id": {"type": "string", "description": "Explicit second project id for isolation checks."},
        "expected_ids": {"type": "array", "items": {"type": "string"}, "description": "Expected memory ids used by hard negative mining."},
        "forbidden_ids": {"type": "array", "items": {"type": "string"}, "description": "Memory ids that must not appear in eval results."},
        "retrieval_event_id": {"type": "string", "description": "Retrieval event id this feedback applies to."},
        "outcome_kind": {"type": "string", "enum": ["wrong_memory", "missing_memory", "stale_memory", "contradiction", "bad_code_hit", "bad_answer", "bug_regression"], "default": "wrong_memory"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
        "apply": {"type": "boolean", "default": false, "description": "Apply auditable lifecycle/feedback/graph changes. Omit for dry-run planning."},
        "helpful_ids": {"type": "array", "items": {"type": "string"}, "description": "Memory ids marked helpful by an agent outcome feedback event."},
        "unhelpful_ids": {"type": "array", "items": {"type": "string"}, "description": "Memory ids marked unhelpful by an agent outcome feedback event."},
        "limit": limit_schema(20, 500),
        "status": status_schema("active"),
        "kind": {"type": "string"},
        "memory_tier": {"type": "string", "enum": ["core", "archival", "conversation"]},
        "mode": search_mode_schema("hybrid"),
        "min_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
        "target_memory_model": {"type": "string", "description": "Target memory embedding model for migration planning."},
        "target_code_model": {"type": "string", "description": "Target code embedding model for migration planning."},
        "as_of": {"type": "string", "description": "Timestamp for temporal context reads, accepted by PostgreSQL timestamptz parsing."},
        "project_id": project_id_schema(),
        "project_path": project_path_schema()
    })
}

fn memory_status_schema(default: &str) -> Value {
    json!({
        "type": "string",
        "enum": ["pending", "active", "superseded", "archived"],
        "default": default
    })
}

fn project_id_schema() -> Value {
    json!({
        "type": "string",
        "description": "Explicit project id. Omit unless the user asked for a specific project."
    })
}

fn project_path_schema() -> Value {
    json!({
        "type": "string",
        "description": "Absolute path to the current project root. Prefer this over project_id when available."
    })
}

fn roots_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "path": {"type": "string"},
                        "name": {"type": "string"}
                    }
                }
            ]
        },
        "description": "Optional client workspace roots. When supplied with project_path, the path must be inside one root."
    })
}

fn resolve_project(arguments: &Value) -> Result<String> {
    if let Some(project_id) = optional_string(arguments, "project_id") {
        return resolve_project_id(Some(project_id));
    }
    if let Some(project_path) = optional_string(arguments, "project_path") {
        validate_project_path_roots(&project_path, arguments)?;
        return resolve_project_id_from_path(project_path);
    }
    resolve_project_id(None)
}

fn validate_project_path_roots(project_path: &str, arguments: &Value) -> Result<()> {
    let Some(roots) = arguments.get("roots").and_then(Value::as_array) else {
        return Ok(());
    };
    if roots.is_empty() {
        return Ok(());
    }
    let project_path = canonical_or_absolute(Path::new(project_path))?;
    let mut root_paths = Vec::new();
    for root in roots {
        let raw = if let Some(value) = root.as_str() {
            Some(value)
        } else {
            root.get("path")
                .and_then(Value::as_str)
                .or_else(|| root.get("uri").and_then(Value::as_str))
        };
        let Some(raw) = raw else {
            continue;
        };
        let raw = raw.strip_prefix("file://").unwrap_or(raw);
        root_paths.push(canonical_or_absolute(Path::new(raw))?);
    }
    if root_paths
        .iter()
        .any(|root_path| project_path.starts_with(root_path))
    {
        Ok(())
    } else {
        bail!(
            "project_path `{}` is outside supplied roots",
            project_path.display()
        )
    }
}

fn canonical_or_absolute(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return path
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {}", path.display()));
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn status_filter_arg(arguments: &Value, default: MemoryStatus) -> Result<StatusFilter> {
    StatusFilter::parse(optional_string(arguments, "status").as_deref(), default)
}

fn memory_status_arg(arguments: &Value, default: MemoryStatus) -> Result<MemoryStatus> {
    match optional_string(arguments, "status") {
        Some(value) => MemoryStatus::parse(&value),
        None => Ok(default),
    }
}

fn required_string(arguments: &Value, field: &str) -> Result<String> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow!("missing required string argument `{field}`"))
}

fn optional_string(arguments: &Value, field: &str) -> Option<String> {
    arguments
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.to_string())
}

fn optional_path(arguments: &Value, field: &str) -> Result<Option<PathBuf>> {
    let Some(value) = optional_string(arguments, field) else {
        return Ok(None);
    };
    if field == "project_path" {
        validate_project_path_roots(&value, arguments)?;
    }
    Ok(Some(PathBuf::from(value)))
}

fn optional_u64(arguments: &Value, field: &str) -> Option<u64> {
    arguments.get(field).and_then(Value::as_u64)
}

fn optional_f64(arguments: &Value, field: &str) -> Option<f64> {
    arguments.get(field).and_then(Value::as_f64)
}

fn optional_bool(arguments: &Value, field: &str) -> Option<bool> {
    arguments.get(field).and_then(Value::as_bool)
}

fn optional_string_array(arguments: &Value, field: &str) -> Result<Vec<String>> {
    let Some(value) = arguments.get(field) else {
        return Ok(Vec::new());
    };
    let array = value
        .as_array()
        .ok_or_else(|| anyhow!("`{field}` must be an array of strings"))?;
    array
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(|value| value.to_string())
                .ok_or_else(|| anyhow!("`{field}` must be an array of strings"))
        })
        .collect()
}

fn optional_string_array_field(arguments: &Value, field: &str) -> Result<Option<Vec<String>>> {
    if arguments.get(field).is_some() {
        optional_string_array(arguments, field).map(Some)
    } else {
        Ok(None)
    }
}

fn strings_from_value(value: Option<&Value>) -> Result<Vec<String>> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("expected array of strings"))
            })
            .collect(),
        Some(_) => Err(anyhow!("expected array of strings")),
    }
}

fn tool_result(text: String, structured: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "structuredContent": structured
    })
}

fn model_roles_json(config: &Config) -> Value {
    let roles = config
        .model_roles()
        .into_iter()
        .map(|(role, model)| (role.to_string(), Value::String(model.to_string())))
        .collect::<serde_json::Map<_, _>>();
    Value::Object(roles)
}

fn tool_error(error: anyhow::Error) -> Value {
    json!({
        "code": -32603,
        "message": error.to_string()
    })
}

fn invalid_params(message: &str) -> Value {
    json!({
        "code": -32602,
        "message": message
    })
}

fn write_message(stdout: &mut impl Write, message: Value) -> Result<()> {
    serde_json::to_writer(&mut *stdout, &message).context("failed to write MCP response")?;
    stdout
        .write_all(b"\n")
        .context("failed to finish MCP response")?;
    stdout.flush().context("failed to flush MCP stdout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_mode_filters_tool_calls() {
        assert_eq!(AuditMode::parse("off").unwrap(), AuditMode::Off);
        assert_eq!(
            AuditMode::parse("writes_only").unwrap(),
            AuditMode::WritesOnly
        );
        assert_eq!(AuditMode::parse("all").unwrap(), AuditMode::All);
        assert!(AuditMode::parse("surprise").is_err());

        assert!(!should_audit_tool_call(
            AuditMode::Off,
            "dukememory_remember"
        ));
        assert!(should_audit_tool_call(AuditMode::All, "dukememory_status"));
        assert!(should_audit_tool_call(
            AuditMode::WritesOnly,
            "dukememory_remember"
        ));
        assert!(should_audit_tool_call(
            AuditMode::WritesOnly,
            "dukememory_code_index"
        ));
        assert!(!should_audit_tool_call(
            AuditMode::WritesOnly,
            "dukememory_search"
        ));
        assert!(!should_audit_tool_call(
            AuditMode::WritesOnly,
            "dukememory_audit_log"
        ));
    }

    #[test]
    fn audit_metrics_summarize_recent_events() {
        let events = vec![
            crate::store::AuditEvent {
                id: "1".to_string(),
                project_id: "p".to_string(),
                actor: "mcp".to_string(),
                action: "dukememory_remember".to_string(),
                target_type: "tool_call".to_string(),
                target_id: None,
                detail: json!({}),
                created_at: "now".to_string(),
            },
            crate::store::AuditEvent {
                id: "2".to_string(),
                project_id: "p".to_string(),
                actor: "mcp".to_string(),
                action: "dukememory_code_index".to_string(),
                target_type: "tool_call".to_string(),
                target_id: None,
                detail: json!({}),
                created_at: "now".to_string(),
            },
        ];
        let metrics = audit_metrics(&events);
        assert_eq!(metrics["total"], 2);
        assert_eq!(metrics["by_actor"]["mcp"], 2);
        assert_eq!(metrics["by_target_type"]["tool_call"], 2);
        assert_eq!(metrics["by_action"]["dukememory_remember"], 1);
    }

    fn documented_tool_set(
        document: &str,
        start_marker: &str,
        end_marker: &str,
    ) -> std::collections::BTreeSet<String> {
        let start = document
            .find(start_marker)
            .expect("documentation start marker exists");
        let end = document[start..]
            .find(end_marker)
            .map(|offset| start + offset)
            .expect("documentation end marker exists");
        document[start..end]
            .lines()
            .filter_map(|line| {
                line.trim()
                    .strip_prefix("- `")
                    .and_then(|value| value.strip_suffix('`'))
                    .filter(|name| name.starts_with("dukememory_"))
                    .map(str::to_string)
            })
            .collect()
    }

    #[test]
    fn documented_mcp_tool_lists_match_registry() {
        let registry = tools()
            .as_array()
            .expect("tools registry is an array")
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .map(str::to_string)
            .collect::<std::collections::BTreeSet<_>>();
        let readme_tools = documented_tool_set(
            include_str!("../README.md"),
            "Current tools:",
            "MCP also exposes read-only resources",
        );
        let agent_tools = documented_tool_set(
            include_str!("../AGENTS.md"),
            "Use:",
            "Do not introduce generic",
        );
        assert_eq!(readme_tools, registry, "README MCP tool list is stale");
        assert_eq!(agent_tools, registry, "AGENTS MCP tool list is stale");
    }

    #[test]
    fn http_status_reason_uses_real_reason_phrases() {
        assert_eq!(http_status_reason(200), "OK");
        assert_eq!(http_status_reason(202), "Accepted");
        assert_eq!(http_status_reason(404), "Not Found");
        assert_eq!(http_status_reason(405), "Method Not Allowed");
        assert_eq!(http_status_reason(413), "Payload Too Large");
        assert_eq!(http_status_reason(500), "Internal Server Error");
    }

    #[test]
    fn http_rejects_wrong_method_with_405() {
        let response =
            http_roundtrip("PUT /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n");
        assert!(
            response.starts_with("HTTP/1.1 405 Method Not Allowed"),
            "{response:?}"
        );
    }

    #[test]
    fn http_get_mcp_returns_endpoint_event() {
        let response = http_roundtrip("GET /mcp HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response:?}");
        assert!(response.contains("Content-Type: text/event-stream"));
        assert!(response.contains("data: /mcp"));
    }

    #[test]
    fn http_rejects_non_local_origin_with_403() {
        let response = http_roundtrip(
            "POST /mcp HTTP/1.1\r\nHost: localhost\r\nOrigin: https://example.com\r\nContent-Length: 0\r\n\r\n",
        );
        assert!(
            response.starts_with("HTTP/1.1 403 Forbidden"),
            "{response:?}"
        );
    }

    #[test]
    fn http_rejects_oversized_body_with_413_before_reading_body() {
        let request = format!(
            "POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            HTTP_BODY_LIMIT_BYTES + 1
        );
        let response = http_roundtrip(&request);
        assert!(
            response.starts_with("HTTP/1.1 413 Payload Too Large"),
            "{response:?}"
        );
    }

    fn http_roundtrip(request: &str) -> String {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
        let addr = listener.local_addr().expect("listener addr");
        let request = request.to_string();
        let client = std::thread::spawn(move || {
            let mut stream = std::net::TcpStream::connect(addr).expect("connect test server");
            stream
                .write_all(request.as_bytes())
                .expect("write test request");
            stream
                .shutdown(std::net::Shutdown::Write)
                .expect("shutdown write side");
            let mut response = String::new();
            stream
                .read_to_string(&mut response)
                .expect("read test response");
            response
        });
        let (mut server_stream, _) = listener.accept().expect("accept test connection");
        let config = test_config("http-roundtrip");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        runtime
            .block_on(handle_http_connection(&config, &mut server_stream))
            .expect("handle test HTTP request");
        drop(server_stream);
        client.join().expect("client thread")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_smoke_exercises_initialize_and_negative_paths() {
        let config = test_config("http-smoke");
        let report = run_http_smoke(&config).await.expect("HTTP smoke passes");
        assert_eq!(report.server_info_name, "dukememory");
        assert_eq!(report.initialize_status, 200);
        assert_eq!(report.sse_status, 200);
        assert_eq!(report.forbidden_origin_status, 403);
        assert_eq!(report.wrong_method_status, 405);
        assert_eq!(report.oversized_body_status, 413);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn initialize_and_tools_list_expose_dukememory_context() {
        let config = test_config("tools-list");
        let initialize = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            }),
        )
        .await
        .expect("initialize returns response");
        assert_eq!(initialize["result"]["serverInfo"]["name"], "dukememory");
        assert!(initialize["result"]["capabilities"]["resources"].is_object());
        assert!(initialize["result"]["capabilities"]["prompts"].is_object());
        assert!(initialize["result"]["capabilities"]["logging"].is_object());

        let tools = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }),
        )
        .await
        .expect("tools/list returns response");
        let names = tools["result"]["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"dukememory_prepare"));
        assert!(names.contains(&"dukememory_context"));
        assert!(names.contains(&"dukememory_extract"));
        assert!(names.contains(&"dukememory_agent_task"));
        assert!(names.contains(&"dukememory_task_eval"));
        assert!(names.contains(&"dukememory_test_plan"));
        assert!(names.contains(&"dukememory_remember_smart"));
        assert!(names.contains(&"dukememory_models"));
        assert!(names.contains(&"dukememory_review_apply"));
        assert!(names.contains(&"dukememory_audit_log"));
        assert!(names.contains(&"dukememory_validate_pending"));
        assert!(names.contains(&"dukememory_compact"));
        assert!(names.contains(&"dukememory_maintenance"));
        assert!(names.contains(&"dukememory_ops_pipeline"));
        assert!(names.contains(&"dukememory_backup"));
        assert!(names.contains(&"dukememory_export"));
        assert!(names.contains(&"dukememory_import"));
        assert!(names.contains(&"dukememory_code_lsif_index"));
        assert!(names.contains(&"dukememory_code_status"));
        assert!(names.contains(&"dukememory_code_files"));
        assert!(names.contains(&"dukememory_code_outline"));
        assert!(names.contains(&"dukememory_code_brief"));
        assert!(names.contains(&"dukememory_code_plan"));
        assert!(names.contains(&"dukememory_code_risk"));
        assert!(names.contains(&"dukememory_code_patterns"));
        assert!(names.contains(&"dukememory_code_duplicates"));
        assert!(names.contains(&"dukememory_code_assist"));
        assert!(names.contains(&"dukememory_code_review_plan"));
        assert!(names.contains(&"dukememory_code_eval"));
        assert!(names.contains(&"dukememory_consistency_check"));
        assert!(names.iter().all(|name| name.starts_with("dukememory_")));
        let graph_tool = tools["result"]["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .find(|tool| tool["name"] == "dukememory_graph")
            .expect("dukememory_graph tool");
        let graph_actions = graph_tool["inputSchema"]["properties"]["action"]["enum"]
            .as_array()
            .expect("graph action enum")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        assert!(graph_actions.contains(&"invalidate_fact"));
        assert!(graph_actions.contains(&"invalidate_edge"));

        let resources = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/list",
                "params": {}
            }),
        )
        .await
        .expect("resources/list returns response");
        let resource_uris = resources["result"]["resources"]
            .as_array()
            .expect("resources array")
            .iter()
            .filter_map(|resource| resource["uri"].as_str())
            .collect::<Vec<_>>();
        assert!(resource_uris.contains(&"dukememory://ontology"));
        assert!(resource_uris.contains(&"dukememory://health"));

        let resource_templates = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/templates/list",
                "params": {}
            }),
        )
        .await
        .expect("resources/templates/list returns response");
        let template_uris = resource_templates["result"]["resourceTemplates"]
            .as_array()
            .expect("resourceTemplates array")
            .iter()
            .filter_map(|template| template["uriTemplate"].as_str())
            .collect::<Vec<_>>();
        assert!(template_uris.contains(&"dukememory://project/{project_id}/profile"));

        let prompts = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "prompts/list",
                "params": {}
            }),
        )
        .await
        .expect("prompts/list returns response");
        let prompt_names = prompts["result"]["prompts"]
            .as_array()
            .expect("prompts array")
            .iter()
            .filter_map(|prompt| prompt["name"].as_str())
            .collect::<Vec<_>>();
        assert!(prompt_names.contains(&"dukememory_agent_before"));
        assert!(prompt_names.contains(&"dukememory_agent_after"));
        assert!(prompt_names.contains(&"dukememory_agent_workflow"));

        let workflow_prompt = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 55,
                "method": "prompts/get",
                "params": {
                    "name": "dukememory_agent_workflow",
                    "arguments": {
                        "query": "change GUI graph",
                        "project_path": "/tmp/project"
                    }
                }
            }),
        )
        .await
        .expect("prompts/get returns workflow response");
        let workflow_text = workflow_prompt["result"]["messages"][0]["content"]["text"]
            .as_str()
            .expect("workflow prompt text");
        assert!(workflow_text.contains("dukememory_prepare"));
        assert!(workflow_text.contains("dukememory_agent_after"));

        let ontology = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": "resources/read",
                "params": {
                    "uri": "dukememory://ontology"
                }
            }),
        )
        .await
        .expect("resources/read returns response");
        let ontology_text = ontology["result"]["contents"][0]["text"]
            .as_str()
            .expect("ontology text");
        assert!(ontology_text.contains("memory_tiers"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn smoke_runner_exercises_core_mcp_memory_flow() -> Result<()> {
        let config = test_config("smoke-runner");
        let report = run_smoke(&config).await?;

        assert!(report.tools_count >= 4);
        assert_eq!(report.search_hits, 1);
        assert_eq!(report.context_hits, 1);
        assert_eq!(report.cross_project_hits, 0);
        assert_eq!(report.total_memories, 1);
        assert_eq!(report.active_memories, 1);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn context_tool_works_in_keyword_mode_without_ollama() {
        let config = test_config("context-keyword");
        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_context",
                    "arguments": {
                        "project_id": "mcp-test",
                        "query": "inventory retrieval",
                        "mode": "keyword",
                        "memory_limit": 4,
                        "code_limit": 0
                    }
                }
            }),
        )
        .await
        .expect("context returns response");
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(
            response["result"]["structuredContent"]["project_id"],
            "mcp-test"
        );
        assert_eq!(response["result"]["structuredContent"]["mode"], "keyword");
        let retrieval = &response["result"]["structuredContent"]["retrieval"];
        assert_eq!(retrieval["requested_mode"], "keyword");
        assert_eq!(retrieval["memory_actual_mode"], "keyword");
        assert_eq!(retrieval["code_actual_mode"], "disabled");
        assert_eq!(retrieval["intent"], "architecture");
        assert!(
            retrieval["sources"]
                .as_array()
                .expect("retrieval sources")
                .iter()
                .any(|source| source["domain"] == "memory" && source["source"] == "keyword")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn context_tool_returns_task_scoped_fragments_without_full_memory_bodies() {
        let config = test_config("context-fragments");
        let store = Store::open(&config.database_marker).expect("store opens");
        store
            .remember(
                "mcp-test",
                NewMemory {
                    scope: DEFAULT_MEMORY_SCOPE.to_string(),
                    memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                    kind: "decision".to_string(),
                    body: "Inventory retrieval uses scoped fragments for matching tasks.\n\nDeployment runbook text should not be needed for inventory questions.".to_string(),
                    tags: vec!["retrieval".to_string()],
                    source: Some("test".to_string()),
                    status: MemoryStatus::Active,
                    importance: 0.8,
                    confidence: 0.9,
                    status_reason: None,
                    allow_sensitive: false,
                },
            )
            .expect("memory stored");

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 35,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_context",
                    "arguments": {
                        "project_id": "mcp-test",
                        "query": "inventory scoped fragments",
                        "mode": "keyword",
                        "memory_limit": 4,
                        "code_limit": 0,
                        "token_budget": 1000
                    }
                }
            }),
        )
        .await
        .expect("context returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["task_scoped"], true);
        let fragments = content["memory_fragments"]
            .as_array()
            .expect("memory fragments");
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].get("text").is_none());
        assert!(fragments[0]["text_chars"].as_u64().unwrap_or(0) > 0);
        assert!(content.get("trace").is_none());
        assert!(content.get("graph").is_none());
        assert!(content.get("graph_summary").is_some());
        let text = response["result"]["content"][0]["text"]
            .as_str()
            .expect("tool text");
        assert!(text.contains("scoped fragments"));
        assert!(
            !text.contains("Deployment runbook"),
            "non-selected chunk leaked into prompt text: {text}"
        );
        let memories = content["memories"].as_array().expect("memory summaries");
        assert_eq!(memories.len(), 1);
        assert!(memories[0].get("body").is_none());
        assert_eq!(memories[0]["included_fragments"], 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn feedback_applies_quality_effect_and_creates_regression_case() -> Result<()> {
        let config = test_config("feedback-loop");
        let store = Store::open(&config.database_marker)?;
        let helpful_id = store.remember(
            "mcp-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Use scoped retrieval for inventory tasks.".to_string(),
                tags: vec!["retrieval".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        let unhelpful_id = store.remember(
            "mcp-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Use global retrieval for inventory tasks.".to_string(),
                tags: vec!["retrieval".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        drop(store);

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 77,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_feedback",
                    "arguments": {
                        "project_id": "mcp-test",
                        "query": "inventory retrieval",
                        "helpful_ids": [helpful_id],
                        "unhelpful_ids": [unhelpful_id],
                        "outcome_kind": "wrong_memory",
                        "severity": "high"
                    }
                }
            }),
        )
        .await
        .expect("feedback returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["feedback_effect"]["helpful_memories_updated"], 1);
        assert_eq!(content["feedback_effect"]["unhelpful_memories_updated"], 1);
        assert_eq!(
            content["regression_eval_case"]["forbidden_ids"][0],
            unhelpful_id
        );

        let store = Store::open(&config.database_marker)?;
        let helpful_after = store.get("mcp-test", &helpful_id)?.expect("helpful memory");
        let unhelpful_after = store
            .get("mcp-test", &unhelpful_id)?
            .expect("unhelpful memory");
        assert!(helpful_after.quality_score > unhelpful_after.quality_score);
        assert!(unhelpful_after.contradiction_risk > helpful_after.contradiction_risk);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn memory_compiler_apply_promotes_stable_rule_to_core() -> Result<()> {
        let config = test_config("memory-compiler-apply");
        let store = Store::open(&config.database_marker)?;
        let memory_id = store.remember(
            "mcp-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "project_rule".to_string(),
                body: "Project rule: task replay must use session-linked retrieval events."
                    .to_string(),
                tags: vec!["task-replay".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.9,
                confidence: 0.92,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        drop(store);

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 177,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_memory_compiler",
                    "arguments": {
                        "project_id": "mcp-test",
                        "limit": 20,
                        "apply": true
                    }
                }
            }),
        )
        .await
        .expect("memory compiler returns response");
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(
            response["result"]["structuredContent"]["counts"]["promote_to_core"],
            1
        );

        let store = Store::open(&config.database_marker)?;
        let memory = store.get("mcp-test", &memory_id)?.expect("memory exists");
        assert_eq!(memory.memory_tier, "core");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn conflict_graph_apply_invalidates_weaker_fact() -> Result<()> {
        let config = test_config("conflict-graph-apply");
        let store = Store::open(&config.database_marker)?;
        let entity = store.upsert_memory_entity(
            "mcp-test",
            "module",
            "retrieval",
            Vec::new(),
            Some("Retrieval subsystem".to_string()),
        )?;
        let weaker = store.add_memory_fact(
            "mcp-test",
            Some(&entity.id),
            None,
            "uses",
            "global memory lookup",
            0.45,
        )?;
        let stronger = store.add_memory_fact(
            "mcp-test",
            Some(&entity.id),
            None,
            "uses",
            "project scoped memory lookup",
            0.92,
        )?;
        drop(store);

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 178,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_conflict_graph",
                    "arguments": {
                        "project_id": "mcp-test",
                        "query": "retrieval",
                        "limit": 20,
                        "apply": true
                    }
                }
            }),
        )
        .await
        .expect("conflict graph returns response");
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(
            response["result"]["structuredContent"]["counts"]["invalidated_facts"],
            1
        );

        let store = Store::open(&config.database_marker)?;
        let graph = store.search_memory_graph("mcp-test", "", 20)?;
        assert!(graph.facts.iter().any(|fact| fact.id == stronger.id));
        assert!(!graph.facts.iter().any(|fact| fact.id == weaker.id));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn outcome_learn_apply_uses_completed_session_memory_ids() -> Result<()> {
        let config = test_config("outcome-learn-apply");
        let store = Store::open(&config.database_marker)?;
        let memory_id = store.remember(
            "mcp-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Task sessions should retain retrieved memory ids for outcome learning."
                    .to_string(),
                tags: vec!["task-session".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.7,
                confidence: 0.75,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        let before = store.get("mcp-test", &memory_id)?.expect("memory exists");
        let session = store.create_task_session(NewTaskSession {
            project_id: "mcp-test",
            query: "task session outcome learning",
            status: "completed",
            phase: "done",
            progress: 100,
            result: json!({"tests_passed": true, "accepted": true}),
        })?;
        store.update_task_session(
            "mcp-test",
            &session.id,
            TaskSessionUpdate {
                status: None,
                phase: None,
                progress: None,
                memory_ids: Some(vec![memory_id.clone()]),
                code_symbol_ids: None,
                file_paths: None,
                test_paths: None,
                summary: Some(Some("completed".to_string())),
                result: None,
            },
        )?;
        drop(store);

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 179,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_outcome_learn",
                    "arguments": {
                        "project_id": "mcp-test",
                        "id": session.id,
                        "apply": true
                    }
                }
            }),
        )
        .await
        .expect("outcome learn returns response");
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(
            response["result"]["structuredContent"]["counts"]["helpful_ids"],
            1
        );

        let store = Store::open(&config.database_marker)?;
        let after = store.get("mcp-test", &memory_id)?.expect("memory exists");
        assert!(after.quality_score > before.quality_score);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn context_tool_filters_unrelated_core_rules_by_task_query() {
        let config = test_config("context-core-query-filter");
        let store = Store::open(&config.database_marker).expect("store opens");
        store
            .remember(
                "mcp-test",
                NewMemory {
                    scope: DEFAULT_MEMORY_SCOPE.to_string(),
                    memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                    kind: "project_rule".to_string(),
                    body:
                        "Deployment release checklist requires staging approval and rollback notes."
                            .to_string(),
                    tags: vec!["deployment".to_string()],
                    source: Some("test".to_string()),
                    status: MemoryStatus::Active,
                    importance: 0.99,
                    confidence: 0.99,
                    status_reason: None,
                    allow_sensitive: false,
                },
            )
            .expect("unrelated core rule stored");
        store
            .remember(
                "mcp-test",
                NewMemory {
                    scope: DEFAULT_MEMORY_SCOPE.to_string(),
                    memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                    kind: "decision".to_string(),
                    body: "Invoice webhook retrieval should use task-scoped fragments.".to_string(),
                    tags: vec!["invoice".to_string()],
                    source: Some("test".to_string()),
                    status: MemoryStatus::Active,
                    importance: 0.7,
                    confidence: 0.9,
                    status_reason: None,
                    allow_sensitive: false,
                },
            )
            .expect("task memory stored");

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 36,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_context",
                    "arguments": {
                        "project_id": "mcp-test",
                        "query": "invoice webhook retrieval",
                        "mode": "keyword",
                        "memory_limit": 4,
                        "core_memory_limit": 5,
                        "code_limit": 0,
                        "token_budget": 1000
                    }
                }
            }),
        )
        .await
        .expect("context returns response");
        assert!(response.get("error").is_none(), "{response}");

        let content = &response["result"]["structuredContent"];
        let fragments = content["memory_fragments"]
            .as_array()
            .expect("memory fragments");
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0]["section"], "task");
        let text = response["result"]["content"][0]["text"]
            .as_str()
            .expect("tool text");
        assert!(text.contains("Invoice webhook retrieval"));
        assert!(
            !text.contains("Deployment release checklist"),
            "unrelated core/project-rule memory leaked into prompt text: {text}"
        );
        assert!(content.get("trace").is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remember_smart_writes_pending_and_deduplicates_without_ollama() {
        let config = test_config("remember-smart");
        let body = "Smart memory policy stores pending offline decisions.";
        let first = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 37,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_remember_smart",
                    "arguments": {
                        "project_id": "remember-smart-test",
                        "body": body,
                        "kind": "decision",
                        "tags": ["policy"],
                        "embed": false
                    }
                }
            }),
        )
        .await
        .expect("remember smart returns response");
        assert!(first.get("error").is_none(), "{first}");
        let content = &first["result"]["structuredContent"];
        assert_eq!(content["outcome"]["inserted"], true);
        assert_eq!(content["status"], "pending");
        assert_eq!(content["policy"]["fallback"], true);
        assert!(content["policy_warning"].is_string());

        let duplicate = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 38,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_remember_smart",
                    "arguments": {
                        "project_id": "remember-smart-test",
                        "body": body,
                        "kind": "decision",
                        "tags": ["policy"],
                        "embed": false
                    }
                }
            }),
        )
        .await
        .expect("remember smart duplicate returns response");
        assert!(duplicate.get("error").is_none(), "{duplicate}");
        assert_eq!(
            duplicate["result"]["structuredContent"]["outcome"]["inserted"],
            false
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn task_session_tool_tracks_agent_progress() {
        let config = test_config("task-session");
        let create = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 37,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_task_session",
                    "arguments": {
                        "project_id": "task-session-test",
                        "action": "create",
                        "query": "implement indexed task workflow",
                        "status": "running",
                        "phase": "prepare",
                        "progress": 12,
                        "result": {"step": "start"}
                    }
                }
            }),
        )
        .await
        .expect("task session create returns response");
        assert!(create.get("error").is_none(), "{create}");
        let session_id = create["result"]["structuredContent"]["session"]["id"]
            .as_str()
            .expect("session id")
            .to_string();
        assert_eq!(
            create["result"]["structuredContent"]["session"]["progress"],
            12
        );

        let update = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 38,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_task_session",
                    "arguments": {
                        "project_id": "task-session-test",
                        "action": "update",
                        "id": session_id,
                        "status": "completed",
                        "phase": "done",
                        "progress": 100,
                        "memory_ids": ["mem-1"],
                        "code_symbol_ids": ["sym-1"],
                        "file_paths": ["src/mcp.rs"],
                        "test_paths": ["src/mcp.rs::task_session_tool_tracks_agent_progress"],
                        "summary": "Task workflow is stored with related artifacts.",
                        "result": {"ok": true}
                    }
                }
            }),
        )
        .await
        .expect("task session update returns response");
        assert!(update.get("error").is_none(), "{update}");
        let updated = &update["result"]["structuredContent"]["session"];
        assert_eq!(updated["status"], "completed");
        assert_eq!(updated["phase"], "done");
        assert!(updated["completed_at"].is_string());
        assert_eq!(updated["memory_ids"][0], "mem-1");
        assert_eq!(updated["file_paths"][0], "src/mcp.rs");

        let get = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 39,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_task_session",
                    "arguments": {
                        "project_id": "task-session-test",
                        "action": "get",
                        "id": session_id
                    }
                }
            }),
        )
        .await
        .expect("task session get returns response");
        assert!(get.get("error").is_none(), "{get}");
        assert_eq!(
            get["result"]["structuredContent"]["session"]["summary"],
            "Task workflow is stored with related artifacts."
        );

        let list = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 40,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_task_session",
                    "arguments": {
                        "project_id": "task-session-test",
                        "action": "list",
                        "status": "completed",
                        "limit": 5
                    }
                }
            }),
        )
        .await
        .expect("task session list returns response");
        assert!(list.get("error").is_none(), "{list}");
        let sessions = list["result"]["structuredContent"]["sessions"]
            .as_array()
            .expect("sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0]["id"], session_id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn task_eval_builds_and_runs_case_from_session() -> Result<()> {
        let config = test_config("task-eval");
        let store = Store::open(&config.database_marker)?;
        let memory_id = store.remember(
            "task-eval-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Invoice retrieval uses task scoped memory fragments for agent context."
                    .to_string(),
                tags: vec!["invoice".to_string(), "retrieval".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        let session = store.create_task_session(NewTaskSession {
            project_id: "task-eval-test",
            query: "invoice retrieval task scoped memory",
            status: "completed",
            phase: "ready",
            progress: 100,
            result: json!({}),
        })?;
        store.update_task_session(
            "task-eval-test",
            &session.id,
            TaskSessionUpdate {
                status: None,
                phase: None,
                progress: None,
                memory_ids: Some(vec![memory_id.clone()]),
                code_symbol_ids: None,
                file_paths: None,
                test_paths: None,
                summary: None,
                result: None,
            },
        )?;

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 44,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_task_eval",
                    "arguments": {
                        "project_id": "task-eval-test",
                        "session_id": session.id,
                        "action": "run",
                        "mode": "keyword",
                        "limit": 5
                    }
                }
            }),
        )
        .await
        .expect("task eval returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["case"]["expected_ids"][0], memory_id);
        assert_eq!(content["eval"]["cases"][0]["passed"], true);
        assert_eq!(content["eval"]["passed_cases"], 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_plan_selects_affected_tests_from_session_files() -> Result<()> {
        let config = test_config("test-plan");
        let root = std::env::temp_dir().join(format!(
            "dukememory-test-plan-fixture-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::create_dir_all(root.join("tests"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"test-plan-fixture\"\n",
        )?;
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn invoice_search() { invoice_helper(); }\nfn invoice_helper() {}\n",
        )?;
        std::fs::write(
            root.join("tests/invoice_test.rs"),
            "#[test]\nfn covers_invoice_search() { invoice_search(); }\n",
        )?;
        let mut store = Store::open(&config.database_marker)?;
        index_project(
            &mut store,
            &root,
            Some("test-plan-fixture".to_string()),
            false,
        )?;
        let session = store.create_task_session(NewTaskSession {
            project_id: "test-plan-fixture",
            query: "change invoice search",
            status: "completed",
            phase: "ready",
            progress: 100,
            result: json!({}),
        })?;
        store.update_task_session(
            "test-plan-fixture",
            &session.id,
            TaskSessionUpdate {
                status: None,
                phase: None,
                progress: None,
                memory_ids: None,
                code_symbol_ids: None,
                file_paths: Some(vec!["src/lib.rs".to_string()]),
                test_paths: None,
                summary: None,
                result: None,
            },
        )?;

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 46,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_test_plan",
                    "arguments": {
                        "project_id": "test-plan-fixture",
                        "session_id": session.id,
                        "limit": 20
                    }
                }
            }),
        )
        .await
        .expect("test plan returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["confidence"], "high");
        assert!(
            content["affected_tests"]
                .as_array()
                .expect("affected tests")
                .iter()
                .any(|file| file == "tests/invoice_test.rs")
        );
        assert!(
            content["commands"]
                .as_array()
                .expect("commands")
                .iter()
                .any(|command| command["command"] == "cargo test --test invoice_test")
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_task_tool_prepares_context_and_records_session() -> Result<()> {
        let config = test_config("agent-task");
        let root = std::env::temp_dir().join(format!(
            "dukememory-agent-task-fixture-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::create_dir_all(root.join("tests"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"agent-task-fixture\"\n",
        )?;
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn invoice_search() { invoice_helper(); }\nfn invoice_helper() {}\n",
        )?;
        std::fs::write(
            root.join("tests/invoice_test.rs"),
            "#[test]\nfn covers_invoice_search() { invoice_search(); }\n",
        )?;

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 41,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_agent_task",
                    "arguments": {
                        "project_id": "agent-task-fixture",
                        "project_path": root,
                        "query": "invoice search",
                        "mode": "keyword",
                        "memory_limit": 1,
                        "code_limit": 5,
                        "token_budget": 1000,
                        "include_code_assist": true,
                        "code_assist_limit": 5
                    }
                }
            }),
        )
        .await
        .expect("agent task returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["project_id"], "agent-task-fixture");
        assert_eq!(content["progress"], 100);
        assert_eq!(content["phase"], "ready");
        assert_eq!(content["session"]["status"], "completed");
        assert_eq!(content["session"]["progress"], 100);
        assert_eq!(content["index"]["files_indexed"], 2);
        assert!(content["consistency"]["readiness_score"].is_number());
        assert!(content["session"]["result"]["consistency"]["status"].is_string());
        assert!(
            content["artifacts"]["code_symbol_ids"]
                .as_array()
                .expect("code symbol ids")
                .iter()
                .any(Value::is_string)
        );
        assert!(
            content["artifacts"]["file_paths"]
                .as_array()
                .expect("file paths")
                .iter()
                .any(|file| file == "src/lib.rs")
        );
        assert!(
            content["artifacts"]["test_paths"]
                .as_array()
                .expect("test paths")
                .iter()
                .any(|file| file == "tests/invoice_test.rs")
        );
        assert_eq!(content["code_assist"]["report"]["actual_mode"], "keyword");

        let store = Store::open(&config.database_marker)?;
        let session_id = content["session"]["id"].as_str().expect("session id");
        let session = store
            .get_task_session("agent-task-fixture", session_id)?
            .expect("session stored");
        assert_eq!(session.phase, "ready");
        assert_eq!(session.progress, 100);
        assert!(session.file_paths.iter().any(|file| file == "src/lib.rs"));
        assert!(
            session
                .test_paths
                .iter()
                .any(|file| file == "tests/invoice_test.rs")
        );
        let linked_events =
            store.list_retrieval_events_for_session("agent-task-fixture", session_id, 10)?;
        assert!(
            linked_events
                .iter()
                .any(|event| event.tool == "dukememory_agent_task"
                    && event.task_session_id.as_deref() == Some(session_id)),
            "expected session-linked retrieval event, got {linked_events:#?}"
        );

        let trace = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_task_replay",
                    "arguments": {
                        "project_id": "agent-task-fixture",
                        "id": session_id,
                        "limit": 5
                    }
                }
            }),
        )
        .await
        .expect("task replay returns response");
        assert!(trace.get("error").is_none(), "{trace}");
        assert!(
            trace["result"]["structuredContent"]["retrieval_events"]
                .as_array()
                .expect("retrieval events")
                .iter()
                .any(|event| event["task_session_id"] == session_id)
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn consistency_check_reports_memory_conflicts_without_ollama() -> Result<()> {
        let config = test_config("consistency-check");
        let store = Store::open(&config.database_marker)?;
        store.remember(
            "consistency-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Feature flags must enable beta dashboard for admin users.".to_string(),
                tags: vec!["flags".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        store.remember(
            "consistency-test",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Feature flags must not enable beta dashboard for admin users.".to_string(),
                tags: vec!["flags".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 43,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_consistency_check",
                    "arguments": {
                        "project_id": "consistency-test",
                        "limit": 20,
                        "status": "active"
                    }
                }
            }),
        )
        .await
        .expect("consistency check returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["action"], "consistency");
        assert!(
            content["readiness_score"].as_f64().unwrap_or(1.0) < 1.0,
            "{content}"
        );
        assert!(
            content["conflict_candidates"]
                .as_array()
                .expect("conflict candidates")
                .iter()
                .any(|candidate| candidate["suggested_action"] == "manual_review_or_supersede")
        );
        assert!(
            content["findings"]
                .as_array()
                .expect("findings")
                .iter()
                .any(|finding| finding.as_str().unwrap_or("").contains("conflict"))
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_tool_skips_without_enough_memories_or_ollama() {
        let config = test_config("compact-skip");
        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 33,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_compact",
                    "arguments": {
                        "project_id": "mcp-test",
                        "limit": 10,
                        "min_memories": 2
                    }
                }
            }),
        )
        .await
        .expect("compact returns response");
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(response["result"]["structuredContent"]["status"], "skipped");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maintenance_tool_default_dry_run_works_without_ollama_when_empty() {
        let config = test_config("maintenance-default");
        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 34,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_maintenance",
                    "arguments": {
                        "project_id": "mcp-test"
                    }
                }
            }),
        )
        .await
        .expect("maintenance returns response");
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(response["result"]["structuredContent"]["apply"], false);
        assert_eq!(
            response["result"]["structuredContent"]["validation"]["pending"],
            0
        );
        assert_eq!(
            response["result"]["structuredContent"]["compaction"]["status"],
            "skipped"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ops_pipeline_returns_safe_dry_run_report() {
        let config = test_config("ops-pipeline");
        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 47,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_ops_pipeline",
                    "arguments": {
                        "project_id": "ops-pipeline-test",
                        "embed_missing": false
                    }
                }
            }),
        )
        .await
        .expect("ops pipeline returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["project_id"], "ops-pipeline-test");
        assert_eq!(content["maintenance"]["apply"], false);
        assert!(content["health"]["schema"].is_string());
        assert!(content["consistency"]["readiness_score"].is_number());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn models_tool_returns_configured_roles_without_ollama() {
        let config = test_config("models");
        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_models",
                    "arguments": {}
                }
            }),
        )
        .await
        .expect("models returns response");
        assert!(response.get("error").is_none(), "{response}");
        let roles = response["result"]["structuredContent"]["roles"]
            .as_array()
            .expect("roles array");
        assert!(
            roles
                .iter()
                .any(|role| { role["role"] == "memory_embed" && role["model"] == "test-embed" })
        );
        assert!(
            roles
                .iter()
                .any(|role| { role["role"] == "fast_embed" && role["model"] == "test-fast-embed" })
        );
        assert!(
            response["result"]["structuredContent"]["warning"]
                .as_str()
                .is_some()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prepare_tool_auto_indexes_code_in_keyword_mode_without_ollama() {
        let config = test_config("prepare-keyword");
        let root =
            std::env::temp_dir().join(format!("dukememory-mcp-prepare-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join("src")).expect("temp project");
        std::fs::write(root.join(".dukememory.toml"), "name = \"prepare-test\"\n")
            .expect("project marker");
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn alpha_prepare() { prepare_helper(); }\nfn prepare_helper() {}\n",
        )
        .expect("rust source");

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_prepare",
                    "arguments": {
                        "project_id": "prepare-test",
                        "project_path": root,
                        "query": "alpha prepare",
                        "mode": "keyword",
                        "memory_limit": 1,
                        "code_limit": 5
                    }
                }
            }),
        )
        .await
        .expect("prepare returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        assert_eq!(content["project_id"], "prepare-test");
        assert_eq!(content["auto_index"], true);
        assert_eq!(content["index"]["files_indexed"], 1);
        assert!(
            content["code"]
                .as_array()
                .expect("code results")
                .iter()
                .any(|result| result["symbol"]["name"] == "alpha_prepare")
        );
        let code_hit = &content["code"].as_array().expect("code results")[0];
        assert!(code_hit["symbol"].get("body").is_none());
        assert_eq!(content["code_body_included"], false);
        let neighborhood = content["code_neighborhood"]
            .as_array()
            .expect("code neighborhood");
        let workflow = content["agent_workflow"]
            .as_array()
            .expect("agent workflow");
        assert_eq!(workflow.len(), 4);
        assert_eq!(workflow[0]["step"], "prepare");
        assert_eq!(workflow[3]["tool"], "dukememory_agent_after");
        assert!(content.get("debug_code_neighborhood").is_none());
        assert!(neighborhood.iter().any(|entry| {
            entry["symbol_name"] == "alpha_prepare"
                && entry["callees"]
                    .as_array()
                    .expect("callees")
                    .iter()
                    .any(|relation| relation["target_name"] == "prepare_helper")
        }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn impact_tool_reports_impacted_files_and_tests() -> Result<()> {
        let config = test_config("impact-files");
        let root = std::env::temp_dir().join(format!(
            "dukememory-impact-fixture-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::create_dir_all(root.join("tests"))?;
        std::fs::write(root.join(".dukememory.toml"), "name = \"impact-fixture\"\n")?;
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn changed() { helper(); }\nfn helper() {}\n",
        )?;
        std::fs::write(
            root.join("tests/integration_test.rs"),
            "#[test]\nfn covers_changed() { changed(); }\n",
        )?;
        let mut store = Store::open(&config.database_marker)?;
        index_project(&mut store, &root, Some("impact-fixture".to_string()), false)?;

        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_impact",
                    "arguments": {
                        "project_id": "impact-fixture",
                        "symbol": "changed",
                        "limit": 20
                    }
                }
            }),
        )
        .await
        .expect("impact returns response");
        assert!(response.get("error").is_none(), "{response}");
        let content = &response["result"]["structuredContent"];
        let files = content["impacted_files"]
            .as_array()
            .expect("impacted files");
        assert!(files.iter().any(|file| file == "src/lib.rs"));
        assert!(files.iter().any(|file| file == "tests/integration_test.rs"));
        let tests = content["affected_tests"]
            .as_array()
            .expect("affected tests");
        assert!(tests.iter().any(|file| file == "tests/integration_test.rs"));
        let callee_edges = content["callee_edges"].as_array().expect("callee edges");
        assert!(callee_edges.iter().any(|edge| {
            edge["relation"]["target_name"] == "helper"
                && edge["source"] == "tree_sitter"
                && edge["confidence"] == "medium"
                && edge["resolution"] == "resolved"
        }));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn code_assist_tools_work_in_keyword_mode_without_ollama() -> Result<()> {
        let config = test_config("code-assist-keyword");
        let root = std::env::temp_dir().join(format!(
            "dukememory-code-assist-fixture-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::create_dir_all(root.join("tests"))?;
        std::fs::write(root.join(".dukememory.toml"), "name = \"assist-fixture\"\n")?;
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn invoice_search() { invoice_helper(); }\nfn invoice_helper() {}\n",
        )?;
        std::fs::write(
            root.join("tests/invoice_test.rs"),
            "#[test]\nfn covers_invoice_search() { invoice_search(); }\n",
        )?;
        let mut store = Store::open(&config.database_marker)?;
        index_project(&mut store, &root, Some("assist-fixture".to_string()), false)?;

        let patterns = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 44,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_code_patterns",
                    "arguments": {
                        "project_id": "assist-fixture",
                        "query": "invoice search",
                        "mode": "keyword",
                        "limit": 5
                    }
                }
            }),
        )
        .await
        .expect("patterns returns response");
        assert!(patterns.get("error").is_none(), "{patterns}");
        let report = &patterns["result"]["structuredContent"]["report"];
        assert!(
            report["symbols"]
                .as_array()
                .expect("symbols")
                .iter()
                .any(|result| result["symbol"]["name"] == "invoice_search")
        );
        assert!(
            report["memory_suggestions"]
                .as_array()
                .expect("memory suggestions")
                .iter()
                .any(|suggestion| suggestion["status"] == "pending")
        );

        let assist = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 45,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_code_assist",
                    "arguments": {
                        "project_id": "assist-fixture",
                        "query": "invoice search",
                        "mode": "keyword",
                        "limit": 5
                    }
                }
            }),
        )
        .await
        .expect("assist returns response");
        assert!(assist.get("error").is_none(), "{assist}");
        let assist_report = &assist["result"]["structuredContent"]["report"];
        assert_eq!(assist_report["actual_mode"], "keyword");
        assert!(
            assist_report["affected_tests"]
                .as_array()
                .expect("affected tests")
                .iter()
                .any(|file| file == "tests/invoice_test.rs")
        );
        assert!(
            assist_report["test_commands"]
                .as_array()
                .expect("test commands")
                .iter()
                .any(|item| item["command"] == "cargo test --test invoice_test")
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn devsystem_tool_records_advisory_session_and_pending_memory() -> Result<()> {
        let config = test_config("devsystem-tool");
        let root = std::env::temp_dir().join(format!(
            "dukememory-devsystem-fixture-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"devsystem-fixture\"\n",
        )?;
        std::fs::write(
            root.join("src/payment_service.py"),
            "def capture_payment(db): db.insert('capture')\n\
             def refund_payment(db): db.update('refund')\n\
             def parse_webhook(request): return request.body\n\
             def generate_invoice(payment): return str(payment)\n\
             def send_email_notification(email): print(email)\n\
             def fraud_decision(user): return user.risk > 10\n\
             def route_provider(provider): return provider.client\n",
        )?;
        let response = handle_message(
            &config,
            json!({
                "jsonrpc": "2.0",
                "id": 46,
                "method": "tools/call",
                "params": {
                    "name": "dukememory_devsystem",
                    "arguments": {
                        "project_id": "devsystem-fixture",
                        "project_path": root,
                        "query": "check payment boundary",
                        "files": ["src/payment_service.py"],
                        "policy": {
                            "required_test_commands": ["cargo test --mcp-policy"]
                        }
                    }
                }
            }),
        )
        .await
        .expect("devsystem returns response");
        assert!(response.get("error").is_none(), "{response}");
        let report = &response["result"]["structuredContent"]["report"];
        assert_eq!(report["product"], "dukedevsystem");
        assert_eq!(report["readiness_percent"], 100);
        assert_eq!(report["telemetry"]["index_run"]["enabled"], true);
        assert!(
            report["telemetry"]["index_run"]["files_indexed"]
                .as_u64()
                .is_some_and(|count| count >= 1)
        );
        assert!(
            report["telemetry"]["index_run"]["symbols_indexed"]
                .as_u64()
                .is_some_and(|count| count >= 1)
        );
        assert!(report["stage_reports"].as_array().unwrap().len() >= 8);
        assert_eq!(report["stage_reports"][0]["role"], "planner");
        assert_eq!(report["stage_reports"][1]["role"], "memory");
        assert!(
            report["telemetry"]["memory_agent"]["recent_task_sessions"]
                .as_array()
                .is_some_and(|sessions| !sessions.is_empty())
        );
        assert!(
            report["telemetry"]["memory_agent"]["file_task_history"]
                .as_array()
                .is_some_and(|history| !history.is_empty())
        );
        assert!(
            report["telemetry"]["policy"]["source"]
                .as_str()
                .is_some_and(|source| source.contains("mcp_override"))
        );
        assert!(
            report["recommended_test_commands"]
                .as_array()
                .is_some_and(|commands| commands
                    .iter()
                    .any(|command| command["command"] == "cargo test --mcp-policy"))
        );
        assert_eq!(
            report["code_review_plan"]["changed_files"][0],
            "src/payment_service.py"
        );
        assert_eq!(report["memory_writes"]["status"], "pending");
        assert!(
            report["memory_writes"]["ids"]
                .as_array()
                .is_some_and(|ids| ids.len() == 4)
        );
        assert!(
            report["memory_writes"]["decision_memory_ids"]
                .as_array()
                .is_some_and(|ids| ids.len() == 1)
        );
        assert!(
            report["memory_writes"]["entropy_memory_ids"]
                .as_array()
                .is_some_and(|ids| ids.len() == 1)
        );
        assert!(
            report["memory_writes"]["graph_memory_ids"]
                .as_array()
                .is_some_and(|ids| ids.len() == 1)
        );
        assert!(
            report["memory_writes"]["graph_candidates"]
                .as_array()
                .is_some_and(|candidates| candidates
                    .iter()
                    .any(|candidate| candidate["relation"] == "needs_boundary_repair_plan"))
        );
        assert_eq!(
            report["file_entropy_reports"][0]["verdict"],
            "split_required"
        );
        assert!(
            report["boundary_repair_plans"]
                .as_array()
                .is_some_and(|plans| !plans.is_empty())
        );
        assert_eq!(
            report["boundary_repair_plans"][0]["source_file"],
            "src/payment_service.py"
        );
        assert_eq!(
            report["quality_gate_summary"]["overall_status"],
            "blocked_by_quality_gate"
        );
        assert!(report["quality_gates"].as_array().is_some_and(|gates| {
            gates
                .iter()
                .any(|gate| gate["id"] == "boundary_repair:src/payment_service.py")
        }));
        assert!(
            report["boundary_repair_plans"][0]["required_tests"]
                .as_array()
                .is_some_and(|tests| tests
                    .iter()
                    .any(|test| test["command"] == "cargo test --mcp-policy"))
        );
        let memory_id = report["memory_writes"]["ids"][0]
            .as_str()
            .expect("memory id");
        let store = Store::open(&config.database_marker)?;
        let memory = store
            .get("devsystem-fixture", memory_id)?
            .expect("pending memory");
        assert_eq!(memory.status, "pending");
        let decision_memory_id = report["memory_writes"]["decision_memory_ids"][0]
            .as_str()
            .expect("decision memory id");
        let decision_memory = store
            .get("devsystem-fixture", decision_memory_id)?
            .expect("decision memory");
        assert_eq!(decision_memory.kind, "decision");
        assert_eq!(decision_memory.status, "pending");
        assert!(
            decision_memory
                .body
                .contains("boundary_repair: recommended")
        );
        assert!(decision_memory.body.contains("src/payment_service.py"));
        let session_id = report["task_session_id"].as_str().expect("session id");
        let session = store
            .get_task_session("devsystem-fixture", session_id)?
            .expect("session");
        assert_eq!(session.status, "completed");
        assert_eq!(session.phase, "done");
        assert_eq!(session.memory_ids.len(), 4);
        assert!(!session.code_symbol_ids.is_empty());
        assert!(
            session
                .file_paths
                .iter()
                .any(|path| path == "src/payment_service.py")
        );
        Ok(())
    }

    fn test_config(name: &str) -> Config {
        Config {
            database_url: "postgresql://dukememory-test@localhost:55432/dukememory_test"
                .to_string(),
            database_marker: std::env::temp_dir().join(format!(
                "dukememory-mcp-test-{name}-{}.schema-marker",
                uuid::Uuid::now_v7()
            )),
            ollama_base_url: "http://127.0.0.1:9".to_string(),
            ollama_embed_model: "test-embed".to_string(),
            ollama_llm_model: "test-llm".to_string(),
            fast_embed_model: "test-fast-embed".to_string(),
            validate_model: "test-validate".to_string(),
            fast_code_model: "test-fast-code".to_string(),
            deep_code_model: "test-deep-code".to_string(),
            agent_code_model: "test-agent-code".to_string(),
            experiment_model: "test-experiment".to_string(),
        }
    }
}

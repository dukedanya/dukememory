use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ollama::OllamaClient;
use crate::store::{CodeRelation, CodeSearchResult, CodeSymbol, Memory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeReasonTask {
    Brief,
    Plan,
    Risk,
}

impl CodeReasonTask {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Brief => "brief",
            Self::Plan => "plan",
            Self::Risk => "risk",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeReasonReport {
    pub task: String,
    pub model: String,
    pub answer: String,
    pub bullets: Vec<String>,
    pub risks: Vec<String>,
    pub next_steps: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawCodeReasonResponse {
    answer: Option<Value>,
    bullets: Option<Vec<Value>>,
    risks: Option<Vec<Value>>,
    next_steps: Option<Vec<Value>>,
}

pub async fn reason_about_symbol(
    ollama: &OllamaClient,
    model: &str,
    project_id: &str,
    symbol: &CodeSymbol,
    callers: &[CodeRelation],
    callees: &[CodeRelation],
) -> Result<CodeReasonReport> {
    let system = code_reason_system_prompt(CodeReasonTask::Brief);
    let user = format!(
        "Project id: {project_id}\n\
         Task: explain the indexed Rust symbol for a coding agent.\n\n\
         Symbol:\n{}\n\n\
         Callers:\n{}\n\n\
         Callees:\n{}\n",
        format_symbol(symbol, true),
        format_relations(callers),
        format_relations(callees)
    );
    reason_json(ollama, model, CodeReasonTask::Brief, system, &user).await
}

pub async fn reason_about_search(
    ollama: &OllamaClient,
    model: &str,
    task: CodeReasonTask,
    project_id: &str,
    query: &str,
    memories: &[Memory],
    code: &[CodeSearchResult],
) -> Result<CodeReasonReport> {
    let system = code_reason_system_prompt(task);
    let user = format!(
        "Project id: {project_id}\n\
         User request/query: {query}\n\n\
         Durable memories:\n{}\n\n\
         Indexed code hits:\n{}\n",
        format_memories(memories),
        format_code_hits(code)
    );
    reason_json(ollama, model, task, system, &user).await
}

async fn reason_json(
    ollama: &OllamaClient,
    model: &str,
    task: CodeReasonTask,
    system: &str,
    user: &str,
) -> Result<CodeReasonReport> {
    let raw = ollama.chat_json_with_model(model, system, user).await?;
    let json = extract_json_object(&raw)?;
    let response = serde_json::from_str::<RawCodeReasonResponse>(json)
        .with_context(|| format!("failed to parse code reasoning JSON: {json}"))?;
    Ok(CodeReasonReport {
        task: task.as_str().to_string(),
        model: model.to_string(),
        answer: truncate(
            &text_value(response.answer.as_ref())
                .unwrap_or_else(|| "No answer was returned.".to_string()),
            4_000,
        ),
        bullets: normalize_list(response.bullets, 12, 500),
        risks: normalize_list(response.risks, 12, 500),
        next_steps: normalize_list(response.next_steps, 12, 500),
    })
}

fn code_reason_system_prompt(task: CodeReasonTask) -> &'static str {
    match task {
        CodeReasonTask::Brief => {
            "You explain Rust code to a coding agent. Return only JSON with keys answer, bullets, risks, next_steps. \
             Be concrete. Use indexed symbol names and file paths. Do not invent APIs not present in the context."
        }
        CodeReasonTask::Plan => {
            "You produce an implementation plan for a Rust coding agent. Return only JSON with keys answer, bullets, risks, next_steps. \
             Ground the plan in provided durable memories and indexed code hits. Keep steps actionable and scoped."
        }
        CodeReasonTask::Risk => {
            "You analyze implementation risk for a Rust coding agent. Return only JSON with keys answer, bullets, risks, next_steps. \
             Focus on likely regressions, missing tests, impacted symbols, and assumptions. Do not exaggerate certainty."
        }
    }
}

fn format_memories(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return "- none".to_string();
    }
    memories
        .iter()
        .map(|memory| {
            format!(
                "- [{}] {} (importance {:.2}, confidence {:.2}): {}",
                memory.kind,
                memory.id,
                memory.importance,
                memory.confidence,
                truncate(&memory.body, 600)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_code_hits(results: &[CodeSearchResult]) -> String {
    if results.is_empty() {
        return "- none".to_string();
    }
    results
        .iter()
        .map(|result| {
            format!(
                "score {:.4}\n{}",
                result.score,
                format_symbol(&result.symbol, false)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_symbol(symbol: &CodeSymbol, include_body: bool) -> String {
    let mut text = format!(
        "- id: {}\n  file: {}:{}-{}\n  kind: {}\n  name: {}\n  signature: {}",
        symbol.id,
        symbol.file_path,
        symbol.start_line,
        symbol.end_line,
        symbol.kind,
        symbol.name,
        symbol.signature
    );
    if include_body {
        text.push_str("\n  body:\n");
        text.push_str(&truncate(&symbol.body, 8_000));
    }
    text
}

fn format_relations(relations: &[CodeRelation]) -> String {
    if relations.is_empty() {
        return "- none".to_string();
    }
    relations
        .iter()
        .map(|relation| {
            format!(
                "- {} {} from {}",
                relation.relation_kind, relation.target_name, relation.from_file_path
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_list(items: Option<Vec<Value>>, limit: usize, max_chars: usize) -> Vec<String> {
    items
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| text_value(Some(&item)))
        .map(|item| truncate(item.trim(), max_chars))
        .filter(|item| !item.is_empty())
        .take(limit)
        .collect()
}

fn text_value(value: Option<&Value>) -> Option<String> {
    let value = value?;
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(_) | Value::Bool(_) => Some(value.to_string()),
        Value::Array(items) => {
            let parts = items
                .iter()
                .filter_map(|item| text_value(Some(item)))
                .collect::<Vec<_>>();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("; "))
            }
        }
        Value::Object(map) => {
            let mut parts = Vec::new();
            if let Some(description) = map.get("description").and_then(|value| value.as_str()) {
                parts.push(description.to_string());
            }
            if let Some(title) = map.get("title").and_then(|value| value.as_str()) {
                parts.push(title.to_string());
            }
            if let Some(details) = text_value(map.get("details")) {
                parts.push(details);
            }
            if parts.is_empty() {
                Some(value.to_string())
            } else {
                Some(parts.join(": "))
            }
        }
        Value::Null => None,
    }
}

fn extract_json_object(text: &str) -> Result<&str> {
    let text = text.trim();
    if text.starts_with('{') && text.ends_with('}') {
        return Ok(text);
    }
    let Some(start) = text.find('{') else {
        bail!("model response did not contain a JSON object");
    };
    let Some(end) = text.rfind('}') else {
        bail!("model response did not contain a complete JSON object");
    };
    Ok(&text[start..=end])
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

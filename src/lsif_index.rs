use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::store::{CodeRelation, CodeSymbol, Store};

const RA_REFERENCE_KIND: &str = "ra_reference";
const RA_CALL_KIND: &str = "ra_call";

#[derive(Debug, Clone)]
pub struct LsifImportReport {
    pub project_id: String,
    pub root_path: PathBuf,
    pub source: String,
    pub documents: usize,
    pub ranges: usize,
    pub definitions_seen: usize,
    pub reference_ranges_seen: usize,
    pub target_symbols_resolved: usize,
    pub relations_imported: usize,
    pub call_relations_imported: usize,
    pub relations_skipped: usize,
    pub stale_relations_removed: usize,
}

#[derive(Debug, Clone)]
struct LsifRange {
    start_line: u32,
    end_line: u32,
    end_character: u32,
}

#[derive(Debug, Clone)]
struct RangeRef {
    document_id: u64,
    range_id: u64,
}

#[derive(Debug, Default)]
struct LsifGraph {
    documents: HashMap<u64, String>,
    ranges: HashMap<u64, LsifRange>,
    range_documents: HashMap<u64, u64>,
    range_result_sets: HashMap<u64, u64>,
    result_definitions: HashMap<u64, u64>,
    result_references: HashMap<u64, u64>,
    result_monikers: HashMap<u64, u64>,
    monikers: HashMap<u64, String>,
    items: HashMap<u64, LsifItems>,
}

#[derive(Debug, Default)]
struct LsifItems {
    definitions: Vec<RangeRef>,
    references: Vec<RangeRef>,
}

pub fn generate_and_import_rust_analyzer_lsif(
    store: &Store,
    project_id: &str,
    root_path: &Path,
) -> Result<LsifImportReport> {
    let root_path = root_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root_path.display()))?;
    let output = Command::new("rust-analyzer")
        .arg("lsif")
        .arg(&root_path)
        .output()
        .context("failed to execute rust-analyzer lsif")?;
    if !output.status.success() {
        bail!(
            "rust-analyzer lsif failed: {}",
            command_output_detail(output.status, &output.stdout, &output.stderr)
        );
    }
    let lsif = String::from_utf8(output.stdout).context("rust-analyzer lsif returned non-UTF8")?;
    import_rust_analyzer_lsif(store, project_id, &root_path, "rust-analyzer lsif", &lsif)
}

pub fn import_rust_analyzer_lsif(
    store: &Store,
    project_id: &str,
    root_path: &Path,
    source: &str,
    lsif: &str,
) -> Result<LsifImportReport> {
    let root_path = root_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root_path.display()))?;
    let graph = parse_lsif(lsif)?;
    let symbols = store.code_symbols_for_project(project_id)?;
    let symbols_by_file = symbols_by_file(&symbols);
    let stale_references_removed =
        store.remove_code_relations_by_kind(project_id, RA_REFERENCE_KIND)?;
    let stale_calls_removed = store.remove_code_relations_by_kind(project_id, RA_CALL_KIND)?;

    let mut report = LsifImportReport {
        project_id: project_id.to_string(),
        root_path,
        source: source.to_string(),
        documents: graph.documents.len(),
        ranges: graph.ranges.len(),
        definitions_seen: 0,
        reference_ranges_seen: 0,
        target_symbols_resolved: 0,
        relations_imported: 0,
        call_relations_imported: 0,
        relations_skipped: 0,
        stale_relations_removed: stale_references_removed + stale_calls_removed,
    };
    let mut source_cache = SourceCache::new(report.root_path.clone());

    let mut target_symbols = HashMap::new();
    for (result_set_id, definition_result_id) in &graph.result_definitions {
        let Some(items) = graph.items.get(definition_result_id) else {
            continue;
        };
        let moniker = graph
            .result_monikers
            .get(result_set_id)
            .and_then(|moniker_id| graph.monikers.get(moniker_id));
        let name_hint = moniker.and_then(|identifier| last_identifier_segment(identifier));
        for range_ref in &items.definitions {
            report.definitions_seen += 1;
            let Some(symbol) = resolve_symbol_for_range(
                &graph,
                &symbols_by_file,
                &report.root_path,
                range_ref,
                name_hint,
            )?
            else {
                continue;
            };
            target_symbols.insert(*result_set_id, (symbol.clone(), moniker.cloned()));
            report.target_symbols_resolved += 1;
            break;
        }
    }

    for (result_set_id, reference_result_id) in &graph.result_references {
        let Some((target_symbol, moniker)) = target_symbols.get(result_set_id) else {
            continue;
        };
        let Some(items) = graph.items.get(reference_result_id) else {
            continue;
        };
        for range_ref in &items.references {
            report.reference_ranges_seen += 1;
            let Some((from_file_path, line)) =
                range_location(&graph, &report.root_path, range_ref)?
            else {
                report.relations_skipped += 1;
                continue;
            };
            if target_symbol.file_path == from_file_path && target_symbol.start_line == line {
                report.relations_skipped += 1;
                continue;
            }
            let from_symbol = containing_symbol(&symbols_by_file, &from_file_path, line);
            let from_symbol_id = from_symbol.as_ref().map(|symbol| symbol.id.clone());
            let target_name = moniker
                .clone()
                .unwrap_or_else(|| target_symbol.name.clone());
            let relation = CodeRelation {
                id: ra_relation_id(
                    RA_REFERENCE_KIND,
                    project_id,
                    &from_file_path,
                    from_symbol_id.as_deref(),
                    &target_symbol.id,
                    range_ref.range_id,
                ),
                project_id: project_id.to_string(),
                from_symbol_id: from_symbol_id.clone(),
                from_file_path: from_file_path.clone(),
                relation_kind: RA_REFERENCE_KIND.to_string(),
                target_name: target_name.clone(),
                target_symbol_id: Some(target_symbol.id.clone()),
            };
            store.insert_code_relation(relation)?;
            report.relations_imported += 1;
            if source_cache.is_call_like_reference(&graph, range_ref, &from_file_path)? {
                store.insert_code_relation(CodeRelation {
                    id: ra_relation_id(
                        RA_CALL_KIND,
                        project_id,
                        &from_file_path,
                        from_symbol_id.as_deref(),
                        &target_symbol.id,
                        range_ref.range_id,
                    ),
                    project_id: project_id.to_string(),
                    from_symbol_id,
                    from_file_path,
                    relation_kind: RA_CALL_KIND.to_string(),
                    target_name,
                    target_symbol_id: Some(target_symbol.id.clone()),
                })?;
                report.call_relations_imported += 1;
            }
        }
    }

    Ok(report)
}

fn parse_lsif(lsif: &str) -> Result<LsifGraph> {
    let mut graph = LsifGraph::default();
    for (line_index, line) in lsif.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(line)
            .with_context(|| format!("failed to parse LSIF JSON line {}", line_index + 1))?;
        match string_field(&value, "type") {
            Some("vertex") => parse_vertex(&mut graph, &value)?,
            Some("edge") => parse_edge(&mut graph, &value)?,
            _ => {}
        }
    }
    Ok(graph)
}

fn parse_vertex(graph: &mut LsifGraph, value: &Value) -> Result<()> {
    let Some(label) = string_field(value, "label") else {
        return Ok(());
    };
    let id = required_u64(value, "id")?;
    match label {
        "document" => {
            if let Some(uri) = string_field(value, "uri") {
                graph.documents.insert(id, uri.to_string());
            }
        }
        "range" => {
            let start = value.get("start").context("LSIF range is missing start")?;
            let end = value.get("end").context("LSIF range is missing end")?;
            graph.ranges.insert(
                id,
                LsifRange {
                    start_line: required_u64(start, "line")? as u32,
                    end_line: required_u64(end, "line")? as u32,
                    end_character: required_u64(end, "character")? as u32,
                },
            );
        }
        "moniker" => {
            if let Some(identifier) = string_field(value, "identifier") {
                graph.monikers.insert(id, identifier.to_string());
            }
        }
        _ => {}
    }
    Ok(())
}

fn parse_edge(graph: &mut LsifGraph, value: &Value) -> Result<()> {
    let Some(label) = string_field(value, "label") else {
        return Ok(());
    };
    match label {
        "contains" => {
            let document_id = required_u64(value, "outV")?;
            for range_id in required_u64_array(value, "inVs")? {
                graph.range_documents.insert(range_id, document_id);
            }
        }
        "next" => {
            graph
                .range_result_sets
                .insert(required_u64(value, "outV")?, required_u64(value, "inV")?);
        }
        "textDocument/definition" => {
            graph
                .result_definitions
                .insert(required_u64(value, "outV")?, required_u64(value, "inV")?);
        }
        "textDocument/references" => {
            graph
                .result_references
                .insert(required_u64(value, "outV")?, required_u64(value, "inV")?);
        }
        "moniker" => {
            graph
                .result_monikers
                .insert(required_u64(value, "outV")?, required_u64(value, "inV")?);
        }
        "item" => parse_item_edge(graph, value)?,
        _ => {}
    }
    Ok(())
}

fn parse_item_edge(graph: &mut LsifGraph, value: &Value) -> Result<()> {
    let result_id = required_u64(value, "outV")?;
    let document_id = required_u64(value, "document")?;
    let range_refs = required_u64_array(value, "inVs")?
        .into_iter()
        .map(|range_id| RangeRef {
            document_id,
            range_id,
        })
        .collect::<Vec<_>>();
    let items = graph.items.entry(result_id).or_default();
    match string_field(value, "property") {
        Some("references") => items.references.extend(range_refs),
        Some("definitions") | None => items.definitions.extend(range_refs),
        _ => {}
    }
    Ok(())
}

fn resolve_symbol_for_range<'a>(
    graph: &LsifGraph,
    symbols_by_file: &'a HashMap<String, Vec<&'a CodeSymbol>>,
    root_path: &Path,
    range_ref: &RangeRef,
    name_hint: Option<&str>,
) -> Result<Option<&'a CodeSymbol>> {
    let Some((file_path, line)) = range_location(graph, root_path, range_ref)? else {
        return Ok(None);
    };
    let Some(symbols) = symbols_by_file.get(&file_path) else {
        return Ok(None);
    };
    let candidates = symbols
        .iter()
        .copied()
        .filter(|symbol| symbol.start_line <= line && line <= symbol.end_line)
        .collect::<Vec<_>>();
    if let Some(name_hint) = name_hint {
        return Ok(smallest_symbol_span(
            candidates
                .iter()
                .copied()
                .filter(|symbol| symbol.name == name_hint)
                .collect::<Vec<_>>(),
        ));
    }
    Ok(smallest_symbol_span(candidates))
}

fn range_location(
    graph: &LsifGraph,
    root_path: &Path,
    range_ref: &RangeRef,
) -> Result<Option<(String, u32)>> {
    let document_id = graph
        .range_documents
        .get(&range_ref.range_id)
        .copied()
        .unwrap_or(range_ref.document_id);
    let Some(uri) = graph.documents.get(&document_id) else {
        return Ok(None);
    };
    let Some(range) = graph.ranges.get(&range_ref.range_id) else {
        return Ok(None);
    };
    let Some(file_path) = uri_to_relative_path(uri, root_path)? else {
        return Ok(None);
    };
    Ok(Some((file_path, range.start_line + 1)))
}

fn containing_symbol<'a>(
    symbols_by_file: &'a HashMap<String, Vec<&'a CodeSymbol>>,
    file_path: &str,
    line: u32,
) -> Option<&'a CodeSymbol> {
    let symbols = symbols_by_file.get(file_path)?;
    smallest_symbol_span(
        symbols
            .iter()
            .copied()
            .filter(|symbol| symbol.start_line <= line && line <= symbol.end_line)
            .collect::<Vec<_>>(),
    )
}

fn smallest_symbol_span(symbols: Vec<&CodeSymbol>) -> Option<&CodeSymbol> {
    symbols
        .into_iter()
        .min_by_key(|symbol| (symbol.end_line - symbol.start_line, symbol.start_line))
}

fn symbols_by_file(symbols: &[CodeSymbol]) -> HashMap<String, Vec<&CodeSymbol>> {
    let mut by_file: HashMap<String, Vec<&CodeSymbol>> = HashMap::new();
    for symbol in symbols {
        by_file
            .entry(symbol.file_path.clone())
            .or_default()
            .push(symbol);
    }
    by_file
}

fn uri_to_relative_path(uri: &str, root_path: &Path) -> Result<Option<String>> {
    let Some(path) = uri.strip_prefix("file://") else {
        return Ok(None);
    };
    let decoded = percent_decode(path);
    let path = PathBuf::from(decoded);
    let relative = match path.strip_prefix(root_path) {
        Ok(relative) => relative,
        Err(_) => return Ok(None),
    };
    Ok(Some(relative.to_string_lossy().replace('\\', "/")))
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let Ok(hex) = std::str::from_utf8(&bytes[index + 1..index + 3])
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            index += 3;
            continue;
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

struct SourceCache {
    root_path: PathBuf,
    lines_by_file: HashMap<String, Vec<String>>,
}

impl SourceCache {
    fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
            lines_by_file: HashMap::new(),
        }
    }

    fn is_call_like_reference(
        &mut self,
        graph: &LsifGraph,
        range_ref: &RangeRef,
        file_path: &str,
    ) -> Result<bool> {
        let Some(range) = graph.ranges.get(&range_ref.range_id) else {
            return Ok(false);
        };
        if range.start_line != range.end_line {
            return Ok(false);
        }
        let lines = self.lines_for(file_path)?;
        let Some(line) = lines.get(range.end_line as usize) else {
            return Ok(false);
        };
        let next = line
            .chars()
            .skip(range.end_character as usize)
            .find(|ch| !ch.is_whitespace());
        Ok(matches!(next, Some('(' | '!')))
    }

    fn lines_for(&mut self, file_path: &str) -> Result<&Vec<String>> {
        if !self.lines_by_file.contains_key(file_path) {
            let source_path = self.root_path.join(file_path);
            let source = std::fs::read_to_string(&source_path)
                .with_context(|| format!("failed to read source {}", source_path.display()))?;
            self.lines_by_file.insert(
                file_path.to_string(),
                source.lines().map(str::to_string).collect(),
            );
        }
        Ok(self
            .lines_by_file
            .get(file_path)
            .expect("source lines were just inserted"))
    }
}

fn last_identifier_segment(identifier: &str) -> Option<&str> {
    identifier
        .rsplit("::")
        .find(|part| !matches!(*part, "impl" | "crate"))
}

fn ra_relation_id(
    relation_kind: &str,
    project_id: &str,
    from_file_path: &str,
    from_symbol_id: Option<&str>,
    target_symbol_id: &str,
    range_id: u64,
) -> String {
    let input = format!(
        "{}:{}:{}:{}:{}:{}",
        relation_kind,
        project_id,
        from_file_path,
        from_symbol_id.unwrap_or("file"),
        target_symbol_id,
        range_id
    );
    format!("raref_{}", blake3::hash(input.as_bytes()).to_hex())
}

fn string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field).and_then(Value::as_str)
}

fn required_u64(value: &Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("LSIF value is missing numeric field `{field}`"))
}

fn required_u64_array(value: &Value, field: &str) -> Result<Vec<u64>> {
    let values = value
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("LSIF value is missing array field `{field}`"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .context("LSIF array contains a non-numeric id")
        })
        .collect()
}

fn command_output_detail(status: std::process::ExitStatus, stdout: &[u8], stderr: &[u8]) -> String {
    let stdout = String::from_utf8_lossy(stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => format!("command exited with {status}"),
        (false, true) => format!("command exited with {status}: {stdout}"),
        (true, false) => format!("command exited with {status}: {stderr}"),
        (false, false) => format!("command exited with {status}: {stderr}; stdout: {stdout}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{CodeFile, Store};

    #[test]
    fn imports_ra_reference_from_minimal_lsif() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("dukememory-lsif-test-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join("src/lib.rs"),
            "fn target() {}\n\nfn caller() {\n    target();\n}\n",
        )?;
        let database_marker = root.join("schema.marker");
        let store = Store::open(&database_marker)?;
        store.upsert_code_file(CodeFile {
            project_id: "lsif-test".to_string(),
            path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            hash: "test".to_string(),
            size_bytes: 44,
            line_count: 5,
        })?;
        store.upsert_code_symbol(CodeSymbol {
            id: "sym_target".to_string(),
            project_id: "lsif-test".to_string(),
            file_path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            name: "target".to_string(),
            kind: "function".to_string(),
            signature: "fn target()".to_string(),
            body: "fn target() {}".to_string(),
            start_line: 1,
            end_line: 1,
            parent_id: None,
        })?;
        store.upsert_code_symbol(CodeSymbol {
            id: "sym_caller".to_string(),
            project_id: "lsif-test".to_string(),
            file_path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            name: "caller".to_string(),
            kind: "function".to_string(),
            signature: "fn caller()".to_string(),
            body: "fn caller() { target(); }".to_string(),
            start_line: 3,
            end_line: 5,
            parent_id: None,
        })?;

        let source_path = root.join("src/lib.rs").canonicalize()?;
        let uri = format!("file://{}", source_path.display());
        let lsif = format!(
            r#"{{"id":1,"type":"vertex","label":"document","uri":"{uri}","languageId":"rust"}}
{{"id":2,"type":"vertex","label":"range","start":{{"line":0,"character":3}},"end":{{"line":0,"character":9}}}}
{{"id":3,"type":"vertex","label":"range","start":{{"line":3,"character":4}},"end":{{"line":3,"character":10}}}}
{{"id":4,"type":"vertex","label":"resultSet"}}
{{"id":5,"type":"edge","label":"contains","outV":1,"inVs":[2,3]}}
{{"id":6,"type":"edge","label":"next","outV":2,"inV":4}}
{{"id":7,"type":"edge","label":"next","outV":3,"inV":4}}
{{"id":8,"type":"vertex","label":"definitionResult"}}
{{"id":9,"type":"edge","label":"textDocument/definition","outV":4,"inV":8}}
{{"id":10,"type":"edge","label":"item","outV":8,"document":1,"property":"definitions","inVs":[2]}}
{{"id":11,"type":"vertex","label":"referenceResult"}}
{{"id":12,"type":"edge","label":"textDocument/references","outV":4,"inV":11}}
{{"id":13,"type":"edge","label":"item","outV":11,"document":1,"property":"references","inVs":[3]}}
{{"id":14,"type":"vertex","label":"moniker","scheme":"rust-analyzer","identifier":"lsif_test::target","unique":"scheme","kind":"export"}}
{{"id":15,"type":"edge","label":"moniker","outV":4,"inV":14}}
"#
        );

        let report = import_rust_analyzer_lsif(&store, "lsif-test", &root, "test", &lsif)?;
        assert_eq!(report.target_symbols_resolved, 1);
        assert_eq!(report.relations_imported, 1);
        assert_eq!(report.call_relations_imported, 1);

        let callers = store.find_callers("lsif-test", "sym_target", 10)?;
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].relation_kind, "ra_call");
        assert_eq!(callers[0].from_symbol_id.as_deref(), Some("sym_caller"));
        assert_eq!(callers[0].target_symbol_id.as_deref(), Some("sym_target"));
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }
}

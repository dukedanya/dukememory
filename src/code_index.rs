use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use tree_sitter::{Language, Node, Parser};

use crate::project::resolve_project_id_from_path;
use crate::store::{CodeFile, CodeRelation, CodeSymbol, Store};

const MAX_FILE_BYTES: u64 = 1_000_000;

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct IndexTimingReport {
    pub total_ms: u64,
    pub scan_ms: u64,
    pub read_hash_ms: u64,
    pub parse_ms: u64,
    pub db_write_ms: u64,
    pub delete_ms: u64,
    pub resolve_ms: u64,
}

#[derive(Debug)]
pub struct IndexReport {
    pub project_id: String,
    pub root_path: PathBuf,
    pub full_rebuild: bool,
    pub files_seen: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub indexed_files: Vec<String>,
    pub symbols_indexed: usize,
    pub relations_indexed: usize,
    pub relation_targets_reset: usize,
    pub calls_resolved: usize,
    pub uses_resolved: usize,
    pub modules_resolved: usize,
    pub timing: IndexTimingReport,
}

#[derive(Debug, serde::Serialize)]
pub struct CodeFreshnessReport {
    pub project_id: String,
    pub root_path: PathBuf,
    pub files_seen: usize,
    pub indexed_files: usize,
    pub stale_files: Vec<String>,
    pub missing_files: Vec<String>,
    pub deleted_files: Vec<String>,
}

impl CodeFreshnessReport {
    pub fn is_fresh(&self) -> bool {
        self.stale_files.is_empty()
            && self.missing_files.is_empty()
            && self.deleted_files.is_empty()
    }
}

#[derive(Debug)]
struct ExtractedRust {
    symbols: Vec<CodeSymbol>,
    relations: Vec<CodeRelation>,
}

#[derive(Debug, Clone)]
struct SymbolDraft {
    id: String,
    name: String,
    kind: String,
    signature: String,
    body: String,
    start_line: u32,
    end_line: u32,
    parent_id: Option<String>,
}

pub fn index_project(
    store: &mut Store,
    root_path: &Path,
    project_id: Option<String>,
    full_rebuild: bool,
) -> Result<IndexReport> {
    let total_started = Instant::now();
    let root_path = root_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root_path.display()))?;
    let project_id = match project_id {
        Some(project_id) => project_id,
        None => resolve_project_id_from_path(&root_path)?,
    };

    let existing_hashes = if full_rebuild {
        store.clear_code_index(&project_id)?;
        HashMap::new()
    } else {
        store.code_file_hashes(&project_id)?
    };

    let mut report = IndexReport {
        project_id: project_id.clone(),
        root_path: root_path.clone(),
        full_rebuild,
        files_seen: 0,
        files_indexed: 0,
        files_skipped: 0,
        files_deleted: 0,
        indexed_files: Vec::new(),
        symbols_indexed: 0,
        relations_indexed: 0,
        relation_targets_reset: 0,
        calls_resolved: 0,
        uses_resolved: 0,
        modules_resolved: 0,
        timing: IndexTimingReport::default(),
    };
    let mut seen_code_paths = HashSet::new();

    for entry in WalkBuilder::new(&root_path)
        .standard_filters(true)
        .hidden(false)
        .build()
    {
        let scan_started = Instant::now();
        let entry = entry?;
        let path = entry.path();
        if !entry
            .file_type()
            .map(|kind| kind.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        if should_skip(path) {
            continue;
        }
        let metadata = fs::metadata(path)?;
        if metadata.len() > MAX_FILE_BYTES || !is_indexable_code_file(path) {
            report.timing.scan_ms += elapsed_ms(scan_started.elapsed());
            continue;
        }
        report.timing.scan_ms += elapsed_ms(scan_started.elapsed());

        let read_started = Instant::now();
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let relative_path = relative_path(&root_path, path);
        report.files_seen += 1;
        let hash = blake3::hash(source.as_bytes()).to_hex().to_string();
        report.timing.read_hash_ms += elapsed_ms(read_started.elapsed());
        if path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
            let db_started = Instant::now();
            index_cargo_toml(
                store,
                &project_id,
                &relative_path,
                &source,
                &existing_hashes,
                &mut seen_code_paths,
                &mut report,
            )?;
            report.timing.db_write_ms += elapsed_ms(db_started.elapsed());
            continue;
        }
        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        if !matches!(
            ext,
            "rs" | "py"
                | "js"
                | "mjs"
                | "jsx"
                | "ts"
                | "tsx"
                | "go"
                | "java"
                | "kt"
                | "kts"
                | "swift"
        ) {
            continue;
        }
        seen_code_paths.insert(relative_path.clone());
        if existing_hashes
            .get(&relative_path)
            .is_some_and(|existing_hash| existing_hash == &hash)
        {
            report.files_skipped += 1;
            continue;
        }

        let line_count = source.lines().count() as u32;
        let lang = match ext {
            "rs" => "rust",
            "py" => "python",
            "js" | "mjs" | "jsx" => "javascript",
            "ts" | "tsx" => "typescript",
            "go" => "go",
            "java" => "java",
            "kt" | "kts" => "kotlin",
            "swift" => "swift",
            _ => "generic",
        };

        let file = CodeFile {
            project_id: project_id.clone(),
            path: relative_path.clone(),
            language: lang.to_string(),
            hash,
            size_bytes: metadata.len(),
            line_count,
        };

        let parse_started = Instant::now();
        let extracted = match lang {
            "rust" => extract_rust(&project_id, &relative_path, &source)?,
            "python" => extract_python(&project_id, &relative_path, &source)?,
            "javascript" | "typescript" => {
                extract_javascript_like(&project_id, &relative_path, &source, lang)?
            }
            "go" => extract_generic_tree_sitter(
                &project_id,
                &relative_path,
                &source,
                "go",
                tree_sitter_go::LANGUAGE.into(),
            )?,
            "java" => extract_generic_tree_sitter(
                &project_id,
                &relative_path,
                &source,
                "java",
                tree_sitter_java::LANGUAGE.into(),
            )?,
            "kotlin" => extract_kotlin_text(&project_id, &relative_path, &source),
            "swift" => extract_generic_tree_sitter(
                &project_id,
                &relative_path,
                &source,
                "swift",
                tree_sitter_swift::LANGUAGE.into(),
            )?,
            _ => ExtractedRust {
                symbols: Vec::new(),
                relations: Vec::new(),
            },
        };
        report.timing.parse_ms += elapsed_ms(parse_started.elapsed());
        let symbols_indexed = extracted.symbols.len();
        let db_started = Instant::now();
        report.relations_indexed += store.upsert_code_file_index(
            file,
            extracted.symbols,
            extracted.relations,
            existing_hashes.contains_key(&relative_path),
        )?;
        report.timing.db_write_ms += elapsed_ms(db_started.elapsed());
        report.symbols_indexed += symbols_indexed;
        report.files_indexed += 1;
        report.indexed_files.push(relative_path);
    }

    if !full_rebuild {
        for existing_path in existing_hashes.keys() {
            if !seen_code_paths.contains(existing_path) {
                let delete_started = Instant::now();
                store.remove_code_file(&project_id, existing_path)?;
                report.timing.delete_ms += elapsed_ms(delete_started.elapsed());
                report.files_deleted += 1;
            }
        }
    }

    let resolve_started = Instant::now();
    let resolution = store.resolve_code_relation_targets(&project_id)?;
    report.timing.resolve_ms = elapsed_ms(resolve_started.elapsed());
    report.relation_targets_reset = resolution.targets_reset;
    report.calls_resolved = resolution.calls_resolved;
    report.uses_resolved = resolution.uses_resolved;
    report.modules_resolved = resolution.modules_resolved;
    report.timing.total_ms = elapsed_ms(total_started.elapsed());

    Ok(report)
}

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis() as u64
}

pub fn check_code_index_freshness(
    store: &Store,
    root_path: &Path,
    project_id: Option<String>,
) -> Result<CodeFreshnessReport> {
    let root_path = root_path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", root_path.display()))?;
    let project_id = match project_id {
        Some(project_id) => project_id,
        None => resolve_project_id_from_path(&root_path)?,
    };
    let existing_hashes = store.code_file_hashes(&project_id)?;
    let mut seen_code_paths = HashSet::new();
    let mut stale_files = Vec::new();
    let mut missing_files = Vec::new();

    for entry in WalkBuilder::new(&root_path)
        .standard_filters(true)
        .hidden(false)
        .build()
    {
        let entry = entry?;
        let path = entry.path();
        if !entry
            .file_type()
            .map(|kind| kind.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        if should_skip(path) {
            continue;
        }
        let metadata = fs::metadata(path)?;
        if metadata.len() > MAX_FILE_BYTES || !is_indexable_code_file(path) {
            continue;
        }
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let relative_path = relative_path(&root_path, path);
        let hash = blake3::hash(source.as_bytes()).to_hex().to_string();
        seen_code_paths.insert(relative_path.clone());
        match existing_hashes.get(&relative_path) {
            Some(existing_hash) if existing_hash == &hash => {}
            Some(_) => stale_files.push(relative_path),
            None => missing_files.push(relative_path),
        }
    }

    let mut deleted_files = existing_hashes
        .keys()
        .filter(|path| !seen_code_paths.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    stale_files.sort();
    missing_files.sort();
    deleted_files.sort();

    Ok(CodeFreshnessReport {
        project_id,
        root_path,
        files_seen: seen_code_paths.len(),
        indexed_files: existing_hashes.len(),
        stale_files,
        missing_files,
        deleted_files,
    })
}

fn index_cargo_toml(
    store: &mut Store,
    project_id: &str,
    relative_path: &str,
    source: &str,
    existing_hashes: &HashMap<String, String>,
    seen_code_paths: &mut HashSet<String>,
    report: &mut IndexReport,
) -> Result<()> {
    let hash = blake3::hash(source.as_bytes()).to_hex().to_string();
    seen_code_paths.insert(relative_path.to_string());
    if existing_hashes
        .get(relative_path)
        .is_some_and(|existing_hash| existing_hash == &hash)
    {
        report.files_skipped += 1;
        return Ok(());
    }
    if existing_hashes.contains_key(relative_path) {
        store.remove_code_file(project_id, relative_path)?;
    }
    store.upsert_code_file(CodeFile {
        project_id: project_id.to_string(),
        path: relative_path.to_string(),
        language: "toml".to_string(),
        hash,
        size_bytes: source.len() as u64,
        line_count: source.lines().count() as u32,
    })?;
    if let Some(package_name) = cargo_package_name(source) {
        report.relations_indexed += store.insert_code_relation(CodeRelation {
            id: relation_id(
                project_id,
                relative_path,
                None,
                "cargo_package",
                &package_name,
            ),
            project_id: project_id.to_string(),
            from_symbol_id: None,
            from_file_path: relative_path.to_string(),
            relation_kind: "cargo_package".to_string(),
            target_name: package_name,
            target_symbol_id: None,
        })?;
    }
    report.files_indexed += 1;
    report.indexed_files.push(relative_path.to_string());
    Ok(())
}

fn extract_python(project_id: &str, file_path: &str, source: &str) -> Result<ExtractedRust> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .context("failed to load tree-sitter Python grammar")?;
    let tree = parser
        .parse(source, None)
        .context("tree-sitter failed to parse Python source")?;
    let root = tree.root_node();

    let mut drafts = Vec::new();
    let mut relations = Vec::new();
    let mut scope = Vec::new();
    collect_python_nodes(
        root,
        project_id,
        file_path,
        source,
        &mut scope,
        &mut drafts,
        &mut relations,
    );

    let symbol_ids = drafts
        .iter()
        .map(|draft| draft.id.clone())
        .collect::<HashSet<_>>();
    let symbol_by_name = drafts
        .iter()
        .map(|draft| (draft.name.clone(), draft.id.clone()))
        .collect::<HashMap<_, _>>();
    for relation in &mut relations {
        if relation.relation_kind == "calls" {
            relation.target_symbol_id = symbol_by_name.get(&relation.target_name).cloned();
        }
    }
    let symbols = drafts
        .into_iter()
        .map(|draft| CodeSymbol {
            id: draft.id,
            project_id: project_id.to_string(),
            file_path: file_path.to_string(),
            language: "python".to_string(),
            name: draft.name,
            kind: draft.kind,
            signature: draft.signature,
            body: draft.body,
            start_line: draft.start_line,
            end_line: draft.end_line,
            parent_id: draft.parent_id.filter(|id| symbol_ids.contains(id)),
        })
        .collect();

    Ok(ExtractedRust { symbols, relations })
}

fn collect_python_nodes(
    node: Node<'_>,
    project_id: &str,
    file_path: &str,
    source: &str,
    scope: &mut Vec<String>,
    symbols: &mut Vec<SymbolDraft>,
    relations: &mut Vec<CodeRelation>,
) {
    let kind = node.kind();
    if (kind == "class_definition" || kind == "function_definition")
        && let Some(name_node) = node.child_by_field_name("name")
        && let Ok(name) = name_node.utf8_text(source.as_bytes())
    {
        let name = name.to_string();
        let kind_str = match kind {
            "class_definition" => "class",
            "function_definition" => "function",
            _ => "symbol",
        };
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let id = symbol_id(project_id, file_path, kind_str, &name, start_line);
        let body = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();

        let full_text = node.utf8_text(source.as_bytes()).unwrap_or("");
        let first_line = full_text.lines().next().unwrap_or("").trim();
        let before_colon = first_line.split(':').next().unwrap_or(first_line).trim();
        let signature = truncate(&compact_whitespace(before_colon), 500);

        let parent_id = scope.last().cloned();

        symbols.push(SymbolDraft {
            id: id.clone(),
            name,
            kind: kind_str.to_string(),
            signature,
            body,
            start_line,
            end_line,
            parent_id,
        });

        if kind == "function_definition" {
            collect_calls(node, project_id, file_path, source, &id, relations, false);
        }

        scope.push(id);
        visit_children(node, |child| {
            collect_python_nodes(
                child, project_id, file_path, source, scope, symbols, relations,
            );
        });
        scope.pop();
        return;
    }

    visit_children(node, |child| {
        collect_python_nodes(
            child, project_id, file_path, source, scope, symbols, relations,
        );
    });
}

fn extract_javascript_like(
    project_id: &str,
    file_path: &str,
    source: &str,
    language: &str,
) -> Result<ExtractedRust> {
    let mut parser = Parser::new();
    if language == "typescript" {
        let ext = Path::new(file_path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        if ext == "tsx" {
            parser
                .set_language(&tree_sitter_typescript::LANGUAGE_TSX.into())
                .context("failed to load tree-sitter TSX grammar")?;
        } else {
            parser
                .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
                .context("failed to load tree-sitter TypeScript grammar")?;
        }
    } else {
        parser
            .set_language(&tree_sitter_javascript::LANGUAGE.into())
            .context("failed to load tree-sitter JavaScript grammar")?;
    }
    let tree = parser
        .parse(source, None)
        .with_context(|| format!("tree-sitter failed to parse {language} source"))?;
    let root = tree.root_node();

    let mut drafts = Vec::new();
    let mut relations = Vec::new();
    let mut scope = Vec::new();
    collect_javascript_nodes(
        root,
        project_id,
        file_path,
        source,
        &mut scope,
        &mut drafts,
        &mut relations,
    );

    let symbol_ids = drafts
        .iter()
        .map(|draft| draft.id.clone())
        .collect::<HashSet<_>>();
    let symbol_by_name = drafts
        .iter()
        .map(|draft| (draft.name.clone(), draft.id.clone()))
        .collect::<HashMap<_, _>>();
    for relation in &mut relations {
        if relation.relation_kind == "calls" {
            relation.target_symbol_id = symbol_by_name.get(&relation.target_name).cloned();
        }
    }
    let symbols = drafts
        .into_iter()
        .map(|draft| CodeSymbol {
            id: draft.id,
            project_id: project_id.to_string(),
            file_path: file_path.to_string(),
            language: language.to_string(),
            name: draft.name,
            kind: draft.kind,
            signature: draft.signature,
            body: draft.body,
            start_line: draft.start_line,
            end_line: draft.end_line,
            parent_id: draft.parent_id.filter(|id| symbol_ids.contains(id)),
        })
        .collect();

    Ok(ExtractedRust { symbols, relations })
}

fn collect_javascript_nodes(
    node: Node<'_>,
    project_id: &str,
    file_path: &str,
    source: &str,
    scope: &mut Vec<String>,
    symbols: &mut Vec<SymbolDraft>,
    relations: &mut Vec<CodeRelation>,
) {
    let kind = node.kind();
    if (kind == "class_declaration"
        || kind == "function_declaration"
        || kind == "method_definition")
        && let Some(name_node) = node.child_by_field_name("name")
        && let Ok(name) = name_node.utf8_text(source.as_bytes())
    {
        let name = name.to_string();
        let kind_str = match kind {
            "class_declaration" => "class",
            "function_declaration" => "function",
            "method_definition" => "method",
            _ => "symbol",
        };
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let id = symbol_id(project_id, file_path, kind_str, &name, start_line);
        let body = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();

        let full_text = node.utf8_text(source.as_bytes()).unwrap_or("");
        let first_line = full_text.lines().next().unwrap_or("").trim();
        let before_brace = first_line.split('{').next().unwrap_or(first_line).trim();
        let signature = truncate(&compact_whitespace(before_brace), 500);

        let parent_id = scope.last().cloned();

        symbols.push(SymbolDraft {
            id: id.clone(),
            name,
            kind: kind_str.to_string(),
            signature,
            body,
            start_line,
            end_line,
            parent_id,
        });

        if kind == "function_declaration" || kind == "method_definition" {
            collect_calls(node, project_id, file_path, source, &id, relations, false);
        }

        scope.push(id);
        visit_children(node, |child| {
            collect_javascript_nodes(
                child, project_id, file_path, source, scope, symbols, relations,
            );
        });
        scope.pop();
        return;
    }

    visit_children(node, |child| {
        collect_javascript_nodes(
            child, project_id, file_path, source, scope, symbols, relations,
        );
    });
}

fn extract_kotlin_text(project_id: &str, file_path: &str, source: &str) -> ExtractedRust {
    let mut symbols = Vec::new();
    let mut relations = Vec::new();
    let mut current_parent: Option<String> = None;
    let mut current_function: Option<String> = None;

    for (index, line) in source.lines().enumerate() {
        let line_number = index as u32 + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if let Some((kind, name)) = kotlin_symbol_decl(trimmed) {
            let id = symbol_id(project_id, file_path, kind, &name, line_number);
            let signature = truncate(
                &compact_whitespace(trimmed.trim_end_matches('{').trim()),
                500,
            );
            symbols.push(CodeSymbol {
                id: id.clone(),
                project_id: project_id.to_string(),
                file_path: file_path.to_string(),
                language: "kotlin".to_string(),
                name,
                kind: kind.to_string(),
                signature,
                body: trimmed.to_string(),
                start_line: line_number,
                end_line: line_number,
                parent_id: if matches!(kind, "function" | "method") {
                    current_parent.clone()
                } else {
                    None
                },
            });
            if matches!(kind, "class" | "interface" | "enum" | "object") {
                current_parent = Some(id);
                current_function = None;
            } else {
                current_function = Some(id);
            }
            continue;
        }
        if let Some(from_symbol_id) = current_function.as_deref() {
            for target_name in call_names_from_line(trimmed) {
                relations.push(CodeRelation {
                    id: relation_id(
                        project_id,
                        file_path,
                        Some(from_symbol_id),
                        "calls",
                        &target_name,
                    ),
                    project_id: project_id.to_string(),
                    from_symbol_id: Some(from_symbol_id.to_string()),
                    from_file_path: file_path.to_string(),
                    relation_kind: "calls".to_string(),
                    target_name,
                    target_symbol_id: None,
                });
            }
        }
    }

    let symbol_by_name = symbols
        .iter()
        .map(|symbol| (symbol.name.clone(), symbol.id.clone()))
        .collect::<HashMap<_, _>>();
    for relation in &mut relations {
        relation.target_symbol_id = symbol_by_name.get(&relation.target_name).cloned();
    }

    ExtractedRust { symbols, relations }
}

fn kotlin_symbol_decl(line: &str) -> Option<(&'static str, String)> {
    for (prefix, kind) in [
        ("class ", "class"),
        ("interface ", "interface"),
        ("enum class ", "enum"),
        ("object ", "object"),
        ("fun ", "function"),
    ] {
        if let Some(rest) = line.strip_prefix(prefix) {
            let name = rest
                .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                return Some((kind, name.to_string()));
            }
        }
    }
    None
}

fn call_names_from_line(line: &str) -> Vec<String> {
    let mut calls = Vec::new();
    for part in line.split('(').take(line.matches('(').count()) {
        let name = part
            .rsplit(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.'))
            .next()
            .unwrap_or("")
            .trim_matches('.')
            .trim();
        if name.is_empty()
            || matches!(
                name,
                "if" | "for" | "while" | "when" | "return" | "println" | "print"
            )
        {
            continue;
        }
        calls.push(name.rsplit('.').next().unwrap_or(name).to_string());
    }
    calls
}

fn extract_generic_tree_sitter(
    project_id: &str,
    file_path: &str,
    source: &str,
    language: &str,
    grammar: Language,
) -> Result<ExtractedRust> {
    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .with_context(|| format!("failed to load tree-sitter {language} grammar"))?;
    let tree = parser
        .parse(source, None)
        .with_context(|| format!("tree-sitter failed to parse {language} source"))?;
    let root = tree.root_node();

    let mut drafts = Vec::new();
    let mut relations = Vec::new();
    let mut scope = Vec::new();
    collect_generic_nodes(
        root,
        project_id,
        file_path,
        source,
        &mut scope,
        &mut drafts,
        &mut relations,
    );

    let symbol_ids = drafts
        .iter()
        .map(|draft| draft.id.clone())
        .collect::<HashSet<_>>();
    let symbol_by_name = drafts
        .iter()
        .map(|draft| (draft.name.clone(), draft.id.clone()))
        .collect::<HashMap<_, _>>();
    for relation in &mut relations {
        if relation.relation_kind == "calls" {
            relation.target_symbol_id = symbol_by_name.get(&relation.target_name).cloned();
        }
    }
    let symbols = drafts
        .into_iter()
        .map(|draft| CodeSymbol {
            id: draft.id,
            project_id: project_id.to_string(),
            file_path: file_path.to_string(),
            language: language.to_string(),
            name: draft.name,
            kind: draft.kind,
            signature: draft.signature,
            body: draft.body,
            start_line: draft.start_line,
            end_line: draft.end_line,
            parent_id: draft.parent_id.filter(|id| symbol_ids.contains(id)),
        })
        .collect();

    Ok(ExtractedRust { symbols, relations })
}

fn collect_generic_nodes(
    node: Node<'_>,
    project_id: &str,
    file_path: &str,
    source: &str,
    scope: &mut Vec<String>,
    symbols: &mut Vec<SymbolDraft>,
    relations: &mut Vec<CodeRelation>,
) {
    let kind = node.kind();
    if let Some(kind_str) = generic_symbol_kind(kind)
        && let Some(name) = generic_symbol_name(node, source)
    {
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let id = symbol_id(project_id, file_path, kind_str, &name, start_line);
        let body = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
        let signature = signature_for(node, source);
        let parent_id = scope.last().cloned();

        symbols.push(SymbolDraft {
            id: id.clone(),
            name,
            kind: kind_str.to_string(),
            signature,
            body,
            start_line,
            end_line,
            parent_id,
        });

        if matches!(kind_str, "function" | "method" | "constructor") {
            collect_calls(node, project_id, file_path, source, &id, relations, false);
        }

        scope.push(id);
        visit_children(node, |child| {
            collect_generic_nodes(
                child, project_id, file_path, source, scope, symbols, relations,
            );
        });
        scope.pop();
        return;
    }

    visit_children(node, |child| {
        collect_generic_nodes(
            child, project_id, file_path, source, scope, symbols, relations,
        );
    });
}

fn generic_symbol_kind(kind: &str) -> Option<&'static str> {
    match kind {
        "function_declaration" => Some("function"),
        "method_declaration" | "method_definition" => Some("method"),
        "constructor_declaration" => Some("constructor"),
        "class_declaration" => Some("class"),
        "interface_declaration" => Some("interface"),
        "enum_declaration" => Some("enum"),
        "struct_declaration" => Some("struct"),
        "protocol_declaration" => Some("protocol"),
        "object_declaration" => Some("object"),
        _ => None,
    }
}

fn generic_symbol_name(node: Node<'_>, source: &str) -> Option<String> {
    node.child_by_field_name("name")
        .and_then(|name| name.utf8_text(source.as_bytes()).ok())
        .map(str::to_string)
}

fn extract_rust(project_id: &str, file_path: &str, source: &str) -> Result<ExtractedRust> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .context("failed to load tree-sitter Rust grammar")?;
    let tree = parser
        .parse(source, None)
        .context("tree-sitter failed to parse Rust source")?;
    let root = tree.root_node();

    let mut drafts = Vec::new();
    let mut relations = Vec::new();
    let mut scope = Vec::new();
    collect_rust_nodes(
        root,
        project_id,
        file_path,
        source,
        &mut scope,
        &mut drafts,
        &mut relations,
    );

    let symbol_ids = drafts
        .iter()
        .map(|draft| draft.id.clone())
        .collect::<HashSet<_>>();
    let symbol_by_name = drafts
        .iter()
        .map(|draft| (draft.name.clone(), draft.id.clone()))
        .collect::<HashMap<_, _>>();
    for relation in &mut relations {
        if relation.relation_kind == "calls" {
            relation.target_symbol_id = symbol_by_name.get(&relation.target_name).cloned();
        }
    }
    let symbols = drafts
        .into_iter()
        .map(|draft| CodeSymbol {
            id: draft.id,
            project_id: project_id.to_string(),
            file_path: file_path.to_string(),
            language: "rust".to_string(),
            name: draft.name,
            kind: draft.kind,
            signature: draft.signature,
            body: draft.body,
            start_line: draft.start_line,
            end_line: draft.end_line,
            parent_id: draft.parent_id.filter(|id| symbol_ids.contains(id)),
        })
        .collect();

    Ok(ExtractedRust { symbols, relations })
}

fn collect_rust_nodes(
    node: Node<'_>,
    project_id: &str,
    file_path: &str,
    source: &str,
    scope: &mut Vec<String>,
    symbols: &mut Vec<SymbolDraft>,
    relations: &mut Vec<CodeRelation>,
) {
    if node.kind() == "use_declaration"
        && let Ok(text) = node.utf8_text(source.as_bytes())
    {
        for target_name in rust_use_targets(text) {
            let relation_kind = if is_external_rust_path(&target_name) {
                "external_use"
            } else {
                "uses"
            };
            relations.push(CodeRelation {
                id: relation_id(project_id, file_path, None, relation_kind, &target_name),
                project_id: project_id.to_string(),
                from_symbol_id: None,
                from_file_path: file_path.to_string(),
                relation_kind: relation_kind.to_string(),
                target_name,
                target_symbol_id: None,
            });
        }
    }

    if node.kind() == "mod_item"
        && is_external_mod_item(node, source)
        && let Some(name) = symbol_name_for(node, source)
    {
        for target in module_file_targets(file_path, &name) {
            relations.push(CodeRelation {
                id: relation_id(project_id, file_path, None, "declares_module", &target),
                project_id: project_id.to_string(),
                from_symbol_id: None,
                from_file_path: file_path.to_string(),
                relation_kind: "declares_module".to_string(),
                target_name: target,
                target_symbol_id: None,
            });
        }
    }

    if is_symbol_node(node.kind())
        && let Some(name) = symbol_name_for(node, source)
    {
        let kind = normalize_symbol_kind(node.kind());
        let start_line = node.start_position().row as u32 + 1;
        let end_line = node.end_position().row as u32 + 1;
        let id = symbol_id(project_id, file_path, kind, &name, start_line);
        let body = node.utf8_text(source.as_bytes()).unwrap_or("").to_string();
        let signature = signature_for(node, source);
        let parent_id = scope.last().cloned();

        symbols.push(SymbolDraft {
            id: id.clone(),
            name,
            kind: kind.to_string(),
            signature,
            body,
            start_line,
            end_line,
            parent_id,
        });

        if node.kind() == "function_item" {
            collect_calls(node, project_id, file_path, source, &id, relations, true);
        }

        scope.push(id);
        visit_children(node, |child| {
            collect_rust_nodes(
                child, project_id, file_path, source, scope, symbols, relations,
            );
        });
        scope.pop();
        return;
    }

    visit_children(node, |child| {
        collect_rust_nodes(
            child, project_id, file_path, source, scope, symbols, relations,
        );
    });
}

fn collect_calls(
    symbol_node: Node<'_>,
    project_id: &str,
    file_path: &str,
    source: &str,
    from_symbol_id: &str,
    relations: &mut Vec<CodeRelation>,
    suppress_common_rust_calls: bool,
) {
    visit_descendants(symbol_node, &mut |node| {
        if !matches!(
            node.kind(),
            "call_expression" | "call" | "method_invocation" | "navigation_expression"
        ) {
            return;
        }
        let Some(function_node) = node
            .child_by_field_name("function")
            .or_else(|| node.child_by_field_name("name"))
            .or_else(|| node.child_by_field_name("called_expression"))
            .or_else(|| node.named_child(0))
        else {
            return;
        };
        let Ok(raw_name) = function_node.utf8_text(source.as_bytes()) else {
            return;
        };
        let target_name = normalize_call_target(raw_name);
        if target_name.is_empty() {
            return;
        }
        let relation_kind =
            if suppress_common_rust_calls && is_external_rust_call(raw_name, &target_name) {
                "external_call"
            } else {
                "calls"
            };
        relations.push(CodeRelation {
            id: relation_id(
                project_id,
                file_path,
                Some(from_symbol_id),
                relation_kind,
                &target_name,
            ),
            project_id: project_id.to_string(),
            from_symbol_id: Some(from_symbol_id.to_string()),
            from_file_path: file_path.to_string(),
            relation_kind: relation_kind.to_string(),
            target_name,
            target_symbol_id: None,
        });
    });
}

fn visit_children(node: Node<'_>, mut f: impl FnMut(Node<'_>)) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        f(child);
    }
}

fn visit_descendants(node: Node<'_>, f: &mut impl FnMut(Node<'_>)) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        f(child);
        visit_descendants(child, f);
    }
}

fn is_symbol_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "impl_item"
            | "mod_item"
            | "const_item"
            | "static_item"
            | "type_item"
            | "macro_definition"
    )
}

fn normalize_symbol_kind(kind: &str) -> &'static str {
    match kind {
        "function_item" => "function",
        "struct_item" => "struct",
        "enum_item" => "enum",
        "trait_item" => "trait",
        "impl_item" => "impl",
        "mod_item" => "module",
        "const_item" => "const",
        "static_item" => "static",
        "type_item" => "type",
        "macro_definition" => "macro",
        _ => "symbol",
    }
}

fn symbol_name_for(node: Node<'_>, source: &str) -> Option<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node
            .utf8_text(source.as_bytes())
            .ok()
            .map(|name| name.to_string());
    }

    if node.kind() == "impl_item" {
        let signature = signature_for(node, source);
        let name = signature
            .strip_prefix("impl")
            .unwrap_or(&signature)
            .trim()
            .trim_end_matches('{')
            .trim();
        if !name.is_empty() {
            return Some(format!("impl {}", truncate(&compact_whitespace(name), 160)));
        }
    }

    None
}

fn signature_for(node: Node<'_>, source: &str) -> String {
    let text = node.utf8_text(source.as_bytes()).unwrap_or("");
    let first_line = text.lines().next().unwrap_or("").trim();
    let before_body = text
        .split('{')
        .next()
        .unwrap_or(first_line)
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ");
    truncate(&compact_whitespace(&before_body), 500)
}

fn normalize_call_target(raw: &str) -> String {
    let raw = raw.trim();
    if raw.contains('.') {
        let raw = raw.rsplit('.').next().unwrap_or(raw);
        let raw = raw.split('<').next().unwrap_or(raw).trim_end_matches(':');
        return raw
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
    }
    raw.chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == ':')
        .collect()
}

fn is_external_rust_call(raw_name: &str, target_name: &str) -> bool {
    if matches!(target_name, "Ok" | "Err" | "Some" | "None") {
        return true;
    }
    if matches!(target_name, "drop" | "sum") {
        return true;
    }
    if is_common_external_rust_associated_call(target_name) || is_external_rust_path(target_name) {
        return true;
    }
    raw_name.contains('.')
        && matches!(
            target_name,
            "add_space"
                | "as_deref"
                | "as_ref"
                | "as_str"
                | "and_then"
                | "any"
                | "arg"
                | "button"
                | "clamp"
                | "chars"
                | "chain"
                | "center"
                | "clear"
                | "clicked"
                | "clone"
                | "cloned"
                | "collect"
                | "contains"
                | "contains_key"
                | "context"
                | "copied"
                | "cmp"
                | "count"
                | "dedup"
                | "entry"
                | "enumerate"
                | "expect"
                | "exists"
                | "execute"
                | "extend"
                | "filter"
                | "filter_map"
                | "find"
                | "file_name"
                | "flat_map"
                | "fold"
                | "flatten"
                | "get"
                | "get_mut"
                | "heading"
                | "horizontal_wrapped"
                | "insert"
                | "into"
                | "into_values"
                | "into_iter"
                | "is_ascii_alphanumeric"
                | "is_empty"
                | "is_none"
                | "is_some"
                | "is_some_and"
                | "iter"
                | "join"
                | "label"
                | "last"
                | "len"
                | "lines"
                | "map"
                | "map_err"
                | "max"
                | "min"
                | "next"
                | "num_columns"
                | "nth"
                | "ok"
                | "or"
                | "ok_or"
                | "ok_or_else"
                | "or_else"
                | "or_default"
                | "or_insert"
                | "or_insert_with"
                | "output"
                | "parse"
                | "parent"
                | "partial_cmp"
                | "push"
                | "push_str"
                | "remove"
                | "replace"
                | "retain"
                | "rsplit"
                | "as_array"
                | "as_bytes"
                | "as_object"
                | "ends_with"
                | "show"
                | "saturating_sub"
                | "saturating_mul"
                | "separator"
                | "sort"
                | "sort_by"
                | "sort_by_key"
                | "spacing"
                | "split"
                | "split_once"
                | "split_whitespace"
                | "starts_with"
                | "striped"
                | "strip_prefix"
                | "success"
                | "take"
                | "then"
                | "then_some"
                | "then_with"
                | "to_ascii_lowercase"
                | "to_hex"
                | "to_owned"
                | "to_path_buf"
                | "to_str"
                | "to_string"
                | "to_string_lossy"
                | "to_vec"
                | "transpose"
                | "trim"
                | "trim_end"
                | "trim_end_matches"
                | "trim_start"
                | "trim_matches"
                | "truncate"
                | "unwrap"
                | "unwrap_or"
                | "unwrap_or_default"
                | "unwrap_or_else"
                | "utf8_text"
                | "write_all"
                | "with_context"
        )
}

fn is_external_rust_path(target_name: &str) -> bool {
    let Some(root) = target_name
        .trim_start_matches("::")
        .split("::")
        .next()
        .filter(|root| !root.is_empty())
    else {
        return false;
    };
    matches!(
        root,
        "anyhow"
            | "axum"
            | "blake3"
            | "chrono"
            | "clap"
            | "egui"
            | "env"
            | "fs"
            | "ignore"
            | "pgvector"
            | "postgres"
            | "reqwest"
            | "serde"
            | "serde_json"
            | "std"
            | "tokio"
            | "toml"
            | "tower_http"
            | "tree_sitter"
            | "Color32"
            | "Pos2"
            | "Uuid"
            | "Vec2"
            | "uuid"
    )
}

fn is_common_external_rust_associated_call(target_name: &str) -> bool {
    if target_name.starts_with("BTreeMap::") && target_name.ends_with("::new") {
        return true;
    }
    matches!(
        target_name,
        "BTreeMap::new"
            | "BTreeSet::new"
            | "Color32::from_rgb"
            | "Command::new"
            | "HashMap::new"
            | "HashSet::new"
            | "PathBuf::from"
            | "PathBuf::new"
            | "Path::new"
            | "Pos2::new"
            | "String::from_utf8_lossy"
            | "String::new"
            | "Uuid::now_v7"
            | "Vec::new"
            | "Vec2::new"
            | "VecDeque::new"
    )
}

fn rust_use_targets(text: &str) -> Vec<String> {
    let tree = text
        .trim()
        .strip_prefix("use")
        .unwrap_or(text)
        .trim()
        .trim_end_matches(';')
        .trim();
    let mut targets = Vec::new();
    expand_rust_use_tree("", tree, &mut targets);
    targets.sort();
    targets.dedup();
    targets
}

fn expand_rust_use_tree(prefix: &str, tree: &str, targets: &mut Vec<String>) {
    let tree = tree.trim();
    if tree.is_empty() {
        return;
    }
    if let Some((open, close)) = top_level_brace_pair(tree) {
        let base = tree[..open].trim().trim_end_matches("::");
        let inside = &tree[open + 1..close];
        let base_prefix = join_rust_use_path(prefix, base);
        for item in split_top_level_commas(inside) {
            expand_rust_use_tree(&base_prefix, item, targets);
        }
        return;
    }
    if let Some(target) = clean_rust_use_leaf(prefix, tree) {
        targets.push(target);
    }
}

fn top_level_brace_pair(text: &str) -> Option<(usize, usize)> {
    let mut depth = 0usize;
    let mut open = None;
    for (index, ch) in text.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    open = Some(index);
                }
                depth += 1;
            }
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return open.map(|open| (open, index));
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_commas(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(text[start..index].trim());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim());
    parts
}

fn clean_rust_use_leaf(prefix: &str, leaf: &str) -> Option<String> {
    let leaf = leaf
        .split_once(" as ")
        .map(|(name, _)| name)
        .unwrap_or(leaf)
        .trim()
        .trim_end_matches(';')
        .trim();
    if leaf.is_empty() || leaf == "*" {
        return None;
    }
    let target = if leaf == "self" {
        prefix.trim().to_string()
    } else {
        join_rust_use_path(prefix, leaf)
    };
    (!target.is_empty() && !target.ends_with("::*")).then_some(target)
}

fn join_rust_use_path(prefix: &str, item: &str) -> String {
    let prefix = prefix.trim().trim_end_matches("::");
    let item = item.trim().trim_start_matches("::").trim_end_matches("::");
    if prefix.is_empty() {
        item.to_string()
    } else if item.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}::{item}")
    }
}

fn is_external_mod_item(node: Node<'_>, source: &str) -> bool {
    node.utf8_text(source.as_bytes())
        .map(|text| text.trim_end().ends_with(';'))
        .unwrap_or(false)
}

fn module_file_targets(file_path: &str, module_name: &str) -> Vec<String> {
    let module_name = module_name.trim();
    if module_name.is_empty() {
        return Vec::new();
    }
    let (parent, stem) = split_file_parent_stem(file_path);
    let base = if matches!(stem.as_str(), "lib" | "main" | "mod") {
        parent
    } else if parent.is_empty() {
        stem
    } else {
        format!("{parent}/{stem}")
    };
    if base.is_empty() {
        vec![format!("{module_name}.rs"), format!("{module_name}/mod.rs")]
    } else {
        vec![
            format!("{base}/{module_name}.rs"),
            format!("{base}/{module_name}/mod.rs"),
        ]
    }
}

fn split_file_parent_stem(file_path: &str) -> (String, String) {
    let (parent, file_name) = file_path.rsplit_once('/').unwrap_or(("", file_path));
    let stem = file_name.strip_suffix(".rs").unwrap_or(file_name);
    (parent.to_string(), stem.to_string())
}

fn cargo_package_name(source: &str) -> Option<String> {
    let mut in_package = false;
    for line in source.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "name" {
            continue;
        }
        return Some(
            value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string(),
        )
        .filter(|value| !value.is_empty());
    }
    None
}

fn symbol_id(project_id: &str, file_path: &str, kind: &str, name: &str, start_line: u32) -> String {
    let input = format!("{project_id}:{file_path}:{kind}:{name}:{start_line}");
    format!("sym_{}", blake3::hash(input.as_bytes()).to_hex())
}

fn relation_id(
    project_id: &str,
    file_path: &str,
    from_symbol_id: Option<&str>,
    relation_kind: &str,
    target_name: &str,
) -> String {
    let input = format!(
        "{}:{}:{}:{}:{}",
        project_id,
        file_path,
        from_symbol_id.unwrap_or("file"),
        relation_kind,
        target_name
    );
    format!("rel_{}", blake3::hash(input.as_bytes()).to_hex())
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn should_skip(path: &Path) -> bool {
    path.components().any(|component| {
        let value = component.as_os_str().to_string_lossy();
        matches!(
            value.as_ref(),
            ".git" | "target" | "node_modules" | ".dukememory" | ".codegraph"
        )
    })
}

fn is_indexable_code_file(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml")
        || path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| {
                matches!(
                    ext,
                    "rs" | "py"
                        | "js"
                        | "mjs"
                        | "jsx"
                        | "ts"
                        | "tsx"
                        | "go"
                        | "java"
                        | "kt"
                        | "kts"
                        | "swift"
                )
            })
}

fn compact_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{CodeSearchOptions, Store};

    #[test]
    fn indexes_rust_symbols_and_calls() -> Result<()> {
        let root = temp_project_dir("code-index")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join(".dukememory.toml"),
            "name = \"code-index-test\"\n",
        )?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
pub fn spawn_enemy() {
    update_ai();
}

fn update_ai() {}

pub struct EnemySpawner;
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index"))?;
        let report = index_project(
            &mut store,
            &root,
            Some("code-index-test".to_string()),
            false,
        )?;
        assert_eq!(report.files_indexed, 1);
        assert!(report.symbols_indexed >= 3);

        let results = store.search_code(
            "code-index-test",
            CodeSearchOptions {
                query: "spawn enemy".to_string(),
                limit: 10,
                kind: Some("function".to_string()),
                file_path: None,
            },
        )?;
        assert!(
            results
                .iter()
                .any(|result| result.symbol.name == "spawn_enemy")
        );

        let callees = store.find_callees("code-index-test", "spawn_enemy", 10)?;
        assert!(
            callees
                .iter()
                .any(|relation| relation.target_name == "update_ai")
        );
        assert!(callees.iter().any(|relation| {
            relation.target_name == "update_ai" && relation.target_symbol_id.is_some()
        }));
        assert!(report.calls_resolved >= 1);
        Ok(())
    }

    #[test]
    fn expands_rust_use_trees_into_individual_targets() {
        let targets = rust_use_targets(
            "use crate::{project::resolve_project_id_from_path, store::{CodeFile, Store as DbStore}};",
        );
        assert_eq!(
            targets,
            vec![
                "crate::project::resolve_project_id_from_path".to_string(),
                "crate::store::CodeFile".to_string(),
                "crate::store::Store".to_string(),
            ]
        );
    }

    #[test]
    fn resolves_expanded_rust_use_targets() -> Result<()> {
        let root = temp_project_dir("code-index-rust-use")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"rust-use\"\n")?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
mod store;

use crate::store::{CodeFile, Store};

pub fn build() {}
"#,
        )?;
        fs::write(
            root.join("src/store.rs"),
            r#"
pub struct CodeFile;
pub struct Store;
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-rust-use"))?;
        let report = index_project(&mut store, &root, Some("rust-use".to_string()), false)?;
        assert!(report.uses_resolved >= 2);

        let symbols = store.code_symbols_for_project("rust-use")?;
        let seed_ids = symbols
            .iter()
            .filter(|symbol| symbol.name == "CodeFile" || symbol.name == "Store")
            .map(|symbol| symbol.id.clone())
            .collect::<Vec<_>>();
        let (_, relations) = store.code_graph_for_symbols("rust-use", &seed_ids, 20)?;
        for target_name in ["crate::store::CodeFile", "crate::store::Store"] {
            assert!(
                relations.iter().any(|relation| {
                    relation.relation_kind == "uses"
                        && relation.target_name == target_name
                        && relation.target_symbol_id.is_some()
                }),
                "expected resolved use relation for {target_name}, got {relations:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn suppresses_common_rust_library_calls_from_project_graph() -> Result<()> {
        let root = temp_project_dir("code-index-rust-call-noise")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join(".dukememory.toml"),
            "name = \"rust-call-noise\"\n",
        )?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
use anyhow::{Context, Result};
use std::fs;

pub fn run(items: Vec<String>) -> Result<usize> {
    helper();
    push();
    let path = std::env::temp_dir().join("dukememory-noise.txt");
    fs::write(path, "noise").context("write fixture")?;
    let mut names = Vec::new();
    names.push("duke".to_string());
    let joined = names.join(",");
    let trimmed = joined.trim();
    let count = items.iter().map(|item| item.to_string()).collect::<Vec<_>>().len();
    Ok(count + trimmed.len())
}

fn helper() {}

fn push() {}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-rust-call-noise"))?;
        let report = index_project(
            &mut store,
            &root,
            Some("rust-call-noise".to_string()),
            false,
        )?;
        assert_eq!(report.calls_resolved, 2);

        let callees = store.find_callees("rust-call-noise", "run", 20)?;
        assert!(callees.iter().any(|relation| {
            relation.target_name == "helper" && relation.target_symbol_id.is_some()
        }));
        assert!(
            callees
                .iter()
                .any(|relation| relation.target_name == "push"
                    && relation.target_symbol_id.is_some())
        );
        for noisy_target in [
            "Vec::new",
            "anyhow::Context",
            "anyhow::Result",
            "context",
            "fs::write",
            "iter",
            "map",
            "std::env::temp_dir",
            "to_string",
            "collect",
            "collectVec_",
            "len",
            "join",
            "trim",
            "Ok",
        ] {
            assert!(
                !callees
                    .iter()
                    .any(|relation| relation.target_name == noisy_target),
                "did not expect noisy target {noisy_target} in {callees:?}"
            );
        }
        let status = store.code_status("rust-call-noise")?;
        assert!(
            status.relations > status.resolved_relations,
            "external relations should remain in total relation counts: {status:#?}"
        );
        assert!(status.relation_counts.external_call > 0);
        assert!(status.relation_counts.external_use > 0);
        assert_eq!(
            status.relation_counts.project_quality_unresolved,
            status.quality.unresolved_relations
        );
        assert_eq!(status.quality.unresolved_relations, 0);
        assert_eq!(status.quality.top_unresolved_targets.len(), 0);
        Ok(())
    }

    #[test]
    fn indexes_python_and_javascript_symbols_and_calls() -> Result<()> {
        let root = temp_project_dir("code-index-polyglot")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join(".dukememory.toml"),
            "name = \"code-index-polyglot\"\n",
        )?;
        fs::write(
            root.join("src/tasks.py"),
            r#"
def orchestrate():
    prepare()

def prepare():
    return True
"#,
        )?;
        fs::write(
            root.join("src/ui.js"),
            r#"
function render() {
  updateView();
}

function updateView() {
  return true;
}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-polyglot"))?;
        let report = index_project(
            &mut store,
            &root,
            Some("code-index-polyglot".to_string()),
            false,
        )?;
        assert_eq!(report.files_indexed, 2);
        assert!(report.symbols_indexed >= 4);
        assert!(report.calls_resolved >= 2);

        let python_callees = store.find_callees("code-index-polyglot", "orchestrate", 10)?;
        let python_call = python_callees
            .iter()
            .find(|relation| relation.target_name == "prepare")
            .expect("orchestrate should call prepare");
        assert!(python_call.target_symbol_id.is_some());

        let javascript_callees = store.find_callees("code-index-polyglot", "render", 10)?;
        let javascript_call = javascript_callees
            .iter()
            .find(|relation| relation.target_name == "updateView")
            .expect("render should call updateView");
        assert!(javascript_call.target_symbol_id.is_some());
        Ok(())
    }

    #[test]
    fn indexes_typescript_and_tsx_files() -> Result<()> {
        let root = temp_project_dir("code-index-typescript")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"typescript\"\n")?;
        fs::write(
            root.join("src/service.ts"),
            r#"
export function loadUser(id: string) {
  return formatUser(id);
}

function formatUser(id: string) {
  return id;
}
"#,
        )?;
        fs::write(
            root.join("src/view.tsx"),
            r#"
class UserCard {
  render() {
    loadUser("1");
    return <div />;
  }
}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-typescript"))?;
        let report = index_project(&mut store, &root, Some("typescript".to_string()), false)?;
        assert_eq!(report.files_indexed, 2);
        let symbols = store.code_symbols_for_project("typescript")?;
        assert!(symbols.iter().any(|symbol| {
            symbol.name == "loadUser"
                && symbol.language == "typescript"
                && symbol.file_path == "src/service.ts"
        }));
        assert!(symbols.iter().any(|symbol| {
            symbol.name == "UserCard"
                && symbol.language == "typescript"
                && symbol.file_path == "src/view.tsx"
        }));
        let fresh = check_code_index_freshness(&store, &root, Some("typescript".to_string()))?;
        assert!(fresh.is_fresh());
        Ok(())
    }

    #[test]
    fn indexes_go_java_kotlin_and_swift_files() -> Result<()> {
        let root = temp_project_dir("code-index-mobile-polyglot")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join(".dukememory.toml"),
            "name = \"mobile-polyglot\"\n",
        )?;
        fs::write(
            root.join("src/service.go"),
            r#"
package service

func LoadUser() {
    FormatUser()
}

func FormatUser() {}
"#,
        )?;
        fs::write(
            root.join("src/UserService.java"),
            r#"
class UserService {
    void loadUser() {
        formatUser();
    }
    void formatUser() {}
}
"#,
        )?;
        fs::write(
            root.join("src/UserView.kt"),
            r#"
class UserView {
    fun render() {
        loadUser()
    }
    fun loadUser() {}
}
"#,
        )?;
        fs::write(
            root.join("src/UserCard.swift"),
            r#"
struct UserCard {
    func render() {
        loadUser()
    }
    func loadUser() {}
}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-mobile-polyglot"))?;
        let report = index_project(
            &mut store,
            &root,
            Some("mobile-polyglot".to_string()),
            false,
        )?;
        assert_eq!(report.files_indexed, 4);
        let symbols = store.code_symbols_for_project("mobile-polyglot")?;
        for (name, language) in [
            ("LoadUser", "go"),
            ("UserService", "java"),
            ("UserView", "kotlin"),
            ("UserCard", "swift"),
        ] {
            assert!(
                symbols
                    .iter()
                    .any(|symbol| symbol.name == name && symbol.language == language),
                "missing {language} symbol {name}"
            );
        }
        for (source_name, source_language, source_file, target_name) in [
            ("LoadUser", "go", "src/service.go", "FormatUser"),
            ("loadUser", "java", "src/UserService.java", "formatUser"),
            ("render", "kotlin", "src/UserView.kt", "loadUser"),
            ("render", "swift", "src/UserCard.swift", "loadUser"),
        ] {
            let source = symbols
                .iter()
                .find(|symbol| {
                    symbol.name == source_name
                        && symbol.language == source_language
                        && symbol.file_path == source_file
                })
                .unwrap_or_else(|| panic!("missing {source_language} source symbol {source_name}"));
            let callees = store.find_callees("mobile-polyglot", &source.id, 10)?;
            assert!(
                callees.iter().any(|relation| {
                    relation.target_name == target_name && relation.target_symbol_id.is_some()
                }),
                "expected {source_language} {source_name} to resolve call to {target_name}, got {callees:?}"
            );
        }
        assert!(
            report.calls_resolved >= 4,
            "expected local calls to resolve across new language fixtures, got {}",
            report.calls_resolved
        );
        let status = store.code_status("mobile-polyglot")?;
        assert_eq!(status.languages.len(), 4);
        assert!(status.languages.iter().any(|language| {
            language.language == "go"
                && language.files == 1
                && language.symbols >= 2
                && language.resolved_relations >= 1
        }));
        assert_eq!(status.quality.unresolved_relations, 0);
        assert_eq!(status.quality.top_unresolved_targets.len(), 0);
        assert!(status.quality.relation_resolution_rate >= 0.75);
        let fresh = check_code_index_freshness(&store, &root, Some("mobile-polyglot".to_string()))?;
        assert!(fresh.is_fresh());
        Ok(())
    }

    #[test]
    fn code_index_freshness_reports_disk_drift() -> Result<()> {
        let root = temp_project_dir("code-index-freshness")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"freshness\"\n")?;
        fs::write(root.join("Cargo.toml"), "[package]\nname = \"freshness\"\n")?;
        fs::write(root.join("src/lib.rs"), "pub fn indexed() {}\n")?;

        let mut store = Store::open(&temp_database_marker("code-index-freshness"))?;
        index_project(&mut store, &root, Some("freshness".to_string()), false)?;
        let fresh = check_code_index_freshness(&store, &root, Some("freshness".to_string()))?;
        assert!(fresh.is_fresh());

        fs::write(root.join("src/lib.rs"), "pub fn indexed_changed() {}\n")?;
        fs::write(root.join("src/new.rs"), "pub fn missing_from_index() {}\n")?;
        fs::remove_file(root.join("Cargo.toml"))?;

        let stale = check_code_index_freshness(&store, &root, Some("freshness".to_string()))?;
        assert!(stale.stale_files.contains(&"src/lib.rs".to_string()));
        assert!(stale.missing_files.contains(&"src/new.rs".to_string()));
        assert!(stale.deleted_files.contains(&"Cargo.toml".to_string()));
        Ok(())
    }

    #[test]
    fn ignores_non_code_files_before_reading_as_utf8() -> Result<()> {
        let root = temp_project_dir("code-index-binary-skip")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"binary-skip\"\n")?;
        fs::write(root.join("src/lib.rs"), "pub fn indexed() {}\n")?;
        fs::write(root.join("schema.marker"), [0xff, 0x00, 0x80, 0x13])?;

        let mut store = Store::open(&temp_database_marker("code-index-binary-skip"))?;
        let report = index_project(&mut store, &root, Some("binary-skip".to_string()), false)?;
        assert_eq!(report.files_indexed, 1);
        assert_eq!(report.files_seen, 1);
        Ok(())
    }

    #[test]
    fn resolves_cross_file_calls_and_module_declarations() -> Result<()> {
        let root = temp_project_dir("code-index-cross-file")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"cross-file\"\n")?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
mod combat;

pub fn tick() {
    combat::apply_hit_stun();
}
"#,
        )?;
        fs::write(
            root.join("src/combat.rs"),
            r#"
pub fn apply_hit_stun() {}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-cross-file"))?;
        let report = index_project(&mut store, &root, Some("cross-file".to_string()), false)?;
        assert_eq!(report.files_indexed, 2);
        assert!(report.calls_resolved >= 1);
        assert!(report.modules_resolved >= 1);

        let callees = store.find_callees("cross-file", "tick", 10)?;
        let call = callees
            .iter()
            .find(|relation| relation.target_name == "combat::apply_hit_stun")
            .expect("tick should call apply_hit_stun");
        let target_id = call
            .target_symbol_id
            .as_deref()
            .expect("cross-file call should be resolved");
        let target = store
            .get_code_symbol("cross-file", target_id)?
            .expect("resolved target symbol should exist");
        assert_eq!(target.name, "apply_hit_stun");
        assert_eq!(target.file_path, "src/combat.rs");

        let callers = store.find_callers("cross-file", target_id, 10)?;
        assert!(callers.iter().any(|relation| {
            relation.from_symbol_id.as_deref() == call.from_symbol_id.as_deref()
                && relation.target_symbol_id.as_deref() == Some(target_id)
        }));
        Ok(())
    }

    #[test]
    fn resolves_qualified_call_when_target_name_is_ambiguous() -> Result<()> {
        let root = temp_project_dir("code-index-qualified-call")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"qualified-call\"\n")?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
mod combat;
mod ui;

pub fn tick() {
    combat::apply();
}
"#,
        )?;
        fs::write(root.join("src/combat.rs"), "pub fn apply() {}\n")?;
        fs::write(root.join("src/ui.rs"), "pub fn apply() {}\n")?;

        let mut store = Store::open(&temp_database_marker("code-index-qualified-call"))?;
        let report = index_project(&mut store, &root, Some("qualified-call".to_string()), false)?;
        assert_eq!(report.files_indexed, 3);
        assert!(report.calls_resolved >= 1);

        let callees = store.find_callees("qualified-call", "tick", 10)?;
        let call = callees
            .iter()
            .find(|relation| relation.target_name == "combat::apply")
            .expect("tick should call combat::apply");
        let target_id = call
            .target_symbol_id
            .as_deref()
            .expect("qualified call should resolve despite duplicate function names");
        let target = store
            .get_code_symbol("qualified-call", target_id)?
            .expect("resolved target symbol should exist");
        assert_eq!(target.name, "apply");
        assert_eq!(target.file_path, "src/combat.rs");
        Ok(())
    }

    #[test]
    fn resolves_type_qualified_impl_method_when_method_name_is_ambiguous() -> Result<()> {
        let root = temp_project_dir("code-index-impl-method")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"impl-method\"\n")?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
pub struct Store;
pub struct Client;

impl Store {
    pub fn open() {}
}

impl Client {
    pub fn open() {}
}

pub fn build() {
    Store::open();
}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-impl-method"))?;
        let report = index_project(&mut store, &root, Some("impl-method".to_string()), false)?;
        assert!(report.calls_resolved >= 1);

        let callees = store.find_callees("impl-method", "build", 10)?;
        let call = callees
            .iter()
            .find(|relation| relation.target_name == "Store::open")
            .expect("build should call Store::open");
        let target_id = call
            .target_symbol_id
            .as_deref()
            .expect("type-qualified impl method should resolve");
        let target = store
            .get_code_symbol("impl-method", target_id)?
            .expect("resolved impl method should exist");
        let parent_id = target
            .parent_id
            .as_deref()
            .expect("resolved method should have impl parent");
        let parent = store
            .get_code_symbol("impl-method", parent_id)?
            .expect("impl parent should exist");
        assert_eq!(target.name, "open");
        assert!(parent.name.contains("Store"));
        Ok(())
    }

    #[test]
    fn resolves_qualified_enum_variant_call_to_project_enum() -> Result<()> {
        let root = temp_project_dir("code-index-enum-variant")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"enum-variant\"\n")?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
pub enum StatusFilter {
    One(u8),
    Any,
}

pub fn build() {
    let _status = StatusFilter::One(1);
}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-enum-variant"))?;
        let report = index_project(&mut store, &root, Some("enum-variant".to_string()), false)?;
        assert!(report.calls_resolved >= 1);

        let callees = store.find_callees("enum-variant", "build", 10)?;
        let call = callees
            .iter()
            .find(|relation| relation.target_name == "StatusFilter::One")
            .expect("build should construct StatusFilter::One");
        let target_id = call
            .target_symbol_id
            .as_deref()
            .expect("qualified enum variant should resolve to enum symbol");
        let target = store
            .get_code_symbol("enum-variant", target_id)?
            .expect("resolved enum should exist");
        assert_eq!(target.kind, "enum");
        assert_eq!(target.name, "StatusFilter");
        Ok(())
    }

    #[test]
    fn indexes_only_changed_or_deleted_rust_files() -> Result<()> {
        let root = temp_project_dir("code-index-incremental")?;
        let src_dir = root.join("src");
        let source_path = src_dir.join("lib.rs");
        fs::create_dir_all(&src_dir)?;
        fs::write(root.join(".dukememory.toml"), "name = \"incremental\"\n")?;
        fs::write(
            &source_path,
            r#"
pub fn alpha() {
    beta();
}

fn beta() {}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-incremental"))?;
        let first = index_project(&mut store, &root, Some("incremental".to_string()), false)?;
        assert_eq!(first.files_indexed, 1);
        assert_eq!(first.files_skipped, 0);
        assert_eq!(first.files_deleted, 0);
        assert_eq!(first.indexed_files, vec!["src/lib.rs".to_string()]);
        let first_status = store.code_status("incremental")?;

        let second = index_project(&mut store, &root, Some("incremental".to_string()), false)?;
        assert_eq!(second.files_indexed, 0);
        assert_eq!(second.files_skipped, 1);
        assert_eq!(second.files_deleted, 0);
        assert!(second.indexed_files.is_empty());
        let second_status = store.code_status("incremental")?;
        assert_eq!(second_status.files, first_status.files);
        assert_eq!(second_status.symbols, first_status.symbols);

        fs::write(
            &source_path,
            r#"
pub fn gamma() {}
"#,
        )?;
        let changed = index_project(&mut store, &root, Some("incremental".to_string()), false)?;
        assert_eq!(changed.files_indexed, 1);
        assert_eq!(changed.files_skipped, 0);
        assert_eq!(changed.files_deleted, 0);
        assert_eq!(changed.indexed_files, vec!["src/lib.rs".to_string()]);

        let old_results = store.search_code(
            "incremental",
            CodeSearchOptions {
                query: "alpha".to_string(),
                limit: 10,
                kind: Some("function".to_string()),
                file_path: None,
            },
        )?;
        assert!(old_results.is_empty());

        let new_results = store.search_code(
            "incremental",
            CodeSearchOptions {
                query: "gamma".to_string(),
                limit: 10,
                kind: Some("function".to_string()),
                file_path: None,
            },
        )?;
        assert!(
            new_results
                .iter()
                .any(|result| result.symbol.name == "gamma")
        );

        fs::remove_file(&source_path)?;
        let deleted = index_project(&mut store, &root, Some("incremental".to_string()), false)?;
        assert_eq!(deleted.files_indexed, 0);
        assert_eq!(deleted.files_deleted, 1);
        assert!(deleted.indexed_files.is_empty());
        let final_status = store.code_status("incremental")?;
        assert_eq!(final_status.files, 0);
        assert_eq!(final_status.symbols, 0);
        assert_eq!(final_status.relations, 0);
        Ok(())
    }

    #[test]
    fn lists_missing_symbol_embeddings_for_selected_indexed_files() -> Result<()> {
        let root = temp_project_dir("code-index-embedding-files")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join(".dukememory.toml"),
            "name = \"embedding-files\"\n",
        )?;
        fs::write(root.join("src/lib.rs"), "pub fn alpha() {}\nfn beta() {}\n")?;
        fs::write(root.join("src/other.rs"), "pub fn omega() {}\n")?;

        let mut store = Store::open(&temp_database_marker("code-index-embedding-files"))?;
        index_project(
            &mut store,
            &root,
            Some("embedding-files".to_string()),
            false,
        )?;

        let lib_only = vec!["src/lib.rs".to_string()];
        let symbols = store.code_symbols_missing_embeddings_for_files(
            "embedding-files",
            "test-embed-model",
            &lib_only,
            100,
        )?;
        assert!(symbols.iter().any(|symbol| symbol.name == "alpha"));
        assert!(symbols.iter().any(|symbol| symbol.name == "beta"));
        assert!(!symbols.iter().any(|symbol| symbol.name == "omega"));

        let alpha = symbols
            .iter()
            .find(|symbol| symbol.name == "alpha")
            .expect("alpha should be indexed");
        store.set_code_symbol_embedding_with_cache(
            "embedding-files",
            &alpha.id,
            "test-embed-model",
            "test-alpha-cache-key",
            &vec![0.1; 1024],
        )?;
        store.set_code_symbol_embedding_kind_with_cache(
            "embedding-files",
            &alpha.id,
            "test-embed-model",
            "signature",
            "test-alpha-signature-cache-key",
            &vec![0.2; 1024],
        )?;
        let remaining = store.code_symbols_missing_embeddings_for_files(
            "embedding-files",
            "test-embed-model",
            &lib_only,
            100,
        )?;
        assert!(!remaining.iter().any(|symbol| symbol.name == "alpha"));
        assert!(remaining.iter().any(|symbol| symbol.name == "beta"));
        Ok(())
    }

    #[test]
    fn indexes_cargo_metadata_and_external_modules() -> Result<()> {
        let root = temp_project_dir("code-index-cargo-mod")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"cargo-mod-test\"\nedition = \"2024\"\n",
        )?;
        fs::write(
            root.join("src/lib.rs"),
            "mod gameplay;\npub fn start() {}\n",
        )?;
        fs::write(root.join("src/gameplay.rs"), "pub fn tick() {}\n")?;

        let mut store = Store::open(&temp_database_marker("code-index-cargo-mod"))?;
        let report = index_project(&mut store, &root, Some("cargo-mod-test".to_string()), false)?;
        assert_eq!(report.files_indexed, 3);
        assert!(report.indexed_files.contains(&"Cargo.toml".to_string()));
        assert!(report.relations_indexed >= 3);
        assert!(report.timing.total_ms >= report.timing.parse_ms);
        assert!(report.timing.total_ms >= report.timing.db_write_ms);
        assert!(report.timing.total_ms >= report.timing.resolve_ms);

        let status = store.code_status("cargo-mod-test")?;
        assert_eq!(status.files, 3);
        assert!(status.relations >= 3);
        Ok(())
    }

    #[test]
    fn relations_indexed_counts_inserted_unique_relations() -> Result<()> {
        let root = temp_project_dir("code-index-relation-count")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(root.join(".dukememory.toml"), "name = \"relation-count\"\n")?;
        fs::write(
            root.join("src/lib.rs"),
            r#"
pub fn tick() {
    update_ai();
    update_ai();
}

fn update_ai() {}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("code-index-relation-count"))?;
        let report = index_project(&mut store, &root, Some("relation-count".to_string()), false)?;
        let status = store.code_status("relation-count")?;

        assert_eq!(
            report.relations_indexed as u64, status.relations,
            "index report should count rows inserted into code_relations, not duplicate insert attempts"
        );
        Ok(())
    }

    #[test]
    fn indexes_python_and_javascript_symbols() -> Result<()> {
        let root = temp_project_dir("multi-lang-index")?;
        fs::create_dir_all(root.join("src"))?;
        fs::write(
            root.join(".dukememory.toml"),
            "name = \"multi-lang-test\"\n",
        )?;

        fs::write(
            root.join("src/utils.py"),
            r#"
class DataProcessor:
    def process_data(self, value):
        return value * 2

def global_compute(x):
    return x + 1
"#,
        )?;

        fs::write(
            root.join("src/helper.js"),
            r#"
class UIBuilder {
    render() {
        console.log("rendering");
    }
}

function debugLog(msg) {
    console.log(msg);
}
"#,
        )?;

        let mut store = Store::open(&temp_database_marker("multi-lang-index"))?;
        let report = index_project(
            &mut store,
            &root,
            Some("multi-lang-test".to_string()),
            false,
        )?;
        assert_eq!(report.files_indexed, 2);

        let symbols = store.code_symbols_for_project("multi-lang-test")?;

        // Assert python symbols
        let py_class = symbols.iter().find(|s| s.name == "DataProcessor").unwrap();
        assert_eq!(py_class.kind, "class");
        assert_eq!(py_class.language, "python");
        assert_eq!(py_class.signature, "class DataProcessor");

        let py_method = symbols.iter().find(|s| s.name == "process_data").unwrap();
        assert_eq!(py_method.kind, "function");
        assert_eq!(py_method.language, "python");
        assert_eq!(py_method.signature, "def process_data(self, value)");
        assert_eq!(py_method.parent_id, Some(py_class.id.clone()));

        let py_func = symbols.iter().find(|s| s.name == "global_compute").unwrap();
        assert_eq!(py_func.kind, "function");
        assert_eq!(py_func.language, "python");
        assert_eq!(py_func.signature, "def global_compute(x)");

        // Assert javascript symbols
        let js_class = symbols.iter().find(|s| s.name == "UIBuilder").unwrap();
        assert_eq!(js_class.kind, "class");
        assert_eq!(js_class.language, "javascript");
        assert_eq!(js_class.signature, "class UIBuilder");

        let js_method = symbols.iter().find(|s| s.name == "render").unwrap();
        assert_eq!(js_method.kind, "method");
        assert_eq!(js_method.language, "javascript");
        assert_eq!(js_method.signature, "render()");
        assert_eq!(js_method.parent_id, Some(js_class.id.clone()));

        let js_func = symbols.iter().find(|s| s.name == "debugLog").unwrap();
        assert_eq!(js_func.kind, "function");
        assert_eq!(js_func.language, "javascript");
        assert_eq!(js_func.signature, "function debugLog(msg)");

        Ok(())
    }

    fn temp_project_dir(name: &str) -> Result<PathBuf> {
        let root =
            std::env::temp_dir().join(format!("dukememory-test-{name}-{}", uuid::Uuid::now_v7()));
        fs::create_dir_all(&root)?;
        Ok(root)
    }

    fn temp_database_marker(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "dukememory-test-{name}-{}.schema-marker",
            uuid::Uuid::now_v7()
        ))
    }
}

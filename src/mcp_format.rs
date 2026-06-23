use serde_json::Value;

use crate::code_assist::CodeAssistReport;
use crate::code_reason::CodeReasonReport;
use crate::store::{
    CodeFile, CodeMemory, CodeRelation, CodeRouteHint, CodeSearchResult, CodeSymbol, Memory,
};

pub fn memory_eval_text(memory: &Memory) -> String {
    format!(
        "{}\n{}\n{}\n{}",
        memory.kind,
        memory.scope,
        memory.body,
        memory.tags.join(" ")
    )
}

pub fn format_memories(header: &str, memories: &[Memory]) -> String {
    let mut text = header.to_string();
    for memory in memories {
        text.push('\n');
        text.push_str(&format_memory("", memory));
    }
    text
}

pub fn format_memory(header: &str, memory: &Memory) -> String {
    let mut text = String::new();
    if !header.is_empty() {
        text.push_str(header);
        text.push('\n');
    }
    text.push_str(&format!(
        "- id: {}\n  scope: {}\n  kind: {}\n  status: {}\n  importance: {:.2}\n  confidence: {:.2}\n  updated_at: {}\n  body: {}",
        memory.id,
        memory.scope,
        memory.kind,
        memory.status,
        memory.importance,
        memory.confidence,
        memory.updated_at,
        memory.body
    ));
    if !memory.tags.is_empty() {
        text.push_str(&format!("\n  tags: {}", memory.tags.join(",")));
    }
    if let Some(source) = &memory.source {
        text.push_str(&format!("\n  source: {source}"));
    }
    if let Some(superseded_by) = &memory.superseded_by {
        text.push_str(&format!("\n  superseded_by: {superseded_by}"));
    }
    if let Some(reason) = &memory.status_reason {
        text.push_str(&format!("\n  status_reason: {reason}"));
    }
    text
}

pub fn format_code_results(header: &str, results: &[CodeSearchResult]) -> String {
    let mut text = header.to_string();
    for result in results {
        text.push('\n');
        text.push_str(&format!("score: {:.4}\n", result.score));
        text.push_str(&format_code_symbol("", &result.symbol, false));
    }
    text
}

pub fn format_code_patterns_text(
    report: &crate::code_assist::CodePatternReport,
    actual_mode: &str,
    warning: Option<&str>,
) -> String {
    let mut text = format!(
        "Code patterns for project `{}` query `{}`: {} symbols, {} pattern groups, mode={actual_mode}",
        report.project_id,
        report.query,
        report.symbols.len(),
        report.patterns.len()
    );
    if let Some(warning) = warning {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    for pattern in &report.patterns {
        text.push_str(&format!(
            "\n- seed `{}` {}:{}",
            pattern.seed.kind, pattern.seed.file_path, pattern.seed.name
        ));
        for related in &pattern.related_symbols {
            text.push_str(&format!(
                "\n  - {:.3} `{}` {}:{}",
                related.score, related.symbol.kind, related.symbol.file_path, related.symbol.name
            ));
        }
    }
    if !report.affected_tests.is_empty() {
        text.push_str("\nAffected tests:");
        for file in &report.affected_tests {
            text.push_str(&format!("\n- {file}"));
        }
    }
    if !report.memory_suggestions.is_empty() {
        text.push_str("\nPending code-memory suggestions:");
        for suggestion in &report.memory_suggestions {
            text.push_str(&format!("\n- [{}] {}", suggestion.kind, suggestion.body));
        }
    }
    text
}

pub fn format_code_assist_text(report: &CodeAssistReport) -> String {
    let mut text = format!(
        "Code assist for project `{}` query `{}`: {} symbols, {} pattern groups, {} duplicate pairs, {} affected tests",
        report.project_id,
        report.query,
        report.symbols.len(),
        report.patterns.len(),
        report.duplicate_pairs.len(),
        report.affected_tests.len()
    );
    text.push_str(&format!("\nactual_mode: {}", report.actual_mode));
    if let Some(warning) = &report.warning {
        text.push_str(&format!("\nwarning: {warning}"));
    }
    if !report.impacted_files.is_empty() {
        text.push_str("\nImpacted files:");
        for file in &report.impacted_files {
            text.push_str(&format!("\n- {file}"));
        }
    }
    if !report.affected_tests.is_empty() {
        text.push_str("\nAffected tests:");
        for file in &report.affected_tests {
            text.push_str(&format!("\n- {file}"));
        }
    }
    if !report.memory_suggestions.is_empty() {
        text.push_str("\nPending code-memory suggestions:");
        for suggestion in &report.memory_suggestions {
            text.push_str(&format!("\n- [{}] {}", suggestion.kind, suggestion.body));
        }
    }
    text
}

pub struct CodeExploreFormat<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub results: &'a [CodeSearchResult],
    pub code_memories: &'a [CodeMemory],
    pub routes: &'a [CodeRouteHint],
    pub impact: &'a [Value],
    pub freshness: Option<&'a Value>,
    pub include_body: bool,
}

pub fn format_code_explore(input: CodeExploreFormat<'_>) -> String {
    let mut text = format!(
        "Code explore for project `{}` query `{}`: {} symbols",
        input.project_id,
        input.query,
        input.results.len()
    );
    for result in input.results {
        text.push('\n');
        text.push_str(&format!("score: {:.4}\n", result.score));
        text.push_str(&format_code_symbol("", &result.symbol, input.include_body));
    }
    if !input.code_memories.is_empty() {
        text.push_str("\n\n");
        text.push_str(&format_code_memories(
            "Related dukememory code memories:",
            input.code_memories,
        ));
    }
    if !input.routes.is_empty() {
        text.push_str("\n\nRoute hints:");
        for route in input.routes {
            text.push_str(&format!(
                "\n- {} {} -> {} ({})\n  file: {}\n  symbol: {}\n  evidence: {}",
                route.method.as_deref().unwrap_or("*"),
                route.route,
                route.symbol_name,
                route.framework,
                route.file_path,
                route.symbol_id,
                route.evidence
            ));
        }
    }
    if !input.impact.is_empty() {
        text.push_str("\n\nImpact:");
        for item in input.impact {
            text.push_str(&format!("\n- {item}"));
        }
    }
    if let Some(freshness) = input.freshness {
        text.push_str(&format!("\n\nFreshness: {freshness}"));
    }
    text
}

pub fn format_code_symbol(header: &str, symbol: &CodeSymbol, include_body: bool) -> String {
    let mut text = String::new();
    if !header.is_empty() {
        text.push_str(header);
        text.push('\n');
    }
    text.push_str(&format!(
        "- id: {}\n  file: {}:{}-{}\n  kind: {}\n  name: {}\n  signature: {}",
        symbol.id,
        symbol.file_path,
        symbol.start_line,
        symbol.end_line,
        symbol.kind,
        symbol.name,
        symbol.signature
    ));
    if let Some(parent_id) = &symbol.parent_id {
        text.push_str(&format!("\n  parent_id: {parent_id}"));
    }
    if include_body {
        text.push_str("\n```rust\n");
        text.push_str(&symbol.body);
        text.push_str("\n```");
    }
    text
}

pub fn format_code_memories(header: &str, memories: &[CodeMemory]) -> String {
    let mut text = header.to_string();
    for memory in memories {
        text.push_str(&format!(
            "\n- id: {}\n  kind: {}\n  status: {}\n  link_status: {}\n  confidence: {:.2}\n  symbol_id: {}\n  file_path: {}\n  updated_at: {}\n  body: {}",
            memory.id,
            memory.kind,
            memory.status,
            memory.link_status,
            memory.confidence,
            memory.symbol_id.as_deref().unwrap_or("-"),
            memory.file_path.as_deref().unwrap_or("-"),
            memory.updated_at,
            memory.body
        ));
        if memory.symbol_name.is_some() || memory.symbol_kind.is_some() {
            text.push_str(&format!(
                "\n  symbol_snapshot: {} {}",
                memory.symbol_kind.as_deref().unwrap_or("-"),
                memory.symbol_name.as_deref().unwrap_or("-")
            ));
        }
        if memory.symbol_start_line.is_some() || memory.symbol_end_line.is_some() {
            text.push_str(&format!(
                "\n  symbol_lines: {}-{}",
                memory
                    .symbol_start_line
                    .map(|line| line.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                memory
                    .symbol_end_line
                    .map(|line| line.to_string())
                    .unwrap_or_else(|| "-".to_string())
            ));
        }
        if memory.relink_attempts > 0 {
            text.push_str(&format!("\n  relink_attempts: {}", memory.relink_attempts));
        }
        if let Some(last_relinked_at) = &memory.last_relinked_at {
            text.push_str(&format!("\n  last_relinked_at: {last_relinked_at}"));
        }
        if !memory.tags.is_empty() {
            text.push_str(&format!("\n  tags: {}", memory.tags.join(",")));
        }
        if let Some(source) = &memory.source {
            text.push_str(&format!("\n  source: {source}"));
        }
        if let Some(reason) = &memory.status_reason {
            text.push_str(&format!("\n  status_reason: {reason}"));
        }
    }
    text
}

pub fn format_code_files(header: &str, files: &[CodeFile]) -> String {
    let mut text = header.to_string();
    for file in files {
        text.push_str(&format!(
            "\n- path: {}\n  language: {}\n  size: {} bytes\n  lines: {}",
            file.path, file.language, file.size_bytes, file.line_count
        ));
    }
    text
}

pub fn format_code_outline(header: &str, symbols: &[CodeSymbol]) -> String {
    let mut text = header.to_string();
    for symbol in symbols {
        text.push_str(&format!(
            "\n- name: {}\n  kind: {}\n  lines: {}-{}\n  id: {}\n  signature: {}",
            symbol.name,
            symbol.kind,
            symbol.start_line,
            symbol.end_line,
            symbol.id,
            symbol.signature
        ));
        if let Some(parent_id) = &symbol.parent_id {
            text.push_str(&format!("\n  parent_id: {parent_id}"));
        }
    }
    text
}

pub fn format_code_reason_report(report: &CodeReasonReport) -> String {
    let mut text = format!(
        "Code {} using `{}`:\n{}",
        report.task, report.model, report.answer
    );
    if !report.bullets.is_empty() {
        text.push_str("\n\nBullets:");
        for item in &report.bullets {
            text.push_str(&format!("\n- {item}"));
        }
    }
    if !report.risks.is_empty() {
        text.push_str("\n\nRisks:");
        for item in &report.risks {
            text.push_str(&format!("\n- {item}"));
        }
    }
    if !report.next_steps.is_empty() {
        text.push_str("\n\nNext steps:");
        for item in &report.next_steps {
            text.push_str(&format!("\n- {item}"));
        }
    }
    text
}

pub fn format_relations(header: &str, relations: &[CodeRelation]) -> String {
    let mut text = header.to_string();
    if relations.is_empty() {
        text.push_str("\n- none");
        return text;
    }
    for relation in relations {
        text.push_str(&format!(
            "\n- {} {} from {}",
            relation.relation_kind, relation.target_name, relation.from_file_path
        ));
        if let Some(from_symbol_id) = &relation.from_symbol_id {
            text.push_str(&format!(" ({from_symbol_id})"));
        }
    }
    text
}

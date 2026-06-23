use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::store::{CodeMemory, CodeSearchResult, Memory, MemoryGraph};

const CHARS_PER_TOKEN: usize = 4;
const MIN_CONTEXT_CHARS: usize = 2_000;
const MAX_CONTEXT_CHARS: usize = 120_000;
const MAX_FRAGMENT_CHARS: usize = 700;
const MAX_CORE_FRAGMENT_CHARS: usize = 520;

#[derive(Debug, Clone, Serialize)]
pub struct MemoryContextFragment {
    pub memory_id: String,
    pub fragment_id: String,
    pub section: String,
    pub rank: usize,
    pub kind: String,
    pub memory_tier: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub importance: f64,
    pub confidence: f64,
    pub memory_score: Option<f64>,
    pub fragment_score: f64,
    pub reason: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextMemorySummary {
    pub id: String,
    pub kind: String,
    pub memory_tier: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub importance: f64,
    pub confidence: f64,
    pub score: Option<f64>,
    pub included_fragments: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeContextSummary {
    pub score: f64,
    pub symbol: CodeSymbolSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeMemorySummary {
    pub id: String,
    pub symbol_id: Option<String>,
    pub file_path: Option<String>,
    pub link_status: String,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub kind: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub confidence: f64,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeSymbolSummary {
    pub id: String,
    pub project_id: String,
    pub language: String,
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub signature: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_id: Option<String>,
}

pub struct ProjectContextFormat<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub mode: &'a str,
    pub memories: &'a [Memory],
    pub memory_fragments: &'a [MemoryContextFragment],
    pub code: &'a [CodeSearchResult],
    pub code_memories: &'a [CodeMemory],
    pub graph: &'a MemoryGraph,
    pub total_memories: u64,
    pub indexed_symbols: u64,
    pub token_budget: usize,
}

pub fn context_char_budget(token_budget: usize) -> usize {
    token_budget
        .saturating_mul(CHARS_PER_TOKEN)
        .clamp(MIN_CONTEXT_CHARS, MAX_CONTEXT_CHARS)
}

pub fn build_memory_fragments(
    query: &str,
    memories: &[Memory],
    token_budget: usize,
) -> Vec<MemoryContextFragment> {
    let terms = query_terms(query);
    let mut fragments = Vec::new();
    let mut used_chars = 0usize;
    let fragment_budget = (context_char_budget(token_budget) / 2).max(MAX_FRAGMENT_CHARS);

    for memory in memories {
        let is_core = is_core_context_memory(memory);
        let max_chars = if is_core {
            MAX_CORE_FRAGMENT_CHARS
        } else {
            MAX_FRAGMENT_CHARS
        };
        let section = if is_core { "core" } else { "task" };
        let reason = if is_core {
            "task-relevant core/project-rule fragment"
        } else {
            "task-relevant retrieved memory fragment"
        };
        let mut chunks = scored_chunks(&memory.body, &terms, max_chars);
        if chunks.is_empty() {
            continue;
        }
        chunks.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.index.cmp(&right.index))
        });
        let per_memory_limit = if is_core || memory.body.chars().count() <= max_chars {
            1
        } else {
            2
        };
        for chunk in chunks.into_iter().take(per_memory_limit) {
            let text = chunk.text.trim();
            if text.is_empty() {
                continue;
            }
            let text_chars = text.chars().count();
            if !fragments.is_empty() && used_chars.saturating_add(text_chars) > fragment_budget {
                break;
            }
            used_chars = used_chars.saturating_add(text_chars);
            let rank = fragments.len() + 1;
            fragments.push(MemoryContextFragment {
                memory_id: memory.id.clone(),
                fragment_id: format!("{}#{}", memory.id, chunk.index + 1),
                section: section.to_string(),
                rank,
                kind: memory.kind.clone(),
                memory_tier: memory.memory_tier.clone(),
                source: memory.source.clone(),
                tags: memory.tags.clone(),
                importance: memory.importance,
                confidence: memory.confidence,
                memory_score: memory.score,
                fragment_score: chunk.score,
                reason: reason.to_string(),
                text: text.to_string(),
            });
        }
        if used_chars >= fragment_budget {
            break;
        }
    }
    fragments
}

pub fn merge_core_and_task_memories(
    core_memories: Vec<Memory>,
    task_memories: Vec<Memory>,
    task_limit: usize,
) -> Vec<Memory> {
    let task_limit = task_limit.clamp(0, 50);
    let mut seen = HashSet::new();
    let mut merged = Vec::with_capacity(core_memories.len().saturating_add(task_limit));
    for memory in core_memories {
        if seen.insert(memory.id.clone()) {
            merged.push(memory);
        }
    }
    let mut task_added = 0usize;
    for memory in task_memories {
        if seen.insert(memory.id.clone()) {
            merged.push(memory);
            task_added += 1;
        }
        if task_added >= task_limit {
            break;
        }
    }
    merged
}

pub fn code_context_summaries(code: &[CodeSearchResult]) -> Vec<CodeContextSummary> {
    code.iter()
        .map(|result| CodeContextSummary {
            score: result.score,
            symbol: CodeSymbolSummary {
                id: result.symbol.id.clone(),
                project_id: result.symbol.project_id.clone(),
                language: result.symbol.language.clone(),
                file_path: result.symbol.file_path.clone(),
                name: result.symbol.name.clone(),
                kind: result.symbol.kind.clone(),
                signature: result.symbol.signature.clone(),
                start_line: result.symbol.start_line,
                end_line: result.symbol.end_line,
                parent_id: result.symbol.parent_id.clone(),
            },
        })
        .collect()
}

pub fn code_memory_summaries(code_memories: &[CodeMemory]) -> Vec<CodeMemorySummary> {
    code_memories
        .iter()
        .map(|memory| CodeMemorySummary {
            id: memory.id.clone(),
            symbol_id: memory.symbol_id.clone(),
            file_path: memory.file_path.clone(),
            link_status: memory.link_status.clone(),
            symbol_name: memory.symbol_name.clone(),
            symbol_kind: memory.symbol_kind.clone(),
            kind: memory.kind.clone(),
            tags: memory.tags.clone(),
            source: memory.source.clone(),
            confidence: memory.confidence,
            score: memory.score,
        })
        .collect()
}

pub fn fragment_memory_ids(fragments: &[MemoryContextFragment]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    for fragment in fragments {
        if seen.insert(fragment.memory_id.clone()) {
            ids.push(fragment.memory_id.clone());
        }
    }
    ids
}

pub fn context_memory_summaries(
    memories: &[Memory],
    fragments: &[MemoryContextFragment],
) -> Vec<ContextMemorySummary> {
    let mut counts = HashMap::<&str, usize>::new();
    for fragment in fragments {
        *counts.entry(fragment.memory_id.as_str()).or_default() += 1;
    }
    memories
        .iter()
        .map(|memory| ContextMemorySummary {
            id: memory.id.clone(),
            kind: memory.kind.clone(),
            memory_tier: memory.memory_tier.clone(),
            source: memory.source.clone(),
            tags: memory.tags.clone(),
            importance: memory.importance,
            confidence: memory.confidence,
            score: memory.score,
            included_fragments: counts.get(memory.id.as_str()).copied().unwrap_or(0),
        })
        .collect()
}

pub fn format_task_context(input: ProjectContextFormat<'_>) -> String {
    let mut text = format!(
        "DUKEMEMORY TASK-SCOPED PROJECT CONTEXT\nproject_id: {}\nquery: {}\nmode: {}\ntask_scoped: true\nselected_memory_hits: {}\nselected_memory_fragments: {}\nindexed_code_hits: {}\nselected_code_memories: {}\nmemory_graph_entities: {}\nmemory_graph_facts: {}\nmemory_graph_edges: {}\ntotal_memories: {}\nindexed_symbols: {}\ncontext_token_budget: {}\n\nUse this context only for the current project and current task. Full project memory is not loaded here: only task-selected fragments are included, including core/project-rule fragments that matched the task query. Code memories are loaded only for task-selected symbols/files. Treat memory graph relations as project-scoped navigation hints. Treat code relations as approximate navigation hints.\n",
        input.project_id,
        input.query,
        input.mode,
        input.memories.len(),
        input.memory_fragments.len(),
        input.code.len(),
        input.code_memories.len(),
        input.graph.entities.len(),
        input.graph.facts.len(),
        input.graph.edges.len(),
        input.total_memories,
        input.indexed_symbols,
        input.token_budget
    );

    push_fragment_block(&mut text, "CORE MEMORIES", input.memory_fragments, "core");
    push_fragment_block(
        &mut text,
        "TASK MEMORY FRAGMENTS",
        input.memory_fragments,
        "task",
    );

    text.push_str("\nSCOPED MEMORY GRAPH\n");
    text.push_str(&format_memory_graph_context(input.graph));

    text.push_str("\nCODE MEMORIES\n");
    if input.code_memories.is_empty() {
        text.push_str("- none\n");
    } else {
        for memory in input.code_memories {
            text.push_str(&format!(
                "- [{}] {} symbol={} file={} link={} status={} confidence {:.2}\n  {}\n",
                memory.kind,
                memory.id,
                memory.symbol_id.as_deref().unwrap_or("-"),
                memory.file_path.as_deref().unwrap_or("-"),
                memory.link_status,
                memory.status,
                memory.confidence,
                memory.body.replace('\n', "\n  ")
            ));
        }
    }

    text.push_str("\nINDEXED CODE\n");
    if input.code.is_empty() {
        text.push_str("- none\n");
    } else {
        for result in input.code {
            let symbol = &result.symbol;
            text.push_str(&format!(
                "- [{}] {} in {}:{}-{} (score {:.4})\n  {}\n",
                symbol.kind,
                symbol.name,
                symbol.file_path,
                symbol.start_line,
                symbol.end_line,
                result.score,
                symbol.signature
            ));
        }
    }

    text
}

fn push_fragment_block(
    text: &mut String,
    title: &str,
    fragments: &[MemoryContextFragment],
    section: &str,
) {
    text.push_str(&format!("\n{title}\n"));
    let mut any = false;
    for fragment in fragments
        .iter()
        .filter(|fragment| fragment.section == section)
    {
        any = true;
        text.push_str(&format!(
            "- [{}] {} (fragment {}, tier {}, importance {:.2}, confidence {:.2}, reason: {})\n  {}\n",
            fragment.kind,
            fragment.memory_id,
            fragment.fragment_id,
            fragment.memory_tier,
            fragment.importance,
            fragment.confidence,
            fragment.reason,
            fragment.text.replace('\n', "\n  ")
        ));
        if !fragment.tags.is_empty() {
            text.push_str(&format!("  tags: {}\n", fragment.tags.join(",")));
        }
    }
    if !any {
        text.push_str("- none\n");
    }
}

fn format_memory_graph_context(graph: &MemoryGraph) -> String {
    if graph.entities.is_empty() && graph.facts.is_empty() && graph.edges.is_empty() {
        return "- none\n".to_string();
    }

    let entity_names = graph
        .entities
        .iter()
        .map(|entity| (entity.id.as_str(), entity.name.as_str()))
        .collect::<HashMap<_, _>>();
    let mut text = String::new();

    if !graph.entities.is_empty() {
        text.push_str("ENTITIES\n");
        for entity in graph.entities.iter().take(12) {
            text.push_str(&format!("- [{}] {}\n", entity.entity_type, entity.name));
        }
    }

    if !graph.facts.is_empty() {
        text.push_str("FACTS\n");
        for fact in graph.facts.iter().take(12) {
            let subject = fact
                .entity_id
                .as_deref()
                .and_then(|id| entity_names.get(id).copied())
                .unwrap_or("memory");
            text.push_str(&format!(
                "- {} {} {} (memory {}, confidence {:.2})\n",
                subject,
                fact.predicate,
                fact.value,
                fact.memory_id.as_deref().unwrap_or("<none>"),
                fact.confidence
            ));
        }
    }

    if !graph.edges.is_empty() {
        text.push_str("RELATIONS\n");
        for edge in graph.edges.iter().take(16) {
            text.push_str(&format!(
                "- {} -{}-> {} (memory {}, confidence {:.2})\n",
                edge.from_entity_name,
                edge.relation_type,
                edge.to_entity_name,
                edge.memory_id.as_deref().unwrap_or("<none>"),
                edge.confidence
            ));
        }
    }

    text
}

#[derive(Debug)]
struct ScoredChunk {
    index: usize,
    score: f64,
    text: String,
}

fn scored_chunks(body: &str, terms: &HashSet<String>, max_chars: usize) -> Vec<ScoredChunk> {
    split_body_chunks(body, max_chars)
        .into_iter()
        .enumerate()
        .map(|(index, text)| {
            let score = chunk_score(&text, terms);
            ScoredChunk { index, score, text }
        })
        .collect()
}

fn is_core_context_memory(memory: &Memory) -> bool {
    memory.memory_tier == "core" || matches!(memory.kind.as_str(), "project_rule" | "constraint")
}

fn split_body_chunks(body: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    for paragraph in body.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        if paragraph.chars().count() <= max_chars {
            chunks.push(paragraph.to_string());
            continue;
        }
        let chars = paragraph.chars().collect::<Vec<_>>();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + max_chars).min(chars.len());
            chunks.push(chars[start..end].iter().collect::<String>());
            start = end;
        }
    }
    if chunks.is_empty() && !body.trim().is_empty() {
        chunks.push(truncate_text(body.trim(), max_chars));
    }
    chunks
}

fn chunk_score(text: &str, terms: &HashSet<String>) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let lower = text.to_ascii_lowercase();
    terms
        .iter()
        .map(|term| lower.matches(term).count() as f64)
        .sum()
}

fn query_terms(query: &str) -> HashSet<String> {
    query
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .filter(|term| term.chars().count() >= 3)
        .map(str::to_string)
        .collect()
}

fn truncate_text(text: &str, max_chars: usize) -> String {
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
    use crate::store::DEFAULT_MEMORY_TIER;

    fn memory(id: &str, tier: &str, body: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: "project".to_string(),
            scope: "project".to_string(),
            memory_tier: tier.to_string(),
            kind: "decision".to_string(),
            body: body.to_string(),
            tags: vec!["retrieval".to_string()],
            source: Some("test".to_string()),
            status: "active".to_string(),
            importance: 0.8,
            confidence: 0.9,
            superseded_by: None,
            status_reason: None,
            score: Some(0.42),
            quality_score: 0.0,
            usage_count: 0,
            last_used_at: None,
            contradiction_risk: 0.0,
            created_at: "now".to_string(),
            updated_at: "now".to_string(),
        }
    }

    #[test]
    fn task_fragments_select_relevant_chunk_without_full_body() {
        let body = "Billing code owns invoice retries and webhook reconciliation.\n\nUnrelated deployment notes mention dashboards and log rotation.";
        let memories = vec![memory("m1", DEFAULT_MEMORY_TIER, body)];
        let fragments = build_memory_fragments("invoice webhook", &memories, 1_000);

        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].text.contains("invoice retries"));
        assert!(!fragments[0].text.contains("log rotation"));
        assert_eq!(fragment_memory_ids(&fragments), vec!["m1".to_string()]);
    }

    #[test]
    fn selected_core_memory_is_labeled_as_task_relevant() {
        let memories = vec![memory(
            "core-1",
            "core",
            "All inventory tools must use dukememory_* names.",
        )];
        let fragments = build_memory_fragments("inventory query", &memories, 1_000);

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].section, "core");
        assert_eq!(
            fragments[0].reason,
            "task-relevant core/project-rule fragment"
        );
    }

    #[test]
    fn memory_summaries_omit_body_and_count_fragments() {
        let memories = vec![memory("m1", DEFAULT_MEMORY_TIER, "One\n\nTwo")];
        let fragments = build_memory_fragments("one two", &memories, 1_000);
        let summaries = context_memory_summaries(&memories, &fragments);

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, "m1");
        assert!(summaries[0].included_fragments >= 1);
    }

    #[test]
    fn core_memories_do_not_consume_task_memory_limit() {
        let merged = merge_core_and_task_memories(
            vec![memory("core-1", "core", "Core rule")],
            vec![
                memory("task-1", DEFAULT_MEMORY_TIER, "Task one"),
                memory("task-2", DEFAULT_MEMORY_TIER, "Task two"),
            ],
            2,
        );

        let ids = merged
            .iter()
            .map(|memory| memory.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["core-1", "task-1", "task-2"]);
    }
}

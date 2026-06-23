use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::ollama::OllamaClient;
use crate::safety::looks_like_sensitive_text;
use crate::store::{Memory, MemoryEntity, Store, normalize_project_type};

const MAX_PROMPT_CHARS: usize = 48_000;
const MAX_ENTITY_NAME_CHARS: usize = 120;
const MAX_DESCRIPTION_CHARS: usize = 500;
const MAX_FACT_VALUE_CHARS: usize = 700;
const MAX_RELATION_CHARS: usize = 80;
const ALLOWED_ENTITY_TYPES: &[&str] = &[
    "project",
    "component",
    "module",
    "tool",
    "command",
    "workflow",
    "rule",
    "concept",
    "external_service",
    "model",
    "file",
    "database",
    "schema",
    "index",
];
const ALLOWED_RELATIONS: &[&str] = &[
    "uses",
    "depends_on",
    "enforces",
    "replaces",
    "documents",
    "runs_before",
    "belongs_to",
    "configures",
    "stores",
    "indexes",
    "retrieves",
    "validates",
    "generates",
    "embeds",
    "extracts",
    "searches",
    "writes",
    "reads",
    "calls",
    "implements",
];

#[derive(Debug, Clone, Serialize)]
pub struct GraphExtractionProposal {
    pub memory_id: String,
    pub entities: Vec<GraphEntityCandidate>,
    pub facts: Vec<GraphFactCandidate>,
    pub edges: Vec<GraphEdgeCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEntityCandidate {
    pub name: String,
    pub entity_type: String,
    pub aliases: Vec<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphFactCandidate {
    pub entity_name: String,
    pub entity_type: String,
    pub predicate: String,
    pub value: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdgeCandidate {
    pub from_name: String,
    pub from_type: String,
    pub to_name: String,
    pub to_type: String,
    pub relation_type: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphExtractionReport {
    pub project_id: String,
    pub apply: bool,
    pub memories: usize,
    pub proposed_entities: usize,
    pub proposed_facts: usize,
    pub proposed_edges: usize,
    pub upserted_entities: usize,
    pub inserted_facts: usize,
    pub duplicate_facts: usize,
    pub inserted_edges: usize,
    pub duplicate_edges: usize,
    pub proposals: Vec<GraphExtractionProposal>,
}

#[derive(Debug, Deserialize)]
struct RawGraphResponse {
    items: Vec<RawGraphItem>,
}

#[derive(Debug, Deserialize)]
struct RawGraphItem {
    memory_id: Option<String>,
    entities: Option<Vec<RawGraphEntity>>,
    facts: Option<Vec<RawGraphFact>>,
    edges: Option<Vec<RawGraphEdge>>,
}

#[derive(Debug, Deserialize)]
struct RawGraphEntity {
    name: Option<String>,
    #[serde(alias = "type")]
    entity_type: Option<String>,
    aliases: Option<Vec<String>>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawGraphFact {
    entity_name: Option<String>,
    #[serde(alias = "entity_type")]
    entity_type: Option<String>,
    predicate: Option<String>,
    value: Option<String>,
    confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawGraphEdge {
    from_name: Option<String>,
    #[serde(alias = "from_entity_type")]
    from_type: Option<String>,
    to_name: Option<String>,
    #[serde(alias = "to_entity_type")]
    to_type: Option<String>,
    relation_type: Option<String>,
    confidence: Option<f64>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct EntityKey {
    entity_type: String,
    canonical_name: String,
}

pub async fn extract_memory_graph(
    ollama: &OllamaClient,
    project_id: &str,
    memories: &[Memory],
) -> Result<Vec<GraphExtractionProposal>> {
    if memories.is_empty() {
        return Ok(Vec::new());
    }
    let system = graph_extraction_system_prompt();
    let user = graph_extraction_user_prompt(project_id, memories);
    let raw = ollama.chat_json(system, &user).await?;
    let json = extract_json_object(&raw)?;
    normalize_graph_response(project_id, memories, json)
}

pub fn apply_graph_extraction(
    store: &Store,
    project_id: &str,
    proposals: Vec<GraphExtractionProposal>,
    apply: bool,
) -> Result<GraphExtractionReport> {
    let proposed_entities = proposals
        .iter()
        .map(|proposal| proposal.entities.len())
        .sum::<usize>();
    let proposed_facts = proposals
        .iter()
        .map(|proposal| proposal.facts.len())
        .sum::<usize>();
    let proposed_edges = proposals
        .iter()
        .map(|proposal| proposal.edges.len())
        .sum::<usize>();

    let mut report = GraphExtractionReport {
        project_id: project_id.to_string(),
        apply,
        memories: proposals.len(),
        proposed_entities,
        proposed_facts,
        proposed_edges,
        upserted_entities: 0,
        inserted_facts: 0,
        duplicate_facts: 0,
        inserted_edges: 0,
        duplicate_edges: 0,
        proposals,
    };

    if !apply {
        return Ok(report);
    }

    let mut entities = HashMap::<EntityKey, MemoryEntity>::new();
    for proposal in &report.proposals {
        for entity in &proposal.entities {
            ensure_entity(store, project_id, &mut entities, entity)?;
            report.upserted_entities += 1;
        }
    }

    for proposal in &report.proposals {
        for fact in &proposal.facts {
            let entity = ensure_fact_entity(store, project_id, &mut entities, fact)?;
            let (_fact, inserted) = store.add_memory_fact_deduplicated(
                project_id,
                Some(&entity.id),
                Some(&proposal.memory_id),
                &fact.predicate,
                &fact.value,
                fact.confidence,
            )?;
            if inserted {
                report.inserted_facts += 1;
            } else {
                report.duplicate_facts += 1;
            }
        }

        for edge in &proposal.edges {
            let from = ensure_edge_entity(
                store,
                project_id,
                &mut entities,
                &edge.from_name,
                &edge.from_type,
            )?;
            let to = ensure_edge_entity(
                store,
                project_id,
                &mut entities,
                &edge.to_name,
                &edge.to_type,
            )?;
            if from.id == to.id {
                continue;
            }
            let (_edge, inserted) = store.add_memory_edge_deduplicated(
                project_id,
                &from.id,
                &to.id,
                &edge.relation_type,
                Some(&proposal.memory_id),
                edge.confidence,
            )?;
            if inserted {
                report.inserted_edges += 1;
            } else {
                report.duplicate_edges += 1;
            }
        }
    }

    Ok(report)
}

fn ensure_entity(
    store: &Store,
    project_id: &str,
    entities: &mut HashMap<EntityKey, MemoryEntity>,
    candidate: &GraphEntityCandidate,
) -> Result<MemoryEntity> {
    let key = entity_key(&candidate.entity_type, &candidate.name);
    if let Some(entity) = entities.get(&key) {
        return Ok(entity.clone());
    }
    let entity = store.upsert_memory_entity(
        project_id,
        &candidate.entity_type,
        &candidate.name,
        candidate.aliases.clone(),
        candidate.description.clone(),
    )?;
    entities.insert(key, entity.clone());
    Ok(entity)
}

fn ensure_fact_entity(
    store: &Store,
    project_id: &str,
    entities: &mut HashMap<EntityKey, MemoryEntity>,
    fact: &GraphFactCandidate,
) -> Result<MemoryEntity> {
    let candidate = GraphEntityCandidate {
        name: fact.entity_name.clone(),
        entity_type: fact.entity_type.clone(),
        aliases: Vec::new(),
        description: None,
    };
    ensure_entity(store, project_id, entities, &candidate)
}

fn ensure_edge_entity(
    store: &Store,
    project_id: &str,
    entities: &mut HashMap<EntityKey, MemoryEntity>,
    name: &str,
    entity_type: &str,
) -> Result<MemoryEntity> {
    let candidate = GraphEntityCandidate {
        name: name.to_string(),
        entity_type: entity_type.to_string(),
        aliases: Vec::new(),
        description: None,
    };
    ensure_entity(store, project_id, entities, &candidate)
}

fn normalize_graph_response(
    _project_id: &str,
    memories: &[Memory],
    json: &str,
) -> Result<Vec<GraphExtractionProposal>> {
    let response = serde_json::from_str::<RawGraphResponse>(json)
        .with_context(|| format!("failed to parse graph extraction JSON: {json}"))?;
    let valid_ids = memories
        .iter()
        .map(|memory| memory.id.as_str())
        .collect::<HashSet<_>>();
    let mut proposals = Vec::new();
    let mut seen_item_ids = HashSet::new();
    let mut global_entities = HashSet::new();
    let mut global_entity_names = HashSet::new();
    let mut global_facts = HashSet::new();
    let mut global_edges = HashSet::new();

    for item in response.items {
        let Some(memory_id) = item.memory_id.map(|value| value.trim().to_string()) else {
            continue;
        };
        if !valid_ids.contains(memory_id.as_str()) || !seen_item_ids.insert(memory_id.clone()) {
            continue;
        }

        let mut entities = Vec::new();
        let mut seen_entities = HashSet::new();
        for raw in item.entities.unwrap_or_default().into_iter().take(12) {
            if let Some(entity) = normalize_entity(raw) {
                let key = entity_key(&entity.entity_type, &entity.name);
                let name_key = canonical_name(&entity.name);
                if seen_entities.insert(key.clone())
                    && global_entities.insert(key)
                    && global_entity_names.insert(name_key)
                {
                    entities.push(entity);
                }
            }
        }

        let mut facts = Vec::new();
        let mut seen_facts = HashSet::new();
        for raw in item.facts.unwrap_or_default().into_iter().take(20) {
            if let Some(fact) = normalize_fact(raw) {
                let key = format!(
                    "{}\n{}\n{}\n{}",
                    fact.entity_type,
                    canonical_name(&fact.entity_name),
                    fact.predicate,
                    fact.value
                );
                if seen_facts.insert(key.clone()) && global_facts.insert(key) {
                    facts.push(fact);
                }
            }
        }

        let mut edges = Vec::new();
        let mut seen_edges = HashSet::new();
        for raw in item.edges.unwrap_or_default().into_iter().take(20) {
            if let Some(edge) = normalize_edge(raw) {
                let key = format!(
                    "{}\n{}\n{}\n{}\n{}",
                    edge.from_type,
                    canonical_name(&edge.from_name),
                    edge.relation_type,
                    edge.to_type,
                    canonical_name(&edge.to_name)
                );
                if seen_edges.insert(key.clone()) && global_edges.insert(key) {
                    edges.push(edge);
                }
            }
        }

        if !entities.is_empty() || !facts.is_empty() || !edges.is_empty() {
            proposals.push(GraphExtractionProposal {
                memory_id,
                entities,
                facts,
                edges,
            });
        }
    }

    Ok(proposals)
}

fn normalize_entity(raw: RawGraphEntity) -> Option<GraphEntityCandidate> {
    let name = clean_name(raw.name?)?;
    let entity_type = normalize_graph_type(raw.entity_type.as_deref().unwrap_or("concept"));
    let description = raw
        .description
        .and_then(|value| clean_text(&value, MAX_DESCRIPTION_CHARS));
    let aliases = raw
        .aliases
        .unwrap_or_default()
        .into_iter()
        .filter_map(clean_name)
        .filter(|alias| canonical_name(alias) != canonical_name(&name))
        .take(8)
        .collect::<Vec<_>>();
    Some(GraphEntityCandidate {
        name,
        entity_type,
        aliases,
        description,
    })
}

fn normalize_fact(raw: RawGraphFact) -> Option<GraphFactCandidate> {
    let entity_name = clean_name(raw.entity_name?)?;
    let entity_type = normalize_graph_type(raw.entity_type.as_deref().unwrap_or("concept"));
    let predicate = normalize_relation(raw.predicate?)?;
    let value = clean_text(&raw.value?, MAX_FACT_VALUE_CHARS)?;
    Some(GraphFactCandidate {
        entity_name,
        entity_type,
        predicate,
        value,
        confidence: clamp_score(raw.confidence.unwrap_or(0.7)),
    })
}

fn normalize_edge(raw: RawGraphEdge) -> Option<GraphEdgeCandidate> {
    let from_name = clean_name(raw.from_name?)?;
    let from_type = normalize_graph_type(raw.from_type.as_deref().unwrap_or("concept"));
    let to_name = clean_name(raw.to_name?)?;
    let to_type = normalize_graph_type(raw.to_type.as_deref().unwrap_or("concept"));
    let relation_type = normalize_relation(raw.relation_type?)?;
    if from_type == to_type && canonical_name(&from_name) == canonical_name(&to_name) {
        return None;
    }
    Some(GraphEdgeCandidate {
        from_name,
        from_type,
        to_name,
        to_type,
        relation_type,
        confidence: clamp_score(raw.confidence.unwrap_or(0.7)),
    })
}

fn graph_extraction_system_prompt() -> &'static str {
    "You build a durable project memory knowledge graph for a local coding agent. \
     Return only JSON. Extract stable entities, facts, and relationships that help \
     navigate a project weeks later. Do not extract secrets, credentials, access \
     tokens, private personal data, temporary status, guesses, or one-off chatter. \
     Prefer concise entity names. Use stable snake_case relation and predicate names. \
     Use only these entity_type values: project, component, module, tool, command, \
     workflow, rule, concept, external_service, model, file, database, schema, index. \
     Use only these relation and predicate values: uses, depends_on, enforces, \
     replaces, documents, runs_before, belongs_to, configures, stores, indexes, \
     retrieves, validates, generates, embeds, extracts, searches, writes, reads, \
     calls, implements."
}

fn graph_extraction_user_prompt(project_id: &str, memories: &[Memory]) -> String {
    let mut text = format!(
        "Project id: {project_id}\n\n\
         Return JSON exactly in this shape:\n\
         {{\"items\":[{{\"memory_id\":\"...\",\"entities\":[{{\"name\":\"Dukememory\",\"entity_type\":\"project\",\"aliases\":[],\"description\":\"...\"}}],\"facts\":[{{\"entity_name\":\"Dukememory\",\"entity_type\":\"project\",\"predicate\":\"uses\",\"value\":\"PostgreSQL\",\"confidence\":0.8}}],\"edges\":[{{\"from_name\":\"Dukememory\",\"from_type\":\"project\",\"to_name\":\"PostgreSQL\",\"to_type\":\"tool\",\"relation_type\":\"uses\",\"confidence\":0.8}}]}}]}}\n\n\
         Rules:\n\
         - Use only memory_id values from the input.\n\
         - Return at most 8 entities, 10 facts, and 10 edges per memory.\n\
         - Use only allowed entity_type and relation/predicate values from the system message.\n\
         - Facts should be short subject-predicate-value statements backed by that memory.\n\
         - Edges should show meaningful dependencies, ownership, usage, replacement, workflow order, or constraints.\n\
         - If a memory has no durable graph content, omit it.\n\n\
         Memories:\n"
    );

    for memory in memories {
        text.push_str(&format!(
            "\n[{}]\nkind: {}\ntags: {}\nsource: {}\nbody:\n{}\n",
            memory.id,
            memory.kind,
            memory.tags.join(","),
            memory.source.clone().unwrap_or_default(),
            truncate_text(&memory.body, 2_500)
        ));
    }

    truncate_text(&text, MAX_PROMPT_CHARS)
}

fn clean_name(value: String) -> Option<String> {
    clean_text(&value, MAX_ENTITY_NAME_CHARS)
}

fn clean_text(value: &str, max_chars: usize) -> Option<String> {
    let value = collapse_whitespace(value);
    if value.is_empty() || looks_like_sensitive_text(&value) {
        return None;
    }
    Some(truncate_text(&value, max_chars))
}

fn normalize_graph_type(value: &str) -> String {
    let normalized = normalize_project_type(value);
    if ALLOWED_ENTITY_TYPES.contains(&normalized.as_str()) {
        normalized
    } else {
        "concept".to_string()
    }
}

fn normalize_relation(value: String) -> Option<String> {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    let normalized = truncate_text(&normalized, MAX_RELATION_CHARS);
    if normalized.is_empty() || looks_like_sensitive_text(&normalized) {
        return None;
    }
    let normalized = relation_alias(&normalized);
    if ALLOWED_RELATIONS.contains(&normalized.as_str()) {
        Some(normalized)
    } else {
        None
    }
}

fn relation_alias(value: &str) -> String {
    match value {
        "use" | "used_by" | "uses_model" | "uses_tool" => "uses".to_string(),
        "depend" | "depends" | "dependency" | "requires" | "required_by" => {
            "depends_on".to_string()
        }
        "enforce" | "requires_rule" => "enforces".to_string(),
        "replace" | "supersedes" | "superseded_by" => "replaces".to_string(),
        "doc" | "docs" | "documented_by" => "documents".to_string(),
        "before" | "precedes" | "runs_after" => "runs_before".to_string(),
        "belong_to" | "part_of" | "owned_by" => "belongs_to".to_string(),
        "configure" => "configures".to_string(),
        "store" | "persists" | "persists_to" => "stores".to_string(),
        "index" => "indexes".to_string(),
        "retrieve" | "loads" | "loads_from" => "retrieves".to_string(),
        "validate" => "validates".to_string(),
        "generate" => "generates".to_string(),
        "embed" => "embeds".to_string(),
        "extract" => "extracts".to_string(),
        "search" => "searches".to_string(),
        "write" => "writes".to_string(),
        "read" => "reads".to_string(),
        "call" => "calls".to_string(),
        "implement" => "implements".to_string(),
        other => other.to_string(),
    }
}

fn entity_key(entity_type: &str, name: &str) -> EntityKey {
    EntityKey {
        entity_type: normalize_graph_type(entity_type),
        canonical_name: canonical_name(name),
    }
}

fn canonical_name(value: &str) -> String {
    collapse_whitespace(value).to_ascii_lowercase()
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clamp_score(value: f64) -> f64 {
    if value.is_nan() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
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

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(id: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: "project".to_string(),
            scope: "project".to_string(),
            memory_tier: crate::store::DEFAULT_MEMORY_TIER.to_string(),
            kind: "decision".to_string(),
            body: "Dukememory uses PostgreSQL and Ollama.".to_string(),
            tags: Vec::new(),
            source: None,
            status: "active".to_string(),
            importance: 0.7,
            confidence: 0.8,
            superseded_by: None,
            status_reason: None,
            score: None,
            quality_score: 0.0,
            usage_count: 0,
            last_used_at: None,
            contradiction_risk: 0.0,
            created_at: "now".to_string(),
            updated_at: "now".to_string(),
        }
    }

    #[test]
    fn normalizes_graph_response_and_drops_unknown_memories() -> Result<()> {
        let memories = vec![memory("019eef06-ee5f-72c2-b056-3c4686924e0d")];
        let json = r#"{
            "items": [
                {
                    "memory_id": "019eef06-ee5f-72c2-b056-3c4686924e0d",
                    "entities": [{"name": " Dukememory ", "entity_type": "Project"}],
                    "facts": [{"entity_name": "Dukememory", "predicate": "Uses", "value": "PostgreSQL", "confidence": 2.0}],
                    "edges": [{"from_name": "Dukememory", "to_name": "Ollama", "relation_type": "uses model", "confidence": 0.8}]
                },
                {"memory_id": "unknown", "entities": [{"name": "Bad"}]}
            ]
        }"#;

        let proposals = normalize_graph_response("project", &memories, json)?;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].entities[0].entity_type, "project");
        assert_eq!(proposals[0].facts[0].predicate, "uses");
        assert_eq!(proposals[0].facts[0].confidence, 1.0);
        assert_eq!(proposals[0].edges[0].relation_type, "uses");
        Ok(())
    }

    #[test]
    fn graph_response_deduplicates_globally_and_drops_unknown_relations() -> Result<()> {
        let memories = vec![
            memory("019eef06-ee5f-72c2-b056-3c4686924e0d"),
            memory("019eef06-eba2-7fb0-b43c-b8d981a7478c"),
        ];
        let json = r#"{
            "items": [
                {
                    "memory_id": "019eef06-ee5f-72c2-b056-3c4686924e0d",
                    "entities": [{"name": "Dukememory", "entity_type": "Project"}],
                    "facts": [{"entity_name": "Dukememory", "predicate": "stores", "value": "PostgreSQL", "confidence": 0.8}],
                    "edges": [{"from_name": "Dukememory", "to_name": "PostgreSQL", "relation_type": "stores", "confidence": 0.8}]
                },
                {
                    "memory_id": "019eef06-eba2-7fb0-b43c-b8d981a7478c",
                    "entities": [{"name": "Dukememory", "entity_type": "UnknownThing"}],
                    "facts": [{"entity_name": "Dukememory", "predicate": "vaguely_mentions", "value": "PostgreSQL", "confidence": 0.8}],
                    "edges": [{"from_name": "Dukememory", "to_name": "PostgreSQL", "relation_type": "vaguely mentions", "confidence": 0.8}]
                }
            ]
        }"#;

        let proposals = normalize_graph_response("project", &memories, json)?;
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].entities.len(), 1);
        assert_eq!(proposals[0].facts.len(), 1);
        assert_eq!(proposals[0].edges.len(), 1);
        Ok(())
    }
}

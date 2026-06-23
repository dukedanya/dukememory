use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::ollama::OllamaClient;
use crate::store::Memory;

const MAX_COMPACTION_INPUT_CHARS: usize = 36_000;
const MAX_MEMORY_BODY_CHARS: usize = 1_200;
const MAX_SUMMARY_CHARS: usize = 4_000;

#[derive(Debug, Clone, Serialize)]
pub struct CompactionProposal {
    pub project_id: String,
    pub model: String,
    pub source_ids: Vec<String>,
    pub summary_kind: String,
    pub summary_body: String,
    pub tags: Vec<String>,
    pub importance: f64,
    pub confidence: f64,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
struct RawCompactionResponse {
    summary: Option<String>,
    tags: Option<Vec<String>>,
    importance: Option<f64>,
    confidence: Option<f64>,
    reason: Option<String>,
}

pub async fn propose_compaction(
    ollama: &OllamaClient,
    model: &str,
    project_id: &str,
    memories: &[Memory],
) -> Result<CompactionProposal> {
    if memories.len() < 2 {
        bail!("compaction requires at least 2 memories");
    }
    let system = compaction_system_prompt();
    let user = compaction_user_prompt(project_id, memories);
    let raw = ollama.chat_json_with_model(model, system, &user).await?;
    let json = extract_json_object(&raw)?;
    let response = serde_json::from_str::<RawCompactionResponse>(json)
        .with_context(|| format!("failed to parse compaction JSON: {json}"))?;
    normalize_response(model, project_id, memories, response)
}

fn normalize_response(
    model: &str,
    project_id: &str,
    memories: &[Memory],
    response: RawCompactionResponse,
) -> Result<CompactionProposal> {
    let summary_body = truncate(
        response.summary.unwrap_or_default().trim(),
        MAX_SUMMARY_CHARS,
    );
    if summary_body.len() < 48 {
        bail!("compaction model returned an empty or too-short summary");
    }
    let mut tags = response
        .tags
        .unwrap_or_default()
        .into_iter()
        .map(|tag| tag.trim().to_ascii_lowercase().replace('-', "_"))
        .filter(|tag| !tag.is_empty())
        .take(12)
        .collect::<Vec<_>>();
    tags.push("compacted".to_string());
    tags.sort();
    tags.dedup();
    let source_ids = memories
        .iter()
        .map(|memory| memory.id.clone())
        .collect::<Vec<_>>();
    Ok(CompactionProposal {
        project_id: project_id.to_string(),
        model: model.to_string(),
        source_ids,
        summary_kind: "project_summary".to_string(),
        summary_body,
        tags,
        importance: clamp_score(response.importance.unwrap_or(0.75)),
        confidence: clamp_score(response.confidence.unwrap_or(0.75)),
        reason: truncate(
            response
                .reason
                .unwrap_or_else(|| "Compacted older active memories into one summary.".to_string())
                .trim(),
            500,
        ),
    })
}

fn compaction_system_prompt() -> &'static str {
    "You compact durable project memories for a local coding agent. Return only JSON. \
     Preserve stable decisions, project rules, recurring pitfalls, codebase lessons, asset pipeline facts, and game design facts. \
     Remove duplicates, temporary task status, weak guesses, and wording noise. \
     Do not invent facts. Do not include secrets, credentials, private keys, access tokens, or private personal data. \
     The summary must remain useful weeks later and must not mention that it came from a chat transcript."
}

fn compaction_user_prompt(project_id: &str, memories: &[Memory]) -> String {
    let mut text = format!(
        "Project id: {project_id}\n\
         Source memories: {}\n\n\
         Return JSON exactly in this shape:\n\
         {{\"summary\":\"durable compacted memory text\",\"tags\":[\"rust\"],\"importance\":0.8,\"confidence\":0.8,\"reason\":\"why this summary preserves the useful facts\"}}\n\n\
         Memories:\n",
        memories.len()
    );
    for memory in memories {
        text.push_str(&format!(
            "\n---\nid: {}\nkind: {}\nimportance: {:.2}\nconfidence: {:.2}\ntags: {}\nbody:\n{}\n",
            memory.id,
            memory.kind,
            memory.importance,
            memory.confidence,
            memory.tags.join(","),
            truncate(&memory.body, MAX_MEMORY_BODY_CHARS)
        ));
        if text.chars().count() > MAX_COMPACTION_INPUT_CHARS {
            text = truncate(&text, MAX_COMPACTION_INPUT_CHARS);
            break;
        }
    }
    text
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

fn clamp_score(value: f64) -> f64 {
    if value.is_nan() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
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

    #[test]
    fn normalize_response_builds_project_summary() -> Result<()> {
        let memories = vec![Memory {
            id: "m1".to_string(),
            project_id: "game-a".to_string(),
            scope: "project".to_string(),
            memory_tier: crate::store::DEFAULT_MEMORY_TIER.to_string(),
            kind: "decision".to_string(),
            body: "Use deterministic fixed-step simulation for combat.".to_string(),
            tags: vec!["combat".to_string()],
            source: None,
            status: "active".to_string(),
            importance: 0.8,
            confidence: 0.9,
            superseded_by: None,
            status_reason: None,
            score: None,
            quality_score: 0.0,
            usage_count: 0,
            last_used_at: None,
            contradiction_risk: 0.0,
            created_at: "2026-01-01 00:00:00".to_string(),
            updated_at: "2026-01-01 00:00:00".to_string(),
        }];
        let response = RawCompactionResponse {
            summary: Some(
                "Combat systems should use deterministic fixed-step simulation so replay and tests remain stable."
                    .to_string(),
            ),
            tags: Some(vec!["Combat".to_string(), "rust-code".to_string()]),
            importance: Some(1.5),
            confidence: Some(0.8),
            reason: Some("Preserves the durable combat simulation rule.".to_string()),
        };
        let proposal = normalize_response("qwen3:14b", "game-a", &memories, response)?;
        assert_eq!(proposal.summary_kind, "project_summary");
        assert_eq!(proposal.importance, 1.0);
        assert!(proposal.tags.contains(&"compacted".to_string()));
        assert!(proposal.tags.contains(&"rust_code".to_string()));
        Ok(())
    }

    #[test]
    fn extract_json_object_accepts_wrapped_response() -> Result<()> {
        let json = extract_json_object("Result:\n{\"summary\":\"ok\"}\nDone")?;
        assert_eq!(json, "{\"summary\":\"ok\"}");
        Ok(())
    }
}

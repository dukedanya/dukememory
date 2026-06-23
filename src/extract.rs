use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ollama::OllamaClient;
use crate::safety::looks_like_sensitive_text;

const MAX_INPUT_CHARS: usize = 48_000;
const MAX_BODY_CHARS: usize = 2_000;
const ALLOWED_KINDS: &[&str] = &[
    "decision",
    "project_rule",
    "constraint",
    "architecture",
    "game_design",
    "code_fact",
    "bug_regression",
    "workflow",
    "setup",
    "external_service",
    "asset_pipeline",
    "user_preference",
    "note",
];

#[derive(Debug, Clone, Serialize)]
pub struct MemoryCandidate {
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub importance: f64,
    pub confidence: f64,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PreparedExtractionInput {
    pub source: String,
    pub project_path: Option<String>,
    pub text: String,
}

#[derive(Debug, Deserialize)]
struct ExtractResponse {
    memories: Vec<RawMemoryCandidate>,
}

#[derive(Debug, Deserialize)]
struct RawMemoryCandidate {
    kind: Option<String>,
    body: Option<String>,
    tags: Option<Vec<String>>,
    importance: Option<f64>,
    confidence: Option<f64>,
    reason: Option<String>,
}

pub async fn extract_memory_candidates(
    ollama: &OllamaClient,
    project_id: &str,
    source: &str,
    input: &str,
    max_candidates: usize,
) -> Result<Vec<MemoryCandidate>> {
    let max_candidates = max_candidates.clamp(1, 20);
    let system = extraction_system_prompt();
    let user = extraction_user_prompt(project_id, source, input, max_candidates);
    let raw = ollama.chat_json(system, &user).await?;
    let json = extract_json_object(&raw)?;
    let response = serde_json::from_str::<ExtractResponse>(json)
        .with_context(|| format!("failed to parse extraction JSON: {json}"))?;

    let mut candidates = Vec::new();
    for raw in response.memories.into_iter().take(max_candidates) {
        if let Some(candidate) = normalize_candidate(raw) {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

pub fn prepare_extraction_input(input: &str, default_source: &str) -> PreparedExtractionInput {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return PreparedExtractionInput {
            source: default_source.to_string(),
            project_path: None,
            text: String::new(),
        };
    }

    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        return PreparedExtractionInput {
            source: default_source.to_string(),
            project_path: None,
            text: input.to_string(),
        };
    };

    let source = find_first_string(
        &value,
        &[
            "source",
            "event",
            "hook_event",
            "hook_event_name",
            "hook",
            "type",
            "name",
        ],
    )
    .map(|value| append_source_suffix(default_source, &value))
    .unwrap_or_else(|| default_source.to_string());
    let project_path = find_first_string(
        &value,
        &[
            "project_path",
            "workspace_root",
            "workspace",
            "repo_root",
            "repository_root",
            "cwd",
            "working_directory",
            "current_dir",
        ],
    );

    let mut parts = Vec::new();
    collect_named_text_fields(
        &value,
        &mut parts,
        &[
            "summary",
            "transcript",
            "conversation",
            "messages",
            "input_messages",
            "output",
            "history",
            "turns",
            "prompt",
            "user_prompt",
            "last_user_message",
            "response",
            "result",
            "text",
            "input",
        ],
    );

    let text = if parts.is_empty() {
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| input.to_string())
    } else {
        parts.join("\n\n")
    };

    PreparedExtractionInput {
        source,
        project_path,
        text,
    }
}

fn extraction_system_prompt() -> &'static str {
    "You extract durable project memory for a local coding agent. Return only JSON. \
     Extract only stable facts, user preferences, project rules, constraints, \
     architecture decisions, workflows, setup facts, external service facts, \
     codebase lessons, verified bug regressions, asset pipeline facts, and domain facts. \
     Use kind bug_regression only for known wrong behavior with an observed symptom and durable trigger or context; \
     Prefer generic memory kinds and use tags for domains like game, webapp, research, or ops. \
     Reject secrets, credentials, API keys, private keys, access tokens, temporary chatter, \
     raw logs, guesses, one-off task status, 'continue/do it' chatter, and facts from other projects. \
     Every extracted memory must be useful weeks later."
}

fn append_source_suffix(source: &str, suffix: &str) -> String {
    let suffix = suffix.trim();
    if suffix.is_empty() || source.rsplit(':').next() == Some(suffix) {
        source.to_string()
    } else {
        format!("{source}:{suffix}")
    }
}

fn extraction_user_prompt(
    project_id: &str,
    source: &str,
    input: &str,
    max_candidates: usize,
) -> String {
    format!(
        "Project id: {project_id}\n\
         Source: {source}\n\
         Max memories: {max_candidates}\n\n\
         Allowed kinds: decision, project_rule, constraint, architecture, game_design, code_fact, bug_regression, workflow, setup, external_service, asset_pipeline, user_preference, note.\n\
         For bug_regression, body must state the symptom, trigger/context, and expected safe behavior; keep uncertain bugs out.\n\n\
         Return JSON exactly in this shape:\n\
         {{\"memories\":[{{\"kind\":\"decision\",\"body\":\"...\",\"tags\":[\"rust\"],\"importance\":0.7,\"confidence\":0.8,\"reason\":\"why this is durable\"}}]}}\n\n\
         If there is nothing durable, return {{\"memories\":[]}}.\n\n\
         Text:\n{}",
        truncate(input, MAX_INPUT_CHARS)
    )
}

fn normalize_candidate(raw: RawMemoryCandidate) -> Option<MemoryCandidate> {
    let body = truncate(raw.body?.trim(), MAX_BODY_CHARS);
    if body.is_empty() || looks_like_sensitive_text(&body) || looks_transient(&body) {
        return None;
    }

    let kind = raw
        .kind
        .as_deref()
        .map(normalize_kind)
        .unwrap_or_else(|| "note".to_string());
    let tags = raw
        .tags
        .unwrap_or_default()
        .into_iter()
        .map(|tag| tag.trim().to_ascii_lowercase())
        .filter(|tag| !tag.is_empty() && !looks_like_sensitive_text(tag))
        .take(12)
        .collect::<Vec<_>>();

    Some(MemoryCandidate {
        kind,
        body,
        tags,
        importance: clamp_score(raw.importance.unwrap_or(0.5)),
        confidence: clamp_score(raw.confidence.unwrap_or(0.6)),
        reason: raw.reason.map(|reason| truncate(reason.trim(), 500)),
    })
}

fn normalize_kind(kind: &str) -> String {
    let kind = kind.trim().to_ascii_lowercase().replace('-', "_");
    if ALLOWED_KINDS.contains(&kind.as_str()) {
        kind
    } else {
        "note".to_string()
    }
}

fn clamp_score(value: f64) -> f64 {
    if value.is_nan() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
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

fn looks_transient(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    lower.len() < 24
        || matches!(
            lower.as_str(),
            "continue" | "продолжай" | "делай" | "давай" | "ok" | "ок"
        )
        || lower.contains("we are currently")
        || lower.contains("next i will")
        || lower.contains("сейчас я")
}

fn find_first_string(value: &Value, keys: &[&str]) -> Option<String> {
    match value {
        Value::Object(map) => {
            for key in keys {
                if let Some(text) = map
                    .iter()
                    .find(|(candidate, _)| key_matches(candidate, key))
                    .and_then(|(_, value)| value.as_str())
                {
                    let text = text.trim();
                    if !text.is_empty() {
                        return Some(text.to_string());
                    }
                }
            }
            for child in map.values() {
                if let Some(found) = find_first_string(child, keys) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(values) => values
            .iter()
            .find_map(|child| find_first_string(child, keys)),
        _ => None,
    }
}

fn collect_named_text_fields(value: &Value, parts: &mut Vec<String>, keys: &[&str]) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if keys.iter().any(|candidate| key_matches(key, candidate)) {
                    collect_text_value(key, child, parts);
                } else {
                    collect_named_text_fields(child, parts, keys);
                }
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_named_text_fields(child, parts, keys);
            }
        }
        _ => {}
    }
}

fn collect_text_value(label: &str, value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(format!("{label}: {text}"));
            }
        }
        Value::Array(values) => {
            for item in values {
                collect_text_value(label, item, parts);
            }
        }
        Value::Object(map) => {
            let role = map
                .get("role")
                .or_else(|| map.get("author"))
                .or_else(|| map.get("type"))
                .and_then(Value::as_str)
                .unwrap_or(label);
            if let Some(content) = map.get("content") {
                collect_content_parts(role, content, parts);
            } else {
                collect_named_text_fields(
                    value,
                    parts,
                    &[
                        "text",
                        "summary",
                        "transcript",
                        "prompt",
                        "message",
                        "input_text",
                        "output_text",
                        "final_response",
                        "assistant_response",
                    ],
                );
            }
        }
        _ => {}
    }
}

fn collect_content_parts(label: &str, value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(format!("{label}: {text}"));
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_content_parts(label, value, parts);
            }
        }
        Value::Object(map) => {
            let part_label = map.get("type").and_then(Value::as_str).unwrap_or(label);
            if let Some(text) = map
                .get("text")
                .or_else(|| map.get("input_text"))
                .or_else(|| map.get("output_text"))
                .or_else(|| map.get("value"))
                .and_then(Value::as_str)
            {
                let text = text.trim();
                if !text.is_empty() {
                    parts.push(format!("{label}/{part_label}: {text}"));
                }
            } else if let Some(content) = map.get("content").or_else(|| map.get("parts")) {
                collect_content_parts(label, content, parts);
            } else {
                collect_named_text_fields(
                    value,
                    parts,
                    &["text", "input_text", "output_text", "summary", "message"],
                );
            }
        }
        _ => {}
    }
}

fn key_matches(actual: &str, expected: &str) -> bool {
    canonical_key(actual) == canonical_key(expected)
}

fn canonical_key(key: &str) -> String {
    key.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
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
    fn normalize_candidate_rejects_secrets() {
        let candidate = normalize_candidate(RawMemoryCandidate {
            kind: Some("decision".to_string()),
            body: Some("API_KEY=sk-test1234567890abcdef should be stored in config.".to_string()),
            tags: Some(vec!["security".to_string()]),
            importance: Some(0.9),
            confidence: Some(0.9),
            reason: None,
        });
        assert!(candidate.is_none());
    }

    #[test]
    fn normalize_candidate_defaults_unknown_kind_and_clamps_scores() {
        let candidate = normalize_candidate(RawMemoryCandidate {
            kind: Some("random-kind".to_string()),
            body: Some("Use project-scoped memory for all retrieval.".to_string()),
            tags: Some(vec![" Memory ".to_string(), "".to_string()]),
            importance: Some(10.0),
            confidence: Some(-5.0),
            reason: Some("Stable project rule.".to_string()),
        })
        .expect("candidate should be accepted");

        assert_eq!(candidate.kind, "note");
        assert_eq!(candidate.tags, vec!["memory"]);
        assert_eq!(candidate.importance, 1.0);
        assert_eq!(candidate.confidence, 0.0);
    }

    #[test]
    fn extracts_json_object_from_wrapped_model_text() -> Result<()> {
        let json = extract_json_object("Here:\n{\"memories\":[]}\nDone")?;
        assert_eq!(json, "{\"memories\":[]}");
        Ok(())
    }

    #[test]
    fn prepares_json_hook_payload_with_project_path_and_messages() {
        let prepared = prepare_extraction_input(
            r#"{
              "event": "Stop",
              "cwd": "/tmp/game",
              "messages": [
                {"role": "user", "content": "Use RON for item data."},
                {"role": "assistant", "content": "Implemented parser."}
              ]
            }"#,
            "dukememory_hook",
        );

        assert_eq!(prepared.source, "dukememory_hook:Stop");
        assert_eq!(prepared.project_path.as_deref(), Some("/tmp/game"));
        assert!(prepared.text.contains("Use RON for item data."));
        assert!(prepared.text.contains("Implemented parser."));
    }

    #[test]
    fn prepare_json_hook_payload_does_not_duplicate_source_event_suffix() {
        let prepared = prepare_extraction_input(
            r#"{
              "event": "Stop",
              "cwd": "/tmp/game",
              "summary": "Use RON for item data."
            }"#,
            "dukememory_hook:Stop",
        );

        assert_eq!(prepared.source, "dukememory_hook:Stop");
        assert_eq!(prepared.project_path.as_deref(), Some("/tmp/game"));
        assert!(prepared.text.contains("Use RON for item data."));
    }

    #[test]
    fn prepares_codex_payload_with_nested_content_parts_and_camel_case_metadata() {
        let prepared = prepare_extraction_input(
            r#"{
              "hookEventName": "PreCompact",
              "metadata": {
                "workspaceRoot": "/tmp/duke-game"
              },
              "inputMessages": [
                {
                  "role": "user",
                  "content": [
                    {"type": "input_text", "text": "Use ECS for combat systems."}
                  ]
                },
                {
                  "role": "assistant",
                  "content": [
                    {"type": "output_text", "text": "Added Bevy schedule notes."}
                  ]
                }
              ],
              "finalResponse": {
                "content": [
                  {"type": "text", "text": "Remember that combat systems run after movement."}
                ]
              }
            }"#,
            "dukememory_hook",
        );

        assert_eq!(prepared.source, "dukememory_hook:PreCompact");
        assert_eq!(prepared.project_path.as_deref(), Some("/tmp/duke-game"));
        assert!(prepared.text.contains("Use ECS for combat systems."));
        assert!(prepared.text.contains("Added Bevy schedule notes."));
        assert!(
            prepared
                .text
                .contains("Remember that combat systems run after movement.")
        );
    }
}

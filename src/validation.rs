use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::ollama::OllamaClient;
use crate::store::Memory;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum ValidationAction {
    Promote,
    Archive,
    Keep,
}

impl ValidationAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Promote => "promote",
            Self::Archive => "archive",
            Self::Keep => "keep",
        }
    }

    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "promote" | "active" | "approve" | "accept" => Self::Promote,
            "archive" | "reject" | "delete" | "discard" => Self::Archive,
            _ => Self::Keep,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationDecision {
    pub id: String,
    pub action: ValidationAction,
    pub confidence: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationReport {
    pub model: String,
    pub decisions: Vec<ValidationDecision>,
}

#[derive(Debug, Deserialize)]
struct RawValidationResponse {
    decisions: Vec<RawValidationDecision>,
}

#[derive(Debug, Deserialize)]
struct RawValidationDecision {
    id: Option<String>,
    action: Option<String>,
    confidence: Option<f64>,
    reason: Option<String>,
}

pub async fn validate_memories(
    ollama: &OllamaClient,
    model: &str,
    project_id: &str,
    memories: &[Memory],
) -> Result<ValidationReport> {
    if memories.is_empty() {
        return Ok(ValidationReport {
            model: model.to_string(),
            decisions: Vec::new(),
        });
    }

    let system = validation_system_prompt();
    let user = validation_user_prompt(project_id, memories);
    let raw = ollama.chat_json_with_model(model, system, &user).await?;
    let json = extract_json_object(&raw)?;
    let response = serde_json::from_str::<RawValidationResponse>(json)
        .with_context(|| format!("failed to parse validation JSON: {json}"))?;

    let allowed_ids = memories
        .iter()
        .map(|memory| memory.id.as_str())
        .collect::<HashSet<_>>();
    let mut decisions_by_id = HashMap::new();
    for raw in response.decisions {
        let Some(id) = raw.id else {
            continue;
        };
        if !allowed_ids.contains(id.as_str()) {
            continue;
        }
        let action = raw
            .action
            .as_deref()
            .map(ValidationAction::parse)
            .unwrap_or(ValidationAction::Keep);
        let confidence = raw.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
        let reason = truncate(
            raw.reason
                .as_deref()
                .unwrap_or("validator did not provide a reason"),
            500,
        );
        decisions_by_id.insert(
            id.clone(),
            ValidationDecision {
                id,
                action,
                confidence,
                reason,
            },
        );
    }

    let decisions = memories
        .iter()
        .map(|memory| {
            decisions_by_id
                .remove(&memory.id)
                .unwrap_or_else(|| ValidationDecision {
                    id: memory.id.clone(),
                    action: ValidationAction::Keep,
                    confidence: 0.0,
                    reason: "validator omitted this memory".to_string(),
                })
        })
        .collect();

    Ok(ValidationReport {
        model: model.to_string(),
        decisions,
    })
}

fn validation_system_prompt() -> &'static str {
    "You validate pending durable memory for a project-scoped coding agent. Return only JSON. \
     Promote only stable, reusable project facts, decisions, rules, user preferences, durable game design facts, \
     codebase lessons, asset pipeline facts, and known bug regressions with clear symptom and trigger/context. \
     Keep bug_regression memories for human review when the symptom, trigger, or expected behavior is unclear. \
     Archive secrets, credentials, private data, \
     one-off task status, vague chatter, duplicated facts, speculation, logs, and memories that would not be useful weeks later. \
     Use keep when the memory might be useful but needs human review."
}

fn validation_user_prompt(project_id: &str, memories: &[Memory]) -> String {
    let mut text = format!(
        "Project id: {project_id}\n\
         Return JSON exactly in this shape:\n\
         {{\"decisions\":[{{\"id\":\"memory-id\",\"action\":\"promote|archive|keep\",\"confidence\":0.8,\"reason\":\"short reason\"}}]}}\n\n\
         Pending memories:\n"
    );
    for memory in memories {
        text.push_str(&format!(
            "\n- id: {}\n  kind: {}\n  importance: {:.2}\n  confidence: {:.2}\n  source: {}\n  tags: {}\n  body: {}\n",
            memory.id,
            memory.kind,
            memory.importance,
            memory.confidence,
            memory.source.as_deref().unwrap_or("-"),
            memory.tags.join(","),
            truncate(&memory.body, 1_500)
        ));
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

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

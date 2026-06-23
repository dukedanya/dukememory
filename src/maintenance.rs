use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use crate::backup::{DatabaseBackupReport, create_database_backup};
use crate::compaction::{CompactionProposal, propose_compaction};
use crate::config::Config;
use crate::embedding::embed_missing;
use crate::ollama::OllamaClient;
use crate::store::{
    DEFAULT_MEMORY_SCOPE, DEFAULT_MEMORY_TIER, ListOptions, MemoryStatus, NewMemory,
    RememberOutcome, StatusFilter, Store,
};
use crate::validation::{ValidationAction, ValidationReport, validate_memories};

#[derive(Debug, Clone)]
pub struct MaintenanceOptions {
    pub apply: bool,
    pub backup: bool,
    pub backup_output: Option<PathBuf>,
    pub validate_pending: bool,
    pub validate_limit: usize,
    pub compact: bool,
    pub compact_limit: usize,
    pub compact_min_memories: usize,
    pub feedback: bool,
    pub feedback_limit: usize,
    pub embed_missing: bool,
    pub embed_limit: usize,
    pub embed_scope: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceReport {
    pub project_id: String,
    pub apply: bool,
    pub backup: Option<DatabaseBackupReport>,
    pub validation: Option<MaintenanceValidationReport>,
    pub compaction: Option<MaintenanceCompactionReport>,
    pub feedback: Option<MaintenanceFeedbackReport>,
    pub embeddings: Option<MaintenanceEmbeddingReport>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceValidationReport {
    pub status: String,
    pub model: String,
    pub pending: usize,
    pub apply: bool,
    pub promote: usize,
    pub archive: usize,
    pub keep: usize,
    pub decisions: Option<ValidationReport>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceCompactionReport {
    pub status: String,
    pub candidate_memories: usize,
    pub min_memories: usize,
    pub apply: bool,
    pub proposal: Option<CompactionProposal>,
    pub application: Option<MaintenanceCompactionApplication>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceCompactionApplication {
    pub summary_id: String,
    pub inserted: bool,
    pub duplicate_of: Option<String>,
    pub archived_source_memories: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceFeedbackReport {
    pub status: String,
    pub apply: bool,
    pub considered_events: usize,
    pub unapplied_events: usize,
    pub applied_events: usize,
    pub helpful_memories_updated: usize,
    pub unhelpful_memories_updated: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceEmbeddingReport {
    pub scope: String,
    pub limit: usize,
    pub apply: bool,
    pub memory_model: String,
    pub code_model: String,
    pub memories_missing: usize,
    pub code_symbols_missing: usize,
    pub memories_embedded: usize,
    pub code_symbols_embedded: usize,
}

pub async fn run_maintenance(
    config: &Config,
    project_id: &str,
    options: MaintenanceOptions,
) -> Result<MaintenanceReport> {
    let store = Store::open(&config.database_marker)?;
    let ollama = OllamaClient::new(
        config.ollama_base_url.clone(),
        config.extract_model().to_string(),
    );
    let mut warnings = Vec::new();

    let backup = if options.backup {
        Some(create_database_backup(
            &store,
            &config.database_marker,
            options.backup_output.clone(),
        )?)
    } else {
        None
    };

    let validation = if options.validate_pending {
        match run_validation_step(
            &store,
            &ollama,
            config,
            project_id,
            options.validate_limit,
            options.apply,
        )
        .await
        {
            Ok(report) => Some(report),
            Err(error) => {
                let message = format!("validation step failed: {error:#}");
                warnings.push(message.clone());
                Some(MaintenanceValidationReport {
                    status: "failed".to_string(),
                    model: config.validate_model.clone(),
                    pending: 0,
                    apply: options.apply,
                    promote: 0,
                    archive: 0,
                    keep: 0,
                    decisions: None,
                    error: Some(message),
                })
            }
        }
    } else {
        None
    };

    let compaction = if options.compact {
        let report = run_compaction_step(
            &store,
            &ollama,
            config,
            project_id,
            options.compact_limit,
            options.compact_min_memories,
            options.apply,
        )
        .await?;
        if let Some(error) = &report.error {
            warnings.push(format!("compaction step failed: {error}"));
        }
        Some(report)
    } else {
        None
    };

    let feedback = if options.feedback {
        match run_feedback_step(&store, project_id, options.feedback_limit, options.apply) {
            Ok(report) => Some(report),
            Err(error) => {
                let message = format!("feedback step failed: {error:#}");
                warnings.push(message.clone());
                Some(MaintenanceFeedbackReport {
                    status: "failed".to_string(),
                    apply: options.apply,
                    considered_events: 0,
                    unapplied_events: 0,
                    applied_events: 0,
                    helpful_memories_updated: 0,
                    unhelpful_memories_updated: 0,
                    error: Some(message),
                })
            }
        }
    } else {
        None
    };

    let embeddings = if options.embed_missing {
        match run_embedding_step(
            &store,
            &ollama,
            config,
            project_id,
            options.embed_limit,
            &options.embed_scope,
            options.apply,
        )
        .await
        {
            Ok(report) => Some(report),
            Err(error) => {
                let message = format!("embedding step failed: {error:#}");
                warnings.push(message);
                None
            }
        }
    } else {
        None
    };

    Ok(MaintenanceReport {
        project_id: project_id.to_string(),
        apply: options.apply,
        backup,
        validation,
        compaction,
        feedback,
        embeddings,
        warnings,
    })
}

fn run_feedback_step(
    store: &Store,
    project_id: &str,
    limit: usize,
    apply: bool,
) -> Result<MaintenanceFeedbackReport> {
    let events = store.list_audit_events(project_id, limit.clamp(1, 500))?;
    let applied_markers = events
        .iter()
        .filter(|event| event.action == "dukememory_feedback_apply")
        .filter_map(|event| {
            event
                .detail
                .get("feedback_event_id")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect::<HashSet<_>>();
    let unapplied = events
        .iter()
        .filter(|event| event.action == "dukememory_feedback")
        .filter(|event| !applied_markers.contains(&event.id))
        .collect::<Vec<_>>();
    let mut helpful_memories_updated = 0_usize;
    let mut unhelpful_memories_updated = 0_usize;
    let mut applied_events = 0_usize;
    if apply {
        for event in &unapplied {
            let helpful_ids = string_array_field(&event.detail, "helpful_ids");
            let unhelpful_ids = string_array_field(&event.detail, "unhelpful_ids");
            let effect = store.apply_memory_feedback(project_id, &helpful_ids, &unhelpful_ids)?;
            helpful_memories_updated += effect.helpful_memories_updated;
            unhelpful_memories_updated += effect.unhelpful_memories_updated;
            store.record_audit_event(
                project_id,
                "dukememory_maintenance",
                "dukememory_feedback_apply",
                "audit_event",
                Some(&event.id),
                json!({
                    "feedback_event_id": event.id,
                    "helpful_ids": helpful_ids,
                    "unhelpful_ids": unhelpful_ids,
                    "feedback_effect": effect
                }),
            )?;
            applied_events += 1;
        }
    }
    Ok(MaintenanceFeedbackReport {
        status: if apply { "applied" } else { "dry_run" }.to_string(),
        apply,
        considered_events: events.len(),
        unapplied_events: unapplied.len(),
        applied_events,
        helpful_memories_updated,
        unhelpful_memories_updated,
        error: None,
    })
}

fn string_array_field(value: &Value, field: &str) -> Vec<String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect()
}

async fn run_validation_step(
    store: &Store,
    ollama: &OllamaClient,
    config: &Config,
    project_id: &str,
    limit: usize,
    apply: bool,
) -> Result<MaintenanceValidationReport> {
    let pending = store.list(
        project_id,
        ListOptions {
            limit: limit.clamp(1, 500),
            offset: 0,
            status: StatusFilter::One(MemoryStatus::Pending),
            kind: None,
            memory_tier: None,
        },
    )?;
    let decisions = validate_memories(ollama, &config.validate_model, project_id, &pending).await?;
    if apply {
        for decision in &decisions.decisions {
            match decision.action {
                ValidationAction::Promote => {
                    store.promote(project_id, &decision.id, Some(&decision.reason))?
                }
                ValidationAction::Archive => {
                    store.archive(project_id, &decision.id, Some(&decision.reason))?
                }
                ValidationAction::Keep => {}
            }
        }
    }
    let promote = decisions
        .decisions
        .iter()
        .filter(|decision| decision.action == ValidationAction::Promote)
        .count();
    let archive = decisions
        .decisions
        .iter()
        .filter(|decision| decision.action == ValidationAction::Archive)
        .count();
    let keep = decisions
        .decisions
        .iter()
        .filter(|decision| decision.action == ValidationAction::Keep)
        .count();

    Ok(MaintenanceValidationReport {
        status: "ok".to_string(),
        model: decisions.model.clone(),
        pending: pending.len(),
        apply,
        promote,
        archive,
        keep,
        decisions: Some(decisions),
        error: None,
    })
}

async fn run_compaction_step(
    store: &Store,
    ollama: &OllamaClient,
    config: &Config,
    project_id: &str,
    limit: usize,
    min_memories: usize,
    apply: bool,
) -> Result<MaintenanceCompactionReport> {
    let memories = store.active_memories_for_compaction(project_id, limit.clamp(2, 500), None)?;
    if memories.len() < min_memories.clamp(2, 500) {
        return Ok(MaintenanceCompactionReport {
            status: "skipped".to_string(),
            candidate_memories: memories.len(),
            min_memories,
            apply,
            proposal: None,
            application: None,
            error: None,
        });
    }

    let proposal =
        match propose_compaction(ollama, config.extract_model(), project_id, &memories).await {
            Ok(proposal) => proposal,
            Err(error) => {
                return Ok(MaintenanceCompactionReport {
                    status: "failed".to_string(),
                    candidate_memories: memories.len(),
                    min_memories,
                    apply,
                    proposal: None,
                    application: None,
                    error: Some(format!("{error:#}")),
                });
            }
        };
    let application = if apply {
        Some(apply_compaction(store, project_id, &proposal)?)
    } else {
        None
    };

    Ok(MaintenanceCompactionReport {
        status: if apply { "applied" } else { "prepared" }.to_string(),
        candidate_memories: memories.len(),
        min_memories,
        apply,
        proposal: Some(proposal),
        application,
        error: None,
    })
}

fn apply_compaction(
    store: &Store,
    project_id: &str,
    proposal: &CompactionProposal,
) -> Result<MaintenanceCompactionApplication> {
    let outcome = store.remember_deduplicated(
        project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: proposal.summary_kind.clone(),
            body: proposal.summary_body.clone(),
            tags: proposal.tags.clone(),
            source: Some("dukememory_maintenance:compact".to_string()),
            status: MemoryStatus::Active,
            importance: proposal.importance,
            confidence: proposal.confidence,
            status_reason: Some(proposal.reason.clone()),
            allow_sensitive: false,
        },
    )?;
    let summary_id = summary_id(&outcome);
    for source_id in &proposal.source_ids {
        store.archive(
            project_id,
            source_id,
            Some(&format!("compacted into {summary_id}")),
        )?;
    }
    Ok(MaintenanceCompactionApplication {
        summary_id,
        inserted: outcome.inserted,
        duplicate_of: outcome.duplicate_of,
        archived_source_memories: proposal.source_ids.len(),
    })
}

async fn run_embedding_step(
    store: &Store,
    _ollama: &OllamaClient,
    config: &Config,
    project_id: &str,
    limit: usize,
    scope: &str,
    apply: bool,
) -> Result<MaintenanceEmbeddingReport> {
    let embed_memories = matches!(scope, "all" | "memory" | "memories");
    let embed_code = matches!(scope, "all" | "code" | "symbols" | "code_symbols");
    if !embed_memories && !embed_code {
        bail!("invalid embed scope `{scope}`; use all, memories, or code");
    }
    let limit = limit.clamp(1, 500);
    let memories = if embed_memories {
        store.memories_missing_embeddings(project_id, config.memory_embed_model(), limit)?
    } else {
        Vec::new()
    };
    let code_symbols = if embed_code {
        store.code_symbols_missing_embeddings(project_id, config.code_embed_model(), limit)?
    } else {
        Vec::new()
    };

    let (memories_embedded, code_symbols_embedded) = if apply {
        let report = embed_missing(config, store, project_id, limit, scope).await?;
        (report.memories, report.code_symbols)
    } else {
        (0, 0)
    };

    Ok(MaintenanceEmbeddingReport {
        scope: scope.to_string(),
        limit,
        apply,
        memory_model: config.memory_embed_model().to_string(),
        code_model: config.code_embed_model().to_string(),
        memories_missing: memories.len(),
        code_symbols_missing: code_symbols.len(),
        memories_embedded,
        code_symbols_embedded,
    })
}

fn summary_id(outcome: &RememberOutcome) -> String {
    outcome
        .duplicate_of
        .as_deref()
        .unwrap_or(outcome.id.as_str())
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn maintenance_skips_compaction_when_not_enough_memories_without_ollama() -> Result<()> {
        let config = test_config("maintenance-skip");
        let report = run_maintenance(
            &config,
            "game-a",
            MaintenanceOptions {
                apply: false,
                backup: false,
                backup_output: None,
                validate_pending: false,
                validate_limit: 20,
                compact: true,
                compact_limit: 40,
                compact_min_memories: 20,
                feedback: false,
                feedback_limit: 100,
                embed_missing: false,
                embed_limit: 50,
                embed_scope: "all".to_string(),
            },
        )
        .await?;
        let compaction = report.compaction.expect("compaction report");
        assert_eq!(compaction.status, "skipped");
        assert_eq!(compaction.candidate_memories, 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maintenance_reports_missing_embeddings_without_apply_or_ollama() -> Result<()> {
        let config = test_config("maintenance-embeddings");
        let store = Store::open(&config.database_marker)?;
        store.remember(
            "game-a",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Use deterministic tests for maintenance reports.".to_string(),
                tags: Vec::new(),
                source: None,
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        drop(store);

        let report = run_maintenance(
            &config,
            "game-a",
            MaintenanceOptions {
                apply: false,
                backup: false,
                backup_output: None,
                validate_pending: false,
                validate_limit: 20,
                compact: false,
                compact_limit: 40,
                compact_min_memories: 20,
                feedback: false,
                feedback_limit: 100,
                embed_missing: true,
                embed_limit: 50,
                embed_scope: "memories".to_string(),
            },
        )
        .await?;
        let embeddings = report.embeddings.expect("embedding report");
        assert_eq!(embeddings.memories_missing, 1);
        assert_eq!(embeddings.memories_embedded, 0);
        Ok(())
    }

    fn test_config(name: &str) -> Config {
        Config {
            database_url: "postgresql://dukememory-test@localhost:55432/dukememory_test"
                .to_string(),
            database_marker: std::env::temp_dir().join(format!(
                "dukememory-maintenance-test-{name}-{}.schema-marker",
                uuid::Uuid::now_v7()
            )),
            ollama_base_url: "http://127.0.0.1:1".to_string(),
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

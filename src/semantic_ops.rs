use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::code_index::check_code_index_freshness;
use crate::config::Config;
use crate::context_plan::{ContextPlanRequest, plan_context_access};
use crate::ollama::OllamaClient;
use crate::retrieval::{
    ContextRetrievalRequest, RetrievalMode, build_retrieval_plan, run_context_retrieval,
};
use crate::store::{
    CodeMemorySearchOptions, CodeSearchOptions, CodeVectorSearchOptions, DEFAULT_MEMORY_SCOPE,
    DEFAULT_MEMORY_TIER, ListOptions, Memory, MemoryFact, MemorySimilarityPair,
    MemorySimilarityPairOptions, MemoryStatus, NewEvalCase, NewMemory, RetrievalEvent,
    SearchOptions, StatusFilter, Store, VectorSearchOptions,
};

const DEFAULT_DEDUPE_SIMILARITY: f64 = 0.92;
const DEFAULT_RELATED_SIMILARITY: f64 = 0.72;
const DEFAULT_CLUSTER_SIMILARITY: f64 = 0.78;

#[derive(Debug, Clone)]
pub struct SemanticOperationRequest {
    pub project_id: String,
    pub action: String,
    pub query: Option<String>,
    pub body: Option<String>,
    pub memory_id: Option<String>,
    pub symbol: Option<String>,
    pub file_path: Option<String>,
    pub project_path: Option<String>,
    pub input: Option<String>,
    pub other_project_id: Option<String>,
    pub expected_ids: Vec<String>,
    pub helpful_ids: Vec<String>,
    pub unhelpful_ids: Vec<String>,
    pub limit: usize,
    pub status: StatusFilter,
    pub kind: Option<String>,
    pub memory_tier: Option<String>,
    pub mode: RetrievalMode,
    pub min_similarity: f64,
    pub target_memory_model: Option<String>,
    pub target_code_model: Option<String>,
    pub as_of: Option<String>,
    pub retrieval_event_id: Option<String>,
    pub outcome_kind: Option<String>,
    pub severity: Option<String>,
    pub apply: bool,
}

pub async fn run_semantic_operation(
    config: &Config,
    store: &Store,
    request: SemanticOperationRequest,
) -> Result<Value> {
    match request.action.as_str() {
        "policy" | "policy_decision" => semantic_policy(config, store, &request).await,
        "retrieval_quality" | "quality" => {
            semantic_retrieval_quality(config, store, &request).await
        }
        "auto_eval" | "task_eval_case" => semantic_auto_eval(config, store, &request).await,
        "ab_compare" | "model_ab_compare" => semantic_ab_compare(config, store, &request).await,
        "lifecycle" | "lifecycle_review" => semantic_lifecycle(config, store, &request).await,
        "consistency" | "consistency_check" => semantic_consistency(config, store, &request),
        "code_memory_suggest" | "code_memory_suggestions" => {
            semantic_code_memory_suggest(config, store, &request).await
        }
        "verify_conflicts" | "contradiction_verify" => {
            semantic_verify_conflicts(config, store, &request).await
        }
        "topic_map" | "ontology_drift" => semantic_topic_map(store, &request),
        "budget_optimize" | "context_budget_optimizer" => {
            semantic_budget_optimizer(config, store, &request).await
        }
        "feedback" | "outcome_feedback" => semantic_feedback(store, &request),
        "self_heal" | "self_healing" => semantic_self_heal(config, store, &request).await,
        "outcome_learn" | "task_outcome" | "task_learning" => {
            semantic_outcome_learn(store, &request)
        }
        "conflict_graph" | "memory_conflict_graph" => semantic_conflict_graph(store, &request),
        "memory_compiler" | "compile_memory" => {
            semantic_memory_compiler(config, store, &request).await
        }
        "policy_ab" | "policy_ab_runner" | "policy_ab_compare" => {
            semantic_policy_ab(config, store, &request).await
        }
        "context_policy" | "policy_learn" | "learned_policy" => {
            semantic_context_policy(config, store, &request).await
        }
        "trace" | "flight_recorder" | "task_replay" | "replay" => semantic_trace(store, &request),
        "counterfactual" | "counterfactual_eval" => {
            semantic_counterfactual_eval(config, store, &request).await
        }
        "causality" | "code_causality" | "memory_impact" => {
            semantic_causality(config, store, &request).await
        }
        "temporal_context" | "as_of_context" => semantic_temporal_context(store, &request),
        "dedupe" => semantic_dedupe(config, store, &request),
        "related" => semantic_related(config, store, &request).await,
        "review" | "conflicts" | "supersede_suggestions" => {
            semantic_review(config, store, &request).await
        }
        "route" => semantic_route(&request),
        "clusters" | "compact_clusters" => semantic_clusters(config, store, &request),
        "tag" | "tags" => semantic_tags(store, &request),
        "stale" => semantic_stale(store, &request),
        "eval_cases" | "eval_generator" => semantic_eval_cases(config, store, &request),
        "hard_negatives" => semantic_hard_negatives(config, store, &request).await,
        "health" | "embedding_health" => semantic_embedding_health(config, store, &request),
        "migration" | "model_migration" => semantic_model_migration(config, store, &request),
        "isolation_check" => semantic_isolation_check(config, store, &request).await,
        "hints" => semantic_hints(config, store, &request).await,
        other => bail!("invalid dukememory_semantic action `{other}`"),
    }
}

fn semantic_dedupe(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let pairs = store.memory_similarity_pairs(
        &request.project_id,
        MemorySimilarityPairOptions {
            embedding_model: config.memory_embed_model().to_string(),
            limit: request.limit,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
            min_similarity: similarity_or(request.min_similarity, DEFAULT_DEDUPE_SIMILARITY),
        },
    )?;
    let groups = grouped_memory_pairs(&pairs);
    Ok(json!({
        "action": "dedupe",
        "project_id": request.project_id,
        "model": config.memory_embed_model(),
        "min_similarity": similarity_or(request.min_similarity, DEFAULT_DEDUPE_SIMILARITY),
        "pairs": pairs.iter().map(memory_pair_json).collect::<Vec<_>>(),
        "groups": groups,
        "candidate_pairs": pairs.len()
    }))
}

fn semantic_consistency(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let status = store.status(&request.project_id)?;
    let code_status = store.code_status(&request.project_id)?;
    let stale = semantic_stale(store, request)?;
    let stale_candidates = stale["candidates"].as_array().cloned().unwrap_or_default();
    let conflict_candidates = memory_conflict_candidates(store, request)?;
    let mut warnings = Vec::new();
    let duplicate_pairs = match store.memory_similarity_pairs(
        &request.project_id,
        MemorySimilarityPairOptions {
            embedding_model: config.memory_embed_model().to_string(),
            limit: request.limit.clamp(1, 200),
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
            min_similarity: similarity_or(request.min_similarity, DEFAULT_DEDUPE_SIMILARITY),
        },
    ) {
        Ok(pairs) => pairs,
        Err(error) => {
            warnings.push(format!("duplicate similarity scan skipped: {error}"));
            Vec::new()
        }
    };
    let freshness = if let Some(path) = &request.project_path {
        match check_code_index_freshness(
            store,
            &PathBuf::from(path),
            Some(request.project_id.clone()),
        ) {
            Ok(report) => {
                if !report.stale_files.is_empty() {
                    warnings.push(format!(
                        "{} indexed files changed on disk",
                        report.stale_files.len()
                    ));
                }
                if !report.missing_files.is_empty() {
                    warnings.push(format!(
                        "{} indexed files are missing on disk",
                        report.missing_files.len()
                    ));
                }
                if !report.deleted_files.is_empty() {
                    warnings.push(format!(
                        "{} indexed files should be deleted from index",
                        report.deleted_files.len()
                    ));
                }
                json!({
                    "checked": true,
                    "stale_files": report.stale_files,
                    "missing_files": report.missing_files,
                    "deleted_files": report.deleted_files,
                    "files_seen": report.files_seen,
                    "indexed_files": report.indexed_files,
                    "project_id": report.project_id,
                    "root_path": report.root_path
                })
            }
            Err(error) => {
                warnings.push(format!("code index freshness check skipped: {error}"));
                json!({"checked": false, "error": error.to_string()})
            }
        }
    } else {
        json!({
            "checked": false,
            "reason": "project_path not supplied"
        })
    };

    let freshness_issues = freshness["stale_files"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0)
        + freshness["missing_files"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0)
        + freshness["deleted_files"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
    let mut findings = Vec::new();
    if !conflict_candidates.is_empty() {
        findings.push(format!(
            "{} lexical memory conflict candidates need review",
            conflict_candidates.len()
        ));
    }
    if !duplicate_pairs.is_empty() {
        findings.push(format!(
            "{} near-duplicate memory pairs may need dedupe",
            duplicate_pairs.len()
        ));
    }
    if !stale_candidates.is_empty() {
        findings.push(format!(
            "{} stale code-memory candidates need repair/archive",
            stale_candidates.len()
        ));
    }
    if status.pending_memories > 0 {
        findings.push(format!(
            "{} pending memories are excluded from default active retrieval",
            status.pending_memories
        ));
    }
    if freshness_issues > 0 {
        findings.push(format!(
            "{freshness_issues} code index freshness issues require reindex"
        ));
    }
    if code_status.symbols == 0 {
        findings.push("code graph has no indexed symbols".to_string());
    }
    findings.extend(warnings.iter().cloned());

    let mut penalty = 0.0_f64;
    if !conflict_candidates.is_empty() {
        penalty += 0.30;
    }
    if !duplicate_pairs.is_empty() {
        penalty += 0.15;
    }
    if !stale_candidates.is_empty() {
        penalty += 0.20;
    }
    if status.pending_memories > 0 {
        penalty += 0.10;
    }
    if freshness_issues > 0 {
        penalty += 0.20;
    }
    if code_status.symbols == 0 {
        penalty += 0.10;
    }
    let readiness_score = (1.0 - penalty).clamp(0.0, 1.0);
    let status_label = if readiness_score >= 0.85 {
        "ready"
    } else if readiness_score >= 0.60 {
        "needs_attention"
    } else {
        "blocked_by_consistency"
    };

    Ok(json!({
        "action": "consistency",
        "project_id": request.project_id,
        "readiness_score": readiness_score,
        "status": status_label,
        "findings": findings,
        "warnings": warnings,
        "memory_status": status,
        "code_status": code_status,
        "freshness": freshness,
        "stale_code_memories": stale_candidates,
        "conflict_candidates": conflict_candidates,
        "duplicate_pairs": duplicate_pairs.iter().map(memory_pair_json).collect::<Vec<_>>(),
        "min_duplicate_similarity": similarity_or(request.min_similarity, DEFAULT_DEDUPE_SIMILARITY)
    }))
}

async fn semantic_policy(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let body = request
        .body
        .as_deref()
        .or(request.input.as_deref())
        .ok_or_else(|| anyhow!("policy action requires body or input"))?;
    let review_request = SemanticOperationRequest {
        action: "review".to_string(),
        body: Some(body.to_string()),
        limit: request.limit.max(8),
        ..request.clone()
    };
    let review = semantic_review(config, store, &review_request).await?;
    let suggestions = review["suggestions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut decision = json!({
        "action": "insert",
        "confidence": 0.72,
        "reason": "no close active semantic neighbor requires lifecycle action"
    });
    for suggestion in &suggestions {
        let class = suggestion["classification"].as_str().unwrap_or("");
        let memory_id = suggestion["memory"]["id"].as_str().unwrap_or("");
        let score = suggestion["memory"]["score"].as_f64().unwrap_or(0.0);
        match class {
            "duplicate" => {
                decision = json!({
                    "action": "skip_duplicate",
                    "confidence": score.max(0.9),
                    "target_id": memory_id,
                    "reason": "proposed memory is semantically duplicate of an existing live memory"
                });
                break;
            }
            "possible_conflict" => {
                decision = json!({
                    "action": "needs_review",
                    "confidence": score.max(0.75),
                    "target_id": memory_id,
                    "reason": "proposed memory may conflict with an existing memory"
                });
                break;
            }
            "possible_update" if decision["action"] == "insert" => {
                decision = json!({
                    "action": "supersede_candidate",
                    "confidence": score.max(0.8),
                    "target_id": memory_id,
                    "reason": "proposed memory looks like an update to an existing memory"
                });
            }
            _ => {}
        }
    }
    Ok(json!({
        "action": "policy",
        "project_id": request.project_id,
        "decision": decision,
        "review": review
    }))
}

async fn semantic_retrieval_quality(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("retrieval_quality requires query"))?;
    let output = run_context_retrieval(
        config,
        store,
        ContextRetrievalRequest {
            project_id: &request.project_id,
            query,
            memory_limit: request.limit.clamp(1, 50),
            code_limit: request.limit.clamp(0, 50),
            mode: request.mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let ids = output
        .memories
        .iter()
        .map(|memory| memory.id.clone())
        .collect::<Vec<_>>();
    let similarities =
        store.memory_pair_similarities(&request.project_id, &ids, config.memory_embed_model())?;
    let max_redundancy = similarities.values().copied().fold(0.0, f64::max);
    let kinds = output
        .memories
        .iter()
        .map(|memory| memory.kind.clone())
        .collect::<HashSet<_>>();
    let memory_score = if output.memories.is_empty() {
        0.0
    } else {
        0.35
    };
    let code_score = if output.code.is_empty() { 0.0 } else { 0.2 };
    let diversity_score = (1.0 - max_redundancy).clamp(0.0, 1.0) * 0.25;
    let kind_score = (kinds.len() as f64 / output.memories.len().max(1) as f64).min(1.0) * 0.1;
    let warning_score = if output.diagnostics.warnings.is_empty() {
        0.1
    } else {
        0.0
    };
    let quality = memory_score + code_score + diversity_score + kind_score + warning_score;
    let mut findings = Vec::new();
    if output.memories.is_empty() {
        findings.push("no memory hits".to_string());
    }
    if output.code.is_empty() {
        findings.push("no code hits".to_string());
    }
    if max_redundancy >= 0.88 {
        findings.push(format!("high memory redundancy ({max_redundancy:.2})"));
    }
    findings.extend(output.diagnostics.warnings.clone());
    Ok(json!({
        "action": "retrieval_quality",
        "project_id": request.project_id,
        "query": query,
        "quality_score": quality.clamp(0.0, 1.0),
        "memory_hits": output.memories.len(),
        "code_hits": output.code.len(),
        "memory_kind_diversity": kinds.len(),
        "max_memory_redundancy": max_redundancy,
        "diagnostics": output.diagnostics,
        "findings": findings,
        "top_memory_ids": ids,
        "top_code_symbols": output.code.iter().map(|result| compact_code_symbol(&result.symbol)).collect::<Vec<_>>()
    }))
}

async fn semantic_auto_eval(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let text = request
        .input
        .as_deref()
        .or(request.body.as_deref())
        .or(request.query.as_deref())
        .ok_or_else(|| anyhow!("auto_eval requires input, body, or query"))?;
    let query = request
        .query
        .clone()
        .unwrap_or_else(|| top_terms(text, 8).join(" "));
    let related_request = SemanticOperationRequest {
        action: "related".to_string(),
        query: Some(query.clone()),
        limit: request.limit.max(8),
        ..request.clone()
    };
    let related = semantic_related(config, store, &related_request).await?;
    let expected_ids = related["memories"]
        .as_array()
        .map(|memories| {
            memories
                .iter()
                .filter_map(|memory| memory["id"].as_str().map(str::to_string))
                .take(3)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let case = json!({
        "name": format!("agent-task-{}", short_id(&uuid::Uuid::now_v7().to_string())),
        "query": query,
        "expected_ids": expected_ids,
        "expected_contains": top_terms(text, 4),
        "min_results": 1
    });
    Ok(json!({
        "action": "auto_eval",
        "project_id": request.project_id,
        "case": case,
        "related": related
    }))
}

async fn semantic_ab_compare(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("ab_compare requires query"))?;
    let baseline_model = config.memory_embed_model().to_string();
    let target_model = request
        .target_memory_model
        .clone()
        .ok_or_else(|| anyhow!("ab_compare requires target_memory_model"))?;
    let baseline =
        semantic_memory_search_with_model(config, store, request, query, &baseline_model).await?;
    let target =
        semantic_memory_search_with_model(config, store, request, query, &target_model).await?;
    let baseline_ids = memory_ids(&baseline);
    let target_ids = memory_ids(&target);
    let overlap = jaccard(&baseline_ids, &target_ids);
    Ok(json!({
        "action": "ab_compare",
        "project_id": request.project_id,
        "query": query,
        "baseline_model": baseline_model,
        "target_model": target_model,
        "limit": request.limit,
        "overlap": overlap,
        "baseline": compact_memories(&baseline),
        "target": compact_memories(&target),
        "baseline_only_ids": set_difference(&baseline_ids, &target_ids),
        "target_only_ids": set_difference(&target_ids, &baseline_ids)
    }))
}

async fn semantic_lifecycle(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let dedupe = semantic_dedupe(config, store, request)?;
    let clusters = semantic_clusters(config, store, request)?;
    let stale = semantic_stale(store, request)?;
    let tags = semantic_tags(store, request)?;
    let health = semantic_embedding_health(config, store, request)?;
    Ok(json!({
        "action": "lifecycle",
        "project_id": request.project_id,
        "review_bundle": {
            "dedupe_pairs": dedupe["candidate_pairs"].clone(),
            "dedupe_groups": dedupe["groups"].clone(),
            "clusters": clusters["clusters"].clone(),
            "stale_candidates": stale["candidates"].clone(),
            "tag_suggestions": tags["suggestions"].clone(),
            "embedding_health": health
        }
    }))
}

async fn semantic_code_memory_suggest(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .or(request.input.as_deref())
        .ok_or_else(|| anyhow!("code_memory_suggest requires query or input"))?;
    let results = if let Some(file_path) = &request.file_path {
        store.search_code(
            &request.project_id,
            CodeSearchOptions {
                query: query.to_string(),
                limit: request.limit,
                kind: request.kind.clone(),
                file_path: Some(file_path.clone()),
            },
        )?
    } else {
        let ollama = ollama_from_config(config);
        match ollama
            .embed_with_model(config.code_embed_model(), query)
            .await
        {
            Ok(embedding) if embedding.len() == 1024 => store.search_code_vectors(
                &request.project_id,
                CodeVectorSearchOptions {
                    embedding,
                    embedding_model: config.code_embed_model().to_string(),
                    limit: request.limit,
                    kind: request.kind.clone(),
                    file_path: None,
                },
            )?,
            _ => store.search_code(
                &request.project_id,
                CodeSearchOptions {
                    query: query.to_string(),
                    limit: request.limit,
                    kind: request.kind.clone(),
                    file_path: None,
                },
            )?,
        }
    };
    let suggestions = results
        .iter()
        .map(|result| {
            json!({
                "symbol": compact_code_symbol(&result.symbol),
                "score": result.score,
                "suggested_code_memory": {
                    "symbol_id": result.symbol.id.clone(),
                    "file_path": result.symbol.file_path.clone(),
                    "kind": "usage",
                    "status": "pending",
                    "confidence": 0.72,
                    "tags": ["code-memory", "semantic-suggestion"],
                    "body": format!(
                        "Task `{}` is related to `{}` in `{}`; review whether a durable invariant, risk, or usage note should be stored.",
                        truncate(query, 140),
                        result.symbol.name,
                        result.symbol.file_path
                    )
                }
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "action": "code_memory_suggest",
        "project_id": request.project_id,
        "query": query,
        "suggestions": suggestions
    }))
}

async fn semantic_verify_conflicts(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let review = semantic_review(config, store, request).await?;
    let candidates = review["suggestions"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|item| item["classification"] == "possible_conflict")
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(json!({
            "action": "verify_conflicts",
            "project_id": request.project_id,
            "verified": [],
            "review": review
        }));
    }
    let body = request.body.as_deref().unwrap_or_default();
    let ollama = ollama_from_config(config);
    let mut verified = Vec::new();
    for candidate in candidates {
        let prompt = format!(
            "New memory:\n{}\n\nExisting memory:\n{}\n\nReturn JSON with keys verdict(one of conflict, update, duplicate, related), confidence number 0..1, reason string.",
            body,
            candidate["memory"]["body_preview"].as_str().unwrap_or("")
        );
        let llm = ollama
            .chat_json_with_model(
                config.extract_model(),
                "You verify whether two project memory statements contradict each other. Return only JSON.",
                &prompt,
            )
            .await;
        let verdict = match llm {
            Ok(text) => serde_json::from_str::<Value>(&text).unwrap_or_else(
                |_| json!({"verdict": "needs_review", "confidence": 0.5, "reason": text}),
            ),
            Err(error) => json!({
                "verdict": "needs_review",
                "confidence": 0.5,
                "reason": format!("LLM verifier unavailable: {error}")
            }),
        };
        verified.push(json!({
            "candidate": candidate,
            "verification": verdict
        }));
    }
    Ok(json!({
        "action": "verify_conflicts",
        "project_id": request.project_id,
        "verified": verified
    }))
}

fn semantic_topic_map(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let memories = store.list(
        &request.project_id,
        ListOptions {
            limit: request.limit.clamp(1, 500),
            offset: 0,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )?;
    let mut topics = HashMap::<String, usize>::new();
    let mut kinds = HashMap::<String, usize>::new();
    for memory in &memories {
        *kinds.entry(memory.kind.clone()).or_default() += 1;
        for tag in suggest_tags(memory, &memories) {
            *topics.entry(tag).or_default() += 1;
        }
    }
    Ok(json!({
        "action": "topic_map",
        "project_id": request.project_id,
        "sampled_memories": memories.len(),
        "topics": ranked_counts(topics),
        "kinds": ranked_counts(kinds),
        "drift_signals": topic_drift_signals(&memories)
    }))
}

async fn semantic_budget_optimizer(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("budget_optimize requires query"))?;
    let candidates = [4usize, 8, 12, 20]
        .into_iter()
        .filter(|limit| *limit <= request.limit.max(4))
        .collect::<Vec<_>>();
    let mut plans = Vec::new();
    for limit in candidates {
        let quality_request = SemanticOperationRequest {
            limit,
            query: Some(query.to_string()),
            ..request.clone()
        };
        let quality = semantic_retrieval_quality(config, store, &quality_request).await?;
        let score = quality["quality_score"].as_f64().unwrap_or(0.0);
        let estimated_tokens = (limit * 700) / 4;
        let utility_per_1k = if estimated_tokens == 0 {
            score
        } else {
            score / (estimated_tokens as f64 / 1000.0)
        };
        plans.push(json!({
            "memory_limit": limit,
            "code_limit": limit,
            "estimated_tokens": estimated_tokens,
            "quality_score": score,
            "utility_per_1k_tokens": utility_per_1k,
            "findings": quality["findings"].clone()
        }));
    }
    plans.sort_by(|a, b| {
        b["utility_per_1k_tokens"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&a["utility_per_1k_tokens"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(json!({
        "action": "budget_optimize",
        "project_id": request.project_id,
        "query": query,
        "recommended": plans.first().cloned().unwrap_or(Value::Null),
        "plans": plans
    }))
}

fn semantic_feedback(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let retrieval_event = request
        .retrieval_event_id
        .as_deref()
        .map(|id| store.get_retrieval_event(&request.project_id, id))
        .transpose()?
        .flatten();
    let query = request
        .query
        .as_deref()
        .or_else(|| retrieval_event.as_ref().map(|event| event.query.as_str()))
        .ok_or_else(|| anyhow!("feedback requires query or retrieval_event_id"))?;
    if request.helpful_ids.is_empty() && request.unhelpful_ids.is_empty() {
        bail!("feedback requires helpful_ids or unhelpful_ids");
    }
    let outcome_kind = validate_feedback_outcome_kind(request.outcome_kind.as_deref())?;
    let severity = validate_feedback_severity(request.severity.as_deref())?;
    let effect = store.apply_memory_feedback(
        &request.project_id,
        &request.helpful_ids,
        &request.unhelpful_ids,
    )?;
    let regression_eval_case = Some(store.record_eval_case(
        &request.project_id,
        NewEvalCase {
            suite_name: "feedback-regressions",
            name: &feedback_eval_case_name(query, &outcome_kind),
            query,
            expected_contains: Vec::new(),
            forbidden_contains: Vec::new(),
            expected_ids: request.helpful_ids.clone(),
            forbidden_ids: request.unhelpful_ids.clone(),
            min_results: Some(1),
        },
    )?);
    let event_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        "dukememory_feedback",
        "retrieval",
        request.retrieval_event_id.as_deref(),
        json!({
            "query": query,
            "retrieval_event_id": request.retrieval_event_id,
            "outcome_kind": outcome_kind,
            "severity": severity,
            "helpful_ids": request.helpful_ids.clone(),
            "unhelpful_ids": request.unhelpful_ids.clone(),
            "note": request.body.as_deref().or(request.input.as_deref()),
            "feedback_effect": effect,
            "regression_eval_case": regression_eval_case
        }),
    )?;
    Ok(json!({
        "action": "feedback",
        "project_id": request.project_id,
        "event_id": event_id,
        "query": query,
        "retrieval_event_id": request.retrieval_event_id,
        "outcome_kind": outcome_kind,
        "severity": severity,
        "helpful_ids": request.helpful_ids.clone(),
        "unhelpful_ids": request.unhelpful_ids.clone(),
        "feedback_effect": effect,
        "regression_eval_case": regression_eval_case,
        "effect": "stored as typed audit feedback, applied to memory quality scores, and converted into a feedback-regression eval case"
    }))
}

fn validate_feedback_outcome_kind(value: Option<&str>) -> Result<String> {
    let value = value.unwrap_or("wrong_memory").trim();
    match value {
        "wrong_memory" | "missing_memory" | "stale_memory" | "contradiction" | "bad_code_hit"
        | "bad_answer" | "bug_regression" => Ok(value.to_string()),
        other => bail!(
            "invalid feedback outcome_kind `{other}`; use wrong_memory, missing_memory, stale_memory, contradiction, bad_code_hit, bad_answer, or bug_regression"
        ),
    }
}

fn validate_feedback_severity(value: Option<&str>) -> Result<String> {
    let value = value.unwrap_or("medium").trim();
    match value {
        "low" | "medium" | "high" => Ok(value.to_string()),
        other => bail!("invalid feedback severity `{other}`; use low, medium, or high"),
    }
}

fn feedback_eval_case_name(query: &str, outcome_kind: &str) -> String {
    let slug = query
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() || *ch == '-')
        .collect::<String>()
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        format!("feedback-{outcome_kind}")
    } else {
        format!("feedback-{outcome_kind}-{slug}")
    }
}

async fn semantic_self_heal(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .or(request.input.as_deref())
        .unwrap_or("project memory self healing");
    let review_request = SemanticOperationRequest {
        action: "lifecycle".to_string(),
        query: Some(query.to_string()),
        limit: request.limit.clamp(5, 100),
        status: StatusFilter::Any,
        ..request.clone()
    };
    let lifecycle = semantic_lifecycle(config, store, &review_request).await?;
    let conflict_graph = semantic_conflict_graph(store, &review_request)?;
    let compiler = semantic_memory_compiler(config, store, &review_request).await?;
    let outcome = semantic_outcome_learn(store, &review_request)?;
    let mut applied_events = Vec::new();
    if request.apply {
        let effect = apply_self_heal_actions(store, request, &compiler, &conflict_graph)?;
        applied_events.push(effect);
    }
    let event_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        if request.apply {
            "dukememory_self_heal_apply"
        } else {
            "dukememory_self_heal_plan"
        },
        "project",
        Some(&request.project_id),
        json!({
            "query": query,
            "apply": request.apply,
            "compiler_counts": compiler.get("counts").cloned().unwrap_or(Value::Null),
            "conflict_counts": conflict_graph.get("counts").cloned().unwrap_or(Value::Null),
            "outcome_counts": outcome.get("counts").cloned().unwrap_or(Value::Null),
            "applied": applied_events
        }),
    )?;
    Ok(json!({
        "action": "self_heal",
        "project_id": request.project_id,
        "query": query,
        "apply": request.apply,
        "audit_event_id": event_id,
        "readiness": if request.apply { "applied" } else { "dry_run" },
        "lifecycle": lifecycle,
        "conflict_graph": conflict_graph,
        "memory_compiler": compiler,
        "outcome_learning": outcome,
        "applied": applied_events
    }))
}

fn semantic_outcome_learn(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let session = if let Some(id) = request.memory_id.as_deref() {
        store.get_task_session(&request.project_id, id)?
    } else {
        None
    };
    let sessions = if let Some(session) = session {
        vec![session]
    } else {
        store.list_task_sessions(&request.project_id, Some("completed"), request.limit)?
    };
    let mut observations = Vec::new();
    let mut helpful_ids = Vec::new();
    let mut unhelpful_ids = Vec::new();
    for session in sessions {
        let events =
            store.list_retrieval_events_for_session(&request.project_id, &session.id, 50)?;
        let mut memory_ids = session.memory_ids.clone();
        for event in &events {
            memory_ids.extend(memory_ids_from_retrieval_event(event));
        }
        memory_ids.sort();
        memory_ids.dedup();
        let outcome = classify_task_outcome(&session.result, &session.status);
        match outcome.as_str() {
            "positive" => helpful_ids.extend(memory_ids.iter().cloned()),
            "negative" => unhelpful_ids.extend(memory_ids.iter().cloned()),
            _ => {}
        }
        observations.push(json!({
            "session_id": session.id,
            "query": session.query,
            "status": session.status,
            "phase": session.phase,
            "progress": session.progress,
            "outcome": outcome,
            "retrieval_events": events.iter().map(compact_retrieval_event).collect::<Vec<_>>(),
            "memory_ids": memory_ids
        }));
    }
    helpful_ids.sort();
    helpful_ids.dedup();
    unhelpful_ids.sort();
    unhelpful_ids.dedup();
    let effect = if request.apply {
        Some(store.apply_memory_feedback(&request.project_id, &helpful_ids, &unhelpful_ids)?)
    } else {
        None
    };
    let event_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        if request.apply {
            "dukememory_outcome_learn_apply"
        } else {
            "dukememory_outcome_learn_plan"
        },
        "task_session",
        request.memory_id.as_deref(),
        json!({
            "apply": request.apply,
            "helpful_ids": helpful_ids,
            "unhelpful_ids": unhelpful_ids,
            "effect": effect,
            "observations": observations
        }),
    )?;
    Ok(json!({
        "action": "outcome_learn",
        "project_id": request.project_id,
        "apply": request.apply,
        "audit_event_id": event_id,
        "counts": {
            "sessions": observations.len(),
            "helpful_ids": helpful_ids.len(),
            "unhelpful_ids": unhelpful_ids.len()
        },
        "helpful_ids": helpful_ids,
        "unhelpful_ids": unhelpful_ids,
        "feedback_effect": effect,
        "observations": observations
    }))
}

fn semantic_conflict_graph(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let query = request.query.as_deref().unwrap_or("");
    let memory_conflicts = memory_conflict_candidates(store, request)?;
    let graph = store.search_memory_graph(&request.project_id, query, request.limit.max(50))?;
    let mut fact_conflicts = fact_conflict_candidates(&graph.facts);
    fact_conflicts.truncate(request.limit.clamp(1, 100));
    let mut invalidated = Vec::new();
    if request.apply {
        for conflict in &fact_conflicts {
            if conflict.get("suggested_action").and_then(Value::as_str)
                != Some("invalidate_weaker_fact")
            {
                continue;
            }
            if let Some(loser_id) = conflict.get("weaker_fact_id").and_then(Value::as_str) {
                let winner_id = conflict.get("stronger_fact_id").and_then(Value::as_str);
                let fact = store.invalidate_memory_fact(
                    &request.project_id,
                    loser_id,
                    winner_id,
                    request.as_of.as_deref(),
                )?;
                invalidated.push(json!({
                    "fact_id": fact.id,
                    "invalidated_by": fact.invalidated_by,
                    "valid_to": fact.valid_to
                }));
            }
        }
    }
    let event_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        if request.apply {
            "dukememory_conflict_graph_apply"
        } else {
            "dukememory_conflict_graph_plan"
        },
        "memory_graph",
        Some(&request.project_id),
        json!({
            "query": query,
            "apply": request.apply,
            "memory_conflicts": memory_conflicts,
            "fact_conflicts": fact_conflicts,
            "invalidated": invalidated
        }),
    )?;
    Ok(json!({
        "action": "conflict_graph",
        "project_id": request.project_id,
        "query": query,
        "apply": request.apply,
        "audit_event_id": event_id,
        "counts": {
            "memory_conflicts": memory_conflicts.len(),
            "fact_conflicts": fact_conflicts.len(),
            "invalidated_facts": invalidated.len()
        },
        "memory_conflicts": memory_conflicts,
        "fact_conflicts": fact_conflicts,
        "invalidated": invalidated
    }))
}

async fn semantic_memory_compiler(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let memories = store.list(
        &request.project_id,
        ListOptions {
            limit: request.limit.clamp(10, 500),
            offset: 0,
            status: StatusFilter::One(MemoryStatus::Active),
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )?;
    let duplicate_pairs = store
        .memory_similarity_pairs(
            &request.project_id,
            MemorySimilarityPairOptions {
                embedding_model: config.memory_embed_model().to_string(),
                limit: request.limit.clamp(1, 200),
                status: StatusFilter::One(MemoryStatus::Active),
                kind: request.kind.clone(),
                memory_tier: request.memory_tier.clone(),
                min_similarity: similarity_or(request.min_similarity, DEFAULT_DEDUPE_SIMILARITY),
            },
        )
        .unwrap_or_default();
    let mut promote_to_core = Vec::new();
    let mut archive_candidates = Vec::new();
    let mut split_candidates = Vec::new();
    for memory in &memories {
        if should_compile_to_core(memory) {
            promote_to_core.push(memory.clone());
        }
        if should_compile_archive(memory) {
            archive_candidates.push(memory.clone());
        }
        if memory.body.chars().count() > 1_600 {
            split_candidates.push(memory.clone());
        }
    }
    let mut duplicate_archive = Vec::new();
    for pair in &duplicate_pairs {
        duplicate_archive.push(lower_quality_memory(&pair.left, &pair.right).clone());
    }
    dedupe_memories(&mut duplicate_archive);
    let mut applied = Vec::new();
    if request.apply {
        for memory in &promote_to_core {
            store.set_memory_tier(
                &request.project_id,
                &memory.id,
                "core",
                Some("dukememory_memory_compiler promoted stable high-confidence memory"),
            )?;
            applied
                .push(json!({"action": "set_memory_tier", "id": memory.id, "memory_tier": "core"}));
        }
        for memory in archive_candidates.iter().chain(duplicate_archive.iter()) {
            store.archive(
                &request.project_id,
                &memory.id,
                Some("dukememory_memory_compiler archived low-signal or duplicate memory"),
            )?;
            applied.push(json!({"action": "archive", "id": memory.id}));
        }
        for memory in &split_candidates {
            for (index, chunk) in split_memory_body(&memory.body).into_iter().enumerate() {
                let id = store.remember_deduplicated(
                    &request.project_id,
                    NewMemory {
                        scope: DEFAULT_MEMORY_SCOPE.to_string(),
                        memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                        kind: memory.kind.clone(),
                        body: chunk,
                        tags: memory.tags.clone(),
                        source: Some("dukememory_memory_compiler".to_string()),
                        status: MemoryStatus::Pending,
                        importance: memory.importance,
                        confidence: (memory.confidence * 0.95).clamp(0.0, 1.0),
                        status_reason: Some(format!(
                            "split candidate {} from long memory {}",
                            index + 1,
                            memory.id
                        )),
                        allow_sensitive: false,
                    },
                )?;
                applied.push(json!({
                    "action": "create_pending_split",
                    "source_id": memory.id,
                    "id": id.id,
                    "inserted": id.inserted
                }));
            }
        }
    }
    let event_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        if request.apply {
            "dukememory_memory_compiler_apply"
        } else {
            "dukememory_memory_compiler_plan"
        },
        "memory",
        Some(&request.project_id),
        json!({
            "apply": request.apply,
            "promote_to_core": promote_to_core.iter().map(compact_memory).collect::<Vec<_>>(),
            "archive_candidates": archive_candidates.iter().map(compact_memory).collect::<Vec<_>>(),
            "duplicate_archive": duplicate_archive.iter().map(compact_memory).collect::<Vec<_>>(),
            "split_candidates": split_candidates.iter().map(compact_memory).collect::<Vec<_>>(),
            "applied": applied
        }),
    )?;
    Ok(json!({
        "action": "memory_compiler",
        "project_id": request.project_id,
        "apply": request.apply,
        "audit_event_id": event_id,
        "counts": {
            "active_memories_considered": memories.len(),
            "promote_to_core": promote_to_core.len(),
            "archive_candidates": archive_candidates.len(),
            "duplicate_archive": duplicate_archive.len(),
            "split_candidates": split_candidates.len(),
            "applied": applied.len()
        },
        "promote_to_core": promote_to_core.iter().map(compact_memory).collect::<Vec<_>>(),
        "archive_candidates": archive_candidates.iter().map(compact_memory).collect::<Vec<_>>(),
        "duplicate_archive": duplicate_archive.iter().map(compact_memory).collect::<Vec<_>>(),
        "split_candidates": split_candidates.iter().map(compact_memory).collect::<Vec<_>>(),
        "duplicate_pairs": duplicate_pairs.iter().map(memory_pair_json).collect::<Vec<_>>(),
        "applied": applied
    }))
}

async fn semantic_policy_ab(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .or(request.input.as_deref())
        .ok_or_else(|| anyhow!("policy_ab requires query or input"))?;
    let policies = [
        (
            "balanced",
            request.limit.clamp(4, 16),
            request.limit.clamp(4, 16),
            RetrievalMode::Hybrid,
        ),
        (
            "memory_heavy",
            request.limit.clamp(8, 30),
            4,
            RetrievalMode::Hybrid,
        ),
        (
            "code_heavy",
            4,
            request.limit.clamp(8, 30),
            RetrievalMode::Hybrid,
        ),
        (
            "keyword_control",
            request.limit.clamp(4, 16),
            request.limit.clamp(4, 16),
            RetrievalMode::Keyword,
        ),
    ];
    let mut results = Vec::new();
    for (name, memory_limit, code_limit, mode) in policies {
        let output = run_context_retrieval(
            config,
            store,
            ContextRetrievalRequest {
                project_id: &request.project_id,
                query,
                memory_limit,
                code_limit,
                mode,
                allow_hybrid_fallback: true,
            },
        )
        .await?;
        let score = policy_trial_score(query, &output.memories, output.code.len());
        results.push(json!({
            "policy": name,
            "mode": mode.as_str(),
            "memory_limit": memory_limit,
            "code_limit": code_limit,
            "score": score,
            "memory_ids": output.memories.iter().map(|memory| memory.id.clone()).collect::<Vec<_>>(),
            "code_symbol_ids": output.code.iter().map(|result| result.symbol.id.clone()).collect::<Vec<_>>(),
            "diagnostics": output.diagnostics
        }));
    }
    results.sort_by(|left, right| {
        right["score"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&left["score"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let recommended = results.first().cloned().unwrap_or(Value::Null);
    let event_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        "dukememory_policy_ab",
        "retrieval_policy",
        None,
        json!({
            "query": query,
            "recommended": recommended,
            "results": results
        }),
    )?;
    Ok(json!({
        "action": "policy_ab",
        "project_id": request.project_id,
        "query": query,
        "audit_event_id": event_id,
        "recommended_policy": recommended,
        "trials": results
    }))
}

async fn semantic_context_policy(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .or(request.input.as_deref())
        .ok_or_else(|| anyhow!("context_policy requires query or input"))?;
    let base = plan_context_access(ContextPlanRequest {
        query,
        memory_limit: request.limit.clamp(1, 30),
        core_memory_limit: 5,
        code_limit: request.limit.clamp(1, 30),
        token_budget: 6_000,
    });
    let quality_request = SemanticOperationRequest {
        action: "retrieval_quality".to_string(),
        query: Some(query.to_string()),
        limit: request.limit.clamp(4, 30),
        ..request.clone()
    };
    let quality = semantic_retrieval_quality(config, store, &quality_request).await?;
    let events = store.list_retrieval_events(&request.project_id, None, request.limit.max(50))?;
    let query_terms = top_terms(query, 12).into_iter().collect::<HashSet<_>>();
    let relevant = events
        .iter()
        .filter(|event| {
            event.task_type == base.task_type
                || term_overlap(&query_terms, &top_terms(&event.query, 12)) >= 0.25
        })
        .cloned()
        .collect::<Vec<_>>();
    let policy = learned_policy_from_events(&base, &relevant, quality["quality_score"].as_f64());
    let feedback_events = store
        .list_audit_events(&request.project_id, request.limit.max(50))?
        .into_iter()
        .filter(|event| event.action == "dukememory_feedback")
        .filter(|event| {
            event
                .detail
                .get("query")
                .and_then(Value::as_str)
                .map(|value| value.contains(query) || query.contains(value))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "action": "context_policy",
        "project_id": request.project_id,
        "query": query,
        "baseline_plan": base,
        "quality_probe": quality,
        "history": {
            "events_considered": events.len(),
            "relevant_events": relevant.iter().map(compact_retrieval_event).collect::<Vec<_>>(),
            "feedback_events": feedback_events
        },
        "recommended_policy": policy
    }))
}

fn semantic_trace(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let id = request.memory_id.as_deref();
    let session = if let Some(id) = id {
        store
            .get_task_session(&request.project_id, id)
            .unwrap_or(None)
    } else {
        None
    };
    let direct_event = if let Some(id) = id {
        store
            .get_retrieval_event(&request.project_id, id)
            .unwrap_or(None)
    } else {
        None
    };
    let trace_query = request
        .query
        .as_deref()
        .or_else(|| session.as_ref().map(|session| session.query.as_str()))
        .or_else(|| direct_event.as_ref().map(|event| event.query.as_str()))
        .map(str::to_string);
    let retrieval_events = if let Some(event) = direct_event {
        vec![event]
    } else if let Some(session) = &session {
        let events = store.list_retrieval_events_for_session(
            &request.project_id,
            &session.id,
            request.limit.max(20),
        )?;
        if events.is_empty() {
            store.list_retrieval_events(
                &request.project_id,
                trace_query.as_deref(),
                request.limit.max(20),
            )?
        } else {
            events
        }
    } else {
        store.list_retrieval_events(
            &request.project_id,
            trace_query.as_deref(),
            request.limit.max(20),
        )?
    };
    let mut memory_ids = session
        .as_ref()
        .map(|session| session.memory_ids.clone())
        .unwrap_or_default();
    let mut code_symbol_ids = session
        .as_ref()
        .map(|session| session.code_symbol_ids.clone())
        .unwrap_or_default();
    for event in &retrieval_events {
        memory_ids.extend(memory_ids_from_retrieval_event(event));
        code_symbol_ids.extend(code_symbol_ids_from_retrieval_event(event));
    }
    memory_ids.sort();
    memory_ids.dedup();
    code_symbol_ids.sort();
    code_symbol_ids.dedup();
    let memories = memory_ids
        .iter()
        .filter_map(|id| store.get(&request.project_id, id).ok().flatten())
        .collect::<Vec<_>>();
    let code_symbols = code_symbol_ids
        .iter()
        .filter_map(|id| {
            store
                .get_code_symbol(&request.project_id, id)
                .ok()
                .flatten()
        })
        .collect::<Vec<_>>();
    let feedback_events = store
        .list_audit_events(&request.project_id, request.limit.max(50))?
        .into_iter()
        .filter(|event| event.action == "dukememory_feedback")
        .filter(|event| {
            trace_query
                .as_deref()
                .zip(event.detail.get("query").and_then(Value::as_str))
                .map(|(query, value)| value.contains(query) || query.contains(value))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "action": "trace",
        "project_id": request.project_id,
        "id": id,
        "query": trace_query,
        "task_session": session,
        "retrieval_events": retrieval_events.iter().map(compact_retrieval_event).collect::<Vec<_>>(),
        "selected_memories": compact_memories(&memories),
        "selected_code_symbols": code_symbols.iter().map(compact_code_symbol).collect::<Vec<_>>(),
        "feedback_events": feedback_events,
        "replay": {
            "command": "dukememory_prepare or dukememory_context with the trace query, mode, limits, and token budget shown in the retrieval event",
            "deterministic_inputs": retrieval_events.iter().map(|event| json!({
                "event_id": event.id,
                "tool": event.tool,
                "query": event.query,
                "task_type": event.task_type,
                "token_budget": event.token_budget,
                "plan": event.plan
            })).collect::<Vec<_>>()
        }
    }))
}

async fn semantic_counterfactual_eval(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("counterfactual_eval requires query"))?;
    let output = run_context_retrieval(
        config,
        store,
        ContextRetrievalRequest {
            project_id: &request.project_id,
            query,
            memory_limit: request.limit.clamp(2, 30),
            code_limit: request.limit.clamp(0, 30),
            mode: request.mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let ids = output
        .memories
        .iter()
        .map(|memory| memory.id.clone())
        .collect::<Vec<_>>();
    let similarities =
        store.memory_pair_similarities(&request.project_id, &ids, config.memory_embed_model())?;
    let expected = request.expected_ids.iter().cloned().collect::<HashSet<_>>();
    let ablations = output
        .memories
        .iter()
        .map(|memory| {
            let other_ids = ids
                .iter()
                .filter(|id| *id != &memory.id)
                .cloned()
                .collect::<Vec<_>>();
            let redundancy_without = max_similarity_for_ids(&similarities, &other_ids);
            let expected_loss = expected.contains(&memory.id);
            let quality_delta = if expected_loss {
                -0.35
            } else if ((memory.importance * 0.55) + (memory.confidence * 0.45)) >= 0.75 {
                -0.15
            } else if redundancy_without < 0.35 {
                -0.10
            } else {
                0.03
            };
            json!({
                "removed_memory": compact_memory(memory),
                "counterfactual": {
                    "remaining_memory_ids": other_ids,
                    "max_redundancy_without": redundancy_without,
                    "expected_id_loss": expected_loss,
                    "estimated_quality_delta": quality_delta,
                    "verdict": if expected_loss || quality_delta < -0.1 { "keep_or_promote_eval_expected" } else if quality_delta > 0.0 { "candidate_for_lower_rank_or_archive_review" } else { "neutral" }
                }
            })
        })
        .collect::<Vec<_>>();
    let hard_negative_request = SemanticOperationRequest {
        action: "hard_negatives".to_string(),
        query: Some(query.to_string()),
        expected_ids: ids.iter().take(3).cloned().collect(),
        limit: request.limit.max(8),
        ..request.clone()
    };
    let hard_negatives = semantic_hard_negatives(config, store, &hard_negative_request).await?;
    Ok(json!({
        "action": "counterfactual_eval",
        "project_id": request.project_id,
        "query": query,
        "mode": request.mode.as_str(),
        "baseline": {
            "memory_ids": ids,
            "code_symbol_ids": output.code.iter().map(|result| result.symbol.id.clone()).collect::<Vec<_>>(),
            "diagnostics": output.diagnostics
        },
        "leave_one_out": ablations,
        "hard_negatives": hard_negatives,
        "suggested_eval_case": {
            "name": format!("counterfactual-{}", short_id(&uuid::Uuid::now_v7().to_string())),
            "query": query,
            "expected_ids": request.expected_ids.iter().cloned().chain(output.memories.iter().take(2).map(|memory| memory.id.clone())).collect::<Vec<_>>(),
            "forbidden_contains": [],
            "min_results": 1
        }
    }))
}

async fn semantic_causality(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .or(request.input.as_deref())
        .or(request.body.as_deref())
        .unwrap_or("memory code causality");
    let output = run_context_retrieval(
        config,
        store,
        ContextRetrievalRequest {
            project_id: &request.project_id,
            query,
            memory_limit: request.limit.clamp(1, 30),
            code_limit: request.limit.clamp(1, 30),
            mode: request.mode,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let memory_ids = if let Some(memory_id) = &request.memory_id {
        vec![memory_id.clone()]
    } else {
        output
            .memories
            .iter()
            .map(|memory| memory.id.clone())
            .collect()
    };
    let code_symbol_ids = if let Some(symbol) = &request.symbol {
        store
            .get_code_symbol(&request.project_id, symbol)?
            .map(|symbol| vec![symbol.id])
            .unwrap_or_default()
    } else {
        output
            .code
            .iter()
            .map(|result| result.symbol.id.clone())
            .collect()
    };
    let memories = memory_ids
        .iter()
        .filter_map(|id| store.get(&request.project_id, id).ok().flatten())
        .collect::<Vec<_>>();
    let symbols = code_symbol_ids
        .iter()
        .filter_map(|id| {
            store
                .get_code_symbol(&request.project_id, id)
                .ok()
                .flatten()
        })
        .collect::<Vec<_>>();
    let graph =
        store.memory_graph_for_memories(&request.project_id, &memory_ids, request.limit.max(10))?;
    let (_related_symbols, relations) = store.code_graph_for_symbols(
        &request.project_id,
        &code_symbol_ids,
        request.limit.max(50),
    )?;
    let causal_edges = memory_code_edges(&memories, &symbols);
    let impacted_files = symbols
        .iter()
        .map(|symbol| symbol.file_path.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let affected_tests = store.affected_test_files(
        &request.project_id,
        &impacted_files,
        5,
        request.limit.max(25),
    )?;
    Ok(json!({
        "action": "causality",
        "project_id": request.project_id,
        "query": query,
        "memories": compact_memories(&memories),
        "code_symbols": symbols.iter().map(compact_code_symbol).collect::<Vec<_>>(),
        "causal_edges": causal_edges,
        "memory_graph": {
            "entities": graph.entities,
            "facts": graph.facts,
            "edges": graph.edges
        },
        "code_relations": relations,
        "impacted_files": impacted_files,
        "affected_tests": affected_tests,
        "diagnostics": output.diagnostics
    }))
}

fn semantic_temporal_context(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("temporal_context requires query"))?;
    let as_of = request
        .as_of
        .as_deref()
        .or(request.input.as_deref())
        .ok_or_else(|| anyhow!("temporal_context requires as_of or input timestamp"))?;
    let memories =
        store.search_memories_as_of(&request.project_id, query, as_of, request.limit.max(8))?;
    let graph = store.search_memory_graph_at(
        &request.project_id,
        query,
        request.limit.max(8),
        Some(as_of),
    )?;
    Ok(json!({
        "action": "temporal_context",
        "project_id": request.project_id,
        "query": query,
        "as_of": as_of,
        "memories": compact_memories(&memories),
        "graph": {
            "entities": graph.entities,
            "facts": graph.facts,
            "edges": graph.edges
        },
        "temporal_model": {
            "memory_rule": "includes active memories created before as_of plus memories superseded or archived after as_of",
            "graph_rule": "uses fact/edge valid_from, valid_to, and invalidated_by windows"
        }
    }))
}

async fn semantic_related(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    if let Some(memory_id) = &request.memory_id {
        let memories = store.search_related_memories(
            &request.project_id,
            memory_id,
            MemorySimilarityPairOptions {
                embedding_model: config.memory_embed_model().to_string(),
                limit: request.limit,
                status: request.status,
                kind: request.kind.clone(),
                memory_tier: request.memory_tier.clone(),
                min_similarity: similarity_or(request.min_similarity, DEFAULT_RELATED_SIMILARITY),
            },
        )?;
        return Ok(json!({
            "action": "related",
            "project_id": request.project_id,
            "source": {"type": "memory", "id": memory_id},
            "memory_model": config.memory_embed_model(),
            "memories": compact_memories(&memories),
            "code": []
        }));
    }

    if let Some(symbol_ref) = &request.symbol {
        let symbol =
            store.resolve_code_symbol_reference(&request.project_id, symbol_ref, None, None)?;
        let code = store.search_related_code_symbols(
            &request.project_id,
            &symbol.id,
            config.code_embed_model(),
            request.limit,
            similarity_or(request.min_similarity, DEFAULT_RELATED_SIMILARITY),
        )?;
        return Ok(json!({
            "action": "related",
            "project_id": request.project_id,
            "source": {"type": "code_symbol", "id": symbol.id, "name": symbol.name},
            "code_model": config.code_embed_model(),
            "memories": [],
            "code": code.iter().map(|result| json!({
                "score": result.score,
                "symbol": compact_code_symbol(&result.symbol)
            })).collect::<Vec<_>>()
        }));
    }

    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("related action requires query, memory_id, or symbol"))?;
    let ollama = ollama_from_config(config);
    let memory_embedding = ollama
        .embed_with_model(config.memory_embed_model(), query)
        .await?;
    if memory_embedding.len() != 4096 {
        bail!(
            "memory embedding dimension mismatch: expected 4096, got {}",
            memory_embedding.len()
        );
    }
    let memories = store.search_memory_vectors(
        &request.project_id,
        VectorSearchOptions {
            embedding: memory_embedding,
            embedding_model: config.memory_embed_model().to_string(),
            limit: request.limit,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )?;
    let code_embedding = ollama
        .embed_with_model(config.code_embed_model(), query)
        .await?;
    let code = if code_embedding.len() == 1024 {
        store.search_code_vectors(
            &request.project_id,
            CodeVectorSearchOptions {
                embedding: code_embedding,
                embedding_model: config.code_embed_model().to_string(),
                limit: request.limit,
                kind: None,
                file_path: None,
            },
        )?
    } else {
        Vec::new()
    };
    Ok(json!({
        "action": "related",
        "project_id": request.project_id,
        "source": {"type": "query", "query": query},
        "memory_model": config.memory_embed_model(),
        "code_model": config.code_embed_model(),
        "memories": compact_memories(&memories),
        "code": code.iter().map(|result| json!({
            "score": result.score,
            "symbol": compact_code_symbol(&result.symbol)
        })).collect::<Vec<_>>()
    }))
}

async fn semantic_review(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let body = request
        .body
        .as_deref()
        .ok_or_else(|| anyhow!("review action requires body"))?;
    let ollama = ollama_from_config(config);
    let embedding = ollama
        .embed_with_model(config.memory_embed_model(), body)
        .await?;
    if embedding.len() != 4096 {
        bail!(
            "memory embedding dimension mismatch: expected 4096, got {}",
            embedding.len()
        );
    }
    let neighbors = store.search_memory_vectors(
        &request.project_id,
        VectorSearchOptions {
            embedding,
            embedding_model: config.memory_embed_model().to_string(),
            limit: request.limit,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )?;
    let suggestions = neighbors
        .iter()
        .map(|memory| {
            let classification = classify_relation(body, &memory.body, memory.score.unwrap_or(0.0));
            json!({
                "memory": compact_memory(memory),
                "classification": classification.0,
                "recommended_action": classification.1,
                "reason": classification.2
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "action": "review",
        "project_id": request.project_id,
        "model": config.memory_embed_model(),
        "suggestions": suggestions
    }))
}

fn semantic_route(request: &SemanticOperationRequest) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("route action requires query"))?;
    let plan = build_retrieval_plan(query);
    Ok(json!({
        "action": "route",
        "project_id": request.project_id,
        "query": query,
        "intent": plan.intent,
        "weights": {
            "memory": plan.memory_weight,
            "code": plan.code_weight,
            "graph": plan.graph_weight,
            "keyword": plan.keyword_weight,
            "semantic": plan.semantic_weight,
            "symbol": plan.symbol_weight
        },
        "notes": plan.notes
    }))
}

fn semantic_clusters(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let min_similarity = similarity_or(request.min_similarity, DEFAULT_CLUSTER_SIMILARITY);
    let pairs = store.memory_similarity_pairs(
        &request.project_id,
        MemorySimilarityPairOptions {
            embedding_model: config.memory_embed_model().to_string(),
            limit: request.limit.clamp(1, 500),
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
            min_similarity,
        },
    )?;
    Ok(json!({
        "action": "clusters",
        "project_id": request.project_id,
        "model": config.memory_embed_model(),
        "min_similarity": min_similarity,
        "clusters": grouped_memory_pairs(&pairs),
        "pairs": pairs.iter().map(memory_pair_json).collect::<Vec<_>>()
    }))
}

fn semantic_tags(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let memories = if let Some(query) = &request.query {
        store.search(
            &request.project_id,
            SearchOptions {
                query: query.clone(),
                limit: request.limit,
                status: request.status,
                kind: request.kind.clone(),
                memory_tier: request.memory_tier.clone(),
            },
        )?
    } else {
        store.list(
            &request.project_id,
            ListOptions {
                limit: request.limit,
                offset: 0,
                status: request.status,
                kind: request.kind.clone(),
                memory_tier: request.memory_tier.clone(),
            },
        )?
    };
    let tagged_examples = store.list(
        &request.project_id,
        ListOptions {
            limit: 500,
            offset: 0,
            status: StatusFilter::One(MemoryStatus::Active),
            kind: None,
            memory_tier: None,
        },
    )?;
    let suggestions = memories
        .iter()
        .map(|memory| {
            json!({
                "memory": compact_memory(memory),
                "suggested_tags": suggest_tags(memory, &tagged_examples)
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "action": "tag",
        "project_id": request.project_id,
        "suggestions": suggestions
    }))
}

fn semantic_stale(store: &Store, request: &SemanticOperationRequest) -> Result<Value> {
    let code_memories = store.search_code_memories(
        &request.project_id,
        CodeMemorySearchOptions {
            query: request.query.clone(),
            limit: request.limit,
            status: "any".to_string(),
            kind: None,
            symbol_ids: Vec::new(),
            file_paths: Vec::new(),
        },
    )?;
    let mut results = Vec::new();
    for memory in code_memories {
        let mut reasons = Vec::new();
        if memory.link_status == "stale" {
            reasons.push("link_status is stale".to_string());
        }
        if let Some(symbol_id) = &memory.symbol_id {
            match store.get_code_symbol(&request.project_id, symbol_id)? {
                Some(symbol) => {
                    let overlap = token_overlap(
                        &memory.body,
                        &format!("{} {}", symbol.signature, symbol.body),
                    );
                    if overlap < 0.08 {
                        reasons.push(format!(
                            "low lexical overlap with linked symbol ({overlap:.2})"
                        ));
                    }
                }
                None => reasons.push("linked symbol no longer exists".to_string()),
            }
        } else if memory.file_path.is_none() {
            reasons.push("code memory has no symbol_id or file_path".to_string());
        }
        if !reasons.is_empty() {
            results.push(json!({
                "id": memory.id,
                "symbol_id": memory.symbol_id,
                "file_path": memory.file_path,
                "link_status": memory.link_status,
                "kind": memory.kind,
                "status": memory.status,
                "reasons": reasons
            }));
        }
    }
    Ok(json!({
        "action": "stale",
        "project_id": request.project_id,
        "candidates": results
    }))
}

fn memory_conflict_candidates(
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Vec<Value>> {
    let memories = store.list(
        &request.project_id,
        ListOptions {
            limit: request.limit.clamp(2, 300),
            offset: 0,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )?;
    let mut candidates = Vec::new();
    for (left_index, left) in memories.iter().enumerate() {
        for right in memories.iter().skip(left_index + 1) {
            if !has_conflict_markers(&left.body, &right.body) {
                continue;
            }
            let overlap = token_overlap(&left.body, &right.body);
            if overlap < 0.18 {
                continue;
            }
            candidates.push(json!({
                "left": compact_memory(left),
                "right": compact_memory(right),
                "overlap": overlap,
                "reason": "memories share topic terms and contain opposing lifecycle/policy language",
                "suggested_action": "manual_review_or_supersede"
            }));
        }
    }
    candidates.sort_by(|left, right| {
        let left_score = left["overlap"].as_f64().unwrap_or(0.0);
        let right_score = right["overlap"].as_f64().unwrap_or(0.0);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(request.limit.clamp(1, 100));
    Ok(candidates)
}

fn semantic_eval_cases(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let memories = store.list(
        &request.project_id,
        ListOptions {
            limit: request.limit,
            offset: 0,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )?;
    let mut cases = Vec::new();
    for memory in &memories {
        let query = eval_query_for_memory(memory);
        let hard_negatives = store
            .search_related_memories(
                &request.project_id,
                &memory.id,
                MemorySimilarityPairOptions {
                    embedding_model: config.memory_embed_model().to_string(),
                    limit: 5,
                    status: request.status,
                    kind: None,
                    memory_tier: None,
                    min_similarity: 0.0,
                },
            )
            .unwrap_or_default()
            .into_iter()
            .filter(|candidate| candidate.id != memory.id)
            .map(|candidate| candidate.id)
            .collect::<Vec<_>>();
        cases.push(json!({
            "name": format!("semantic-{}", short_id(&memory.id)),
            "query": query,
            "expected_ids": [memory.id.clone()],
            "expected_contains": memory.tags.iter().take(2).cloned().collect::<Vec<_>>(),
            "hard_negative_ids": hard_negatives
        }));
    }
    Ok(json!({
        "action": "eval_cases",
        "project_id": request.project_id,
        "mode": "hybrid",
        "cases": cases
    }))
}

async fn semantic_hard_negatives(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let expected = request.expected_ids.iter().cloned().collect::<HashSet<_>>();
    let mut negatives = Vec::new();
    if !request.expected_ids.is_empty() {
        for id in &request.expected_ids {
            let related = store.search_related_memories(
                &request.project_id,
                id,
                MemorySimilarityPairOptions {
                    embedding_model: config.memory_embed_model().to_string(),
                    limit: request.limit,
                    status: request.status,
                    kind: request.kind.clone(),
                    memory_tier: request.memory_tier.clone(),
                    min_similarity: similarity_or(request.min_similarity, 0.0),
                },
            )?;
            negatives.extend(
                related
                    .into_iter()
                    .filter(|memory| !expected.contains(&memory.id)),
            );
        }
    } else if let Some(query) = &request.query {
        let related = semantic_related(config, store, request).await?;
        return Ok(json!({
            "action": "hard_negatives",
            "project_id": request.project_id,
            "query": query,
            "note": "no expected_ids supplied; returning nearest semantic candidates for manual labeling",
            "candidates": related["memories"].clone()
        }));
    } else {
        bail!("hard_negatives requires expected_ids or query");
    }
    dedupe_memories(&mut negatives);
    negatives.truncate(request.limit);
    Ok(json!({
        "action": "hard_negatives",
        "project_id": request.project_id,
        "expected_ids": request.expected_ids,
        "candidates": compact_memories(&negatives)
    }))
}

fn semantic_embedding_health(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let status = store.status(&request.project_id)?;
    let code_status = store.code_status(&request.project_id)?;
    let stats = store.embedding_model_stats(&request.project_id)?;
    Ok(json!({
        "action": "embedding_health",
        "project_id": request.project_id,
        "memory_model": config.memory_embed_model(),
        "code_model": config.code_embed_model(),
        "memory_coverage": coverage(status.memory_embeddings, status.total_memories),
        "code_coverage": coverage(code_status.symbol_embeddings, code_status.symbols),
        "memory_embeddings": status.memory_embeddings,
        "total_memories": status.total_memories,
        "code_symbol_embeddings": code_status.symbol_embeddings,
        "code_symbols": code_status.symbols,
        "models": stats
    }))
}

fn semantic_model_migration(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let status = store.status(&request.project_id)?;
    let code_status = store.code_status(&request.project_id)?;
    let stats = store.embedding_model_stats(&request.project_id)?;
    let target_memory = request
        .target_memory_model
        .clone()
        .unwrap_or_else(|| config.memory_embed_model().to_string());
    let target_code = request
        .target_code_model
        .clone()
        .unwrap_or_else(|| config.code_embed_model().to_string());
    let target_memory_count = complete_embedding_count_for_model(
        &stats.memory_models,
        &target_memory,
        &["body", "metadata"],
    );
    let target_code_count = complete_embedding_count_for_model(
        &stats.code_models,
        &target_code,
        &["body", "signature"],
    );
    Ok(json!({
        "action": "model_migration",
        "project_id": request.project_id,
        "current": {
            "memory_model": config.memory_embed_model(),
            "code_model": config.code_embed_model()
        },
        "target": {
            "memory_model": target_memory,
            "code_model": target_code
        },
        "missing_for_target": {
            "memories": status.total_memories.saturating_sub(target_memory_count),
            "code_symbols": code_status.symbols.saturating_sub(target_code_count)
        },
        "models": stats,
        "recommended_steps": [
            "run dukememory_embed_missing for the target memory model after configuration change",
            "run dukememory_code_index with embed_symbols or dukememory_embed_missing scope=code_symbols after code model change",
            "run dukememory_eval before and after switching default models"
        ]
    }))
}

async fn semantic_isolation_check(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("isolation_check requires query"))?;
    let other_project_id = request
        .other_project_id
        .as_deref()
        .ok_or_else(|| anyhow!("isolation_check requires other_project_id"))?;
    let primary = run_context_retrieval(
        config,
        store,
        ContextRetrievalRequest {
            project_id: &request.project_id,
            query,
            memory_limit: request.limit,
            code_limit: 0,
            mode: RetrievalMode::Hybrid,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let other = run_context_retrieval(
        config,
        store,
        ContextRetrievalRequest {
            project_id: other_project_id,
            query,
            memory_limit: request.limit,
            code_limit: 0,
            mode: RetrievalMode::Hybrid,
            allow_hybrid_fallback: true,
        },
    )
    .await?;
    let primary_ids = primary
        .memories
        .iter()
        .map(|m| m.id.clone())
        .collect::<HashSet<_>>();
    let leaked = other
        .memories
        .iter()
        .filter(|memory| primary_ids.contains(&memory.id))
        .cloned()
        .collect::<Vec<_>>();
    Ok(json!({
        "action": "isolation_check",
        "project_id": request.project_id,
        "other_project_id": other_project_id,
        "query": query,
        "primary_hits": primary.memories.len(),
        "other_hits": other.memories.len(),
        "leaked_ids": leaked.iter().map(|memory| memory.id.clone()).collect::<Vec<_>>(),
        "passed": leaked.is_empty()
    }))
}

async fn semantic_hints(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
) -> Result<Value> {
    let query = request
        .query
        .as_deref()
        .ok_or_else(|| anyhow!("hints action requires query"))?;
    let route = semantic_route(request)?;
    let related = semantic_related(config, store, request).await?;
    let stale = semantic_stale(store, request)?;
    Ok(json!({
        "action": "hints",
        "project_id": request.project_id,
        "query": query,
        "route": route,
        "related_memories": related["memories"].clone(),
        "related_code": related["code"].clone(),
        "stale_candidates": stale["candidates"].clone()
    }))
}

pub fn diversify_memories_by_embedding(
    store: &Store,
    project_id: &str,
    model: &str,
    memories: Vec<Memory>,
    limit: usize,
) -> Result<Vec<Memory>> {
    let limit = limit.clamp(1, 50);
    if memories.len() <= limit {
        return Ok(memories);
    }
    let ids = memories
        .iter()
        .map(|memory| memory.id.clone())
        .collect::<Vec<_>>();
    let similarities = store.memory_pair_similarities(project_id, &ids, model)?;
    if similarities.is_empty() {
        return Ok(memories.into_iter().take(limit).collect());
    }
    Ok(mmr_select(memories, &similarities, limit))
}

fn mmr_select(
    memories: Vec<Memory>,
    similarities: &HashMap<(String, String), f64>,
    limit: usize,
) -> Vec<Memory> {
    let mut remaining = memories;
    let mut selected = Vec::new();
    while !remaining.is_empty() && selected.len() < limit {
        let best_index = remaining
            .iter()
            .enumerate()
            .map(|(index, memory)| {
                let relevance = memory.score.unwrap_or(0.0);
                let redundancy = selected
                    .iter()
                    .map(|chosen: &Memory| {
                        similarities
                            .get(&(memory.id.clone(), chosen.id.clone()))
                            .or_else(|| similarities.get(&(chosen.id.clone(), memory.id.clone())))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .fold(0.0, f64::max);
                let score = 0.72 * relevance - 0.28 * redundancy;
                (index, score)
            })
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        selected.push(remaining.remove(best_index));
    }
    selected
}

fn grouped_memory_pairs(pairs: &[MemorySimilarityPair]) -> Vec<Value> {
    let mut parent = HashMap::<String, String>::new();
    let mut memory_by_id = HashMap::<String, &Memory>::new();
    for pair in pairs {
        memory_by_id.insert(pair.left.id.clone(), &pair.left);
        memory_by_id.insert(pair.right.id.clone(), &pair.right);
        union(&mut parent, &pair.left.id, &pair.right.id);
    }
    let mut groups = HashMap::<String, Vec<String>>::new();
    for id in memory_by_id.keys() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push(id.clone());
    }
    let mut output = groups
        .into_values()
        .filter(|ids| ids.len() > 1)
        .map(|mut ids| {
            ids.sort();
            let memories = ids
                .iter()
                .filter_map(|id| memory_by_id.get(id).copied())
                .map(compact_memory)
                .collect::<Vec<_>>();
            json!({
                "size": memories.len(),
                "memory_ids": ids,
                "memories": memories
            })
        })
        .collect::<Vec<_>>();
    output.sort_by_key(|group| std::cmp::Reverse(group["size"].as_u64().unwrap_or(0)));
    output
}

fn union(parent: &mut HashMap<String, String>, left: &str, right: &str) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        parent.insert(right_root, left_root);
    }
}

fn find(parent: &mut HashMap<String, String>, id: &str) -> String {
    let current = parent
        .entry(id.to_string())
        .or_insert_with(|| id.to_string())
        .clone();
    if current == id {
        return current;
    }
    let root = find(parent, &current);
    parent.insert(id.to_string(), root.clone());
    root
}

fn memory_pair_json(pair: &MemorySimilarityPair) -> Value {
    json!({
        "similarity": pair.similarity,
        "left": compact_memory(&pair.left),
        "right": compact_memory(&pair.right)
    })
}

fn learned_policy_from_events(
    base: &crate::context_plan::ContextPlan,
    events: &[RetrievalEvent],
    quality_score: Option<f64>,
) -> Value {
    let best = events.iter().max_by(|left, right| {
        event_utility(left)
            .partial_cmp(&event_utility(right))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let avg = |value: fn(&RetrievalEvent) -> usize| -> f64 {
        if events.is_empty() {
            0.0
        } else {
            events.iter().map(value).sum::<usize>() as f64 / events.len() as f64
        }
    };
    let avg_memory = avg(|event| event.memory_fragments);
    let avg_code = avg(|event| event.code_hits);
    let avg_code_memories = avg(|event| event.code_memories);
    let avg_tokens = avg(|event| event.estimated_tokens);
    let mut memory_limit = base.memory_limit;
    let mut code_limit = base.code_limit;
    let mut token_budget = base.budget_plan.effective_token_budget;
    let mut reasons = base.reasons.clone();
    if let Some(best) = best {
        memory_limit = memory_limit.max(best.memory_fragments.clamp(4, 24));
        code_limit = code_limit.max(best.code_hits.clamp(4, 24));
        if best.estimated_tokens > 0 {
            token_budget = (best.estimated_tokens * 5 / 4).clamp(1_000, 30_000);
        }
        reasons.push(format!(
            "best historical event {} utility={:.2}",
            best.id,
            event_utility(best)
        ));
    }
    if avg_code > avg_memory * 1.5 {
        code_limit = code_limit.saturating_add(4).min(30);
        reasons.push("history is code-heavy; increased code_limit".to_string());
    }
    if avg_code_memories >= 3.0 {
        reasons
            .push("history repeatedly used code memories; keep code_memories enabled".to_string());
    }
    if quality_score.unwrap_or(1.0) < 0.55 {
        memory_limit = memory_limit.saturating_add(4).min(30);
        code_limit = code_limit.saturating_add(4).min(30);
        token_budget = (token_budget + 2_000).min(30_000);
        reasons.push("quality probe is weak; widened retrieval limits and budget".to_string());
    }
    json!({
        "task_type": base.task_type,
        "memory_limit": memory_limit,
        "core_memory_limit": base.core_memory_limit,
        "code_limit": code_limit,
        "graph_limit": base.graph_limit.max(memory_limit).max(code_limit).min(40),
        "code_memory_limit": base.code_memory_limit.max(code_limit.min(24)),
        "token_budget": token_budget,
        "source_plan": base.source_plan,
        "history_stats": {
            "events": events.len(),
            "avg_memory_fragments": avg_memory,
            "avg_code_hits": avg_code,
            "avg_code_memories": avg_code_memories,
            "avg_estimated_tokens": avg_tokens
        },
        "reasons": reasons
    })
}

fn event_utility(event: &RetrievalEvent) -> f64 {
    let signal = event.memory_fragments as f64
        + event.code_hits as f64
        + event.code_memories as f64 * 1.25
        + event.graph_items as f64 * 0.35;
    let token_k = (event.estimated_tokens.max(250) as f64) / 1000.0;
    let latency_penalty = (event.latency_ms as f64 / 10_000.0).min(1.0);
    (signal / token_k) * (1.0 - latency_penalty * 0.25)
}

fn apply_self_heal_actions(
    store: &Store,
    request: &SemanticOperationRequest,
    compiler: &Value,
    conflict_graph: &Value,
) -> Result<Value> {
    let compiler_applied = compiler
        .get("applied")
        .and_then(Value::as_array)
        .map(|values| values.len())
        .unwrap_or(0);
    let conflict_invalidated = conflict_graph
        .get("invalidated")
        .and_then(Value::as_array)
        .map(|values| values.len())
        .unwrap_or(0);
    let audit_id = store.record_audit_event(
        &request.project_id,
        "semantic_ops",
        "dukememory_self_heal_apply_summary",
        "project",
        Some(&request.project_id),
        json!({
            "compiler_applied": compiler_applied,
            "conflict_invalidated": conflict_invalidated
        }),
    )?;
    Ok(json!({
        "audit_event_id": audit_id,
        "compiler_applied": compiler_applied,
        "conflict_invalidated": conflict_invalidated
    }))
}

fn classify_task_outcome(result: &Value, status: &str) -> String {
    if status == "failed" {
        return "negative".to_string();
    }
    let tests_passed = result
        .pointer("/tests/passed")
        .and_then(Value::as_bool)
        .or_else(|| result.get("tests_passed").and_then(Value::as_bool));
    let accepted = result
        .get("accepted")
        .and_then(Value::as_bool)
        .or_else(|| result.get("user_accepted").and_then(Value::as_bool));
    let failed = result
        .get("failed")
        .and_then(Value::as_bool)
        .or_else(|| result.get("bug_regression").and_then(Value::as_bool));
    if failed == Some(true) || tests_passed == Some(false) || accepted == Some(false) {
        "negative".to_string()
    } else if status == "completed" || tests_passed == Some(true) || accepted == Some(true) {
        "positive".to_string()
    } else {
        "unknown".to_string()
    }
}

fn fact_conflict_candidates(facts: &[MemoryFact]) -> Vec<Value> {
    let mut groups: HashMap<(Option<String>, String), Vec<&MemoryFact>> = HashMap::new();
    for fact in facts.iter().filter(|fact| fact.invalidated_by.is_none()) {
        groups
            .entry((fact.entity_id.clone(), fact.predicate.to_lowercase()))
            .or_default()
            .push(fact);
    }
    let mut conflicts = Vec::new();
    for ((_entity_id, predicate), group) in groups {
        for (left_index, left) in group.iter().enumerate() {
            for right in group.iter().skip(left_index + 1) {
                if normalized_fact_value(&left.value) == normalized_fact_value(&right.value) {
                    continue;
                }
                let (stronger, weaker) = if left.confidence >= right.confidence {
                    (*left, *right)
                } else {
                    (*right, *left)
                };
                conflicts.push(json!({
                    "predicate": predicate,
                    "left": compact_fact(left),
                    "right": compact_fact(right),
                    "stronger_fact_id": stronger.id,
                    "weaker_fact_id": weaker.id,
                    "confidence_delta": (stronger.confidence - weaker.confidence).abs(),
                    "suggested_action": if (stronger.confidence - weaker.confidence).abs() >= 0.15 {
                        "invalidate_weaker_fact"
                    } else {
                        "manual_review"
                    }
                }));
            }
        }
    }
    conflicts.sort_by(|left, right| {
        right["confidence_delta"]
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&left["confidence_delta"].as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    conflicts
}

fn compact_fact(fact: &MemoryFact) -> Value {
    json!({
        "id": fact.id,
        "entity_id": fact.entity_id,
        "memory_id": fact.memory_id,
        "predicate": fact.predicate,
        "value": fact.value,
        "confidence": fact.confidence,
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "invalidated_by": fact.invalidated_by
    })
}

fn normalized_fact_value(value: &str) -> String {
    value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn should_compile_to_core(memory: &Memory) -> bool {
    memory.status == "active"
        && memory.memory_tier != "core"
        && matches!(
            memory.kind.as_str(),
            "project_rule" | "constraint" | "architecture" | "decision"
        )
        && memory.importance >= 0.75
        && memory.confidence >= 0.75
        && memory.quality_score >= 0.60
        && memory.contradiction_risk <= 0.35
}

fn should_compile_archive(memory: &Memory) -> bool {
    memory.status == "active"
        && ((memory.memory_tier == "conversation" && memory.quality_score < 0.55)
            || memory.contradiction_risk >= 0.75
            || (memory.quality_score < 0.25 && memory.usage_count > 0))
}

fn lower_quality_memory<'a>(left: &'a Memory, right: &'a Memory) -> &'a Memory {
    let left_score = memory_compile_score(left);
    let right_score = memory_compile_score(right);
    if left_score <= right_score {
        left
    } else {
        right
    }
}

fn memory_compile_score(memory: &Memory) -> f64 {
    memory.quality_score * 0.45 + memory.importance * 0.25 + memory.confidence * 0.20
        - memory.contradiction_risk * 0.25
        + if memory.memory_tier == "core" {
            0.10
        } else {
            0.0
        }
}

fn split_memory_body(body: &str) -> Vec<String> {
    let paragraphs = body
        .split("\n\n")
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in paragraphs {
        if !current.is_empty() && current.len() + paragraph.len() > 900 {
            chunks.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    if chunks.len() <= 1 {
        let mut fallback = Vec::new();
        let mut current = String::new();
        for ch in body.chars() {
            current.push(ch);
            if current.chars().count() >= 900 {
                fallback.push(current.trim().to_string());
                current.clear();
            }
        }
        if !current.trim().is_empty() {
            fallback.push(current.trim().to_string());
        }
        fallback
    } else {
        chunks
    }
}

fn policy_trial_score(query: &str, memories: &[Memory], code_hits: usize) -> f64 {
    if memories.is_empty() && code_hits == 0 {
        return 0.0;
    }
    let relevance = if memories.is_empty() {
        0.0
    } else {
        memories
            .iter()
            .map(|memory| token_overlap(query, &memory.body))
            .sum::<f64>()
            / memories.len() as f64
    };
    let quality = if memories.is_empty() {
        0.0
    } else {
        memories.iter().map(memory_compile_score).sum::<f64>() / memories.len() as f64
    };
    let coverage =
        (memories.len() as f64 / 8.0).min(1.0) * 0.25 + (code_hits as f64 / 8.0).min(1.0) * 0.20;
    (relevance * 0.35 + quality * 0.45 + coverage).clamp(0.0, 1.0)
}

fn compact_retrieval_event(event: &RetrievalEvent) -> Value {
    json!({
        "id": event.id,
        "task_session_id": event.task_session_id,
        "tool": event.tool,
        "query": event.query,
        "task_type": event.task_type,
        "token_budget": event.token_budget,
        "estimated_tokens": event.estimated_tokens,
        "latency_ms": event.latency_ms,
        "memory_fragments": event.memory_fragments,
        "code_hits": event.code_hits,
        "graph_items": event.graph_items,
        "code_memories": event.code_memories,
        "utility_per_1k_tokens": event_utility(event),
        "created_at": event.created_at,
        "warnings": event.audit.pointer("/warnings").cloned().unwrap_or(Value::Null)
    })
}

fn memory_ids_from_retrieval_event(event: &RetrievalEvent) -> Vec<String> {
    event
        .audit
        .get("memory_fragments")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|fragment| fragment.get("memory_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn code_symbol_ids_from_retrieval_event(event: &RetrievalEvent) -> Vec<String> {
    event
        .audit
        .get("code_hits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|hit| hit.get("symbol_id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn max_similarity_for_ids(similarities: &HashMap<(String, String), f64>, ids: &[String]) -> f64 {
    let id_set = ids.iter().collect::<HashSet<_>>();
    similarities
        .iter()
        .filter(|((left, right), _)| id_set.contains(left) && id_set.contains(right))
        .map(|(_, score)| *score)
        .fold(0.0, f64::max)
}

fn memory_code_edges(memories: &[Memory], symbols: &[crate::store::CodeSymbol]) -> Vec<Value> {
    let mut edges = Vec::new();
    for memory in memories {
        let body = memory.body.to_ascii_lowercase();
        for symbol in symbols {
            let name = symbol.name.to_ascii_lowercase();
            let file = symbol.file_path.to_ascii_lowercase();
            let (relation, confidence, evidence) = if !name.is_empty() && body.contains(&name) {
                ("mentions_symbol", 0.9, symbol.name.clone())
            } else if body.contains(&file) {
                ("mentions_file", 0.85, symbol.file_path.clone())
            } else if memory
                .tags
                .iter()
                .any(|tag| !tag.is_empty() && file.contains(&tag.to_ascii_lowercase()))
            {
                ("tag_matches_file", 0.55, memory.tags.join(","))
            } else if edges.len() < 100 {
                ("co_selected", 0.35, "same retrieval context".to_string())
            } else {
                continue;
            };
            edges.push(json!({
                "from_memory_id": memory.id,
                "to_symbol_id": symbol.id,
                "relation": relation,
                "confidence": confidence,
                "evidence": evidence,
                "file_path": symbol.file_path,
                "symbol_name": symbol.name
            }));
        }
    }
    edges
}

fn term_overlap(left: &HashSet<String>, right: &[String]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let right = right.iter().cloned().collect::<HashSet<_>>();
    let intersection = left.intersection(&right).count() as f64;
    let union = left.union(&right).count().max(1) as f64;
    intersection / union
}

fn compact_memories(memories: &[Memory]) -> Vec<Value> {
    memories.iter().map(compact_memory).collect()
}

fn compact_memory(memory: &Memory) -> Value {
    json!({
        "id": memory.id,
        "kind": memory.kind,
        "memory_tier": memory.memory_tier,
        "status": memory.status,
        "tags": memory.tags,
        "source": memory.source,
        "score": memory.score,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "body_preview": truncate(&memory.body, 220)
    })
}

fn compact_code_symbol(symbol: &crate::store::CodeSymbol) -> Value {
    json!({
        "id": symbol.id,
        "file_path": symbol.file_path,
        "language": symbol.language,
        "name": symbol.name,
        "kind": symbol.kind,
        "signature": symbol.signature,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line
    })
}

fn classify_relation(
    new_body: &str,
    old_body: &str,
    similarity: f64,
) -> (&'static str, &'static str, String) {
    if similarity >= 0.94 {
        return (
            "duplicate",
            "archive_or_skip_new",
            format!("semantic similarity is {similarity:.2}"),
        );
    }
    if has_conflict_markers(new_body, old_body) && similarity >= 0.72 {
        return (
            "possible_conflict",
            "manual_review",
            "nearby memory contains opposing lifecycle or policy language".to_string(),
        );
    }
    if similarity >= 0.84 {
        return (
            "possible_update",
            "consider_supersede",
            format!("semantic similarity is {similarity:.2} but text is not identical"),
        );
    }
    (
        "related",
        "keep_separate",
        format!("semantic similarity is {similarity:.2}"),
    )
}

fn has_conflict_markers(left: &str, right: &str) -> bool {
    let left = left.to_ascii_lowercase();
    let right = right.to_ascii_lowercase();
    let pairs = [
        ("must", "must not"),
        ("always", "never"),
        ("enable", "disable"),
        ("active", "archived"),
        ("pending", "active"),
        ("required", "forbidden"),
        ("true", "false"),
    ];
    pairs.iter().any(|(a, b)| {
        (left.contains(a) && right.contains(b)) || (left.contains(b) && right.contains(a))
    })
}

fn suggest_tags(memory: &Memory, examples: &[Memory]) -> Vec<String> {
    let mut scores = HashMap::<String, f64>::new();
    for tag in keyword_tags(&memory.body) {
        *scores.entry(tag).or_default() += 1.0;
    }
    for example in examples {
        if example.id == memory.id || example.tags.is_empty() {
            continue;
        }
        let overlap = token_overlap(&memory.body, &example.body);
        if overlap <= 0.0 {
            continue;
        }
        for tag in &example.tags {
            *scores.entry(tag.clone()).or_default() += overlap;
        }
    }
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.into_iter().take(6).map(|(tag, _)| tag).collect()
}

fn keyword_tags(body: &str) -> Vec<String> {
    let lower = body.to_ascii_lowercase();
    [
        (
            "retrieval",
            ["retrieval", "search", "context", "rerank"].as_slice(),
        ),
        (
            "embedding",
            ["embedding", "vector", "semantic", "pgvector"].as_slice(),
        ),
        (
            "code-index",
            ["code index", "symbol", "call graph", "lsif"].as_slice(),
        ),
        ("mcp", ["mcp", "tool", "stdio", "jsonrpc"].as_slice()),
        ("graph", ["graph", "entity", "edge", "fact"].as_slice()),
        (
            "testing",
            ["test", "eval", "smoke", "regression"].as_slice(),
        ),
        (
            "database",
            ["postgres", "schema", "migration", "sql"].as_slice(),
        ),
        (
            "security",
            ["secret", "credential", "safety", "isolation"].as_slice(),
        ),
    ]
    .into_iter()
    .filter_map(|(tag, needles)| {
        if needles.iter().any(|needle| lower.contains(needle)) {
            Some(tag.to_string())
        } else {
            None
        }
    })
    .collect()
}

fn eval_query_for_memory(memory: &Memory) -> String {
    let mut terms = memory.tags.clone();
    terms.extend(top_terms(&memory.body, 5));
    terms.sort();
    terms.dedup();
    if terms.is_empty() {
        memory.kind.clone()
    } else {
        terms.into_iter().take(8).collect::<Vec<_>>().join(" ")
    }
}

fn top_terms(text: &str, limit: usize) -> Vec<String> {
    let stop = [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "when",
        "then",
        "must",
        "should",
        "memory",
        "dukememory",
    ];
    let stop = stop.iter().copied().collect::<HashSet<_>>();
    let mut counts = HashMap::<String, usize>::new();
    for token in tokens(text) {
        if token.len() < 4 || stop.contains(token.as_str()) {
            continue;
        }
        *counts.entry(token).or_default() += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    ranked
        .into_iter()
        .take(limit)
        .map(|(term, _)| term)
        .collect()
}

fn token_overlap(left: &str, right: &str) -> f64 {
    let left = tokens(left).into_iter().collect::<HashSet<_>>();
    let right = tokens(right).into_iter().collect::<HashSet<_>>();
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let shared = left.intersection(&right).count();
    shared as f64 / left.len().min(right.len()) as f64
}

fn tokens(text: &str) -> Vec<String> {
    text.chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn dedupe_memories(memories: &mut Vec<Memory>) {
    let mut seen = HashSet::new();
    memories.retain(|memory| seen.insert(memory.id.clone()));
}

fn coverage(done: u64, total: u64) -> f64 {
    if total == 0 {
        1.0
    } else {
        done as f64 / total as f64
    }
}

async fn semantic_memory_search_with_model(
    config: &Config,
    store: &Store,
    request: &SemanticOperationRequest,
    query: &str,
    model: &str,
) -> Result<Vec<Memory>> {
    let ollama = ollama_from_config(config);
    let embedding = ollama.embed_with_model(model, query).await?;
    if embedding.len() != 4096 {
        bail!(
            "memory embedding dimension mismatch for model `{model}`: expected 4096, got {}",
            embedding.len()
        );
    }
    store.search_memory_vectors(
        &request.project_id,
        VectorSearchOptions {
            embedding,
            embedding_model: model.to_string(),
            limit: request.limit,
            status: request.status,
            kind: request.kind.clone(),
            memory_tier: request.memory_tier.clone(),
        },
    )
}

fn memory_ids(memories: &[Memory]) -> HashSet<String> {
    memories.iter().map(|memory| memory.id.clone()).collect()
}

fn jaccard(left: &HashSet<String>, right: &HashSet<String>) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn set_difference(left: &HashSet<String>, right: &HashSet<String>) -> Vec<String> {
    let mut values = left.difference(right).cloned().collect::<Vec<_>>();
    values.sort();
    values
}

fn ranked_counts(counts: HashMap<String, usize>) -> Vec<Value> {
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked
        .into_iter()
        .map(|(name, count)| json!({ "name": name, "count": count }))
        .collect()
}

fn topic_drift_signals(memories: &[Memory]) -> Vec<Value> {
    let midpoint = memories.len() / 2;
    if midpoint == 0 {
        return Vec::new();
    }
    let recent = &memories[..midpoint];
    let older = &memories[midpoint..];
    let recent_tags = tag_counts(recent);
    let older_tags = tag_counts(older);
    let mut signals = Vec::new();
    for (tag, recent_count) in recent_tags {
        let older_count = older_tags.get(&tag).copied().unwrap_or(0);
        if recent_count >= older_count.saturating_mul(2).max(1) {
            signals.push(json!({
                "topic": tag,
                "signal": "growing",
                "recent": recent_count,
                "older": older_count
            }));
        }
    }
    signals
}

fn tag_counts(memories: &[Memory]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for memory in memories {
        for tag in &memory.tags {
            *counts.entry(tag.clone()).or_default() += 1;
        }
    }
    counts
}

fn complete_embedding_count_for_model(
    entries: &[crate::store::EmbeddingModelCount],
    model: &str,
    required_kinds: &[&str],
) -> u64 {
    required_kinds
        .iter()
        .map(|kind| {
            entries
                .iter()
                .find(|entry| entry.model == model && entry.embedding_kind == *kind)
                .map(|entry| entry.count)
                .unwrap_or(0)
        })
        .min()
        .unwrap_or(0)
}

fn similarity_or(value: f64, default: f64) -> f64 {
    if value > 0.0 { value } else { default }
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

fn ollama_from_config(config: &Config) -> OllamaClient {
    OllamaClient::new(
        config.ollama_base_url.clone(),
        config.extract_model().to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::DEFAULT_MEMORY_TIER;

    fn memory(id: &str, body: &str, score: f64) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: "project".to_string(),
            scope: "project".to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: "note".to_string(),
            body: body.to_string(),
            tags: Vec::new(),
            source: None,
            status: "active".to_string(),
            importance: 0.5,
            confidence: 0.8,
            superseded_by: None,
            status_reason: None,
            score: Some(score),
            quality_score: 0.0,
            usage_count: 0,
            last_used_at: None,
            contradiction_risk: 0.0,
            created_at: "now".to_string(),
            updated_at: "now".to_string(),
        }
    }

    fn retrieval_event(id: &str, memory_fragments: usize, code_hits: usize) -> RetrievalEvent {
        RetrievalEvent {
            id: id.to_string(),
            project_id: "project".to_string(),
            task_session_id: None,
            tool: "dukememory_prepare".to_string(),
            query: "code graph memory".to_string(),
            task_type: "feature".to_string(),
            token_budget: 6000,
            estimated_tokens: 1200,
            latency_ms: 100,
            memory_fragments,
            code_hits,
            graph_items: 2,
            code_memories: 1,
            plan: json!({}),
            audit: json!({
                "memory_fragments": [{"memory_id": "mem-a"}, {"memory_id": "mem-b"}],
                "code_hits": [{"symbol_id": "sym-a"}]
            }),
            created_at: "now".to_string(),
        }
    }

    fn code_symbol(id: &str, name: &str, file_path: &str) -> crate::store::CodeSymbol {
        crate::store::CodeSymbol {
            id: id.to_string(),
            project_id: "project".to_string(),
            file_path: file_path.to_string(),
            language: "rust".to_string(),
            name: name.to_string(),
            kind: "function".to_string(),
            signature: format!("fn {name}()"),
            body: String::new(),
            start_line: 1,
            end_line: 3,
            parent_id: None,
        }
    }

    #[test]
    fn mmr_penalizes_redundant_memories() {
        let memories = vec![
            memory("a", "retrieval alpha", 0.95),
            memory("b", "retrieval alpha duplicate", 0.94),
            memory("c", "graph beta", 0.70),
        ];
        let similarities = HashMap::from([
            (("a".to_string(), "b".to_string()), 0.99),
            (("a".to_string(), "c".to_string()), 0.05),
            (("b".to_string(), "c".to_string()), 0.05),
        ]);
        let selected = mmr_select(memories, &similarities, 2);
        let ids = selected
            .into_iter()
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["a", "c"]);
    }

    #[test]
    fn relation_classifier_marks_close_duplicates() {
        let (class, action, _) =
            classify_relation("Use semantic dedupe.", "Use semantic dedupe.", 0.98);
        assert_eq!(class, "duplicate");
        assert_eq!(action, "archive_or_skip_new");
    }

    #[test]
    fn retrieval_event_extracts_trace_ids() {
        let event = retrieval_event("event-a", 2, 1);
        assert_eq!(
            memory_ids_from_retrieval_event(&event),
            vec!["mem-a".to_string(), "mem-b".to_string()]
        );
        assert_eq!(
            code_symbol_ids_from_retrieval_event(&event),
            vec!["sym-a".to_string()]
        );
        assert!(event_utility(&event) > 0.0);
    }

    #[test]
    fn learned_policy_widens_code_heavy_history() {
        let base = plan_context_access(ContextPlanRequest {
            query: "implement code graph feature",
            memory_limit: 6,
            core_memory_limit: 3,
            code_limit: 6,
            token_budget: 4000,
        });
        let policy = learned_policy_from_events(
            &base,
            &[
                retrieval_event("event-a", 4, 14),
                retrieval_event("event-b", 3, 12),
            ],
            Some(0.9),
        );
        assert!(policy["code_limit"].as_u64().unwrap() > base.code_limit as u64);
    }

    #[test]
    fn memory_code_edges_prefers_direct_mentions() {
        let memories = vec![memory(
            "mem-a",
            "The retrieval policy calls build_context_payload in src/mcp.rs.",
            0.9,
        )];
        let symbols = vec![code_symbol("sym-a", "build_context_payload", "src/mcp.rs")];
        let edges = memory_code_edges(&memories, &symbols);
        assert_eq!(edges[0]["relation"], "mentions_symbol");
        assert!(edges[0]["confidence"].as_f64().unwrap() >= 0.85);
    }
}

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;

use crate::config::Config;
use crate::ollama::OllamaClient;
use crate::retrieval::RetrievalMode;
use crate::store::{
    CodeSearchOptions, CodeSearchResult, CodeVectorSearchOptions, Memory, SearchOptions,
    StatusFilter, Store, VectorSearchOptions,
};

pub struct MemorySearchRequest<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub limit: usize,
    pub status: StatusFilter,
    pub kind: Option<String>,
    pub memory_tier: Option<String>,
    pub mode: RetrievalMode,
    pub allow_hybrid_fallback: bool,
}

pub struct CodeSearchRequest<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub limit: usize,
    pub kind: Option<String>,
    pub file_path: Option<String>,
    pub mode: RetrievalMode,
    pub allow_hybrid_fallback: bool,
}

pub struct SearchReport<T> {
    pub results: Vec<T>,
    pub actual_mode: String,
    pub warning: Option<String>,
}

pub async fn search_memories(
    config: &Config,
    store: &Store,
    request: MemorySearchRequest<'_>,
) -> Result<Vec<Memory>> {
    Ok(search_memories_with_mode(config, store, request)
        .await?
        .results)
}

pub async fn search_memories_with_mode(
    config: &Config,
    store: &Store,
    request: MemorySearchRequest<'_>,
) -> Result<SearchReport<Memory>> {
    let keyword = if request.mode.uses_keyword() {
        store.search(
            request.project_id,
            SearchOptions {
                query: request.query.to_string(),
                limit: request.limit.saturating_mul(2).clamp(1, 100),
                status: request.status,
                kind: request.kind.clone(),
                memory_tier: request.memory_tier.clone(),
            },
        )?
    } else {
        Vec::new()
    };

    let semantic = if request.mode.uses_semantic() {
        let ollama = ollama_from_config(config);
        match ollama
            .embed_with_model(config.memory_embed_model(), request.query)
            .await
        {
            Ok(embedding) => {
                if embedding.len() != 4096 {
                    let warning = format!(
                        "semantic memory search skipped: embedding_dimension_mismatch expected=4096 actual={}",
                        embedding.len()
                    );
                    if request.mode.allows_keyword_fallback() && request.allow_hybrid_fallback {
                        return Ok(SearchReport {
                            results: keyword
                                .into_iter()
                                .take(request.limit.clamp(1, 50))
                                .collect(),
                            actual_mode: "keyword".to_string(),
                            warning: Some(warning),
                        });
                    }
                    bail!("{warning}");
                }
                store.search_memory_vectors(
                    request.project_id,
                    VectorSearchOptions {
                        embedding,
                        embedding_model: config.memory_embed_model().to_string(),
                        limit: request.limit.saturating_mul(2).clamp(1, 100),
                        status: request.status,
                        kind: request.kind,
                        memory_tier: request.memory_tier,
                    },
                )?
            }
            Err(error)
                if request.mode.allows_keyword_fallback() && request.allow_hybrid_fallback =>
            {
                return Ok(SearchReport {
                    results: keyword
                        .into_iter()
                        .take(request.limit.clamp(1, 50))
                        .collect(),
                    actual_mode: "keyword".to_string(),
                    warning: Some(format!("semantic memory search skipped: {error}")),
                });
            }
            Err(error) => return Err(error),
        }
    } else {
        Vec::new()
    };

    let results = match request.mode {
        RetrievalMode::Keyword => keyword
            .into_iter()
            .take(request.limit.clamp(1, 50))
            .collect(),
        RetrievalMode::Semantic => semantic
            .into_iter()
            .take(request.limit.clamp(1, 50))
            .collect(),
        RetrievalMode::Hybrid => merge_memories(keyword, semantic, request.limit),
        RetrievalMode::Rerank => {
            let candidates = merge_memories(keyword, semantic, request.limit.saturating_mul(3));
            rerank_memories(config, request.query, candidates, request.limit).await?
        }
    };
    Ok(SearchReport {
        results,
        actual_mode: request.mode.as_str().to_string(),
        warning: None,
    })
}

pub async fn search_memories_tuple(
    config: &Config,
    store: &Store,
    request: MemorySearchRequest<'_>,
) -> Result<(Vec<Memory>, String, Option<String>)> {
    let report = search_memories_with_mode(config, store, request).await?;
    Ok((report.results, report.actual_mode, report.warning))
}

pub async fn search_code(
    config: &Config,
    store: &Store,
    request: CodeSearchRequest<'_>,
) -> Result<Vec<CodeSearchResult>> {
    Ok(search_code_with_mode(config, store, request).await?.results)
}

pub async fn search_code_with_mode(
    config: &Config,
    store: &Store,
    request: CodeSearchRequest<'_>,
) -> Result<SearchReport<CodeSearchResult>> {
    let keyword = if request.mode.uses_keyword() {
        store.search_code(
            request.project_id,
            CodeSearchOptions {
                query: request.query.to_string(),
                limit: request.limit.saturating_mul(2).clamp(1, 100),
                kind: request.kind.clone(),
                file_path: request.file_path.clone(),
            },
        )?
    } else {
        Vec::new()
    };

    let semantic = if request.mode.uses_semantic() {
        let ollama = ollama_from_config(config);
        match ollama
            .embed_with_model(config.code_embed_model(), request.query)
            .await
        {
            Ok(embedding) => {
                if embedding.len() != 1024 {
                    let warning = format!(
                        "semantic code search skipped: embedding_dimension_mismatch expected=1024 actual={}",
                        embedding.len()
                    );
                    if request.mode.allows_keyword_fallback() && request.allow_hybrid_fallback {
                        return Ok(SearchReport {
                            results: keyword
                                .into_iter()
                                .take(request.limit.clamp(1, 50))
                                .collect(),
                            actual_mode: "keyword".to_string(),
                            warning: Some(warning),
                        });
                    }
                    bail!("{warning}");
                }
                store.search_code_vectors(
                    request.project_id,
                    CodeVectorSearchOptions {
                        embedding,
                        embedding_model: config.code_embed_model().to_string(),
                        limit: request.limit.saturating_mul(2).clamp(1, 100),
                        kind: request.kind,
                        file_path: request.file_path,
                    },
                )?
            }
            Err(error)
                if request.mode.allows_keyword_fallback() && request.allow_hybrid_fallback =>
            {
                return Ok(SearchReport {
                    results: keyword
                        .into_iter()
                        .take(request.limit.clamp(1, 50))
                        .collect(),
                    actual_mode: "keyword".to_string(),
                    warning: Some(format!("semantic code search skipped: {error}")),
                });
            }
            Err(error) => return Err(error),
        }
    } else {
        Vec::new()
    };

    let results = match request.mode {
        RetrievalMode::Keyword => keyword
            .into_iter()
            .take(request.limit.clamp(1, 50))
            .collect(),
        RetrievalMode::Semantic => semantic
            .into_iter()
            .take(request.limit.clamp(1, 50))
            .collect(),
        RetrievalMode::Hybrid | RetrievalMode::Rerank => {
            merge_code_results(keyword, semantic, request.limit)
        }
    };
    Ok(SearchReport {
        results,
        actual_mode: request.mode.as_str().to_string(),
        warning: None,
    })
}

pub async fn search_code_tuple(
    config: &Config,
    store: &Store,
    request: CodeSearchRequest<'_>,
) -> Result<(Vec<CodeSearchResult>, String, Option<String>)> {
    let report = search_code_with_mode(config, store, request).await?;
    Ok((report.results, report.actual_mode, report.warning))
}

pub fn ollama_from_config(config: &Config) -> OllamaClient {
    OllamaClient::new(
        config.ollama_base_url.clone(),
        config.extract_model().to_string(),
    )
}

pub fn code_model_for_role<'a>(config: &'a Config, role: &str) -> Result<&'a str> {
    match role {
        "fast" | "fast_code" => Ok(&config.fast_code_model),
        "deep" | "deep_code" => Ok(&config.deep_code_model),
        "agent" | "agent_code" => Ok(&config.agent_code_model),
        "experiment" => Ok(&config.experiment_model),
        other => Err(anyhow!(
            "invalid code model role `{other}`; use fast_code, deep_code, agent_code, or experiment"
        )),
    }
}

trait RetrievalModeExt {
    fn uses_keyword(self) -> bool;
    fn uses_semantic(self) -> bool;
    fn allows_keyword_fallback(self) -> bool;
}

impl RetrievalModeExt for RetrievalMode {
    fn uses_keyword(self) -> bool {
        matches!(
            self,
            RetrievalMode::Keyword | RetrievalMode::Hybrid | RetrievalMode::Rerank
        )
    }

    fn uses_semantic(self) -> bool {
        matches!(
            self,
            RetrievalMode::Semantic | RetrievalMode::Hybrid | RetrievalMode::Rerank
        )
    }

    fn allows_keyword_fallback(self) -> bool {
        matches!(self, RetrievalMode::Hybrid | RetrievalMode::Rerank)
    }
}

fn merge_memories(keyword: Vec<Memory>, semantic: Vec<Memory>, limit: usize) -> Vec<Memory> {
    let mut ranked: HashMap<String, (f64, Memory)> = HashMap::new();
    for (rank, memory) in keyword.into_iter().enumerate() {
        add_memory_rank(&mut ranked, memory, rank);
    }
    for (rank, memory) in semantic.into_iter().enumerate() {
        add_memory_rank(&mut ranked, memory, rank);
    }

    let mut results = ranked
        .into_values()
        .map(|(score, mut memory)| {
            memory.score = Some(memory_quality_adjusted_score(score, &memory));
            memory
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit.clamp(1, 50));
    results
}

fn add_memory_rank(ranked: &mut HashMap<String, (f64, Memory)>, memory: Memory, rank: usize) {
    let entry = ranked
        .entry(memory.id.clone())
        .or_insert_with(|| (0.0, memory));
    entry.0 += reciprocal_rank(rank);
}

fn memory_quality_adjusted_score(score: f64, memory: &Memory) -> f64 {
    let quality_multiplier = 0.75 + (memory.quality_score.clamp(0.0, 1.0) * 0.5);
    let risk_multiplier = 1.0 - (memory.contradiction_risk.clamp(0.0, 1.0) * 0.35);
    score * quality_multiplier * risk_multiplier
}

fn merge_code_results(
    keyword: Vec<CodeSearchResult>,
    semantic: Vec<CodeSearchResult>,
    limit: usize,
) -> Vec<CodeSearchResult> {
    let mut ranked: HashMap<String, (f64, CodeSearchResult)> = HashMap::new();
    for (rank, result) in keyword.into_iter().enumerate() {
        add_code_rank(&mut ranked, result, rank);
    }
    for (rank, result) in semantic.into_iter().enumerate() {
        add_code_rank(&mut ranked, result, rank);
    }

    let mut results = ranked
        .into_values()
        .map(|(score, mut result)| {
            result.score = score;
            result
        })
        .collect::<Vec<_>>();
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit.clamp(1, 50));
    results
}

fn add_code_rank(
    ranked: &mut HashMap<String, (f64, CodeSearchResult)>,
    result: CodeSearchResult,
    rank: usize,
) {
    let entry = ranked
        .entry(result.symbol.id.clone())
        .or_insert_with(|| (0.0, result));
    entry.0 += reciprocal_rank(rank);
}

fn reciprocal_rank(rank: usize) -> f64 {
    1.0 / (60.0 + rank as f64 + 1.0)
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    ranked_ids: Vec<String>,
}

async fn rerank_memories(
    config: &Config,
    query: &str,
    candidates: Vec<Memory>,
    limit: usize,
) -> Result<Vec<Memory>> {
    if candidates.len() <= 1 {
        return Ok(candidates);
    }
    let candidate_text = candidates
        .iter()
        .enumerate()
        .map(|(index, memory)| {
            format!(
                "{}. id={} kind={} importance={:.2} confidence={:.2} quality={:.2} contradiction_risk={:.2}\n{}",
                index + 1,
                memory.id,
                memory.kind,
                memory.importance,
                memory.confidence,
                memory.quality_score,
                memory.contradiction_risk,
                truncate_text(&memory.body, 700)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let system = "You rerank project memory search results. Return strict JSON only: {\"ranked_ids\":[\"id\",...]}. Include only ids from the candidate list, most useful first.";
    let user = format!("Query:\n{query}\n\nCandidates:\n{candidate_text}");
    let ollama = ollama_from_config(config);
    let response = ollama
        .chat_json_with_model(&config.validate_model, system, &user)
        .await;
    let Ok(response) = response else {
        return Ok(candidates.into_iter().take(limit.clamp(1, 50)).collect());
    };
    let parsed = serde_json::from_str::<RerankResponse>(&response);
    let Ok(parsed) = parsed else {
        return Ok(candidates.into_iter().take(limit.clamp(1, 50)).collect());
    };
    let mut by_id = candidates
        .into_iter()
        .map(|memory| (memory.id.clone(), memory))
        .collect::<HashMap<_, _>>();
    let mut reranked = Vec::new();
    for id in parsed.ranked_ids {
        if let Some(memory) = by_id.remove(&id) {
            reranked.push(memory);
        }
    }
    reranked.extend(by_id.into_values());
    Ok(reranked.into_iter().take(limit.clamp(1, 50)).collect())
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, anyhow, bail};
use serde::Serialize;

use crate::config::Config;
use crate::ollama::OllamaClient;
use crate::semantic_ops::diversify_memories_by_embedding;
use crate::store::{
    CodeSearchOptions, CodeSearchResult, CodeVectorSearchOptions, Memory, MemoryStatus,
    SearchOptions, StatusFilter, Store, VectorSearchOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    Keyword,
    Semantic,
    Hybrid,
    Rerank,
}

impl RetrievalMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "keyword" => Ok(Self::Keyword),
            "semantic" => Ok(Self::Semantic),
            "hybrid" => Ok(Self::Hybrid),
            "rerank" => Ok(Self::Rerank),
            other => Err(anyhow!(
                "invalid search mode `{other}`; use keyword, semantic, hybrid, or rerank"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keyword => "keyword",
            Self::Semantic => "semantic",
            Self::Hybrid => "hybrid",
            Self::Rerank => "rerank",
        }
    }

    fn uses_keyword(self) -> bool {
        matches!(self, Self::Keyword | Self::Hybrid | Self::Rerank)
    }

    fn uses_semantic(self) -> bool {
        matches!(self, Self::Semantic | Self::Hybrid | Self::Rerank)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    Code,
    Memory,
    Architecture,
    Debugging,
    Mixed,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalPlan {
    pub intent: QueryIntent,
    pub memory_weight: f64,
    pub code_weight: f64,
    pub graph_weight: f64,
    pub keyword_weight: f64,
    pub semantic_weight: f64,
    pub symbol_weight: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalSourceDiagnostic {
    pub domain: String,
    pub source: String,
    pub requested: bool,
    pub returned: usize,
    pub weight: f64,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalExplanation {
    pub domain: String,
    pub id: String,
    pub score: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalDiagnostics {
    pub requested_mode: String,
    pub memory_actual_mode: String,
    pub code_actual_mode: String,
    pub intent: QueryIntent,
    pub plan: RetrievalPlan,
    pub sources: Vec<RetrievalSourceDiagnostic>,
    pub warnings: Vec<String>,
    pub memory_candidates: usize,
    pub code_candidates: usize,
    pub reranker: String,
    pub explanations: Vec<RetrievalExplanation>,
}

pub struct ContextRetrievalRequest<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub memory_limit: usize,
    pub code_limit: usize,
    pub mode: RetrievalMode,
    pub allow_hybrid_fallback: bool,
}

pub struct ContextRetrievalOutput {
    pub memories: Vec<Memory>,
    pub code: Vec<CodeSearchResult>,
    pub diagnostics: RetrievalDiagnostics,
}

struct MemoryPipelineOutput {
    results: Vec<Memory>,
    actual_mode: String,
    candidates: usize,
    sources: Vec<RetrievalSourceDiagnostic>,
    warnings: Vec<String>,
}

struct CodePipelineOutput {
    results: Vec<CodeSearchResult>,
    actual_mode: String,
    candidates: usize,
    sources: Vec<RetrievalSourceDiagnostic>,
    warnings: Vec<String>,
}

pub async fn run_context_retrieval(
    config: &Config,
    store: &Store,
    request: ContextRetrievalRequest<'_>,
) -> Result<ContextRetrievalOutput> {
    let plan = build_retrieval_plan(request.query);
    let memory = if request.memory_limit == 0 {
        MemoryPipelineOutput {
            results: Vec::new(),
            actual_mode: "disabled".to_string(),
            candidates: 0,
            sources: Vec::new(),
            warnings: Vec::new(),
        }
    } else {
        search_memory_pipeline(config, store, &request, &plan).await?
    };
    let code = if request.code_limit == 0 {
        CodePipelineOutput {
            results: Vec::new(),
            actual_mode: "disabled".to_string(),
            candidates: 0,
            sources: Vec::new(),
            warnings: Vec::new(),
        }
    } else {
        search_code_pipeline(config, store, &request, &plan).await?
    };

    let mut sources = memory.sources;
    sources.extend(code.sources);
    let mut warnings = memory.warnings;
    warnings.extend(code.warnings);
    let explanations = retrieval_explanations(
        &memory.results,
        &code.results,
        &memory.actual_mode,
        &code.actual_mode,
    );
    let diagnostics = RetrievalDiagnostics {
        requested_mode: request.mode.as_str().to_string(),
        memory_actual_mode: memory.actual_mode,
        code_actual_mode: code.actual_mode,
        intent: plan.intent,
        plan,
        sources,
        warnings,
        memory_candidates: memory.candidates,
        code_candidates: code.candidates,
        reranker: "deterministic_rrf_lexical_importance".to_string(),
        explanations,
    };

    Ok(ContextRetrievalOutput {
        memories: memory.results,
        code: code.results,
        diagnostics,
    })
}

pub fn format_retrieval_diagnostics(diagnostics: &RetrievalDiagnostics) -> String {
    let mut text = String::new();
    text.push_str("\nRETRIEVAL PIPELINE\n");
    text.push_str(&format!(
        "- mode={} memory_actual={} code_actual={} intent={:?} reranker={}\n",
        diagnostics.requested_mode,
        diagnostics.memory_actual_mode,
        diagnostics.code_actual_mode,
        diagnostics.intent,
        diagnostics.reranker
    ));
    text.push_str(&format!(
        "- weights: memory={:.2} code={:.2} graph={:.2} keyword={:.2} semantic={:.2} symbol={:.2}\n",
        diagnostics.plan.memory_weight,
        diagnostics.plan.code_weight,
        diagnostics.plan.graph_weight,
        diagnostics.plan.keyword_weight,
        diagnostics.plan.semantic_weight,
        diagnostics.plan.symbol_weight
    ));
    for source in &diagnostics.sources {
        text.push_str(&format!(
            "- {}:{} requested={} returned={} weight={:.2}",
            source.domain, source.source, source.requested, source.returned, source.weight
        ));
        if let Some(warning) = &source.warning {
            text.push_str(&format!(" warning={warning}"));
        }
        text.push('\n');
    }
    if !diagnostics.warnings.is_empty() {
        text.push_str("WARNINGS\n");
        for warning in &diagnostics.warnings {
            text.push_str(&format!("- {warning}\n"));
        }
    }
    text
}

fn retrieval_explanations(
    memories: &[Memory],
    code: &[CodeSearchResult],
    memory_actual_mode: &str,
    code_actual_mode: &str,
) -> Vec<RetrievalExplanation> {
    let mut explanations = Vec::new();
    explanations.extend(memories.iter().take(10).map(|memory| RetrievalExplanation {
        domain: "memory".to_string(),
        id: memory.id.clone(),
        score: memory.score.unwrap_or(0.0),
        reason: format!(
            "selected by {memory_actual_mode} memory retrieval; kind={} importance={:.2} confidence={:.2} quality={:.2} contradiction_risk={:.2}",
            memory.kind, memory.importance, memory.confidence, memory.quality_score, memory.contradiction_risk
        ),
    }));
    explanations.extend(code.iter().take(10).map(|result| RetrievalExplanation {
        domain: "code".to_string(),
        id: result.symbol.id.clone(),
        score: result.score,
        reason: format!(
            "selected by {code_actual_mode} code retrieval; symbol={} kind={} file={}",
            result.symbol.name, result.symbol.kind, result.symbol.file_path
        ),
    }));
    explanations
}

fn semantic_query_variants(query: &str, intent: QueryIntent) -> Vec<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut variants = vec![trimmed.to_string()];
    match intent {
        QueryIntent::Code | QueryIntent::Debugging => {
            variants.push(format!("code symbols implementation {trimmed}"));
            variants.push(format!("callers callees tests {trimmed}"));
        }
        QueryIntent::Memory => {
            variants.push(format!("project memory decision rule {trimmed}"));
        }
        QueryIntent::Architecture | QueryIntent::Mixed => {
            variants.push(format!("architecture workflow components {trimmed}"));
            variants.push(format!("project memory and code {trimmed}"));
        }
    }
    variants.sort();
    variants.dedup();
    variants.truncate(3);
    variants
}

pub fn build_retrieval_plan(query: &str) -> RetrievalPlan {
    let normalized = query.to_lowercase();
    let has_code = contains_any(
        &normalized,
        &[
            "code",
            "код",
            "файл",
            "symbol",
            "символ",
            "function",
            "функц",
            "struct",
            "trait",
            "impl",
            "call",
            "caller",
            "callee",
            "вызов",
            "grep",
            "ast",
            "scip",
        ],
    );
    let has_memory = contains_any(
        &normalized,
        &[
            "memory",
            "памят",
            "решени",
            "decision",
            "истори",
            "контекст",
            "remember",
            "факт",
        ],
    );
    let has_architecture = contains_any(
        &normalized,
        &[
            "architecture",
            "архитект",
            "pipeline",
            "graph",
            "граф",
            "связ",
            "зависим",
            "retrieval",
            "rag",
        ],
    );
    let has_debugging = contains_any(
        &normalized,
        &[
            "bug",
            "debug",
            "ошиб",
            "слом",
            "test",
            "тест",
            "panic",
            "failure",
            "regression",
        ],
    );
    let intent = match (has_code, has_memory, has_architecture, has_debugging) {
        (true, false, false, false) => QueryIntent::Code,
        (false, true, false, false) => QueryIntent::Memory,
        (_, _, true, false) => QueryIntent::Architecture,
        (_, _, _, true) => QueryIntent::Debugging,
        (true, true, false, false) => QueryIntent::Code,
        _ => QueryIntent::Mixed,
    };

    let mut plan = RetrievalPlan {
        intent,
        memory_weight: 1.0,
        code_weight: 1.0,
        graph_weight: 1.0,
        keyword_weight: 1.0,
        semantic_weight: 1.0,
        symbol_weight: 1.0,
        notes: Vec::new(),
    };
    match intent {
        QueryIntent::Code => {
            plan.code_weight = 1.35;
            plan.symbol_weight = 1.35;
            plan.keyword_weight = 1.15;
            plan.notes.push("code-oriented query".to_string());
        }
        QueryIntent::Memory => {
            plan.memory_weight = 1.35;
            plan.semantic_weight = 1.15;
            plan.notes.push("memory-oriented query".to_string());
        }
        QueryIntent::Architecture => {
            plan.memory_weight = 1.15;
            plan.code_weight = 1.15;
            plan.graph_weight = 1.35;
            plan.semantic_weight = 1.1;
            plan.notes.push("architecture/graph query".to_string());
        }
        QueryIntent::Debugging => {
            plan.code_weight = 1.25;
            plan.keyword_weight = 1.2;
            plan.symbol_weight = 1.2;
            plan.notes.push("debugging query".to_string());
        }
        QueryIntent::Mixed => {
            plan.graph_weight = 1.1;
            plan.notes.push("mixed query".to_string());
        }
    }
    plan
}

async fn search_memory_pipeline(
    config: &Config,
    store: &Store,
    request: &ContextRetrievalRequest<'_>,
    plan: &RetrievalPlan,
) -> Result<MemoryPipelineOutput> {
    let mut sources = Vec::new();
    let mut warnings = Vec::new();
    let keyword_requested = request.mode.uses_keyword();
    let semantic_requested = request.mode.uses_semantic();
    let graph_requested = !request.query.trim().is_empty();
    let keyword_limit = request.memory_limit.saturating_mul(3).clamp(1, 100);
    let semantic_limit = request.memory_limit.saturating_mul(3).clamp(1, 100);
    let graph_limit = request.memory_limit.saturating_mul(3).clamp(1, 100);

    let keyword = if keyword_requested {
        store.search(
            request.project_id,
            SearchOptions {
                query: request.query.to_string(),
                limit: keyword_limit,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
            },
        )?
    } else {
        Vec::new()
    };
    sources.push(RetrievalSourceDiagnostic {
        domain: "memory".to_string(),
        source: "keyword".to_string(),
        requested: keyword_requested,
        returned: keyword.len(),
        weight: plan.keyword_weight * plan.memory_weight,
        warning: None,
    });

    let query_variants = semantic_query_variants(request.query, plan.intent);
    let mut semantic_warning = None;
    let semantic = if semantic_requested {
        let ollama = ollama_from_config(config);
        let mut variant_results = Vec::new();
        for variant in &query_variants {
            match ollama
                .embed_with_model(config.memory_embed_model(), variant)
                .await
            {
                Ok(embedding) => {
                    if embedding.len() != 4096 {
                        let warning = format!(
                            "semantic memory search skipped: embedding_dimension_mismatch expected=4096 actual={}",
                            embedding.len()
                        );
                        if matches!(request.mode, RetrievalMode::Hybrid | RetrievalMode::Rerank)
                            && request.allow_hybrid_fallback
                        {
                            semantic_warning = Some(warning);
                            variant_results.clear();
                            break;
                        } else {
                            bail!("{warning}");
                        }
                    }
                    variant_results.push(store.search_memory_vectors(
                        request.project_id,
                        VectorSearchOptions {
                            embedding,
                            embedding_model: config.memory_embed_model().to_string(),
                            limit: semantic_limit,
                            status: StatusFilter::One(MemoryStatus::Active),
                            kind: None,
                            memory_tier: None,
                        },
                    )?);
                }
                Err(error)
                    if matches!(request.mode, RetrievalMode::Hybrid | RetrievalMode::Rerank)
                        && request.allow_hybrid_fallback =>
                {
                    semantic_warning = Some(format!("semantic memory search skipped: {error}"));
                    variant_results.clear();
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        merge_semantic_memory_variants(variant_results, semantic_limit)
    } else {
        Vec::new()
    };
    if let Some(warning) = &semantic_warning {
        warnings.push(warning.clone());
    }
    sources.push(RetrievalSourceDiagnostic {
        domain: "memory".to_string(),
        source: "semantic".to_string(),
        requested: semantic_requested,
        returned: semantic.len(),
        weight: plan.semantic_weight * plan.memory_weight,
        warning: semantic_warning.clone(),
    });
    sources.push(RetrievalSourceDiagnostic {
        domain: "memory".to_string(),
        source: "semantic_decomposition".to_string(),
        requested: semantic_requested,
        returned: if semantic_requested {
            query_variants.len()
        } else {
            0
        },
        weight: plan.semantic_weight,
        warning: None,
    });

    let mut graph_warning = None;
    let graph = if graph_requested {
        match search_graph_memories(store, request.project_id, request.query, graph_limit) {
            Ok(results) => results,
            Err(error) => {
                graph_warning = Some(format!("memory graph search skipped: {error}"));
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    if let Some(warning) = &graph_warning {
        warnings.push(warning.clone());
    }
    sources.push(RetrievalSourceDiagnostic {
        domain: "memory".to_string(),
        source: "graph".to_string(),
        requested: graph_requested,
        returned: graph.len(),
        weight: plan.graph_weight * plan.memory_weight,
        warning: graph_warning.clone(),
    });

    let candidates = keyword.len() + semantic.len() + graph.len();
    let actual_mode = actual_mode_with_graph(
        request.mode,
        keyword_requested,
        semantic_requested,
        &semantic,
        graph_requested,
        &graph,
    );
    let mut results = match request.mode {
        RetrievalMode::Keyword => merge_memories_weighted(
            keyword.clone(),
            Vec::new(),
            graph.clone(),
            request.memory_limit.saturating_mul(3).clamp(1, 100),
            plan,
        ),
        RetrievalMode::Semantic => merge_memories_weighted(
            Vec::new(),
            semantic.clone(),
            graph.clone(),
            request.memory_limit.saturating_mul(3).clamp(1, 100),
            plan,
        ),
        RetrievalMode::Hybrid | RetrievalMode::Rerank => merge_memories_weighted(
            keyword.clone(),
            semantic.clone(),
            graph.clone(),
            request.memory_limit.saturating_mul(3).clamp(1, 100),
            plan,
        ),
    };
    if matches!(request.mode, RetrievalMode::Rerank) {
        rerank_memories_deterministic(request.query, &mut results, plan);
    }
    let result_limit = request.memory_limit.clamp(1, 50);
    let fallback_results = results.clone();
    results = match diversify_memories_by_embedding(
        store,
        request.project_id,
        config.memory_embed_model(),
        results,
        result_limit,
    ) {
        Ok(diverse) => diverse,
        Err(error) => {
            warnings.push(format!("memory diversity packing skipped: {error}"));
            let mut fallback = fallback_results;
            fallback.truncate(result_limit);
            fallback
        }
    };

    Ok(MemoryPipelineOutput {
        results,
        actual_mode,
        candidates,
        sources,
        warnings,
    })
}

async fn search_code_pipeline(
    config: &Config,
    store: &Store,
    request: &ContextRetrievalRequest<'_>,
    plan: &RetrievalPlan,
) -> Result<CodePipelineOutput> {
    let mut sources = Vec::new();
    let mut warnings = Vec::new();
    let keyword_requested = request.mode.uses_keyword();
    let semantic_requested = request.mode.uses_semantic();
    let keyword_limit = request.code_limit.saturating_mul(3).clamp(1, 100);
    let semantic_limit = request.code_limit.saturating_mul(3).clamp(1, 100);
    let graph_limit = request.code_limit.saturating_mul(4).clamp(1, 120);

    let keyword = if keyword_requested {
        store.search_code(
            request.project_id,
            CodeSearchOptions {
                query: request.query.to_string(),
                limit: keyword_limit,
                kind: None,
                file_path: None,
            },
        )?
    } else {
        Vec::new()
    };
    sources.push(RetrievalSourceDiagnostic {
        domain: "code".to_string(),
        source: "keyword".to_string(),
        requested: keyword_requested,
        returned: keyword.len(),
        weight: plan.keyword_weight * plan.code_weight * plan.symbol_weight,
        warning: None,
    });

    let query_variants = semantic_query_variants(request.query, plan.intent);
    let mut semantic_warning = None;
    let semantic = if semantic_requested {
        let ollama = ollama_from_config(config);
        let mut variant_results = Vec::new();
        for variant in &query_variants {
            match ollama
                .embed_with_model(config.code_embed_model(), variant)
                .await
            {
                Ok(embedding) => {
                    if embedding.len() != 1024 {
                        let warning = format!(
                            "semantic code search skipped: embedding_dimension_mismatch expected=1024 actual={}",
                            embedding.len()
                        );
                        if matches!(request.mode, RetrievalMode::Hybrid | RetrievalMode::Rerank)
                            && request.allow_hybrid_fallback
                        {
                            semantic_warning = Some(warning);
                            variant_results.clear();
                            break;
                        } else {
                            bail!("{warning}");
                        }
                    }
                    variant_results.push(store.search_code_vectors(
                        request.project_id,
                        CodeVectorSearchOptions {
                            embedding,
                            embedding_model: config.code_embed_model().to_string(),
                            limit: semantic_limit,
                            kind: None,
                            file_path: None,
                        },
                    )?);
                }
                Err(error)
                    if matches!(request.mode, RetrievalMode::Hybrid | RetrievalMode::Rerank)
                        && request.allow_hybrid_fallback =>
                {
                    semantic_warning = Some(format!("semantic code search skipped: {error}"));
                    variant_results.clear();
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        merge_semantic_code_variants(variant_results, semantic_limit)
    } else {
        Vec::new()
    };
    if let Some(warning) = &semantic_warning {
        warnings.push(warning.clone());
    }
    sources.push(RetrievalSourceDiagnostic {
        domain: "code".to_string(),
        source: "semantic".to_string(),
        requested: semantic_requested,
        returned: semantic.len(),
        weight: plan.semantic_weight * plan.code_weight,
        warning: semantic_warning.clone(),
    });
    sources.push(RetrievalSourceDiagnostic {
        domain: "code".to_string(),
        source: "semantic_decomposition".to_string(),
        requested: semantic_requested,
        returned: if semantic_requested {
            query_variants.len()
        } else {
            0
        },
        weight: plan.semantic_weight,
        warning: None,
    });

    let seed_results = keyword
        .iter()
        .chain(semantic.iter())
        .cloned()
        .collect::<Vec<_>>();
    let graph_requested = !seed_results.is_empty();
    let mut graph_warning = None;
    let graph = if graph_requested {
        match search_code_graph_neighbors(store, request.project_id, &seed_results, graph_limit) {
            Ok(results) => results,
            Err(error) => {
                graph_warning = Some(format!("code graph expansion skipped: {error}"));
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    if let Some(warning) = &graph_warning {
        warnings.push(warning.clone());
    }
    sources.push(RetrievalSourceDiagnostic {
        domain: "code".to_string(),
        source: "graph".to_string(),
        requested: graph_requested,
        returned: graph.len(),
        weight: plan.graph_weight * plan.code_weight,
        warning: graph_warning.clone(),
    });

    let candidates = keyword.len() + semantic.len() + graph.len();
    let actual_mode = actual_mode_with_graph(
        request.mode,
        keyword_requested,
        semantic_requested,
        &semantic,
        graph_requested,
        &graph,
    );
    let mut results = match request.mode {
        RetrievalMode::Keyword => merge_code_weighted(
            keyword,
            Vec::new(),
            graph,
            request.code_limit.saturating_mul(3).clamp(1, 100),
            plan,
        ),
        RetrievalMode::Semantic => merge_code_weighted(
            Vec::new(),
            semantic,
            graph,
            request.code_limit.saturating_mul(3).clamp(1, 100),
            plan,
        ),
        RetrievalMode::Hybrid | RetrievalMode::Rerank => merge_code_weighted(
            keyword,
            semantic,
            graph,
            request.code_limit.saturating_mul(3).clamp(1, 100),
            plan,
        ),
    };
    if matches!(request.mode, RetrievalMode::Rerank) {
        rerank_code_deterministic(request.query, &mut results, plan);
    }
    results.truncate(request.code_limit.clamp(1, 50));

    Ok(CodePipelineOutput {
        results,
        actual_mode,
        candidates,
        sources,
        warnings,
    })
}

fn actual_mode<T>(
    mode: RetrievalMode,
    keyword_requested: bool,
    semantic_requested: bool,
    semantic: &[T],
) -> String {
    if matches!(mode, RetrievalMode::Hybrid | RetrievalMode::Rerank)
        && keyword_requested
        && semantic_requested
        && semantic.is_empty()
    {
        "keyword".to_string()
    } else {
        mode.as_str().to_string()
    }
}

fn actual_mode_with_graph<T, U>(
    mode: RetrievalMode,
    keyword_requested: bool,
    semantic_requested: bool,
    semantic: &[T],
    graph_requested: bool,
    graph: &[U],
) -> String {
    let base = actual_mode(mode, keyword_requested, semantic_requested, semantic);
    if graph_requested && !graph.is_empty() {
        format!("{base}+graph")
    } else {
        base
    }
}

fn search_graph_memories(
    store: &Store,
    project_id: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<Memory>> {
    let graph = store.search_memory_graph(project_id, query, limit)?;
    let mut ranked = BTreeMap::<String, f64>::new();
    for (rank, fact) in graph.facts.iter().enumerate() {
        if let Some(memory_id) = &fact.memory_id {
            *ranked.entry(memory_id.clone()).or_default() +=
                reciprocal_rank(rank) * fact.confidence.max(0.1);
        }
    }
    for (rank, edge) in graph.edges.iter().enumerate() {
        if let Some(memory_id) = &edge.memory_id {
            *ranked.entry(memory_id.clone()).or_default() +=
                reciprocal_rank(rank) * edge.confidence.max(0.1);
        }
    }

    let mut ids = ranked.into_iter().collect::<Vec<_>>();
    ids.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut memories = Vec::new();
    for (id, score) in ids.into_iter().take(limit.clamp(1, 100)) {
        let Some(mut memory) = store.get(project_id, &id)? else {
            continue;
        };
        if memory.status != MemoryStatus::Active.as_str() {
            continue;
        }
        memory.score = Some(score);
        memories.push(memory);
    }
    Ok(memories)
}

fn search_code_graph_neighbors(
    store: &Store,
    project_id: &str,
    seeds: &[CodeSearchResult],
    limit: usize,
) -> Result<Vec<CodeSearchResult>> {
    let mut seed_ids = seeds
        .iter()
        .map(|result| result.symbol.id.clone())
        .collect::<Vec<_>>();
    seed_ids.sort();
    seed_ids.dedup();
    if seed_ids.is_empty() {
        return Ok(Vec::new());
    }

    let (symbols, relations) = store.code_graph_for_symbols(project_id, &seed_ids, limit)?;
    let mut relation_degree = HashMap::<String, usize>::new();
    for relation in &relations {
        if let Some(from_symbol_id) = &relation.from_symbol_id {
            *relation_degree.entry(from_symbol_id.clone()).or_default() += 1;
        }
        if let Some(target_symbol_id) = &relation.target_symbol_id {
            *relation_degree.entry(target_symbol_id.clone()).or_default() += 1;
        }
    }
    let seed_set = seed_ids.into_iter().collect::<HashSet<_>>();
    let mut results = symbols
        .into_iter()
        .enumerate()
        .map(|(rank, symbol)| {
            let degree = relation_degree.get(&symbol.id).copied().unwrap_or_default() as f64;
            let seed_boost = if seed_set.contains(&symbol.id) {
                0.012
            } else {
                0.0
            };
            CodeSearchResult {
                score: reciprocal_rank(rank) + degree * 0.002 + seed_boost,
                symbol,
            }
        })
        .collect::<Vec<_>>();
    sort_code(&mut results);
    results.truncate(limit.clamp(1, 120));
    Ok(results)
}

fn merge_memories_weighted(
    keyword: Vec<Memory>,
    semantic: Vec<Memory>,
    graph: Vec<Memory>,
    limit: usize,
    plan: &RetrievalPlan,
) -> Vec<Memory> {
    let mut ranked = HashMap::<String, (f64, Memory)>::new();
    for (rank, memory) in keyword.into_iter().enumerate() {
        add_memory_rank(
            &mut ranked,
            memory,
            rank,
            plan.keyword_weight * plan.memory_weight,
        );
    }
    for (rank, memory) in semantic.into_iter().enumerate() {
        add_memory_rank(
            &mut ranked,
            memory,
            rank,
            plan.semantic_weight * plan.memory_weight,
        );
    }
    for (rank, memory) in graph.into_iter().enumerate() {
        add_memory_rank(
            &mut ranked,
            memory,
            rank,
            plan.graph_weight * plan.memory_weight,
        );
    }
    let mut results = ranked
        .into_values()
        .map(|(score, mut memory)| {
            memory.score = Some(memory_quality_adjusted_score(score, &memory));
            memory
        })
        .collect::<Vec<_>>();
    sort_memories(&mut results);
    results.truncate(limit.clamp(1, 100));
    results
}

fn merge_semantic_memory_variants(variants: Vec<Vec<Memory>>, limit: usize) -> Vec<Memory> {
    let mut ranked = HashMap::<String, (f64, Memory)>::new();
    for results in variants {
        for (rank, memory) in results.into_iter().enumerate() {
            add_memory_rank(&mut ranked, memory, rank, 1.0);
        }
    }
    let mut results = ranked
        .into_values()
        .map(|(score, mut memory)| {
            memory.score = Some(memory_quality_adjusted_score(score, &memory));
            memory
        })
        .collect::<Vec<_>>();
    sort_memories(&mut results);
    results.truncate(limit.clamp(1, 100));
    results
}

fn add_memory_rank(
    ranked: &mut HashMap<String, (f64, Memory)>,
    memory: Memory,
    rank: usize,
    weight: f64,
) {
    let entry = ranked
        .entry(memory.id.clone())
        .or_insert_with(|| (0.0, memory));
    entry.0 += reciprocal_rank(rank) * weight;
}

fn merge_code_weighted(
    keyword: Vec<CodeSearchResult>,
    semantic: Vec<CodeSearchResult>,
    graph: Vec<CodeSearchResult>,
    limit: usize,
    plan: &RetrievalPlan,
) -> Vec<CodeSearchResult> {
    let mut ranked = HashMap::<String, (f64, CodeSearchResult)>::new();
    for (rank, result) in keyword.into_iter().enumerate() {
        add_code_rank(
            &mut ranked,
            result,
            rank,
            plan.keyword_weight * plan.code_weight * plan.symbol_weight,
        );
    }
    for (rank, result) in semantic.into_iter().enumerate() {
        add_code_rank(
            &mut ranked,
            result,
            rank,
            plan.semantic_weight * plan.code_weight,
        );
    }
    for (rank, result) in graph.into_iter().enumerate() {
        add_code_rank(
            &mut ranked,
            result,
            rank,
            plan.graph_weight * plan.code_weight,
        );
    }
    let mut results = ranked
        .into_values()
        .map(|(score, mut result)| {
            result.score = score;
            result
        })
        .collect::<Vec<_>>();
    sort_code(&mut results);
    results.truncate(limit.clamp(1, 100));
    results
}

fn merge_semantic_code_variants(
    variants: Vec<Vec<CodeSearchResult>>,
    limit: usize,
) -> Vec<CodeSearchResult> {
    let mut ranked = HashMap::<String, (f64, CodeSearchResult)>::new();
    for results in variants {
        for (rank, result) in results.into_iter().enumerate() {
            add_code_rank(&mut ranked, result, rank, 1.0);
        }
    }
    let mut results = ranked
        .into_values()
        .map(|(score, mut result)| {
            result.score = score;
            result
        })
        .collect::<Vec<_>>();
    sort_code(&mut results);
    results.truncate(limit.clamp(1, 100));
    results
}

fn add_code_rank(
    ranked: &mut HashMap<String, (f64, CodeSearchResult)>,
    result: CodeSearchResult,
    rank: usize,
    weight: f64,
) {
    let entry = ranked
        .entry(result.symbol.id.clone())
        .or_insert_with(|| (0.0, result));
    entry.0 += reciprocal_rank(rank) * weight;
}

fn rerank_memories_deterministic(query: &str, memories: &mut [Memory], plan: &RetrievalPlan) {
    let terms = query_terms(query);
    for memory in memories.iter_mut() {
        let base = memory.score.unwrap_or(0.0);
        let text = format!(
            "{} {} {} {}",
            memory.kind,
            memory.tags.join(" "),
            memory.source.as_deref().unwrap_or(""),
            memory.body
        );
        let lexical = lexical_score(&text, &terms) * 0.004 * plan.keyword_weight;
        let quality = (memory.importance * 0.0015)
            + (memory.confidence * 0.001)
            + (memory.quality_score.clamp(0.0, 1.0) * 0.002)
            - (memory.contradiction_risk.clamp(0.0, 1.0) * 0.003);
        let tier = if memory.memory_tier == "core" {
            0.004
        } else {
            0.0
        };
        memory.score = Some(base + lexical + quality + tier);
    }
    sort_memories(memories);
}

fn rerank_code_deterministic(query: &str, results: &mut [CodeSearchResult], plan: &RetrievalPlan) {
    let terms = query_terms(query);
    for result in results.iter_mut() {
        let symbol = &result.symbol;
        let text = format!(
            "{} {} {} {} {} {}",
            symbol.name,
            symbol.kind,
            symbol.file_path,
            symbol.language,
            symbol.signature,
            symbol.body
        );
        let lexical = lexical_score(&text, &terms) * 0.004 * plan.keyword_weight;
        let symbol_name = symbol_name_score(&symbol.name, &terms) * 0.006 * plan.symbol_weight;
        let kind = if matches!(
            symbol.kind.as_str(),
            "function" | "struct" | "enum" | "trait"
        ) {
            0.001
        } else {
            0.0
        };
        result.score += lexical + symbol_name + kind;
    }
    sort_code(results);
}

fn sort_memories(memories: &mut [Memory]) {
    memories.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .quality_score
                    .partial_cmp(&left.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                left.contradiction_risk
                    .partial_cmp(&right.contradiction_risk)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .importance
                    .partial_cmp(&left.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.updated_at.cmp(&left.updated_at))
    });
}

fn memory_quality_adjusted_score(score: f64, memory: &Memory) -> f64 {
    let quality_multiplier = 0.75 + (memory.quality_score.clamp(0.0, 1.0) * 0.5);
    let risk_multiplier = 1.0 - (memory.contradiction_risk.clamp(0.0, 1.0) * 0.35);
    score * quality_multiplier * risk_multiplier
}

fn sort_code(results: &mut [CodeSearchResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.symbol.file_path.cmp(&right.symbol.file_path))
            .then_with(|| left.symbol.start_line.cmp(&right.symbol.start_line))
    });
}

fn reciprocal_rank(rank: usize) -> f64 {
    1.0 / (60.0 + rank as f64 + 1.0)
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn query_terms(query: &str) -> HashSet<String> {
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .map(|part| part.trim().to_lowercase())
        .filter(|part| part.chars().count() >= 2)
        .collect()
}

fn lexical_score(text: &str, terms: &HashSet<String>) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let normalized = text.to_lowercase();
    terms
        .iter()
        .map(|term| if normalized.contains(term) { 1.0 } else { 0.0 })
        .sum::<f64>()
        / terms.len() as f64
}

fn symbol_name_score(name: &str, terms: &HashSet<String>) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let normalized = name.to_lowercase();
    terms
        .iter()
        .map(|term| {
            if normalized == *term {
                3.0
            } else if normalized.starts_with(term) || normalized.contains(term) {
                1.5
            } else {
                0.0
            }
        })
        .sum()
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

    #[test]
    fn query_plan_detects_code_queries() {
        let plan = build_retrieval_plan("find callers for run_memory_viewer in code");

        assert_eq!(plan.intent, QueryIntent::Code);
        assert!(plan.code_weight > plan.memory_weight);
        assert!(plan.symbol_weight > 1.0);
    }

    #[test]
    fn query_plan_detects_memory_queries() {
        let plan = build_retrieval_plan("какое решение мы приняли в памяти");

        assert_eq!(plan.intent, QueryIntent::Memory);
        assert!(plan.memory_weight > plan.code_weight);
    }

    #[test]
    fn deterministic_code_reranker_boosts_symbol_name_match() {
        let plan = build_retrieval_plan("alpha_handler");
        let mut results = vec![
            code_result("1", "beta_handler", 0.02),
            code_result("2", "alpha_handler", 0.01),
        ];

        rerank_code_deterministic("alpha_handler", &mut results, &plan);

        assert_eq!(results[0].symbol.name, "alpha_handler");
    }

    #[test]
    fn memory_merge_includes_graph_candidates() {
        let plan = build_retrieval_plan("GUI graph");
        let results = merge_memories_weighted(
            Vec::new(),
            Vec::new(),
            vec![memory_result("graph-memory", "Graph fact")],
            5,
            &plan,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "graph-memory");
        assert!(results[0].score.is_some());
    }

    #[test]
    fn memory_merge_applies_feedback_quality_signals() {
        let plan = build_retrieval_plan("retrieval policy");
        let mut low_quality = memory_result("low", "retrieval policy");
        low_quality.quality_score = 0.1;
        low_quality.contradiction_risk = 0.8;
        let mut high_quality = memory_result("high", "retrieval policy");
        high_quality.quality_score = 0.95;
        high_quality.contradiction_risk = 0.0;

        let results =
            merge_memories_weighted(vec![low_quality], Vec::new(), vec![high_quality], 5, &plan);

        assert_eq!(results[0].id, "high");
    }

    #[test]
    fn actual_mode_reports_graph_augmented_source() {
        let mode = actual_mode_with_graph(
            RetrievalMode::Keyword,
            true,
            false,
            &[] as &[Memory],
            true,
            &[memory_result("graph-memory", "Graph fact")],
        );

        assert_eq!(mode, "keyword+graph");
    }

    #[test]
    fn code_merge_includes_graph_neighbors() {
        let plan = build_retrieval_plan("alpha_handler callers");
        let results = merge_code_weighted(
            vec![code_result("seed", "alpha_handler", 0.02)],
            Vec::new(),
            vec![code_result("neighbor", "caller_of_alpha", 0.03)],
            5,
            &plan,
        );

        assert!(results.iter().any(|result| result.symbol.id == "seed"));
        assert!(results.iter().any(|result| result.symbol.id == "neighbor"));
    }

    fn memory_result(id: &str, body: &str) -> Memory {
        Memory {
            id: id.to_string(),
            project_id: "project".to_string(),
            scope: "project".to_string(),
            memory_tier: "archival".to_string(),
            kind: "decision".to_string(),
            body: body.to_string(),
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

    fn code_result(id: &str, name: &str, score: f64) -> CodeSearchResult {
        CodeSearchResult {
            score,
            symbol: crate::store::CodeSymbol {
                id: id.to_string(),
                project_id: "project".to_string(),
                file_path: "src/main.rs".to_string(),
                language: "rust".to_string(),
                name: name.to_string(),
                kind: "function".to_string(),
                signature: format!("fn {name}()"),
                body: String::new(),
                start_line: 1,
                end_line: 2,
                parent_id: None,
            },
        }
    }
}

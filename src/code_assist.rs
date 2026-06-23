use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::code_index::CodeFreshnessReport;
use crate::store::{
    CodeMemory, CodeRouteHint, CodeSearchResult, CodeSimilarityPair, CodeSymbol, NewCodeMemory,
    RememberOutcome, Store,
};

#[derive(Debug, Clone, Serialize)]
pub struct CodePatternReport {
    pub project_id: String,
    pub query: String,
    pub symbols: Vec<CodeSearchResult>,
    pub patterns: Vec<CodePattern>,
    pub route_hints: Vec<CodeRouteHint>,
    pub code_memories: Vec<CodeMemory>,
    pub impacted_files: Vec<String>,
    pub affected_tests: Vec<String>,
    pub test_commands: Vec<TestCommandRecommendation>,
    pub memory_suggestions: Vec<CodeMemorySuggestion>,
    pub pattern_promotions: Vec<CodeMemorySuggestion>,
}

#[derive(Debug, Clone)]
pub struct CodeAssistReportInput<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub actual_mode: String,
    pub warning: Option<String>,
    pub pattern_report: CodePatternReport,
    pub duplicate_pairs: Vec<CodeSimilarityPair>,
    pub applied_memory_suggestions: Vec<AppliedCodeMemorySuggestion>,
    pub index_guard: Option<CodeIndexGuard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodePattern {
    pub seed: CodeSymbol,
    pub related_symbols: Vec<CodeSearchResult>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeMemorySuggestion {
    pub symbol_id: Option<String>,
    pub file_path: Option<String>,
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub source: String,
    pub status: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AppliedCodeMemorySuggestion {
    pub suggestion: CodeMemorySuggestion,
    pub outcome: RememberOutcome,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestCommandRecommendation {
    pub command: String,
    pub reason: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeIndexGuard {
    pub fresh: bool,
    pub stale_files: Vec<String>,
    pub missing_files: Vec<String>,
    pub deleted_files: Vec<String>,
    pub recommended_action: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeterministicCodeReasonReport {
    pub task: String,
    pub answer: String,
    pub bullets: Vec<String>,
    pub risks: Vec<String>,
    pub next_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeReviewPlanReport {
    pub project_id: String,
    pub query: String,
    pub changed_files: Vec<String>,
    pub changed_symbols: Vec<CodeSymbol>,
    pub impacted_files: Vec<String>,
    pub affected_tests: Vec<String>,
    pub test_commands: Vec<TestCommandRecommendation>,
    pub duplicate_pairs: Vec<CodeSimilarityPair>,
    pub memory_suggestions: Vec<CodeMemorySuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEvalCaseInput {
    pub name: String,
    pub query: String,
    #[serde(default)]
    pub expected_ids: Vec<String>,
    #[serde(default)]
    pub expected_symbols: Vec<String>,
    #[serde(default)]
    pub expected_contains: Vec<String>,
    #[serde(default)]
    pub forbidden_contains: Vec<String>,
    #[serde(default)]
    pub min_results: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeEvalCaseReport {
    pub name: String,
    pub query: String,
    pub passed: bool,
    pub hits: usize,
    pub expected_ids: Vec<String>,
    pub matched_expected_ids: Vec<String>,
    pub missing_expected_ids: Vec<String>,
    pub expected_symbols: Vec<String>,
    pub matched_expected_symbols: Vec<String>,
    pub missing_expected_symbols: Vec<String>,
    pub recall_at_k: Option<f64>,
    pub precision_at_k: Option<f64>,
    pub mrr: Option<f64>,
    pub missing_expected: Vec<String>,
    pub forbidden_found: Vec<String>,
    pub top_ids: Vec<String>,
    pub top_symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeEvalReport {
    pub project_id: String,
    pub mode: String,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub cases: Vec<CodeEvalCaseReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeAssistReport {
    pub project_id: String,
    pub query: String,
    pub actual_mode: String,
    pub warning: Option<String>,
    pub symbols: Vec<CodeSearchResult>,
    pub patterns: Vec<CodePattern>,
    pub duplicate_pairs: Vec<CodeSimilarityPair>,
    pub route_hints: Vec<CodeRouteHint>,
    pub code_memories: Vec<CodeMemory>,
    pub impacted_files: Vec<String>,
    pub affected_tests: Vec<String>,
    pub test_commands: Vec<TestCommandRecommendation>,
    pub memory_suggestions: Vec<CodeMemorySuggestion>,
    pub pattern_promotions: Vec<CodeMemorySuggestion>,
    pub applied_memory_suggestions: Vec<AppliedCodeMemorySuggestion>,
    pub index_guard: Option<CodeIndexGuard>,
}

pub fn build_code_pattern_report(
    store: &Store,
    project_id: &str,
    query: &str,
    symbols: Vec<CodeSearchResult>,
    embedding_model: &str,
    limit: usize,
    min_similarity: f64,
) -> Result<CodePatternReport> {
    let patterns = related_patterns_for_symbols(
        store,
        project_id,
        &symbols,
        embedding_model,
        limit,
        min_similarity,
    )?;
    let route_hints = store.route_hints(project_id, query, limit.max(8))?;
    let code_memories = store.code_memories_for_code_results(project_id, &symbols, limit.max(8))?;
    let impacted_files = impacted_files_for_symbols(&symbols);
    let affected_tests =
        store.affected_test_files(project_id, &impacted_files, 5, limit.max(25))?;
    let test_commands = recommend_test_commands(&affected_tests, &impacted_files);
    let memory_suggestions = code_memory_suggestions(query, &symbols, &patterns, &route_hints);
    let pattern_promotions = pattern_promotion_suggestions(&patterns);

    Ok(CodePatternReport {
        project_id: project_id.to_string(),
        query: query.to_string(),
        symbols,
        patterns,
        route_hints,
        code_memories,
        impacted_files,
        affected_tests,
        test_commands,
        memory_suggestions,
        pattern_promotions,
    })
}

pub fn build_code_assist_report(input: CodeAssistReportInput<'_>) -> CodeAssistReport {
    CodeAssistReport {
        project_id: input.project_id.to_string(),
        query: input.query.to_string(),
        actual_mode: input.actual_mode,
        warning: input.warning,
        symbols: input.pattern_report.symbols,
        patterns: input.pattern_report.patterns,
        duplicate_pairs: input.duplicate_pairs,
        route_hints: input.pattern_report.route_hints,
        code_memories: input.pattern_report.code_memories,
        impacted_files: input.pattern_report.impacted_files,
        affected_tests: input.pattern_report.affected_tests,
        test_commands: input.pattern_report.test_commands,
        memory_suggestions: input.pattern_report.memory_suggestions,
        pattern_promotions: input.pattern_report.pattern_promotions,
        applied_memory_suggestions: input.applied_memory_suggestions,
        index_guard: input.index_guard,
    }
}

pub fn related_patterns_for_symbols(
    store: &Store,
    project_id: &str,
    symbols: &[CodeSearchResult],
    embedding_model: &str,
    limit: usize,
    min_similarity: f64,
) -> Result<Vec<CodePattern>> {
    let per_symbol_limit = limit.clamp(1, 20).min(5);
    let mut patterns = Vec::new();
    for result in symbols.iter().take(limit.clamp(1, 20)) {
        let related = store.search_related_code_symbols(
            project_id,
            &result.symbol.id,
            embedding_model,
            per_symbol_limit,
            min_similarity,
        )?;
        if related.is_empty() {
            continue;
        }
        patterns.push(CodePattern {
            seed: result.symbol.clone(),
            related_symbols: related,
            reason: "embedding-nearest indexed symbols expose existing implementation patterns"
                .to_string(),
        });
    }
    Ok(patterns)
}

pub fn impacted_files_for_symbols(symbols: &[CodeSearchResult]) -> Vec<String> {
    let mut files = BTreeSet::new();
    for result in symbols {
        files.insert(result.symbol.file_path.clone());
    }
    files.into_iter().collect()
}

pub fn recommend_test_commands(
    affected_tests: &[String],
    impacted_files: &[String],
) -> Vec<TestCommandRecommendation> {
    let mut commands = BTreeSet::new();
    let mut recommendations = Vec::new();
    for file in affected_tests {
        let command = if let Some(name) = integration_test_name(file) {
            format!("cargo test --test {name}")
        } else {
            "cargo test".to_string()
        };
        if commands.insert(command.clone()) {
            recommendations.push(TestCommandRecommendation {
                command,
                reason: format!(
                    "affected indexed test file `{file}` is connected to impacted code"
                ),
                confidence: 0.85,
            });
        }
    }
    if !impacted_files.is_empty() && commands.insert("cargo check".to_string()) {
        recommendations.push(TestCommandRecommendation {
            command: "cargo check".to_string(),
            reason: "impacted Rust source files should compile before focused tests".to_string(),
            confidence: 0.9,
        });
    }
    if recommendations.is_empty() && !impacted_files.is_empty() {
        recommendations.push(TestCommandRecommendation {
            command: "cargo test".to_string(),
            reason: "no indexed affected test file was found; run the project test suite"
                .to_string(),
            confidence: 0.55,
        });
    }
    recommendations
}

fn integration_test_name(file: &str) -> Option<String> {
    let rest = file.strip_prefix("tests/")?;
    let name = rest.strip_suffix(".rs")?;
    if name.contains('/') {
        None
    } else {
        Some(name.to_string())
    }
}

pub fn code_memory_suggestions(
    query: &str,
    symbols: &[CodeSearchResult],
    patterns: &[CodePattern],
    route_hints: &[CodeRouteHint],
) -> Vec<CodeMemorySuggestion> {
    let mut suggestions = Vec::new();
    for result in symbols.iter().take(5) {
        suggestions.push(CodeMemorySuggestion {
            symbol_id: Some(result.symbol.id.clone()),
            file_path: Some(result.symbol.file_path.clone()),
            kind: "implementation_note".to_string(),
            body: format!(
                "For task `{}`, `{}` in `{}` was identified as a relevant implementation entry point.",
                truncate(query, 180),
                result.symbol.name,
                result.symbol.file_path
            ),
            tags: vec!["agent-workflow".to_string(), "code-assist".to_string()],
            source: "dukememory_code_assist".to_string(),
            status: "pending".to_string(),
            confidence: 0.7,
        });
    }

    for pattern in patterns.iter().take(3) {
        let related = pattern
            .related_symbols
            .iter()
            .take(3)
            .map(|result| result.symbol.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        if related.is_empty() {
            continue;
        }
        suggestions.push(CodeMemorySuggestion {
            symbol_id: Some(pattern.seed.id.clone()),
            file_path: Some(pattern.seed.file_path.clone()),
            kind: "pattern".to_string(),
            body: format!(
                "`{}` has embedding-near implementation patterns: {}.",
                pattern.seed.name, related
            ),
            tags: vec!["pattern-reuse".to_string(), "embedding".to_string()],
            source: "dukememory_code_patterns".to_string(),
            status: "pending".to_string(),
            confidence: 0.75,
        });
    }

    if !route_hints.is_empty() {
        let routes = route_hints
            .iter()
            .take(5)
            .map(|hint| hint.route.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        suggestions.push(CodeMemorySuggestion {
            symbol_id: None,
            file_path: None,
            kind: "route_hint".to_string(),
            body: format!(
                "Task `{}` matched reusable code routes: {}.",
                truncate(query, 180),
                routes
            ),
            tags: vec!["route-hint".to_string(), "agent-workflow".to_string()],
            source: "dukememory_code_assist".to_string(),
            status: "pending".to_string(),
            confidence: 0.65,
        });
    }

    suggestions
}

pub fn pattern_promotion_suggestions(patterns: &[CodePattern]) -> Vec<CodeMemorySuggestion> {
    patterns
        .iter()
        .filter(|pattern| pattern.related_symbols.len() >= 2)
        .take(10)
        .map(|pattern| {
            let related = pattern
                .related_symbols
                .iter()
                .take(5)
                .map(|result| format!("{} in {}", result.symbol.name, result.symbol.file_path))
                .collect::<Vec<_>>()
                .join("; ");
            CodeMemorySuggestion {
                symbol_id: Some(pattern.seed.id.clone()),
                file_path: Some(pattern.seed.file_path.clone()),
                kind: "pattern".to_string(),
                body: format!(
                    "Reusable implementation pattern for `{}`: compare with {}.",
                    pattern.seed.name, related
                ),
                tags: vec!["pattern-promotion".to_string(), "embedding".to_string()],
                source: "dukememory_code_pattern_promote".to_string(),
                status: "pending".to_string(),
                confidence: 0.78,
            }
        })
        .collect()
}

pub fn apply_code_memory_suggestions(
    store: &Store,
    project_id: &str,
    suggestions: &[CodeMemorySuggestion],
    limit: usize,
) -> Result<Vec<AppliedCodeMemorySuggestion>> {
    suggestions
        .iter()
        .take(limit)
        .map(|suggestion| {
            let outcome = store.remember_code_memory(
                project_id,
                NewCodeMemory {
                    symbol_id: suggestion.symbol_id.clone(),
                    symbol_kind: None,
                    file_path: suggestion.file_path.clone(),
                    status: suggestion.status.clone(),
                    kind: suggestion.kind.clone(),
                    body: suggestion.body.clone(),
                    tags: suggestion.tags.clone(),
                    source: Some(suggestion.source.clone()),
                    confidence: suggestion.confidence,
                },
                true,
            )?;
            Ok(AppliedCodeMemorySuggestion {
                suggestion: suggestion.clone(),
                outcome,
            })
        })
        .collect()
}

pub fn code_index_guard_from_freshness(report: &CodeFreshnessReport) -> CodeIndexGuard {
    let fresh = report.is_fresh();
    CodeIndexGuard {
        fresh,
        stale_files: report.stale_files.clone(),
        missing_files: report.missing_files.clone(),
        deleted_files: report.deleted_files.clone(),
        recommended_action: if fresh {
            None
        } else {
            Some("Run dukememory_code_index for this project_path before trusting code-assist results.".to_string())
        },
    }
}

pub fn deterministic_reason_report(
    task: &str,
    query: &str,
    assist: &CodeAssistReport,
) -> DeterministicCodeReasonReport {
    let symbol_names = assist
        .symbols
        .iter()
        .take(8)
        .map(|result| format!("{} ({})", result.symbol.name, result.symbol.file_path))
        .collect::<Vec<_>>();
    let mut bullets = Vec::new();
    if !symbol_names.is_empty() {
        bullets.push(format!(
            "Relevant indexed symbols: {}.",
            symbol_names.join(", ")
        ));
    }
    if !assist.patterns.is_empty() {
        bullets.push(format!(
            "Embedding-near reusable pattern groups found: {}.",
            assist.patterns.len()
        ));
    }
    if !assist.affected_tests.is_empty() {
        bullets.push(format!(
            "Affected indexed tests: {}.",
            assist.affected_tests.join(", ")
        ));
    }
    if !assist.test_commands.is_empty() {
        bullets.push(format!(
            "Recommended commands: {}.",
            assist
                .test_commands
                .iter()
                .map(|item| item.command.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    let mut risks = Vec::new();
    if !assist.duplicate_pairs.is_empty() {
        risks.push(format!(
            "{} near-duplicate code pairs may indicate reuse/refactor risk.",
            assist.duplicate_pairs.len()
        ));
    }
    if let Some(guard) = &assist.index_guard
        && !guard.fresh
    {
        risks.push("Code index is stale; refresh before relying on symbol coverage.".to_string());
    }
    if risks.is_empty() {
        risks.push("No indexed high-risk duplicate or stale-index signal was found.".to_string());
    }
    let next_steps = if task == "risk" {
        vec![
            "Inspect impacted callers/callees for the top symbols.".to_string(),
            "Run the recommended focused test commands.".to_string(),
            "Store durable code-memory notes after verification if the pattern repeats."
                .to_string(),
        ]
    } else {
        vec![
            "Start with the top relevant symbols and reuse the nearest pattern groups.".to_string(),
            "Patch only the impacted files unless callers reveal a wider contract change."
                .to_string(),
            "Run recommended commands, then record durable code memories for stable learnings."
                .to_string(),
        ]
    };
    DeterministicCodeReasonReport {
        task: task.to_string(),
        answer: format!(
            "Deterministic {task} for `{query}` used indexed symbols, embedding-near patterns, duplicate candidates, affected tests, and stored code memories without an LLM call."
        ),
        bullets,
        risks,
        next_steps,
    }
}

pub fn build_code_review_plan_report(
    store: &Store,
    project_id: &str,
    query: &str,
    changed_files: Vec<String>,
    duplicate_pairs: Vec<CodeSimilarityPair>,
    limit: usize,
) -> Result<CodeReviewPlanReport> {
    let mut changed_symbols = Vec::new();
    for file in &changed_files {
        changed_symbols.extend(store.code_symbols_for_file(project_id, file)?);
    }
    changed_symbols.truncate(limit.clamp(1, 100));
    let mut impacted = BTreeSet::new();
    for file in &changed_files {
        impacted.insert(file.clone());
    }
    let affected_tests = store.affected_test_files(project_id, &changed_files, 5, limit.max(25))?;
    for file in &affected_tests {
        impacted.insert(file.clone());
    }
    let impacted_files = impacted.into_iter().collect::<Vec<_>>();
    let test_commands = recommend_test_commands(&affected_tests, &impacted_files);
    let search_results = changed_symbols
        .iter()
        .cloned()
        .map(|symbol| CodeSearchResult { symbol, score: 1.0 })
        .collect::<Vec<_>>();
    let memory_suggestions = code_memory_suggestions(query, &search_results, &[], &[]);
    Ok(CodeReviewPlanReport {
        project_id: project_id.to_string(),
        query: query.to_string(),
        changed_files,
        changed_symbols,
        impacted_files,
        affected_tests,
        test_commands,
        duplicate_pairs,
        memory_suggestions,
    })
}

pub fn evaluate_code_cases(
    project_id: &str,
    mode: &str,
    cases: Vec<(CodeEvalCaseInput, Vec<CodeSearchResult>)>,
) -> CodeEvalReport {
    let reports = cases
        .into_iter()
        .map(|(case, results)| evaluate_code_case(case, results))
        .collect::<Vec<_>>();
    let passed_cases = reports.iter().filter(|case| case.passed).count();
    let failed_cases = reports.len().saturating_sub(passed_cases);
    CodeEvalReport {
        project_id: project_id.to_string(),
        mode: mode.to_string(),
        total_cases: reports.len(),
        passed_cases,
        failed_cases,
        cases: reports,
    }
}

fn evaluate_code_case(
    case: CodeEvalCaseInput,
    results: Vec<CodeSearchResult>,
) -> CodeEvalCaseReport {
    let top_ids = results
        .iter()
        .map(|result| result.symbol.id.clone())
        .collect::<Vec<_>>();
    let top_symbols = results
        .iter()
        .map(|result| result.symbol.name.clone())
        .collect::<Vec<_>>();
    let haystack = results
        .iter()
        .map(|result| {
            format!(
                "{}\n{}\n{}\n{}\n{}",
                result.symbol.file_path,
                result.symbol.kind,
                result.symbol.name,
                result.symbol.signature,
                result.symbol.body
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_ascii_lowercase();
    let matched_expected_ids = matched(&case.expected_ids, &top_ids);
    let missing_expected_ids = missing(&case.expected_ids, &top_ids);
    let matched_expected_symbols = matched(&case.expected_symbols, &top_symbols);
    let missing_expected_symbols = missing(&case.expected_symbols, &top_symbols);
    let missing_expected = case
        .expected_contains
        .iter()
        .filter(|needle| !haystack.contains(&needle.to_ascii_lowercase()))
        .cloned()
        .collect::<Vec<_>>();
    let forbidden_found = case
        .forbidden_contains
        .iter()
        .filter(|needle| haystack.contains(&needle.to_ascii_lowercase()))
        .cloned()
        .collect::<Vec<_>>();
    let expected_total = case.expected_ids.len() + case.expected_symbols.len();
    let matched_total = matched_expected_ids.len() + matched_expected_symbols.len();
    let recall_at_k = if expected_total == 0 {
        None
    } else {
        Some(matched_total as f64 / expected_total as f64)
    };
    let precision_at_k = if expected_total == 0 {
        None
    } else if top_ids.is_empty() {
        Some(0.0)
    } else {
        Some(matched_total as f64 / top_ids.len() as f64)
    };
    let expected_positions = case
        .expected_ids
        .iter()
        .filter_map(|id| top_ids.iter().position(|top| top == id))
        .chain(
            case.expected_symbols
                .iter()
                .filter_map(|name| top_symbols.iter().position(|top| top == name)),
        )
        .collect::<Vec<_>>();
    let mrr = if expected_total == 0 {
        None
    } else {
        Some(
            expected_positions
                .into_iter()
                .min()
                .map(|index| 1.0 / (index as f64 + 1.0))
                .unwrap_or(0.0),
        )
    };
    let min_results_ok = case
        .min_results
        .map(|min_results| results.len() >= min_results)
        .unwrap_or(true);
    let passed = missing_expected_ids.is_empty()
        && missing_expected_symbols.is_empty()
        && missing_expected.is_empty()
        && forbidden_found.is_empty()
        && min_results_ok;
    CodeEvalCaseReport {
        name: case.name,
        query: case.query,
        passed,
        hits: results.len(),
        expected_ids: case.expected_ids,
        matched_expected_ids,
        missing_expected_ids,
        expected_symbols: case.expected_symbols,
        matched_expected_symbols,
        missing_expected_symbols,
        recall_at_k,
        precision_at_k,
        mrr,
        missing_expected,
        forbidden_found,
        top_ids,
        top_symbols,
    }
}

fn matched(expected: &[String], observed: &[String]) -> Vec<String> {
    expected
        .iter()
        .filter(|value| observed.iter().any(|observed| observed == *value))
        .cloned()
        .collect()
}

fn missing(expected: &[String], observed: &[String]) -> Vec<String> {
    expected
        .iter()
        .filter(|value| !observed.iter().any(|observed| observed == *value))
        .cloned()
        .collect()
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut output = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    output.push_str("...");
    output
}

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use crate::code_assist::{
    CodeReviewPlanReport, TestCommandRecommendation, build_code_review_plan_report,
};
use crate::code_index::{
    CodeFreshnessReport, IndexReport, IndexTimingReport, check_code_index_freshness, index_project,
};
use crate::store::{
    CodeMemory, CodeMemorySearchOptions, CodeRelation, CodeSimilarityPairOptions, CodeSymbol,
    DEFAULT_MEMORY_SCOPE, DEFAULT_MEMORY_TIER, ListOptions, Memory, MemoryStatus, NewMemory,
    NewTaskSession, RememberOutcome, SearchOptions, StatusFilter, Store, TaskSession,
    TaskSessionUpdate,
};

#[derive(Debug, Clone)]
pub struct DevsystemRequest {
    pub project_id: String,
    pub query: String,
    pub files: Vec<String>,
    pub project_path: Option<PathBuf>,
    pub write_memory: bool,
    pub auto_index: bool,
    pub full_rebuild: bool,
    pub embed_symbols: bool,
    pub embed_symbol_limit: usize,
    pub precomputed_index_run: Option<IndexRunSummary>,
    pub run_evidence: bool,
    pub evidence_timeout_seconds: u64,
    pub max_evidence_commands: usize,
    pub allowed_evidence_commands: Vec<String>,
    pub code_embedding_model: Option<String>,
    pub duplicate_similarity: f64,
    pub review_limit: usize,
    pub policy_override: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentRunReport {
    pub product: String,
    pub readiness_percent: u8,
    pub task_session_id: String,
    pub task_intent: String,
    pub stage_reports: Vec<AgentStageReport>,
    pub role_reports: RoleReports,
    pub telemetry: DevsystemTelemetry,
    pub code_review_plan: CodeReviewPlanReport,
    pub file_entropy_reports: Vec<FileEntropyReport>,
    pub boundary_repair_plans: Vec<BoundaryRepairPlan>,
    pub quality_gates: Vec<QualityGate>,
    pub quality_gate_summary: QualityGateSummary,
    pub recommended_tests: Vec<String>,
    pub recommended_test_commands: Vec<TestCommandRecommendation>,
    pub quality_evidence_reports: Vec<QualityEvidenceReport>,
    pub memory_writes: MemoryWrites,
    pub final_verdict: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentStageReport {
    pub role: String,
    pub status: String,
    pub readiness_percent: u8,
    pub summary: String,
    pub artifacts: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoleReports {
    pub planner: Value,
    pub memory: Value,
    pub architect: Value,
    pub coder: Value,
    pub test: Value,
    pub critic: Value,
    pub refactor: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryWrites {
    pub status: String,
    pub ids: Vec<String>,
    pub entries: Vec<MemoryWriteEntry>,
    pub intent_memory_ids: Vec<String>,
    pub decision_memory_ids: Vec<String>,
    pub entropy_memory_ids: Vec<String>,
    pub graph_memory_ids: Vec<String>,
    pub quality_observation_ids: Vec<String>,
    pub graph_candidates: Vec<IntentGraphCandidate>,
    pub inserted_count: usize,
    pub duplicate_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryWriteEntry {
    pub category: String,
    pub kind: String,
    pub id: String,
    pub inserted: bool,
    pub duplicate_of: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IntentGraphCandidate {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub source_memory_id: Option<String>,
    pub source_memory_category: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DevsystemTelemetry {
    pub index_run: Option<IndexRunSummary>,
    pub index_guard: Option<DevsystemIndexGuard>,
    pub missing_signals: Vec<String>,
    pub memory_agent: MemoryStageOutput,
    pub policy: DevsystemPolicyReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexRunSummary {
    pub enabled: bool,
    pub reason: Option<String>,
    pub project_id: String,
    pub root_path: Option<PathBuf>,
    pub full_rebuild: bool,
    pub files_seen: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub indexed_files: Vec<String>,
    pub symbols_indexed: usize,
    pub relations_indexed: usize,
    pub relation_targets_reset: usize,
    pub calls_resolved: usize,
    pub uses_resolved: usize,
    pub modules_resolved: usize,
    pub embed_symbols: bool,
    pub embed_symbol_limit: usize,
    pub embedded_symbols: usize,
    pub timing: IndexTimingReport,
}

impl IndexRunSummary {
    pub fn disabled(
        project_id: String,
        root_path: Option<PathBuf>,
        reason: impl Into<String>,
        full_rebuild: bool,
        embed_symbols: bool,
        embed_symbol_limit: usize,
    ) -> Self {
        Self {
            enabled: false,
            reason: Some(reason.into()),
            project_id,
            root_path,
            full_rebuild,
            files_seen: 0,
            files_indexed: 0,
            files_skipped: 0,
            files_deleted: 0,
            indexed_files: Vec::new(),
            symbols_indexed: 0,
            relations_indexed: 0,
            relation_targets_reset: 0,
            calls_resolved: 0,
            uses_resolved: 0,
            modules_resolved: 0,
            embed_symbols,
            embed_symbol_limit,
            embedded_symbols: 0,
            timing: IndexTimingReport::default(),
        }
    }

    pub fn from_index_report(
        report: IndexReport,
        embed_symbols: bool,
        embed_symbol_limit: usize,
        embedded_symbols: usize,
    ) -> Self {
        Self {
            enabled: true,
            reason: None,
            project_id: report.project_id,
            root_path: Some(report.root_path),
            full_rebuild: report.full_rebuild,
            files_seen: report.files_seen,
            files_indexed: report.files_indexed,
            files_skipped: report.files_skipped,
            files_deleted: report.files_deleted,
            indexed_files: report.indexed_files,
            symbols_indexed: report.symbols_indexed,
            relations_indexed: report.relations_indexed,
            relation_targets_reset: report.relation_targets_reset,
            calls_resolved: report.calls_resolved,
            uses_resolved: report.uses_resolved,
            modules_resolved: report.modules_resolved,
            embed_symbols,
            embed_symbol_limit,
            embedded_symbols,
            timing: report.timing,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DevsystemIndexGuard {
    pub fresh: bool,
    pub stale_files: Vec<String>,
    pub missing_files: Vec<String>,
    pub deleted_files: Vec<String>,
    pub recommended_action: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileEntropyReport {
    pub file_path: String,
    pub score: u32,
    pub verdict: String,
    pub responsibility_count: usize,
    pub responsibility_summary: Vec<String>,
    pub signals: FileEntropySignals,
    pub missing_signals: Vec<String>,
    pub boundary_repair_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileEntropySignals {
    pub lines: u32,
    pub fan_in: usize,
    pub fan_out: usize,
    pub public_api_surface: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cyclomatic_complexity: Option<u32>,
    pub semantic_sections: Vec<String>,
    pub unrelated_reasons_to_change: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_frequency: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number_of_authors: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_coverage: Option<f64>,
    pub co_change_hotspots: Vec<String>,
    pub affected_tests: Vec<String>,
    pub task_history_touches: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryRepairPlan {
    pub source_file: String,
    pub risk_level: String,
    pub reason: String,
    pub retain_orchestrator: bool,
    pub proposed_modules: Vec<ProposedBoundaryModule>,
    pub move_order: Vec<String>,
    pub affected_imports: Vec<String>,
    pub required_tests: Vec<TestCommandRecommendation>,
    pub non_goals: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProposedBoundaryModule {
    pub module_path: String,
    pub responsibility: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityGate {
    pub id: String,
    pub status: String,
    pub severity: String,
    pub message: String,
    pub files: Vec<String>,
    pub required_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityGateSummary {
    pub overall_status: String,
    pub pass_count: usize,
    pub warning_count: usize,
    pub decision_count: usize,
    pub blocker_count: usize,
    pub recommended_next_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityEvidenceReport {
    pub command: String,
    pub reason: String,
    pub status: String,
    pub exit_code: Option<i32>,
    pub duration_ms: u128,
    pub stdout_excerpt: String,
    pub stderr_excerpt: String,
    pub evidence_source: String,
    pub affects_gate_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlannerStageOutput {
    pub task_intent: String,
    pub scope_files: Vec<String>,
    pub constraints: Vec<String>,
    pub acceptance_criteria: Vec<String>,
    pub likely_risk: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryStageOutput {
    pub active_memories: Vec<MemorySummary>,
    pub pending_memories: Vec<MemorySummary>,
    pub relevant_memories: Vec<MemorySummary>,
    pub recent_task_sessions: Vec<TaskSessionSummary>,
    pub file_task_history: Vec<FileTaskHistorySummary>,
    pub code_memories: Vec<CodeMemorySummary>,
    pub stale_code_memories: Vec<CodeMemorySummary>,
    pub index_run: Option<IndexRunSummary>,
    pub index_guard: Option<DevsystemIndexGuard>,
    pub missing_signals: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySummary {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub memory_tier: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub confidence: f64,
    pub importance: f64,
    pub score: Option<f64>,
    pub body_excerpt: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskSessionSummary {
    pub id: String,
    pub status: String,
    pub phase: String,
    pub progress: usize,
    pub query: String,
    pub file_paths: Vec<String>,
    pub test_paths: Vec<String>,
    pub summary: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileTaskHistorySummary {
    pub file_path: String,
    pub touches: usize,
    pub recent_session_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeMemorySummary {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub link_status: String,
    pub file_path: Option<String>,
    pub symbol_id: Option<String>,
    pub symbol_name: Option<String>,
    pub confidence: f64,
    pub quality_score: f64,
    pub usage_count: u64,
    pub contradiction_risk: f64,
    pub body_excerpt: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DevsystemPolicy {
    pub boundary_repair_score_threshold: u32,
    pub split_score_threshold: u32,
    pub boundary_repair_responsibility_count: usize,
    pub split_responsibility_count: usize,
    pub line_signal_step: u32,
    pub max_line_signal: u32,
    pub low_coverage_threshold: f64,
    pub ignored_file_patterns: Vec<String>,
    pub static_metadata_patterns: Vec<String>,
    pub generated_file_patterns: Vec<String>,
    pub required_test_commands: Vec<String>,
    pub responsibility_keywords: Vec<ResponsibilityKeywordPolicy>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsibilityKeywordPolicy {
    pub responsibility: String,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DevsystemPolicyReport {
    pub source: String,
    pub effective: DevsystemPolicy,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyLoadResult {
    source: String,
    effective: DevsystemPolicy,
    warnings: Vec<String>,
}

impl Default for DevsystemPolicy {
    fn default() -> Self {
        Self {
            boundary_repair_score_threshold: 45,
            split_score_threshold: 75,
            boundary_repair_responsibility_count: 3,
            split_responsibility_count: 7,
            line_signal_step: 200,
            max_line_signal: 10,
            low_coverage_threshold: 0.35,
            ignored_file_patterns: Vec::new(),
            static_metadata_patterns: vec!["countries".to_string(), "metadata".to_string()],
            generated_file_patterns: vec![
                "generated/".to_string(),
                ".generated.".to_string(),
                "_generated.".to_string(),
                "target/".to_string(),
            ],
            required_test_commands: Vec::new(),
            responsibility_keywords: default_responsibility_keyword_policy(),
        }
    }
}

fn default_responsibility_keyword_policy() -> Vec<ResponsibilityKeywordPolicy> {
    RESPONSIBILITY_KEYWORDS
        .iter()
        .map(|(responsibility, keywords)| ResponsibilityKeywordPolicy {
            responsibility: (*responsibility).to_string(),
            keywords: keywords
                .iter()
                .map(|keyword| (*keyword).to_string())
                .collect(),
        })
        .collect()
}

fn load_devsystem_policy(
    project_path: Option<&Path>,
    override_value: Option<&Value>,
) -> Result<PolicyLoadResult> {
    let mut policy = DevsystemPolicy::default();
    let mut sources = vec!["defaults".to_string()];
    let mut warnings = Vec::new();
    if let Some(root) = project_path {
        let config_path = root.join(".dukememory.toml");
        if config_path.exists() {
            let text = std::fs::read_to_string(&config_path).with_context(|| {
                format!(
                    "failed to read devsystem policy from {}",
                    config_path.display()
                )
            })?;
            if apply_toml_devsystem_policy(&mut policy, &text, &mut warnings) {
                sources.push("project_config".to_string());
            }
        }
    }
    if let Some(value) = override_value {
        apply_json_devsystem_policy(&mut policy, value, &mut warnings)?;
        sources.push("mcp_override".to_string());
    }
    validate_devsystem_policy(&mut policy, &mut warnings);
    Ok(PolicyLoadResult {
        source: sources.join("+"),
        effective: policy,
        warnings,
    })
}

fn apply_toml_devsystem_policy(
    policy: &mut DevsystemPolicy,
    text: &str,
    warnings: &mut Vec<String>,
) -> bool {
    let mut in_devsystem = false;
    let mut applied = false;
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_devsystem = line.trim_matches(&['[', ']'][..]).trim() == "devsystem";
            continue;
        }
        if !in_devsystem {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            warnings.push(format!(
                "project_config: ignored malformed devsystem policy line {}",
                line_index + 1
            ));
            continue;
        };
        if apply_policy_key_value(policy, key.trim(), value.trim(), warnings, "project_config") {
            applied = true;
        }
    }
    applied
}

fn apply_json_devsystem_policy(
    policy: &mut DevsystemPolicy,
    value: &Value,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let Some(object) = value.as_object() else {
        bail!("dukememory_devsystem policy must be a JSON object");
    };
    for (key, value) in object {
        match key.as_str() {
            "boundary_repair_score_threshold" => {
                if let Some(number) = json_u32(value) {
                    policy.boundary_repair_score_threshold = number;
                } else {
                    warnings.push(
                        "mcp_override: boundary_repair_score_threshold must be an integer"
                            .to_string(),
                    );
                }
            }
            "split_score_threshold" => {
                if let Some(number) = json_u32(value) {
                    policy.split_score_threshold = number;
                } else {
                    warnings
                        .push("mcp_override: split_score_threshold must be an integer".to_string());
                }
            }
            "boundary_repair_responsibility_count" => {
                if let Some(number) = json_usize(value) {
                    policy.boundary_repair_responsibility_count = number;
                } else {
                    warnings.push(
                        "mcp_override: boundary_repair_responsibility_count must be an integer"
                            .to_string(),
                    );
                }
            }
            "split_responsibility_count" => {
                if let Some(number) = json_usize(value) {
                    policy.split_responsibility_count = number;
                } else {
                    warnings.push(
                        "mcp_override: split_responsibility_count must be an integer".to_string(),
                    );
                }
            }
            "line_signal_step" => {
                if let Some(number) = json_u32(value) {
                    policy.line_signal_step = number;
                } else {
                    warnings.push("mcp_override: line_signal_step must be an integer".to_string());
                }
            }
            "max_line_signal" => {
                if let Some(number) = json_u32(value) {
                    policy.max_line_signal = number;
                } else {
                    warnings.push("mcp_override: max_line_signal must be an integer".to_string());
                }
            }
            "low_coverage_threshold" => {
                if let Some(number) = value.as_f64() {
                    policy.low_coverage_threshold = number;
                } else {
                    warnings
                        .push("mcp_override: low_coverage_threshold must be a number".to_string());
                }
            }
            "ignored_file_patterns" => {
                apply_json_string_array(&mut policy.ignored_file_patterns, value, warnings, key);
            }
            "static_metadata_patterns" => {
                apply_json_string_array(&mut policy.static_metadata_patterns, value, warnings, key);
            }
            "generated_file_patterns" => {
                apply_json_string_array(&mut policy.generated_file_patterns, value, warnings, key);
            }
            "required_test_commands" => {
                apply_json_string_array(&mut policy.required_test_commands, value, warnings, key);
            }
            "responsibility_keywords" => {
                apply_json_responsibility_keywords(policy, value, warnings);
            }
            other => warnings.push(format!(
                "mcp_override: ignored unknown policy key `{other}`"
            )),
        }
    }
    Ok(())
}

fn apply_policy_key_value(
    policy: &mut DevsystemPolicy,
    key: &str,
    value: &str,
    warnings: &mut Vec<String>,
    source: &str,
) -> bool {
    match key {
        "boundary_repair_score_threshold" => parse_u32_value(value)
            .map(|number| policy.boundary_repair_score_threshold = number)
            .is_some(),
        "split_score_threshold" => parse_u32_value(value)
            .map(|number| policy.split_score_threshold = number)
            .is_some(),
        "boundary_repair_responsibility_count" => parse_usize_value(value)
            .map(|number| policy.boundary_repair_responsibility_count = number)
            .is_some(),
        "split_responsibility_count" => parse_usize_value(value)
            .map(|number| policy.split_responsibility_count = number)
            .is_some(),
        "line_signal_step" => parse_u32_value(value)
            .map(|number| policy.line_signal_step = number)
            .is_some(),
        "max_line_signal" => parse_u32_value(value)
            .map(|number| policy.max_line_signal = number)
            .is_some(),
        "low_coverage_threshold" => parse_f64_value(value)
            .map(|number| policy.low_coverage_threshold = number)
            .is_some(),
        "ignored_file_patterns" => parse_string_array(value)
            .map(|items| policy.ignored_file_patterns = items)
            .is_some(),
        "static_metadata_patterns" => parse_string_array(value)
            .map(|items| policy.static_metadata_patterns = items)
            .is_some(),
        "generated_file_patterns" => parse_string_array(value)
            .map(|items| policy.generated_file_patterns = items)
            .is_some(),
        "required_test_commands" => parse_string_array(value)
            .map(|items| policy.required_test_commands = items)
            .is_some(),
        other => {
            warnings.push(format!("{source}: ignored unknown policy key `{other}`"));
            false
        }
    }
}

fn validate_devsystem_policy(policy: &mut DevsystemPolicy, warnings: &mut Vec<String>) {
    if policy.line_signal_step == 0 {
        warnings.push("policy: line_signal_step must be greater than zero; using 200".to_string());
        policy.line_signal_step = 200;
    }
    if policy.boundary_repair_score_threshold > policy.split_score_threshold {
        warnings.push(
            "policy: boundary_repair_score_threshold was above split_score_threshold; clamping boundary threshold to split threshold".to_string(),
        );
        policy.boundary_repair_score_threshold = policy.split_score_threshold;
    }
    policy.low_coverage_threshold = policy.low_coverage_threshold.clamp(0.0, 1.0);
    policy.ignored_file_patterns.sort();
    policy.ignored_file_patterns.dedup();
    policy.static_metadata_patterns.sort();
    policy.static_metadata_patterns.dedup();
    policy.generated_file_patterns.sort();
    policy.generated_file_patterns.dedup();
    policy.required_test_commands.sort();
    policy.required_test_commands.dedup();
}

fn json_u32(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|number| u32::try_from(number).ok())
}

fn json_usize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
}

fn apply_json_string_array(
    target: &mut Vec<String>,
    value: &Value,
    warnings: &mut Vec<String>,
    key: &str,
) {
    if let Some(items) = value.as_array() {
        *target = items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect();
    } else {
        warnings.push(format!("mcp_override: {key} must be an array of strings"));
    }
}

fn apply_json_responsibility_keywords(
    policy: &mut DevsystemPolicy,
    value: &Value,
    warnings: &mut Vec<String>,
) {
    let Some(object) = value.as_object() else {
        warnings.push("mcp_override: responsibility_keywords must be an object".to_string());
        return;
    };
    policy.responsibility_keywords = object
        .iter()
        .filter_map(|(responsibility, keywords)| {
            let keywords = keywords.as_array()?;
            Some(ResponsibilityKeywordPolicy {
                responsibility: responsibility.clone(),
                keywords: keywords
                    .iter()
                    .filter_map(|keyword| keyword.as_str().map(str::to_string))
                    .collect(),
            })
        })
        .collect();
}

fn parse_u32_value(value: &str) -> Option<u32> {
    value.parse::<u32>().ok()
}

fn parse_usize_value(value: &str) -> Option<usize> {
    value.parse::<usize>().ok()
}

fn parse_f64_value(value: &str) -> Option<f64> {
    value.parse::<f64>().ok()
}

fn parse_string_array(value: &str) -> Option<Vec<String>> {
    let value = value.trim();
    let inner = value.strip_prefix('[')?.strip_suffix(']')?;
    Some(
        inner
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(unquote_policy_string)
            .collect(),
    )
}

fn unquote_policy_string(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

pub fn build_devsystem_report(
    store: &mut Store,
    request: DevsystemRequest,
) -> Result<AgentRunReport> {
    let files = normalize_files(&request.files);
    if files.is_empty() {
        bail!("dukedevsystem requires at least one project-relative file");
    }
    let policy = load_devsystem_policy(
        request.project_path.as_deref(),
        request.policy_override.as_ref(),
    )?;
    let mut stage_reports = Vec::new();
    let session = store.create_task_session(NewTaskSession {
        project_id: &request.project_id,
        query: &request.query,
        status: "running",
        phase: "planner",
        progress: 5,
        result: json!({
            "product": "dukedevsystem",
            "pipeline": [
                "planner", "memory", "architect", "coder", "test", "critic", "refactor", "memory"
            ]
        }),
    })?;
    let planner = run_planner_stage(&request.query, &files, &policy.effective);
    stage_reports.push(planner_stage_report(&planner));

    let index_run = match request.precomputed_index_run.clone() {
        Some(index_run) => Some(index_run),
        None => Some(run_devsystem_auto_index(store, &request)?),
    };
    let index_freshness = match request.project_path.as_deref() {
        Some(root) => Some(check_code_index_freshness(
            store,
            root,
            Some(request.project_id.clone()),
        )?),
        None => None,
    };
    let index_guard = index_freshness.as_ref().map(devsystem_index_guard);
    let memory_stage = run_memory_stage(
        store,
        &request.project_id,
        &request.query,
        &files,
        index_run.clone(),
        index_guard.clone(),
        request.project_path.as_deref(),
    )?;
    stage_reports.push(memory_stage_report(&memory_stage));

    let duplicate_pairs = match request.code_embedding_model.as_deref() {
        Some(model) => store.code_similarity_pairs(
            &request.project_id,
            CodeSimilarityPairOptions {
                embedding_model: model.to_string(),
                limit: request.review_limit,
                kind: None,
                file_path: None,
                min_similarity: request.duplicate_similarity,
            },
        )?,
        None => Vec::new(),
    };
    let code_review_plan = build_code_review_plan_report(
        store,
        &request.project_id,
        &request.query,
        files.clone(),
        duplicate_pairs,
        request.review_limit,
    )?;

    let mut reports = Vec::new();
    for file in &files {
        reports.push(build_file_entropy_report(
            store,
            &request.project_id,
            file,
            request.project_path.as_deref(),
            index_freshness.as_ref(),
            &policy.effective,
        )?);
    }

    let recommended_tests = reports
        .iter()
        .flat_map(|report| report.signals.affected_tests.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let recommended_test_commands =
        merge_required_test_commands(&code_review_plan.test_commands, &policy.effective);
    let quality_evidence_reports = collect_quality_evidence(
        &request,
        &recommended_test_commands,
        request.project_path.as_deref(),
    )?;
    let boundary_repair_plans =
        build_boundary_repair_plans(&reports, &recommended_test_commands, &policy.effective);
    let quality_gates = build_quality_gates(QualityGateInput {
        reports: &reports,
        boundary_repair_plans: &boundary_repair_plans,
        recommended_tests: &recommended_tests,
        recommended_test_commands: &recommended_test_commands,
        quality_evidence_reports: &quality_evidence_reports,
        index_guard: index_guard.as_ref(),
        index_run: index_run.as_ref(),
        memory_stage: &memory_stage,
        policy: &policy,
    });
    let quality_gate_summary = summarize_quality_gates(&quality_gates);

    let final_verdict = final_verdict(&reports, index_guard.as_ref(), &quality_gate_summary);
    let role_reports = build_role_reports(RoleReportInput {
        query: &request.query,
        files: &files,
        reports: &reports,
        recommended_tests: &recommended_tests,
        recommended_test_commands: &recommended_test_commands,
        quality_evidence_reports: &quality_evidence_reports,
        boundary_repair_plans: &boundary_repair_plans,
        quality_gates: &quality_gates,
        quality_gate_summary: &quality_gate_summary,
        code_review_plan: &code_review_plan,
        index_guard: index_guard.as_ref(),
        index_run: index_run.as_ref(),
        memory_stage: &memory_stage,
        policy: &policy,
    });
    stage_reports.extend(build_stage_reports(StageReportInput {
        reports: &reports,
        recommended_tests: &recommended_tests,
        recommended_test_commands: &recommended_test_commands,
        quality_evidence_reports: &quality_evidence_reports,
        boundary_repair_plans: &boundary_repair_plans,
        quality_gates: &quality_gates,
        quality_gate_summary: &quality_gate_summary,
        code_review_plan: &code_review_plan,
        final_verdict: &final_verdict,
        index_guard: index_guard.as_ref(),
        index_run: index_run.as_ref(),
        memory_stage: &memory_stage,
        policy: &policy,
    }));

    let memory_writes = if request.write_memory {
        write_pending_devsystem_memories(DevsystemMemoryWriteInput {
            store,
            project_id: &request.project_id,
            query: &request.query,
            planner: &planner,
            reports: &reports,
            boundary_repair_plans: &boundary_repair_plans,
            quality_gates: &quality_gates,
            quality_gate_summary: &quality_gate_summary,
            recommended_tests: &recommended_tests,
            recommended_test_commands: &recommended_test_commands,
            quality_evidence_reports: &quality_evidence_reports,
            final_verdict: &final_verdict,
        })?
    } else {
        MemoryWrites::empty("disabled")
    };
    let memory_ids = memory_writes.ids.clone();
    let code_symbol_ids = code_symbol_ids_for_files(store, &request.project_id, &files)?;

    let report = AgentRunReport {
        product: "dukedevsystem".to_string(),
        readiness_percent: 100,
        task_session_id: session.id.clone(),
        task_intent: request.query.clone(),
        stage_reports,
        role_reports,
        telemetry: DevsystemTelemetry {
            index_run: index_run.clone(),
            index_guard: index_guard.clone(),
            missing_signals: memory_stage.missing_signals.clone(),
            memory_agent: memory_stage,
            policy: DevsystemPolicyReport {
                source: policy.source,
                effective: policy.effective,
                warnings: policy.warnings,
            },
        },
        code_review_plan,
        file_entropy_reports: reports,
        boundary_repair_plans,
        quality_gates,
        quality_gate_summary,
        recommended_tests,
        recommended_test_commands,
        quality_evidence_reports,
        memory_writes,
        final_verdict,
    };

    store.update_task_session(
        &request.project_id,
        &session.id,
        TaskSessionUpdate {
            status: Some("completed".to_string()),
            phase: Some("done".to_string()),
            progress: Some(100),
            memory_ids: Some(memory_ids),
            code_symbol_ids: Some(code_symbol_ids),
            file_paths: Some(files),
            test_paths: Some(report.recommended_tests.clone()),
            summary: Some(Some(format!(
                "dukedevsystem advisory run completed with final verdict `{}`.",
                report.final_verdict
            ))),
            result: Some(serde_json::to_value(&report)?),
        },
    )?;

    Ok(report)
}

fn run_devsystem_auto_index(
    store: &mut Store,
    request: &DevsystemRequest,
) -> Result<IndexRunSummary> {
    if !request.auto_index {
        return Ok(IndexRunSummary::disabled(
            request.project_id.clone(),
            request.project_path.clone(),
            "auto_index disabled by request",
            request.full_rebuild,
            request.embed_symbols,
            request.embed_symbol_limit,
        ));
    }
    let Some(root) = request.project_path.as_deref() else {
        return Ok(IndexRunSummary::disabled(
            request.project_id.clone(),
            None,
            "auto_index requested but project_path was not supplied",
            request.full_rebuild,
            request.embed_symbols,
            request.embed_symbol_limit,
        ));
    };
    if request.embed_symbols {
        bail!(
            "dukememory_devsystem embed_symbols requires precomputed async index telemetry from the MCP layer"
        );
    }
    let report = index_project(
        store,
        root,
        Some(request.project_id.clone()),
        request.full_rebuild,
    )?;
    Ok(IndexRunSummary::from_index_report(
        report,
        request.embed_symbols,
        request.embed_symbol_limit,
        0,
    ))
}

pub fn build_file_entropy_report(
    store: &Store,
    project_id: &str,
    file_path: &str,
    project_path: Option<&Path>,
    index_freshness: Option<&CodeFreshnessReport>,
    policy: &DevsystemPolicy,
) -> Result<FileEntropyReport> {
    if policy_file_match(file_path, &policy.ignored_file_patterns) {
        return policy_suppressed_file_report(
            store,
            project_id,
            file_path,
            "ignored by devsystem policy",
        );
    }
    if policy_file_match(file_path, &policy.generated_file_patterns) {
        return policy_suppressed_file_report(
            store,
            project_id,
            file_path,
            "generated file by devsystem policy",
        );
    }
    let indexed_file = store
        .code_files_for_project(project_id)?
        .into_iter()
        .find(|file| file.path == file_path);
    let symbols = store.code_symbols_for_file(project_id, file_path)?;
    let mut missing_signals = Vec::new();
    let source = read_project_file(project_path, file_path);
    if source.is_none() {
        match project_path {
            Some(_) => {
                missing_signals.push("source: file not readable from project_path".to_string())
            }
            None => missing_signals.push("source: project_path not supplied".to_string()),
        }
    }
    if indexed_file.is_none() {
        missing_signals.push("code_index: file not indexed".to_string());
    }
    if let Some(freshness) = index_freshness {
        if freshness.stale_files.iter().any(|path| path == file_path) {
            missing_signals.push("code_index: file is stale".to_string());
        }
        if freshness.missing_files.iter().any(|path| path == file_path) {
            missing_signals.push("code_index: file missing from index".to_string());
        }
        if freshness.deleted_files.iter().any(|path| path == file_path) {
            missing_signals.push("code_index: indexed file was deleted from disk".to_string());
        }
    }
    let analysis_source = source.clone().or_else(|| symbol_body_source(&symbols));
    let lines = indexed_file
        .as_ref()
        .map(|file| file.line_count)
        .or_else(|| {
            analysis_source
                .as_ref()
                .map(|text| text.lines().count() as u32)
        })
        .unwrap_or(0);
    let relations = code_relations_for_file(store, project_id, &symbols)?;
    let related_symbols = relation_symbol_map(store, project_id, &symbols)?;
    let fan_in = fan_in(file_path, &relations, &related_symbols);
    let fan_out = fan_out(file_path, &relations, &related_symbols);
    let public_api_surface = public_api_surface(&symbols);
    let semantic_sections =
        semantic_sections(file_path, &symbols, analysis_source.as_deref(), policy);
    let responsibility_summary = responsibility_summary(
        &semantic_sections,
        file_path,
        analysis_source.as_deref(),
        policy,
    );
    let responsibility_count = responsibility_summary.len().max(1);
    let affected_tests = store.affected_test_files(project_id, &[file_path.to_string()], 5, 100)?;
    let git = git_history(project_path, file_path, &mut missing_signals);
    let coverage = coverage_for_file(project_path, file_path, &mut missing_signals);
    let task_history_touches = task_history_touches(store, project_id, file_path)?;
    let unrelated_reasons_to_change = unrelated_reasons(&responsibility_summary);
    let cyclomatic_complexity = analysis_source
        .as_deref()
        .map(estimate_cyclomatic_complexity);

    let signals = FileEntropySignals {
        lines,
        fan_in,
        fan_out,
        public_api_surface,
        cyclomatic_complexity,
        semantic_sections,
        unrelated_reasons_to_change,
        change_frequency: git.as_ref().map(|history| history.change_frequency),
        number_of_authors: git.as_ref().map(|history| history.number_of_authors),
        test_coverage: coverage,
        co_change_hotspots: git
            .as_ref()
            .map(|history| history.co_change_hotspots.clone())
            .unwrap_or_default(),
        affected_tests,
        task_history_touches,
    };
    let score = entropy_score(&signals, responsibility_count, policy);
    let verdict = entropy_verdict(score, responsibility_count, &signals, policy);
    let boundary_repair_suggestions =
        boundary_repair_suggestions(&responsibility_summary, &verdict);

    Ok(FileEntropyReport {
        file_path: file_path.to_string(),
        score,
        verdict,
        responsibility_count,
        responsibility_summary,
        signals,
        missing_signals,
        boundary_repair_suggestions,
    })
}

fn normalize_files(files: &[String]) -> Vec<String> {
    files
        .iter()
        .map(|file| file.trim().trim_start_matches("./").to_string())
        .filter(|file| !file.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn policy_suppressed_file_report(
    store: &Store,
    project_id: &str,
    file_path: &str,
    reason: &str,
) -> Result<FileEntropyReport> {
    let indexed_file = store
        .code_files_for_project(project_id)?
        .into_iter()
        .find(|file| file.path == file_path);
    let affected_tests = store.affected_test_files(project_id, &[file_path.to_string()], 5, 100)?;
    Ok(FileEntropyReport {
        file_path: file_path.to_string(),
        score: 0,
        verdict: "ok".to_string(),
        responsibility_count: 1,
        responsibility_summary: vec![reason.to_string()],
        signals: FileEntropySignals {
            lines: indexed_file.map(|file| file.line_count).unwrap_or(0),
            fan_in: 0,
            fan_out: 0,
            public_api_surface: 0,
            cyclomatic_complexity: None,
            semantic_sections: vec![reason.to_string()],
            unrelated_reasons_to_change: Vec::new(),
            change_frequency: None,
            number_of_authors: None,
            test_coverage: None,
            co_change_hotspots: Vec::new(),
            affected_tests,
            task_history_touches: task_history_touches(store, project_id, file_path)?,
        },
        missing_signals: vec![format!("policy: {reason}")],
        boundary_repair_suggestions: Vec::new(),
    })
}

fn policy_file_match(file_path: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|pattern| wildcard_match(file_path, pattern) || file_path.contains(pattern))
}

fn wildcard_match(value: &str, pattern: &str) -> bool {
    let pattern = pattern.trim();
    if pattern.is_empty() {
        return false;
    }
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return value == pattern;
    }
    let parts = pattern.split('*').collect::<Vec<_>>();
    let mut remaining = value;
    if let Some(first) = parts.first()
        && !first.is_empty()
    {
        let Some(stripped) = remaining.strip_prefix(first) else {
            return false;
        };
        remaining = stripped;
    }
    for part in parts
        .iter()
        .skip(1)
        .take(parts.len().saturating_sub(2))
        .filter(|part| !part.is_empty())
    {
        let Some(index) = remaining.find(part) else {
            return false;
        };
        remaining = &remaining[index + part.len()..];
    }
    if let Some(last) = parts.last()
        && !last.is_empty()
    {
        return remaining.ends_with(last);
    }
    true
}

fn run_planner_stage(
    query: &str,
    files: &[String],
    policy: &DevsystemPolicy,
) -> PlannerStageOutput {
    let likely_risk = if files.len() > 5 {
        "broad_change".to_string()
    } else {
        "bounded_change".to_string()
    };
    PlannerStageOutput {
        task_intent: query.to_string(),
        scope_files: files.to_vec(),
        constraints: vec![
            "MCP-facing contract only".to_string(),
            "advisory recommendations, no auto-applied boundary repair".to_string(),
            "automatic memory writes remain pending".to_string(),
            "project-scoped dukememory access only".to_string(),
        ],
        acceptance_criteria: vec![
            "emit structured stage reports".to_string(),
            "include File Entropy Score reports for all touched files".to_string(),
            "include missing telemetry signals instead of failing".to_string(),
            "include affected tests and executable test commands".to_string(),
            format!(
                "apply devsystem policy split_score_threshold={} split_responsibility_count={}",
                policy.split_score_threshold, policy.split_responsibility_count
            ),
        ],
        likely_risk,
    }
}

fn planner_stage_report(planner: &PlannerStageOutput) -> AgentStageReport {
    stage_report(
        "planner",
        10,
        "Task intent, touched files, constraints, acceptance criteria, and likely risk were normalized.",
        json!(planner),
    )
}

fn run_memory_stage(
    store: &Store,
    project_id: &str,
    query: &str,
    files: &[String],
    index_run: Option<IndexRunSummary>,
    index_guard: Option<DevsystemIndexGuard>,
    project_path: Option<&Path>,
) -> Result<MemoryStageOutput> {
    let mut missing_signals = Vec::new();
    if project_path.is_none() {
        missing_signals.push("code_index_freshness: project_path not supplied".to_string());
    }
    if index_guard.is_none() {
        missing_signals.push("code_index_freshness: not evaluated".to_string());
    }
    match &index_run {
        Some(run) if run.enabled => {}
        Some(run) => missing_signals.push(format!(
            "code_index: {}",
            run.reason
                .as_deref()
                .unwrap_or("auto_index did not run for an unspecified reason")
        )),
        None => missing_signals.push("code_index: auto_index telemetry not available".to_string()),
    }

    let active_memories = store
        .list(
            project_id,
            ListOptions {
                limit: 8,
                offset: 0,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
            },
        )?
        .into_iter()
        .map(memory_summary)
        .collect::<Vec<_>>();
    let pending_memories = store
        .list(
            project_id,
            ListOptions {
                limit: 8,
                offset: 0,
                status: StatusFilter::One(MemoryStatus::Pending),
                kind: None,
                memory_tier: None,
            },
        )?
        .into_iter()
        .map(memory_summary)
        .collect::<Vec<_>>();
    let relevant_memories = store
        .search(
            project_id,
            SearchOptions {
                query: query.to_string(),
                limit: 8,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
            },
        )?
        .into_iter()
        .map(memory_summary)
        .collect::<Vec<_>>();
    let recent_sessions = store.list_task_sessions(project_id, None, 20)?;
    let recent_task_sessions = recent_sessions
        .iter()
        .take(8)
        .cloned()
        .map(task_session_summary)
        .collect::<Vec<_>>();
    let file_task_history = files
        .iter()
        .map(|file| file_task_history_summary(file, &recent_sessions))
        .collect::<Vec<_>>();
    let code_memories = store
        .search_code_memories(
            project_id,
            CodeMemorySearchOptions {
                query: None,
                limit: 30,
                status: "any".to_string(),
                kind: None,
                symbol_ids: Vec::new(),
                file_paths: files.to_vec(),
            },
        )?
        .into_iter()
        .map(code_memory_summary)
        .collect::<Vec<_>>();
    let stale_code_memories = store
        .search_code_memories(
            project_id,
            CodeMemorySearchOptions {
                query: None,
                limit: 30,
                status: "any".to_string(),
                kind: None,
                symbol_ids: Vec::new(),
                file_paths: files.to_vec(),
            },
        )?
        .into_iter()
        .filter(|memory| memory.link_status == "stale")
        .map(code_memory_summary)
        .collect::<Vec<_>>();
    Ok(MemoryStageOutput {
        active_memories,
        pending_memories,
        relevant_memories,
        recent_task_sessions,
        file_task_history,
        code_memories,
        stale_code_memories,
        index_run,
        index_guard,
        missing_signals,
    })
}

fn memory_stage_report(memory: &MemoryStageOutput) -> AgentStageReport {
    stage_report(
        "memory",
        25,
        "Project-scoped dukememory telemetry was loaded from active/pending memories, task sessions, code memories, stale candidates, and code-index freshness.",
        json!(memory),
    )
}

fn merge_required_test_commands(
    commands: &[TestCommandRecommendation],
    policy: &DevsystemPolicy,
) -> Vec<TestCommandRecommendation> {
    let mut seen = BTreeSet::new();
    let mut merged = Vec::new();
    for command in commands {
        if seen.insert(command.command.clone()) {
            merged.push(command.clone());
        }
    }
    for command in &policy.required_test_commands {
        if seen.insert(command.clone()) {
            merged.push(TestCommandRecommendation {
                command: command.clone(),
                reason: "required by dukedevsystem policy".to_string(),
                confidence: 0.95,
            });
        }
    }
    merged
}

fn collect_quality_evidence(
    request: &DevsystemRequest,
    recommended_test_commands: &[TestCommandRecommendation],
    project_path: Option<&Path>,
) -> Result<Vec<QualityEvidenceReport>> {
    let max_commands = request.max_evidence_commands.clamp(1, 50);
    let allowed_filter = request
        .allowed_evidence_commands
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let commands = recommended_test_commands
        .iter()
        .filter(|command| {
            !request.run_evidence
                || allowed_filter.is_empty()
                || allowed_filter.contains(&command.command)
        })
        .take(max_commands)
        .collect::<Vec<_>>();
    if commands.is_empty() {
        return Ok(Vec::new());
    }
    let mut allowed = recommended_test_commands
        .iter()
        .map(|command| command.command.clone())
        .collect::<BTreeSet<_>>();
    allowed.extend(allowed_filter);

    commands
        .into_iter()
        .map(|command| {
            if !request.run_evidence {
                return Ok(quality_evidence_report(
                    command, "not_run", None, 0, "", "", "disabled",
                ));
            }
            if !allowed.contains(&command.command) {
                return Ok(quality_evidence_report(
                    command,
                    "skipped",
                    None,
                    0,
                    "",
                    "command was not present in the quality evidence allowlist",
                    "allowlist",
                ));
            }
            run_quality_evidence_command(
                command,
                request.evidence_timeout_seconds.clamp(1, 3600),
                project_path,
            )
        })
        .collect()
}

fn run_quality_evidence_command(
    command: &TestCommandRecommendation,
    timeout_seconds: u64,
    project_path: Option<&Path>,
) -> Result<QualityEvidenceReport> {
    let started = Instant::now();
    let argv = match parse_evidence_command(&command.command) {
        Ok(argv) => argv,
        Err(error) => {
            return Ok(quality_evidence_report(
                command,
                "skipped",
                None,
                started.elapsed().as_millis(),
                "",
                &error,
                "parser",
            ));
        }
    };
    let Some((program, args)) = argv.split_first() else {
        return Ok(quality_evidence_report(
            command,
            "skipped",
            None,
            started.elapsed().as_millis(),
            "",
            "command was empty",
            "parser",
        ));
    };
    if is_shell_program(program) {
        return Ok(quality_evidence_report(
            command,
            "skipped",
            None,
            started.elapsed().as_millis(),
            "",
            "shell executables are not allowed for quality evidence",
            "safety",
        ));
    }

    let mut process = Command::new(program);
    process.args(args);
    if let Some(root) = project_path {
        process.current_dir(root);
    }
    process.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            return Ok(quality_evidence_report(
                command,
                "skipped",
                None,
                started.elapsed().as_millis(),
                "",
                &format!("failed to start command: {error}"),
                "process",
            ));
        }
    };
    let timeout = Duration::from_secs(timeout_seconds);
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            let exit_code = output.status.code();
            let status = if output.status.success() {
                "passed"
            } else {
                "failed"
            };
            return Ok(quality_evidence_report(
                command,
                status,
                exit_code,
                started.elapsed().as_millis(),
                &String::from_utf8_lossy(&output.stdout),
                &String::from_utf8_lossy(&output.stderr),
                "process",
            ));
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(quality_evidence_report(
                command,
                "timed_out",
                None,
                started.elapsed().as_millis(),
                "",
                &format!("command exceeded {timeout_seconds}s timeout"),
                "timeout",
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn quality_evidence_report(
    command: &TestCommandRecommendation,
    status: &str,
    exit_code: Option<i32>,
    duration_ms: u128,
    stdout: &str,
    stderr: &str,
    evidence_source: &str,
) -> QualityEvidenceReport {
    QualityEvidenceReport {
        command: command.command.clone(),
        reason: command.reason.clone(),
        status: status.to_string(),
        exit_code,
        duration_ms,
        stdout_excerpt: evidence_excerpt(stdout),
        stderr_excerpt: evidence_excerpt(stderr),
        evidence_source: evidence_source.to_string(),
        affects_gate_ids: vec!["test_plan".to_string(), "quality_evidence".to_string()],
    }
}

fn parse_evidence_command(command: &str) -> std::result::Result<Vec<String>, String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in command.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        match quote {
            Some(active) if ch == active => quote = None,
            Some(_) => current.push(ch),
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
            }
            None if matches!(ch, '|' | ';' | '&' | '>' | '<' | '`') => {
                return Err(format!("shell control character `{ch}` is not allowed"));
            }
            None => current.push(ch),
        }
    }
    if escaped {
        current.push('\\');
    }
    if quote.is_some() {
        return Err("unterminated quote in evidence command".to_string());
    }
    if !current.is_empty() {
        parts.push(current);
    }
    Ok(parts)
}

fn is_shell_program(program: &str) -> bool {
    let name = Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(program);
    matches!(
        name,
        "sh" | "bash" | "zsh" | "fish" | "csh" | "tcsh" | "dash" | "ksh"
    )
}

fn evidence_excerpt(output: &str) -> String {
    const MAX_CHARS: usize = 4000;
    let trimmed = output.trim();
    if trimmed.chars().count() <= MAX_CHARS {
        return trimmed.to_string();
    }
    let mut excerpt = trimmed.chars().take(MAX_CHARS).collect::<String>();
    excerpt.push_str("...");
    excerpt
}

fn memory_summary(memory: Memory) -> MemorySummary {
    MemorySummary {
        id: memory.id,
        kind: memory.kind,
        status: memory.status,
        memory_tier: memory.memory_tier,
        source: memory.source,
        tags: memory.tags,
        confidence: memory.confidence,
        importance: memory.importance,
        score: memory.score,
        body_excerpt: excerpt(&memory.body, 240),
        updated_at: memory.updated_at,
    }
}

fn task_session_summary(session: TaskSession) -> TaskSessionSummary {
    TaskSessionSummary {
        id: session.id,
        status: session.status,
        phase: session.phase,
        progress: session.progress,
        query: session.query,
        file_paths: session.file_paths,
        test_paths: session.test_paths,
        summary: session.summary,
        updated_at: session.updated_at,
    }
}

fn file_task_history_summary(file_path: &str, sessions: &[TaskSession]) -> FileTaskHistorySummary {
    let recent_session_ids = sessions
        .iter()
        .filter(|session| session.file_paths.iter().any(|path| path == file_path))
        .take(8)
        .map(|session| session.id.clone())
        .collect::<Vec<_>>();
    FileTaskHistorySummary {
        file_path: file_path.to_string(),
        touches: recent_session_ids.len(),
        recent_session_ids,
    }
}

fn code_memory_summary(memory: CodeMemory) -> CodeMemorySummary {
    CodeMemorySummary {
        id: memory.id,
        kind: memory.kind,
        status: memory.status,
        link_status: memory.link_status,
        file_path: memory.file_path,
        symbol_id: memory.symbol_id,
        symbol_name: memory.symbol_name,
        confidence: memory.confidence,
        quality_score: memory.quality_score,
        usage_count: memory.usage_count,
        contradiction_risk: memory.contradiction_risk,
        body_excerpt: excerpt(&memory.body, 240),
        updated_at: memory.updated_at,
    }
}

fn excerpt(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= limit {
        compact
    } else {
        let mut shortened = compact
            .chars()
            .take(limit.saturating_sub(3))
            .collect::<String>();
        shortened.push_str("...");
        shortened
    }
}

fn read_project_file(project_path: Option<&Path>, file_path: &str) -> Option<String> {
    let root = project_path?;
    std::fs::read_to_string(root.join(file_path)).ok()
}

fn symbol_body_source(symbols: &[CodeSymbol]) -> Option<String> {
    let body = symbols
        .iter()
        .filter(|symbol| !symbol.body.trim().is_empty())
        .map(|symbol| symbol.body.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if body.is_empty() { None } else { Some(body) }
}

fn code_relations_for_file(
    store: &Store,
    project_id: &str,
    symbols: &[CodeSymbol],
) -> Result<Vec<CodeRelation>> {
    let symbol_ids = symbols
        .iter()
        .map(|symbol| symbol.id.clone())
        .collect::<Vec<_>>();
    let (_, relations) = store.code_graph_for_symbols(project_id, &symbol_ids, 500)?;
    Ok(relations)
}

fn relation_symbol_map(
    store: &Store,
    project_id: &str,
    symbols: &[CodeSymbol],
) -> Result<HashMap<String, CodeSymbol>> {
    let symbol_ids = symbols
        .iter()
        .map(|symbol| symbol.id.clone())
        .collect::<Vec<_>>();
    let (symbols, _) = store.code_graph_for_symbols(project_id, &symbol_ids, 500)?;
    Ok(symbols
        .into_iter()
        .map(|symbol| (symbol.id.clone(), symbol))
        .collect())
}

fn fan_in(
    file_path: &str,
    relations: &[CodeRelation],
    symbols: &HashMap<String, CodeSymbol>,
) -> usize {
    relations
        .iter()
        .filter(|relation| relation.from_file_path != file_path)
        .filter(|relation| {
            relation
                .target_symbol_id
                .as_ref()
                .and_then(|id| symbols.get(id))
                .map(|symbol| symbol.file_path == file_path)
                .unwrap_or(false)
        })
        .map(|relation| relation.from_file_path.clone())
        .collect::<BTreeSet<_>>()
        .len()
}

fn fan_out(
    file_path: &str,
    relations: &[CodeRelation],
    symbols: &HashMap<String, CodeSymbol>,
) -> usize {
    relations
        .iter()
        .filter(|relation| relation.from_file_path == file_path)
        .filter_map(|relation| {
            relation
                .target_symbol_id
                .as_ref()
                .and_then(|id| symbols.get(id))
                .map(|symbol| symbol.file_path.clone())
        })
        .filter(|target_file| target_file != file_path)
        .collect::<BTreeSet<_>>()
        .len()
}

fn public_api_surface(symbols: &[CodeSymbol]) -> usize {
    symbols
        .iter()
        .filter(|symbol| {
            let signature = symbol.signature.trim();
            signature.starts_with("pub ")
                || signature.starts_with("pub(")
                || signature.starts_with("export ")
                || signature.starts_with("public ")
                || signature.contains(" pub ")
        })
        .count()
}

fn semantic_sections(
    file_path: &str,
    symbols: &[CodeSymbol],
    source: Option<&str>,
    policy: &DevsystemPolicy,
) -> Vec<String> {
    let mut haystack = String::new();
    haystack.push_str(file_path);
    haystack.push('\n');
    for symbol in symbols {
        haystack.push_str(&symbol.name);
        haystack.push('\n');
        haystack.push_str(&symbol.signature);
        haystack.push('\n');
    }
    if let Some(source) = source {
        haystack.push_str(source);
    }
    let lower = haystack.to_ascii_lowercase();
    let mut sections = BTreeSet::new();
    for item in &policy.responsibility_keywords {
        if item
            .keywords
            .iter()
            .any(|keyword| lower.contains(&keyword.to_ascii_lowercase()))
        {
            sections.insert(item.responsibility.clone());
        }
    }
    if sections.is_empty() && looks_like_static_metadata(file_path, source, policy) {
        sections.insert("static metadata".to_string());
    }
    if sections.is_empty() {
        sections.insert("implementation".to_string());
    }
    sections.into_iter().collect()
}

fn responsibility_summary(
    semantic_sections: &[String],
    file_path: &str,
    source: Option<&str>,
    policy: &DevsystemPolicy,
) -> Vec<String> {
    if looks_like_static_metadata(file_path, source, policy) {
        return vec!["static country metadata".to_string()];
    }
    semantic_sections.to_vec()
}

fn looks_like_static_metadata(
    file_path: &str,
    source: Option<&str>,
    policy: &DevsystemPolicy,
) -> bool {
    let lower_path = file_path.to_ascii_lowercase();
    if policy.static_metadata_patterns.iter().any(|pattern| {
        wildcard_match(&lower_path, &pattern.to_ascii_lowercase())
            || lower_path.contains(&pattern.to_ascii_lowercase())
    }) {
        let source = source.unwrap_or("");
        let lower = source.to_ascii_lowercase();
        let has_dynamic_markers = [
            "fetch(", "axios", "select ", "insert ", "update ", "delete ", "async ", "await ",
            "http", "db.",
        ]
        .iter()
        .any(|needle| lower.contains(needle));
        return !has_dynamic_markers;
    }
    false
}

fn unrelated_reasons(responsibilities: &[String]) -> Vec<String> {
    if responsibilities.len() <= 1 {
        return Vec::new();
    }
    responsibilities
        .iter()
        .map(|responsibility| format!("change because of {responsibility}"))
        .collect()
}

fn estimate_cyclomatic_complexity(source: &str) -> u32 {
    let lower = source.to_ascii_lowercase();
    let markers = [
        " if ",
        "\nif ",
        " else if ",
        " match ",
        " for ",
        "\nfor ",
        " while ",
        "&&",
        "||",
        "?;",
        " catch ",
        " except ",
    ];
    1 + markers
        .iter()
        .map(|marker| lower.matches(marker).count() as u32)
        .sum::<u32>()
}

#[derive(Debug)]
struct GitHistory {
    change_frequency: usize,
    number_of_authors: usize,
    co_change_hotspots: Vec<String>,
}

fn git_history(
    project_path: Option<&Path>,
    file_path: &str,
    missing_signals: &mut Vec<String>,
) -> Option<GitHistory> {
    let Some(root) = project_path else {
        missing_signals.push("git_history: project_path not supplied".to_string());
        return None;
    };
    let inside_git = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("rev-parse")
        .arg("--is-inside-work-tree")
        .output()
        .ok()
        .filter(|output| output.status.success());
    if inside_git.is_none() {
        missing_signals.push("git_history: .git repository not available".to_string());
        return None;
    }
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("log")
        .arg("--follow")
        .arg("--format=%an")
        .arg("--")
        .arg(file_path)
        .output()
        .ok()
        .or_else(|| {
            missing_signals.push("git_history: git log command failed".to_string());
            None
        })?;
    if !output.status.success() {
        missing_signals.push("git_history: git log failed".to_string());
        return None;
    }
    let authors = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let co_change_hotspots = git_co_change_hotspots(root, file_path, missing_signals);
    Some(GitHistory {
        change_frequency: authors.len(),
        number_of_authors: authors.iter().collect::<BTreeSet<_>>().len(),
        co_change_hotspots,
    })
}

fn git_co_change_hotspots(
    root: &Path,
    file_path: &str,
    missing_signals: &mut Vec<String>,
) -> Vec<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .arg("log")
        .arg("--follow")
        .arg("--name-only")
        .arg("--format=__DUKEMEMORY_COMMIT__")
        .arg("--")
        .arg(file_path)
        .output();
    let Ok(output) = output else {
        missing_signals.push("git_history: co-change query failed".to_string());
        return Vec::new();
    };
    if !output.status.success() {
        missing_signals.push("git_history: co-change query returned non-zero status".to_string());
        return Vec::new();
    }
    let mut counts = HashMap::<String, usize>::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let changed = line.trim();
        if changed.is_empty() || changed == "__DUKEMEMORY_COMMIT__" || changed == file_path {
            continue;
        }
        *counts.entry(changed.to_string()).or_insert(0) += 1;
    }
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked
        .into_iter()
        .take(8)
        .map(|(path, count)| format!("{path} ({count})"))
        .collect()
}

fn coverage_for_file(
    project_path: Option<&Path>,
    file_path: &str,
    missing_signals: &mut Vec<String>,
) -> Option<f64> {
    let Some(root) = project_path else {
        missing_signals.push("coverage: project_path not supplied".to_string());
        return None;
    };
    for relative in [
        "lcov.info",
        "coverage/lcov.info",
        "target/llvm-cov/lcov.info",
        "target/coverage/lcov.info",
    ] {
        let path = root.join(relative);
        if path.exists() {
            return parse_lcov_for_file(&path, file_path).or_else(|| {
                missing_signals.push(format!("coverage: `{relative}` has no entry for file"));
                None
            });
        }
    }
    missing_signals.push("coverage: no lcov report found".to_string());
    None
}

fn parse_lcov_for_file(path: &Path, file_path: &str) -> Option<f64> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut in_file = false;
    let mut found = false;
    let mut lines_found = None;
    let mut lines_hit = None;
    let mut da_found = 0.0;
    let mut da_hit = 0.0;
    for line in content.lines() {
        if let Some(sf) = line.strip_prefix("SF:") {
            in_file = sf.ends_with(file_path) || sf == file_path;
            found |= in_file;
            lines_found = None;
            lines_hit = None;
            da_found = 0.0;
            da_hit = 0.0;
        } else if in_file && let Some(value) = line.strip_prefix("LF:") {
            lines_found = value.parse::<f64>().ok();
        } else if in_file && let Some(value) = line.strip_prefix("LH:") {
            lines_hit = value.parse::<f64>().ok();
        } else if in_file && let Some(value) = line.strip_prefix("DA:") {
            if let Some((_, hits)) = value.split_once(',') {
                da_found += 1.0;
                if hits.parse::<u64>().unwrap_or(0) > 0 {
                    da_hit += 1.0;
                }
            }
        } else if in_file && line == "end_of_record" {
            if let (Some(found), Some(hit)) = (lines_found, lines_hit) {
                return if found > 0.0 {
                    Some(hit / found)
                } else {
                    Some(0.0)
                };
            }
            if da_found > 0.0 {
                return Some(da_hit / da_found);
            }
        }
    }
    if found { Some(0.0) } else { None }
}

fn task_history_touches(store: &Store, project_id: &str, file_path: &str) -> Result<usize> {
    Ok(store
        .list_task_sessions(project_id, None, 100)?
        .iter()
        .filter(|session| session.file_paths.iter().any(|path| path == file_path))
        .count())
}

fn entropy_score(
    signals: &FileEntropySignals,
    responsibility_count: usize,
    policy: &DevsystemPolicy,
) -> u32 {
    let mut score = 0_u32;
    score += responsibility_count.saturating_sub(1).min(5) as u32 * 14;
    if responsibility_count > 1 {
        score += (signals.lines / policy.line_signal_step).min(policy.max_line_signal);
    }
    score += signals.fan_in.min(8) as u32;
    score += signals.fan_out.min(8) as u32;
    score += (signals.public_api_surface / 2).min(8) as u32;
    score += signals
        .cyclomatic_complexity
        .map(|value| value.saturating_sub(8).min(15))
        .unwrap_or(0);
    score += signals
        .change_frequency
        .map(|value| value.saturating_sub(8).min(12) as u32)
        .unwrap_or(0);
    score += signals
        .number_of_authors
        .map(|value| value.saturating_sub(2).min(8) as u32)
        .unwrap_or(0);
    score += signals.co_change_hotspots.len().min(6) as u32;
    score += signals.task_history_touches.saturating_sub(2).min(8) as u32;
    if signals
        .test_coverage
        .is_some_and(|coverage| coverage < policy.low_coverage_threshold)
    {
        score += 8;
    }
    score.min(100)
}

fn entropy_verdict(
    score: u32,
    responsibility_count: usize,
    signals: &FileEntropySignals,
    policy: &DevsystemPolicy,
) -> String {
    if responsibility_count <= 1 && signals.unrelated_reasons_to_change.is_empty() {
        return "ok".to_string();
    }
    if responsibility_count >= policy.split_responsibility_count
        || (score >= policy.split_score_threshold
            && responsibility_count >= policy.boundary_repair_responsibility_count)
    {
        "split_required".to_string()
    } else if score >= policy.boundary_repair_score_threshold
        && responsibility_count >= policy.boundary_repair_responsibility_count
    {
        "boundary_repair_recommended".to_string()
    } else {
        "watch".to_string()
    }
}

fn boundary_repair_suggestions(responsibilities: &[String], verdict: &str) -> Vec<String> {
    if !matches!(verdict, "boundary_repair_recommended" | "split_required") {
        return Vec::new();
    }
    responsibilities
        .iter()
        .filter(|responsibility| responsibility.as_str() != "implementation")
        .map(|responsibility| {
            format!(
                "Extract `{}` into a stable boundary and keep the original file as an orchestrator only if concepts still change together.",
                responsibility.replace(' ', "_")
            )
        })
        .collect()
}

fn build_boundary_repair_plans(
    reports: &[FileEntropyReport],
    required_tests: &[TestCommandRecommendation],
    policy: &DevsystemPolicy,
) -> Vec<BoundaryRepairPlan> {
    reports
        .iter()
        .filter(|report| {
            matches!(
                report.verdict.as_str(),
                "boundary_repair_recommended" | "split_required"
            )
        })
        .filter(|report| !policy_file_match(&report.file_path, &policy.ignored_file_patterns))
        .filter(|report| !policy_file_match(&report.file_path, &policy.generated_file_patterns))
        .filter(|report| {
            !policy_file_match(&report.file_path, &policy.static_metadata_patterns)
                || report.responsibility_count > 1
        })
        .map(|report| build_boundary_repair_plan(report, required_tests))
        .collect()
}

struct QualityGateInput<'a> {
    reports: &'a [FileEntropyReport],
    boundary_repair_plans: &'a [BoundaryRepairPlan],
    recommended_tests: &'a [String],
    recommended_test_commands: &'a [TestCommandRecommendation],
    quality_evidence_reports: &'a [QualityEvidenceReport],
    index_guard: Option<&'a DevsystemIndexGuard>,
    index_run: Option<&'a IndexRunSummary>,
    memory_stage: &'a MemoryStageOutput,
    policy: &'a PolicyLoadResult,
}

fn build_quality_gates(input: QualityGateInput<'_>) -> Vec<QualityGate> {
    let mut gates = Vec::new();
    gates.push(index_freshness_gate(input.index_guard, input.index_run));
    gates.push(test_coverage_gate(
        input.recommended_tests,
        input.recommended_test_commands,
    ));
    gates.push(quality_evidence_gate(input.quality_evidence_reports));
    gates.extend(boundary_repair_gates(
        input.reports,
        input.boundary_repair_plans,
    ));
    gates.extend(coverage_gates(input.reports, &input.policy.effective));
    gates.push(stale_code_memory_gate(input.memory_stage));
    gates.push(policy_gate(input.policy));
    gates
}

fn index_freshness_gate(
    index_guard: Option<&DevsystemIndexGuard>,
    index_run: Option<&IndexRunSummary>,
) -> QualityGate {
    if let Some(run) = index_run
        && !run.enabled
    {
        let reason = run
            .reason
            .as_deref()
            .unwrap_or("auto_index did not run for an unspecified reason");
        return quality_gate(
            "code_index_freshness",
            "needs_human_decision",
            "warning",
            format!("Code index freshness could not be guaranteed because {reason}."),
            Vec::new(),
            vec![
                "Enable auto_index and pass project_path, or run dukememory_code_index before dukememory_devsystem.".to_string(),
            ],
        );
    }
    match index_guard {
        Some(guard) if guard.fresh => quality_gate(
            "code_index_freshness",
            "pass",
            "info",
            "Code index freshness telemetry is clean.",
            Vec::new(),
            Vec::new(),
        ),
        Some(guard) => {
            let files = guard
                .stale_files
                .iter()
                .chain(guard.missing_files.iter())
                .chain(guard.deleted_files.iter())
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            quality_gate(
                "code_index_freshness",
                "needs_human_decision",
                "warning",
                "Code index is stale or incomplete.",
                files,
                vec!["Run dukememory_code_index for this project_path.".to_string()],
            )
        }
        None => quality_gate(
            "code_index_freshness",
            "needs_human_decision",
            "warning",
            "Code index freshness was not evaluated because project_path was not supplied.",
            Vec::new(),
            vec!["Call dukememory_devsystem with project_path.".to_string()],
        ),
    }
}

fn test_coverage_gate(
    recommended_tests: &[String],
    recommended_test_commands: &[TestCommandRecommendation],
) -> QualityGate {
    if !recommended_test_commands.is_empty() {
        quality_gate(
            "test_plan",
            "pass",
            "info",
            "Executable test commands are available.",
            recommended_tests.to_vec(),
            recommended_test_commands
                .iter()
                .map(|command| command.command.clone())
                .collect(),
        )
    } else {
        quality_gate(
            "test_plan",
            "needs_human_decision",
            "warning",
            "No affected tests or fallback test commands were identified.",
            Vec::new(),
            vec!["Define or run an appropriate project test suite.".to_string()],
        )
    }
}

fn quality_evidence_gate(evidence: &[QualityEvidenceReport]) -> QualityGate {
    if evidence.is_empty() {
        return quality_gate(
            "quality_evidence",
            "pass",
            "info",
            "No executable quality evidence commands were available.",
            Vec::new(),
            Vec::new(),
        );
    }
    let failed = evidence
        .iter()
        .filter(|report| report.status == "failed")
        .map(|report| report.command.clone())
        .collect::<Vec<_>>();
    if !failed.is_empty() {
        return quality_gate(
            "quality_evidence",
            "blocked_by_quality_gate",
            "blocker",
            "One or more quality evidence commands failed.",
            Vec::new(),
            failed,
        );
    }
    let timed_out = evidence
        .iter()
        .filter(|report| report.status == "timed_out")
        .map(|report| report.command.clone())
        .collect::<Vec<_>>();
    if !timed_out.is_empty() {
        return quality_gate(
            "quality_evidence",
            "needs_human_decision",
            "warning",
            "One or more quality evidence commands timed out.",
            Vec::new(),
            timed_out,
        );
    }
    let skipped = evidence
        .iter()
        .filter(|report| report.status == "skipped")
        .map(|report| format!("{} ({})", report.command, report.stderr_excerpt))
        .collect::<Vec<_>>();
    if !skipped.is_empty() {
        return quality_gate(
            "quality_evidence",
            "needs_human_decision",
            "warning",
            "One or more quality evidence commands were skipped.",
            Vec::new(),
            skipped,
        );
    }
    if evidence.iter().all(|report| report.status == "not_run") {
        return quality_gate(
            "quality_evidence",
            "pass",
            "info",
            "Quality evidence commands were identified but not run.",
            Vec::new(),
            evidence
                .iter()
                .map(|report| report.command.clone())
                .collect(),
        );
    }
    quality_gate(
        "quality_evidence",
        "pass",
        "info",
        "All executed quality evidence commands passed.",
        Vec::new(),
        evidence
            .iter()
            .filter(|report| report.status == "passed")
            .map(|report| report.command.clone())
            .collect(),
    )
}

fn boundary_repair_gates(
    reports: &[FileEntropyReport],
    boundary_repair_plans: &[BoundaryRepairPlan],
) -> Vec<QualityGate> {
    let mut gates = Vec::new();
    let plan_files = boundary_repair_plans
        .iter()
        .map(|plan| plan.source_file.clone())
        .collect::<BTreeSet<_>>();
    let high_entropy = reports
        .iter()
        .filter(|report| {
            matches!(
                report.verdict.as_str(),
                "boundary_repair_recommended" | "split_required"
            )
        })
        .collect::<Vec<_>>();
    if high_entropy.is_empty() {
        gates.push(quality_gate(
            "boundary_repair",
            "pass",
            "info",
            "No high-entropy boundary repair is required.",
            Vec::new(),
            Vec::new(),
        ));
        return gates;
    }
    for report in high_entropy {
        if plan_files.contains(&report.file_path) {
            let status = if report.verdict == "split_required" {
                "blocked_by_quality_gate"
            } else {
                "needs_human_decision"
            };
            let severity = if report.verdict == "split_required" {
                "blocker"
            } else {
                "warning"
            };
            gates.push(quality_gate(
                format!("boundary_repair:{}", report.file_path),
                status,
                severity,
                format!(
                    "{} requires boundary repair before appending new behavior.",
                    report.file_path
                ),
                vec![report.file_path.clone()],
                vec![format!(
                    "Review boundary_repair_plans for `{}`.",
                    report.file_path
                )],
            ));
        } else {
            gates.push(quality_gate(
                format!("boundary_repair_missing_plan:{}", report.file_path),
                "blocked_by_quality_gate",
                "blocker",
                format!(
                    "{} is high entropy but has no boundary repair plan.",
                    report.file_path
                ),
                vec![report.file_path.clone()],
                vec![
                    "Create a structured BoundaryRepairPlan before changing this file.".to_string(),
                ],
            ));
        }
    }
    gates
}

fn coverage_gates(reports: &[FileEntropyReport], policy: &DevsystemPolicy) -> Vec<QualityGate> {
    reports
        .iter()
        .filter_map(|report| match report.signals.test_coverage {
            Some(coverage) if coverage < policy.low_coverage_threshold => Some(quality_gate(
                format!("coverage:{}", report.file_path),
                "needs_human_decision",
                "warning",
                format!(
                    "{} coverage {:.2} is below policy threshold {:.2}.",
                    report.file_path, coverage, policy.low_coverage_threshold
                ),
                vec![report.file_path.clone()],
                vec!["Add or run focused tests before trusting the change.".to_string()],
            )),
            Some(_) => None,
            None => Some(quality_gate(
                format!("coverage_missing:{}", report.file_path),
                "needs_human_decision",
                "warning",
                format!("{} has no coverage telemetry.", report.file_path),
                vec![report.file_path.clone()],
                vec!["Provide coverage telemetry or run fallback test commands.".to_string()],
            )),
        })
        .collect()
}

fn stale_code_memory_gate(memory_stage: &MemoryStageOutput) -> QualityGate {
    if memory_stage.stale_code_memories.is_empty() {
        quality_gate(
            "stale_code_memories",
            "pass",
            "info",
            "No stale code memories overlap scoped files.",
            Vec::new(),
            Vec::new(),
        )
    } else {
        quality_gate(
            "stale_code_memories",
            "needs_human_decision",
            "warning",
            "Stale code memories overlap scoped files.",
            memory_stage
                .stale_code_memories
                .iter()
                .filter_map(|memory| memory.file_path.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            vec![
                "Repair or archive stale code memories before relying on remembered code facts."
                    .to_string(),
            ],
        )
    }
}

fn policy_gate(policy: &PolicyLoadResult) -> QualityGate {
    if policy.warnings.is_empty() {
        quality_gate(
            "policy",
            "pass",
            "info",
            format!("Devsystem policy loaded from {}.", policy.source),
            Vec::new(),
            Vec::new(),
        )
    } else {
        quality_gate(
            "policy",
            "needs_human_decision",
            "warning",
            "Devsystem policy loaded with warnings.",
            Vec::new(),
            policy.warnings.clone(),
        )
    }
}

fn quality_gate(
    id: impl Into<String>,
    status: impl Into<String>,
    severity: impl Into<String>,
    message: impl Into<String>,
    files: Vec<String>,
    required_actions: Vec<String>,
) -> QualityGate {
    QualityGate {
        id: id.into(),
        status: status.into(),
        severity: severity.into(),
        message: message.into(),
        files,
        required_actions,
    }
}

fn summarize_quality_gates(gates: &[QualityGate]) -> QualityGateSummary {
    let pass_count = gates.iter().filter(|gate| gate.status == "pass").count();
    let warning_count = gates
        .iter()
        .filter(|gate| gate.severity == "warning")
        .count();
    let decision_count = gates
        .iter()
        .filter(|gate| gate.status == "needs_human_decision")
        .count();
    let blocker_count = gates
        .iter()
        .filter(|gate| gate.severity == "blocker")
        .count();
    let overall_status = if blocker_count > 0 {
        "blocked_by_quality_gate"
    } else if decision_count > 0 {
        "needs_human_decision"
    } else {
        "ready"
    }
    .to_string();
    let recommended_next_actions = gates
        .iter()
        .flat_map(|gate| gate.required_actions.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    QualityGateSummary {
        overall_status,
        pass_count,
        warning_count,
        decision_count,
        blocker_count,
        recommended_next_actions,
    }
}

fn build_boundary_repair_plan(
    report: &FileEntropyReport,
    required_tests: &[TestCommandRecommendation],
) -> BoundaryRepairPlan {
    let responsibilities = report
        .responsibility_summary
        .iter()
        .filter(|responsibility| {
            !matches!(
                responsibility.as_str(),
                "implementation"
                    | "static metadata"
                    | "static country metadata"
                    | "ignored by devsystem policy"
                    | "generated file by devsystem policy"
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let stem = file_stem_for_module(&report.file_path);
    let proposed_modules = responsibilities
        .iter()
        .map(|responsibility| ProposedBoundaryModule {
            module_path: proposed_module_path(&report.file_path, responsibility),
            responsibility: responsibility.clone(),
            rationale: format!(
                "`{}` changes for a different reason than the other responsibilities in `{}`.",
                responsibility, report.file_path
            ),
        })
        .collect::<Vec<_>>();
    let retain_orchestrator = responsibilities.len() > 1
        && (report
            .signals
            .semantic_sections
            .iter()
            .any(|section| matches!(section.as_str(), "provider routing" | "business logic"))
            || report.signals.fan_in > 0);
    let mut move_order = proposed_modules
        .iter()
        .map(|module| {
            format!(
                "Move `{}` responsibility from `{}` to `{}`.",
                module.responsibility, report.file_path, module.module_path
            )
        })
        .collect::<Vec<_>>();
    if retain_orchestrator {
        move_order.push(format!(
            "Keep `{}` as a thin `{stem}` orchestrator that coordinates extracted boundaries without owning their internals.",
            report.file_path
        ));
    } else {
        move_order.push(format!(
            "Replace direct callers of `{}` with the extracted boundary modules after tests pass.",
            report.file_path
        ));
    }
    BoundaryRepairPlan {
        source_file: report.file_path.clone(),
        risk_level: if report.verdict == "split_required" {
            "high".to_string()
        } else {
            "medium".to_string()
        },
        reason: format!(
            "{} has File Entropy Score {} with {} responsibilities: {}.",
            report.file_path,
            report.score,
            report.responsibility_count,
            report.responsibility_summary.join(", ")
        ),
        retain_orchestrator,
        proposed_modules,
        move_order,
        affected_imports: vec![
            format!(
                "Update imports/callers that depend on `{}` public symbols.",
                report.file_path
            ),
            "Keep public API compatibility until focused tests pass.".to_string(),
        ],
        required_tests: required_tests.to_vec(),
        non_goals: vec![
            "Do not split static metadata or generated files.".to_string(),
            "Do not auto-apply this plan from dukememory_devsystem.".to_string(),
            "Do not create tiny files for responsibilities that always change together."
                .to_string(),
        ],
    }
}

fn file_stem_for_module(file_path: &str) -> String {
    Path::new(file_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("module")
        .to_string()
}

fn proposed_module_path(file_path: &str, responsibility: &str) -> String {
    let path = Path::new(file_path);
    let parent = path
        .parent()
        .and_then(|parent| parent.to_str())
        .unwrap_or("");
    let extension = path.extension().and_then(|extension| extension.to_str());
    let stem = file_stem_for_module(file_path);
    let responsibility_slug = responsibility.replace([' ', '/', '-'], "_");
    let filename = match extension {
        Some(extension) if !extension.is_empty() => {
            format!("{stem}_{responsibility_slug}.{extension}")
        }
        _ => format!("{stem}_{responsibility_slug}"),
    };
    if parent.is_empty() {
        filename
    } else {
        format!("{parent}/{filename}")
    }
}

fn final_verdict(
    reports: &[FileEntropyReport],
    index_guard: Option<&DevsystemIndexGuard>,
    gate_summary: &QualityGateSummary,
) -> String {
    if gate_summary.overall_status == "blocked_by_quality_gate" {
        return "boundary_repair_recommended".to_string();
    }
    if gate_summary.overall_status == "needs_human_decision" {
        return "needs_human_decision".to_string();
    }
    if index_guard.is_some_and(|guard| !guard.fresh) {
        return "needs_human_decision".to_string();
    }
    if reports.iter().any(|report| {
        report.verdict == "split_required" || report.verdict == "boundary_repair_recommended"
    }) {
        "boundary_repair_recommended".to_string()
    } else if reports.iter().any(|report| report.verdict == "watch") {
        "needs_human_decision".to_string()
    } else {
        "ready".to_string()
    }
}

fn devsystem_index_guard(report: &CodeFreshnessReport) -> DevsystemIndexGuard {
    let fresh = report.is_fresh();
    DevsystemIndexGuard {
        fresh,
        stale_files: report.stale_files.clone(),
        missing_files: report.missing_files.clone(),
        deleted_files: report.deleted_files.clone(),
        recommended_action: if fresh {
            None
        } else {
            Some("Run dukememory_code_index for this project_path before relying on devsystem code telemetry.".to_string())
        },
    }
}

fn stage_report(
    role: &str,
    readiness_percent: u8,
    summary: &str,
    artifacts: Value,
) -> AgentStageReport {
    AgentStageReport {
        role: role.to_string(),
        status: "completed".to_string(),
        readiness_percent,
        summary: summary.to_string(),
        artifacts,
    }
}

struct StageReportInput<'a> {
    reports: &'a [FileEntropyReport],
    recommended_tests: &'a [String],
    recommended_test_commands: &'a [TestCommandRecommendation],
    quality_evidence_reports: &'a [QualityEvidenceReport],
    boundary_repair_plans: &'a [BoundaryRepairPlan],
    quality_gates: &'a [QualityGate],
    quality_gate_summary: &'a QualityGateSummary,
    code_review_plan: &'a CodeReviewPlanReport,
    final_verdict: &'a str,
    index_guard: Option<&'a DevsystemIndexGuard>,
    index_run: Option<&'a IndexRunSummary>,
    memory_stage: &'a MemoryStageOutput,
    policy: &'a PolicyLoadResult,
}

fn build_stage_reports(input: StageReportInput<'_>) -> Vec<AgentStageReport> {
    vec![
        run_architect_stage(
            input.reports,
            input.index_guard,
            input.index_run,
            input.memory_stage,
            input.policy,
        ),
        run_coder_stage(input.code_review_plan),
        run_test_stage(
            input.recommended_tests,
            input.recommended_test_commands,
            input.quality_evidence_reports,
        ),
        run_critic_stage(CriticStageInput {
            reports: input.reports,
            recommended_tests: input.recommended_tests,
            quality_evidence_reports: input.quality_evidence_reports,
            boundary_repair_plans: input.boundary_repair_plans,
            quality_gates: input.quality_gates,
            quality_gate_summary: input.quality_gate_summary,
            index_guard: input.index_guard,
            index_run: input.index_run,
            memory_stage: input.memory_stage,
            policy: input.policy,
        }),
        run_refactor_stage(input.reports, input.boundary_repair_plans),
        run_final_memory_stage(input.final_verdict, input.memory_stage),
    ]
}

fn run_architect_stage(
    reports: &[FileEntropyReport],
    index_guard: Option<&DevsystemIndexGuard>,
    index_run: Option<&IndexRunSummary>,
    memory_stage: &MemoryStageOutput,
    policy: &PolicyLoadResult,
) -> AgentStageReport {
    let boundary_files = reports
        .iter()
        .filter(|report| !report.boundary_repair_suggestions.is_empty())
        .map(|report| report.file_path.clone())
        .collect::<Vec<_>>();
    stage_report(
        "architect",
        40,
        "Responsibility boundaries, public API surface, fan-in, fan-out, task history, and index freshness were checked.",
        json!({
            "high_entropy_files": boundary_files,
            "index_run": index_run,
            "index_fresh": index_guard.map(|guard| guard.fresh),
            "file_count": reports.len(),
            "policy_source": &policy.source,
            "policy_thresholds": {
                "boundary_repair_score_threshold": policy.effective.boundary_repair_score_threshold,
                "split_score_threshold": policy.effective.split_score_threshold,
                "boundary_repair_responsibility_count": policy.effective.boundary_repair_responsibility_count,
                "split_responsibility_count": policy.effective.split_responsibility_count
            },
            "file_task_history": &memory_stage.file_task_history,
            "public_api_surface": reports.iter()
                .map(|report| json!({
                    "file_path": &report.file_path,
                    "public_api_surface": report.signals.public_api_surface,
                    "fan_in": report.signals.fan_in,
                    "fan_out": report.signals.fan_out
                }))
                .collect::<Vec<_>>()
        }),
    )
}

fn run_coder_stage(code_review_plan: &CodeReviewPlanReport) -> AgentStageReport {
    stage_report(
        "coder",
        55,
        "Implementation direction was limited to advisory patch planning; boundary repairs are not auto-applied.",
        json!({
            "changed_symbols": code_review_plan.changed_symbols.len(),
            "impacted_files": &code_review_plan.impacted_files,
            "memory_suggestions": &code_review_plan.memory_suggestions
        }),
    )
}

fn run_test_stage(
    recommended_tests: &[String],
    recommended_test_commands: &[TestCommandRecommendation],
    quality_evidence_reports: &[QualityEvidenceReport],
) -> AgentStageReport {
    stage_report(
        "test",
        70,
        "Changed files were mapped to affected indexed tests, executable fallback commands, and optional quality evidence.",
        json!({
            "affected_tests": recommended_tests,
            "commands": recommended_test_commands,
            "quality_evidence": quality_evidence_reports
        }),
    )
}

struct CriticStageInput<'a> {
    reports: &'a [FileEntropyReport],
    recommended_tests: &'a [String],
    quality_evidence_reports: &'a [QualityEvidenceReport],
    boundary_repair_plans: &'a [BoundaryRepairPlan],
    quality_gates: &'a [QualityGate],
    quality_gate_summary: &'a QualityGateSummary,
    index_guard: Option<&'a DevsystemIndexGuard>,
    index_run: Option<&'a IndexRunSummary>,
    memory_stage: &'a MemoryStageOutput,
    policy: &'a PolicyLoadResult,
}

fn run_critic_stage(input: CriticStageInput<'_>) -> AgentStageReport {
    stage_report(
        "critic",
        85,
        "Regression, hidden-coupling, stale-telemetry, stale-code-memory, and high-entropy append risks were evaluated.",
        json!({
            "warnings": critic_warnings(
                input.reports,
                input.recommended_tests,
                input.quality_evidence_reports,
                input.boundary_repair_plans,
                input.index_guard,
                input.index_run,
                input.memory_stage,
                &input.policy.effective
            ),
            "stale_code_memories": &input.memory_stage.stale_code_memories,
            "policy_warnings": &input.policy.warnings,
            "boundary_repair_plan_count": input.boundary_repair_plans.len(),
            "quality_gates": input.quality_gates,
            "quality_gate_summary": input.quality_gate_summary
        }),
    )
}

fn run_refactor_stage(
    reports: &[FileEntropyReport],
    boundary_repair_plans: &[BoundaryRepairPlan],
) -> AgentStageReport {
    stage_report(
        "refactor",
        95,
        "Structured boundary repair plans were proposed only for files with divergent responsibilities.",
        json!({
            "boundary_repair": reports.iter()
                .filter(|report| !report.boundary_repair_suggestions.is_empty())
                .map(|report| json!({
                    "file_path": &report.file_path,
                    "suggestions": &report.boundary_repair_suggestions
                }))
                .collect::<Vec<_>>(),
            "boundary_repair_plans": boundary_repair_plans
        }),
    )
}

fn run_final_memory_stage(
    final_verdict: &str,
    memory_stage: &MemoryStageOutput,
) -> AgentStageReport {
    stage_report(
        "memory",
        100,
        "Final task result is ready for pending dukememory recording.",
        json!({
            "final_verdict": final_verdict,
            "write_policy": "pending",
            "active_memory_count": memory_stage.active_memories.len(),
            "pending_memory_count": memory_stage.pending_memories.len(),
            "relevant_memory_count": memory_stage.relevant_memories.len(),
            "code_memory_count": memory_stage.code_memories.len()
        }),
    )
}

fn critic_warnings(
    reports: &[FileEntropyReport],
    recommended_tests: &[String],
    quality_evidence_reports: &[QualityEvidenceReport],
    boundary_repair_plans: &[BoundaryRepairPlan],
    index_guard: Option<&DevsystemIndexGuard>,
    index_run: Option<&IndexRunSummary>,
    memory_stage: &MemoryStageOutput,
    policy: &DevsystemPolicy,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if recommended_tests.is_empty() {
        warnings
            .push("No indexed affected tests were found; use fallback test commands.".to_string());
    }
    for report in quality_evidence_reports {
        match report.status.as_str() {
            "failed" => warnings.push(format!(
                "Quality evidence command `{}` failed with exit_code {:?}.",
                report.command, report.exit_code
            )),
            "timed_out" => warnings.push(format!(
                "Quality evidence command `{}` timed out after {} ms.",
                report.command, report.duration_ms
            )),
            "skipped" => warnings.push(format!(
                "Quality evidence command `{}` was skipped: {}.",
                report.command, report.stderr_excerpt
            )),
            _ => {}
        }
    }
    if index_guard.is_some_and(|guard| !guard.fresh) {
        warnings.push(
            "Code index is stale or incomplete; refresh before treating telemetry as complete."
                .to_string(),
        );
    }
    if let Some(run) = index_run
        && !run.enabled
    {
        warnings.push(format!(
            "Auto-index did not run: {}.",
            run.reason.as_deref().unwrap_or("reason was not provided")
        ));
    }
    if !memory_stage.stale_code_memories.is_empty() {
        warnings.push(format!(
            "{} stale code memories overlap scoped files; repair or archive them before relying on remembered code facts.",
            memory_stage.stale_code_memories.len()
        ));
    }
    for report in reports {
        if matches!(
            report.verdict.as_str(),
            "boundary_repair_recommended" | "split_required"
        ) {
            warnings.push(format!(
                "{} has high responsibility density; avoid appending new behavior before boundary repair.",
                report.file_path
            ));
            if !boundary_repair_plans
                .iter()
                .any(|plan| plan.source_file == report.file_path)
            {
                warnings.push(format!(
                    "{} is high entropy but has no structured boundary repair plan.",
                    report.file_path
                ));
            }
        }
        if report.signals.test_coverage.is_none() {
            warnings.push(format!(
                "{} has no coverage signal in available telemetry.",
                report.file_path
            ));
        } else if report
            .signals
            .test_coverage
            .is_some_and(|coverage| coverage < policy.low_coverage_threshold)
        {
            warnings.push(format!(
                "{} coverage is below policy threshold {:.2}.",
                report.file_path, policy.low_coverage_threshold
            ));
        }
    }
    for plan in boundary_repair_plans {
        if policy_file_match(&plan.source_file, &policy.ignored_file_patterns)
            || policy_file_match(&plan.source_file, &policy.generated_file_patterns)
        {
            warnings.push(format!(
                "{} has a boundary repair plan despite policy suppression.",
                plan.source_file
            ));
        }
        if plan.required_tests.is_empty() {
            warnings.push(format!(
                "{} boundary repair plan has no required test commands.",
                plan.source_file
            ));
        }
    }
    warnings.sort();
    warnings.dedup();
    warnings
}

struct RoleReportInput<'a> {
    query: &'a str,
    files: &'a [String],
    reports: &'a [FileEntropyReport],
    recommended_tests: &'a [String],
    recommended_test_commands: &'a [TestCommandRecommendation],
    quality_evidence_reports: &'a [QualityEvidenceReport],
    boundary_repair_plans: &'a [BoundaryRepairPlan],
    quality_gates: &'a [QualityGate],
    quality_gate_summary: &'a QualityGateSummary,
    code_review_plan: &'a CodeReviewPlanReport,
    index_guard: Option<&'a DevsystemIndexGuard>,
    index_run: Option<&'a IndexRunSummary>,
    memory_stage: &'a MemoryStageOutput,
    policy: &'a PolicyLoadResult,
}

fn build_role_reports(input: RoleReportInput<'_>) -> RoleReports {
    let high_entropy = input
        .reports
        .iter()
        .filter(|report| {
            matches!(
                report.verdict.as_str(),
                "boundary_repair_recommended" | "split_required"
            )
        })
        .map(|report| report.file_path.clone())
        .collect::<Vec<_>>();
    RoleReports {
        planner: json!({
            "task": input.query,
            "scope_files": input.files,
            "acceptance": "Produce advisory quality report; do not auto-apply repairs."
        }),
        memory: json!({
            "substrate": "dukememory",
            "mode": "full capability set",
            "writes": "pending only",
            "index_run": input.index_run,
            "index_guard": input.index_guard,
            "policy_source": &input.policy.source,
            "active_memories": &input.memory_stage.active_memories,
            "pending_memories": &input.memory_stage.pending_memories,
            "relevant_memories": &input.memory_stage.relevant_memories,
            "recent_task_sessions": &input.memory_stage.recent_task_sessions,
            "code_memories": &input.memory_stage.code_memories,
            "stale_code_memories": &input.memory_stage.stale_code_memories
        }),
        architect: json!({
            "rule": "File size is a signal, not a verdict. Responsibility density is the verdict.",
            "high_entropy_files": high_entropy,
            "index_run": input.index_run,
            "policy": &input.policy.effective,
            "public_api_surface": input.reports.iter()
                .map(|report| json!({
                    "file_path": &report.file_path,
                    "public_api_surface": report.signals.public_api_surface,
                    "fan_in": report.signals.fan_in,
                    "fan_out": report.signals.fan_out
                }))
                .collect::<Vec<_>>()
        }),
        coder: json!({
            "mode": "advisory",
            "implementation_direction": "Patch only files with stable responsibility boundaries; request boundary repair first for high-entropy files.",
            "changed_symbols": input.code_review_plan.changed_symbols.iter()
                .take(20)
                .map(|symbol| json!({
                    "name": &symbol.name,
                    "kind": &symbol.kind,
                    "file_path": &symbol.file_path
                }))
                .collect::<Vec<_>>()
        }),
        test: json!({
            "recommended_tests": input.recommended_tests,
            "recommended_commands": input.recommended_test_commands,
            "quality_evidence": input.quality_evidence_reports,
            "fallback": if input.recommended_tests.is_empty() { "run project suite" } else { "focused indexed tests" }
        }),
        critic: json!({
            "checks": [
                "hidden coupling",
                "missing tests",
                "stale telemetry",
                "append-code-downward behavior in high-entropy files"
            ],
            "warnings": critic_warnings(
                input.reports,
                input.recommended_tests,
                input.quality_evidence_reports,
                input.boundary_repair_plans,
                input.index_guard,
                input.index_run,
                input.memory_stage,
                &input.policy.effective
            ),
            "quality_gates": input.quality_gates,
            "quality_gate_summary": input.quality_gate_summary
        }),
        refactor: json!({
            "policy": "Split when responsibilities diverge; keep together when concepts change together.",
            "boundary_repair_files": input.reports.iter()
                .filter(|report| !report.boundary_repair_suggestions.is_empty())
                .map(|report| report.file_path.clone())
                .collect::<Vec<_>>(),
            "boundary_repair_plans": input.boundary_repair_plans
        }),
    }
}

impl MemoryWrites {
    fn pending() -> Self {
        Self {
            status: "pending".to_string(),
            ids: Vec::new(),
            entries: Vec::new(),
            intent_memory_ids: Vec::new(),
            decision_memory_ids: Vec::new(),
            entropy_memory_ids: Vec::new(),
            graph_memory_ids: Vec::new(),
            quality_observation_ids: Vec::new(),
            graph_candidates: Vec::new(),
            inserted_count: 0,
            duplicate_count: 0,
        }
    }

    fn empty(status: &str) -> Self {
        Self {
            status: status.to_string(),
            ..Self::pending()
        }
    }

    fn record(&mut self, category: &str, kind: &str, outcome: RememberOutcome) -> String {
        if outcome.inserted {
            self.inserted_count += 1;
        } else {
            self.duplicate_count += 1;
        }
        let id = outcome.id.clone();
        self.ids.push(id.clone());
        match category {
            "intent" => self.intent_memory_ids.push(id.clone()),
            "decision" => self.decision_memory_ids.push(id.clone()),
            "entropy" => {
                self.entropy_memory_ids.push(id.clone());
                self.quality_observation_ids.push(id.clone());
            }
            "intent_graph" => self.graph_memory_ids.push(id.clone()),
            _ => {}
        }
        self.entries.push(MemoryWriteEntry {
            category: category.to_string(),
            kind: kind.to_string(),
            id: id.clone(),
            inserted: outcome.inserted,
            duplicate_of: outcome.duplicate_of,
        });
        id
    }
}

struct DevsystemMemoryWriteInput<'a> {
    store: &'a Store,
    project_id: &'a str,
    query: &'a str,
    planner: &'a PlannerStageOutput,
    reports: &'a [FileEntropyReport],
    boundary_repair_plans: &'a [BoundaryRepairPlan],
    quality_gates: &'a [QualityGate],
    quality_gate_summary: &'a QualityGateSummary,
    recommended_tests: &'a [String],
    recommended_test_commands: &'a [TestCommandRecommendation],
    quality_evidence_reports: &'a [QualityEvidenceReport],
    final_verdict: &'a str,
}

fn write_pending_devsystem_memories(input: DevsystemMemoryWriteInput<'_>) -> Result<MemoryWrites> {
    let mut writes = MemoryWrites::pending();
    let intent_id = writes.record(
        "intent",
        "task_intent",
        write_pending_memory(
            input.store,
            input.project_id,
            PendingMemorySpec {
                kind: "task_intent",
                body: intent_memory_body(&input),
                tags: vec!["dukedevsystem", "task-intent", "intent-graph"],
                importance: 0.8,
                confidence: 0.86,
                status_reason: "Pending review for normalized task intent.",
            },
        )?,
    );
    let decision_id = writes.record(
        "decision",
        "decision",
        write_pending_memory(
            input.store,
            input.project_id,
            PendingMemorySpec {
                kind: "decision",
                body: decision_memory_body(&input),
                tags: vec!["dukedevsystem", "decision-memory", "quality-gate"],
                importance: 0.82,
                confidence: 0.84,
                status_reason: "Pending review for devsystem decision memory.",
            },
        )?,
    );

    let mut entropy_source_ids = HashMap::new();
    for report in input.reports {
        let entropy_id = writes.record(
            "entropy",
            "quality_observation",
            write_pending_memory(
                input.store,
                input.project_id,
                PendingMemorySpec {
                    kind: "quality_observation",
                    body: entropy_memory_body(report),
                    tags: vec!["dukedevsystem", "file-entropy-score", "quality-observation"],
                    importance: 0.76,
                    confidence: 0.82,
                    status_reason: "Pending review for file entropy observation.",
                },
            )?,
        );
        entropy_source_ids.insert(report.file_path.clone(), entropy_id);
    }

    let graph_candidates =
        build_intent_graph_candidates(&input, &intent_id, &decision_id, &entropy_source_ids);
    let graph_id = writes.record(
        "intent_graph",
        "intent_graph",
        write_pending_memory(
            input.store,
            input.project_id,
            PendingMemorySpec {
                kind: "intent_graph",
                body: intent_graph_memory_body(input.query, &graph_candidates),
                tags: vec!["dukedevsystem", "intent-graph", "graph-candidate"],
                importance: 0.72,
                confidence: 0.8,
                status_reason: "Pending review for intent graph candidate links.",
            },
        )?,
    );
    writes.graph_candidates = graph_candidates
        .into_iter()
        .chain(std::iter::once(IntentGraphCandidate {
            from: format!("task_intent:{}", input.query),
            relation: "documents_graph_candidate_set".to_string(),
            to: format!("memory:{graph_id}"),
            source_memory_id: Some(graph_id),
            source_memory_category: "intent_graph".to_string(),
        }))
        .collect();
    Ok(writes)
}

struct PendingMemorySpec<'a> {
    kind: &'a str,
    body: String,
    tags: Vec<&'a str>,
    importance: f64,
    confidence: f64,
    status_reason: &'a str,
}

fn write_pending_memory(
    store: &Store,
    project_id: &str,
    spec: PendingMemorySpec<'_>,
) -> Result<RememberOutcome> {
    store.remember_deduplicated(
        project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: spec.kind.to_string(),
            body: spec.body,
            tags: spec.tags.into_iter().map(str::to_string).collect(),
            source: Some("dukememory_devsystem".to_string()),
            status: MemoryStatus::Pending,
            importance: spec.importance,
            confidence: spec.confidence,
            status_reason: Some(spec.status_reason.to_string()),
            allow_sensitive: false,
        },
    )
}

fn intent_memory_body(input: &DevsystemMemoryWriteInput<'_>) -> String {
    format!(
        "dukedevsystem task intent\nintent: {}\nscope_files:\n{}\nlikely_risk: {}\nacceptance_criteria:\n{}\nfinal_verdict: {}",
        input.query,
        bullet_lines(&input.planner.scope_files),
        input.planner.likely_risk,
        bullet_lines(&input.planner.acceptance_criteria),
        input.final_verdict
    )
}

fn decision_memory_body(input: &DevsystemMemoryWriteInput<'_>) -> String {
    let boundary_decision = if input.boundary_repair_plans.is_empty() {
        "boundary_repair: not recommended; scoped files keep stable responsibility boundaries."
            .to_string()
    } else {
        format!(
            "boundary_repair: recommended for {}.",
            input
                .boundary_repair_plans
                .iter()
                .map(|plan| plan.source_file.clone())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let gate_summary = input
        .quality_gates
        .iter()
        .map(|gate| format!("{}={} ({})", gate.id, gate.status, gate.severity))
        .collect::<Vec<_>>();
    let commands = input
        .recommended_test_commands
        .iter()
        .map(|command| command.command.clone())
        .collect::<Vec<_>>();
    let evidence = executed_evidence_summary(input.quality_evidence_reports);
    let evidence_section = if evidence.is_empty() {
        String::new()
    } else {
        format!("\nquality_evidence:\n{}", bullet_lines(&evidence))
    };
    format!(
        "dukedevsystem decision memory\nintent: {}\nfinal_verdict: {}\n{}\nquality_gate_summary: status={} blockers={} decisions={} warnings={}\nquality_gates:\n{}\nrecommended_tests:\n{}\nrecommended_test_commands:\n{}{}",
        input.query,
        input.final_verdict,
        boundary_decision,
        input.quality_gate_summary.overall_status,
        input.quality_gate_summary.blocker_count,
        input.quality_gate_summary.decision_count,
        input.quality_gate_summary.warning_count,
        bullet_lines(&gate_summary),
        bullet_lines(input.recommended_tests),
        bullet_lines(&commands),
        evidence_section
    )
}

fn executed_evidence_summary(evidence: &[QualityEvidenceReport]) -> Vec<String> {
    evidence
        .iter()
        .filter(|report| report.status != "not_run")
        .map(|report| {
            format!(
                "{} => {} exit_code={:?} duration_ms={}",
                report.command, report.status, report.exit_code, report.duration_ms
            )
        })
        .collect()
}

fn entropy_memory_body(report: &FileEntropyReport) -> String {
    format!(
        "dukedevsystem file entropy observation\nfile_path: {}\nscore: {}\nverdict: {}\nresponsibility_count: {}\nresponsibility_summary:\n{}\nmissing_signals:\n{}\nboundary_repair_suggestions:\n{}",
        report.file_path,
        report.score,
        report.verdict,
        report.responsibility_count,
        bullet_lines(&report.responsibility_summary),
        bullet_lines(&report.missing_signals),
        bullet_lines(&report.boundary_repair_suggestions)
    )
}

fn build_intent_graph_candidates(
    input: &DevsystemMemoryWriteInput<'_>,
    intent_id: &str,
    decision_id: &str,
    entropy_source_ids: &HashMap<String, String>,
) -> Vec<IntentGraphCandidate> {
    let intent_node = format!("task_intent:{}", input.query);
    let mut candidates = Vec::new();
    for report in input.reports {
        candidates.push(IntentGraphCandidate {
            from: intent_node.clone(),
            relation: "touches_file".to_string(),
            to: format!("file:{}", report.file_path),
            source_memory_id: Some(intent_id.to_string()),
            source_memory_category: "intent".to_string(),
        });
        candidates.push(IntentGraphCandidate {
            from: format!("file:{}", report.file_path),
            relation: "has_entropy_verdict".to_string(),
            to: format!("entropy_verdict:{}", report.verdict),
            source_memory_id: entropy_source_ids.get(&report.file_path).cloned(),
            source_memory_category: "entropy".to_string(),
        });
    }
    for plan in input.boundary_repair_plans {
        candidates.push(IntentGraphCandidate {
            from: format!("file:{}", plan.source_file),
            relation: "needs_boundary_repair_plan".to_string(),
            to: format!("boundary_repair_plan:{}", plan.risk_level),
            source_memory_id: Some(decision_id.to_string()),
            source_memory_category: "decision".to_string(),
        });
    }
    for gate in input.quality_gates {
        candidates.push(IntentGraphCandidate {
            from: intent_node.clone(),
            relation: "evaluated_by_quality_gate".to_string(),
            to: format!("quality_gate:{}={}", gate.id, gate.status),
            source_memory_id: Some(decision_id.to_string()),
            source_memory_category: "decision".to_string(),
        });
    }
    for test in input.recommended_tests {
        candidates.push(IntentGraphCandidate {
            from: intent_node.clone(),
            relation: "recommends_test".to_string(),
            to: format!("test:{test}"),
            source_memory_id: Some(decision_id.to_string()),
            source_memory_category: "decision".to_string(),
        });
    }
    for evidence in input
        .quality_evidence_reports
        .iter()
        .filter(|report| report.status != "not_run")
    {
        let evidence_node = format!("evidence_command:{}", evidence.command);
        candidates.push(IntentGraphCandidate {
            from: intent_node.clone(),
            relation: "has_quality_evidence".to_string(),
            to: evidence_node.clone(),
            source_memory_id: Some(decision_id.to_string()),
            source_memory_category: "decision".to_string(),
        });
        for gate_id in &evidence.affects_gate_ids {
            candidates.push(IntentGraphCandidate {
                from: evidence_node.clone(),
                relation: "affects_quality_gate".to_string(),
                to: format!("quality_gate:{gate_id}"),
                source_memory_id: Some(decision_id.to_string()),
                source_memory_category: "decision".to_string(),
            });
        }
        if evidence.status == "failed" {
            candidates.push(IntentGraphCandidate {
                from: evidence_node,
                relation: "requires_test_decision".to_string(),
                to: "decision:failed_quality_evidence".to_string(),
                source_memory_id: Some(decision_id.to_string()),
                source_memory_category: "decision".to_string(),
            });
        }
    }
    candidates
}

fn intent_graph_memory_body(query: &str, candidates: &[IntentGraphCandidate]) -> String {
    let lines = candidates
        .iter()
        .map(|candidate| {
            format!(
                "{} --{}--> {} [source_category={}, source_memory_id={}]",
                candidate.from,
                candidate.relation,
                candidate.to,
                candidate.source_memory_category,
                candidate.source_memory_id.as_deref().unwrap_or("none")
            )
        })
        .collect::<Vec<_>>();
    format!(
        "dukedevsystem intent graph candidates\nintent: {query}\nlinks:\n{}",
        bullet_lines(&lines)
    )
}

fn bullet_lines(items: &[String]) -> String {
    if items.is_empty() {
        "- none".to_string()
    } else {
        items
            .iter()
            .map(|item| format!("- {item}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn code_symbol_ids_for_files(
    store: &Store,
    project_id: &str,
    files: &[String],
) -> Result<Vec<String>> {
    let mut ids = BTreeSet::new();
    for file in files {
        for symbol in store.code_symbols_for_file(project_id, file)? {
            ids.insert(symbol.id);
        }
    }
    Ok(ids.into_iter().collect())
}

const RESPONSIBILITY_KEYWORDS: &[(&str, &[&str])] = &[
    ("payment capture", &["capture", "charge", "authorize"]),
    ("refund", &["refund", "reversal"]),
    (
        "provider routing",
        &["provider", "routing", "route_provider"],
    ),
    ("webhook parsing", &["webhook", "parse_webhook"]),
    ("fraud decision", &["fraud", "risk_score"]),
    ("invoice generation", &["invoice", "receipt"]),
    (
        "email notification",
        &["email", "mailer", "send_email", "notification"],
    ),
    ("business logic", &["business", "domain", "policy"]),
    (
        "database access",
        &[
            "database",
            "repository",
            "sql",
            "select ",
            "insert ",
            "update ",
            "delete ",
            "db.",
        ],
    ),
    (
        "http transport",
        &[
            "http",
            "request",
            "response",
            "route(",
            "handler",
            "controller",
        ],
    ),
    ("validation", &["validate", "schema", "sanitize"]),
    (
        "ui formatting",
        &["ui", "format_", "render", "component", "view"],
    ),
    (
        "permissions",
        &["permission", "authz", "authorize", "acl", "role"],
    ),
    (
        "feature flags",
        &["feature_flag", "feature flag", "experiment", "toggle"],
    ),
    (
        "integrations",
        &["integration", "client", "api", "external"],
    ),
    ("retry/cache", &["retry", "cache", "backoff", "memo"]),
    ("logging", &["log", "logger", "tracing"]),
    (
        "migration compatibility",
        &["migration", "legacy", "compat", "backfill"],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_index::index_project;
    use crate::config::Config;

    fn test_config(name: &str) -> Config {
        Config {
            database_url: "postgresql://dukememory-test@localhost:55432/dukememory_test"
                .to_string(),
            database_marker: std::env::temp_dir().join(format!(
                "dukememory-devsystem-test-{name}-{}.schema-marker",
                uuid::Uuid::now_v7()
            )),
            ollama_base_url: "http://127.0.0.1:9".to_string(),
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

    fn write_file(root: &Path, path: &str, body: &str) {
        let full = root.join(path);
        std::fs::create_dir_all(full.parent().expect("parent")).expect("create parent");
        std::fs::write(full, body).expect("write fixture");
    }

    #[test]
    fn large_static_metadata_is_ok_without_git() -> Result<()> {
        let config = test_config("static");
        let root = std::env::temp_dir().join(format!("dukememory-static-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(root.join(".dukememory.toml"), "name = \"static-fixture\"\n")?;
        let entries = (0..1800)
            .map(|index| format!("  {{ code: \"C{index}\", name: \"Country {index}\" }},"))
            .collect::<Vec<_>>()
            .join("\n");
        write_file(
            &root,
            "src/countries.ts",
            &format!("export const countries = [\n{entries}\n];\n"),
        );
        let mut store = Store::open(&config.database_marker)?;
        index_project(&mut store, &root, Some("static-fixture".to_string()), false)?;

        let report = build_file_entropy_report(
            &store,
            "static-fixture",
            "src/countries.ts",
            Some(&root),
            None,
            &DevsystemPolicy::default(),
        )?;

        assert_eq!(report.verdict, "ok");
        assert_eq!(report.responsibility_count, 1);
        assert!(
            report
                .missing_signals
                .iter()
                .any(|signal| signal.contains(".git repository not available"))
        );
        assert!(
            build_boundary_repair_plans(&[report], &[], &DevsystemPolicy::default()).is_empty()
        );
        Ok(())
    }

    #[test]
    fn mixed_payment_service_requires_split() -> Result<()> {
        let config = test_config("payment");
        let root =
            std::env::temp_dir().join(format!("dukememory-payment-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"payment-fixture\"\n",
        )?;
        write_file(
            &root,
            "src/payment_service.py",
            r#"
def capture_payment(db, request): db.insert("capture")
def refund_payment(db, request): db.update("refund")
def route_provider(provider, request): return provider.client.request(request)
def parse_webhook(http_request): return validate(http_request.body)
def fraud_decision(user, invoice): return risk_score(user) > 10
def generate_invoice(payment): return format_invoice(payment)
def send_email_notification(email, invoice): logger.info(email)
def retry_cached_provider_call(cache, client): return cache.get("x") or client.retry()
def legacy_migration_compat(row): return backfill(row)
"#,
        );
        let mut store = Store::open(&config.database_marker)?;
        index_project(
            &mut store,
            &root,
            Some("payment-fixture".to_string()),
            false,
        )?;

        let report = build_file_entropy_report(
            &store,
            "payment-fixture",
            "src/payment_service.py",
            Some(&root),
            None,
            &DevsystemPolicy::default(),
        )?;

        assert_eq!(report.verdict, "split_required");
        assert!(report.responsibility_count >= 7, "{report:#?}");
        assert!(!report.boundary_repair_suggestions.is_empty());
        let plans = build_boundary_repair_plans(&[report], &[], &DevsystemPolicy::default());
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].source_file, "src/payment_service.py");
        assert_eq!(plans[0].risk_level, "high");
        assert!(plans[0].retain_orchestrator);
        assert!(plans[0].proposed_modules.len() >= 7);
        assert!(
            plans[0]
                .proposed_modules
                .iter()
                .any(|module| module.responsibility == "webhook parsing")
        );
        Ok(())
    }

    #[test]
    fn devsystem_pipeline_records_session_and_pending_memory() -> Result<()> {
        let config = test_config("pipeline");
        let root =
            std::env::temp_dir().join(format!("dukememory-pipeline-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"pipeline-fixture\"\n",
        )?;
        write_file(
            &root,
            "src/lib.rs",
            "pub fn one_boundary() -> &'static str { \"ok\" }\n",
        );
        let mut store = Store::open(&config.database_marker)?;
        index_project(
            &mut store,
            &root,
            Some("pipeline-fixture".to_string()),
            false,
        )?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "pipeline-fixture".to_string(),
                query: "check one boundary".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: true,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        assert_eq!(report.readiness_percent, 100);
        assert!(report.stage_reports.len() >= 8);
        assert_eq!(report.stage_reports[0].role, "planner");
        assert_eq!(report.stage_reports[1].role, "memory");
        assert!(
            report
                .telemetry
                .memory_agent
                .file_task_history
                .iter()
                .any(|history| history.file_path == "src/lib.rs")
        );
        assert!(
            report
                .telemetry
                .memory_agent
                .recent_task_sessions
                .iter()
                .any(|session| session.id == report.task_session_id)
        );
        assert_eq!(report.memory_writes.status, "pending");
        assert_eq!(report.memory_writes.ids.len(), 4);
        assert_eq!(report.memory_writes.intent_memory_ids.len(), 1);
        assert_eq!(report.memory_writes.decision_memory_ids.len(), 1);
        assert_eq!(report.memory_writes.entropy_memory_ids.len(), 1);
        assert_eq!(report.memory_writes.graph_memory_ids.len(), 1);
        assert!(!report.memory_writes.graph_candidates.is_empty());
        assert_eq!(report.file_entropy_reports.len(), 1);
        assert!(report.boundary_repair_plans.is_empty());
        assert_eq!(report.quality_gate_summary.blocker_count, 0);
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "boundary_repair" && gate.status == "pass")
        );
        assert_eq!(report.code_review_plan.changed_files, vec!["src/lib.rs"]);
        assert!(!report.recommended_test_commands.is_empty());
        let session = store
            .get_task_session("pipeline-fixture", &report.task_session_id)?
            .expect("session");
        assert_eq!(session.status, "completed");
        assert_eq!(session.phase, "done");
        assert_eq!(session.memory_ids.len(), report.memory_writes.ids.len());
        assert!(!session.code_symbol_ids.is_empty());
        assert!(session.file_paths.iter().any(|path| path == "src/lib.rs"));
        let decision = store
            .get(
                "pipeline-fixture",
                &report.memory_writes.decision_memory_ids[0],
            )?
            .expect("decision memory");
        assert_eq!(decision.status, "pending");
        assert_eq!(decision.kind, "decision");
        assert!(decision.body.contains("boundary_repair: not recommended"));
        Ok(())
    }

    #[test]
    fn devsystem_rejects_empty_file_scope() -> Result<()> {
        let config = test_config("empty-scope");
        let mut store = Store::open(&config.database_marker)?;
        let error = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "empty-scope".to_string(),
                query: "nothing to inspect".to_string(),
                files: Vec::new(),
                project_path: None,
                write_memory: false,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )
        .expect_err("empty scope should fail");
        assert!(
            error
                .to_string()
                .contains("requires at least one project-relative file")
        );
        Ok(())
    }

    #[test]
    fn policy_override_changes_split_threshold() -> Result<()> {
        let config = test_config("policy-override");
        let root = std::env::temp_dir().join(format!(
            "dukememory-policy-override-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"policy-override-fixture\"\n",
        )?;
        write_file(
            &root,
            "src/checkout.py",
            r#"
def capture_payment(db): db.insert("capture")
def parse_webhook(request): return request.body
def generate_invoice(payment): return str(payment)
"#,
        );
        let mut store = Store::open(&config.database_marker)?;
        index_project(
            &mut store,
            &root,
            Some("policy-override-fixture".to_string()),
            false,
        )?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "policy-override-fixture".to_string(),
                query: "strict checkout policy".to_string(),
                files: vec!["src/checkout.py".to_string()],
                project_path: Some(root),
                write_memory: false,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: Some(json!({
                    "split_responsibility_count": 3,
                    "required_test_commands": ["cargo test --strict-policy"]
                })),
            },
        )?;

        assert_eq!(report.file_entropy_reports[0].verdict, "split_required");
        assert_eq!(report.boundary_repair_plans.len(), 1);
        assert!(report.quality_gate_summary.blocker_count >= 1);
        assert_eq!(
            report.quality_gate_summary.overall_status,
            "blocked_by_quality_gate"
        );
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "boundary_repair:src/checkout.py")
        );
        assert!(
            report.boundary_repair_plans[0]
                .required_tests
                .iter()
                .any(|command| command.command == "cargo test --strict-policy")
        );
        assert!(report.telemetry.policy.source.contains("mcp_override"));
        assert_eq!(
            report.telemetry.policy.effective.split_responsibility_count,
            3
        );
        assert!(
            report
                .recommended_test_commands
                .iter()
                .any(|command| command.command == "cargo test --strict-policy")
        );
        Ok(())
    }

    #[test]
    fn project_policy_ignores_generated_files_and_adds_required_commands() -> Result<()> {
        let config = test_config("project-policy");
        let root = std::env::temp_dir().join(format!(
            "dukememory-project-policy-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src/generated"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"project-policy-fixture\"\n\
             [devsystem]\n\
             generated_file_patterns = [\"src/generated/\"]\n\
             required_test_commands = [\"cargo test --policy-required\"]\n",
        )?;
        write_file(
            &root,
            "src/generated/payment_service.py",
            r#"
def capture_payment(db): db.insert("capture")
def refund_payment(db): db.update("refund")
def route_provider(provider): return provider.client
def parse_webhook(request): return request.body
def fraud_decision(user): return user.risk > 10
def generate_invoice(payment): return str(payment)
def send_email_notification(email): print(email)
"#,
        );
        let mut store = Store::open(&config.database_marker)?;
        index_project(
            &mut store,
            &root,
            Some("project-policy-fixture".to_string()),
            false,
        )?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "project-policy-fixture".to_string(),
                query: "generated payment boundary".to_string(),
                files: vec!["src/generated/payment_service.py".to_string()],
                project_path: Some(root),
                write_memory: false,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        assert_eq!(report.file_entropy_reports[0].verdict, "ok");
        assert!(report.boundary_repair_plans.is_empty());
        assert!(
            report.file_entropy_reports[0].responsibility_summary[0].contains("generated file")
        );
        assert!(report.telemetry.policy.source.contains("project_config"));
        assert!(
            report
                .recommended_test_commands
                .iter()
                .any(|command| command.command == "cargo test --policy-required")
        );
        Ok(())
    }

    #[test]
    fn devsystem_auto_indexes_before_analysis() -> Result<()> {
        let config = test_config("auto-index");
        let root =
            std::env::temp_dir().join(format!("dukememory-auto-index-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"auto-index-fixture\"\n",
        )?;
        write_file(
            &root,
            "src/lib.rs",
            "pub fn indexed_by_devsystem() -> &'static str { \"fresh\" }\n",
        );
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "auto-index-fixture".to_string(),
                query: "verify automatic indexing".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: false,
                auto_index: true,
                full_rebuild: true,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        let index_run = report.telemetry.index_run.as_ref().expect("index run");
        assert!(index_run.enabled);
        assert!(index_run.full_rebuild);
        assert!(index_run.files_indexed >= 1, "{index_run:#?}");
        assert!(index_run.symbols_indexed >= 1, "{index_run:#?}");
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "code_index_freshness" && gate.status == "pass")
        );
        assert!(
            report.file_entropy_reports[0]
                .missing_signals
                .iter()
                .all(|signal| !signal.contains("file not indexed"))
        );
        Ok(())
    }

    #[test]
    fn disabled_auto_index_requires_human_decision() -> Result<()> {
        let config = test_config("auto-index-disabled");
        let root = std::env::temp_dir().join(format!(
            "dukememory-auto-index-disabled-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"auto-index-disabled-fixture\"\n",
        )?;
        write_file(
            &root,
            "src/lib.rs",
            "pub fn not_indexed_by_request() -> &'static str { \"stale\" }\n",
        );
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "auto-index-disabled-fixture".to_string(),
                query: "verify disabled automatic indexing".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: false,
                auto_index: false,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        let index_run = report.telemetry.index_run.as_ref().expect("index run");
        assert!(!index_run.enabled);
        assert_eq!(
            index_run.reason.as_deref(),
            Some("auto_index disabled by request")
        );
        assert_eq!(
            report.quality_gate_summary.overall_status,
            "needs_human_decision"
        );
        assert!(report.quality_gates.iter().any(|gate| {
            gate.id == "code_index_freshness" && gate.status == "needs_human_decision"
        }));
        assert!(
            report
                .telemetry
                .missing_signals
                .iter()
                .any(|signal| signal.contains("auto_index disabled"))
        );
        assert_eq!(report.memory_writes.status, "disabled");
        assert!(report.memory_writes.ids.is_empty());
        Ok(())
    }

    #[test]
    fn devsystem_repeated_run_deduplicates_pending_memory_candidates() -> Result<()> {
        let config = test_config("dedupe-devsystem");
        let root = std::env::temp_dir().join(format!(
            "dukememory-dedupe-devsystem-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"dedupe-devsystem-fixture\"\n",
        )?;
        write_file(
            &root,
            "src/lib.rs",
            "pub fn stable_boundary() -> &'static str { \"stable\" }\n",
        );
        let mut store = Store::open(&config.database_marker)?;

        let request = || DevsystemRequest {
            project_id: "dedupe-devsystem-fixture".to_string(),
            query: "deduplicate devsystem memory candidates".to_string(),
            files: vec!["src/lib.rs".to_string()],
            project_path: Some(root.clone()),
            write_memory: true,
            auto_index: true,
            full_rebuild: false,
            embed_symbols: false,
            embed_symbol_limit: 200,
            precomputed_index_run: None,
            run_evidence: false,
            evidence_timeout_seconds: 120,
            max_evidence_commands: 5,
            allowed_evidence_commands: Vec::new(),
            code_embedding_model: None,
            duplicate_similarity: 0.92,
            review_limit: 50,
            policy_override: None,
        };

        let first = build_devsystem_report(&mut store, request())?;
        let second = build_devsystem_report(&mut store, request())?;

        assert_eq!(first.memory_writes.ids.len(), 4);
        assert_eq!(second.memory_writes.ids, first.memory_writes.ids);
        assert_eq!(second.memory_writes.duplicate_count, 4);
        assert_eq!(second.memory_writes.inserted_count, 0);
        Ok(())
    }

    #[test]
    fn quality_evidence_is_not_run_by_default() -> Result<()> {
        let config = test_config("evidence-not-run");
        let root = std::env::temp_dir().join(format!(
            "dukememory-evidence-not-run-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"evidence-not-run-fixture\"\n\
             [devsystem]\n\
             required_test_commands = [\"/bin/echo should-not-run\"]\n",
        )?;
        write_file(&root, "src/lib.rs", "pub fn evidence_default() {}\n");
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "evidence-not-run-fixture".to_string(),
                query: "do not run evidence by default".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: true,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: false,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: Vec::new(),
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        assert!(
            report
                .quality_evidence_reports
                .iter()
                .any(|evidence| evidence.command == "/bin/echo should-not-run"
                    && evidence.status == "not_run")
        );
        let decision = store
            .get(
                "evidence-not-run-fixture",
                &report.memory_writes.decision_memory_ids[0],
            )?
            .expect("decision memory");
        assert!(!decision.body.contains("quality_evidence:"));
        Ok(())
    }

    #[test]
    fn quality_evidence_passed_command_updates_gate_and_memory() -> Result<()> {
        let config = test_config("evidence-passed");
        let root = std::env::temp_dir().join(format!(
            "dukememory-evidence-passed-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"evidence-passed-fixture\"\n\
             [devsystem]\n\
             required_test_commands = [\"/bin/echo evidence-ok\"]\n",
        )?;
        write_file(&root, "src/lib.rs", "pub fn evidence_passed() {}\n");
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "evidence-passed-fixture".to_string(),
                query: "run passing evidence".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: true,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: true,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: vec!["/bin/echo evidence-ok".to_string()],
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        let evidence = report
            .quality_evidence_reports
            .iter()
            .find(|evidence| evidence.command == "/bin/echo evidence-ok")
            .expect("evidence report");
        assert_eq!(evidence.status, "passed");
        assert_eq!(evidence.exit_code, Some(0));
        assert!(evidence.stdout_excerpt.contains("evidence-ok"));
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "quality_evidence" && gate.status == "pass"),
            "gates={:#?} evidence={:#?}",
            report.quality_gates,
            report.quality_evidence_reports
        );
        let decision = store
            .get(
                "evidence-passed-fixture",
                &report.memory_writes.decision_memory_ids[0],
            )?
            .expect("decision memory");
        assert!(decision.body.contains("quality_evidence:"));
        assert!(
            report
                .memory_writes
                .graph_candidates
                .iter()
                .any(|candidate| candidate.relation == "has_quality_evidence")
        );
        Ok(())
    }

    #[test]
    fn quality_evidence_failed_command_blocks_gate() -> Result<()> {
        let config = test_config("evidence-failed");
        let root = std::env::temp_dir().join(format!(
            "dukememory-evidence-failed-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"evidence-failed-fixture\"\n\
             [devsystem]\n\
             required_test_commands = [\"/usr/bin/false\"]\n",
        )?;
        write_file(&root, "src/lib.rs", "pub fn evidence_failed() {}\n");
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "evidence-failed-fixture".to_string(),
                query: "run failing evidence".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: true,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: true,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: vec!["/usr/bin/false".to_string()],
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        assert_eq!(report.quality_evidence_reports[0].status, "failed");
        assert_eq!(
            report.quality_gate_summary.overall_status,
            "blocked_by_quality_gate"
        );
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "quality_evidence"
                    && gate.status == "blocked_by_quality_gate")
        );
        assert!(
            report
                .memory_writes
                .graph_candidates
                .iter()
                .any(|candidate| candidate.relation == "requires_test_decision")
        );
        Ok(())
    }

    #[test]
    fn quality_evidence_timeout_requires_human_decision() -> Result<()> {
        let config = test_config("evidence-timeout");
        let root = std::env::temp_dir().join(format!(
            "dukememory-evidence-timeout-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"evidence-timeout-fixture\"\n\
             [devsystem]\n\
             required_test_commands = [\"/bin/sleep 2\"]\n",
        )?;
        write_file(&root, "src/lib.rs", "pub fn evidence_timeout() {}\n");
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "evidence-timeout-fixture".to_string(),
                query: "run timeout evidence".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: false,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: true,
                evidence_timeout_seconds: 1,
                max_evidence_commands: 5,
                allowed_evidence_commands: vec!["/bin/sleep 2".to_string()],
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        assert_eq!(report.quality_evidence_reports[0].status, "timed_out");
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "quality_evidence" && gate.status == "needs_human_decision")
        );
        Ok(())
    }

    #[test]
    fn quality_evidence_unsafe_command_is_skipped() -> Result<()> {
        let config = test_config("evidence-skipped");
        let root = std::env::temp_dir().join(format!(
            "dukememory-evidence-skipped-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(root.join("src"))?;
        std::fs::write(
            root.join(".dukememory.toml"),
            "name = \"evidence-skipped-fixture\"\n\
             [devsystem]\n\
             required_test_commands = [\"/bin/echo ok; /bin/echo no\"]\n",
        )?;
        write_file(&root, "src/lib.rs", "pub fn evidence_skipped() {}\n");
        let mut store = Store::open(&config.database_marker)?;

        let report = build_devsystem_report(
            &mut store,
            DevsystemRequest {
                project_id: "evidence-skipped-fixture".to_string(),
                query: "skip unsafe evidence".to_string(),
                files: vec!["src/lib.rs".to_string()],
                project_path: Some(root),
                write_memory: false,
                auto_index: true,
                full_rebuild: false,
                embed_symbols: false,
                embed_symbol_limit: 200,
                precomputed_index_run: None,
                run_evidence: true,
                evidence_timeout_seconds: 120,
                max_evidence_commands: 5,
                allowed_evidence_commands: vec!["/bin/echo ok; /bin/echo no".to_string()],
                code_embedding_model: None,
                duplicate_similarity: 0.92,
                review_limit: 50,
                policy_override: None,
            },
        )?;

        assert_eq!(report.quality_evidence_reports[0].status, "skipped");
        assert!(
            report.quality_evidence_reports[0]
                .stderr_excerpt
                .contains("shell control character")
        );
        assert!(
            report
                .quality_gates
                .iter()
                .any(|gate| gate.id == "quality_evidence" && gate.status == "needs_human_decision")
        );
        Ok(())
    }

    #[test]
    fn lcov_da_lines_are_used_when_summary_counts_are_absent() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("dukememory-lcov-da-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(
            root.join("lcov.info"),
            "TN:\nSF:src/lib.rs\nDA:1,1\nDA:2,0\nDA:3,3\nend_of_record\n",
        )?;
        let mut missing = Vec::new();
        let coverage = coverage_for_file(Some(&root), "src/lib.rs", &mut missing);
        assert_eq!(coverage, Some(2.0 / 3.0));
        assert!(missing.is_empty());
        Ok(())
    }
}

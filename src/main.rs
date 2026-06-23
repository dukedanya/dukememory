mod backup;
mod code_assist;
mod code_index;
mod code_reason;
mod codex_config;
mod codex_hooks;
mod codex_integration;
mod compaction;
mod config;
mod context_pack;
mod context_plan;
mod devsystem;
mod embedding;
mod extract;
mod graph_extract;
mod gui;
mod launchd_maintenance;
mod lsif_index;
mod maintenance;
mod mcp;
mod mcp_format;
mod ollama;
mod production_audit;
mod project;
mod retrieval;
mod safety;
mod search;
mod semantic_ops;
mod store;
mod validation;

use std::collections::{BTreeSet, HashSet};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::backup::create_database_backup;
use crate::code_assist::{
    CodeAssistReport, CodeAssistReportInput, CodeEvalCaseInput, apply_code_memory_suggestions,
    build_code_assist_report, build_code_pattern_report, build_code_review_plan_report,
    code_index_guard_from_freshness, deterministic_reason_report, evaluate_code_cases,
};
use crate::code_index::{CodeFreshnessReport, check_code_index_freshness, index_project};
use crate::code_reason::{CodeReasonTask, reason_about_search, reason_about_symbol};
use crate::codex_config::{default_codex_config_path, install_mcp_config, mcp_snippet};
use crate::codex_hooks::{default_codex_hooks_path, hooks_snippet, install_hooks};
use crate::codex_integration::{run_codex_hook_payload_audit, run_codex_integration_audit};
use crate::compaction::{CompactionProposal, propose_compaction};
use crate::config::{Config, model_name_matches};
use crate::context_pack::{
    ProjectContextFormat as TaskProjectContextFormat, build_memory_fragments, format_task_context,
    fragment_memory_ids, merge_core_and_task_memories,
};
use crate::context_plan::{ContextPlanRequest, estimate_context_tokens, plan_context_access};
use crate::devsystem::{DevsystemRequest, build_devsystem_report};
use crate::embedding::{embed_indexed_code_symbols, embed_memory, embed_missing};
use crate::extract::{MemoryCandidate, extract_memory_candidates, prepare_extraction_input};
use crate::graph_extract::{GraphExtractionReport, apply_graph_extraction, extract_memory_graph};
use crate::launchd_maintenance::{
    DEFAULT_LABEL, MaintenanceLaunchdOptions, default_log_paths, default_plist_path,
    install_maintenance_launchd, load_launch_agent, maintenance_launchd_plist, unload_launch_agent,
};
use crate::lsif_index::{generate_and_import_rust_analyzer_lsif, import_rust_analyzer_lsif};
use crate::maintenance::{MaintenanceOptions, MaintenanceReport, run_maintenance};
use crate::mcp::{run_http, run_http_smoke, run_smoke, run_stdio};
use crate::production_audit::run_production_audit;
use crate::project::{resolve_project_id, resolve_project_id_from_path};
use crate::retrieval::RetrievalMode as SearchMode;
use crate::retrieval::{
    ContextRetrievalOutput, ContextRetrievalRequest, RetrievalMode, format_retrieval_diagnostics,
    run_context_retrieval,
};
use crate::search::{
    CodeSearchRequest, MemorySearchRequest, code_model_for_role, ollama_from_config, search_code,
    search_code_with_mode, search_memories,
};
use crate::semantic_ops::{SemanticOperationRequest, run_semantic_operation};
use crate::store::{
    CORE_MEMORY_KINDS, CodeMemory, CodeMemorySearchOptions, CodeRelation, CodeRouteHint,
    CodeSearchResult, CodeSimilarityPairOptions, CodeSymbol, DEFAULT_MEMORY_SCOPE,
    DEFAULT_MEMORY_TIER, EvalRunRecord, ListOptions, MEMORY_SCOPES, Memory, MemoryEdge,
    MemoryEntity, MemoryFact, MemoryStatus, NewCodeMemory, NewMemory, ProjectExport,
    ProjectProfileUpdate, RememberOutcome, SearchOptions, StatusFilter, Store,
};
use crate::validation::{ValidationAction, validate_memories};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Project-scoped memory engine for Codex and local agents"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Check local configuration, database path, and Ollama model availability.
    Doctor {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        deep: bool,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        skip_eval: bool,
        #[arg(long)]
        skip_rust_analyzer_diagnostics: bool,
        #[arg(long)]
        skip_launchd: bool,
    },
    /// Store a project-scoped memory entry.
    Remember {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value = "note")]
        kind: String,
        #[arg(long)]
        source: Option<String>,
        #[arg(long, default_value = "active")]
        status: String,
        #[arg(long, default_value = DEFAULT_MEMORY_TIER)]
        memory_tier: String,
        #[arg(long, default_value_t = 0.5)]
        importance: f64,
        #[arg(long, default_value_t = 0.7)]
        confidence: f64,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long)]
        no_embed: bool,
        #[arg(long)]
        deduplicate: bool,
        #[arg(long)]
        allow_sensitive: bool,
        text: String,
    },
    /// Search only the current or explicitly selected project memory.
    Search {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value = "active")]
        status: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        memory_tier: Option<String>,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        query: String,
    },
    /// Build prompt-ready project context from memory and indexed code.
    Context {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        memory_limit: usize,
        #[arg(long, default_value_t = 5)]
        core_memory_limit: usize,
        #[arg(long, default_value_t = 8)]
        code_limit: usize,
        #[arg(long, default_value_t = 6000)]
        token_budget: usize,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        query: String,
    },
    /// Plan minimal memory/code/graph access for one task without loading full context.
    ContextPlan {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        memory_limit: usize,
        #[arg(long, default_value_t = 5)]
        core_memory_limit: usize,
        #[arg(long, default_value_t = 8)]
        code_limit: usize,
        #[arg(long, default_value_t = 6000)]
        token_budget: usize,
        #[arg(long)]
        json: bool,
        query: String,
    },
    /// Auto-index the project and then build prompt-ready context.
    Prepare {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        memory_limit: usize,
        #[arg(long, default_value_t = 5)]
        core_memory_limit: usize,
        #[arg(long, default_value_t = 8)]
        code_limit: usize,
        #[arg(long, default_value_t = 6000)]
        token_budget: usize,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long)]
        no_index: bool,
        #[arg(long)]
        full: bool,
        #[arg(long)]
        embed: bool,
        #[arg(long, default_value_t = 500)]
        embed_limit: usize,
        #[arg(long)]
        validate_pending: bool,
        #[arg(long)]
        apply_policy: bool,
        #[arg(long, default_value_t = 20)]
        validate_limit: usize,
        query: String,
    },
    /// List memory entries for the current or explicitly selected project.
    List {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[arg(long, default_value = "active")]
        status: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        memory_tier: Option<String>,
    },
    /// Open a native read-only memory viewer window.
    #[command(name = "dukememory_app")]
    DukememoryApp {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long, default_value = "any")]
        status: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long, default_value_t = 100)]
        limit: usize,
    },
    /// Review pending memories in a compact promotion-oriented format.
    Review {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Apply pending memory review decisions in batch.
    ReviewApply {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long = "decision")]
        decisions: Vec<String>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Validate pending memories with the configured validation model.
    ValidatePending {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        apply: bool,
    },
    /// Read one memory by id, scoped to the current or selected project.
    Get {
        #[arg(long)]
        project: Option<String>,
        id: String,
    },
    /// Promote a pending memory to active.
    Promote {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        reason: Option<String>,
        id: String,
    },
    /// Archive a memory without physically deleting it.
    Archive {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        reason: Option<String>,
        id: String,
    },
    /// Archive pending memories, optionally limited to low-confidence candidates.
    PrunePending {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        max_confidence: Option<f64>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long, default_value = "pruned pending memory")]
        reason: String,
    },
    /// Compact older active memories into one durable project summary.
    Compact {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 40)]
        limit: usize,
        #[arg(long, default_value_t = 20)]
        min_memories: usize,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        apply: bool,
        #[arg(long)]
        no_embed: bool,
    },
    /// Run project maintenance: backup, validation, compaction, and embedding rebuild checks.
    Maintenance {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        apply: bool,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        backup: bool,
        #[arg(long)]
        backup_output: Option<PathBuf>,
        #[arg(long)]
        validate_pending: bool,
        #[arg(long, default_value_t = 20)]
        validate_limit: usize,
        #[arg(long)]
        compact: bool,
        #[arg(long, default_value_t = 40)]
        compact_limit: usize,
        #[arg(long, default_value_t = 20)]
        compact_min_memories: usize,
        #[arg(long)]
        feedback: bool,
        #[arg(long, default_value_t = 100)]
        feedback_limit: usize,
        #[arg(long)]
        embed_missing: bool,
        #[arg(long, default_value_t = 50)]
        embed_limit: usize,
        #[arg(long, default_value = "all")]
        embed_scope: String,
        #[arg(long)]
        json: bool,
    },
    /// Print, install, or load a macOS LaunchAgent for recurring maintenance.
    MaintenanceLaunchd {
        #[arg(long)]
        install: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        load: bool,
        #[arg(long)]
        unload: bool,
        #[arg(long)]
        plist: Option<PathBuf>,
        #[arg(long)]
        command: Option<PathBuf>,
        #[arg(long, default_value_t = 21_600)]
        interval_seconds: u64,
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        no_all: bool,
        #[arg(long)]
        no_backup: bool,
        #[arg(long)]
        stdout: Option<PathBuf>,
        #[arg(long)]
        stderr: Option<PathBuf>,
    },
    /// Replace an old memory with a new active version and preserve history.
    Supersede {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        old_id: String,
        #[arg(long, default_value = "note")]
        kind: String,
        #[arg(long)]
        source: Option<String>,
        #[arg(long, default_value_t = 0.7)]
        importance: f64,
        #[arg(long, default_value_t = 0.8)]
        confidence: f64,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long)]
        reason: Option<String>,
        #[arg(long)]
        no_embed: bool,
        #[arg(long)]
        allow_sensitive: bool,
        text: String,
    },
    /// Show memory counts by lifecycle status.
    Status {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Show PostgreSQL store metrics and index coverage.
    Health {
        #[arg(long)]
        json: bool,
    },
    /// Show recent project audit events.
    AuditLog {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Print a compact operations report: status, health, last eval, and recent audit.
    OpsReport {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        audit_limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Drop temporary PostgreSQL schemas created by smoke tests and audits.
    CleanupSchemas {
        #[arg(long)]
        apply: bool,
    },
    /// Upsert a project-scoped memory graph entity.
    GraphEntity {
        #[arg(long)]
        project: Option<String>,
        #[arg(long = "type", default_value = "concept")]
        entity_type: String,
        #[arg(long = "alias")]
        aliases: Vec<String>,
        #[arg(long)]
        description: Option<String>,
        name: String,
    },
    /// Add a project-scoped memory graph fact.
    GraphFact {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        entity_id: Option<String>,
        #[arg(long)]
        entity_name: Option<String>,
        #[arg(long = "entity-type", default_value = "concept")]
        entity_type: String,
        #[arg(long)]
        memory_id: Option<String>,
        #[arg(long, default_value_t = 0.7)]
        confidence: f64,
        predicate: String,
        value: String,
    },
    /// Add a project-scoped memory graph edge.
    GraphEdge {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        from_id: Option<String>,
        #[arg(long)]
        from_name: Option<String>,
        #[arg(long = "from-type", default_value = "concept")]
        from_type: String,
        #[arg(long)]
        to_id: Option<String>,
        #[arg(long)]
        to_name: Option<String>,
        #[arg(long = "to-type", default_value = "concept")]
        to_type: String,
        #[arg(long)]
        memory_id: Option<String>,
        #[arg(long, default_value_t = 0.7)]
        confidence: f64,
        relation_type: String,
    },
    /// Search project-scoped memory graph entities, facts, and edges.
    GraphSearch {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        query: String,
    },
    /// Extract a memory knowledge graph from project memories using the configured LLM.
    #[command(name = "dukememory_graph_extract")]
    DukememoryGraphExtract {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value = "active")]
        status: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        apply: bool,
    },
    /// Show or update the universal project memory profile.
    Profile {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        root_path: Option<PathBuf>,
        #[arg(long)]
        project_type: Option<String>,
        #[arg(long)]
        description: Option<String>,
        #[arg(long = "domain")]
        domains: Vec<String>,
    },
    /// Print universal memory ontology: core kinds, scopes, and domain examples.
    Ontology,
    /// Run a retrieval regression eval suite for one project.
    Eval {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value = "keyword")]
        mode: String,
        #[arg(long)]
        compare_last: bool,
        #[arg(long)]
        json: bool,
    },
    /// Create a consistent PostgreSQL backup of the whole dukememory database.
    Backup {
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Export one project as portable JSON without embedding blobs.
    Export {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        no_code: bool,
    },
    /// Import a dukememory project JSON export.
    Import {
        file: PathBuf,
        #[arg(long)]
        overwrite: bool,
    },
    /// Index Rust code for the current or selected project.
    CodeIndex {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        full: bool,
        #[arg(long)]
        embed: bool,
        #[arg(long, default_value_t = 500)]
        embed_limit: usize,
    },
    /// Import rust-analyzer LSIF definitions/references into the code graph.
    CodeLsifIndex {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Show indexed code counts for the current or selected project.
    CodeStatus {
        #[arg(long)]
        project: Option<String>,
    },
    /// List all indexed code files in the current or selected project.
    CodeFiles {
        #[arg(long)]
        project: Option<String>,
    },
    /// Outline all code symbols in a specific indexed file.
    CodeOutline {
        #[arg(long)]
        project: Option<String>,
        file_path: String,
    },
    /// Search indexed code symbols.
    CodeSearch {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        file_path: Option<String>,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        query: String,
    },
    /// One-call code exploration: symbols, code memories, routes, impact, freshness.
    CodeExplore {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value_t = 12)]
        relation_limit: usize,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        file_path: Option<String>,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long)]
        no_body: bool,
        query: String,
    },
    /// Create, search, list, or archive durable code memories.
    CodeMemory {
        #[arg(long, default_value = "search")]
        action: String,
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        id: Option<String>,
        #[arg(long)]
        symbol_id: Option<String>,
        #[arg(long)]
        symbol: Option<String>,
        #[arg(long)]
        symbol_kind: Option<String>,
        #[arg(long)]
        file_path: Option<String>,
        #[arg(long, default_value = "note")]
        kind: String,
        #[arg(long)]
        source: Option<String>,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long, default_value_t = 0.8)]
        confidence: f64,
        #[arg(long, default_value = "active")]
        status: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        apply: bool,
        #[arg(long)]
        reason: Option<String>,
        text: Option<String>,
    },
    /// Find likely affected test files from changed source files.
    CodeAffected {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 5)]
        depth: usize,
        #[arg(long, default_value_t = 100)]
        limit: usize,
        files: Vec<String>,
    },
    /// Find reusable implementation patterns near a task using code embeddings.
    CodePatterns {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        file_path: Option<String>,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, default_value_t = 0.72)]
        min_similarity: f64,
        #[arg(long)]
        apply_memory_suggestions: bool,
        #[arg(long)]
        promote_patterns: bool,
        #[arg(long, default_value_t = 20)]
        apply_limit: usize,
        query: String,
    },
    /// Find near-duplicate indexed code symbols using stored code embeddings.
    CodeDuplicates {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        file_path: Option<String>,
        #[arg(long, default_value_t = 0.92)]
        min_similarity: f64,
    },
    /// Build a full embedding-assisted development report for a task.
    CodeAssist {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, default_value_t = 0.72)]
        pattern_similarity: f64,
        #[arg(long, default_value_t = 0.92)]
        duplicate_similarity: f64,
        #[arg(long)]
        apply_memory_suggestions: bool,
        #[arg(long)]
        promote_patterns: bool,
        #[arg(long, default_value_t = 20)]
        apply_limit: usize,
        query: String,
    },
    /// Build a review plan from changed files.
    CodeReviewPlan {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long, default_value_t = 0.92)]
        duplicate_similarity: f64,
        #[arg(long)]
        query: Option<String>,
        files: Vec<String>,
    },
    /// Run the dukedevsystem advisory orchestrator and File Entropy Score.
    #[command(name = "dukememory-devsystem")]
    DukememoryDevsystem {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        query: String,
        #[arg(long)]
        no_memory_write: bool,
        files: Vec<String>,
    },
    /// Run code-search eval cases against indexed code symbols.
    CodeEval {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long)]
        json: bool,
    },
    /// Explain one indexed code symbol using a configured code model.
    CodeBrief {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value = "fast_code")]
        model_role: String,
        symbol: String,
    },
    /// Build a code implementation plan from memory and indexed code.
    CodePlan {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        memory_limit: usize,
        #[arg(long, default_value_t = 8)]
        code_limit: usize,
        #[arg(long, default_value = "agent_code")]
        model_role: String,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long)]
        deterministic: bool,
        query: String,
    },
    /// Analyze code change risk from memory and indexed code.
    CodeRisk {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 8)]
        memory_limit: usize,
        #[arg(long, default_value_t = 8)]
        code_limit: usize,
        #[arg(long, default_value = "deep_code")]
        model_role: String,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long)]
        deterministic: bool,
        query: String,
    },
    /// Build missing embeddings for stored memories and indexed code symbols.
    EmbedMissing {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long, default_value = "all")]
        scope: String,
    },
    /// Run embedding-backed semantic operations such as dedupe, related lookup, clusters, and health.
    Semantic {
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        action: String,
        #[arg(long)]
        query: Option<String>,
        #[arg(long)]
        body: Option<String>,
        #[arg(long)]
        memory_id: Option<String>,
        #[arg(long)]
        symbol: Option<String>,
        #[arg(long)]
        file_path: Option<String>,
        #[arg(long)]
        input: Option<String>,
        #[arg(long)]
        other_project_id: Option<String>,
        #[arg(long = "expected-id")]
        expected_ids: Vec<String>,
        #[arg(long = "helpful-id")]
        helpful_ids: Vec<String>,
        #[arg(long = "unhelpful-id")]
        unhelpful_ids: Vec<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long, default_value = "active")]
        status: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        memory_tier: Option<String>,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, default_value_t = 0.0)]
        min_similarity: f64,
        #[arg(long)]
        target_memory_model: Option<String>,
        #[arg(long)]
        target_code_model: Option<String>,
        #[arg(long)]
        as_of: Option<String>,
        #[arg(long)]
        retrieval_event_id: Option<String>,
        #[arg(long)]
        outcome_kind: Option<String>,
        #[arg(long)]
        severity: Option<String>,
        #[arg(long)]
        apply: bool,
    },
    /// Extract pending memory candidates from stdin, a file, or --input using the configured LLM.
    DukememoryExtract {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value = "dukememory_extract")]
        source: String,
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long)]
        input: Option<String>,
        #[arg(long, default_value_t = 8)]
        max_candidates: usize,
        #[arg(long)]
        no_embed: bool,
        #[arg(long)]
        validate: bool,
        #[arg(long)]
        apply_policy: bool,
    },
    /// Read an indexed code symbol by id or exact name.
    ReadSymbol {
        #[arg(long)]
        project: Option<String>,
        symbol: String,
    },
    /// Find approximate callers for an indexed symbol.
    FindCallers {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        symbol: String,
    },
    /// Find approximate callees for an indexed symbol.
    FindCallees {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        symbol: String,
    },
    /// Show approximate callers and callees for an indexed symbol.
    Impact {
        #[arg(long)]
        project: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        symbol: String,
    },
    /// Test a configured Ollama embedding role and print vector dimensions.
    Embed {
        #[arg(long, default_value = "memory")]
        role: String,
        text: String,
    },
    /// Run the MCP stdio server exposing dukememory_* tools.
    Mcp,
    /// Run an experimental localhost Streamable HTTP-style MCP endpoint.
    McpHttp {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8765)]
        port: u16,
    },
    /// Run an isolated JSON-RPC smoke test through the MCP handler.
    McpSmoke,
    /// Run an end-to-end smoke test through the localhost MCP HTTP transport.
    McpHttpSmoke,
    /// Run an isolated end-to-end production audit against a temporary project and database.
    ProductionAudit,
    /// Verify installed Codex MCP config, hooks, and hook wrapper without modifying files.
    CodexAudit,
    /// Run a non-empty Codex hook payload through the wrapper into an isolated database.
    CodexHookAudit,
    /// Print or install Codex MCP config for the dukememory server.
    CodexConfig {
        #[arg(long)]
        install: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        command: Option<PathBuf>,
    },
    /// Print or install Codex hooks for pending memory extraction.
    CodexHooks {
        #[arg(long)]
        install: bool,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        script: Option<PathBuf>,
        #[arg(long = "event")]
        events: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::from_env()?;

    match cli.command {
        Command::Doctor {
            project,
            deep,
            json,
            skip_eval,
            skip_rust_analyzer_diagnostics,
            skip_launchd,
        } => {
            doctor(
                config,
                project,
                deep,
                json,
                skip_eval,
                skip_rust_analyzer_diagnostics,
                skip_launchd,
            )
            .await
        }
        Command::Remember {
            project,
            kind,
            source,
            status,
            memory_tier,
            importance,
            confidence,
            tags,
            no_embed,
            deduplicate,
            allow_sensitive,
            text,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let memory = NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier,
                kind,
                body: text.clone(),
                tags,
                source,
                status: MemoryStatus::parse(&status)?,
                importance,
                confidence,
                status_reason: None,
                allow_sensitive,
            };
            let outcome = if deduplicate {
                store.remember_deduplicated(&project_id, memory)?
            } else {
                let id = store.remember(&project_id, memory)?;
                RememberOutcome {
                    id,
                    inserted: true,
                    duplicate_of: None,
                }
            };
            if outcome.inserted {
                println!("stored memory {}", outcome.id);
            } else {
                println!(
                    "duplicate skipped {}",
                    outcome.duplicate_of.as_deref().unwrap_or(&outcome.id)
                );
            }
            println!("project: {project_id}");
            if !no_embed && outcome.inserted {
                match embed_memory(&config, &store, &project_id, &outcome.id, &text).await {
                    Ok(dimensions) => println!(
                        "embedding: stored model={} dimensions={dimensions}",
                        config.memory_embed_model()
                    ),
                    Err(error) => println!("embedding: skipped ({error})"),
                }
            }
            Ok(())
        }
        Command::Search {
            project,
            limit,
            status,
            kind,
            memory_tier,
            mode,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            let results = search_memories(
                &config,
                &store,
                MemorySearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit,
                    status: StatusFilter::parse(Some(status.as_str()), MemoryStatus::Active)?,
                    kind,
                    memory_tier,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            println!("project: {project_id}");
            println!("mode: {}", mode.as_str());
            println!("results: {}", results.len());
            for memory in results {
                print_memory(&memory);
            }
            Ok(())
        }
        Command::Context {
            project,
            memory_limit,
            core_memory_limit,
            code_limit,
            token_budget,
            mode,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = RetrievalMode::parse(&mode)?;
            let context_plan = plan_context_access(ContextPlanRequest {
                query: &query,
                memory_limit,
                core_memory_limit,
                code_limit,
                token_budget,
            });
            let ContextRetrievalOutput {
                memories: task_memories,
                code,
                diagnostics,
            } = run_context_retrieval(
                &config,
                &store,
                ContextRetrievalRequest {
                    project_id: &project_id,
                    query: &query,
                    memory_limit: context_plan.memory_limit,
                    code_limit: context_plan.code_limit,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let core_memories = if context_plan.core_memory_limit == 0 {
                Vec::new()
            } else {
                store.core_context_memories_for_query(
                    &project_id,
                    &query,
                    context_plan.core_memory_limit,
                )?
            };
            let memories = merge_core_and_task_memories(
                core_memories,
                task_memories,
                context_plan.memory_limit,
            );
            let code_memories = store.code_memories_for_code_results(
                &project_id,
                &code,
                context_plan.code_memory_limit,
            )?;
            let status = store.status(&project_id)?;
            let code_status = store.code_status(&project_id)?;
            let memory_fragments = build_memory_fragments(
                &query,
                &memories,
                context_plan.budget_plan.effective_token_budget,
            );
            let graph_memory_ids = fragment_memory_ids(&memory_fragments);
            let graph_limit = context_plan
                .graph_limit
                .max(memory_fragments.len())
                .max(context_plan.memory_limit)
                .max(context_plan.code_limit)
                .max(8);
            let graph =
                store.memory_graph_for_memories(&project_id, &graph_memory_ids, graph_limit)?;
            let mut text = format_task_context(TaskProjectContextFormat {
                project_id: &project_id,
                query: &query,
                mode: mode.as_str(),
                memories: &memories,
                memory_fragments: &memory_fragments,
                code: &code,
                code_memories: &code_memories,
                graph: &graph,
                total_memories: status.total_memories,
                indexed_symbols: code_status.symbols,
                token_budget: context_plan.budget_plan.effective_token_budget,
            });
            text.push_str(&format_cli_context_plan(&context_plan));
            text.push_str(&format_retrieval_diagnostics(&diagnostics));
            println!("{text}");
            Ok(())
        }
        Command::ContextPlan {
            project,
            memory_limit,
            core_memory_limit,
            code_limit,
            token_budget,
            json,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let status = store.status(&project_id)?;
            let code_status = store.code_status(&project_id)?;
            let plan = plan_context_access(ContextPlanRequest {
                query: &query,
                memory_limit,
                core_memory_limit,
                code_limit,
                token_budget,
            });
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "project_id": project_id,
                        "query": query,
                        "plan": plan,
                        "status": {
                            "active_memories": status.active_memories,
                            "pending_memories": status.pending_memories,
                            "indexed_symbols": code_status.symbols,
                            "code_symbol_embeddings": code_status.symbol_embeddings
                        }
                    }))?
                );
            } else {
                println!("project: {project_id}");
                println!("task_type: {}", plan.task_type);
                println!("token_budget: {}", plan.budget_plan.effective_token_budget);
                println!(
                    "limits: memory={} core={} code={} graph={} code_memories={}",
                    plan.memory_limit,
                    plan.core_memory_limit,
                    plan.code_limit,
                    plan.graph_limit,
                    plan.code_memory_limit
                );
                println!(
                    "sources: memories={} graph={} code_index={} code_neighborhood={} code_memories={} eval_history={}",
                    plan.source_plan.memories,
                    plan.source_plan.memory_graph,
                    plan.source_plan.code_index,
                    plan.source_plan.code_neighborhood,
                    plan.source_plan.code_memories,
                    plan.source_plan.eval_history
                );
                println!(
                    "status: active_memories={} indexed_symbols={}",
                    status.active_memories, code_status.symbols
                );
            }
            Ok(())
        }
        Command::Prepare {
            project,
            path,
            memory_limit,
            core_memory_limit,
            code_limit,
            token_budget,
            mode,
            no_index,
            full,
            embed,
            embed_limit,
            validate_pending,
            apply_policy,
            validate_limit,
            query,
        } => {
            let mut store = Store::open(&config.database_marker)?;
            let project_id = if no_index {
                match (project.as_ref(), path.as_ref()) {
                    (Some(project), _) => project.clone(),
                    (None, Some(path)) => resolve_project_id_from_path(path)?,
                    (None, None) => resolve_project_id(None)?,
                }
            } else {
                let path = path.unwrap_or(std::env::current_dir()?);
                let report = index_project(&mut store, &path, project, full)?;
                let embedded_symbols = if embed {
                    embed_indexed_code_symbols(
                        &config,
                        &store,
                        &report.project_id,
                        &report.indexed_files,
                        embed_limit,
                    )
                    .await?
                } else {
                    0
                };
                println!("auto_index: true");
                println!("project: {}", report.project_id);
                println!("root: {}", report.root_path.display());
                println!("full_rebuild: {}", report.full_rebuild);
                println!("files_indexed: {}", report.files_indexed);
                println!("files_skipped: {}", report.files_skipped);
                println!("files_deleted: {}", report.files_deleted);
                println!("symbols_indexed: {}", report.symbols_indexed);
                println!("relations_indexed: {}", report.relations_indexed);
                println!("relation_targets_reset: {}", report.relation_targets_reset);
                println!("calls_resolved: {}", report.calls_resolved);
                println!("uses_resolved: {}", report.uses_resolved);
                println!("modules_resolved: {}", report.modules_resolved);
                println!(
                    "index_timing_ms: total={} scan={} read_hash={} parse={} db_write={} delete={} resolve={}",
                    report.timing.total_ms,
                    report.timing.scan_ms,
                    report.timing.read_hash_ms,
                    report.timing.parse_ms,
                    report.timing.db_write_ms,
                    report.timing.delete_ms,
                    report.timing.resolve_ms
                );
                if embed {
                    println!("code_symbols_embedded: {embedded_symbols}");
                }
                println!();
                report.project_id
            };
            if no_index {
                println!("auto_index: false");
                println!("project: {project_id}");
                println!();
            }

            if validate_pending {
                let pending = store.list(
                    &project_id,
                    ListOptions {
                        limit: validate_limit,
                        offset: 0,
                        status: StatusFilter::One(MemoryStatus::Pending),
                        kind: None,
                        memory_tier: None,
                    },
                )?;
                let ollama = ollama_from_config(&config);
                let report =
                    validate_memories(&ollama, &config.validate_model, &project_id, &pending)
                        .await?;
                println!("memory_policy_model: {}", report.model);
                println!("memory_policy_apply: {apply_policy}");
                println!("memory_policy_decisions: {}", report.decisions.len());
                for decision in &report.decisions {
                    if apply_policy {
                        match decision.action {
                            ValidationAction::Promote => {
                                store.promote(&project_id, &decision.id, Some(&decision.reason))?
                            }
                            ValidationAction::Archive => {
                                store.archive(&project_id, &decision.id, Some(&decision.reason))?
                            }
                            ValidationAction::Keep => {}
                        }
                    }
                    println!(
                        "- {} {} {:.2}: {}",
                        decision.id,
                        decision.action.as_str(),
                        decision.confidence,
                        decision.reason
                    );
                }
                println!();
            }

            let mode = RetrievalMode::parse(&mode)?;
            let context_plan = plan_context_access(ContextPlanRequest {
                query: &query,
                memory_limit,
                core_memory_limit,
                code_limit,
                token_budget,
            });
            let ContextRetrievalOutput {
                memories: task_memories,
                code,
                diagnostics,
            } = run_context_retrieval(
                &config,
                &store,
                ContextRetrievalRequest {
                    project_id: &project_id,
                    query: &query,
                    memory_limit: context_plan.memory_limit,
                    code_limit: context_plan.code_limit,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let core_memories = if context_plan.core_memory_limit == 0 {
                Vec::new()
            } else {
                store.core_context_memories_for_query(
                    &project_id,
                    &query,
                    context_plan.core_memory_limit,
                )?
            };
            let memories = merge_core_and_task_memories(
                core_memories,
                task_memories,
                context_plan.memory_limit,
            );
            let code_memories = store.code_memories_for_code_results(
                &project_id,
                &code,
                context_plan.code_memory_limit,
            )?;
            let status = store.status(&project_id)?;
            let code_status = store.code_status(&project_id)?;
            let memory_fragments = build_memory_fragments(
                &query,
                &memories,
                context_plan.budget_plan.effective_token_budget,
            );
            let graph_memory_ids = fragment_memory_ids(&memory_fragments);
            let graph_limit = context_plan
                .graph_limit
                .max(memory_fragments.len())
                .max(context_plan.memory_limit)
                .max(context_plan.code_limit)
                .max(8);
            let graph =
                store.memory_graph_for_memories(&project_id, &graph_memory_ids, graph_limit)?;
            let mut text = format_task_context(TaskProjectContextFormat {
                project_id: &project_id,
                query: &query,
                mode: mode.as_str(),
                memories: &memories,
                memory_fragments: &memory_fragments,
                code: &code,
                code_memories: &code_memories,
                graph: &graph,
                total_memories: status.total_memories,
                indexed_symbols: code_status.symbols,
                token_budget: context_plan.budget_plan.effective_token_budget,
            });
            text.push_str(&format_cli_context_plan(&context_plan));
            text.push_str(&format_retrieval_diagnostics(&diagnostics));
            println!("{text}");
            Ok(())
        }
        Command::List {
            project,
            limit,
            offset,
            status,
            kind,
            memory_tier,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let results = store.list(
                &project_id,
                ListOptions {
                    limit,
                    offset,
                    status: StatusFilter::parse(Some(status.as_str()), MemoryStatus::Active)?,
                    kind,
                    memory_tier,
                },
            )?;
            println!("project: {project_id}");
            println!("results: {}", results.len());
            for memory in results {
                print_memory(&memory);
            }
            Ok(())
        }
        Command::DukememoryApp {
            project,
            path,
            status,
            kind,
            limit,
        } => {
            let project_id = match (project, path.as_ref()) {
                (Some(project), _) => resolve_project_id(Some(project))?,
                (None, Some(path)) => resolve_project_id_from_path(path)?,
                (None, None) => resolve_project_id(None)?,
            };
            gui::run_memory_viewer(&config, project_id, status, kind, limit)
        }
        Command::Review { project, limit } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let results = store.list(
                &project_id,
                ListOptions {
                    limit,
                    offset: 0,
                    status: StatusFilter::One(MemoryStatus::Pending),
                    kind: None,
                    memory_tier: None,
                },
            )?;
            println!("project: {project_id}");
            println!("pending: {}", results.len());
            for memory in results {
                print_review_memory(&memory);
            }
            Ok(())
        }
        Command::ReviewApply {
            project,
            file,
            decisions,
            dry_run,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let decisions = load_review_decisions(file.as_deref(), &decisions)?;
            let report = apply_review_decisions(&store, &project_id, &decisions, dry_run)?;
            println!("project: {project_id}");
            println!("dry_run: {dry_run}");
            println!("decisions: {}", report.decisions.len());
            println!("promoted: {}", report.promoted);
            println!("archived: {}", report.archived);
            println!("kept: {}", report.kept);
            for decision in &report.decisions {
                println!(
                    "- {} {}{}",
                    decision.action,
                    decision.id,
                    decision
                        .reason
                        .as_ref()
                        .map(|reason| format!(" reason={reason}"))
                        .unwrap_or_default()
                );
            }
            Ok(())
        }
        Command::ValidatePending {
            project,
            limit,
            apply,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let pending = store.list(
                &project_id,
                ListOptions {
                    limit,
                    offset: 0,
                    status: StatusFilter::One(MemoryStatus::Pending),
                    kind: None,
                    memory_tier: None,
                },
            )?;
            let ollama = ollama_from_config(&config);
            let report =
                validate_memories(&ollama, &config.validate_model, &project_id, &pending).await?;
            println!("project: {project_id}");
            println!("model: {}", report.model);
            println!("apply: {apply}");
            println!("decisions: {}", report.decisions.len());
            for decision in &report.decisions {
                if apply {
                    match decision.action {
                        ValidationAction::Promote => {
                            store.promote(&project_id, &decision.id, Some(&decision.reason))?
                        }
                        ValidationAction::Archive => {
                            store.archive(&project_id, &decision.id, Some(&decision.reason))?
                        }
                        ValidationAction::Keep => {}
                    }
                }
                println!();
                println!("id: {}", decision.id);
                println!("action: {}", decision.action.as_str());
                println!("confidence: {:.2}", decision.confidence);
                println!("reason: {}", decision.reason);
            }
            Ok(())
        }
        Command::Get { project, id } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            match store.get(&project_id, &id)? {
                Some(memory) => print_memory(&memory),
                None => println!("memory `{id}` was not found in project `{project_id}`"),
            }
            Ok(())
        }
        Command::Promote {
            project,
            reason,
            id,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            store.promote(&project_id, &id, reason.as_deref())?;
            println!("promoted memory {id}");
            println!("project: {project_id}");
            Ok(())
        }
        Command::Archive {
            project,
            reason,
            id,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            store.archive(&project_id, &id, reason.as_deref())?;
            println!("archived memory {id}");
            println!("project: {project_id}");
            Ok(())
        }
        Command::PrunePending {
            project,
            limit,
            max_confidence,
            dry_run,
            reason,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let pruned =
                store.prune_pending(&project_id, limit, max_confidence, dry_run, Some(&reason))?;
            println!("project: {project_id}");
            println!("dry_run: {dry_run}");
            println!("matched: {}", pruned.len());
            for memory in pruned {
                print_review_memory(&memory);
            }
            Ok(())
        }
        Command::Compact {
            project,
            limit,
            min_memories,
            kind,
            apply,
            no_embed,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let memories = store.active_memories_for_compaction(&project_id, limit, kind)?;
            println!("project: {project_id}");
            println!("candidate_memories: {}", memories.len());
            println!("min_memories: {min_memories}");
            println!("apply: {apply}");
            if memories.len() < min_memories {
                println!("status: skipped");
                println!("reason: not enough active memories for compaction");
                return Ok(());
            }

            let ollama = ollama_from_config(&config);
            let proposal =
                propose_compaction(&ollama, config.extract_model(), &project_id, &memories).await?;
            print_compaction_proposal(&proposal);
            if apply {
                let outcome =
                    apply_compaction(&config, &store, &project_id, &proposal, !no_embed).await?;
                println!("summary_memory: {}", outcome.id);
                println!("summary_inserted: {}", outcome.inserted);
                if let Some(duplicate_of) = outcome.duplicate_of {
                    println!("summary_duplicate_of: {duplicate_of}");
                }
            }
            Ok(())
        }
        Command::Maintenance {
            project,
            apply,
            all,
            backup,
            backup_output,
            validate_pending,
            validate_limit,
            compact,
            compact_limit,
            compact_min_memories,
            feedback,
            feedback_limit,
            embed_missing,
            embed_limit,
            embed_scope,
            json,
        } => {
            let project_id = resolve_project_id(project)?;
            let any_step =
                all || backup || validate_pending || compact || feedback || embed_missing;
            let report = run_maintenance(
                &config,
                &project_id,
                MaintenanceOptions {
                    apply,
                    backup: all || backup,
                    backup_output,
                    validate_pending: all || validate_pending || !any_step,
                    validate_limit,
                    compact: all || compact || !any_step,
                    compact_limit,
                    compact_min_memories,
                    feedback: all || feedback,
                    feedback_limit,
                    embed_missing: all || embed_missing,
                    embed_limit,
                    embed_scope,
                },
            )
            .await?;
            let store = Store::open(&config.database_marker)?;
            store.record_audit_event(
                &project_id,
                "cli",
                "maintenance_run",
                "maintenance",
                None,
                serde_json::to_value(&report)?,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_maintenance_report(&report);
            }
            Ok(())
        }
        Command::MaintenanceLaunchd {
            install,
            force,
            load,
            unload,
            plist,
            command,
            interval_seconds,
            project,
            dry_run,
            no_all,
            no_backup,
            stdout,
            stderr,
        } => {
            let plist_path = plist.unwrap_or(default_plist_path()?);
            if unload {
                unload_launch_agent(&plist_path)?;
                println!("unloaded: {}", plist_path.display());
                return Ok(());
            }

            let command = match command {
                Some(command) => command,
                None => std::env::current_exe().context("failed to resolve current executable")?,
            };
            let (default_stdout, default_stderr) = default_log_paths()?;
            let stdout_path = stdout.unwrap_or(default_stdout);
            let stderr_path = stderr.unwrap_or(default_stderr);
            let project = Some(match project {
                Some(project) => project,
                None => resolve_project_id(None)?,
            });
            let options = MaintenanceLaunchdOptions {
                command,
                label: DEFAULT_LABEL.to_string(),
                interval_seconds,
                project,
                apply: !dry_run,
                all: !no_all,
                backup: !no_backup,
                stdout_path: stdout_path.clone(),
                stderr_path: stderr_path.clone(),
            };
            let plist_text = maintenance_launchd_plist(&config, &options);

            if install {
                let result = install_maintenance_launchd(
                    &plist_path,
                    &plist_text,
                    &stdout_path,
                    &stderr_path,
                    force,
                )?;
                println!("installed: {}", result.installed);
                println!("plist: {}", result.plist_path.display());
                if let Some(backup_path) = result.backup_path {
                    println!("backup: {}", backup_path.display());
                }
                if load {
                    load_launch_agent(&plist_path)?;
                    println!("loaded: {}", plist_path.display());
                }
            } else if load {
                load_launch_agent(&plist_path)?;
                println!("loaded: {}", plist_path.display());
            } else {
                println!("{plist_text}");
            }
            Ok(())
        }
        Command::Supersede {
            project,
            old_id,
            kind,
            source,
            importance,
            confidence,
            tags,
            reason,
            no_embed,
            allow_sensitive,
            text,
        } => {
            let project_id = resolve_project_id(project)?;
            let mut store = Store::open(&config.database_marker)?;
            let new_id = store.supersede(
                &project_id,
                &old_id,
                NewMemory {
                    scope: DEFAULT_MEMORY_SCOPE.to_string(),
                    memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                    kind,
                    body: text.clone(),
                    tags,
                    source,
                    status: MemoryStatus::Active,
                    importance,
                    confidence,
                    status_reason: reason.clone(),
                    allow_sensitive,
                },
                reason.as_deref(),
            )?;
            println!("superseded memory {old_id}");
            println!("new memory {new_id}");
            println!("project: {project_id}");
            if !no_embed {
                match embed_memory(&config, &store, &project_id, &new_id, &text).await {
                    Ok(dimensions) => println!(
                        "embedding: stored model={} dimensions={dimensions}",
                        config.memory_embed_model()
                    ),
                    Err(error) => println!("embedding: skipped ({error})"),
                }
            }
            Ok(())
        }
        Command::Status { project, json } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let status = store.status(&project_id)?;
            let code_status = store.code_status(&project_id)?;
            let schema_version = store.schema_version()?;
            let integrity_check = store.integrity_check()?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "project": status,
                        "code_index": code_status,
                        "schema_version": schema_version,
                        "integrity_check": integrity_check,
                        "database_url": config.database_url,
                        "database_marker": config.database_marker
                    }))?
                );
                return Ok(());
            }
            println!("project: {}", status.project_id);
            println!("project_type: {}", status.project_type);
            println!("pending: {}", status.pending_memories);
            println!("active: {}", status.active_memories);
            println!("superseded: {}", status.superseded_memories);
            println!("archived: {}", status.archived_memories);
            println!("total: {}", status.total_memories);
            println!("memory_embeddings: {}", status.memory_embeddings);
            println!("code_files: {}", code_status.files);
            println!("code_symbols: {}", code_status.symbols);
            println!("code_relations: {}", code_status.relations);
            println!(
                "code_resolved_relations: {}",
                code_status.resolved_relations
            );
            println!("code_ra_references: {}", code_status.ra_references);
            println!("code_ra_calls: {}", code_status.ra_calls);
            println!("code_symbol_embeddings: {}", code_status.symbol_embeddings);
            println!("schema_version: {}", schema_version);
            println!("integrity_check: {}", integrity_check);
            Ok(())
        }
        Command::Health { json } => {
            let store = Store::open(&config.database_marker)?;
            let health = store.health()?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "database_url": config.database_url,
                        "database_marker": config.database_marker,
                        "health": health
                    }))?
                );
                return Ok(());
            }
            println!("database_url: {}", config.database_url);
            println!("schema: {}", health.schema);
            println!("database_size_bytes: {}", health.database_size_bytes);
            println!("projects: {}", health.projects);
            println!("memories: {}", health.memories);
            println!("memory_embeddings: {}", health.memory_embeddings);
            println!("code_files: {}", health.code_files);
            println!("code_symbols: {}", health.code_symbols);
            println!("code_relations: {}", health.code_relations);
            println!("code_symbol_embeddings: {}", health.code_symbol_embeddings);
            println!("memory_entities: {}", health.memory_entities);
            println!("memory_facts: {}", health.memory_facts);
            println!("memory_edges: {}", health.memory_edges);
            println!("audit_events: {}", health.audit_events);
            println!("temp_schemas: {}", health.temp_schemas);
            Ok(())
        }
        Command::AuditLog {
            project,
            limit,
            json,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let events = store.list_audit_events(&project_id, limit)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "project_id": project_id,
                        "events": events
                    }))?
                );
                return Ok(());
            }
            println!("project: {project_id}");
            println!("events: {}", events.len());
            for event in events {
                println!();
                println!("id: {}", event.id);
                println!("created_at: {}", event.created_at);
                println!("actor: {}", event.actor);
                println!("action: {}", event.action);
                println!("target_type: {}", event.target_type);
                println!(
                    "target_id: {}",
                    event.target_id.as_deref().unwrap_or("<none>")
                );
                println!("detail: {}", serde_json::to_string(&event.detail)?);
            }
            Ok(())
        }
        Command::OpsReport {
            project,
            audit_limit,
            json,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let project_status = store.status(&project_id)?;
            let code_status = store.code_status(&project_id)?;
            let health = store.health()?;
            let latest_eval = store.latest_eval_run_for_project(&project_id)?;
            let audit_events = store.list_audit_events(&project_id, audit_limit)?;
            let report = json!({
                "project_id": project_id,
                "database_url": config.database_url,
                "database_marker": config.database_marker,
                "schema_version": store.schema_version()?,
                "integrity_check": store.integrity_check()?,
                "project": project_status,
                "code_index": code_status,
                "health": health,
                "latest_eval": latest_eval,
                "audit_events": audit_events
            });
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!("project: {}", report["project_id"].as_str().unwrap_or(""));
                println!("schema_version: {}", report["schema_version"]);
                println!("integrity_check: {}", report["integrity_check"]);
                println!(
                    "memories: total={} active={} pending={} archived={}",
                    report["project"]["total_memories"],
                    report["project"]["active_memories"],
                    report["project"]["pending_memories"],
                    report["project"]["archived_memories"]
                );
                println!(
                    "code_index: files={} symbols={} embeddings={} relations={} resolved={} ra_refs={} ra_calls={}",
                    report["code_index"]["files"],
                    report["code_index"]["symbols"],
                    report["code_index"]["symbol_embeddings"],
                    report["code_index"]["relations"],
                    report["code_index"]["resolved_relations"],
                    report["code_index"]["ra_references"],
                    report["code_index"]["ra_calls"]
                );
                println!(
                    "health: audit_events={} temp_schemas={} database_size_bytes={}",
                    report["health"]["audit_events"],
                    report["health"]["temp_schemas"],
                    report["health"]["database_size_bytes"]
                );
                if report["latest_eval"].is_null() {
                    println!("latest_eval: <none>");
                } else {
                    println!(
                        "latest_eval: id={} passed={}/{} failed={} mode={}",
                        report["latest_eval"]["id"],
                        report["latest_eval"]["passed_cases"],
                        report["latest_eval"]["total_cases"],
                        report["latest_eval"]["failed_cases"],
                        report["latest_eval"]["mode"]
                    );
                }
                println!(
                    "recent_audit_events: {}",
                    report["audit_events"].as_array().map(Vec::len).unwrap_or(0)
                );
            }
            Ok(())
        }
        Command::CleanupSchemas { apply } => {
            let store = Store::open(&config.database_marker)?;
            let report = store.cleanup_temp_schemas(!apply)?;
            let project_id = resolve_project_id(None)?;
            store.record_audit_event(
                &project_id,
                "cli",
                "cleanup_schemas",
                "database",
                None,
                json!({
                    "dry_run": report.dry_run,
                    "dropped": report.dropped.len(),
                    "kept": report.kept.len()
                }),
            )?;
            println!("dry_run: {}", report.dry_run);
            println!("dropped: {}", report.dropped.len());
            for schema in &report.dropped {
                println!("- dropped {schema}");
            }
            println!("kept: {}", report.kept.len());
            for schema in &report.kept {
                println!("- kept {schema}");
            }
            Ok(())
        }
        Command::GraphEntity {
            project,
            entity_type,
            aliases,
            description,
            name,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let entity = store.upsert_memory_entity(
                &project_id,
                &entity_type,
                &name,
                aliases,
                description,
            )?;
            print_memory_entity(&entity);
            Ok(())
        }
        Command::GraphFact {
            project,
            entity_id,
            entity_name,
            entity_type,
            memory_id,
            confidence,
            predicate,
            value,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let entity_id = match (entity_id, entity_name) {
                (Some(id), _) => Some(id),
                (None, Some(name)) => Some(
                    store
                        .upsert_memory_entity(&project_id, &entity_type, &name, Vec::new(), None)?
                        .id,
                ),
                (None, None) => None,
            };
            let fact = store.add_memory_fact(
                &project_id,
                entity_id.as_deref(),
                memory_id.as_deref(),
                &predicate,
                &value,
                confidence,
            )?;
            print_memory_fact(&fact);
            Ok(())
        }
        Command::GraphEdge {
            project,
            from_id,
            from_name,
            from_type,
            to_id,
            to_name,
            to_type,
            memory_id,
            confidence,
            relation_type,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let from_id = match (from_id, from_name) {
                (Some(id), _) => id,
                (None, Some(name)) => {
                    store
                        .upsert_memory_entity(&project_id, &from_type, &name, Vec::new(), None)?
                        .id
                }
                (None, None) => bail!("graph-edge requires --from-id or --from-name"),
            };
            let to_id = match (to_id, to_name) {
                (Some(id), _) => id,
                (None, Some(name)) => {
                    store
                        .upsert_memory_entity(&project_id, &to_type, &name, Vec::new(), None)?
                        .id
                }
                (None, None) => bail!("graph-edge requires --to-id or --to-name"),
            };
            let edge = store.add_memory_edge(
                &project_id,
                &from_id,
                &to_id,
                &relation_type,
                memory_id.as_deref(),
                confidence,
            )?;
            print_memory_edge(&edge);
            Ok(())
        }
        Command::GraphSearch {
            project,
            limit,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let graph = store.search_memory_graph(&project_id, &query, limit)?;
            println!("project: {project_id}");
            println!("entities: {}", graph.entities.len());
            for entity in &graph.entities {
                print_memory_entity(entity);
            }
            println!("facts: {}", graph.facts.len());
            for fact in &graph.facts {
                print_memory_fact(fact);
            }
            println!("edges: {}", graph.edges.len());
            for edge in &graph.edges {
                print_memory_edge(edge);
            }
            Ok(())
        }
        Command::DukememoryGraphExtract {
            project,
            limit,
            status,
            kind,
            query,
            apply,
        } => {
            let project_id = resolve_project_id(project)?;
            let status = StatusFilter::parse(Some(status.as_str()), MemoryStatus::Active)?;
            let store = Store::open(&config.database_marker)?;
            let memories =
                if let Some(query) = query.as_deref().filter(|value| !value.trim().is_empty()) {
                    store.search(
                        &project_id,
                        SearchOptions {
                            query: query.to_string(),
                            limit,
                            status,
                            kind,
                            memory_tier: None,
                        },
                    )?
                } else {
                    store.list(
                        &project_id,
                        ListOptions {
                            limit,
                            offset: 0,
                            status,
                            kind,
                            memory_tier: None,
                        },
                    )?
                };
            let ollama = ollama_from_config(&config);
            let proposals = extract_memory_graph(&ollama, &project_id, &memories).await?;
            let report = apply_graph_extraction(&store, &project_id, proposals, apply)?;
            print_graph_extraction_report(&report);
            Ok(())
        }
        Command::Profile {
            project,
            name,
            root_path,
            project_type,
            description,
            domains,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let has_update = name.is_some()
                || root_path.is_some()
                || project_type.is_some()
                || description.is_some()
                || !domains.is_empty();
            let profile = if has_update {
                store.update_project_profile(
                    &project_id,
                    ProjectProfileUpdate {
                        name,
                        root_path: root_path.map(|path| path.display().to_string()),
                        project_type,
                        description,
                        domains: if domains.is_empty() {
                            None
                        } else {
                            Some(domains)
                        },
                    },
                )?
            } else {
                store.project_profile(&project_id)?
            };
            print_project_profile(&profile);
            Ok(())
        }
        Command::Ontology => {
            print_ontology();
            Ok(())
        }
        Command::Eval {
            project,
            file,
            limit,
            mode,
            compare_last,
            json,
        } => {
            let default_project_id = resolve_project_id(project)?;
            let suite_text = std::fs::read_to_string(&file)
                .with_context(|| format!("failed to read eval suite {}", file.display()))?;
            let suite: EvalSuite = serde_json::from_str(&suite_text)
                .with_context(|| format!("failed to parse eval suite {}", file.display()))?;
            let suite_hash = blake3::hash(suite_text.as_bytes()).to_hex().to_string();
            let suite_name = suite.name.clone().or_else(|| {
                file.file_stem()
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
            });
            let mode = SearchMode::parse(&mode)?;
            let store = Store::open(&config.database_marker)?;
            let report = run_eval_suite(
                &config,
                &store,
                EvalSuiteRun {
                    default_project_id: &default_project_id,
                    suite,
                    suite_name,
                    suite_hash: Some(suite_hash),
                    mode,
                    limit,
                },
            )
            .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_eval_report(&report, compare_last);
            }
            if report.failed_cases > 0 {
                bail!("eval failed: {} cases failed", report.failed_cases);
            }
            Ok(())
        }
        Command::Backup { output } => {
            let store = Store::open(&config.database_marker)?;
            let report = create_database_backup(&store, &config.database_marker, output)?;
            println!("database_url: {}", config.database_url);
            println!("database_marker: {}", report.source.display());
            println!("backup: {}", report.output.display());
            println!("size_bytes: {}", report.size_bytes);
            Ok(())
        }
        Command::Export {
            project,
            output,
            no_code,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let export = store.export_project(&project_id, !no_code)?;
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create export directory {}", parent.display())
                })?;
            }
            let json = serde_json::to_string_pretty(&export)?;
            std::fs::write(&output, json)
                .with_context(|| format!("failed to write export {}", output.display()))?;
            println!("project: {}", export.project.id);
            println!("output: {}", output.display());
            println!("schema_version: {}", export.schema_version);
            println!("include_code: {}", export.includes_code);
            println!("memories: {}", export.memories.len());
            println!("code_files: {}", export.code_files.len());
            println!("code_symbols: {}", export.code_symbols.len());
            println!("code_relations: {}", export.code_relations.len());
            println!("embeddings: not exported; run embed-missing after import");
            Ok(())
        }
        Command::Import { file, overwrite } => {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("failed to read import file {}", file.display()))?;
            let export: ProjectExport = serde_json::from_str(&text)
                .with_context(|| format!("failed to parse import file {}", file.display()))?;
            let mut store = Store::open(&config.database_marker)?;
            let report = store.import_project(export, overwrite)?;
            println!("project: {}", report.project_id);
            println!("overwrite: {}", report.overwrite);
            println!("memories_imported: {}", report.memories_imported);
            println!("memories_skipped: {}", report.memories_skipped);
            println!("code_files_imported: {}", report.code_files_imported);
            println!("code_files_skipped: {}", report.code_files_skipped);
            println!("code_symbols_imported: {}", report.code_symbols_imported);
            println!("code_symbols_skipped: {}", report.code_symbols_skipped);
            println!(
                "code_relations_imported: {}",
                report.code_relations_imported
            );
            println!("code_relations_skipped: {}", report.code_relations_skipped);
            println!("embeddings: run embed-missing for restored memories/code symbols");
            Ok(())
        }
        Command::CodeIndex {
            project,
            path,
            full,
            embed,
            embed_limit,
        } => {
            let path = path.unwrap_or(std::env::current_dir()?);
            let mut store = Store::open(&config.database_marker)?;
            let report = index_project(&mut store, &path, project, full)?;
            let embedded_symbols = if embed {
                embed_indexed_code_symbols(
                    &config,
                    &store,
                    &report.project_id,
                    &report.indexed_files,
                    embed_limit,
                )
                .await?
            } else {
                0
            };
            store.record_audit_event(
                &report.project_id,
                "cli",
                "code_index",
                "code_index",
                None,
                json!({
                    "root_path": report.root_path.display().to_string(),
                    "full_rebuild": report.full_rebuild,
                    "files_seen": report.files_seen,
                    "files_indexed": report.files_indexed,
                    "files_skipped": report.files_skipped,
                    "files_deleted": report.files_deleted,
                    "symbols_indexed": report.symbols_indexed,
                    "relations_indexed": report.relations_indexed,
                    "relation_targets_reset": report.relation_targets_reset,
                    "calls_resolved": report.calls_resolved,
                    "uses_resolved": report.uses_resolved,
                    "modules_resolved": report.modules_resolved,
                    "embed": embed,
                    "embedded_symbols": embedded_symbols
                }),
            )?;
            println!("project: {}", report.project_id);
            println!("root: {}", report.root_path.display());
            println!("full_rebuild: {}", report.full_rebuild);
            println!("files_seen: {}", report.files_seen);
            println!("files_indexed: {}", report.files_indexed);
            println!("files_skipped: {}", report.files_skipped);
            println!("files_deleted: {}", report.files_deleted);
            println!("symbols_indexed: {}", report.symbols_indexed);
            println!("relations_indexed: {}", report.relations_indexed);
            println!("relation_targets_reset: {}", report.relation_targets_reset);
            println!("calls_resolved: {}", report.calls_resolved);
            println!("uses_resolved: {}", report.uses_resolved);
            println!("modules_resolved: {}", report.modules_resolved);
            if embed {
                println!("code_symbols_embedded: {embedded_symbols}");
            }
            Ok(())
        }
        Command::CodeLsifIndex {
            project,
            path,
            input,
        } => {
            let path = path.unwrap_or(std::env::current_dir()?);
            let project_id = match project {
                Some(project) => resolve_project_id(Some(project))?,
                None => resolve_project_id_from_path(&path)?,
            };
            let store = Store::open(&config.database_marker)?;
            let report = if let Some(input) = input {
                let lsif = std::fs::read_to_string(&input)
                    .with_context(|| format!("failed to read LSIF {}", input.display()))?;
                import_rust_analyzer_lsif(
                    &store,
                    &project_id,
                    &path,
                    &input.display().to_string(),
                    &lsif,
                )?
            } else {
                generate_and_import_rust_analyzer_lsif(&store, &project_id, &path)?
            };
            store.record_audit_event(
                &project_id,
                "cli",
                "code_lsif_index",
                "code_index",
                None,
                json!({
                    "root_path": report.root_path.display().to_string(),
                    "source": &report.source,
                    "documents": report.documents,
                    "ranges": report.ranges,
                    "definitions_seen": report.definitions_seen,
                    "reference_ranges_seen": report.reference_ranges_seen,
                    "target_symbols_resolved": report.target_symbols_resolved,
                    "stale_relations_removed": report.stale_relations_removed,
                    "relations_imported": report.relations_imported,
                    "call_relations_imported": report.call_relations_imported,
                    "relations_skipped": report.relations_skipped
                }),
            )?;
            println!("project: {}", report.project_id);
            println!("root: {}", report.root_path.display());
            println!("source: {}", report.source);
            println!("documents: {}", report.documents);
            println!("ranges: {}", report.ranges);
            println!("definitions_seen: {}", report.definitions_seen);
            println!("reference_ranges_seen: {}", report.reference_ranges_seen);
            println!(
                "target_symbols_resolved: {}",
                report.target_symbols_resolved
            );
            println!(
                "stale_relations_removed: {}",
                report.stale_relations_removed
            );
            println!("relations_imported: {}", report.relations_imported);
            println!(
                "call_relations_imported: {}",
                report.call_relations_imported
            );
            println!("relations_skipped: {}", report.relations_skipped);
            Ok(())
        }
        Command::CodeStatus { project } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let status = store.code_status(&project_id)?;
            println!("project: {}", status.project_id);
            println!("files: {}", status.files);
            println!("symbols: {}", status.symbols);
            println!("relations: {}", status.relations);
            println!("resolved_relations: {}", status.resolved_relations);
            println!("ra_references: {}", status.ra_references);
            println!("ra_calls: {}", status.ra_calls);
            println!("symbol_embeddings: {}", status.symbol_embeddings);
            println!("relation_counts:");
            println!(
                "- total={}, resolved={}",
                status.relation_counts.total, status.relation_counts.total_resolved
            );
            println!(
                "- project_quality={}, resolved={}, unresolved={}",
                status.relation_counts.project_quality,
                status.relation_counts.project_quality_resolved,
                status.relation_counts.project_quality_unresolved
            );
            println!(
                "- external={}, external_call={}, external_use={}, cargo_package={}",
                status.relation_counts.external,
                status.relation_counts.external_call,
                status.relation_counts.external_use,
                status.relation_counts.cargo_package
            );
            println!(
                "- ra_reference={}, ra_call={}",
                status.relation_counts.ra_reference, status.relation_counts.ra_call
            );
            println!(
                "relation_resolution_rate: {:.3}",
                status.quality.relation_resolution_rate
            );
            println!(
                "unresolved_relations: {}",
                status.quality.unresolved_relations
            );
            println!(
                "ambiguous_unresolved_targets: {}",
                status.quality.ambiguous_unresolved_targets
            );
            if !status.languages.is_empty() {
                println!("languages:");
                for language in &status.languages {
                    println!(
                        "- {}: files={}, symbols={}, relations={}, resolved={}, embeddings={}",
                        language.language,
                        language.files,
                        language.symbols,
                        language.relations,
                        language.resolved_relations,
                        language.symbol_embeddings
                    );
                }
            }
            if !status.quality.top_unresolved_targets.is_empty() {
                println!("top_unresolved_targets:");
                for target in &status.quality.top_unresolved_targets {
                    println!(
                        "- {} {}: {}",
                        target.relation_kind, target.target_name, target.count
                    );
                }
            }
            Ok(())
        }
        Command::CodeFiles { project } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let files = store.code_files_for_project(&project_id)?;
            println!(
                "Found {} indexed files in project `{}`:",
                files.len(),
                project_id
            );
            for file in files {
                println!(
                    "- path: {}\n  language: {}\n  size: {} bytes\n  lines: {}",
                    file.path, file.language, file.size_bytes, file.line_count
                );
            }
            Ok(())
        }
        Command::CodeOutline { project, file_path } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let symbols = store.code_symbols_for_file(&project_id, &file_path)?;
            println!(
                "Outline for `{}` in project `{}` ({} symbols):",
                file_path,
                project_id,
                symbols.len()
            );
            for symbol in symbols {
                print!(
                    "- name: {}\n  kind: {}\n  lines: {}-{}\n  id: {}\n  signature: {}",
                    symbol.name,
                    symbol.kind,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.id,
                    symbol.signature
                );
                if let Some(parent_id) = &symbol.parent_id {
                    print!("\n  parent_id: {}", parent_id);
                }
                println!();
            }
            Ok(())
        }
        Command::CodeSearch {
            project,
            limit,
            kind,
            file_path,
            mode,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            let results = search_code(
                &config,
                &store,
                CodeSearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit,
                    kind,
                    file_path,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            println!("project: {project_id}");
            println!("mode: {}", mode.as_str());
            println!("results: {}", results.len());
            for result in results {
                print_code_search_result(&result);
            }
            Ok(())
        }
        Command::CodeExplore {
            project,
            path,
            limit,
            relation_limit,
            kind,
            file_path,
            mode,
            no_body,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            let results = search_code(
                &config,
                &store,
                CodeSearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit,
                    kind,
                    file_path,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let code_memories =
                store.code_memories_for_code_results(&project_id, &results, limit.max(8))?;
            let routes = store.route_hints(&project_id, &query, limit)?;
            println!("project: {project_id}");
            println!("query: {query}");
            if let Some(path) = path {
                let freshness =
                    check_code_index_freshness(&store, &path, Some(project_id.clone()))?;
                println!(
                    "freshness: stale_files={} missing_files={} deleted_files={}",
                    freshness.stale_files.len(),
                    freshness.missing_files.len(),
                    freshness.deleted_files.len()
                );
            }
            println!("symbols: {}", results.len());
            for result in &results {
                print_code_search_result(result);
                if !no_body {
                    println!("{}", result.symbol.body);
                }
                let callers = store.find_callers(&project_id, &result.symbol.id, relation_limit)?;
                let callees = store.find_callees(&project_id, &result.symbol.id, relation_limit)?;
                println!("callers: {}", callers.len());
                for relation in callers {
                    print_code_relation(&relation);
                }
                println!("callees: {}", callees.len());
                for relation in callees {
                    print_code_relation(&relation);
                }
            }
            print_code_memories("Related code memories:", &code_memories);
            print_route_hints("Route hints:", &routes);
            Ok(())
        }
        Command::CodeMemory {
            action,
            project,
            id,
            symbol_id,
            symbol,
            symbol_kind,
            file_path,
            kind,
            source,
            tags,
            confidence,
            status,
            limit,
            apply,
            reason,
            text,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let symbol_ref = symbol_id.or(symbol);
            match action.as_str() {
                "remember" => {
                    let body = match text {
                        Some(text) => text,
                        None => {
                            let mut input = String::new();
                            io::stdin().read_to_string(&mut input)?;
                            input
                        }
                    };
                    let outcome = store.remember_code_memory(
                        &project_id,
                        NewCodeMemory {
                            symbol_id: symbol_ref.clone(),
                            symbol_kind: symbol_kind.clone(),
                            file_path: file_path.clone(),
                            status: status.clone(),
                            kind,
                            body,
                            tags,
                            source,
                            confidence,
                        },
                        true,
                    )?;
                    println!("project: {project_id}");
                    println!("id: {}", outcome.id);
                    println!("inserted: {}", outcome.inserted);
                }
                "archive" => {
                    let id = id.context("code-memory --action archive requires --id")?;
                    store.archive_code_memory(&project_id, &id, reason.as_deref())?;
                    println!("archived: {id}");
                }
                "promote" => {
                    let id = id.context("code-memory --action promote requires --id")?;
                    store.promote_code_memory(&project_id, &id, reason.as_deref())?;
                    println!("promoted: {id}");
                }
                "repair" => {
                    let report = store.repair_code_memory_links(&project_id, limit, apply)?;
                    println!("project: {}", report.project_id);
                    println!("dry_run: {}", report.dry_run);
                    println!("scanned: {}", report.scanned);
                    println!("repaired: {}", report.repaired);
                    println!("ambiguous: {}", report.ambiguous);
                    println!("stale: {}", report.stale);
                    for result in report.results {
                        println!();
                        println!("id: {}", result.memory_id);
                        println!("status: {}", result.status);
                        println!("reason: {}", result.reason);
                        println!("candidates: {}", result.candidates);
                        println!(
                            "old_symbol_id: {}",
                            result.old_symbol_id.as_deref().unwrap_or("-")
                        );
                        println!(
                            "new_symbol_id: {}",
                            result.new_symbol_id.as_deref().unwrap_or("-")
                        );
                        println!("file_path: {}", result.file_path.as_deref().unwrap_or("-"));
                        println!(
                            "symbol: {} {}",
                            result.symbol_kind.as_deref().unwrap_or("-"),
                            result.symbol_name.as_deref().unwrap_or("-")
                        );
                    }
                }
                "list" | "search" => {
                    let results = store.search_code_memories(
                        &project_id,
                        CodeMemorySearchOptions {
                            query: text,
                            limit,
                            status,
                            kind: if kind == "note" { None } else { Some(kind) },
                            symbol_ids: match symbol_ref {
                                Some(symbol_ref) => vec![
                                    store
                                        .resolve_code_symbol_reference(
                                            &project_id,
                                            &symbol_ref,
                                            file_path.as_deref(),
                                            symbol_kind.as_deref(),
                                        )?
                                        .id,
                                ],
                                None => Vec::new(),
                            },
                            file_paths: file_path.into_iter().collect(),
                        },
                    )?;
                    print_code_memories(
                        &format!(
                            "Found {} code memories in project `{project_id}`:",
                            results.len()
                        ),
                        &results,
                    );
                }
                other => bail!("invalid code-memory action `{other}`"),
            }
            Ok(())
        }
        Command::CodeAffected {
            project,
            depth,
            limit,
            files,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let tests = store.affected_test_files(&project_id, &files, depth, limit)?;
            println!("project: {project_id}");
            println!("changed_files: {}", files.len());
            println!("affected_tests: {}", tests.len());
            for file in tests {
                println!("- {file}");
            }
            Ok(())
        }
        Command::CodePatterns {
            project,
            limit,
            kind,
            file_path,
            mode,
            min_similarity,
            apply_memory_suggestions,
            promote_patterns,
            apply_limit,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            let symbols = search_code(
                &config,
                &store,
                CodeSearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit,
                    kind,
                    file_path,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let report = build_code_pattern_report(
                &store,
                &project_id,
                &query,
                symbols,
                config.code_embed_model(),
                limit,
                min_similarity,
            )?;
            if apply_memory_suggestions {
                let applied = apply_code_memory_suggestions(
                    &store,
                    &project_id,
                    &report.memory_suggestions,
                    apply_limit,
                )?;
                println!("applied_memory_suggestions: {}", applied.len());
            }
            if promote_patterns {
                let applied = apply_code_memory_suggestions(
                    &store,
                    &project_id,
                    &report.pattern_promotions,
                    apply_limit,
                )?;
                println!("applied_pattern_promotions: {}", applied.len());
            }
            print_code_pattern_report(&report)?;
            Ok(())
        }
        Command::CodeDuplicates {
            project,
            limit,
            kind,
            file_path,
            min_similarity,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let pairs = store.code_similarity_pairs(
                &project_id,
                CodeSimilarityPairOptions {
                    embedding_model: config.code_embed_model().to_string(),
                    limit,
                    kind,
                    file_path,
                    min_similarity,
                },
            )?;
            println!("project: {project_id}");
            println!("model: {}", config.code_embed_model());
            println!("min_similarity: {min_similarity:.2}");
            println!("duplicate_pairs: {}", pairs.len());
            for pair in pairs {
                println!(
                    "- {:.3} {} {}:{} <-> {} {}:{}",
                    pair.similarity,
                    pair.left.kind,
                    pair.left.file_path,
                    pair.left.name,
                    pair.right.kind,
                    pair.right.file_path,
                    pair.right.name
                );
            }
            Ok(())
        }
        Command::CodeAssist {
            project,
            path,
            limit,
            mode,
            pattern_similarity,
            duplicate_similarity,
            apply_memory_suggestions,
            promote_patterns,
            apply_limit,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            let search_report = search_code_with_mode(
                &config,
                &store,
                CodeSearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit,
                    kind: None,
                    file_path: None,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let symbols = search_report.results;
            let pattern_report = build_code_pattern_report(
                &store,
                &project_id,
                &query,
                symbols,
                config.code_embed_model(),
                limit,
                pattern_similarity,
            )?;
            let duplicates = store.code_similarity_pairs(
                &project_id,
                CodeSimilarityPairOptions {
                    embedding_model: config.code_embed_model().to_string(),
                    limit,
                    kind: None,
                    file_path: None,
                    min_similarity: duplicate_similarity,
                },
            )?;
            let index_guard = path
                .as_ref()
                .map(|path| {
                    check_code_index_freshness(&store, path, Some(project_id.clone()))
                        .map(|report| code_index_guard_from_freshness(&report))
                })
                .transpose()?;
            let mut applied = Vec::new();
            if apply_memory_suggestions {
                applied.extend(apply_code_memory_suggestions(
                    &store,
                    &project_id,
                    &pattern_report.memory_suggestions,
                    apply_limit,
                )?);
            }
            if promote_patterns {
                applied.extend(apply_code_memory_suggestions(
                    &store,
                    &project_id,
                    &pattern_report.pattern_promotions,
                    apply_limit,
                )?);
            }
            let report = build_code_assist_report(CodeAssistReportInput {
                project_id: &project_id,
                query: &query,
                actual_mode: search_report.actual_mode,
                warning: search_report.warning,
                pattern_report,
                duplicate_pairs: duplicates,
                applied_memory_suggestions: applied,
                index_guard,
            });
            print_code_assist_report(&report)?;
            Ok(())
        }
        Command::CodeReviewPlan {
            project,
            limit,
            duplicate_similarity,
            query,
            files,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let duplicates = store.code_similarity_pairs(
                &project_id,
                CodeSimilarityPairOptions {
                    embedding_model: config.code_embed_model().to_string(),
                    limit,
                    kind: None,
                    file_path: None,
                    min_similarity: duplicate_similarity,
                },
            )?;
            let report = build_code_review_plan_report(
                &store,
                &project_id,
                query.as_deref().unwrap_or("changed files review"),
                files,
                duplicates,
                limit,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::DukememoryDevsystem {
            project,
            path,
            query,
            no_memory_write,
            files,
        } => {
            let project_id = resolve_project_id(project)?;
            let mut store = Store::open(&config.database_marker)?;
            let report = build_devsystem_report(
                &mut store,
                DevsystemRequest {
                    project_id,
                    query,
                    files,
                    project_path: path,
                    write_memory: !no_memory_write,
                    auto_index: true,
                    full_rebuild: false,
                    embed_symbols: false,
                    embed_symbol_limit: 200,
                    precomputed_index_run: None,
                    run_evidence: false,
                    evidence_timeout_seconds: 120,
                    max_evidence_commands: 5,
                    allowed_evidence_commands: Vec::new(),
                    code_embedding_model: Some(config.code_embed_model().to_string()),
                    duplicate_similarity: 0.92,
                    review_limit: 50,
                    policy_override: None,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::CodeEval {
            project,
            file,
            limit,
            mode,
            json: json_output,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("failed to read code eval suite {}", file.display()))?;
            let cases = if let Ok(cases) = serde_json::from_str::<Vec<CodeEvalCaseInput>>(&text) {
                cases
            } else {
                #[derive(Deserialize)]
                struct Envelope {
                    cases: Vec<CodeEvalCaseInput>,
                }
                serde_json::from_str::<Envelope>(&text)
                    .with_context(|| format!("failed to parse code eval suite {}", file.display()))?
                    .cases
            };
            let mut observed = Vec::new();
            for case in cases {
                let results = search_code(
                    &config,
                    &store,
                    CodeSearchRequest {
                        project_id: &project_id,
                        query: &case.query,
                        limit,
                        kind: None,
                        file_path: None,
                        mode,
                        allow_hybrid_fallback: true,
                    },
                )
                .await?;
                observed.push((case, results));
            }
            let report = evaluate_code_cases(&project_id, mode.as_str(), observed);
            if json_output {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                println!(
                    "code_eval: {}/{} passed, {} failed",
                    report.passed_cases, report.total_cases, report.failed_cases
                );
                for case in &report.cases {
                    println!(
                        "- {} passed={} hits={} mrr={:.3}",
                        case.name,
                        case.passed,
                        case.hits,
                        case.mrr.unwrap_or(0.0)
                    );
                }
            }
            if report.failed_cases > 0 {
                bail!("code eval failed: {} cases failed", report.failed_cases);
            }
            Ok(())
        }
        Command::CodeBrief {
            project,
            model_role,
            symbol,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let Some(symbol_data) = store.get_code_symbol(&project_id, &symbol)? else {
                bail!("indexed symbol `{symbol}` was not found in project `{project_id}`");
            };
            let callers = store.find_callers(&project_id, &symbol, 50)?;
            let callees = store.find_callees(&project_id, &symbol, 50)?;
            let model = code_model_for_role(&config, &model_role)?;
            let ollama = ollama_from_config(&config);
            let report = reason_about_symbol(
                &ollama,
                model,
                &project_id,
                &symbol_data,
                &callers,
                &callees,
            )
            .await?;
            print_code_reason_report(&report);
            Ok(())
        }
        Command::CodePlan {
            project,
            memory_limit,
            code_limit,
            model_role,
            mode,
            deterministic,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            if deterministic {
                let search_report = search_code_with_mode(
                    &config,
                    &store,
                    CodeSearchRequest {
                        project_id: &project_id,
                        query: &query,
                        limit: code_limit,
                        kind: None,
                        file_path: None,
                        mode,
                        allow_hybrid_fallback: true,
                    },
                )
                .await?;
                let symbols = search_report.results;
                let pattern_report = build_code_pattern_report(
                    &store,
                    &project_id,
                    &query,
                    symbols,
                    config.code_embed_model(),
                    code_limit,
                    0.72,
                )?;
                let duplicates = store.code_similarity_pairs(
                    &project_id,
                    CodeSimilarityPairOptions {
                        embedding_model: config.code_embed_model().to_string(),
                        limit: code_limit,
                        kind: None,
                        file_path: None,
                        min_similarity: 0.92,
                    },
                )?;
                let assist = build_code_assist_report(CodeAssistReportInput {
                    project_id: &project_id,
                    query: &query,
                    actual_mode: search_report.actual_mode,
                    warning: search_report.warning,
                    pattern_report,
                    duplicate_pairs: duplicates,
                    applied_memory_suggestions: Vec::new(),
                    index_guard: None,
                });
                let report = deterministic_reason_report("plan", &query, &assist);
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }
            let memories = search_memories(
                &config,
                &store,
                MemorySearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit: memory_limit,
                    status: StatusFilter::One(MemoryStatus::Active),
                    kind: None,
                    memory_tier: None,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let code = search_code(
                &config,
                &store,
                CodeSearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit: code_limit,
                    kind: None,
                    file_path: None,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let model = code_model_for_role(&config, &model_role)?;
            let ollama = ollama_from_config(&config);
            let report = reason_about_search(
                &ollama,
                model,
                CodeReasonTask::Plan,
                &project_id,
                &query,
                &memories,
                &code,
            )
            .await?;
            print_code_reason_report(&report);
            Ok(())
        }
        Command::CodeRisk {
            project,
            memory_limit,
            code_limit,
            model_role,
            mode,
            deterministic,
            query,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let mode = SearchMode::parse(&mode)?;
            if deterministic {
                let search_report = search_code_with_mode(
                    &config,
                    &store,
                    CodeSearchRequest {
                        project_id: &project_id,
                        query: &query,
                        limit: code_limit,
                        kind: None,
                        file_path: None,
                        mode,
                        allow_hybrid_fallback: true,
                    },
                )
                .await?;
                let symbols = search_report.results;
                let pattern_report = build_code_pattern_report(
                    &store,
                    &project_id,
                    &query,
                    symbols,
                    config.code_embed_model(),
                    code_limit,
                    0.72,
                )?;
                let duplicates = store.code_similarity_pairs(
                    &project_id,
                    CodeSimilarityPairOptions {
                        embedding_model: config.code_embed_model().to_string(),
                        limit: code_limit,
                        kind: None,
                        file_path: None,
                        min_similarity: 0.92,
                    },
                )?;
                let assist = build_code_assist_report(CodeAssistReportInput {
                    project_id: &project_id,
                    query: &query,
                    actual_mode: search_report.actual_mode,
                    warning: search_report.warning,
                    pattern_report,
                    duplicate_pairs: duplicates,
                    applied_memory_suggestions: Vec::new(),
                    index_guard: None,
                });
                let report = deterministic_reason_report("risk", &query, &assist);
                println!("{}", serde_json::to_string_pretty(&report)?);
                return Ok(());
            }
            let memories = search_memories(
                &config,
                &store,
                MemorySearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit: memory_limit,
                    status: StatusFilter::One(MemoryStatus::Active),
                    kind: None,
                    memory_tier: None,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let code = search_code(
                &config,
                &store,
                CodeSearchRequest {
                    project_id: &project_id,
                    query: &query,
                    limit: code_limit,
                    kind: None,
                    file_path: None,
                    mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let model = code_model_for_role(&config, &model_role)?;
            let ollama = ollama_from_config(&config);
            let report = reason_about_search(
                &ollama,
                model,
                CodeReasonTask::Risk,
                &project_id,
                &query,
                &memories,
                &code,
            )
            .await?;
            print_code_reason_report(&report);
            Ok(())
        }
        Command::EmbedMissing {
            project,
            limit,
            scope,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let report = embed_missing(&config, &store, &project_id, limit, &scope).await?;
            println!("project: {project_id}");
            println!("memory_model: {}", config.memory_embed_model());
            println!("code_model: {}", config.code_embed_model());
            println!("memories_embedded: {}", report.memories);
            println!("code_symbols_embedded: {}", report.code_symbols);
            println!("code_symbols_cached: {}", report.code_symbols_cached);
            println!("code_symbols_reused: {}", report.code_symbols_reused);
            println!("code_symbols_generated: {}", report.code_symbols_generated);
            Ok(())
        }
        Command::Semantic {
            project,
            action,
            query,
            body,
            memory_id,
            symbol,
            file_path,
            input,
            other_project_id,
            expected_ids,
            helpful_ids,
            unhelpful_ids,
            limit,
            status,
            kind,
            memory_tier,
            mode,
            min_similarity,
            target_memory_model,
            target_code_model,
            as_of,
            retrieval_event_id,
            outcome_kind,
            severity,
            apply,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let report = run_semantic_operation(
                &config,
                &store,
                SemanticOperationRequest {
                    project_id,
                    action,
                    query,
                    body,
                    memory_id,
                    symbol,
                    file_path,
                    project_path: None,
                    input,
                    other_project_id,
                    expected_ids,
                    helpful_ids,
                    unhelpful_ids,
                    limit,
                    status: StatusFilter::parse(Some(status.as_str()), MemoryStatus::Active)?,
                    kind,
                    memory_tier,
                    mode: RetrievalMode::parse(&mode)?,
                    min_similarity,
                    target_memory_model,
                    target_code_model,
                    as_of,
                    retrieval_event_id,
                    outcome_kind,
                    severity,
                    apply,
                },
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::DukememoryExtract {
            project,
            source,
            file,
            input,
            max_candidates,
            no_embed,
            validate,
            apply_policy,
        } => {
            let text = read_input_text(file.as_ref(), input.as_deref())?;
            let prepared = prepare_extraction_input(&text, &source);
            let project_id = match (project, prepared.project_path.as_deref()) {
                (Some(project), _) => resolve_project_id(Some(project))?,
                (None, Some(project_path)) => resolve_project_id_from_path(project_path)?,
                (None, None) => resolve_project_id(None)?,
            };
            let store = Store::open(&config.database_marker)?;
            let episode = store.add_memory_episode(
                &project_id,
                &prepared.source,
                Some("dukememory_extract input"),
                prepared.project_path.as_deref(),
                json!({
                    "source": prepared.source,
                    "project_path": prepared.project_path,
                    "text": truncate_text(&prepared.text, 50_000)
                }),
            )?;
            let ollama = ollama_from_config(&config);
            let candidates = extract_memory_candidates(
                &ollama,
                &project_id,
                &prepared.source,
                &prepared.text,
                max_candidates,
            )
            .await?;
            println!("project: {project_id}");
            println!("source: {}", prepared.source);
            if let Some(project_path) = &prepared.project_path {
                println!("project_path: {project_path}");
            }
            println!("episode_id: {}", episode.id);
            println!("candidates: {}", candidates.len());
            let mut inserted_ids = Vec::new();
            for candidate in candidates {
                let outcome = store_candidate(&store, &project_id, &prepared.source, &candidate)?;
                println!();
                if outcome.inserted {
                    println!("stored pending memory: {}", outcome.id);
                    inserted_ids.push(outcome.id.clone());
                } else {
                    println!(
                        "duplicate skipped: {}",
                        outcome.duplicate_of.as_deref().unwrap_or(&outcome.id)
                    );
                }
                println!("kind: {}", candidate.kind);
                println!("importance: {:.2}", candidate.importance);
                println!("confidence: {:.2}", candidate.confidence);
                println!("{}", candidate.body);
                if !no_embed && outcome.inserted {
                    match embed_memory(&config, &store, &project_id, &outcome.id, &candidate.body)
                        .await
                    {
                        Ok(dimensions) => println!(
                            "embedding: stored model={} dimensions={dimensions}",
                            config.memory_embed_model()
                        ),
                        Err(error) => println!("embedding: skipped ({error})"),
                    }
                }
            }
            if validate && !inserted_ids.is_empty() {
                let mut pending = Vec::new();
                for id in &inserted_ids {
                    if let Some(memory) = store.get(&project_id, id)?
                        && memory.status == MemoryStatus::Pending.as_str()
                    {
                        pending.push(memory);
                    }
                }
                let ollama = ollama_from_config(&config);
                let report =
                    validate_memories(&ollama, &config.validate_model, &project_id, &pending)
                        .await?;
                println!();
                println!("memory_policy_model: {}", report.model);
                println!("memory_policy_apply: {apply_policy}");
                println!("memory_policy_decisions: {}", report.decisions.len());
                for decision in &report.decisions {
                    if apply_policy {
                        match decision.action {
                            ValidationAction::Promote => {
                                store.promote(&project_id, &decision.id, Some(&decision.reason))?
                            }
                            ValidationAction::Archive => {
                                store.archive(&project_id, &decision.id, Some(&decision.reason))?
                            }
                            ValidationAction::Keep => {}
                        }
                    }
                    println!(
                        "- {} {} {:.2}: {}",
                        decision.id,
                        decision.action.as_str(),
                        decision.confidence,
                        decision.reason
                    );
                }
            }
            Ok(())
        }
        Command::ReadSymbol { project, symbol } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            match store.get_code_symbol(&project_id, &symbol)? {
                Some(symbol) => print_code_symbol(&symbol, true),
                None => println!("symbol `{symbol}` was not found in project `{project_id}`"),
            }
            Ok(())
        }
        Command::FindCallers {
            project,
            limit,
            symbol,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let callers = store.find_callers(&project_id, &symbol, limit)?;
            println!("project: {project_id}");
            println!("callers: {}", callers.len());
            for relation in callers {
                print_code_relation(&relation);
            }
            Ok(())
        }
        Command::FindCallees {
            project,
            limit,
            symbol,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let callees = store.find_callees(&project_id, &symbol, limit)?;
            println!("project: {project_id}");
            println!("callees: {}", callees.len());
            for relation in callees {
                print_code_relation(&relation);
            }
            Ok(())
        }
        Command::Impact {
            project,
            limit,
            symbol,
        } => {
            let project_id = resolve_project_id(project)?;
            let store = Store::open(&config.database_marker)?;
            let callers = store.find_callers(&project_id, &symbol, limit)?;
            let callees = store.find_callees(&project_id, &symbol, limit)?;
            let impacted_files =
                impacted_files_for_relations(&store, &project_id, &symbol, &callers, &callees)?;
            let affected_tests =
                store.affected_test_files(&project_id, &impacted_files, 5, limit)?;
            println!("project: {project_id}");
            println!("symbol: {symbol}");
            println!("callers: {}", callers.len());
            for relation in &callers {
                print_code_relation(relation);
            }
            println!("callees: {}", callees.len());
            for relation in &callees {
                print_code_relation(relation);
            }
            println!("impacted_files: {}", impacted_files.len());
            for file in &impacted_files {
                println!("- {file}");
            }
            println!("affected_tests: {}", affected_tests.len());
            for file in &affected_tests {
                println!("- {file}");
            }
            Ok(())
        }
        Command::Embed { role, text } => {
            let model = match role.as_str() {
                "memory" | "memory_embed" | "embed" => config.memory_embed_model(),
                "code" | "code_embed" | "fast_embed" => config.code_embed_model(),
                _ => {
                    bail!(
                        "embed can only test embedding roles; use memory or code. Use doctor or dukememory_models to inspect all roles."
                    );
                }
            };
            let ollama = ollama_from_config(&config);
            let embedding = ollama.embed_with_model(model, &text).await?;
            println!("role: {role}");
            println!("model: {model}");
            println!("dimensions: {}", embedding.len());
            Ok(())
        }
        Command::Mcp => run_stdio(config).await,
        Command::McpHttp { host, port } => run_http(config, &host, port).await,
        Command::McpSmoke => {
            let report = run_smoke(&config).await?;
            println!("mcp_smoke: ok");
            println!("project: {}", report.project_id);
            println!("database_url: {}", config.database_url);
            println!("database_marker: {}", report.database_marker.display());
            println!("tools: {}", report.tools_count);
            println!("remembered_id: {}", report.remembered_id);
            println!("search_hits: {}", report.search_hits);
            println!("context_hits: {}", report.context_hits);
            println!("cross_project_hits: {}", report.cross_project_hits);
            println!("total_memories: {}", report.total_memories);
            println!("active_memories: {}", report.active_memories);
            Ok(())
        }
        Command::McpHttpSmoke => {
            let report = run_http_smoke(&config).await?;
            println!("mcp_http_smoke: ok");
            println!("address: {}", report.address);
            println!("server_info_name: {}", report.server_info_name);
            println!("initialize_status: {}", report.initialize_status);
            println!("sse_status: {}", report.sse_status);
            println!(
                "forbidden_origin_status: {}",
                report.forbidden_origin_status
            );
            println!("wrong_method_status: {}", report.wrong_method_status);
            println!("oversized_body_status: {}", report.oversized_body_status);
            Ok(())
        }
        Command::ProductionAudit => {
            let report = run_production_audit(&config).await?;
            println!("production_audit: ok");
            println!("project: {}", report.project_id);
            println!("root: {}", report.root_path.display());
            println!("database_url: {}", config.database_url);
            println!("database_marker: {}", report.database_marker.display());
            println!("files_indexed: {}", report.files_indexed);
            println!("files_skipped: {}", report.files_skipped);
            println!("code_symbols: {}", report.code_symbols);
            println!("code_relations: {}", report.code_relations);
            println!("resolved_relations: {}", report.resolved_relations);
            println!("memory_id: {}", report.memory_id);
            println!("memory_hits: {}", report.memory_hits);
            println!("cross_project_hits: {}", report.cross_project_hits);
            println!("code_hits: {}", report.code_hits);
            println!("caller_edges: {}", report.caller_edges);
            println!("callee_edges: {}", report.callee_edges);
            println!("backup: {}", report.backup_path.display());
            println!("backup_size_bytes: {}", report.backup_size_bytes);
            println!("export: {}", report.export_path.display());
            println!("export_memories: {}", report.export_memories);
            println!("export_code_files: {}", report.export_code_files);
            println!("export_code_symbols: {}", report.export_code_symbols);
            println!("export_code_relations: {}", report.export_code_relations);
            println!("import_memories: {}", report.import_memories);
            println!("import_code_files: {}", report.import_code_files);
            println!("import_code_symbols: {}", report.import_code_symbols);
            println!("import_code_relations: {}", report.import_code_relations);
            println!(
                "maintenance_backup_size_bytes: {}",
                report.maintenance_backup_size_bytes
            );
            println!(
                "maintenance_compaction_status: {}",
                report.maintenance_compaction_status
            );
            Ok(())
        }
        Command::CodexAudit => {
            let command =
                std::env::current_exe().context("failed to resolve current executable")?;
            let script = default_hook_script_path()?;
            let report = run_codex_integration_audit(&config, &command, &script)?;
            println!("codex_audit: ok");
            println!("config: {}", report.config_path.display());
            println!("hooks: {}", report.hooks_path.display());
            println!("script: {}", report.script_path.display());
            println!("command: {}", report.command_path.display());
            println!("mcp_server_configured: {}", report.mcp_server_configured);
            println!("mcp_args_configured: {}", report.mcp_args_configured);
            println!(
                "env_keys: {}/{}",
                report.env_keys_configured, report.env_keys_expected
            );
            println!("hook_events: {}", report.hook_events_configured.join(","));
            println!("script_exists: {}", report.script_exists);
            println!("script_executable: {}", report.script_executable);
            println!("hook_dry_run_ok: {}", report.hook_dry_run_ok);
            Ok(())
        }
        Command::CodexHookAudit => {
            let command =
                std::env::current_exe().context("failed to resolve current executable")?;
            let script = default_hook_script_path()?;
            let report = run_codex_hook_payload_audit(&config, &command, &script)?;
            println!("codex_hook_audit: ok");
            println!("project: {}", report.project_id);
            println!("root: {}", report.root_path.display());
            println!("database_url: {}", config.database_url);
            println!("database_marker: {}", report.database_marker.display());
            println!("script: {}", report.script_path.display());
            println!("command: {}", report.command_path.display());
            println!("event: {}", report.event);
            println!("pending_memories: {}", report.pending_memories);
            println!("total_memories: {}", report.total_memories);
            println!("memory_embeddings: {}", report.memory_embeddings);
            println!("first_memory_id: {}", report.first_memory_id);
            if let Some(source) = report.first_memory_source {
                println!("first_memory_source: {source}");
            }
            println!("stdout_bytes: {}", report.stdout_bytes);
            println!("stderr_bytes: {}", report.stderr_bytes);
            Ok(())
        }
        Command::CodexConfig {
            install,
            force,
            config: codex_config_path,
            command,
        } => {
            let command = match command {
                Some(command) => command,
                None => std::env::current_exe().context("failed to resolve current executable")?,
            };
            let snippet = mcp_snippet(&config, &command);
            if install {
                let config_path = match codex_config_path {
                    Some(path) => path,
                    None => default_codex_config_path()?,
                };
                let result = install_mcp_config(&config_path, &snippet, force)?;
                println!("installed: {}", result.installed);
                println!("config: {}", result.config_path.display());
                if let Some(backup_path) = result.backup_path {
                    println!("backup: {}", backup_path.display());
                }
            } else {
                println!("{snippet}");
            }
            Ok(())
        }
        Command::CodexHooks {
            install,
            force,
            config: hooks_path,
            script,
            events,
        } => {
            let script = match script {
                Some(script) => script,
                None => default_hook_script_path()?,
            };
            let snippet = hooks_snippet(&script, &events)?;
            if install {
                let hooks_path = match hooks_path {
                    Some(path) => path,
                    None => default_codex_hooks_path()?,
                };
                let result = install_hooks(&hooks_path, &script, &events, force)?;
                println!("hooks: {}", result.hooks_path.display());
                println!("events: {}", result.installed_events.join(","));
                if let Some(backup_path) = result.backup_path {
                    println!("backup: {}", backup_path.display());
                }
            } else {
                println!("{snippet}");
            }
            Ok(())
        }
    }
}

#[derive(Debug, Deserialize)]
struct EvalSuite {
    name: Option<String>,
    cases: Vec<EvalCase>,
}

#[derive(Debug, Deserialize)]
struct EvalCase {
    name: Option<String>,
    #[serde(default = "default_eval_target")]
    target: String,
    project_id: Option<String>,
    query: String,
    #[serde(default)]
    semantic_action: Option<String>,
    #[serde(default)]
    as_of: Option<String>,
    #[serde(default)]
    decisions: Vec<ReviewDecisionInput>,
    #[serde(default)]
    expected_contains: Vec<String>,
    #[serde(default)]
    forbidden_contains: Vec<String>,
    #[serde(default)]
    expected_ids: Vec<String>,
    #[serde(default)]
    forbidden_ids: Vec<String>,
    #[serde(default)]
    min_results: Option<usize>,
    #[serde(default)]
    max_latency_ms: Option<u64>,
    #[serde(default)]
    max_estimated_tokens: Option<usize>,
}

fn default_eval_target() -> String {
    "memory".to_string()
}

#[derive(Debug, Serialize)]
struct EvalReport {
    run_id: String,
    suite_name: Option<String>,
    suite_hash: Option<String>,
    previous_run: Option<crate::store::EvalRunSummary>,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    cases: Vec<EvalCaseReport>,
}

#[derive(Debug, Serialize)]
struct EvalCaseReport {
    name: String,
    target: String,
    project_id: String,
    query: String,
    hits: usize,
    passed: bool,
    expected_ids: Vec<String>,
    forbidden_ids: Vec<String>,
    matched_expected_ids: Vec<String>,
    missing_expected_ids: Vec<String>,
    matched_forbidden_ids: Vec<String>,
    recall_at_k: Option<f64>,
    precision_at_k: Option<f64>,
    mrr: Option<f64>,
    ndcg_at_k: Option<f64>,
    latency_ms: u64,
    max_latency_ms: Option<u64>,
    latency_ok: bool,
    estimated_tokens: usize,
    max_estimated_tokens: Option<usize>,
    token_budget_ok: bool,
    missing_expected: Vec<String>,
    forbidden_found: Vec<String>,
    top_ids: Vec<String>,
}

struct EvalObserved {
    hits: usize,
    haystack: String,
    top_ids: Vec<String>,
    estimated_tokens: usize,
}

fn mean_reciprocal_rank(top_ids: &[String], expected_ids: &[String]) -> Option<f64> {
    if expected_ids.is_empty() {
        return None;
    }
    let expected = expected_ids.iter().collect::<HashSet<_>>();
    top_ids
        .iter()
        .position(|id| expected.contains(id))
        .map(|index| 1.0 / (index as f64 + 1.0))
        .or(Some(0.0))
}

fn ndcg_at_k(top_ids: &[String], expected_ids: &[String]) -> Option<f64> {
    if expected_ids.is_empty() {
        return None;
    }
    let expected = expected_ids.iter().collect::<HashSet<_>>();
    let dcg = top_ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| {
            if expected.contains(id) {
                Some(1.0 / ((index as f64) + 2.0).log2())
            } else {
                None
            }
        })
        .sum::<f64>();
    let ideal_hits = expected_ids.len().min(top_ids.len());
    if ideal_hits == 0 {
        return Some(0.0);
    }
    let idcg = (0..ideal_hits)
        .map(|index| 1.0 / ((index as f64) + 2.0).log2())
        .sum::<f64>();
    Some(if idcg > 0.0 { dcg / idcg } else { 0.0 })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewDecisionInput {
    id: String,
    action: String,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReviewApplyReport {
    project_id: String,
    dry_run: bool,
    promoted: usize,
    archived: usize,
    kept: usize,
    decisions: Vec<ReviewDecisionInput>,
}

fn load_review_decisions(
    file: Option<&Path>,
    inline_decisions: &[String],
) -> Result<Vec<ReviewDecisionInput>> {
    let mut decisions = Vec::new();
    if let Some(file) = file {
        let text = std::fs::read_to_string(file)
            .with_context(|| format!("failed to read review decision file {}", file.display()))?;
        let value = serde_json::from_str::<Value>(&text)
            .with_context(|| format!("failed to parse review decision file {}", file.display()))?;
        let parsed = if value.is_array() {
            serde_json::from_value::<Vec<ReviewDecisionInput>>(value)?
        } else {
            serde_json::from_value::<ReviewDecisionEnvelope>(value)?.decisions
        };
        decisions.extend(parsed);
    }
    for raw in inline_decisions {
        decisions.push(parse_review_decision(raw)?);
    }
    if decisions.is_empty() {
        bail!("review-apply requires --decision or --file");
    }
    Ok(decisions)
}

#[derive(Debug, Deserialize)]
struct ReviewDecisionEnvelope {
    decisions: Vec<ReviewDecisionInput>,
}

fn parse_review_decision(raw: &str) -> Result<ReviewDecisionInput> {
    let mut parts = raw.splitn(3, ':');
    let action = parts
        .next()
        .filter(|value| !value.trim().is_empty())
        .context("review decision is missing action")?;
    let id = parts
        .next()
        .filter(|value| !value.trim().is_empty())
        .context("review decision is missing id")?;
    let reason = parts
        .next()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string);
    Ok(ReviewDecisionInput {
        id: id.to_string(),
        action: action.to_string(),
        reason,
    })
}

fn apply_review_decisions(
    store: &Store,
    project_id: &str,
    decisions: &[ReviewDecisionInput],
    dry_run: bool,
) -> Result<ReviewApplyReport> {
    let mut report = ReviewApplyReport {
        project_id: project_id.to_string(),
        dry_run,
        promoted: 0,
        archived: 0,
        kept: 0,
        decisions: decisions.to_vec(),
    };
    for decision in decisions {
        match decision.action.as_str() {
            "promote" => {
                if !dry_run {
                    store.promote(project_id, &decision.id, decision.reason.as_deref())?;
                    store.record_audit_event(
                        project_id,
                        "cli",
                        "review_promote",
                        "memory",
                        Some(&decision.id),
                        json!({"reason": decision.reason}),
                    )?;
                }
                report.promoted += 1;
            }
            "archive" => {
                if !dry_run {
                    store.archive(project_id, &decision.id, decision.reason.as_deref())?;
                    store.record_audit_event(
                        project_id,
                        "cli",
                        "review_archive",
                        "memory",
                        Some(&decision.id),
                        json!({"reason": decision.reason}),
                    )?;
                }
                report.archived += 1;
            }
            "keep" => {
                if !dry_run {
                    store.record_audit_event(
                        project_id,
                        "cli",
                        "review_keep",
                        "memory",
                        Some(&decision.id),
                        json!({"reason": decision.reason}),
                    )?;
                }
                report.kept += 1;
            }
            other => bail!("invalid review action `{other}`; use promote, archive, or keep"),
        }
    }
    Ok(report)
}

struct EvalSuiteRun<'a> {
    default_project_id: &'a str,
    suite: EvalSuite,
    suite_name: Option<String>,
    suite_hash: Option<String>,
    mode: SearchMode,
    limit: usize,
}

async fn run_eval_suite(
    config: &Config,
    store: &Store,
    run: EvalSuiteRun<'_>,
) -> Result<EvalReport> {
    let mut cases = Vec::new();
    for (index, case) in run.suite.cases.iter().enumerate() {
        let project_id = case
            .project_id
            .clone()
            .unwrap_or_else(|| run.default_project_id.to_string());
        let target = case.target.trim().to_ascii_lowercase();
        let started_at = Instant::now();
        let observed = observe_eval_case(config, store, &project_id, case, &target, &run).await?;
        let latency_ms = started_at.elapsed().as_millis() as u64;
        let top_id_set = observed.top_ids.iter().collect::<HashSet<_>>();
        let matched_expected_ids = case
            .expected_ids
            .iter()
            .filter(|id| top_id_set.contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let matched_forbidden_ids = case
            .forbidden_ids
            .iter()
            .filter(|id| top_id_set.contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let missing_expected_ids = case
            .expected_ids
            .iter()
            .filter(|id| !top_id_set.contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let recall_at_k = if case.expected_ids.is_empty() {
            None
        } else {
            Some(matched_expected_ids.len() as f64 / case.expected_ids.len() as f64)
        };
        let precision_at_k = if case.expected_ids.is_empty() {
            None
        } else if observed.top_ids.is_empty() {
            Some(0.0)
        } else {
            Some(matched_expected_ids.len() as f64 / observed.top_ids.len() as f64)
        };
        let mrr = mean_reciprocal_rank(&observed.top_ids, &case.expected_ids);
        let ndcg_at_k = ndcg_at_k(&observed.top_ids, &case.expected_ids);
        let haystack = observed.haystack.to_ascii_lowercase();
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
        let min_results_ok = case
            .min_results
            .map(|min_results| observed.hits >= min_results)
            .unwrap_or(true);
        let latency_ok = case
            .max_latency_ms
            .map(|max_latency_ms| latency_ms <= max_latency_ms)
            .unwrap_or(true);
        let token_budget_ok = case
            .max_estimated_tokens
            .map(|max_tokens| observed.estimated_tokens <= max_tokens)
            .unwrap_or(true);
        let passed = missing_expected.is_empty()
            && missing_expected_ids.is_empty()
            && matched_forbidden_ids.is_empty()
            && forbidden_found.is_empty()
            && min_results_ok
            && latency_ok
            && token_budget_ok;
        cases.push(EvalCaseReport {
            name: case
                .name
                .clone()
                .unwrap_or_else(|| format!("case-{}", index + 1)),
            target,
            project_id,
            query: case.query.clone(),
            hits: observed.hits,
            passed,
            expected_ids: case.expected_ids.clone(),
            forbidden_ids: case.forbidden_ids.clone(),
            matched_expected_ids,
            missing_expected_ids,
            matched_forbidden_ids,
            recall_at_k,
            precision_at_k,
            mrr,
            ndcg_at_k,
            latency_ms,
            max_latency_ms: case.max_latency_ms,
            latency_ok,
            estimated_tokens: observed.estimated_tokens,
            max_estimated_tokens: case.max_estimated_tokens,
            token_budget_ok,
            missing_expected,
            forbidden_found,
            top_ids: observed.top_ids,
        });
    }
    let total_cases = cases.len();
    let passed_cases = cases.iter().filter(|case| case.passed).count();
    let failed_cases = total_cases.saturating_sub(passed_cases);
    let previous_run = store.latest_eval_run_summary(
        run.default_project_id,
        run.suite_name.as_deref(),
        run.suite_hash.as_deref(),
        run.mode.as_str(),
        None,
    )?;
    let detail = json!({
        "default_project_id": run.default_project_id,
        "suite_name": &run.suite_name,
        "suite_hash": &run.suite_hash,
        "limit": run.limit,
        "mode": run.mode.as_str(),
        "cases": &cases
    });
    let run_id = store.record_eval_run(EvalRunRecord {
        project_id: run.default_project_id,
        suite_name: run.suite_name.as_deref(),
        suite_hash: run.suite_hash.as_deref(),
        mode: run.mode.as_str(),
        total_cases,
        passed_cases,
        failed_cases,
        detail,
    })?;
    store.record_audit_event(
        run.default_project_id,
        "cli",
        "eval_run",
        "eval_run",
        Some(&run_id),
        json!({
            "suite_name": &run.suite_name,
            "suite_hash": &run.suite_hash,
            "mode": run.mode.as_str(),
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases
        }),
    )?;
    Ok(EvalReport {
        run_id,
        suite_name: run.suite_name,
        suite_hash: run.suite_hash,
        previous_run,
        total_cases,
        passed_cases,
        failed_cases,
        cases,
    })
}

async fn observe_eval_case(
    config: &Config,
    store: &Store,
    project_id: &str,
    case: &EvalCase,
    target: &str,
    run: &EvalSuiteRun<'_>,
) -> Result<EvalObserved> {
    match target {
        "memory" => {
            let results = search_memories(
                config,
                store,
                MemorySearchRequest {
                    project_id,
                    query: &case.query,
                    limit: run.limit,
                    status: StatusFilter::One(MemoryStatus::Active),
                    kind: None,
                    memory_tier: None,
                    mode: run.mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let haystack = results
                .iter()
                .map(memory_eval_text)
                .collect::<Vec<_>>()
                .join("\n");
            Ok(EvalObserved {
                hits: results.len(),
                top_ids: results.iter().map(|memory| memory.id.clone()).collect(),
                estimated_tokens: estimate_context_tokens(&[haystack.chars().count()]),
                haystack,
            })
        }
        "code" => {
            let results = search_code(
                config,
                store,
                CodeSearchRequest {
                    project_id,
                    query: &case.query,
                    limit: run.limit,
                    kind: None,
                    file_path: None,
                    mode: run.mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let haystack = results
                .iter()
                .map(code_eval_text)
                .collect::<Vec<_>>()
                .join("\n");
            Ok(EvalObserved {
                hits: results.len(),
                top_ids: results
                    .iter()
                    .map(|result| result.symbol.id.clone())
                    .collect(),
                estimated_tokens: estimate_context_tokens(&[haystack.chars().count()]),
                haystack,
            })
        }
        "context" => {
            let memories = search_memories(
                config,
                store,
                MemorySearchRequest {
                    project_id,
                    query: &case.query,
                    limit: run.limit,
                    status: StatusFilter::One(MemoryStatus::Active),
                    kind: None,
                    memory_tier: None,
                    mode: run.mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let code = search_code(
                config,
                store,
                CodeSearchRequest {
                    project_id,
                    query: &case.query,
                    limit: run.limit,
                    kind: None,
                    file_path: None,
                    mode: run.mode,
                    allow_hybrid_fallback: true,
                },
            )
            .await?;
            let top_ids = memories
                .iter()
                .map(|memory| memory.id.clone())
                .chain(code.iter().map(|result| result.symbol.id.clone()))
                .collect::<Vec<_>>();
            let haystack = memories
                .iter()
                .map(memory_eval_text)
                .chain(code.iter().map(code_eval_text))
                .collect::<Vec<_>>()
                .join("\n");
            Ok(EvalObserved {
                hits: top_ids.len(),
                estimated_tokens: estimate_context_tokens(&[haystack.chars().count()]),
                haystack,
                top_ids,
            })
        }
        "review_apply" => {
            if case.decisions.is_empty() {
                bail!("review_apply eval case requires at least one decision");
            }
            let report = apply_review_decisions(store, project_id, &case.decisions, true)?;
            let haystack = serde_json::to_string(&report)?;
            Ok(EvalObserved {
                hits: report.decisions.len(),
                estimated_tokens: estimate_context_tokens(&[haystack.chars().count()]),
                haystack,
                top_ids: report
                    .decisions
                    .into_iter()
                    .map(|decision| decision.id)
                    .collect(),
            })
        }
        "semantic" => {
            let action = case
                .semantic_action
                .clone()
                .unwrap_or_else(|| "retrieval_quality".to_string());
            let report = run_semantic_operation(
                config,
                store,
                SemanticOperationRequest {
                    project_id: project_id.to_string(),
                    action,
                    query: Some(case.query.clone()),
                    body: None,
                    memory_id: None,
                    symbol: None,
                    file_path: None,
                    project_path: None,
                    input: None,
                    other_project_id: None,
                    expected_ids: case.expected_ids.clone(),
                    helpful_ids: Vec::new(),
                    unhelpful_ids: Vec::new(),
                    limit: run.limit,
                    status: StatusFilter::One(MemoryStatus::Active),
                    kind: None,
                    memory_tier: None,
                    mode: run.mode,
                    min_similarity: 0.0,
                    target_memory_model: None,
                    target_code_model: None,
                    as_of: case.as_of.clone(),
                    retrieval_event_id: None,
                    outcome_kind: None,
                    severity: None,
                    apply: false,
                },
            )
            .await?;
            let haystack = serde_json::to_string(&report)?;
            let top_ids = semantic_eval_top_ids(&report);
            Ok(EvalObserved {
                hits: top_ids.len().max(semantic_eval_hit_count(&report)),
                estimated_tokens: estimate_context_tokens(&[haystack.chars().count()]),
                haystack,
                top_ids,
            })
        }
        other => bail!(
            "invalid eval target `{other}`; use memory, code, context, review_apply, or semantic"
        ),
    }
}

fn semantic_eval_hit_count(value: &Value) -> usize {
    [
        "/top_memory_ids",
        "/baseline/memory_ids",
        "/memories",
        "/leave_one_out",
        "/hard_negatives/candidates",
        "/recommended_policy/reasons",
        "/retrieval_events",
    ]
    .iter()
    .filter_map(|path| value.pointer(path).and_then(Value::as_array).map(Vec::len))
    .max()
    .unwrap_or(0)
}

fn semantic_eval_top_ids(value: &Value) -> Vec<String> {
    let mut ids = Vec::new();
    collect_string_array_pointer(value, "/top_memory_ids", &mut ids);
    collect_string_array_pointer(value, "/baseline/memory_ids", &mut ids);
    collect_string_array_pointer(value, "/suggested_eval_case/expected_ids", &mut ids);
    collect_object_ids_pointer(value, "/memories", &mut ids);
    collect_object_ids_pointer(value, "/hard_negatives/candidates", &mut ids);
    collect_object_ids_pointer(value, "/selected_memories", &mut ids);
    ids.sort();
    ids.dedup();
    ids
}

fn collect_string_array_pointer(value: &Value, pointer: &str, ids: &mut Vec<String>) {
    if let Some(values) = value.pointer(pointer).and_then(Value::as_array) {
        ids.extend(values.iter().filter_map(Value::as_str).map(str::to_string));
    }
}

fn collect_object_ids_pointer(value: &Value, pointer: &str, ids: &mut Vec<String>) {
    if let Some(values) = value.pointer(pointer).and_then(Value::as_array) {
        ids.extend(
            values
                .iter()
                .filter_map(|item| item.get("id").and_then(Value::as_str))
                .map(str::to_string),
        );
    }
}

fn memory_eval_text(memory: &Memory) -> String {
    format!(
        "{}\n{}\n{}\n{}",
        memory.kind,
        memory.scope,
        memory.body,
        memory.tags.join(" ")
    )
}

fn code_eval_text(result: &CodeSearchResult) -> String {
    let symbol = &result.symbol;
    format!(
        "{}\n{}\n{}\n{}\n{}",
        symbol.language, symbol.kind, symbol.name, symbol.signature, symbol.body
    )
}

fn store_candidate(
    store: &Store,
    project_id: &str,
    source: &str,
    candidate: &MemoryCandidate,
) -> Result<RememberOutcome> {
    store.remember_deduplicated(
        project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: candidate.kind.clone(),
            body: candidate.body.clone(),
            tags: candidate.tags.clone(),
            source: Some(source.to_string()),
            status: MemoryStatus::Pending,
            importance: candidate.importance,
            confidence: candidate.confidence,
            status_reason: candidate.reason.clone(),
            allow_sensitive: false,
        },
    )
}

async fn apply_compaction(
    config: &Config,
    store: &Store,
    project_id: &str,
    proposal: &CompactionProposal,
    embed: bool,
) -> Result<RememberOutcome> {
    let outcome = store.remember_deduplicated(
        project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: proposal.summary_kind.clone(),
            body: proposal.summary_body.clone(),
            tags: proposal.tags.clone(),
            source: Some("dukememory_compact".to_string()),
            status: MemoryStatus::Active,
            importance: proposal.importance,
            confidence: proposal.confidence,
            status_reason: Some(proposal.reason.clone()),
            allow_sensitive: false,
        },
    )?;
    let summary_id = outcome
        .duplicate_of
        .as_deref()
        .unwrap_or(outcome.id.as_str())
        .to_string();
    for source_id in &proposal.source_ids {
        store.archive(
            project_id,
            source_id,
            Some(&format!("compacted into {summary_id}")),
        )?;
    }
    if embed && outcome.inserted {
        match embed_memory(
            config,
            store,
            project_id,
            &outcome.id,
            &proposal.summary_body,
        )
        .await
        {
            Ok(dimensions) => println!(
                "summary_embedding: stored model={} dimensions={dimensions}",
                config.memory_embed_model()
            ),
            Err(error) => println!("summary_embedding: skipped ({error})"),
        }
    }
    Ok(outcome)
}

fn read_input_text(file: Option<&PathBuf>, input: Option<&str>) -> Result<String> {
    if let Some(input) = input {
        return Ok(input.to_string());
    }
    if let Some(file) = file {
        return std::fs::read_to_string(file)
            .with_context(|| format!("failed to read input file {}", file.display()));
    }
    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .context("failed to read stdin")?;
    Ok(text)
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ReadinessState {
    #[serde(rename = "ok")]
    Ok,
    #[serde(rename = "fail")]
    Fail,
    #[serde(rename = "skip")]
    Skip,
}

impl ReadinessState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Fail => "fail",
            Self::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ReadinessCheck {
    name: String,
    #[serde(rename = "status")]
    state: ReadinessState,
    detail: String,
}

impl ReadinessCheck {
    fn ok(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: ReadinessState::Ok,
            detail: detail.into(),
        }
    }

    fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: ReadinessState::Fail,
            detail: detail.into(),
        }
    }

    fn skip(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: ReadinessState::Skip,
            detail: detail.into(),
        }
    }
}

async fn doctor(
    config: Config,
    project: Option<String>,
    deep: bool,
    json_output: bool,
    skip_eval: bool,
    skip_rust_analyzer_diagnostics: bool,
    skip_launchd: bool,
) -> Result<()> {
    let store = Store::open(&config.database_marker)?;
    let schema_version = store.schema_version()?;
    let integrity_check = store.integrity_check()?;
    if !json_output {
        println!("database_url: {}", config.database_url);
        println!("database_marker: {}", config.database_marker.display());
        println!("ollama_base_url: {}", config.ollama_base_url);
        println!("schema_version: {}", schema_version);
        println!("integrity_check: {}", integrity_check);
    }

    let ollama = ollama_from_config(&config);
    let tags = ollama.tags().await?;
    let installed = tags
        .models
        .iter()
        .map(|model| model.name.clone())
        .collect::<Vec<_>>();
    let mut missing_roles = Vec::new();
    let mut model_roles = Vec::new();
    if !json_output {
        println!();
        println!("configured model roles:");
    }
    for (role, model) in config.model_roles() {
        let available = installed
            .iter()
            .any(|installed| model_name_matches(model, installed.as_str()));
        if !available {
            missing_roles.push(format!("{role}: {model}"));
        }
        model_roles.push(json!({
            "role": role,
            "model": model,
            "available": available
        }));
        if !json_output {
            println!(
                "- {role}: {model} ({})",
                if available { "available" } else { "missing" }
            );
        }
    }
    if !json_output {
        println!();
        println!("ollama models:");
    }
    let ollama_models = tags.models;
    for model in &ollama_models {
        let caps = model.capabilities.as_deref().unwrap_or(&[]).join(",");
        let details = model.details.as_ref();
        let context = details
            .and_then(|details| details.context_length)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let embed = details
            .and_then(|details| details.embedding_length)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let params = details
            .and_then(|details| details.parameter_size.clone())
            .unwrap_or_else(|| "-".to_string());
        if !json_output {
            println!(
                "- {} params={} context={} embedding={} caps={}",
                model.name, params, context, embed, caps
            );
        }
    }

    let project_id = resolve_project_id(project)?;
    let rust_analyzer = rust_analyzer_status();
    if !json_output {
        println!();
        println!("current_project_id: {project_id}");
        println!("memory embedding model: {}", config.memory_embed_model());
        println!("code embedding model: {}", config.code_embed_model());
        println!("extraction model: {}", config.extract_model());
        println!("rust_analyzer: {}", rust_analyzer);
    }

    if deep {
        let status = store.status(&project_id)?;
        let code_status = store.code_status(&project_id)?;
        let mut checks = Vec::new();
        checks.push(model_role_readiness_check(&missing_roles));
        checks.push(match run_smoke(&config).await {
            Ok(report) => ReadinessCheck::ok(
                "mcp_smoke",
                format!(
                    "{} tools, search_hits={}, context_hits={}, cross_project_hits={}",
                    report.tools_count,
                    report.search_hits,
                    report.context_hits,
                    report.cross_project_hits
                ),
            ),
            Err(error) => ReadinessCheck::fail("mcp_smoke", error.to_string()),
        });
        checks.push(match run_production_audit(&config).await {
            Ok(report) => ReadinessCheck::ok(
                "production_audit",
                format!(
                    "files={}, symbols={}, relations={}, memory_hits={}, cross_project_hits={}, import_memories={}",
                    report.files_indexed,
                    report.code_symbols,
                    report.code_relations,
                    report.memory_hits,
                    report.cross_project_hits,
                    report.import_memories
                ),
            ),
            Err(error) => ReadinessCheck::fail("production_audit", error.to_string()),
        });
        checks.push(
            match (
                std::env::current_exe().context("failed to resolve current executable"),
                default_hook_script_path(),
            ) {
                (Ok(command), Ok(script)) => {
                    match run_codex_integration_audit(&config, &command, &script) {
                        Ok(report) => ReadinessCheck::ok(
                            "codex_integration",
                            format!(
                                "mcp configured, env={}/{}, hooks={}, hook dry-run ok",
                                report.env_keys_configured,
                                report.env_keys_expected,
                                report.hook_events_configured.join(",")
                            ),
                        ),
                        Err(error) => ReadinessCheck::fail("codex_integration", error.to_string()),
                    }
                }
                (Err(error), _) | (_, Err(error)) => {
                    ReadinessCheck::fail("codex_integration", error.to_string())
                }
            },
        );
        checks.extend(embedding_readiness_checks(
            status.total_memories,
            status.memory_embeddings,
            code_status.symbols,
            code_status.symbol_embeddings,
        ));
        checks.push(code_relation_readiness_check(
            code_status.relations,
            code_status.resolved_relations,
        ));
        checks.push(code_index_freshness_readiness_check(
            &check_code_index_freshness(
                &store,
                &std::env::current_dir()?,
                Some(project_id.clone()),
            )?,
        ));
        checks.push(ra_reference_readiness_check(
            code_status.symbols,
            code_status.ra_references,
        ));
        checks.push(ra_call_readiness_check(
            code_status.symbols,
            code_status.ra_references,
            code_status.ra_calls,
        ));
        if skip_eval {
            checks.push(ReadinessCheck::skip("eval", "skipped by --skip-eval"));
        } else {
            checks.push(eval_readiness_check(&config, &store, &project_id).await);
        }
        if skip_rust_analyzer_diagnostics {
            checks.push(ReadinessCheck::skip(
                "rust_analyzer_diagnostics",
                "skipped by --skip-rust-analyzer-diagnostics",
            ));
        } else {
            checks.push(rust_analyzer_diagnostics_check(&std::env::current_dir()?));
        }
        if skip_launchd {
            checks.push(ReadinessCheck::skip("launchd", "skipped by --skip-launchd"));
        } else {
            checks.push(launchd_readiness_check(&project_id));
        }

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "database_url": config.database_url,
                    "database_marker": config.database_marker,
                    "ollama_base_url": config.ollama_base_url,
                    "schema_version": schema_version,
                    "integrity_check": integrity_check,
                    "project_id": project_id,
                    "memory_embedding_model": config.memory_embed_model(),
                    "code_embedding_model": config.code_embed_model(),
                    "extraction_model": config.extract_model(),
                    "rust_analyzer": rust_analyzer,
                    "model_roles": model_roles,
                    "ollama_models": ollama_models,
                    "deep": true,
                    "readiness": checks
                }))?
            );
        } else {
            println!();
            println!("deep readiness:");
            for check in &checks {
                println!(
                    "- {}: {} ({})",
                    check.name,
                    check.state.as_str(),
                    check.detail
                );
            }
        }
        let failures = checks
            .iter()
            .filter(|check| check.state == ReadinessState::Fail)
            .map(|check| format!("{}: {}", check.name, check.detail))
            .collect::<Vec<_>>();
        if !failures.is_empty() {
            bail!("doctor deep failed: {}", failures.join("; "));
        }
        if !json_output {
            println!("readiness: ok");
        }
    } else if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "database_url": config.database_url,
                "database_marker": config.database_marker,
                "ollama_base_url": config.ollama_base_url,
                "schema_version": schema_version,
                "integrity_check": integrity_check,
                "project_id": project_id,
                "memory_embedding_model": config.memory_embed_model(),
                "code_embedding_model": config.code_embed_model(),
                "extraction_model": config.extract_model(),
                "rust_analyzer": rust_analyzer,
                "model_roles": model_roles,
                "ollama_models": ollama_models,
                "deep": false
            }))?
        );
    }

    Ok(())
}

fn model_role_readiness_check(missing_roles: &[String]) -> ReadinessCheck {
    if missing_roles.is_empty() {
        ReadinessCheck::ok("model_roles", "all configured roles are available")
    } else {
        ReadinessCheck::fail(
            "model_roles",
            format!("missing {}", missing_roles.join(", ")),
        )
    }
}

fn embedding_readiness_checks(
    total_memories: u64,
    memory_embeddings: u64,
    code_symbols: u64,
    code_symbol_embeddings: u64,
) -> Vec<ReadinessCheck> {
    let memory_check = if memory_embeddings >= total_memories {
        ReadinessCheck::ok(
            "memory_embeddings",
            format!("{memory_embeddings}/{total_memories} memories embedded"),
        )
    } else {
        ReadinessCheck::fail(
            "memory_embeddings",
            format!("{memory_embeddings}/{total_memories} memories embedded"),
        )
    };

    let code_check = if code_symbols == 0 {
        ReadinessCheck::skip("code_symbol_embeddings", "no indexed code symbols")
    } else if code_symbol_embeddings >= code_symbols {
        ReadinessCheck::ok(
            "code_symbol_embeddings",
            format!("{code_symbol_embeddings}/{code_symbols} symbols embedded"),
        )
    } else {
        ReadinessCheck::fail(
            "code_symbol_embeddings",
            format!("{code_symbol_embeddings}/{code_symbols} symbols embedded"),
        )
    };

    vec![memory_check, code_check]
}

fn code_relation_readiness_check(relations: u64, resolved_relations: u64) -> ReadinessCheck {
    if relations == 0 {
        ReadinessCheck::skip("code_relation_targets", "no indexed code relations")
    } else if resolved_relations == 0 {
        ReadinessCheck::fail(
            "code_relation_targets",
            format!("{resolved_relations}/{relations} relations resolved"),
        )
    } else {
        ReadinessCheck::ok(
            "code_relation_targets",
            format!("{resolved_relations}/{relations} relations resolved"),
        )
    }
}

fn code_index_freshness_readiness_check(report: &CodeFreshnessReport) -> ReadinessCheck {
    if report.files_seen == 0 && report.indexed_files == 0 {
        return ReadinessCheck::skip("code_index_freshness", "no indexable code files");
    }
    if report.is_fresh() {
        return ReadinessCheck::ok(
            "code_index_freshness",
            format!(
                "{} indexed files match disk for {}",
                report.indexed_files,
                report.root_path.display()
            ),
        );
    }

    ReadinessCheck::fail(
        "code_index_freshness",
        format!(
            "stale={}, missing={}, deleted={} for project {}; run code-index",
            summarize_paths(&report.stale_files),
            summarize_paths(&report.missing_files),
            summarize_paths(&report.deleted_files),
            report.project_id
        ),
    )
}

fn format_cli_context_plan(plan: &crate::context_plan::ContextPlan) -> String {
    format!(
        "\nCONTEXT ACCESS PLAN\n- task_type: {}\n- token_budget: {}\n- limits: memory={} core={} code={} graph={} code_memories={}\n- sources: memories={} graph={} code_index={} code_neighborhood={} code_memories={} eval_history={}\n",
        plan.task_type,
        plan.budget_plan.effective_token_budget,
        plan.memory_limit,
        plan.core_memory_limit,
        plan.code_limit,
        plan.graph_limit,
        plan.code_memory_limit,
        plan.source_plan.memories,
        plan.source_plan.memory_graph,
        plan.source_plan.code_index,
        plan.source_plan.code_neighborhood,
        plan.source_plan.code_memories,
        plan.source_plan.eval_history
    )
}

fn summarize_paths(paths: &[String]) -> String {
    if paths.is_empty() {
        return "0".to_string();
    }
    let mut summary = paths.iter().take(3).cloned().collect::<Vec<_>>().join(",");
    if paths.len() > 3 {
        summary.push_str(&format!("+{}", paths.len() - 3));
    }
    summary
}

fn default_hook_script_path() -> Result<PathBuf> {
    Ok(std::env::current_dir()
        .context("failed to resolve current directory")?
        .join("scripts")
        .join("dukememory_codex_hook.sh"))
}

fn ra_reference_readiness_check(code_symbols: u64, ra_references: u64) -> ReadinessCheck {
    if code_symbols == 0 {
        ReadinessCheck::skip("ra_references", "no indexed code symbols")
    } else if ra_references == 0 {
        ReadinessCheck::fail(
            "ra_references",
            "run code-lsif-index or dukememory_code_lsif_index",
        )
    } else {
        ReadinessCheck::ok(
            "ra_references",
            format!("{ra_references} rust-analyzer references imported"),
        )
    }
}

fn ra_call_readiness_check(code_symbols: u64, ra_references: u64, ra_calls: u64) -> ReadinessCheck {
    if code_symbols == 0 {
        ReadinessCheck::skip("ra_calls", "no indexed code symbols")
    } else if ra_references == 0 {
        ReadinessCheck::skip("ra_calls", "no rust-analyzer references imported")
    } else if ra_calls == 0 {
        ReadinessCheck::fail("ra_calls", "no rust-analyzer call edges imported")
    } else {
        ReadinessCheck::ok(
            "ra_calls",
            format!("{ra_calls} rust-analyzer call edges imported"),
        )
    }
}

fn rust_analyzer_status() -> String {
    match ProcessCommand::new("rust-analyzer")
        .arg("--version")
        .output()
    {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if version.is_empty() {
                "available".to_string()
            } else {
                version
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if stderr.is_empty() {
                format!("unavailable ({})", output.status)
            } else {
                format!("unavailable ({stderr})")
            }
        }
        Err(error) => format!("missing ({error})"),
    }
}

async fn eval_readiness_check(
    config: &Config,
    _store: &Store,
    _project_id: &str,
) -> ReadinessCheck {
    let token = uuid::Uuid::now_v7().to_string().replace('-', "");
    let project_id = format!("doctor-eval-{token}");
    let root = std::env::temp_dir().join(format!("dukememory-doctor-eval-{token}"));
    let marker = root.join("schema.marker");
    let result = async {
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create {}", root.display()))?;
        let store = Store::open(&marker)?;
        store.remember(
            &project_id,
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "project_rule".to_string(),
                body: "Dukememory doctor eval sentinel requires project scoped keyword retrieval."
                    .to_string(),
                tags: vec!["doctor".to_string(), "eval".to_string()],
                source: Some("doctor_eval".to_string()),
                status: MemoryStatus::Active,
                importance: 0.5,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;
        let suite = EvalSuite {
            name: Some("doctor-readiness".to_string()),
            cases: vec![EvalCase {
                name: Some("sentinel-keyword".to_string()),
                target: default_eval_target(),
                project_id: None,
                query: "doctor eval sentinel keyword retrieval".to_string(),
                semantic_action: None,
                as_of: None,
                decisions: Vec::new(),
                expected_contains: vec!["project scoped keyword retrieval".to_string()],
                forbidden_contains: vec!["unrelated project".to_string()],
                expected_ids: Vec::new(),
                forbidden_ids: Vec::new(),
                min_results: Some(1),
                max_latency_ms: None,
                max_estimated_tokens: None,
            }],
        };
        run_eval_suite(
            config,
            &store,
            EvalSuiteRun {
                default_project_id: &project_id,
                suite,
                suite_name: Some("doctor-readiness".to_string()),
                suite_hash: Some("built-in".to_string()),
                mode: SearchMode::Keyword,
                limit: 5,
            },
        )
        .await
    }
    .await;
    match result {
        Ok(report) if report.failed_cases == 0 => ReadinessCheck::ok(
            "eval",
            format!(
                "built-in eval passed {}/{} cases",
                report.passed_cases, report.total_cases
            ),
        ),
        Ok(report) => ReadinessCheck::fail(
            "eval",
            format!(
                "built-in eval failed {}/{} cases",
                report.failed_cases, report.total_cases
            ),
        ),
        Err(error) => ReadinessCheck::fail("eval", error.to_string()),
    }
}

fn rust_analyzer_diagnostics_check(root: &Path) -> ReadinessCheck {
    match ProcessCommand::new("rust-analyzer")
        .arg("diagnostics")
        .arg(root)
        .output()
    {
        Ok(output) if output.status.success() => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let warnings = stderr.matches("WeakWarning").count();
            if warnings == 0 {
                ReadinessCheck::ok("rust_analyzer_diagnostics", "diagnostic scan complete")
            } else {
                ReadinessCheck::ok(
                    "rust_analyzer_diagnostics",
                    format!("diagnostic scan complete with {warnings} weak warnings"),
                )
            }
        }
        Ok(output) => {
            let detail = command_failure_detail(output.status, &output.stdout, &output.stderr);
            ReadinessCheck::fail("rust_analyzer_diagnostics", detail)
        }
        Err(error) => ReadinessCheck::fail("rust_analyzer_diagnostics", error.to_string()),
    }
}

fn launchd_readiness_check(project_id: &str) -> ReadinessCheck {
    #[cfg(target_os = "macos")]
    {
        let domain = match current_launchctl_domain() {
            Ok(domain) => domain,
            Err(error) => return ReadinessCheck::fail("launchd", error.to_string()),
        };
        match ProcessCommand::new("launchctl")
            .arg("print")
            .arg(format!("{domain}/{DEFAULT_LABEL}"))
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                launchd_output_readiness_check(&stdout, project_id)
            }
            Ok(output) => ReadinessCheck::fail(
                "launchd",
                command_failure_detail(output.status, &output.stdout, &output.stderr),
            ),
            Err(error) => ReadinessCheck::fail("launchd", error.to_string()),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = project_id;
        ReadinessCheck::skip("launchd", "not macOS")
    }
}

fn current_launchctl_domain() -> Result<String> {
    let output = ProcessCommand::new("id")
        .arg("-u")
        .output()
        .context("failed to execute id -u")?;
    if !output.status.success() {
        bail!(
            "id -u failed: {}",
            command_failure_detail(output.status, &output.stdout, &output.stderr)
        );
    }
    let uid = String::from_utf8(output.stdout)
        .context("id -u returned invalid UTF-8")?
        .trim()
        .to_string();
    Ok(format!("gui/{uid}"))
}

fn launchd_output_readiness_check(output: &str, project_id: &str) -> ReadinessCheck {
    let mut missing = Vec::new();
    if !output.contains("last exit code = 0") {
        missing.push("last exit code is not 0");
    }
    if !output.contains("--project") || !output.contains(project_id) {
        missing.push("project id is not pinned");
    }
    if !output.contains("run interval =") {
        missing.push("run interval is missing");
    }
    if missing.is_empty() {
        ReadinessCheck::ok(
            "launchd",
            format!("{DEFAULT_LABEL} loaded for project {project_id}"),
        )
    } else {
        ReadinessCheck::fail("launchd", missing.join(", "))
    }
}

fn command_failure_detail(
    status: std::process::ExitStatus,
    stdout: &[u8],
    stderr: &[u8],
) -> String {
    let stdout = String::from_utf8_lossy(stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => format!("command exited with {status}"),
        (false, true) => format!("command exited with {status}: {stdout}"),
        (true, false) => format!("command exited with {status}: {stderr}"),
        (false, false) => format!("command exited with {status}: {stderr}; stdout: {stdout}"),
    }
}

fn print_code_reason_report(report: &crate::code_reason::CodeReasonReport) {
    println!("task: {}", report.task);
    println!("model: {}", report.model);
    println!();
    println!("{}", report.answer);
    if !report.bullets.is_empty() {
        println!();
        println!("bullets:");
        for item in &report.bullets {
            println!("- {item}");
        }
    }
    if !report.risks.is_empty() {
        println!();
        println!("risks:");
        for item in &report.risks {
            println!("- {item}");
        }
    }
    if !report.next_steps.is_empty() {
        println!();
        println!("next_steps:");
        for item in &report.next_steps {
            println!("- {item}");
        }
    }
}

fn print_code_pattern_report(report: &crate::code_assist::CodePatternReport) -> Result<()> {
    println!("project: {}", report.project_id);
    println!("query: {}", report.query);
    println!("symbols: {}", report.symbols.len());
    for result in &report.symbols {
        print_code_search_result(result);
    }
    println!("patterns: {}", report.patterns.len());
    for pattern in &report.patterns {
        println!(
            "- seed: {} {}:{}",
            pattern.seed.kind, pattern.seed.file_path, pattern.seed.name
        );
        for related in &pattern.related_symbols {
            println!(
                "  related {:.3}: {} {}:{}",
                related.score, related.symbol.kind, related.symbol.file_path, related.symbol.name
            );
        }
    }
    println!("affected_tests: {}", report.affected_tests.len());
    for file in &report.affected_tests {
        println!("- {file}");
    }
    println!("memory_suggestions: {}", report.memory_suggestions.len());
    println!(
        "{}",
        serde_json::to_string_pretty(&report.memory_suggestions)?
    );
    Ok(())
}

fn print_code_assist_report(report: &CodeAssistReport) -> Result<()> {
    println!("project: {}", report.project_id);
    println!("query: {}", report.query);
    println!("actual_mode: {}", report.actual_mode);
    if let Some(warning) = &report.warning {
        println!("warning: {warning}");
    }
    println!("symbols: {}", report.symbols.len());
    println!("patterns: {}", report.patterns.len());
    println!("duplicate_pairs: {}", report.duplicate_pairs.len());
    println!("impacted_files: {}", report.impacted_files.len());
    for file in &report.impacted_files {
        println!("- {file}");
    }
    println!("affected_tests: {}", report.affected_tests.len());
    for file in &report.affected_tests {
        println!("- {file}");
    }
    println!("memory_suggestions: {}", report.memory_suggestions.len());
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

fn print_compaction_proposal(proposal: &CompactionProposal) {
    println!("model: {}", proposal.model);
    println!("summary_kind: {}", proposal.summary_kind);
    println!("source_memories: {}", proposal.source_ids.len());
    println!("importance: {:.2}", proposal.importance);
    println!("confidence: {:.2}", proposal.confidence);
    if !proposal.tags.is_empty() {
        println!("tags: {}", proposal.tags.join(","));
    }
    println!("reason: {}", proposal.reason);
    println!();
    println!("{}", proposal.summary_body);
}

fn print_maintenance_report(report: &MaintenanceReport) {
    println!("project: {}", report.project_id);
    println!("apply: {}", report.apply);
    if let Some(backup) = &report.backup {
        println!("backup: {}", backup.output.display());
        println!("backup_size_bytes: {}", backup.size_bytes);
    }
    if let Some(validation) = &report.validation {
        println!("validation_status: {}", validation.status);
        println!("validation_model: {}", validation.model);
        println!("pending: {}", validation.pending);
        println!("validation_apply: {}", validation.apply);
        println!("validation_promote: {}", validation.promote);
        println!("validation_archive: {}", validation.archive);
        println!("validation_keep: {}", validation.keep);
        if let Some(error) = &validation.error {
            println!("validation_error: {error}");
        }
    }
    if let Some(compaction) = &report.compaction {
        println!("compaction_status: {}", compaction.status);
        println!("compaction_candidates: {}", compaction.candidate_memories);
        println!("compaction_min_memories: {}", compaction.min_memories);
        if let Some(error) = &compaction.error {
            println!("compaction_error: {error}");
        }
        if let Some(proposal) = &compaction.proposal {
            println!("compaction_summary_kind: {}", proposal.summary_kind);
            println!("compaction_source_memories: {}", proposal.source_ids.len());
        }
        if let Some(application) = &compaction.application {
            println!("compaction_summary_id: {}", application.summary_id);
            println!("compaction_inserted: {}", application.inserted);
            if let Some(duplicate_of) = &application.duplicate_of {
                println!("compaction_duplicate_of: {duplicate_of}");
            }
            println!(
                "compaction_archived_source_memories: {}",
                application.archived_source_memories
            );
        }
    }
    if let Some(feedback) = &report.feedback {
        println!("feedback_status: {}", feedback.status);
        println!("feedback_apply: {}", feedback.apply);
        println!("feedback_considered_events: {}", feedback.considered_events);
        println!("feedback_unapplied_events: {}", feedback.unapplied_events);
        println!("feedback_applied_events: {}", feedback.applied_events);
        println!(
            "feedback_helpful_memories_updated: {}",
            feedback.helpful_memories_updated
        );
        println!(
            "feedback_unhelpful_memories_updated: {}",
            feedback.unhelpful_memories_updated
        );
        if let Some(error) = &feedback.error {
            println!("feedback_error: {error}");
        }
    }
    if let Some(embeddings) = &report.embeddings {
        println!("embedding_scope: {}", embeddings.scope);
        println!("embedding_apply: {}", embeddings.apply);
        println!("memory_model: {}", embeddings.memory_model);
        println!("code_model: {}", embeddings.code_model);
        println!(
            "memories_missing_embeddings: {}",
            embeddings.memories_missing
        );
        println!(
            "code_symbols_missing_embeddings: {}",
            embeddings.code_symbols_missing
        );
        println!("memories_embedded: {}", embeddings.memories_embedded);
        println!(
            "code_symbols_embedded: {}",
            embeddings.code_symbols_embedded
        );
    }
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
}

fn print_memory(memory: &Memory) {
    println!();
    println!("id: {}", memory.id);
    println!("project: {}", memory.project_id);
    println!("scope: {}", memory.scope);
    println!("kind: {}", memory.kind);
    println!("status: {}", memory.status);
    println!("importance: {:.2}", memory.importance);
    println!("confidence: {:.2}", memory.confidence);
    if let Some(score) = memory.score {
        println!("score: {score:.4}");
    }
    println!("created_at: {}", memory.created_at);
    println!("updated_at: {}", memory.updated_at);
    if !memory.tags.is_empty() {
        println!("tags: {}", memory.tags.join(","));
    }
    if let Some(source) = &memory.source {
        println!("source: {source}");
    }
    if let Some(superseded_by) = &memory.superseded_by {
        println!("superseded_by: {superseded_by}");
    }
    if let Some(reason) = &memory.status_reason {
        println!("status_reason: {reason}");
    }
    println!("{}", memory.body);
}

fn print_memory_entity(entity: &MemoryEntity) {
    println!();
    println!("entity_id: {}", entity.id);
    println!("project: {}", entity.project_id);
    println!("entity_type: {}", entity.entity_type);
    println!("name: {}", entity.name);
    if !entity.aliases.is_empty() {
        println!("aliases: {}", entity.aliases.join(","));
    }
    if let Some(description) = &entity.description {
        println!("description: {description}");
    }
}

fn print_memory_fact(fact: &MemoryFact) {
    println!();
    println!("fact_id: {}", fact.id);
    println!("project: {}", fact.project_id);
    if let Some(entity_id) = &fact.entity_id {
        println!("entity_id: {entity_id}");
    }
    if let Some(memory_id) = &fact.memory_id {
        println!("memory_id: {memory_id}");
    }
    println!("predicate: {}", fact.predicate);
    println!("value: {}", fact.value);
    println!("confidence: {:.2}", fact.confidence);
    println!("observed_at: {}", fact.observed_at);
}

fn print_memory_edge(edge: &MemoryEdge) {
    println!();
    println!("edge_id: {}", edge.id);
    println!("project: {}", edge.project_id);
    println!("from: {} ({})", edge.from_entity_name, edge.from_entity_id);
    println!("relation_type: {}", edge.relation_type);
    println!("to: {} ({})", edge.to_entity_name, edge.to_entity_id);
    if let Some(memory_id) = &edge.memory_id {
        println!("memory_id: {memory_id}");
    }
    println!("confidence: {:.2}", edge.confidence);
    println!("observed_at: {}", edge.observed_at);
}

fn print_graph_extraction_report(report: &GraphExtractionReport) {
    println!("project: {}", report.project_id);
    println!("apply: {}", report.apply);
    println!("memories: {}", report.memories);
    println!("proposed_entities: {}", report.proposed_entities);
    println!("proposed_facts: {}", report.proposed_facts);
    println!("proposed_edges: {}", report.proposed_edges);
    if report.apply {
        println!("upserted_entities: {}", report.upserted_entities);
        println!("inserted_facts: {}", report.inserted_facts);
        println!("duplicate_facts: {}", report.duplicate_facts);
        println!("inserted_edges: {}", report.inserted_edges);
        println!("duplicate_edges: {}", report.duplicate_edges);
    }

    for proposal in &report.proposals {
        println!();
        println!("memory_id: {}", proposal.memory_id);
        println!("entities: {}", proposal.entities.len());
        for entity in &proposal.entities {
            println!("- entity {} [{}]", entity.name, entity.entity_type);
            if let Some(description) = &entity.description {
                println!("  description: {description}");
            }
        }
        println!("facts: {}", proposal.facts.len());
        for fact in &proposal.facts {
            println!(
                "- fact {} [{}] {} {} ({:.2})",
                fact.entity_name, fact.entity_type, fact.predicate, fact.value, fact.confidence
            );
        }
        println!("edges: {}", proposal.edges.len());
        for edge in &proposal.edges {
            println!(
                "- edge {} [{}] -{}-> {} [{}] ({:.2})",
                edge.from_name,
                edge.from_type,
                edge.relation_type,
                edge.to_name,
                edge.to_type,
                edge.confidence
            );
        }
    }
}

fn print_project_profile(profile: &crate::store::ProjectRecord) {
    println!("project: {}", profile.id);
    println!("name: {}", profile.name);
    println!("project_type: {}", profile.project_type);
    println!(
        "root_path: {}",
        profile.root_path.as_deref().unwrap_or("<unset>")
    );
    println!(
        "description: {}",
        profile.description.as_deref().unwrap_or("<unset>")
    );
    println!(
        "domains: {}",
        if profile.domains.is_empty() {
            "<none>".to_string()
        } else {
            profile.domains.join(",")
        }
    );
    println!("created_at: {}", profile.created_at);
    println!("updated_at: {}", profile.updated_at);
}

fn print_ontology() {
    println!("core_memory_kinds:");
    for kind in CORE_MEMORY_KINDS {
        println!("- {kind}");
    }
    println!("memory_scopes:");
    for scope in MEMORY_SCOPES {
        println!("- {scope}");
    }
    println!("domain_examples:");
    println!("- generic: decisions, constraints, workflows, setup, external services");
    println!("- game: mechanics, assets, levels, balance, lore");
    println!("- webapp: routes, ux rules, auth, billing, api contracts");
    println!("- library: public api, invariants, benchmarks, compatibility");
    println!("- research: hypotheses, sources, conclusions, experiments");
    println!("- ops: environments, incidents, runbooks, deployment facts");
}

fn print_eval_report(report: &EvalReport, compare_last: bool) {
    println!("eval_run_id: {}", report.run_id);
    if let Some(name) = &report.suite_name {
        println!("eval_suite: {name}");
    }
    if let Some(hash) = &report.suite_hash {
        println!("eval_suite_hash: {hash}");
    }
    println!("eval_total: {}", report.total_cases);
    println!("eval_passed: {}", report.passed_cases);
    println!("eval_failed: {}", report.failed_cases);
    if compare_last {
        match &report.previous_run {
            Some(previous) => {
                println!("previous_run_id: {}", previous.id);
                println!("previous_created_at: {}", previous.created_at);
                println!("previous_passed: {}", previous.passed_cases);
                println!("previous_failed: {}", previous.failed_cases);
                println!(
                    "passed_delta: {}",
                    report.passed_cases as i64 - previous.passed_cases as i64
                );
                println!(
                    "failed_delta: {}",
                    report.failed_cases as i64 - previous.failed_cases as i64
                );
            }
            None => println!("previous_run: <none>"),
        }
    }
    for case in &report.cases {
        println!();
        println!("case: {}", case.name);
        println!("target: {}", case.target);
        println!("project: {}", case.project_id);
        println!("query: {}", case.query);
        println!("passed: {}", case.passed);
        println!("hits: {}", case.hits);
        println!("latency_ms: {}", case.latency_ms);
        if let Some(max_latency_ms) = case.max_latency_ms {
            println!("max_latency_ms: {max_latency_ms}");
            println!("latency_ok: {}", case.latency_ok);
        }
        if let Some(recall) = case.recall_at_k {
            println!("recall_at_k: {:.3}", recall);
        }
        if let Some(precision) = case.precision_at_k {
            println!("precision_at_k: {:.3}", precision);
        }
        if !case.expected_ids.is_empty() {
            println!("expected_ids: {}", case.expected_ids.join(","));
        }
        if !case.matched_expected_ids.is_empty() {
            println!(
                "matched_expected_ids: {}",
                case.matched_expected_ids.join(",")
            );
        }
        if !case.missing_expected_ids.is_empty() {
            println!(
                "missing_expected_ids: {}",
                case.missing_expected_ids.join(",")
            );
        }
        if !case.top_ids.is_empty() {
            println!("top_ids: {}", case.top_ids.join(","));
        }
        if !case.missing_expected.is_empty() {
            println!("missing_expected: {}", case.missing_expected.join(" | "));
        }
        if !case.forbidden_found.is_empty() {
            println!("forbidden_found: {}", case.forbidden_found.join(" | "));
        }
    }
}

fn print_review_memory(memory: &Memory) {
    println!();
    println!("id: {}", memory.id);
    println!("scope: {}", memory.scope);
    println!("kind: {}", memory.kind);
    println!("importance: {:.2}", memory.importance);
    println!("confidence: {:.2}", memory.confidence);
    if !memory.tags.is_empty() {
        println!("tags: {}", memory.tags.join(","));
    }
    if let Some(reason) = &memory.status_reason {
        println!("reason: {reason}");
    }
    println!("body: {}", memory.body);
    println!("promote: cargo run -- promote {}", memory.id);
    println!("archive: cargo run -- archive {}", memory.id);
}

fn impacted_files_for_relations(
    store: &Store,
    project_id: &str,
    symbol: &str,
    callers: &[CodeRelation],
    callees: &[CodeRelation],
) -> Result<Vec<String>> {
    let mut files = BTreeSet::new();
    if let Some(symbol) = store.get_code_symbol(project_id, symbol)? {
        files.insert(symbol.file_path);
    }
    for relation in callers.iter().chain(callees.iter()) {
        files.insert(relation.from_file_path.clone());
        if let Some(target_symbol_id) = &relation.target_symbol_id
            && let Some(target_symbol) = store.get_code_symbol(project_id, target_symbol_id)?
        {
            files.insert(target_symbol.file_path);
        }
    }
    Ok(files.into_iter().collect())
}

fn print_code_search_result(result: &CodeSearchResult) {
    println!();
    println!("score: {:.4}", result.score);
    print_code_symbol(&result.symbol, false);
}

fn print_code_symbol(symbol: &CodeSymbol, include_body: bool) {
    println!("id: {}", symbol.id);
    println!(
        "file: {}:{}-{}",
        symbol.file_path, symbol.start_line, symbol.end_line
    );
    println!("kind: {}", symbol.kind);
    println!("name: {}", symbol.name);
    println!("signature: {}", symbol.signature);
    if let Some(parent_id) = &symbol.parent_id {
        println!("parent_id: {parent_id}");
    }
    if include_body {
        println!("{}", symbol.body);
    }
}

fn print_code_relation(relation: &CodeRelation) {
    println!();
    println!("id: {}", relation.id);
    println!("kind: {}", relation.relation_kind);
    println!("from_file: {}", relation.from_file_path);
    if let Some(from_symbol_id) = &relation.from_symbol_id {
        println!("from_symbol_id: {from_symbol_id}");
    }
    println!("target: {}", relation.target_name);
    if let Some(target_symbol_id) = &relation.target_symbol_id {
        println!("target_symbol_id: {target_symbol_id}");
    }
}

fn print_code_memories(header: &str, memories: &[CodeMemory]) {
    println!("{header}");
    if memories.is_empty() {
        println!("- none");
        return;
    }
    for memory in memories {
        println!();
        println!("id: {}", memory.id);
        println!("kind: {}", memory.kind);
        println!("status: {}", memory.status);
        println!("link_status: {}", memory.link_status);
        println!("confidence: {:.2}", memory.confidence);
        println!("symbol_id: {}", memory.symbol_id.as_deref().unwrap_or("-"));
        println!("file_path: {}", memory.file_path.as_deref().unwrap_or("-"));
        if memory.symbol_name.is_some() || memory.symbol_kind.is_some() {
            println!(
                "symbol_snapshot: {} {}",
                memory.symbol_kind.as_deref().unwrap_or("-"),
                memory.symbol_name.as_deref().unwrap_or("-")
            );
        }
        if memory.symbol_start_line.is_some() || memory.symbol_end_line.is_some() {
            println!(
                "symbol_lines: {}-{}",
                memory
                    .symbol_start_line
                    .map(|line| line.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                memory
                    .symbol_end_line
                    .map(|line| line.to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
        if memory.relink_attempts > 0 {
            println!("relink_attempts: {}", memory.relink_attempts);
        }
        if let Some(last_relinked_at) = &memory.last_relinked_at {
            println!("last_relinked_at: {last_relinked_at}");
        }
        println!("updated_at: {}", memory.updated_at);
        if !memory.tags.is_empty() {
            println!("tags: {}", memory.tags.join(","));
        }
        if let Some(source) = &memory.source {
            println!("source: {source}");
        }
        println!("body: {}", memory.body);
    }
}

fn print_route_hints(header: &str, routes: &[CodeRouteHint]) {
    println!("{header}");
    if routes.is_empty() {
        println!("- none");
        return;
    }
    for route in routes {
        println!();
        println!("framework: {}", route.framework);
        println!("route: {}", route.route);
        println!("method: {}", route.method.as_deref().unwrap_or("*"));
        println!("file: {}", route.file_path);
        println!("symbol: {} ({})", route.symbol_name, route.symbol_id);
        println!("evidence: {}", route.evidence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::CodeFile;

    #[test]
    fn embedding_readiness_fails_on_missing_vectors() {
        let checks = embedding_readiness_checks(3, 2, 10, 10);
        assert_eq!(checks[0].state, ReadinessState::Fail);
        assert_eq!(checks[1].state, ReadinessState::Ok);

        let checks = embedding_readiness_checks(3, 3, 10, 9);
        assert_eq!(checks[0].state, ReadinessState::Ok);
        assert_eq!(checks[1].state, ReadinessState::Fail);
    }

    #[test]
    fn embedding_readiness_skips_code_when_no_symbols_are_indexed() {
        let checks = embedding_readiness_checks(0, 0, 0, 0);
        assert_eq!(checks[0].state, ReadinessState::Ok);
        assert_eq!(checks[1].state, ReadinessState::Skip);
    }

    #[test]
    fn code_relation_readiness_requires_some_resolved_targets_when_relations_exist() {
        assert_eq!(
            code_relation_readiness_check(0, 0).state,
            ReadinessState::Skip
        );
        assert_eq!(
            code_relation_readiness_check(10, 0).state,
            ReadinessState::Fail
        );
        assert_eq!(
            code_relation_readiness_check(10, 4).state,
            ReadinessState::Ok
        );
    }

    #[test]
    fn code_index_freshness_readiness_reports_stale_files() {
        let fresh = CodeFreshnessReport {
            project_id: "game".to_string(),
            root_path: PathBuf::from("/tmp/game"),
            files_seen: 2,
            indexed_files: 2,
            stale_files: Vec::new(),
            missing_files: Vec::new(),
            deleted_files: Vec::new(),
        };
        assert_eq!(
            code_index_freshness_readiness_check(&fresh).state,
            ReadinessState::Ok
        );

        let stale = CodeFreshnessReport {
            stale_files: vec!["src/lib.rs".to_string()],
            missing_files: vec!["src/new.rs".to_string()],
            deleted_files: vec!["Cargo.toml".to_string()],
            ..fresh
        };
        let check = code_index_freshness_readiness_check(&stale);
        assert_eq!(check.state, ReadinessState::Fail);
        assert!(check.detail.contains("run code-index"));
    }

    #[test]
    fn ra_reference_readiness_requires_lsif_import_when_code_is_indexed() {
        assert_eq!(
            ra_reference_readiness_check(0, 0).state,
            ReadinessState::Skip
        );
        assert_eq!(
            ra_reference_readiness_check(10, 0).state,
            ReadinessState::Fail
        );
        assert_eq!(
            ra_reference_readiness_check(10, 4).state,
            ReadinessState::Ok
        );
    }

    #[test]
    fn ra_call_readiness_requires_lsif_call_edges_when_references_exist() {
        assert_eq!(ra_call_readiness_check(0, 0, 0).state, ReadinessState::Skip);
        assert_eq!(
            ra_call_readiness_check(10, 0, 0).state,
            ReadinessState::Skip
        );
        assert_eq!(
            ra_call_readiness_check(10, 4, 0).state,
            ReadinessState::Fail
        );
        assert_eq!(ra_call_readiness_check(10, 4, 2).state, ReadinessState::Ok);
    }

    #[test]
    fn launchd_readiness_requires_project_pin_and_successful_last_exit() {
        let output = r#"
gui/501/com.dukememory.maintenance = {
    arguments = {
        /tmp/dukememory
        maintenance
        --all
        --backup
        --apply
        --project
        game-a
    }
    last exit code = 0
    run interval = 21600 seconds
}
"#;
        assert_eq!(
            launchd_output_readiness_check(output, "game-a").state,
            ReadinessState::Ok
        );
        assert_eq!(
            launchd_output_readiness_check(output, "game-b").state,
            ReadinessState::Fail
        );
        assert_eq!(
            launchd_output_readiness_check(&output.replace("last exit code = 0", ""), "game-a")
                .state,
            ReadinessState::Fail
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cli_hybrid_memory_search_falls_back_to_keyword_when_embedder_unavailable() -> Result<()>
    {
        let config = test_config("cli-hybrid-memory");
        let store = Store::open(&config.database_marker)?;
        store.remember(
            "cli-hybrid-memory",
            NewMemory {
                scope: DEFAULT_MEMORY_SCOPE.to_string(),
                memory_tier: DEFAULT_MEMORY_TIER.to_string(),
                kind: "decision".to_string(),
                body: "Inventory retrieval uses keyword fallback when embeddings are offline."
                    .to_string(),
                tags: vec!["retrieval".to_string()],
                source: Some("test".to_string()),
                status: MemoryStatus::Active,
                importance: 0.8,
                confidence: 0.9,
                status_reason: None,
                allow_sensitive: false,
            },
        )?;

        let results = search_memories(
            &config,
            &store,
            MemorySearchRequest {
                project_id: "cli-hybrid-memory",
                query: "inventory fallback",
                limit: 8,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
                mode: SearchMode::Hybrid,
                allow_hybrid_fallback: true,
            },
        )
        .await?;

        assert_eq!(results.len(), 1);
        assert!(results[0].body.contains("keyword fallback"));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cli_hybrid_code_search_falls_back_to_keyword_when_embedder_unavailable() -> Result<()>
    {
        let config = test_config("cli-hybrid-code");
        let store = Store::open(&config.database_marker)?;
        store.upsert_code_file(CodeFile {
            project_id: "cli-hybrid-code".to_string(),
            path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            hash: "test-hash".to_string(),
            size_bytes: 32,
            line_count: 1,
        })?;
        store.upsert_code_symbol(CodeSymbol {
            id: "sym_cli_hybrid_code_lookup".to_string(),
            project_id: "cli-hybrid-code".to_string(),
            file_path: "src/lib.rs".to_string(),
            language: "rust".to_string(),
            name: "inventory_lookup".to_string(),
            kind: "function".to_string(),
            signature: "pub fn inventory_lookup()".to_string(),
            body: "pub fn inventory_lookup() {}".to_string(),
            start_line: 1,
            end_line: 1,
            parent_id: None,
        })?;

        let results = search_code(
            &config,
            &store,
            CodeSearchRequest {
                project_id: "cli-hybrid-code",
                query: "inventory lookup",
                limit: 8,
                kind: None,
                file_path: None,
                mode: SearchMode::Hybrid,
                allow_hybrid_fallback: true,
            },
        )
        .await?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].symbol.name, "inventory_lookup");
        Ok(())
    }

    fn test_config(name: &str) -> Config {
        Config {
            database_url: "postgresql://dukememory-test@localhost:55432/dukememory_test"
                .to_string(),
            database_marker: std::env::temp_dir().join(format!(
                "dukememory-cli-test-{name}-{}.schema-marker",
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
}

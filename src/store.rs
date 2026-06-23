use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use pgvector::Vector;
use postgres::types::ToSql;
use postgres::{Client, NoTls, Row, SimpleQueryMessage, Transaction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::safety::{enforce_memory_safety, inspect_memory_safety};

pub const SCHEMA_VERSION: i64 = 15;
pub const PROJECT_EXPORT_FORMAT: &str = "dukememory.project-export.v1";
pub const DEFAULT_PROJECT_TYPE: &str = "generic";
pub const DEFAULT_MEMORY_SCOPE: &str = "project";
pub const DEFAULT_MEMORY_TIER: &str = "archival";

pub const CORE_MEMORY_KINDS: &[&str] = &[
    "decision",
    "project_rule",
    "constraint",
    "architecture",
    "code_fact",
    "bug_regression",
    "workflow",
    "setup",
    "external_service",
    "user_preference",
    "project_summary",
    "note",
];

pub const MEMORY_SCOPES: &[&str] = &["project", "session", "module", "user", "global"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStatus {
    Pending,
    Active,
    Superseded,
    Archived,
}

impl MemoryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Archived => "archived",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "pending" => Ok(Self::Pending),
            "active" => Ok(Self::Active),
            "superseded" => Ok(Self::Superseded),
            "archived" => Ok(Self::Archived),
            other => bail!("invalid memory status `{other}`"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusFilter {
    One(MemoryStatus),
    Any,
}

impl StatusFilter {
    pub fn parse(value: Option<&str>, default: MemoryStatus) -> Result<Self> {
        match value {
            Some("any") => Ok(Self::Any),
            Some(value) => Ok(Self::One(MemoryStatus::parse(value)?)),
            None => Ok(Self::One(default)),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: String,
    pub project_id: String,
    pub scope: String,
    pub memory_tier: String,
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub status: String,
    pub importance: f64,
    pub confidence: f64,
    pub superseded_by: Option<String>,
    pub status_reason: Option<String>,
    pub score: Option<f64>,
    pub quality_score: f64,
    pub usage_count: u64,
    pub last_used_at: Option<String>,
    pub contradiction_risk: f64,
    pub created_at: String,
    pub updated_at: String,
}

pub struct NewMemory {
    pub scope: String,
    pub memory_tier: String,
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub status: MemoryStatus,
    pub importance: f64,
    pub confidence: f64,
    pub status_reason: Option<String>,
    pub allow_sensitive: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RememberOutcome {
    pub id: String,
    pub inserted: bool,
    pub duplicate_of: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalCaseRecord {
    pub id: String,
    pub suite_id: String,
    pub project_id: String,
    pub name: String,
    pub query: String,
    pub expected_contains: Vec<String>,
    pub forbidden_contains: Vec<String>,
    pub expected_ids: Vec<String>,
    pub forbidden_ids: Vec<String>,
    pub min_results: Option<u64>,
    pub created_at: String,
}

pub struct NewEvalCase<'a> {
    pub suite_name: &'a str,
    pub name: &'a str,
    pub query: &'a str,
    pub expected_contains: Vec<String>,
    pub forbidden_contains: Vec<String>,
    pub expected_ids: Vec<String>,
    pub forbidden_ids: Vec<String>,
    pub min_results: Option<u64>,
}

pub struct SearchOptions {
    pub query: String,
    pub limit: usize,
    pub status: StatusFilter,
    pub kind: Option<String>,
    pub memory_tier: Option<String>,
}

pub struct ListOptions {
    pub limit: usize,
    pub offset: usize,
    pub status: StatusFilter,
    pub kind: Option<String>,
    pub memory_tier: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectStatus {
    pub project_id: String,
    pub project_type: String,
    pub pending_memories: u64,
    pub active_memories: u64,
    pub superseded_memories: u64,
    pub archived_memories: u64,
    pub total_memories: u64,
    pub memory_embeddings: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFile {
    pub project_id: String,
    pub path: String,
    pub language: String,
    pub hash: String,
    pub size_bytes: u64,
    pub line_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSymbol {
    pub id: String,
    pub project_id: String,
    pub file_path: String,
    pub language: String,
    pub name: String,
    pub kind: String,
    pub signature: String,
    pub body: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRelation {
    pub id: String,
    pub project_id: String,
    pub from_symbol_id: Option<String>,
    pub from_file_path: String,
    pub relation_kind: String,
    pub target_name: String,
    pub target_symbol_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntity {
    pub id: String,
    pub project_id: String,
    pub entity_type: String,
    pub name: String,
    pub aliases: Vec<String>,
    pub description: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    pub id: String,
    pub project_id: String,
    pub entity_id: Option<String>,
    pub memory_id: Option<String>,
    pub episode_id: Option<String>,
    pub predicate: String,
    pub value: String,
    pub confidence: f64,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub invalidated_by: Option<String>,
    pub observed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: String,
    pub project_id: String,
    pub from_entity_id: String,
    pub from_entity_name: String,
    pub to_entity_id: String,
    pub to_entity_name: String,
    pub relation_type: String,
    pub memory_id: Option<String>,
    pub episode_id: Option<String>,
    pub confidence: f64,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
    pub invalidated_by: Option<String>,
    pub observed_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryGraph {
    pub entities: Vec<MemoryEntity>,
    pub facts: Vec<MemoryFact>,
    pub edges: Vec<MemoryEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEpisode {
    pub id: String,
    pub project_id: String,
    pub source: String,
    pub summary: Option<String>,
    pub raw_ref: Option<String>,
    pub raw_payload: Value,
    pub observed_at: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoreHealth {
    pub schema: String,
    pub projects: u64,
    pub memories: u64,
    pub memory_embeddings: u64,
    pub code_files: u64,
    pub code_symbols: u64,
    pub code_relations: u64,
    pub code_symbol_embeddings: u64,
    pub memory_entities: u64,
    pub memory_facts: u64,
    pub memory_edges: u64,
    pub audit_events: u64,
    pub task_sessions: u64,
    pub temp_schemas: u64,
    pub database_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    pub id: String,
    pub project_id: String,
    pub actor: String,
    pub action: String,
    pub target_type: String,
    pub target_id: Option<String>,
    pub detail: Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalRunSummary {
    pub id: String,
    pub project_id: String,
    pub suite_name: Option<String>,
    pub suite_hash: Option<String>,
    pub mode: String,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub created_at: String,
}

pub struct EvalRunRecord<'a> {
    pub project_id: &'a str,
    pub suite_name: Option<&'a str>,
    pub suite_hash: Option<&'a str>,
    pub mode: &'a str,
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: usize,
    pub detail: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskSession {
    pub id: String,
    pub project_id: String,
    pub query: String,
    pub status: String,
    pub phase: String,
    pub progress: usize,
    pub memory_ids: Vec<String>,
    pub code_symbol_ids: Vec<String>,
    pub file_paths: Vec<String>,
    pub test_paths: Vec<String>,
    pub summary: Option<String>,
    pub result: Value,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
}

pub struct NewTaskSession<'a> {
    pub project_id: &'a str,
    pub query: &'a str,
    pub status: &'a str,
    pub phase: &'a str,
    pub progress: usize,
    pub result: Value,
}

pub struct TaskSessionUpdate {
    pub status: Option<String>,
    pub phase: Option<String>,
    pub progress: Option<usize>,
    pub memory_ids: Option<Vec<String>>,
    pub code_symbol_ids: Option<Vec<String>>,
    pub file_paths: Option<Vec<String>>,
    pub test_paths: Option<Vec<String>>,
    pub summary: Option<Option<String>>,
    pub result: Option<Value>,
}

pub struct RetrievalEventRecord<'a> {
    pub project_id: &'a str,
    pub task_session_id: Option<&'a str>,
    pub tool: &'a str,
    pub query: &'a str,
    pub task_type: &'a str,
    pub token_budget: usize,
    pub estimated_tokens: usize,
    pub latency_ms: u64,
    pub memory_fragments: usize,
    pub code_hits: usize,
    pub graph_items: usize,
    pub code_memories: usize,
    pub plan: Value,
    pub audit: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalEvent {
    pub id: String,
    pub project_id: String,
    pub task_session_id: Option<String>,
    pub tool: String,
    pub query: String,
    pub task_type: String,
    pub token_budget: usize,
    pub estimated_tokens: usize,
    pub latency_ms: u64,
    pub memory_fragments: usize,
    pub code_hits: usize,
    pub graph_items: usize,
    pub code_memories: usize,
    pub plan: Value,
    pub audit: Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeedbackEffect {
    pub helpful_memories_updated: usize,
    pub unhelpful_memories_updated: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CleanupSchemasReport {
    pub dry_run: bool,
    pub dropped: Vec<String>,
    pub kept: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeSearchResult {
    pub symbol: CodeSymbol,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRecord {
    pub id: String,
    pub name: String,
    pub root_path: Option<String>,
    pub project_type: String,
    pub description: Option<String>,
    pub domains: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectProfileUpdate {
    pub name: Option<String>,
    pub root_path: Option<String>,
    pub project_type: Option<String>,
    pub description: Option<String>,
    pub domains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMemory {
    pub id: String,
    pub project_id: String,
    #[serde(default = "default_memory_scope")]
    pub scope: String,
    #[serde(default = "default_memory_tier")]
    pub memory_tier: String,
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub status: String,
    pub importance: f64,
    pub confidence: f64,
    pub superseded_by: Option<String>,
    pub status_reason: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectExport {
    pub format: String,
    pub schema_version: i64,
    pub exported_at: String,
    pub project: ProjectRecord,
    pub includes_code: bool,
    pub includes_embeddings: bool,
    pub memories: Vec<ExportedMemory>,
    #[serde(default)]
    pub code_files: Vec<CodeFile>,
    #[serde(default)]
    pub code_symbols: Vec<CodeSymbol>,
    #[serde(default)]
    pub code_relations: Vec<CodeRelation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProjectImportReport {
    pub project_id: String,
    pub overwrite: bool,
    pub memories_imported: usize,
    pub memories_skipped: usize,
    pub code_files_imported: usize,
    pub code_files_skipped: usize,
    pub code_symbols_imported: usize,
    pub code_symbols_skipped: usize,
    pub code_relations_imported: usize,
    pub code_relations_skipped: usize,
}

pub struct CodeSearchOptions {
    pub query: String,
    pub limit: usize,
    pub kind: Option<String>,
    pub file_path: Option<String>,
}

pub struct CodeSimilarityPairOptions {
    pub embedding_model: String,
    pub limit: usize,
    pub kind: Option<String>,
    pub file_path: Option<String>,
    pub min_similarity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeSimilarityPair {
    pub left: CodeSymbol,
    pub right: CodeSymbol,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeMemory {
    pub id: String,
    pub project_id: String,
    pub symbol_id: Option<String>,
    pub file_path: Option<String>,
    pub link_status: String,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub symbol_signature: Option<String>,
    pub symbol_start_line: Option<u32>,
    pub symbol_end_line: Option<u32>,
    pub symbol_body_hash: Option<String>,
    pub last_relinked_at: Option<String>,
    pub relink_attempts: u32,
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub status: String,
    pub confidence: f64,
    pub quality_score: f64,
    pub usage_count: u64,
    pub last_used_at: Option<String>,
    pub contradiction_risk: f64,
    pub status_reason: Option<String>,
    pub score: Option<f64>,
    pub created_at: String,
    pub updated_at: String,
}

pub struct NewCodeMemory {
    pub symbol_id: Option<String>,
    pub symbol_kind: Option<String>,
    pub file_path: Option<String>,
    pub status: String,
    pub kind: String,
    pub body: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub confidence: f64,
}

pub struct CodeMemorySearchOptions {
    pub query: Option<String>,
    pub limit: usize,
    pub status: String,
    pub kind: Option<String>,
    pub symbol_ids: Vec<String>,
    pub file_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeMemoryRepairReport {
    pub project_id: String,
    pub scanned: usize,
    pub repaired: usize,
    pub ambiguous: usize,
    pub stale: usize,
    pub dry_run: bool,
    pub results: Vec<CodeMemoryRepairResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeMemoryRepairResult {
    pub memory_id: String,
    pub old_symbol_id: Option<String>,
    pub new_symbol_id: Option<String>,
    pub file_path: Option<String>,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub status: String,
    pub reason: String,
    pub candidates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRouteHint {
    pub project_id: String,
    pub file_path: String,
    pub symbol_id: String,
    pub symbol_name: String,
    pub framework: String,
    pub route: String,
    pub method: Option<String>,
    pub evidence: String,
}

pub struct VectorSearchOptions {
    pub embedding: Vec<f32>,
    pub embedding_model: String,
    pub limit: usize,
    pub status: StatusFilter,
    pub kind: Option<String>,
    pub memory_tier: Option<String>,
}

pub struct CodeVectorSearchOptions {
    pub embedding: Vec<f32>,
    pub embedding_model: String,
    pub limit: usize,
    pub kind: Option<String>,
    pub file_path: Option<String>,
}

pub struct MemorySimilarityPairOptions {
    pub embedding_model: String,
    pub limit: usize,
    pub status: StatusFilter,
    pub kind: Option<String>,
    pub memory_tier: Option<String>,
    pub min_similarity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySimilarityPair {
    pub left: Memory,
    pub right: Memory,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingModelCount {
    pub model: String,
    pub embedding_kind: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingModelStats {
    pub memory_models: Vec<EmbeddingModelCount>,
    pub code_models: Vec<EmbeddingModelCount>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeIndexStatus {
    pub project_id: String,
    pub files: u64,
    pub symbols: u64,
    pub relations: u64,
    pub resolved_relations: u64,
    pub ra_references: u64,
    pub ra_calls: u64,
    pub symbol_embeddings: u64,
    pub languages: Vec<CodeLanguageStatus>,
    pub relation_counts: CodeRelationCounts,
    pub quality: CodeRelationQuality,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeLanguageStatus {
    pub language: String,
    pub files: u64,
    pub symbols: u64,
    pub relations: u64,
    pub resolved_relations: u64,
    pub symbol_embeddings: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeRelationQuality {
    pub relation_resolution_rate: f64,
    pub unresolved_relations: u64,
    pub ambiguous_unresolved_targets: u64,
    pub top_unresolved_targets: Vec<CodeUnresolvedTarget>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeRelationCounts {
    pub total: u64,
    pub total_resolved: u64,
    pub project_quality: u64,
    pub project_quality_resolved: u64,
    pub project_quality_unresolved: u64,
    pub external: u64,
    pub external_call: u64,
    pub external_use: u64,
    pub cargo_package: u64,
    pub ra_reference: u64,
    pub ra_call: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeUnresolvedTarget {
    pub relation_kind: String,
    pub target_name: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodeRelationResolutionReport {
    pub project_id: String,
    pub targets_reset: usize,
    pub calls_resolved: usize,
    pub uses_resolved: usize,
    pub modules_resolved: usize,
}

pub struct Store {
    database_url: String,
    clients: Mutex<Vec<Client>>,
    schema: Option<String>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let database_url = env::var("DUKEMEMORY_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(default_database_url);
        let mut client = connect_postgres(&database_url)?;
        let schema = schema_for_path(path)?;
        let setup_result = setup_connection(&mut client, schema.as_deref())
            .context("failed to prepare PostgreSQL connection");
        if let Err(error) = setup_result {
            close_client(client);
            return Err(error);
        }
        let migration_key = migration_cache_key(&database_url, schema.as_deref());
        let migration_result = ensure_migrations_applied(&mut client, &migration_key)
            .context("failed to apply PostgreSQL migrations");
        if let Err(error) = migration_result {
            close_client(client);
            return Err(error);
        }
        Ok(Self {
            database_url,
            clients: Mutex::new(vec![client]),
            schema,
        })
    }

    pub fn schema_version(&self) -> Result<i64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COALESCE(MAX(version), 0)::BIGINT FROM dukememory_schema_migrations",
            &[],
        )?;
        Ok(row.get::<_, i64>(0))
    }

    pub fn integrity_check(&self) -> Result<String> {
        let mut client = self.client()?;
        client.simple_query("SELECT 1")?;
        Ok("ok".to_string())
    }

    pub fn backup_to(&self, output_path: &Path) -> Result<u64> {
        if output_path.exists() {
            bail!("backup target already exists: {}", output_path.display());
        }
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create backup directory {}", parent.display())
            })?;
        }
        let database_url = env::var("DUKEMEMORY_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(default_database_url);
        let mut command = std::process::Command::new(pg_dump_path());
        command
            .arg("--format=custom")
            .arg("--file")
            .arg(output_path);
        if let Some(schema) = &self.schema {
            command.arg("--schema").arg(schema);
        }
        let status = command
            .arg(&database_url)
            .status()
            .context("failed to run pg_dump")?;
        if !status.success() {
            bail!("pg_dump failed with status {status}");
        }
        Ok(fs::metadata(output_path)?.len())
    }

    pub fn remember(&self, project_id: &str, memory: NewMemory) -> Result<String> {
        Ok(self.remember_inner(project_id, memory, false)?.id)
    }

    pub fn remember_deduplicated(
        &self,
        project_id: &str,
        memory: NewMemory,
    ) -> Result<RememberOutcome> {
        self.remember_inner(project_id, memory, true)
    }

    fn remember_inner(
        &self,
        project_id: &str,
        memory: NewMemory,
        deduplicate: bool,
    ) -> Result<RememberOutcome> {
        validate_score("importance", memory.importance)?;
        validate_score("confidence", memory.confidence)?;
        let scope = normalize_memory_scope(&memory.scope)?;
        let memory_tier = normalize_memory_tier(&memory.memory_tier)?;
        let safety = inspect_memory_safety(&memory.body, &memory.tags, memory.source.as_deref());
        enforce_memory_safety(&safety, memory.allow_sensitive)?;
        self.ensure_project(project_id)?;

        let body_hash = memory_body_hash(&memory.body);
        if deduplicate
            && let Some(existing_id) = self.find_live_memory_by_hash(project_id, &body_hash)?
        {
            return Ok(RememberOutcome {
                id: existing_id.clone(),
                inserted: false,
                duplicate_of: Some(existing_id),
            });
        }

        let id = Uuid::now_v7();
        let tags = json_array(normalize_tags(memory.tags));
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO memories (
                id, project_id, scope, memory_tier, kind, body, body_hash, tags, source, status,
                importance, confidence, status_reason
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13)",
            &[
                &id,
                &project_id,
                &scope,
                &memory_tier,
                &memory.kind,
                &memory.body,
                &body_hash,
                &tags,
                &memory.source,
                &memory.status.as_str(),
                &memory.importance,
                &memory.confidence,
                &memory.status_reason,
            ],
        )?;
        sync_memory_fragments(&mut client, project_id, id, &body_hash, &memory.body)?;
        Ok(RememberOutcome {
            id: id.to_string(),
            inserted: true,
            duplicate_of: None,
        })
    }

    pub fn search(&self, project_id: &str, options: SearchOptions) -> Result<Vec<Memory>> {
        self.search_memory_fragments(project_id, options)
    }

    pub fn search_memory_fragments(
        &self,
        project_id: &str,
        options: SearchOptions,
    ) -> Result<Vec<Memory>> {
        let query = to_tsquery(&options.query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = memory_fragment_select_sql(
            "ts_rank_cd(f.search_vector, websearch_to_tsquery('simple', $2))::double precision",
        );
        sql.push_str(
            " WHERE f.project_id = $1
                AND f.search_vector @@ websearch_to_tsquery('simple', $2)",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> =
            vec![Box::new(project_id.to_string()), Box::new(query)];
        append_pg_filters(&mut sql, &mut params, options.status, options.kind);
        append_memory_tier_filter(&mut sql, &mut params, options.memory_tier)?;
        push_param(&mut sql, &mut params, options.limit.clamp(1, 50) as i64);
        sql.push_str(
            " ORDER BY score DESC, m.quality_score DESC, m.contradiction_risk ASC, m.importance DESC, m.updated_at DESC, f.chunk_index ASC LIMIT $",
        );
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn list(&self, project_id: &str, options: ListOptions) -> Result<Vec<Memory>> {
        self.ensure_project(project_id)?;
        let mut sql = memory_select_sql("NULL::double precision");
        sql.push_str(" WHERE m.project_id = $1");
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![Box::new(project_id.to_string())];
        append_pg_filters(&mut sql, &mut params, options.status, options.kind);
        append_memory_tier_filter(&mut sql, &mut params, options.memory_tier)?;
        push_param(&mut sql, &mut params, options.limit.clamp(1, 100) as i64);
        let limit_index = params.len();
        push_param(&mut sql, &mut params, options.offset as i64);
        sql.push_str(&format!(
            " ORDER BY m.updated_at DESC, m.created_at DESC LIMIT ${limit_index} OFFSET ${}",
            params.len()
        ));
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn core_context_memories_for_query(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        self.ensure_project(project_id)?;
        let query = to_tsquery(query);
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let sql = format!(
            "{} WHERE f.project_id = $1
                AND m.status = 'active'
                AND (m.memory_tier = 'core' OR m.kind IN ('project_rule', 'constraint'))
                AND f.search_vector @@ websearch_to_tsquery('simple', $2)
             ORDER BY
                score DESC,
                m.quality_score DESC,
                m.contradiction_risk ASC,
                CASE WHEN m.memory_tier = 'core' THEN 0 ELSE 1 END,
                m.importance DESC,
                m.confidence DESC,
                m.updated_at DESC,
                m.created_at DESC
             LIMIT $3",
            memory_fragment_select_sql(
                "ts_rank_cd(f.search_vector, websearch_to_tsquery('simple', $2))::double precision"
            )
        );
        let mut client = self.client()?;
        let rows = client.query(&sql, &[&project_id, &query, &(limit.clamp(1, 10) as i64)])?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn get(&self, project_id: &str, id: &str) -> Result<Option<Memory>> {
        let memory_id = parse_uuid(id)?;
        let sql = format!(
            "{} WHERE m.project_id = $1 AND m.id = $2",
            memory_select_sql("NULL::double precision")
        );
        let mut client = self.client()?;
        client
            .query_opt(&sql, &[&project_id, &memory_id])?
            .map(|row| memory_from_row(&row))
            .transpose()
    }

    pub fn search_memories_as_of(
        &self,
        project_id: &str,
        query: &str,
        as_of: &str,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        self.ensure_project(project_id)?;
        let query = to_tsquery(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "{} WHERE m.project_id = $1
                AND m.created_at <= $3::text::timestamptz
                AND (
                    m.status = 'active'
                    OR (m.status IN ('superseded', 'archived') AND m.updated_at > $3::text::timestamptz)
                )
                AND m.search_vector @@ websearch_to_tsquery('simple', $2)
             ORDER BY ts_rank_cd(m.search_vector, websearch_to_tsquery('simple', $2)) DESC,
                      m.quality_score DESC,
                      m.updated_at DESC
             LIMIT $4",
            memory_select_sql(
                "ts_rank_cd(m.search_vector, websearch_to_tsquery('simple', $2))::double precision"
            )
        );
        let mut client = self.client()?;
        let rows = client.query(
            &sql,
            &[&project_id, &query, &as_of, &(limit.clamp(1, 100) as i64)],
        )?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn promote(&self, project_id: &str, id: &str, reason: Option<&str>) -> Result<()> {
        self.set_status(project_id, id, MemoryStatus::Active, reason)
    }

    pub fn archive(&self, project_id: &str, id: &str, reason: Option<&str>) -> Result<()> {
        self.set_status(project_id, id, MemoryStatus::Archived, reason)
    }

    pub fn set_memory_tier(
        &self,
        project_id: &str,
        id: &str,
        memory_tier: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        self.ensure_project(project_id)?;
        let id = parse_uuid(id)?;
        let memory_tier = normalize_memory_tier(memory_tier)?;
        let mut client = self.client()?;
        client.execute(
            "UPDATE memories
             SET memory_tier = $1,
                 status_reason = COALESCE($2, status_reason),
                 updated_at = now()
             WHERE project_id = $3 AND id = $4",
            &[&memory_tier, &reason, &project_id, &id],
        )?;
        Ok(())
    }

    pub fn prune_pending(
        &self,
        project_id: &str,
        limit: usize,
        max_confidence: Option<f64>,
        dry_run: bool,
        reason: Option<&str>,
    ) -> Result<Vec<Memory>> {
        if let Some(max_confidence) = max_confidence {
            validate_score("max_confidence", max_confidence)?;
        }
        let mut sql = memory_select_sql("NULL::double precision");
        sql.push_str(" WHERE m.project_id = $1 AND m.status = 'pending'");
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![Box::new(project_id.to_string())];
        if let Some(max_confidence) = max_confidence {
            push_param(&mut sql, &mut params, max_confidence);
            sql.push_str(" AND m.confidence <= $");
            sql.push_str(&params.len().to_string());
        }
        push_param(&mut sql, &mut params, limit.clamp(1, 500) as i64);
        sql.push_str(" ORDER BY m.updated_at ASC, m.created_at ASC LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        let memories = rows
            .iter()
            .map(memory_from_row)
            .collect::<Result<Vec<_>>>()?;
        drop(client);
        if !dry_run {
            for memory in &memories {
                self.archive(project_id, &memory.id, reason)?;
            }
        }
        Ok(memories)
    }

    pub fn active_memories_for_compaction(
        &self,
        project_id: &str,
        limit: usize,
        kind: Option<String>,
    ) -> Result<Vec<Memory>> {
        self.ensure_project(project_id)?;
        let mut sql = memory_select_sql("NULL::double precision");
        sql.push_str(" WHERE m.project_id = $1 AND m.status = 'active'");
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![Box::new(project_id.to_string())];
        if let Some(kind) = kind {
            push_param(&mut sql, &mut params, kind);
            sql.push_str(" AND m.kind = $");
            sql.push_str(&params.len().to_string());
        } else {
            sql.push_str(" AND m.kind <> 'project_summary'");
        }
        sql.push_str(" AND m.memory_tier <> 'conversation'");
        push_param(&mut sql, &mut params, limit.clamp(2, 500) as i64);
        sql.push_str(" ORDER BY m.updated_at ASC, m.created_at ASC LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn supersede(
        &mut self,
        project_id: &str,
        old_id: &str,
        new_memory: NewMemory,
        reason: Option<&str>,
    ) -> Result<String> {
        self.ensure_memory_exists(project_id, old_id)?;
        validate_score("importance", new_memory.importance)?;
        validate_score("confidence", new_memory.confidence)?;
        let scope = normalize_memory_scope(&new_memory.scope)?;
        let memory_tier = normalize_memory_tier(&new_memory.memory_tier)?;
        let safety = inspect_memory_safety(
            &new_memory.body,
            &new_memory.tags,
            new_memory.source.as_deref(),
        );
        enforce_memory_safety(&safety, new_memory.allow_sensitive)?;
        let old_uuid = parse_uuid(old_id)?;
        let new_id = Uuid::now_v7();
        let body_hash = memory_body_hash(&new_memory.body);
        let tags = json_array(normalize_tags(new_memory.tags));
        let mut client = self.client()?;
        let mut tx = client.transaction()?;
        tx.execute(
            "INSERT INTO memories (
                id, project_id, scope, memory_tier, kind, body, body_hash, tags, source, status,
                importance, confidence, status_reason
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13)",
            &[
                &new_id,
                &project_id,
                &scope,
                &memory_tier,
                &new_memory.kind,
                &new_memory.body,
                &body_hash,
                &tags,
                &new_memory.source,
                &new_memory.status.as_str(),
                &new_memory.importance,
                &new_memory.confidence,
                &new_memory.status_reason,
            ],
        )?;
        tx.execute(
            "UPDATE memories
             SET status = 'superseded',
                 superseded_by = $1,
                 status_reason = $2,
                 updated_at = now()
             WHERE project_id = $3 AND id = $4",
            &[&new_id, &reason, &project_id, &old_uuid],
        )?;
        tx.commit()?;
        let mut client = self.client()?;
        let body_hash = memory_body_hash(&new_memory.body);
        sync_memory_fragments(
            &mut client,
            project_id,
            new_id,
            &body_hash,
            &new_memory.body,
        )?;
        Ok(new_id.to_string())
    }

    pub fn status(&self, project_id: &str) -> Result<ProjectStatus> {
        self.ensure_project(project_id)?;
        let project = self.project_record(project_id)?;
        Ok(ProjectStatus {
            project_id: project_id.to_string(),
            project_type: project.project_type,
            pending_memories: self.count_by_status(project_id, Some(MemoryStatus::Pending))?,
            active_memories: self.count_by_status(project_id, Some(MemoryStatus::Active))?,
            superseded_memories: self
                .count_by_status(project_id, Some(MemoryStatus::Superseded))?,
            archived_memories: self.count_by_status(project_id, Some(MemoryStatus::Archived))?,
            total_memories: self.count_by_status(project_id, None)?,
            memory_embeddings: self.count_memory_embeddings(project_id)?,
        })
    }

    pub fn project_profile(&self, project_id: &str) -> Result<ProjectRecord> {
        self.ensure_project(project_id)?;
        self.project_record(project_id)
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectRecord>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, name, root_path, project_type, description, domains,
                    ts(created_at), ts(updated_at)
             FROM projects
             ORDER BY updated_at DESC, name ASC, id ASC",
            &[],
        )?;
        rows.iter().map(project_record_from_row).collect()
    }

    pub fn update_project_profile(
        &self,
        project_id: &str,
        update: ProjectProfileUpdate,
    ) -> Result<ProjectRecord> {
        self.ensure_project(project_id)?;
        let current = self.project_record(project_id)?;
        let name = update.name.unwrap_or(current.name);
        let root_path = update.root_path.or(current.root_path);
        let project_type = update
            .project_type
            .as_deref()
            .map(normalize_project_type)
            .unwrap_or(current.project_type);
        let description = update.description.or(current.description);
        let domains = json_array(normalize_tags(update.domains.unwrap_or(current.domains)));
        let mut client = self.client()?;
        client.execute(
            "UPDATE projects
             SET name = $1, root_path = $2, project_type = $3, description = $4,
                 domains = $5::jsonb, updated_at = now()
             WHERE id = $6",
            &[
                &name,
                &root_path,
                &project_type,
                &description,
                &domains,
                &project_id,
            ],
        )?;
        drop(client);
        self.project_record(project_id)
    }

    pub fn upsert_memory_entity(
        &self,
        project_id: &str,
        entity_type: &str,
        name: &str,
        aliases: Vec<String>,
        description: Option<String>,
    ) -> Result<MemoryEntity> {
        self.ensure_project(project_id)?;
        let entity_type = normalize_project_type(entity_type);
        let name = name.trim();
        if name.is_empty() {
            bail!("entity name cannot be empty");
        }
        let aliases = json_array(normalize_tags(aliases));
        let id = Uuid::now_v7();
        let mut client = self.client()?;
        let row = client.query_one(
            "INSERT INTO memory_entities (
                id, project_id, entity_type, name, aliases, description
             )
             VALUES ($1, $2, $3, $4, $5::jsonb, $6)
             ON CONFLICT(project_id, entity_type, name) DO UPDATE SET
                aliases = excluded.aliases,
                description = COALESCE(excluded.description, memory_entities.description),
                updated_at = now()
             RETURNING id, project_id, entity_type, name, aliases, description,
                       ts(created_at), ts(updated_at)",
            &[
                &id,
                &project_id,
                &entity_type,
                &name,
                &aliases,
                &description,
            ],
        )?;
        memory_entity_from_row(&row)
    }

    pub fn add_memory_fact(
        &self,
        project_id: &str,
        entity_id: Option<&str>,
        memory_id: Option<&str>,
        predicate: &str,
        value: &str,
        confidence: f64,
    ) -> Result<MemoryFact> {
        self.ensure_project(project_id)?;
        validate_score("confidence", confidence)?;
        let entity_id = entity_id.map(parse_uuid).transpose()?;
        let memory_id = memory_id.map(parse_uuid).transpose()?;
        let predicate = predicate.trim();
        let value = value.trim();
        if predicate.is_empty() || value.is_empty() {
            bail!("fact predicate and value cannot be empty");
        }
        let id = Uuid::now_v7();
        let mut client = self.client()?;
        let row = client.query_one(
            "INSERT INTO memory_facts (
                id, project_id, entity_id, memory_id, predicate, value, confidence
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             RETURNING id, project_id, entity_id, memory_id, episode_id, predicate, value,
                       confidence, ts(valid_from), ts(valid_to), invalidated_by, ts(observed_at)",
            &[
                &id,
                &project_id,
                &entity_id,
                &memory_id,
                &predicate,
                &value,
                &confidence,
            ],
        )?;
        memory_fact_from_row(&row)
    }

    pub fn add_memory_fact_deduplicated(
        &self,
        project_id: &str,
        entity_id: Option<&str>,
        memory_id: Option<&str>,
        predicate: &str,
        value: &str,
        confidence: f64,
    ) -> Result<(MemoryFact, bool)> {
        self.ensure_project(project_id)?;
        validate_score("confidence", confidence)?;
        let entity_id = entity_id.map(parse_uuid).transpose()?;
        let memory_id = memory_id.map(parse_uuid).transpose()?;
        let predicate = predicate.trim();
        let value = value.trim();
        if predicate.is_empty() || value.is_empty() {
            bail!("fact predicate and value cannot be empty");
        }
        let mut client = self.client()?;
        if let Some(row) = client.query_opt(
            "SELECT id, project_id, entity_id, memory_id, episode_id, predicate, value,
                    confidence, ts(valid_from), ts(valid_to), invalidated_by, ts(observed_at)
             FROM memory_facts
             WHERE project_id = $1
               AND (($2::uuid IS NULL AND entity_id IS NULL) OR entity_id = $2)
               AND (($3::uuid IS NULL AND memory_id IS NULL) OR memory_id = $3)
               AND predicate = $4
               AND value = $5
             ORDER BY observed_at DESC
             LIMIT 1",
            &[&project_id, &entity_id, &memory_id, &predicate, &value],
        )? {
            return Ok((memory_fact_from_row(&row)?, false));
        }

        let id = Uuid::now_v7();
        let row = client.query_one(
            "INSERT INTO memory_facts (
                id, project_id, entity_id, memory_id, predicate, value, confidence
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             RETURNING id, project_id, entity_id, memory_id, episode_id, predicate, value,
                       confidence, ts(valid_from), ts(valid_to), invalidated_by, ts(observed_at)",
            &[
                &id,
                &project_id,
                &entity_id,
                &memory_id,
                &predicate,
                &value,
                &confidence,
            ],
        )?;
        Ok((memory_fact_from_row(&row)?, true))
    }

    pub fn add_memory_edge(
        &self,
        project_id: &str,
        from_entity_id: &str,
        to_entity_id: &str,
        relation_type: &str,
        memory_id: Option<&str>,
        confidence: f64,
    ) -> Result<MemoryEdge> {
        self.ensure_project(project_id)?;
        validate_score("confidence", confidence)?;
        let from_entity_id = parse_uuid(from_entity_id)?;
        let to_entity_id = parse_uuid(to_entity_id)?;
        let memory_id = memory_id.map(parse_uuid).transpose()?;
        let relation_type = relation_type.trim();
        if relation_type.is_empty() {
            bail!("edge relation_type cannot be empty");
        }
        let id = Uuid::now_v7();
        let mut client = self.client()?;
        let row = client.query_one(
            "WITH inserted AS (
                INSERT INTO memory_edges (
                    id, project_id, from_entity_id, to_entity_id, relation_type,
                    memory_id, confidence
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
             )
             SELECT e.id, e.project_id, e.from_entity_id, from_entity.name,
                    e.to_entity_id, to_entity.name, e.relation_type, e.memory_id,
                    e.episode_id, e.confidence, ts(e.valid_from), ts(e.valid_to),
                    e.invalidated_by, ts(e.observed_at)
             FROM inserted e
             JOIN memory_entities from_entity ON from_entity.id = e.from_entity_id
             JOIN memory_entities to_entity ON to_entity.id = e.to_entity_id",
            &[
                &id,
                &project_id,
                &from_entity_id,
                &to_entity_id,
                &relation_type,
                &memory_id,
                &confidence,
            ],
        )?;
        Ok(memory_edge_from_row(&row))
    }

    pub fn add_memory_edge_deduplicated(
        &self,
        project_id: &str,
        from_entity_id: &str,
        to_entity_id: &str,
        relation_type: &str,
        memory_id: Option<&str>,
        confidence: f64,
    ) -> Result<(MemoryEdge, bool)> {
        self.ensure_project(project_id)?;
        validate_score("confidence", confidence)?;
        let from_entity_id = parse_uuid(from_entity_id)?;
        let to_entity_id = parse_uuid(to_entity_id)?;
        let memory_id = memory_id.map(parse_uuid).transpose()?;
        let relation_type = relation_type.trim();
        if relation_type.is_empty() {
            bail!("edge relation_type cannot be empty");
        }
        let mut client = self.client()?;
        if let Some(row) = client.query_opt(
            "SELECT e.id, e.project_id, e.from_entity_id, from_entity.name,
                    e.to_entity_id, to_entity.name, e.relation_type, e.memory_id,
                    e.episode_id, e.confidence, ts(e.valid_from), ts(e.valid_to),
                    e.invalidated_by, ts(e.observed_at)
             FROM memory_edges e
             JOIN memory_entities from_entity ON from_entity.id = e.from_entity_id
             JOIN memory_entities to_entity ON to_entity.id = e.to_entity_id
             WHERE e.project_id = $1
               AND e.from_entity_id = $2
               AND e.to_entity_id = $3
               AND e.relation_type = $4
               AND (($5::uuid IS NULL AND e.memory_id IS NULL) OR e.memory_id = $5)
             ORDER BY e.observed_at DESC
             LIMIT 1",
            &[
                &project_id,
                &from_entity_id,
                &to_entity_id,
                &relation_type,
                &memory_id,
            ],
        )? {
            return Ok((memory_edge_from_row(&row), false));
        }

        let id = Uuid::now_v7();
        let row = client.query_one(
            "WITH inserted AS (
                INSERT INTO memory_edges (
                    id, project_id, from_entity_id, to_entity_id, relation_type,
                    memory_id, confidence
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
             )
             SELECT e.id, e.project_id, e.from_entity_id, from_entity.name,
                    e.to_entity_id, to_entity.name, e.relation_type, e.memory_id,
                    e.episode_id, e.confidence, ts(e.valid_from), ts(e.valid_to),
                    e.invalidated_by, ts(e.observed_at)
             FROM inserted e
             JOIN memory_entities from_entity ON from_entity.id = e.from_entity_id
             JOIN memory_entities to_entity ON to_entity.id = e.to_entity_id",
            &[
                &id,
                &project_id,
                &from_entity_id,
                &to_entity_id,
                &relation_type,
                &memory_id,
                &confidence,
            ],
        )?;
        Ok((memory_edge_from_row(&row), true))
    }

    pub fn invalidate_memory_fact(
        &self,
        project_id: &str,
        fact_id: &str,
        invalidated_by: Option<&str>,
        valid_to: Option<&str>,
    ) -> Result<MemoryFact> {
        self.ensure_project(project_id)?;
        let fact_id = parse_uuid(fact_id)?;
        let invalidated_by = invalidated_by
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(parse_uuid)
            .transpose()?
            .unwrap_or(fact_id);
        let valid_to = valid_to.map(str::trim).filter(|value| !value.is_empty());
        let mut client = self.client()?;
        let row = client.query_one(
            "UPDATE memory_facts
             SET invalidated_by = $3,
                 valid_to = COALESCE($4::text::timestamptz, valid_to, now())
             WHERE project_id = $1 AND id = $2
             RETURNING id, project_id, entity_id, memory_id, episode_id, predicate, value,
                       confidence, ts(valid_from), ts(valid_to), invalidated_by, ts(observed_at)",
            &[&project_id, &fact_id, &invalidated_by, &valid_to],
        )?;
        memory_fact_from_row(&row)
    }

    pub fn invalidate_memory_edge(
        &self,
        project_id: &str,
        edge_id: &str,
        invalidated_by: Option<&str>,
        valid_to: Option<&str>,
    ) -> Result<MemoryEdge> {
        self.ensure_project(project_id)?;
        let edge_id = parse_uuid(edge_id)?;
        let invalidated_by = invalidated_by
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(parse_uuid)
            .transpose()?
            .unwrap_or(edge_id);
        let valid_to = valid_to.map(str::trim).filter(|value| !value.is_empty());
        let mut client = self.client()?;
        let row = client.query_one(
            "WITH updated AS (
                UPDATE memory_edges
                SET invalidated_by = $3,
                    valid_to = COALESCE($4::text::timestamptz, valid_to, now())
                WHERE project_id = $1 AND id = $2
                RETURNING *
             )
             SELECT e.id, e.project_id, e.from_entity_id, from_entity.name,
                    e.to_entity_id, to_entity.name, e.relation_type, e.memory_id,
                    e.episode_id, e.confidence, ts(e.valid_from), ts(e.valid_to),
                    e.invalidated_by, ts(e.observed_at)
             FROM updated e
             JOIN memory_entities from_entity ON from_entity.id = e.from_entity_id
             JOIN memory_entities to_entity ON to_entity.id = e.to_entity_id",
            &[&project_id, &edge_id, &invalidated_by, &valid_to],
        )?;
        Ok(memory_edge_from_row(&row))
    }

    pub fn add_memory_episode(
        &self,
        project_id: &str,
        source: &str,
        summary: Option<&str>,
        raw_ref: Option<&str>,
        raw_payload: Value,
    ) -> Result<MemoryEpisode> {
        self.ensure_project(project_id)?;
        let source = source.trim();
        if source.is_empty() {
            bail!("episode source cannot be empty");
        }
        let id = Uuid::now_v7();
        let mut client = self.client()?;
        let row = client.query_one(
            "INSERT INTO memory_episodes (id, project_id, source, summary, raw_ref, raw_payload)
             VALUES ($1, $2, $3, $4, $5, $6::jsonb)
             RETURNING id, project_id, source, summary, raw_ref, raw_payload,
                       ts(observed_at), ts(created_at)",
            &[&id, &project_id, &source, &summary, &raw_ref, &raw_payload],
        )?;
        memory_episode_from_row(&row)
    }

    pub fn search_memory_episodes(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEpisode>> {
        self.ensure_project(project_id)?;
        let pattern = format!("%{}%", query.trim());
        let limit = limit.clamp(1, 100) as i64;
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, source, summary, raw_ref, raw_payload,
                    ts(observed_at), ts(created_at)
             FROM memory_episodes
             WHERE project_id = $1
               AND ($2 = '%%'
                    OR source ILIKE $2
                    OR summary ILIKE $2
                    OR raw_ref ILIKE $2
                    OR raw_payload::text ILIKE $2)
             ORDER BY observed_at DESC
             LIMIT $3",
            &[&project_id, &pattern, &limit],
        )?;
        rows.iter().map(memory_episode_from_row).collect()
    }

    pub fn search_memory_graph(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<MemoryGraph> {
        self.search_memory_graph_at(project_id, query, limit, None)
    }

    pub fn search_memory_graph_at(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
        as_of: Option<&str>,
    ) -> Result<MemoryGraph> {
        self.ensure_project(project_id)?;
        let pattern = format!("%{}%", query.trim());
        let limit = limit.clamp(1, 500) as i64;
        let as_of = as_of.map(str::trim).filter(|value| !value.is_empty());
        let mut client = self.client()?;
        let entities = client.query(
            "SELECT id, project_id, entity_type, name, aliases, description,
                    ts(created_at), ts(updated_at)
             FROM memory_entities
             WHERE project_id = $1
               AND ($2 = '%%' OR name ILIKE $2 OR description ILIKE $2)
             ORDER BY updated_at DESC
             LIMIT $3",
            &[&project_id, &pattern, &limit],
        )?;
        let facts = client.query(
            "SELECT id, project_id, entity_id, memory_id, episode_id, predicate, value,
                    confidence, ts(valid_from), ts(valid_to), invalidated_by, ts(observed_at)
             FROM memory_facts
             WHERE project_id = $1
               AND ($2 = '%%' OR predicate ILIKE $2 OR value ILIKE $2)
               AND invalidated_by IS NULL
               AND (
                    $4::text IS NULL OR
                    ((valid_from IS NULL OR valid_from <= $4::text::timestamptz)
                     AND (valid_to IS NULL OR valid_to > $4::text::timestamptz))
               )
             ORDER BY observed_at DESC
             LIMIT $3",
            &[&project_id, &pattern, &limit, &as_of],
        )?;
        let edges = client.query(
            "SELECT e.id, e.project_id, e.from_entity_id, from_entity.name,
                    e.to_entity_id, to_entity.name, e.relation_type, e.memory_id,
                    e.episode_id, e.confidence, ts(e.valid_from), ts(e.valid_to),
                    e.invalidated_by, ts(e.observed_at)
             FROM memory_edges e
             JOIN memory_entities from_entity ON from_entity.id = e.from_entity_id
             JOIN memory_entities to_entity ON to_entity.id = e.to_entity_id
             WHERE e.project_id = $1
               AND ($2 = '%%'
                    OR e.relation_type ILIKE $2
                    OR from_entity.name ILIKE $2
                    OR to_entity.name ILIKE $2)
               AND e.invalidated_by IS NULL
               AND (
                    $4::text IS NULL OR
                    ((e.valid_from IS NULL OR e.valid_from <= $4::text::timestamptz)
                     AND (e.valid_to IS NULL OR e.valid_to > $4::text::timestamptz))
               )
             ORDER BY e.observed_at DESC
             LIMIT $3",
            &[&project_id, &pattern, &limit, &as_of],
        )?;
        Ok(MemoryGraph {
            entities: entities
                .iter()
                .map(memory_entity_from_row)
                .collect::<Result<Vec<_>>>()?,
            facts: facts
                .iter()
                .map(memory_fact_from_row)
                .collect::<Result<Vec<_>>>()?,
            edges: edges.iter().map(memory_edge_from_row).collect(),
        })
    }

    pub fn memory_graph_for_memories(
        &self,
        project_id: &str,
        memory_ids: &[String],
        limit: usize,
    ) -> Result<MemoryGraph> {
        self.ensure_project(project_id)?;
        let memory_uuids = memory_ids
            .iter()
            .map(|id| parse_uuid(id))
            .collect::<Result<Vec<_>>>()?;
        if memory_uuids.is_empty() {
            return Ok(MemoryGraph {
                entities: Vec::new(),
                facts: Vec::new(),
                edges: Vec::new(),
            });
        }
        let limit = limit.clamp(1, 500) as i64;
        let mut client = self.client()?;
        let facts = client.query(
            "SELECT id, project_id, entity_id, memory_id, episode_id, predicate, value,
                    confidence, ts(valid_from), ts(valid_to), invalidated_by, ts(observed_at)
             FROM memory_facts
             WHERE project_id = $1
               AND memory_id = ANY($2)
               AND invalidated_by IS NULL
             ORDER BY observed_at DESC
             LIMIT $3",
            &[&project_id, &memory_uuids, &limit],
        )?;
        let edges = client.query(
            "SELECT e.id, e.project_id, e.from_entity_id, from_entity.name,
                    e.to_entity_id, to_entity.name, e.relation_type, e.memory_id,
                    e.episode_id, e.confidence, ts(e.valid_from), ts(e.valid_to),
                    e.invalidated_by, ts(e.observed_at)
             FROM memory_edges e
             JOIN memory_entities from_entity ON from_entity.id = e.from_entity_id
             JOIN memory_entities to_entity ON to_entity.id = e.to_entity_id
             WHERE e.project_id = $1
               AND e.memory_id = ANY($2)
               AND e.invalidated_by IS NULL
             ORDER BY e.observed_at DESC
             LIMIT $3",
            &[&project_id, &memory_uuids, &limit],
        )?;
        let entities = client.query(
            "WITH selected_entities AS (
                SELECT entity_id AS id
                FROM memory_facts
                WHERE project_id = $1
                  AND memory_id = ANY($2)
                  AND entity_id IS NOT NULL
                  AND invalidated_by IS NULL
                UNION
                SELECT from_entity_id AS id
                FROM memory_edges
                WHERE project_id = $1
                  AND memory_id = ANY($2)
                  AND invalidated_by IS NULL
                UNION
                SELECT to_entity_id AS id
                FROM memory_edges
                WHERE project_id = $1
                  AND memory_id = ANY($2)
                  AND invalidated_by IS NULL
             )
             SELECT e.id, e.project_id, e.entity_type, e.name, e.aliases, e.description,
                    ts(e.created_at), ts(e.updated_at)
             FROM memory_entities e
             JOIN selected_entities s ON s.id = e.id
             WHERE e.project_id = $1
             ORDER BY e.updated_at DESC
             LIMIT $3",
            &[&project_id, &memory_uuids, &(limit.saturating_mul(3))],
        )?;
        Ok(MemoryGraph {
            entities: entities
                .iter()
                .map(memory_entity_from_row)
                .collect::<Result<Vec<_>>>()?,
            facts: facts
                .iter()
                .map(memory_fact_from_row)
                .collect::<Result<Vec<_>>>()?,
            edges: edges.iter().map(memory_edge_from_row).collect(),
        })
    }

    pub fn health(&self) -> Result<StoreHealth> {
        let mut client = self.client()?;
        let schema = client
            .query_one("SELECT current_schema()", &[])?
            .get::<_, String>(0);
        let temp_schemas = client
            .query_one(
                "SELECT COUNT(*)::BIGINT
                 FROM information_schema.schemata
                 WHERE schema_name LIKE 'dm_%'",
                &[],
            )?
            .get::<_, i64>(0) as u64;
        let database_size_bytes = client
            .query_one("SELECT pg_database_size(current_database())::BIGINT", &[])?
            .get::<_, i64>(0) as u64;
        Ok(StoreHealth {
            schema,
            projects: count_all(&mut client, "projects")?,
            memories: count_all(&mut client, "memories")?,
            memory_embeddings: count_distinct_project_keys(
                &mut client,
                "memory_embeddings_4096",
                "memory_id",
            )?,
            code_files: count_all(&mut client, "code_files")?,
            code_symbols: count_all(&mut client, "code_symbols")?,
            code_relations: count_all(&mut client, "code_relations")?,
            code_symbol_embeddings: count_distinct_project_keys(
                &mut client,
                "code_symbol_embeddings_1024",
                "symbol_id",
            )?,
            memory_entities: count_all(&mut client, "memory_entities")?,
            memory_facts: count_all(&mut client, "memory_facts")?,
            memory_edges: count_all(&mut client, "memory_edges")?,
            audit_events: count_all(&mut client, "dukememory_audit_events")?,
            task_sessions: count_all(&mut client, "dukememory_task_sessions")?,
            temp_schemas,
            database_size_bytes,
        })
    }

    pub fn embedding_model_stats(&self, project_id: &str) -> Result<EmbeddingModelStats> {
        self.ensure_project(project_id)?;
        let mut client = self.client()?;
        let memory_rows = client.query(
            "SELECT model, embedding_kind, COUNT(DISTINCT memory_id)::BIGINT
             FROM memory_embeddings_4096
             WHERE project_id = $1
             GROUP BY model, embedding_kind
             ORDER BY model, embedding_kind",
            &[&project_id],
        )?;
        let code_rows = client.query(
            "SELECT model, embedding_kind, COUNT(DISTINCT symbol_id)::BIGINT
             FROM code_symbol_embeddings_1024
             WHERE project_id = $1
             GROUP BY model, embedding_kind
             ORDER BY model, embedding_kind",
            &[&project_id],
        )?;
        Ok(EmbeddingModelStats {
            memory_models: memory_rows
                .iter()
                .map(|row| EmbeddingModelCount {
                    model: row.get(0),
                    embedding_kind: row.get(1),
                    count: row.get::<_, i64>(2) as u64,
                })
                .collect(),
            code_models: code_rows
                .iter()
                .map(|row| EmbeddingModelCount {
                    model: row.get(0),
                    embedding_kind: row.get(1),
                    count: row.get::<_, i64>(2) as u64,
                })
                .collect(),
        })
    }

    pub fn cleanup_temp_schemas(&self, dry_run: bool) -> Result<CleanupSchemasReport> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT schema_name
             FROM information_schema.schemata
             WHERE schema_name LIKE 'dm_%'
             ORDER BY schema_name",
            &[],
        )?;
        let mut report = CleanupSchemasReport {
            dry_run,
            dropped: Vec::new(),
            kept: Vec::new(),
        };
        for row in rows {
            let schema = row.get::<_, String>(0);
            if dry_run {
                report.kept.push(schema);
            } else {
                let schema_ident = quote_ident(&schema)?;
                client.query(&format!("DROP SCHEMA {schema_ident} CASCADE"), &[])?;
                report.dropped.push(schema);
            }
        }
        Ok(report)
    }

    pub fn record_eval_run(&self, record: EvalRunRecord<'_>) -> Result<String> {
        self.ensure_project(record.project_id)?;
        let mode = record.mode.trim();
        if mode.is_empty() {
            bail!("eval mode cannot be empty");
        }
        let id = Uuid::now_v7();
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO eval_runs (
                id, suite_id, project_id, suite_name, suite_hash, mode,
                total_cases, passed_cases, failed_cases, detail
             )
             VALUES ($1, NULL, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)",
            &[
                &id,
                &record.project_id,
                &record.suite_name,
                &record.suite_hash,
                &mode,
                &(record.total_cases as i32),
                &(record.passed_cases as i32),
                &(record.failed_cases as i32),
                &record.detail,
            ],
        )?;
        Ok(id.to_string())
    }

    pub fn latest_eval_run_summary(
        &self,
        project_id: &str,
        suite_name: Option<&str>,
        suite_hash: Option<&str>,
        mode: &str,
        exclude_run_id: Option<&str>,
    ) -> Result<Option<EvalRunSummary>> {
        let exclude_run_id = match exclude_run_id {
            Some(id) => Some(parse_uuid(id)?),
            None => None,
        };
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, suite_name, suite_hash, mode,
                    total_cases, passed_cases, failed_cases, ts(created_at)
             FROM eval_runs
             WHERE project_id = $1
               AND (($2::text IS NULL AND suite_name IS NULL) OR suite_name = $2)
               AND (($3::text IS NULL AND suite_hash IS NULL) OR suite_hash = $3)
               AND mode = $4
               AND ($5::uuid IS NULL OR id <> $5)
             ORDER BY created_at DESC
             LIMIT 1",
            &[
                &project_id,
                &suite_name,
                &suite_hash,
                &mode,
                &exclude_run_id,
            ],
        )?;
        Ok(rows.first().map(eval_run_summary_from_row))
    }

    pub fn latest_eval_run_for_project(&self, project_id: &str) -> Result<Option<EvalRunSummary>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, suite_name, suite_hash, mode,
                    total_cases, passed_cases, failed_cases, ts(created_at)
             FROM eval_runs
             WHERE project_id = $1
             ORDER BY created_at DESC
             LIMIT 1",
            &[&project_id],
        )?;
        Ok(rows.first().map(eval_run_summary_from_row))
    }

    pub fn record_audit_event(
        &self,
        project_id: &str,
        actor: &str,
        action: &str,
        target_type: &str,
        target_id: Option<&str>,
        detail: Value,
    ) -> Result<String> {
        self.ensure_project(project_id)?;
        let actor = actor.trim();
        let action = action.trim();
        let target_type = target_type.trim();
        if actor.is_empty() || action.is_empty() || target_type.is_empty() {
            bail!("audit actor, action, and target_type cannot be empty");
        }
        let id = Uuid::now_v7();
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO dukememory_audit_events (
                id, project_id, actor, action, target_type, target_id, detail
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)",
            &[
                &id,
                &project_id,
                &actor,
                &action,
                &target_type,
                &target_id,
                &detail,
            ],
        )?;
        Ok(id.to_string())
    }

    pub fn record_eval_case(
        &self,
        project_id: &str,
        case: NewEvalCase<'_>,
    ) -> Result<EvalCaseRecord> {
        self.ensure_project(project_id)?;
        let suite_name = case.suite_name.trim();
        let name = case.name.trim();
        let query = case.query.trim();
        if suite_name.is_empty() || name.is_empty() || query.is_empty() {
            bail!("eval suite_name, name, and query cannot be empty");
        }
        let suite_id = Uuid::now_v7();
        let case_id = Uuid::now_v7();
        let min_results = case.min_results.map(|value| value as i32);
        let expected_contains = serde_json::to_value(&case.expected_contains)?;
        let forbidden_contains = serde_json::to_value(&case.forbidden_contains)?;
        let expected_ids = serde_json::to_value(&case.expected_ids)?;
        let forbidden_ids = serde_json::to_value(&case.forbidden_ids)?;
        let mut client = self.client()?;
        let row = client.query_one(
            "WITH suite AS (
                INSERT INTO eval_suites (id, project_id, name)
                VALUES ($1, $2, $3)
                ON CONFLICT(project_id, name) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
             ),
             inserted AS (
                INSERT INTO eval_cases (
                    id, suite_id, project_id, name, query, expected_contains,
                    forbidden_contains, min_results, expected_ids, forbidden_ids
                )
                SELECT $4, suite.id, $2, $5, $6, $7::jsonb, $8::jsonb, $9, $10::jsonb, $11::jsonb
                FROM suite
                RETURNING id, suite_id, project_id, name, query, expected_contains,
                          forbidden_contains, min_results, expected_ids, forbidden_ids,
                          ts(created_at)
             )
             SELECT * FROM inserted",
            &[
                &suite_id,
                &project_id,
                &suite_name,
                &case_id,
                &name,
                &query,
                &expected_contains,
                &forbidden_contains,
                &min_results,
                &expected_ids,
                &forbidden_ids,
            ],
        )?;
        eval_case_from_row(&row)
    }

    pub fn list_audit_events(&self, project_id: &str, limit: usize) -> Result<Vec<AuditEvent>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, actor, action, target_type, target_id, detail, ts(created_at)
             FROM dukememory_audit_events
             WHERE project_id = $1
             ORDER BY created_at DESC
             LIMIT $2",
            &[&project_id, &(limit.clamp(1, 500) as i64)],
        )?;
        Ok(rows.iter().map(audit_event_from_row).collect())
    }

    pub fn record_retrieval_event(&self, event: RetrievalEventRecord<'_>) -> Result<String> {
        self.ensure_project(event.project_id)?;
        let id = Uuid::now_v7();
        let task_session_id = event.task_session_id.map(parse_uuid).transpose()?;
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO dukememory_retrieval_events (
                id, project_id, task_session_id, tool, query, task_type, token_budget, estimated_tokens,
                latency_ms, memory_fragments, code_hits, graph_items, code_memories, plan, audit
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15::jsonb)",
            &[
                &id,
                &event.project_id,
                &task_session_id,
                &event.tool,
                &event.query,
                &event.task_type,
                &(event.token_budget as i32),
                &(event.estimated_tokens as i32),
                &(event.latency_ms as i64),
                &(event.memory_fragments as i32),
                &(event.code_hits as i32),
                &(event.graph_items as i32),
                &(event.code_memories as i32),
                &event.plan,
                &event.audit,
            ],
        )?;
        Ok(id.to_string())
    }

    pub fn get_retrieval_event(
        &self,
        project_id: &str,
        event_id: &str,
    ) -> Result<Option<RetrievalEvent>> {
        let event_uuid = parse_uuid(event_id)?;
        let mut client = self.client()?;
        client
            .query_opt(
                "SELECT id, project_id, task_session_id, tool, query, task_type, token_budget, estimated_tokens,
                        latency_ms, memory_fragments, code_hits, graph_items, code_memories,
                        plan, audit, ts(created_at)
                 FROM dukememory_retrieval_events
                 WHERE project_id = $1 AND id = $2",
                &[&project_id, &event_uuid],
            )?
            .map(|row| retrieval_event_from_row(&row))
            .transpose()
    }

    pub fn list_retrieval_events(
        &self,
        project_id: &str,
        query: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RetrievalEvent>> {
        self.ensure_project(project_id)?;
        let mut client = self.client()?;
        let rows = if let Some(query) = query.filter(|value| !value.trim().is_empty()) {
            client.query(
                "SELECT id, project_id, task_session_id, tool, query, task_type, token_budget, estimated_tokens,
                        latency_ms, memory_fragments, code_hits, graph_items, code_memories,
                        plan, audit, ts(created_at)
                 FROM dukememory_retrieval_events
                 WHERE project_id = $1
                   AND query ILIKE '%' || $2 || '%'
                 ORDER BY created_at DESC
                 LIMIT $3",
                &[&project_id, &query, &(limit.clamp(1, 500) as i64)],
            )?
        } else {
            client.query(
                "SELECT id, project_id, task_session_id, tool, query, task_type, token_budget, estimated_tokens,
                        latency_ms, memory_fragments, code_hits, graph_items, code_memories,
                        plan, audit, ts(created_at)
                 FROM dukememory_retrieval_events
                 WHERE project_id = $1
                 ORDER BY created_at DESC
                 LIMIT $2",
                &[&project_id, &(limit.clamp(1, 500) as i64)],
            )?
        };
        rows.iter().map(retrieval_event_from_row).collect()
    }

    #[allow(dead_code)]
    pub fn list_retrieval_events_for_session(
        &self,
        project_id: &str,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalEvent>> {
        self.ensure_project(project_id)?;
        let session_uuid = parse_uuid(session_id)?;
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, task_session_id, tool, query, task_type, token_budget, estimated_tokens,
                    latency_ms, memory_fragments, code_hits, graph_items, code_memories,
                    plan, audit, ts(created_at)
             FROM dukememory_retrieval_events
             WHERE project_id = $1 AND task_session_id = $2
             ORDER BY created_at DESC
             LIMIT $3",
            &[&project_id, &session_uuid, &(limit.clamp(1, 500) as i64)],
        )?;
        rows.iter().map(retrieval_event_from_row).collect()
    }

    pub fn apply_memory_feedback(
        &self,
        project_id: &str,
        helpful_ids: &[String],
        unhelpful_ids: &[String],
    ) -> Result<FeedbackEffect> {
        let helpful = helpful_ids
            .iter()
            .filter_map(|id| parse_uuid(id).ok())
            .collect::<Vec<_>>();
        let unhelpful = unhelpful_ids
            .iter()
            .filter_map(|id| parse_uuid(id).ok())
            .collect::<Vec<_>>();
        let mut client = self.client()?;
        let helpful_memories_updated = if helpful.is_empty() {
            0
        } else {
            client.execute(
                "UPDATE memories
                 SET usage_count = usage_count + 1,
                     last_used_at = now(),
                     quality_score = LEAST(1.0, quality_score + 0.08),
                     contradiction_risk = GREATEST(0.0, contradiction_risk - 0.03)
                 WHERE project_id = $1 AND id = ANY($2)",
                &[&project_id, &helpful],
            )? as usize
        };
        let unhelpful_memories_updated = if unhelpful.is_empty() {
            0
        } else {
            client.execute(
                "UPDATE memories
                 SET usage_count = usage_count + 1,
                     last_used_at = now(),
                     quality_score = GREATEST(0.0, quality_score - 0.12),
                     contradiction_risk = LEAST(1.0, contradiction_risk + 0.05)
                 WHERE project_id = $1 AND id = ANY($2)",
                &[&project_id, &unhelpful],
            )? as usize
        };
        Ok(FeedbackEffect {
            helpful_memories_updated,
            unhelpful_memories_updated,
        })
    }

    pub fn create_task_session(&self, session: NewTaskSession<'_>) -> Result<TaskSession> {
        self.ensure_project(session.project_id)?;
        validate_task_session_status(session.status)?;
        validate_task_session_phase(session.phase)?;
        let id = Uuid::now_v7();
        let progress = session.progress.clamp(0, 100) as i32;
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO dukememory_task_sessions (
                id, project_id, query, status, phase, progress, result
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)",
            &[
                &id,
                &session.project_id,
                &session.query,
                &session.status,
                &session.phase,
                &progress,
                &session.result,
            ],
        )?;
        drop(client);
        self.get_task_session(session.project_id, &id.to_string())?
            .ok_or_else(|| anyhow!("created task session `{id}` was not found"))
    }

    pub fn update_task_session(
        &self,
        project_id: &str,
        session_id: &str,
        update: TaskSessionUpdate,
    ) -> Result<TaskSession> {
        self.ensure_project(project_id)?;
        let current = self
            .get_task_session(project_id, session_id)?
            .ok_or_else(|| anyhow!("task session `{session_id}` was not found"))?;
        let status = update.status.unwrap_or(current.status);
        validate_task_session_status(&status)?;
        let phase = update.phase.unwrap_or(current.phase);
        validate_task_session_phase(&phase)?;
        let progress = update.progress.unwrap_or(current.progress).clamp(0, 100) as i32;
        let memory_ids = json_array(update.memory_ids.unwrap_or(current.memory_ids));
        let code_symbol_ids = json_array(update.code_symbol_ids.unwrap_or(current.code_symbol_ids));
        let file_paths = json_array(update.file_paths.unwrap_or(current.file_paths));
        let test_paths = json_array(update.test_paths.unwrap_or(current.test_paths));
        let summary = update.summary.unwrap_or(current.summary);
        let result = update.result.unwrap_or(current.result);
        let session_uuid = parse_uuid(session_id)?;
        let completed = matches!(status.as_str(), "completed" | "failed" | "archived");
        let mut client = self.client()?;
        client.execute(
            "UPDATE dukememory_task_sessions
             SET status = $1,
                 phase = $2,
                 progress = $3,
                 memory_ids = $4::jsonb,
                 code_symbol_ids = $5::jsonb,
                 file_paths = $6::jsonb,
                 test_paths = $7::jsonb,
                 summary = $8,
                 result = $9::jsonb,
                 updated_at = now(),
                 completed_at = CASE
                    WHEN $10 THEN COALESCE(completed_at, now())
                    ELSE NULL
                 END
             WHERE project_id = $11 AND id = $12",
            &[
                &status,
                &phase,
                &progress,
                &memory_ids,
                &code_symbol_ids,
                &file_paths,
                &test_paths,
                &summary,
                &result,
                &completed,
                &project_id,
                &session_uuid,
            ],
        )?;
        drop(client);
        self.get_task_session(project_id, session_id)?
            .ok_or_else(|| anyhow!("updated task session `{session_id}` was not found"))
    }

    pub fn get_task_session(
        &self,
        project_id: &str,
        session_id: &str,
    ) -> Result<Option<TaskSession>> {
        let session_uuid = parse_uuid(session_id)?;
        let mut client = self.client()?;
        let row = client.query_opt(
            "SELECT id, project_id, query, status, phase, progress,
                    memory_ids, code_symbol_ids, file_paths, test_paths,
                    summary, result, ts(created_at), ts(updated_at),
                    CASE WHEN completed_at IS NULL THEN NULL ELSE ts(completed_at) END
             FROM dukememory_task_sessions
             WHERE project_id = $1 AND id = $2",
            &[&project_id, &session_uuid],
        )?;
        row.map(|row| task_session_from_row(&row)).transpose()
    }

    pub fn list_task_sessions(
        &self,
        project_id: &str,
        status: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TaskSession>> {
        self.ensure_project(project_id)?;
        let mut client = self.client()?;
        let rows = if let Some(status) = status {
            validate_task_session_status(status)?;
            client.query(
                "SELECT id, project_id, query, status, phase, progress,
                        memory_ids, code_symbol_ids, file_paths, test_paths,
                        summary, result, ts(created_at), ts(updated_at),
                        CASE WHEN completed_at IS NULL THEN NULL ELSE ts(completed_at) END
                 FROM dukememory_task_sessions
                 WHERE project_id = $1 AND status = $2
                 ORDER BY updated_at DESC
                 LIMIT $3",
                &[&project_id, &status, &(limit.clamp(1, 500) as i64)],
            )?
        } else {
            client.query(
                "SELECT id, project_id, query, status, phase, progress,
                        memory_ids, code_symbol_ids, file_paths, test_paths,
                        summary, result, ts(created_at), ts(updated_at),
                        CASE WHEN completed_at IS NULL THEN NULL ELSE ts(completed_at) END
                 FROM dukememory_task_sessions
                 WHERE project_id = $1
                 ORDER BY updated_at DESC
                 LIMIT $2",
                &[&project_id, &(limit.clamp(1, 500) as i64)],
            )?
        };
        rows.iter().map(task_session_from_row).collect()
    }

    pub fn mark_memories_used(&self, project_id: &str, memory_ids: &[String]) -> Result<usize> {
        if memory_ids.is_empty() {
            return Ok(0);
        }
        let ids = memory_ids
            .iter()
            .filter_map(|id| parse_uuid(id).ok())
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Ok(0);
        }
        let mut client = self.client()?;
        Ok(client.execute(
            "UPDATE memories
             SET usage_count = usage_count + 1,
                 last_used_at = now(),
                 quality_score = LEAST(
                    1.0,
                    GREATEST(quality_score, (importance * 0.55) + (confidence * 0.35) + 0.10)
                 )
             WHERE project_id = $1 AND id = ANY($2)",
            &[&project_id, &ids],
        )? as usize)
    }

    pub fn mark_code_memories_used(
        &self,
        project_id: &str,
        memory_ids: &[String],
    ) -> Result<usize> {
        if memory_ids.is_empty() {
            return Ok(0);
        }
        let ids = memory_ids
            .iter()
            .filter_map(|id| parse_uuid(id).ok())
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Ok(0);
        }
        let mut client = self.client()?;
        Ok(client.execute(
            "UPDATE code_memories
             SET usage_count = usage_count + 1,
                 last_used_at = now(),
                 quality_score = LEAST(1.0, GREATEST(quality_score, confidence + 0.05))
             WHERE project_id = $1 AND id = ANY($2)",
            &[&project_id, &ids],
        )? as usize)
    }

    #[allow(dead_code)]
    pub fn set_memory_embedding(
        &self,
        project_id: &str,
        memory_id: &str,
        model: &str,
        embedding: &[f32],
    ) -> Result<()> {
        self.set_memory_embedding_with_metadata(
            project_id, memory_id, model, "body", None, embedding,
        )
    }

    pub fn set_memory_embedding_with_metadata(
        &self,
        project_id: &str,
        memory_id: &str,
        model: &str,
        embedding_kind: &str,
        content_hash: Option<&str>,
        embedding: &[f32],
    ) -> Result<()> {
        self.ensure_memory_exists(project_id, memory_id)?;
        if embedding.len() != 4096 {
            bail!("memory embedding dimension must be 4096 for PostgreSQL pgvector storage");
        }
        let embedding_kind = normalize_embedding_kind(embedding_kind)?;
        let memory_id = parse_uuid(memory_id)?;
        let vector = Vector::from(embedding.to_vec());
        let dimensions = embedding.len() as i32;
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO memory_embeddings_4096 (
                memory_id, project_id, model, embedding_kind, content_hash, dimensions, embedding,
                created_at, updated_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, now(), now())
             ON CONFLICT(memory_id, model, embedding_kind) DO UPDATE SET
                project_id = excluded.project_id,
                content_hash = excluded.content_hash,
                dimensions = excluded.dimensions,
                embedding = excluded.embedding,
                updated_at = now()",
            &[
                &memory_id,
                &project_id,
                &model,
                &embedding_kind,
                &content_hash,
                &dimensions,
                &vector,
            ],
        )?;
        Ok(())
    }

    pub fn search_memory_vectors(
        &self,
        project_id: &str,
        options: VectorSearchOptions,
    ) -> Result<Vec<Memory>> {
        if options.embedding.is_empty() {
            return Ok(Vec::new());
        }
        if options.embedding.len() != 4096 {
            bail!(
                "memory embedding dimension mismatch for pgvector search: expected 4096, got {}",
                options.embedding.len()
            );
        }
        let vector = Vector::from(options.embedding);
        let mut sql = memory_select_sql("e.score");
        sql.push_str(
            " JOIN (
                SELECT memory_id,
                       MIN(embedding <=> $3) AS distance,
                       MAX(1.0 - (embedding <=> $3)::double precision) AS score
                FROM memory_embeddings_4096
                WHERE project_id = $1 AND model = $2
                GROUP BY memory_id
              ) e ON e.memory_id = m.id
              WHERE m.project_id = $1",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![
            Box::new(project_id.to_string()),
            Box::new(options.embedding_model),
            Box::new(vector),
        ];
        append_pg_filters(&mut sql, &mut params, options.status, options.kind);
        append_memory_tier_filter(&mut sql, &mut params, options.memory_tier)?;
        push_param(&mut sql, &mut params, options.limit.clamp(1, 50) as i64);
        sql.push_str(" ORDER BY e.distance LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn search_related_memories(
        &self,
        project_id: &str,
        memory_id: &str,
        options: MemorySimilarityPairOptions,
    ) -> Result<Vec<Memory>> {
        let memory_id = parse_uuid(memory_id)?;
        let mut sql = memory_select_sql("1.0 - (e.embedding <=> seed.embedding)::double precision");
        sql.push_str(
            " JOIN memory_embeddings_4096 e ON e.memory_id = m.id
              JOIN memory_embeddings_4096 seed ON seed.memory_id = $3
             WHERE e.project_id = $1
               AND e.model = $2
               AND e.embedding_kind = 'body'
               AND seed.project_id = $1
               AND seed.model = $2
               AND seed.embedding_kind = 'body'
               AND m.id <> $3
               AND (1.0 - (e.embedding <=> seed.embedding)::double precision) >= $4",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![
            Box::new(project_id.to_string()),
            Box::new(options.embedding_model),
            Box::new(memory_id),
            Box::new(options.min_similarity),
        ];
        append_pg_filters(&mut sql, &mut params, options.status, options.kind);
        append_memory_tier_filter(&mut sql, &mut params, options.memory_tier)?;
        push_param(&mut sql, &mut params, options.limit.clamp(1, 100) as i64);
        sql.push_str(" ORDER BY e.embedding <=> seed.embedding LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn memory_similarity_pairs(
        &self,
        project_id: &str,
        options: MemorySimilarityPairOptions,
    ) -> Result<Vec<MemorySimilarityPair>> {
        self.ensure_project(project_id)?;
        let mut sql = String::from(
            "SELECT l.memory_id, r.memory_id,
                    1.0 - (l.embedding <=> r.embedding)::double precision AS similarity
             FROM memory_embeddings_4096 l
             JOIN memory_embeddings_4096 r
               ON r.project_id = l.project_id
              AND r.model = l.model
              AND r.embedding_kind = l.embedding_kind
              AND r.memory_id > l.memory_id
             JOIN memories m ON m.id = l.memory_id
             JOIN memories mr ON mr.id = r.memory_id
             WHERE l.project_id = $1
               AND l.model = $2
               AND l.embedding_kind = 'body'
               AND (1.0 - (l.embedding <=> r.embedding)::double precision) >= $3",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![
            Box::new(project_id.to_string()),
            Box::new(options.embedding_model),
            Box::new(options.min_similarity),
        ];
        append_pg_filters(&mut sql, &mut params, options.status, options.kind.clone());
        if let StatusFilter::One(status) = options.status {
            push_param(&mut sql, &mut params, status.as_str().to_string());
            sql.push_str(" AND mr.status = $");
            sql.push_str(&params.len().to_string());
        }
        if let Some(kind) = options.kind {
            push_param(&mut sql, &mut params, kind);
            sql.push_str(" AND mr.kind = $");
            sql.push_str(&params.len().to_string());
        }
        append_memory_tier_filter(&mut sql, &mut params, options.memory_tier.clone())?;
        if let Some(memory_tier) = options.memory_tier {
            push_param(&mut sql, &mut params, normalize_memory_tier(&memory_tier)?);
            sql.push_str(" AND mr.memory_tier = $");
            sql.push_str(&params.len().to_string());
        }
        push_param(&mut sql, &mut params, options.limit.clamp(1, 500) as i64);
        sql.push_str(" ORDER BY similarity DESC, l.memory_id, r.memory_id LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        let mut pairs = Vec::with_capacity(rows.len());
        for row in rows {
            let left_id = row.get::<_, Uuid>(0).to_string();
            let right_id = row.get::<_, Uuid>(1).to_string();
            let Some(left) = self.get(project_id, &left_id)? else {
                continue;
            };
            let Some(right) = self.get(project_id, &right_id)? else {
                continue;
            };
            pairs.push(MemorySimilarityPair {
                left,
                right,
                similarity: row.get(2),
            });
        }
        Ok(pairs)
    }

    pub fn memory_pair_similarities(
        &self,
        project_id: &str,
        memory_ids: &[String],
        model: &str,
    ) -> Result<HashMap<(String, String), f64>> {
        if memory_ids.len() < 2 {
            return Ok(HashMap::new());
        }
        let ids = memory_ids
            .iter()
            .map(|id| parse_uuid(id))
            .collect::<Result<Vec<_>>>()?;
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT l.memory_id, r.memory_id,
                    1.0 - (l.embedding <=> r.embedding)::double precision AS similarity
             FROM memory_embeddings_4096 l
             JOIN memory_embeddings_4096 r
               ON r.project_id = l.project_id
              AND r.model = l.model
              AND r.embedding_kind = l.embedding_kind
              AND r.memory_id <> l.memory_id
             WHERE l.project_id = $1
               AND l.model = $2
               AND l.embedding_kind = 'body'
               AND l.memory_id = ANY($3)
               AND r.memory_id = ANY($3)",
            &[&project_id, &model, &ids],
        )?;
        let mut similarities = HashMap::new();
        for row in rows {
            let left = row.get::<_, Uuid>(0).to_string();
            let right = row.get::<_, Uuid>(1).to_string();
            similarities.insert((left, right), row.get(2));
        }
        Ok(similarities)
    }

    pub fn memories_missing_embeddings(
        &self,
        project_id: &str,
        model: &str,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        let sql = format!(
            "{} WHERE m.project_id = $1
               AND (
                    SELECT COUNT(DISTINCT e.embedding_kind)
                    FROM memory_embeddings_4096 e
                    WHERE e.memory_id = m.id
                      AND e.project_id = $1
                      AND e.model = $2
                      AND e.embedding_kind = ANY($4)
               ) < cardinality($4::text[])
             ORDER BY m.updated_at DESC, m.created_at DESC LIMIT $3",
            memory_select_sql("NULL::double precision")
        );
        let mut client = self.client()?;
        let required_kinds = required_memory_embedding_kinds();
        let rows = client.query(
            &sql,
            &[
                &project_id,
                &model,
                &(limit.clamp(1, 500) as i64),
                &required_kinds,
            ],
        )?;
        rows.iter().map(memory_from_row).collect::<Result<Vec<_>>>()
    }

    pub fn clear_code_index(&mut self, project_id: &str) -> Result<()> {
        let mut client = self.client()?;
        let mut tx = client.transaction()?;
        tx.execute(
            "DELETE FROM code_relations WHERE project_id = $1",
            &[&project_id],
        )?;
        tx.execute(
            "DELETE FROM code_symbol_embeddings_1024 WHERE project_id = $1",
            &[&project_id],
        )?;
        tx.execute(
            "DELETE FROM code_symbols WHERE project_id = $1",
            &[&project_id],
        )?;
        tx.execute(
            "DELETE FROM code_files WHERE project_id = $1",
            &[&project_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn code_file_hashes(&self, project_id: &str) -> Result<HashMap<String, String>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT path, hash FROM code_files WHERE project_id = $1",
            &[&project_id],
        )?;
        Ok(rows
            .into_iter()
            .map(|row| (row.get::<_, String>(0), row.get::<_, String>(1)))
            .collect())
    }

    pub fn remove_code_file(&mut self, project_id: &str, file_path: &str) -> Result<()> {
        let mut client = self.client()?;
        let mut tx = client.transaction()?;
        tx.execute(
            "DELETE FROM code_relations WHERE project_id = $1 AND from_file_path = $2",
            &[&project_id, &file_path],
        )?;
        tx.execute(
            "DELETE FROM code_relations
             WHERE project_id = $1
               AND (from_symbol_id IN (SELECT id FROM code_symbols WHERE project_id = $1 AND file_path = $2)
                    OR target_symbol_id IN (SELECT id FROM code_symbols WHERE project_id = $1 AND file_path = $2))",
            &[&project_id, &file_path],
        )?;
        tx.execute(
            "DELETE FROM code_symbol_embeddings_1024
             WHERE project_id = $1
               AND symbol_id IN (SELECT id FROM code_symbols WHERE project_id = $1 AND file_path = $2)",
            &[&project_id, &file_path],
        )?;
        tx.execute(
            "DELETE FROM code_symbols WHERE project_id = $1 AND file_path = $2",
            &[&project_id, &file_path],
        )?;
        tx.execute(
            "DELETE FROM code_files WHERE project_id = $1 AND path = $2",
            &[&project_id, &file_path],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_code_file(&self, file: CodeFile) -> Result<()> {
        self.ensure_project(&file.project_id)?;
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO code_files (
                project_id, path, language, hash, size_bytes, line_count, indexed_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, now())
             ON CONFLICT(project_id, path) DO UPDATE SET
                language = excluded.language,
                hash = excluded.hash,
                size_bytes = excluded.size_bytes,
                line_count = excluded.line_count,
                indexed_at = now()",
            &[
                &file.project_id,
                &file.path,
                &file.language,
                &file.hash,
                &(file.size_bytes as i64),
                &(file.line_count as i32),
            ],
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub fn upsert_code_symbol(&self, symbol: CodeSymbol) -> Result<()> {
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO code_symbols (
                id, project_id, file_path, language, name, kind, signature,
                body, start_line, end_line, parent_id, indexed_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
             ON CONFLICT(id) DO UPDATE SET
                file_path = excluded.file_path,
                language = excluded.language,
                name = excluded.name,
                kind = excluded.kind,
                signature = excluded.signature,
                body = excluded.body,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                parent_id = excluded.parent_id,
                indexed_at = now()",
            &[
                &symbol.id,
                &symbol.project_id,
                &symbol.file_path,
                &symbol.language,
                &symbol.name,
                &symbol.kind,
                &symbol.signature,
                &symbol.body,
                &(symbol.start_line as i32),
                &(symbol.end_line as i32),
                &symbol.parent_id,
            ],
        )?;
        Ok(())
    }

    pub fn insert_code_relation(&self, relation: CodeRelation) -> Result<usize> {
        let mut client = self.client()?;
        let inserted = client.execute(
            "INSERT INTO code_relations (
                id, project_id, from_symbol_id, from_file_path,
                relation_kind, target_name, target_symbol_id
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             ON CONFLICT(id) DO NOTHING",
            &[
                &relation.id,
                &relation.project_id,
                &relation.from_symbol_id,
                &relation.from_file_path,
                &relation.relation_kind,
                &relation.target_name,
                &relation.target_symbol_id,
            ],
        )?;
        Ok(inserted as usize)
    }

    pub fn upsert_code_file_index(
        &self,
        file: CodeFile,
        symbols: Vec<CodeSymbol>,
        relations: Vec<CodeRelation>,
        replace_existing: bool,
    ) -> Result<usize> {
        self.ensure_project(&file.project_id)?;
        let mut client = self.client()?;
        let mut tx = client.transaction()?;
        if replace_existing {
            tx.execute(
                "DELETE FROM code_relations
                 WHERE project_id = $1
                   AND (from_file_path = $2
                        OR from_symbol_id IN (SELECT id FROM code_symbols WHERE project_id = $1 AND file_path = $2)
                        OR target_symbol_id IN (SELECT id FROM code_symbols WHERE project_id = $1 AND file_path = $2))",
                &[&file.project_id, &file.path],
            )?;
            tx.execute(
                "DELETE FROM code_symbol_embeddings_1024
                 WHERE project_id = $1
                   AND symbol_id IN (SELECT id FROM code_symbols WHERE project_id = $1 AND file_path = $2)",
                &[&file.project_id, &file.path],
            )?;
            tx.execute(
                "DELETE FROM code_symbols WHERE project_id = $1 AND file_path = $2",
                &[&file.project_id, &file.path],
            )?;
        }
        tx.execute(
            "INSERT INTO code_files (
                project_id, path, language, hash, size_bytes, line_count, indexed_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, now())
             ON CONFLICT(project_id, path) DO UPDATE SET
                language = excluded.language,
                hash = excluded.hash,
                size_bytes = excluded.size_bytes,
                line_count = excluded.line_count,
                indexed_at = now()",
            &[
                &file.project_id,
                &file.path,
                &file.language,
                &file.hash,
                &(file.size_bytes as i64),
                &(file.line_count as i32),
            ],
        )?;
        for symbol in symbols {
            tx.execute(
                "INSERT INTO code_symbols (
                    id, project_id, file_path, language, name, kind, signature,
                    body, start_line, end_line, parent_id, indexed_at
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
                 ON CONFLICT(id) DO UPDATE SET
                    file_path = excluded.file_path,
                    language = excluded.language,
                    name = excluded.name,
                    kind = excluded.kind,
                    signature = excluded.signature,
                    body = excluded.body,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    parent_id = excluded.parent_id,
                    indexed_at = now()",
                &[
                    &symbol.id,
                    &symbol.project_id,
                    &symbol.file_path,
                    &symbol.language,
                    &symbol.name,
                    &symbol.kind,
                    &symbol.signature,
                    &symbol.body,
                    &(symbol.start_line as i32),
                    &(symbol.end_line as i32),
                    &symbol.parent_id,
                ],
            )?;
        }
        let mut inserted_relations = 0;
        for relation in relations {
            inserted_relations += tx.execute(
                "INSERT INTO code_relations (
                    id, project_id, from_symbol_id, from_file_path,
                    relation_kind, target_name, target_symbol_id
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, $7)
                 ON CONFLICT(id) DO NOTHING",
                &[
                    &relation.id,
                    &relation.project_id,
                    &relation.from_symbol_id,
                    &relation.from_file_path,
                    &relation.relation_kind,
                    &relation.target_name,
                    &relation.target_symbol_id,
                ],
            )? as usize;
        }
        tx.commit()?;
        Ok(inserted_relations)
    }

    pub fn remove_code_relations_by_kind(
        &self,
        project_id: &str,
        relation_kind: &str,
    ) -> Result<usize> {
        let mut client = self.client()?;
        Ok(client.execute(
            "DELETE FROM code_relations WHERE project_id = $1 AND relation_kind = $2",
            &[&project_id, &relation_kind],
        )? as usize)
    }

    pub fn resolve_code_relation_targets(
        &self,
        project_id: &str,
    ) -> Result<CodeRelationResolutionReport> {
        let symbols = self.all_code_symbols(project_id)?;
        let relations = self.unresolved_resolvable_code_relations(project_id)?;
        let resolver = SymbolResolver::new(symbols);
        let mut client = self.client()?;
        let targets_reset = client.execute(
            "UPDATE code_relations
             SET target_symbol_id = NULL
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'uses', 'declares_module')
               AND target_symbol_id IS NOT NULL",
            &[&project_id],
        )? as usize;

        let mut report = CodeRelationResolutionReport {
            project_id: project_id.to_string(),
            targets_reset,
            calls_resolved: 0,
            uses_resolved: 0,
            modules_resolved: 0,
        };

        for relation in relations {
            let resolved = match relation.relation_kind.as_str() {
                "calls" => resolver.resolve_call(&relation.from_file_path, &relation.target_name),
                "uses" => resolver.resolve_use(&relation.target_name),
                "declares_module" => resolver
                    .resolve_module_declaration(&relation.from_file_path, &relation.target_name),
                _ => None,
            };
            let Some(target_symbol_id) = resolved else {
                continue;
            };
            client.execute(
                "UPDATE code_relations SET target_symbol_id = $1 WHERE project_id = $2 AND id = $3",
                &[&target_symbol_id, &project_id, &relation.id],
            )?;
            match relation.relation_kind.as_str() {
                "calls" => report.calls_resolved += 1,
                "uses" => report.uses_resolved += 1,
                "declares_module" => report.modules_resolved += 1,
                _ => {}
            }
        }
        Ok(report)
    }

    #[allow(dead_code)]
    pub fn set_code_symbol_embedding_with_cache(
        &self,
        project_id: &str,
        symbol_id: &str,
        model: &str,
        content_hash: &str,
        embedding: &[f32],
    ) -> Result<()> {
        self.set_code_symbol_embedding_kind_with_cache(
            project_id,
            symbol_id,
            model,
            "body",
            content_hash,
            embedding,
        )?;
        self.set_code_symbol_embedding_kind_with_cache(
            project_id,
            symbol_id,
            model,
            "signature",
            content_hash,
            embedding,
        )
    }

    pub fn set_code_symbol_embedding_kind_with_cache(
        &self,
        project_id: &str,
        symbol_id: &str,
        model: &str,
        embedding_kind: &str,
        content_hash: &str,
        embedding: &[f32],
    ) -> Result<()> {
        self.set_code_symbol_embedding_inner(
            project_id,
            symbol_id,
            model,
            embedding_kind,
            embedding,
            Some(content_hash),
        )
    }

    fn set_code_symbol_embedding_inner(
        &self,
        project_id: &str,
        symbol_id: &str,
        model: &str,
        embedding_kind: &str,
        embedding: &[f32],
        content_hash: Option<&str>,
    ) -> Result<()> {
        if embedding.len() != 1024 {
            bail!("code symbol embedding dimension must be 1024 for PostgreSQL pgvector storage");
        }
        let embedding_kind = normalize_embedding_kind(embedding_kind)?;
        let mut client = self.client()?;
        let exists = client.query_opt(
            "SELECT 1 FROM code_symbols WHERE project_id = $1 AND id = $2",
            &[&project_id, &symbol_id],
        )?;
        if exists.is_none() {
            bail!("code symbol `{symbol_id}` was not found in project `{project_id}`");
        }
        let mut tx = client.transaction()?;
        let vector = Vector::from(embedding.to_vec());
        let dimensions = embedding.len() as i32;
        tx.execute(
            "INSERT INTO code_symbol_embeddings_1024 (
                symbol_id, project_id, model, embedding_kind, content_hash, dimensions, embedding,
                created_at, updated_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, $7, now(), now())
             ON CONFLICT(symbol_id, model, embedding_kind) DO UPDATE SET
                project_id = excluded.project_id,
                content_hash = excluded.content_hash,
                dimensions = excluded.dimensions,
                embedding = excluded.embedding,
                updated_at = now()",
            &[
                &symbol_id,
                &project_id,
                &model,
                &embedding_kind,
                &content_hash,
                &dimensions,
                &vector,
            ],
        )?;
        if let Some(content_hash) = content_hash {
            tx.execute(
                "INSERT INTO code_symbol_embedding_cache_1024 (
                    project_id, model, embedding_kind, content_hash, dimensions, embedding,
                    created_at, updated_at
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, now(), now())
                 ON CONFLICT(project_id, model, embedding_kind, content_hash) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding,
                    updated_at = now()",
                &[
                    &project_id,
                    &model,
                    &embedding_kind,
                    &content_hash,
                    &dimensions,
                    &vector,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn attach_cached_code_symbol_embedding(
        &self,
        project_id: &str,
        symbol_id: &str,
        model: &str,
        content_hash: &str,
    ) -> Result<bool> {
        let body = self.attach_cached_code_symbol_embedding_kind(
            project_id,
            symbol_id,
            model,
            "body",
            content_hash,
        )?;
        let signature = self.attach_cached_code_symbol_embedding_kind(
            project_id,
            symbol_id,
            model,
            "signature",
            content_hash,
        )?;
        Ok(body || signature)
    }

    pub fn attach_cached_code_symbol_embedding_kind(
        &self,
        project_id: &str,
        symbol_id: &str,
        model: &str,
        embedding_kind: &str,
        content_hash: &str,
    ) -> Result<bool> {
        let embedding_kind = normalize_embedding_kind(embedding_kind)?;
        let mut client = self.client()?;
        let rows = client.execute(
            "INSERT INTO code_symbol_embeddings_1024 (
                symbol_id, project_id, model, embedding_kind, content_hash, dimensions, embedding,
                created_at, updated_at
             )
             SELECT $2, $1, $3, $4, c.content_hash, c.dimensions, c.embedding, now(), now()
             FROM code_symbol_embedding_cache_1024 c
             WHERE c.project_id = $1 AND c.model = $3 AND c.embedding_kind = $4 AND c.content_hash = $5
             ON CONFLICT(symbol_id, model, embedding_kind) DO UPDATE SET
                project_id = excluded.project_id,
                content_hash = excluded.content_hash,
                dimensions = excluded.dimensions,
                embedding = excluded.embedding,
                updated_at = now()",
            &[&project_id, &symbol_id, &model, &embedding_kind, &content_hash],
        )?;
        Ok(rows > 0)
    }

    pub fn cache_existing_code_symbol_embedding(
        &self,
        project_id: &str,
        symbol_id: &str,
        model: &str,
        content_hash: &str,
    ) -> Result<bool> {
        let mut client = self.client()?;
        let rows = client.execute(
            "INSERT INTO code_symbol_embedding_cache_1024 (
                project_id, model, embedding_kind, content_hash, dimensions, embedding, created_at, updated_at
             )
             SELECT e.project_id, e.model, e.embedding_kind, $4, e.dimensions, e.embedding, now(), now()
             FROM code_symbol_embeddings_1024 e
             WHERE e.project_id = $1 AND e.symbol_id = $2 AND e.model = $3
             ON CONFLICT(project_id, model, embedding_kind, content_hash) DO NOTHING",
            &[&project_id, &symbol_id, &model, &content_hash],
        )?;
        Ok(rows > 0)
    }

    pub fn code_symbols_with_embeddings(
        &self,
        project_id: &str,
        model: &str,
        limit: usize,
    ) -> Result<Vec<CodeSymbol>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT s.id, s.project_id, s.file_path, s.language, s.name, s.kind,
                    s.signature, s.body, s.start_line, s.end_line, s.parent_id
             FROM code_symbols s
             JOIN code_symbol_embeddings_1024 e ON e.symbol_id = s.id AND e.model = $2
             WHERE s.project_id = $1
             ORDER BY s.file_path, s.start_line
             LIMIT $3",
            &[&project_id, &model, &(limit.clamp(1, 5000) as i64)],
        )?;
        Ok(rows.iter().map(code_symbol_from_row).collect())
    }

    pub fn code_status(&self, project_id: &str) -> Result<CodeIndexStatus> {
        let relations = self.count_code_table("code_relations", project_id)?;
        let resolved_relations = self.count_resolved_code_relations(project_id)?;
        let quality_relations = self.count_project_quality_code_relations(project_id)?;
        let quality_resolved_relations =
            self.count_project_quality_resolved_code_relations(project_id)?;
        let unresolved_relations = quality_relations.saturating_sub(quality_resolved_relations);
        let external_call = self.count_code_relations_by_kind(project_id, "external_call")?;
        let external_use = self.count_code_relations_by_kind(project_id, "external_use")?;
        let cargo_package = self.count_code_relations_by_kind(project_id, "cargo_package")?;
        let ra_references = self.count_code_relations_by_kind(project_id, "ra_reference")?;
        let ra_calls = self.count_code_relations_by_kind(project_id, "ra_call")?;
        Ok(CodeIndexStatus {
            project_id: project_id.to_string(),
            files: self.count_code_table("code_files", project_id)?,
            symbols: self.count_code_table("code_symbols", project_id)?,
            relations,
            resolved_relations,
            ra_references,
            ra_calls,
            symbol_embeddings: self.count_code_symbol_embeddings(project_id)?,
            languages: self.code_language_statuses(project_id)?,
            relation_counts: CodeRelationCounts {
                total: relations,
                total_resolved: resolved_relations,
                project_quality: quality_relations,
                project_quality_resolved: quality_resolved_relations,
                project_quality_unresolved: unresolved_relations,
                external: external_call + external_use,
                external_call,
                external_use,
                cargo_package,
                ra_reference: ra_references,
                ra_call: ra_calls,
            },
            quality: CodeRelationQuality {
                relation_resolution_rate: if quality_relations == 0 {
                    1.0
                } else {
                    quality_resolved_relations as f64 / quality_relations as f64
                },
                unresolved_relations,
                ambiguous_unresolved_targets: self
                    .count_ambiguous_unresolved_code_targets(project_id)?,
                top_unresolved_targets: self.top_unresolved_code_targets(project_id, 10)?,
            },
        })
    }

    pub fn search_code(
        &self,
        project_id: &str,
        options: CodeSearchOptions,
    ) -> Result<Vec<CodeSearchResult>> {
        let query = to_tsquery(&options.query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = code_symbol_select_sql(
            "ts_rank_cd(s.search_vector, websearch_to_tsquery('simple', $2))::double precision",
        );
        sql.push_str(
            " WHERE s.project_id = $1
                AND s.search_vector @@ websearch_to_tsquery('simple', $2)",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> =
            vec![Box::new(project_id.to_string()), Box::new(query)];
        if let Some(kind) = options.kind {
            push_param(&mut sql, &mut params, kind);
            sql.push_str(" AND s.kind = $");
            sql.push_str(&params.len().to_string());
        }
        if let Some(file_path) = options.file_path {
            push_param(&mut sql, &mut params, file_path);
            sql.push_str(" AND s.file_path = $");
            sql.push_str(&params.len().to_string());
        }
        push_param(&mut sql, &mut params, options.limit.clamp(1, 50) as i64);
        sql.push_str(" ORDER BY score DESC, s.file_path, s.start_line LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter()
            .map(code_search_result_from_row)
            .collect::<Result<Vec<_>>>()
    }

    pub fn search_code_vectors(
        &self,
        project_id: &str,
        options: CodeVectorSearchOptions,
    ) -> Result<Vec<CodeSearchResult>> {
        if options.embedding.len() != 1024 {
            bail!(
                "code embedding dimension mismatch for pgvector search: expected 1024, got {}",
                options.embedding.len()
            );
        }
        let vector = Vector::from(options.embedding);
        let mut sql = code_symbol_select_sql("e.score");
        sql.push_str(
            " JOIN (
                SELECT symbol_id,
                       MIN(embedding <=> $3) AS distance,
                       MAX(1.0 - (embedding <=> $3)::double precision) AS score
                FROM code_symbol_embeddings_1024
                WHERE project_id = $1 AND model = $2
                GROUP BY symbol_id
              ) e ON e.symbol_id = s.id
              WHERE s.project_id = $1",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![
            Box::new(project_id.to_string()),
            Box::new(options.embedding_model),
            Box::new(vector),
        ];
        if let Some(kind) = options.kind {
            push_param(&mut sql, &mut params, kind);
            sql.push_str(" AND s.kind = $");
            sql.push_str(&params.len().to_string());
        }
        if let Some(file_path) = options.file_path {
            push_param(&mut sql, &mut params, file_path);
            sql.push_str(" AND s.file_path = $");
            sql.push_str(&params.len().to_string());
        }
        push_param(&mut sql, &mut params, options.limit.clamp(1, 50) as i64);
        sql.push_str(" ORDER BY e.distance LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter()
            .map(code_search_result_from_row)
            .collect::<Result<Vec<_>>>()
    }

    pub fn search_related_code_symbols(
        &self,
        project_id: &str,
        symbol_id: &str,
        embedding_model: &str,
        limit: usize,
        min_similarity: f64,
    ) -> Result<Vec<CodeSearchResult>> {
        let mut sql =
            code_symbol_select_sql("1.0 - (e.embedding <=> seed.embedding)::double precision");
        sql.push_str(
            " JOIN code_symbol_embeddings_1024 e ON e.symbol_id = s.id
              JOIN code_symbol_embeddings_1024 seed ON seed.symbol_id = $3
             WHERE e.project_id = $1
               AND e.model = $2
               AND e.embedding_kind = 'body'
               AND seed.project_id = $1
               AND seed.model = $2
               AND seed.embedding_kind = 'body'
               AND s.id <> $3
               AND (1.0 - (e.embedding <=> seed.embedding)::double precision) >= $4
             ORDER BY e.embedding <=> seed.embedding
             LIMIT $5",
        );
        let mut client = self.client()?;
        let rows = client.query(
            &sql,
            &[
                &project_id,
                &embedding_model,
                &symbol_id,
                &min_similarity,
                &(limit.clamp(1, 100) as i64),
            ],
        )?;
        rows.iter()
            .map(code_search_result_from_row)
            .collect::<Result<Vec<_>>>()
    }

    pub fn code_similarity_pairs(
        &self,
        project_id: &str,
        options: CodeSimilarityPairOptions,
    ) -> Result<Vec<CodeSimilarityPair>> {
        let mut sql = String::from(
            "SELECT
                    l.id, l.project_id, l.file_path, l.language, l.name, l.kind,
                    l.signature, l.body, l.start_line, l.end_line, l.parent_id,
                    r.id, r.project_id, r.file_path, r.language, r.name, r.kind,
                    r.signature, r.body, r.start_line, r.end_line, r.parent_id,
                    1.0 - (le.embedding <=> re.embedding)::double precision AS similarity
             FROM code_symbol_embeddings_1024 le
             JOIN code_symbol_embeddings_1024 re
               ON re.project_id = le.project_id
              AND re.model = le.model
              AND re.embedding_kind = le.embedding_kind
              AND re.symbol_id > le.symbol_id
             JOIN code_symbols l ON l.project_id = le.project_id AND l.id = le.symbol_id
             JOIN code_symbols r ON r.project_id = re.project_id AND r.id = re.symbol_id
             WHERE le.project_id = $1
               AND le.model = $2
               AND le.embedding_kind = 'body'
               AND (1.0 - (le.embedding <=> re.embedding)::double precision) >= $3",
        );
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![
            Box::new(project_id.to_string()),
            Box::new(options.embedding_model),
            Box::new(options.min_similarity.clamp(0.0, 1.0)),
        ];
        if let Some(kind) = options.kind {
            push_param(&mut sql, &mut params, kind);
            sql.push_str(" AND l.kind = $");
            sql.push_str(&params.len().to_string());
            sql.push_str(" AND r.kind = $");
            sql.push_str(&params.len().to_string());
        }
        if let Some(file_path) = options.file_path {
            push_param(&mut sql, &mut params, file_path);
            sql.push_str(" AND (l.file_path = $");
            sql.push_str(&params.len().to_string());
            sql.push_str(" OR r.file_path = $");
            sql.push_str(&params.len().to_string());
            sql.push(')');
        }
        push_param(&mut sql, &mut params, options.limit.clamp(1, 200) as i64);
        sql.push_str(" ORDER BY similarity DESC, l.file_path, l.start_line LIMIT $");
        sql.push_str(&params.len().to_string());

        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        Ok(rows.iter().map(code_similarity_pair_from_row).collect())
    }

    pub fn remember_code_memory(
        &self,
        project_id: &str,
        memory: NewCodeMemory,
        deduplicate: bool,
    ) -> Result<RememberOutcome> {
        validate_score("confidence", memory.confidence)?;
        let status = normalize_code_memory_status(&memory.status)?;
        if memory.symbol_id.is_none() && memory.file_path.is_none() {
            bail!("code memory must include symbol_id or file_path");
        }
        self.ensure_project(project_id)?;
        let symbol_snapshot = if let Some(symbol_ref) = &memory.symbol_id {
            Some(self.resolve_code_symbol_reference(
                project_id,
                symbol_ref,
                memory.file_path.as_deref(),
                memory.symbol_kind.as_deref(),
            )?)
        } else {
            None
        };
        let resolved_symbol_id = symbol_snapshot.as_ref().map(|symbol| symbol.id.clone());
        let file_path = memory.file_path.clone().or_else(|| {
            symbol_snapshot
                .as_ref()
                .map(|symbol| symbol.file_path.clone())
        });
        let body_hash = memory_body_hash(&memory.body);
        if deduplicate
            && let Some(existing_id) = self.find_live_code_memory_by_hash(project_id, &body_hash)?
        {
            return Ok(RememberOutcome {
                id: existing_id.clone(),
                inserted: false,
                duplicate_of: Some(existing_id),
            });
        }

        let id = Uuid::now_v7();
        let tags = json_array(normalize_tags(memory.tags));
        let symbol_name = symbol_snapshot.as_ref().map(|symbol| symbol.name.clone());
        let symbol_kind = symbol_snapshot.as_ref().map(|symbol| symbol.kind.clone());
        let symbol_signature = symbol_snapshot
            .as_ref()
            .map(|symbol| symbol.signature.clone());
        let symbol_body_hash = symbol_snapshot
            .as_ref()
            .map(|symbol| memory_body_hash(&symbol.body));
        let symbol_start_line = symbol_snapshot
            .as_ref()
            .map(|symbol| symbol.start_line as i32);
        let symbol_end_line = symbol_snapshot
            .as_ref()
            .map(|symbol| symbol.end_line as i32);
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO code_memories (
	                id, project_id, symbol_id, file_path, kind, body, body_hash,
	                tags, source, status, confidence,
	                symbol_name, symbol_kind, symbol_signature, symbol_start_line, symbol_end_line,
	                symbol_body_hash, quality_score
	             )
	             VALUES (
	                $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11,
	                $12, $13, $14, $15, $16, $17, $18
	             )",
            &[
                &id,
                &project_id,
                &resolved_symbol_id,
                &file_path,
                &memory.kind,
                &memory.body,
                &body_hash,
                &tags,
                &memory.source,
                &status,
                &memory.confidence,
                &symbol_name,
                &symbol_kind,
                &symbol_signature,
                &symbol_start_line,
                &symbol_end_line,
                &symbol_body_hash,
                &memory.confidence,
            ],
        )?;
        Ok(RememberOutcome {
            id: id.to_string(),
            inserted: true,
            duplicate_of: None,
        })
    }

    pub fn promote_code_memory(
        &self,
        project_id: &str,
        id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        let id = parse_uuid(id)?;
        let mut client = self.client()?;
        client.execute(
            "UPDATE code_memories
             SET status = 'active', status_reason = $1, updated_at = now()
             WHERE project_id = $2 AND id = $3",
            &[&reason, &project_id, &id],
        )?;
        Ok(())
    }

    pub fn archive_code_memory(
        &self,
        project_id: &str,
        id: &str,
        reason: Option<&str>,
    ) -> Result<()> {
        let id = parse_uuid(id)?;
        let mut client = self.client()?;
        client.execute(
            "UPDATE code_memories
             SET status = 'archived', status_reason = $1, updated_at = now()
             WHERE project_id = $2 AND id = $3",
            &[&reason, &project_id, &id],
        )?;
        Ok(())
    }

    pub fn search_code_memories(
        &self,
        project_id: &str,
        options: CodeMemorySearchOptions,
    ) -> Result<Vec<CodeMemory>> {
        let mut params: Vec<Box<dyn ToSql + Sync>> = vec![Box::new(project_id.to_string())];
        let mut sql = String::from(
            "SELECT cm.id, cm.project_id, cm.symbol_id, cm.file_path,
	                    CASE
	                        WHEN cm.symbol_id IS NOT NULL AND cs.id IS NOT NULL THEN 'symbol'
	                        WHEN cm.file_path IS NOT NULL AND cf.path IS NOT NULL THEN 'file'
	                        ELSE 'stale'
	                    END AS link_status,
	                    cm.kind, cm.body, cm.tags, cm.source,
	                    status, confidence, status_reason,
	                    0.0::double precision AS score,
	                    cm.created_at::text, cm.updated_at::text,
	                    cm.symbol_name, cm.symbol_kind, cm.symbol_signature,
	                    cm.symbol_start_line, cm.symbol_end_line,
	                    cm.last_relinked_at::text, cm.relink_attempts,
	                    cm.symbol_body_hash, cm.quality_score, cm.usage_count,
	                    cm.last_used_at::text, cm.contradiction_risk, cs.body
	             FROM code_memories cm
             LEFT JOIN code_symbols cs ON cs.project_id = cm.project_id AND cs.id = cm.symbol_id
             LEFT JOIN code_files cf ON cf.project_id = cm.project_id AND cf.path = cm.file_path
             WHERE cm.project_id = $1",
        );
        if options.status != "any" {
            push_param(
                &mut sql,
                &mut params,
                normalize_code_memory_status(&options.status)?,
            );
            sql.push_str(" AND cm.status = $");
            sql.push_str(&params.len().to_string());
        }
        if let Some(kind) = options.kind {
            push_param(&mut sql, &mut params, kind);
            sql.push_str(" AND cm.kind = $");
            sql.push_str(&params.len().to_string());
        }
        if !options.symbol_ids.is_empty() && !options.file_paths.is_empty() {
            push_param(&mut sql, &mut params, options.symbol_ids);
            let symbol_index = params.len();
            push_param(&mut sql, &mut params, options.file_paths);
            let file_index = params.len();
            sql.push_str(" AND (cm.symbol_id = ANY($");
            sql.push_str(&symbol_index.to_string());
            sql.push_str(") OR cm.file_path = ANY($");
            sql.push_str(&file_index.to_string());
            sql.push_str("))");
        } else if !options.symbol_ids.is_empty() {
            push_param(&mut sql, &mut params, options.symbol_ids);
            sql.push_str(" AND cm.symbol_id = ANY($");
            sql.push_str(&params.len().to_string());
            sql.push(')');
        } else if !options.file_paths.is_empty() {
            push_param(&mut sql, &mut params, options.file_paths);
            sql.push_str(" AND cm.file_path = ANY($");
            sql.push_str(&params.len().to_string());
            sql.push(')');
        }
        if let Some(query) = options.query {
            let query = to_tsquery(&query);
            if !query.is_empty() {
                push_param(&mut sql, &mut params, query);
                let index = params.len().to_string();
                sql = sql.replace(
                    "0.0::double precision AS score",
                    &format!(
                        "ts_rank_cd(cm.search_vector, websearch_to_tsquery('simple', ${index}))::double precision AS score"
                    ),
                );
                sql.push_str(" AND cm.search_vector @@ websearch_to_tsquery('simple', $");
                sql.push_str(&index);
                sql.push(')');
            }
        }
        push_param(&mut sql, &mut params, options.limit.clamp(1, 100) as i64);
        sql.push_str(" ORDER BY score DESC, cm.updated_at DESC LIMIT $");
        sql.push_str(&params.len().to_string());
        let mut client = self.client()?;
        let rows = client.query(&sql, &param_refs(&params))?;
        rows.iter().map(code_memory_from_row).collect()
    }

    pub fn repair_code_memory_links(
        &self,
        project_id: &str,
        limit: usize,
        apply: bool,
    ) -> Result<CodeMemoryRepairReport> {
        self.ensure_project(project_id)?;
        let mut client = self.client()?;
        let stale_rows = client.query(
            "SELECT cm.id, cm.symbol_id, cm.file_path, cm.symbol_name, cm.symbol_kind,
                    cm.symbol_signature, cm.symbol_start_line, cm.symbol_end_line
             FROM code_memories cm
             LEFT JOIN code_symbols cs ON cs.project_id = cm.project_id AND cs.id = cm.symbol_id
             WHERE cm.project_id = $1
               AND cm.status IN ('pending', 'active')
               AND cm.symbol_id IS NOT NULL
               AND cm.symbol_name IS NOT NULL
               AND cm.file_path IS NOT NULL
               AND cs.id IS NULL
             ORDER BY cm.updated_at DESC
             LIMIT $2",
            &[&project_id, &(limit.clamp(1, 500) as i64)],
        )?;

        let mut report = CodeMemoryRepairReport {
            project_id: project_id.to_string(),
            scanned: stale_rows.len(),
            repaired: 0,
            ambiguous: 0,
            stale: 0,
            dry_run: !apply,
            results: Vec::new(),
        };

        for row in stale_rows {
            let memory_id: Uuid = row.get(0);
            let old_symbol_id: Option<String> = row.get(1);
            let file_path: Option<String> = row.get(2);
            let symbol_name: Option<String> = row.get(3);
            let symbol_kind: Option<String> = row.get(4);
            let old_start_line: Option<i32> = row.get(6);

            let Some(file_path_value) = file_path.as_deref() else {
                report.stale += 1;
                report.results.push(CodeMemoryRepairResult {
                    memory_id: memory_id.to_string(),
                    old_symbol_id,
                    new_symbol_id: None,
                    file_path,
                    symbol_name,
                    symbol_kind,
                    status: "stale".to_string(),
                    reason: "missing file_path snapshot".to_string(),
                    candidates: 0,
                });
                continue;
            };
            let Some(symbol_name_value) = symbol_name.as_deref() else {
                report.stale += 1;
                report.results.push(CodeMemoryRepairResult {
                    memory_id: memory_id.to_string(),
                    old_symbol_id,
                    new_symbol_id: None,
                    file_path,
                    symbol_name,
                    symbol_kind,
                    status: "stale".to_string(),
                    reason: "missing symbol_name snapshot".to_string(),
                    candidates: 0,
                });
                continue;
            };

            let candidates = client.query(
                "SELECT id, project_id, file_path, language, name, kind,
                        signature, body, start_line, end_line, parent_id
                 FROM code_symbols
                 WHERE project_id = $1
                   AND file_path = $2
                   AND name = $3
                   AND ($4::text IS NULL OR kind = $4)
                 ORDER BY ABS(start_line - COALESCE($5, start_line)), start_line
                 LIMIT 3",
                &[
                    &project_id,
                    &file_path_value,
                    &symbol_name_value,
                    &symbol_kind,
                    &old_start_line,
                ],
            )?;

            if candidates.len() == 1 {
                let symbol = code_symbol_from_row(&candidates[0]);
                if apply {
                    client.execute(
                        "UPDATE code_memories
                         SET symbol_id = $1,
                             file_path = $2,
                             symbol_name = $3,
                             symbol_kind = $4,
	                            symbol_signature = $5,
	                            symbol_start_line = $6,
	                            symbol_end_line = $7,
	                            symbol_body_hash = $8,
	                            last_relinked_at = now(),
	                            relink_attempts = relink_attempts + 1,
	                            updated_at = now()
	                         WHERE project_id = $9 AND id = $10",
                        &[
                            &symbol.id,
                            &symbol.file_path,
                            &symbol.name,
                            &symbol.kind,
                            &symbol.signature,
                            &(symbol.start_line as i32),
                            &(symbol.end_line as i32),
                            &memory_body_hash(&symbol.body),
                            &project_id,
                            &memory_id,
                        ],
                    )?;
                    report.repaired += 1;
                    report.results.push(CodeMemoryRepairResult {
                        memory_id: memory_id.to_string(),
                        old_symbol_id,
                        new_symbol_id: Some(symbol.id),
                        file_path,
                        symbol_name,
                        symbol_kind,
                        status: "repaired".to_string(),
                        reason: "unique symbol snapshot match".to_string(),
                        candidates: 1,
                    });
                } else {
                    report.results.push(CodeMemoryRepairResult {
                        memory_id: memory_id.to_string(),
                        old_symbol_id,
                        new_symbol_id: Some(symbol.id),
                        file_path,
                        symbol_name,
                        symbol_kind,
                        status: "would_repair".to_string(),
                        reason: "unique symbol snapshot match".to_string(),
                        candidates: 1,
                    });
                }
            } else if candidates.is_empty() {
                report.stale += 1;
                report.results.push(CodeMemoryRepairResult {
                    memory_id: memory_id.to_string(),
                    old_symbol_id,
                    new_symbol_id: None,
                    file_path,
                    symbol_name,
                    symbol_kind,
                    status: "stale".to_string(),
                    reason: "no matching indexed symbol".to_string(),
                    candidates: 0,
                });
            } else {
                report.ambiguous += 1;
                report.results.push(CodeMemoryRepairResult {
                    memory_id: memory_id.to_string(),
                    old_symbol_id,
                    new_symbol_id: None,
                    file_path,
                    symbol_name,
                    symbol_kind,
                    status: "ambiguous".to_string(),
                    reason: "multiple matching indexed symbols".to_string(),
                    candidates: candidates.len(),
                });
            }
        }

        Ok(report)
    }

    pub fn code_memories_for_code_results(
        &self,
        project_id: &str,
        code: &[CodeSearchResult],
        limit: usize,
    ) -> Result<Vec<CodeMemory>> {
        let symbol_ids = code
            .iter()
            .map(|result| result.symbol.id.clone())
            .collect::<Vec<_>>();
        let mut file_paths = code
            .iter()
            .map(|result| result.symbol.file_path.clone())
            .collect::<Vec<_>>();
        file_paths.sort();
        file_paths.dedup();
        self.search_code_memories(
            project_id,
            CodeMemorySearchOptions {
                query: None,
                limit,
                status: "active".to_string(),
                kind: None,
                symbol_ids,
                file_paths,
            },
        )
    }

    pub fn affected_test_files(
        &self,
        project_id: &str,
        changed_files: &[String],
        depth: usize,
        limit: usize,
    ) -> Result<Vec<String>> {
        if changed_files.is_empty() {
            return Ok(Vec::new());
        }
        let mut frontier = changed_files.iter().cloned().collect::<HashSet<_>>();
        let mut seen = frontier.clone();
        let mut tests = HashSet::new();
        for _ in 0..depth.clamp(1, 8) {
            if frontier.is_empty() {
                break;
            }
            let current = frontier.iter().cloned().collect::<Vec<_>>();
            let mut client = self.client()?;
            let rows = client.query(
                "SELECT DISTINCT COALESCE(target.file_path, r.from_file_path), r.from_file_path
                 FROM code_relations r
                 LEFT JOIN code_symbols target
                   ON target.project_id = r.project_id AND target.id = r.target_symbol_id
                 WHERE r.project_id = $1
                   AND (r.from_file_path = ANY($2) OR target.file_path = ANY($2))
                 LIMIT $3",
                &[
                    &project_id,
                    &current,
                    &(limit.saturating_mul(20).clamp(20, 5000) as i64),
                ],
            )?;
            let mut next = HashSet::new();
            for row in rows {
                for index in 0..2 {
                    let file: Option<String> = row.get(index);
                    let Some(file) = file else { continue };
                    if is_test_file(&file) {
                        tests.insert(file.clone());
                    }
                    if seen.insert(file.clone()) {
                        next.insert(file);
                    }
                }
            }
            frontier = next;
        }
        let mut tests = tests.into_iter().collect::<Vec<_>>();
        tests.sort();
        tests.truncate(limit.clamp(1, 500));
        Ok(tests)
    }

    pub fn route_hints(
        &self,
        project_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<CodeRouteHint>> {
        let terms = query
            .split_whitespace()
            .map(|term| term.to_ascii_lowercase())
            .filter(|term| term.len() >= 2)
            .collect::<Vec<_>>();
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1
               AND kind IN ('function', 'method', 'class', 'struct', 'module')
             ORDER BY file_path, start_line
             LIMIT 5000",
            &[&project_id],
        )?;
        let mut hints = Vec::new();
        for row in rows {
            let symbol = code_symbol_from_row(&row);
            for hint in route_hints_for_symbol(&symbol) {
                if !terms.is_empty() {
                    let haystack = format!(
                        "{} {} {} {}",
                        hint.file_path, hint.symbol_name, hint.route, hint.evidence
                    )
                    .to_ascii_lowercase();
                    if !terms.iter().any(|term| haystack.contains(term)) {
                        continue;
                    }
                }
                hints.push(hint);
                if hints.len() >= limit.clamp(1, 100) {
                    return Ok(hints);
                }
            }
        }
        Ok(hints)
    }

    pub fn code_symbols_missing_embeddings(
        &self,
        project_id: &str,
        model: &str,
        limit: usize,
    ) -> Result<Vec<CodeSymbol>> {
        let mut client = self.client()?;
        let required_kinds = required_code_embedding_kinds();
        let rows = client.query(
            "SELECT s.id, s.project_id, s.file_path, s.language, s.name, s.kind,
                    s.signature, s.body, s.start_line, s.end_line, s.parent_id
             FROM code_symbols s
             WHERE s.project_id = $1
               AND (
                    SELECT COUNT(DISTINCT e.embedding_kind)
                    FROM code_symbol_embeddings_1024 e
                    WHERE e.symbol_id = s.id
                      AND e.project_id = $1
                      AND e.model = $2
                      AND e.embedding_kind = ANY($4)
               ) < cardinality($4::text[])
             ORDER BY s.file_path, s.start_line
             LIMIT $3",
            &[
                &project_id,
                &model,
                &(limit.clamp(1, 500) as i64),
                &required_kinds,
            ],
        )?;
        Ok(rows.iter().map(code_symbol_from_row).collect())
    }

    pub fn code_symbols_missing_embeddings_for_files(
        &self,
        project_id: &str,
        model: &str,
        file_paths: &[String],
        limit: usize,
    ) -> Result<Vec<CodeSymbol>> {
        if file_paths.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = String::from(
            "SELECT s.id, s.project_id, s.file_path, s.language, s.name, s.kind,
                    s.signature, s.body, s.start_line, s.end_line, s.parent_id
             FROM code_symbols s
             WHERE s.project_id = $1
               AND s.file_path = ANY($3)
               AND (
                    SELECT COUNT(DISTINCT e.embedding_kind)
                    FROM code_symbol_embeddings_1024 e
                    WHERE e.symbol_id = s.id
                      AND e.project_id = $1
                      AND e.model = $2
                      AND e.embedding_kind = ANY($5)
               ) < cardinality($5::text[])",
        );
        sql.push_str(" ORDER BY s.file_path, s.start_line LIMIT $4");
        let mut client = self.client()?;
        let required_kinds = required_code_embedding_kinds();
        let rows = client.query(
            &sql,
            &[
                &project_id,
                &model,
                &file_paths,
                &(limit.clamp(1, 500) as i64),
                &required_kinds,
            ],
        )?;
        Ok(rows.iter().map(code_symbol_from_row).collect())
    }

    pub fn get_code_symbol(
        &self,
        project_id: &str,
        symbol_ref: &str,
    ) -> Result<Option<CodeSymbol>> {
        let mut client = self.client()?;
        if let Some(row) = client.query_opt(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1 AND id = $2",
            &[&project_id, &symbol_ref],
        )? {
            return Ok(Some(code_symbol_from_row(&row)));
        }
        Ok(client
            .query_opt(
                "SELECT id, project_id, file_path, language, name, kind,
                        signature, body, start_line, end_line, parent_id
                 FROM code_symbols
                 WHERE project_id = $1 AND name = $2
                 ORDER BY file_path, start_line
                 LIMIT 1",
                &[&project_id, &symbol_ref],
            )?
            .map(|row| code_symbol_from_row(&row)))
    }

    pub fn resolve_code_symbol_reference(
        &self,
        project_id: &str,
        symbol_ref: &str,
        file_path: Option<&str>,
        kind: Option<&str>,
    ) -> Result<CodeSymbol> {
        let mut client = self.client()?;
        if let Some(row) = client.query_opt(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1 AND id = $2",
            &[&project_id, &symbol_ref],
        )? {
            let symbol = code_symbol_from_row(&row);
            if let Some(file_path) = file_path
                && symbol.file_path != file_path
            {
                bail!(
                    "code symbol `{symbol_ref}` is in `{}`, not `{file_path}`",
                    symbol.file_path
                );
            }
            if let Some(kind) = kind
                && symbol.kind != kind
            {
                bail!(
                    "code symbol `{symbol_ref}` has kind `{}`, not `{kind}`",
                    symbol.kind
                );
            }
            return Ok(symbol);
        }

        let file_filter = file_path.map(str::to_string);
        let kind_filter = kind.map(str::to_string);
        let rows = client.query(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1
               AND name = $2
               AND ($3::text IS NULL OR file_path = $3)
               AND ($4::text IS NULL OR kind = $4)
             ORDER BY file_path, start_line
             LIMIT 6",
            &[&project_id, &symbol_ref, &file_filter, &kind_filter],
        )?;
        match rows.len() {
            1 => Ok(code_symbol_from_row(&rows[0])),
            0 => bail!("code symbol `{symbol_ref}` was not found in project `{project_id}`"),
            _ => {
                let candidates = rows
                    .iter()
                    .map(|row| {
                        let symbol = code_symbol_from_row(row);
                        format!(
                            "{}:{} {} {} ({})",
                            symbol.file_path,
                            symbol.start_line,
                            symbol.kind,
                            symbol.name,
                            symbol.id
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                bail!(
                    "code symbol `{symbol_ref}` is ambiguous in project `{project_id}`; pass file_path and/or symbol_kind. Candidates: {candidates}"
                );
            }
        }
    }

    pub fn find_callees(
        &self,
        project_id: &str,
        symbol_ref: &str,
        limit: usize,
    ) -> Result<Vec<CodeRelation>> {
        let Some(symbol) = self.get_code_symbol(project_id, symbol_ref)? else {
            return Ok(Vec::new());
        };
        self.relations_by_from_symbol(project_id, &symbol.id, limit)
    }

    pub fn find_callers(
        &self,
        project_id: &str,
        symbol_ref: &str,
        limit: usize,
    ) -> Result<Vec<CodeRelation>> {
        let Some(symbol) = self.get_code_symbol(project_id, symbol_ref)? else {
            return Ok(Vec::new());
        };
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, from_symbol_id, from_file_path,
                    relation_kind, target_name, target_symbol_id
             FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'ra_call')
               AND (target_symbol_id = $2 OR (target_symbol_id IS NULL AND target_name = $3))
             ORDER BY relation_kind, from_file_path
             LIMIT $4",
            &[
                &project_id,
                &symbol.id,
                &symbol.name,
                &(limit.clamp(1, 200) as i64),
            ],
        )?;
        Ok(rows.iter().map(code_relation_from_row).collect())
    }

    fn relations_by_from_symbol(
        &self,
        project_id: &str,
        from_symbol_id: &str,
        limit: usize,
    ) -> Result<Vec<CodeRelation>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, from_symbol_id, from_file_path,
                    relation_kind, target_name, target_symbol_id
             FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'ra_call')
               AND from_symbol_id = $2
             ORDER BY relation_kind, target_name
             LIMIT $3",
            &[&project_id, &from_symbol_id, &(limit.clamp(1, 200) as i64)],
        )?;
        Ok(rows.iter().map(code_relation_from_row).collect())
    }

    pub fn code_graph_for_symbols(
        &self,
        project_id: &str,
        symbol_ids: &[String],
        limit: usize,
    ) -> Result<(Vec<CodeSymbol>, Vec<CodeRelation>)> {
        if symbol_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut client = self.client()?;
        let relation_rows = client.query(
            "SELECT id, project_id, from_symbol_id, from_file_path,
                    relation_kind, target_name, target_symbol_id
             FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN (
                    'calls', 'ra_call', 'uses', 'declares_module',
                    'cargo_package', 'ra_reference'
               )
               AND (from_symbol_id = ANY($2) OR target_symbol_id = ANY($2))
             ORDER BY relation_kind, from_file_path, target_name
             LIMIT $3",
            &[&project_id, &symbol_ids, &(limit.clamp(1, 500) as i64)],
        )?;
        let relations = relation_rows
            .iter()
            .map(code_relation_from_row)
            .collect::<Vec<_>>();
        let mut related_ids = symbol_ids.to_vec();
        for relation in &relations {
            if let Some(from_symbol_id) = &relation.from_symbol_id {
                related_ids.push(from_symbol_id.clone());
            }
            if let Some(target_symbol_id) = &relation.target_symbol_id {
                related_ids.push(target_symbol_id.clone());
            }
        }
        related_ids.sort();
        related_ids.dedup();

        let symbol_rows = client.query(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1 AND id = ANY($2)
             ORDER BY file_path, start_line, id",
            &[&project_id, &related_ids],
        )?;
        Ok((
            symbol_rows.iter().map(code_symbol_from_row).collect(),
            relations,
        ))
    }

    pub fn export_project(&self, project_id: &str, include_code: bool) -> Result<ProjectExport> {
        self.ensure_project(project_id)?;
        let project = self.project_record(project_id)?;
        let mut client = self.client()?;
        let exported_at = client
            .query_one("SELECT ts(now())", &[])?
            .get::<_, String>(0);
        drop(client);
        let memories = self.export_memories(project_id)?;
        let (code_files, code_symbols, code_relations) = if include_code {
            (
                self.export_code_files(project_id)?,
                self.export_code_symbols(project_id)?,
                self.export_code_relations(project_id)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        Ok(ProjectExport {
            format: PROJECT_EXPORT_FORMAT.to_string(),
            schema_version: self.schema_version()?,
            exported_at,
            project,
            includes_code: include_code,
            includes_embeddings: false,
            memories,
            code_files,
            code_symbols,
            code_relations,
        })
    }

    pub fn import_project(
        &mut self,
        export: ProjectExport,
        overwrite: bool,
    ) -> Result<ProjectImportReport> {
        if export.format != PROJECT_EXPORT_FORMAT {
            bail!(
                "unsupported export format `{}`; expected `{}`",
                export.format,
                PROJECT_EXPORT_FORMAT
            );
        }
        if export.schema_version > SCHEMA_VERSION {
            bail!(
                "export schema version {} is newer than supported version {}",
                export.schema_version,
                SCHEMA_VERSION
            );
        }
        validate_export_scope(&export)?;
        let project_id = export.project.id.clone();
        let mut report = ProjectImportReport {
            project_id: project_id.clone(),
            overwrite,
            memories_imported: 0,
            memories_skipped: 0,
            code_files_imported: 0,
            code_files_skipped: 0,
            code_symbols_imported: 0,
            code_symbols_skipped: 0,
            code_relations_imported: 0,
            code_relations_skipped: 0,
        };
        let mut client = self.client()?;
        let mut tx = client.transaction()?;
        if overwrite {
            tx.execute(
                "DELETE FROM code_relations WHERE project_id = $1",
                &[&project_id],
            )?;
            tx.execute(
                "DELETE FROM code_symbol_embeddings_1024 WHERE project_id = $1",
                &[&project_id],
            )?;
            tx.execute(
                "DELETE FROM code_symbols WHERE project_id = $1",
                &[&project_id],
            )?;
            tx.execute(
                "DELETE FROM code_files WHERE project_id = $1",
                &[&project_id],
            )?;
            tx.execute(
                "DELETE FROM memory_embeddings_4096 WHERE project_id = $1",
                &[&project_id],
            )?;
            tx.execute("DELETE FROM memories WHERE project_id = $1", &[&project_id])?;
        }
        let domains = json_array(normalize_tags(export.project.domains));
        tx.execute(
            "INSERT INTO projects (id, name, root_path, project_type, description, domains, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6::jsonb, COALESCE($7::text::timestamptz, now()), COALESCE($8::text::timestamptz, now()))
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                root_path = excluded.root_path,
                project_type = excluded.project_type,
                description = excluded.description,
                domains = excluded.domains,
                updated_at = now()",
            &[
                &export.project.id,
                &export.project.name,
                &export.project.root_path,
                &normalize_project_type(&export.project.project_type),
                &export.project.description,
                &domains,
                &Some(export.project.created_at),
                &Some(export.project.updated_at),
            ],
        )?;
        for memory in export.memories {
            let id = parse_uuid(&memory.id)?;
            let superseded_by = memory
                .superseded_by
                .as_deref()
                .map(parse_uuid)
                .transpose()?;
            let tags = json_array(normalize_tags(memory.tags));
            let changed = tx.execute(
                "INSERT INTO memories (
                    id, project_id, scope, memory_tier, kind, body, body_hash, tags, source, status,
                    importance, confidence, superseded_by, status_reason, created_at, updated_at
                 )
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11, $12, $13, $14,
                         COALESCE($15::text::timestamptz, now()), COALESCE($16::text::timestamptz, now()))
                 ON CONFLICT(id) DO NOTHING",
                &[
                    &id,
                    &memory.project_id,
                    &normalize_memory_scope(&memory.scope)?,
                    &normalize_memory_tier(&memory.memory_tier)?,
                    &memory.kind,
                    &memory.body,
                    &memory_body_hash(&memory.body),
                    &tags,
                    &memory.source,
                    &memory.status,
                    &memory.importance,
                    &memory.confidence,
                    &superseded_by,
                    &memory.status_reason,
                    &Some(memory.created_at),
                    &Some(memory.updated_at),
                ],
            )?;
            if changed == 0 {
                report.memories_skipped += 1;
            } else {
                report.memories_imported += 1;
            }
        }
        if export.includes_code {
            for file in export.code_files {
                let changed = tx.execute(
                    "INSERT INTO code_files (project_id, path, language, hash, size_bytes, line_count, indexed_at)
                     VALUES ($1, $2, $3, $4, $5, $6, now())
                     ON CONFLICT(project_id, path) DO NOTHING",
                    &[
                        &file.project_id,
                        &file.path,
                        &file.language,
                        &file.hash,
                        &(file.size_bytes as i64),
                        &(file.line_count as i32),
                    ],
                )?;
                if changed == 0 {
                    report.code_files_skipped += 1;
                } else {
                    report.code_files_imported += 1;
                }
            }
            for symbol in export.code_symbols {
                let changed = tx.execute(
                    "INSERT INTO code_symbols (
                        id, project_id, file_path, language, name, kind, signature,
                        body, start_line, end_line, parent_id, indexed_at
                     )
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
                     ON CONFLICT(id) DO NOTHING",
                    &[
                        &symbol.id,
                        &symbol.project_id,
                        &symbol.file_path,
                        &symbol.language,
                        &symbol.name,
                        &symbol.kind,
                        &symbol.signature,
                        &symbol.body,
                        &(symbol.start_line as i32),
                        &(symbol.end_line as i32),
                        &symbol.parent_id,
                    ],
                )?;
                if changed == 0 {
                    report.code_symbols_skipped += 1;
                } else {
                    report.code_symbols_imported += 1;
                }
            }
            for relation in export.code_relations {
                let changed = tx.execute(
                    "INSERT INTO code_relations (
                        id, project_id, from_symbol_id, from_file_path,
                        relation_kind, target_name, target_symbol_id
                     )
                     VALUES ($1, $2, $3, $4, $5, $6, $7)
                     ON CONFLICT(id) DO NOTHING",
                    &[
                        &relation.id,
                        &relation.project_id,
                        &relation.from_symbol_id,
                        &relation.from_file_path,
                        &relation.relation_kind,
                        &relation.target_name,
                        &relation.target_symbol_id,
                    ],
                )?;
                if changed == 0 {
                    report.code_relations_skipped += 1;
                } else {
                    report.code_relations_imported += 1;
                }
            }
        }
        tx.commit()?;
        self.rebuild_project_memory_fragments(&report.project_id)?;
        Ok(report)
    }

    pub fn rebuild_project_memory_fragments(&self, project_id: &str) -> Result<usize> {
        let mut client = self.client()?;
        client.execute(
            "DELETE FROM memory_fragments WHERE project_id = $1",
            &[&project_id],
        )?;
        let rows = client.query(
            "SELECT id, body_hash, body
             FROM memories
             WHERE project_id = $1
             ORDER BY created_at, id",
            &[&project_id],
        )?;
        let mut fragments = 0usize;
        for row in rows {
            let memory_id: Uuid = row.get(0);
            let body_hash: String = row.get(1);
            let body: String = row.get(2);
            fragments +=
                sync_memory_fragments(&mut client, project_id, memory_id, &body_hash, &body)?;
        }
        Ok(fragments)
    }

    pub fn code_symbols_for_project(&self, project_id: &str) -> Result<Vec<CodeSymbol>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1
             ORDER BY file_path, start_line, id",
            &[&project_id],
        )?;
        Ok(rows.iter().map(code_symbol_from_row).collect())
    }

    pub fn code_files_for_project(&self, project_id: &str) -> Result<Vec<CodeFile>> {
        self.export_code_files(project_id)
    }

    pub fn code_symbols_for_file(
        &self,
        project_id: &str,
        file_path: &str,
    ) -> Result<Vec<CodeSymbol>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, file_path, language, name, kind,
                    signature, body, start_line, end_line, parent_id
             FROM code_symbols
             WHERE project_id = $1 AND file_path = $2
             ORDER BY start_line, id",
            &[&project_id, &file_path],
        )?;
        Ok(rows.iter().map(code_symbol_from_row).collect())
    }

    fn all_code_symbols(&self, project_id: &str) -> Result<Vec<CodeSymbol>> {
        self.code_symbols_for_project(project_id)
    }

    fn unresolved_resolvable_code_relations(&self, project_id: &str) -> Result<Vec<CodeRelation>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, from_symbol_id, from_file_path,
                    relation_kind, target_name, target_symbol_id
             FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'uses', 'declares_module')
             ORDER BY from_file_path, relation_kind, target_name, id",
            &[&project_id],
        )?;
        Ok(rows.iter().map(code_relation_from_row).collect())
    }

    fn project_record(&self, project_id: &str) -> Result<ProjectRecord> {
        let mut client = self.client()?;
        let row = client
            .query_one(
                "SELECT id, name, root_path, project_type, description, domains,
                        ts(created_at), ts(updated_at)
                 FROM projects WHERE id = $1",
                &[&project_id],
            )
            .with_context(|| format!("failed to read project `{project_id}`"))?;
        project_record_from_row(&row)
    }

    fn export_memories(&self, project_id: &str) -> Result<Vec<ExportedMemory>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, scope, memory_tier, kind, body, tags, source, status,
                    importance, confidence, superseded_by, status_reason,
                    ts(created_at), ts(updated_at)
             FROM memories
             WHERE project_id = $1
             ORDER BY created_at, id",
            &[&project_id],
        )?;
        rows.iter()
            .map(exported_memory_from_row)
            .collect::<Result<Vec<_>>>()
    }

    fn export_code_files(&self, project_id: &str) -> Result<Vec<CodeFile>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT project_id, path, language, hash, size_bytes, line_count
             FROM code_files WHERE project_id = $1 ORDER BY path",
            &[&project_id],
        )?;
        Ok(rows.iter().map(code_file_from_row).collect())
    }

    fn export_code_symbols(&self, project_id: &str) -> Result<Vec<CodeSymbol>> {
        self.code_symbols_for_project(project_id)
    }

    fn export_code_relations(&self, project_id: &str) -> Result<Vec<CodeRelation>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT id, project_id, from_symbol_id, from_file_path,
                    relation_kind, target_name, target_symbol_id
             FROM code_relations
             WHERE project_id = $1
             ORDER BY from_file_path, relation_kind, target_name, id",
            &[&project_id],
        )?;
        Ok(rows.iter().map(code_relation_from_row).collect())
    }

    fn set_status(
        &self,
        project_id: &str,
        id: &str,
        status: MemoryStatus,
        reason: Option<&str>,
    ) -> Result<()> {
        let id = parse_uuid(id)?;
        let mut client = self.client()?;
        let changed = client.execute(
            "UPDATE memories
             SET status = $1, status_reason = $2, updated_at = now()
             WHERE project_id = $3 AND id = $4",
            &[&status.as_str(), &reason, &project_id, &id],
        )?;
        if changed == 0 {
            bail!("memory `{id}` was not found in project `{project_id}`");
        }
        Ok(())
    }

    fn count_by_status(&self, project_id: &str, status: Option<MemoryStatus>) -> Result<u64> {
        let mut client = self.client()?;
        let count: i64 = match status {
            Some(status) => client
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM memories WHERE project_id = $1 AND status = $2",
                    &[&project_id, &status.as_str()],
                )?
                .get(0),
            None => client
                .query_one(
                    "SELECT COUNT(*)::BIGINT FROM memories WHERE project_id = $1",
                    &[&project_id],
                )?
                .get(0),
        };
        Ok(count as u64)
    }

    fn count_code_table(&self, table: &str, project_id: &str) -> Result<u64> {
        validate_known_table(table)?;
        let mut client = self.client()?;
        let row = client.query_one(
            &format!("SELECT COUNT(*)::BIGINT FROM {table} WHERE project_id = $1"),
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn count_memory_embeddings(&self, project_id: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COUNT(DISTINCT memory_id)::BIGINT
             FROM memory_embeddings_4096
             WHERE project_id = $1",
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn count_code_symbol_embeddings(&self, project_id: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COUNT(DISTINCT symbol_id)::BIGINT
             FROM code_symbol_embeddings_1024
             WHERE project_id = $1",
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn count_resolved_code_relations(&self, project_id: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COUNT(*)::BIGINT FROM code_relations
             WHERE project_id = $1 AND target_symbol_id IS NOT NULL",
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn count_project_quality_code_relations(&self, project_id: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COUNT(*)::BIGINT FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'uses', 'declares_module', 'ra_call', 'ra_reference')",
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn count_project_quality_resolved_code_relations(&self, project_id: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COUNT(*)::BIGINT FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'uses', 'declares_module', 'ra_call', 'ra_reference')
               AND target_symbol_id IS NOT NULL",
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn count_code_relations_by_kind(&self, project_id: &str, relation_kind: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "SELECT COUNT(*)::BIGINT FROM code_relations
             WHERE project_id = $1 AND relation_kind = $2",
            &[&project_id, &relation_kind],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn code_language_statuses(&self, project_id: &str) -> Result<Vec<CodeLanguageStatus>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT cf.language,
                    COUNT(DISTINCT cf.path)::BIGINT AS files,
                    COUNT(DISTINCT s.id)::BIGINT AS symbols,
                    COUNT(DISTINCT r.id)::BIGINT AS relations,
                    COUNT(DISTINCT r.id) FILTER (WHERE r.target_symbol_id IS NOT NULL)::BIGINT AS resolved_relations,
                    COUNT(DISTINCT e.symbol_id)::BIGINT AS symbol_embeddings
             FROM code_files cf
             LEFT JOIN code_symbols s
               ON s.project_id = cf.project_id AND s.file_path = cf.path
             LEFT JOIN code_relations r
               ON r.project_id = cf.project_id AND r.from_file_path = cf.path
             LEFT JOIN code_symbol_embeddings_1024 e
               ON e.project_id = cf.project_id AND e.symbol_id = s.id
             WHERE cf.project_id = $1
             GROUP BY cf.language
             ORDER BY cf.language",
            &[&project_id],
        )?;
        Ok(rows
            .iter()
            .map(|row| CodeLanguageStatus {
                language: row.get(0),
                files: row.get::<_, i64>(1) as u64,
                symbols: row.get::<_, i64>(2) as u64,
                relations: row.get::<_, i64>(3) as u64,
                resolved_relations: row.get::<_, i64>(4) as u64,
                symbol_embeddings: row.get::<_, i64>(5) as u64,
            })
            .collect())
    }

    fn top_unresolved_code_targets(
        &self,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<CodeUnresolvedTarget>> {
        let mut client = self.client()?;
        let rows = client.query(
            "SELECT relation_kind, target_name, COUNT(*)::BIGINT AS count
             FROM code_relations
             WHERE project_id = $1
               AND relation_kind IN ('calls', 'uses', 'declares_module')
               AND target_symbol_id IS NULL
             GROUP BY relation_kind, target_name
             ORDER BY count DESC, relation_kind, target_name
             LIMIT $2",
            &[&project_id, &(limit.clamp(1, 50) as i64)],
        )?;
        Ok(rows
            .iter()
            .map(|row| CodeUnresolvedTarget {
                relation_kind: row.get(0),
                target_name: row.get(1),
                count: row.get::<_, i64>(2) as u64,
            })
            .collect())
    }

    fn count_ambiguous_unresolved_code_targets(&self, project_id: &str) -> Result<u64> {
        let mut client = self.client()?;
        let row = client.query_one(
            "WITH ambiguous_names AS (
                 SELECT name FROM code_symbols
                 WHERE project_id = $1
                 GROUP BY name HAVING COUNT(*) > 1
             )
             SELECT COUNT(DISTINCT r.target_name)::BIGINT
             FROM code_relations r
             JOIN ambiguous_names a ON a.name = r.target_name
             WHERE r.project_id = $1
               AND r.relation_kind IN ('calls', 'uses', 'declares_module')
               AND r.target_symbol_id IS NULL",
            &[&project_id],
        )?;
        Ok(row.get::<_, i64>(0) as u64)
    }

    fn ensure_project(&self, project_id: &str) -> Result<()> {
        let mut client = self.client()?;
        client.execute(
            "INSERT INTO projects (id, name) VALUES ($1, $1) ON CONFLICT(id) DO NOTHING",
            &[&project_id],
        )?;
        Ok(())
    }

    fn ensure_memory_exists(&self, project_id: &str, id: &str) -> Result<()> {
        let id = parse_uuid(id)?;
        let mut client = self.client()?;
        if client
            .query_opt(
                "SELECT 1 FROM memories WHERE project_id = $1 AND id = $2",
                &[&project_id, &id],
            )?
            .is_some()
        {
            Ok(())
        } else {
            Err(anyhow!(
                "memory `{id}` was not found in project `{project_id}`"
            ))
        }
    }

    fn find_live_memory_by_hash(
        &self,
        project_id: &str,
        body_hash: &str,
    ) -> Result<Option<String>> {
        let mut client = self.client()?;
        Ok(client
            .query_opt(
                "SELECT id
                 FROM memories
                 WHERE project_id = $1
                   AND body_hash = $2
                   AND status IN ('pending', 'active')
                 ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, updated_at DESC
                 LIMIT 1",
                &[&project_id, &body_hash],
            )?
            .map(|row| row.get::<_, Uuid>(0).to_string()))
    }

    fn find_live_code_memory_by_hash(
        &self,
        project_id: &str,
        body_hash: &str,
    ) -> Result<Option<String>> {
        let mut client = self.client()?;
        Ok(client
            .query_opt(
                "SELECT id
                 FROM code_memories
                 WHERE project_id = $1
                   AND body_hash = $2
                   AND status IN ('pending', 'active')
                 ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, updated_at DESC
                 LIMIT 1",
                &[&project_id, &body_hash],
            )?
            .map(|row| row.get::<_, Uuid>(0).to_string()))
    }

    fn client(&self) -> Result<StoreClient<'_>> {
        let client = self
            .clients
            .lock()
            .map_err(|_| anyhow!("PostgreSQL client pool mutex is poisoned"))?
            .pop();
        let client = match client {
            Some(client) => client,
            None => {
                let mut client = connect_postgres(&self.database_url)?;
                setup_connection(&mut client, self.schema.as_deref())
                    .context("failed to prepare PostgreSQL pooled connection")?;
                client
            }
        };
        Ok(StoreClient {
            pool: &self.clients,
            client: Some(client),
        })
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        if let Ok(mut clients) = self.clients.lock() {
            for client in clients.drain(..) {
                close_client(client);
            }
        }
    }
}

struct StoreClient<'a> {
    pool: &'a Mutex<Vec<Client>>,
    client: Option<Client>,
}

impl StoreClient<'_> {
    fn raw(&mut self) -> &mut Client {
        self.client
            .as_mut()
            .expect("PostgreSQL client was already closed")
    }

    fn execute(
        &mut self,
        query: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> std::result::Result<u64, postgres::Error> {
        pg_block(|| self.raw().execute(query, params))
    }

    fn query(
        &mut self,
        query: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> std::result::Result<Vec<Row>, postgres::Error> {
        pg_block(|| self.raw().query(query, params))
    }

    fn query_one(
        &mut self,
        query: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> std::result::Result<Row, postgres::Error> {
        pg_block(|| self.raw().query_one(query, params))
    }

    fn query_opt(
        &mut self,
        query: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> std::result::Result<Option<Row>, postgres::Error> {
        pg_block(|| self.raw().query_opt(query, params))
    }

    fn simple_query(
        &mut self,
        query: &str,
    ) -> std::result::Result<Vec<SimpleQueryMessage>, postgres::Error> {
        pg_block(|| self.raw().simple_query(query))
    }

    fn transaction(&mut self) -> std::result::Result<StoreTransaction<'_>, postgres::Error> {
        let transaction = pg_block(|| self.raw().transaction())?;
        Ok(StoreTransaction {
            transaction: Some(transaction),
        })
    }
}

impl Drop for StoreClient<'_> {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            if let Ok(mut clients) = self.pool.lock() {
                clients.push(client);
            } else {
                close_client(client);
            }
        }
    }
}

struct StoreTransaction<'a> {
    transaction: Option<Transaction<'a>>,
}

impl StoreTransaction<'_> {
    fn execute(
        &mut self,
        query: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> std::result::Result<u64, postgres::Error> {
        pg_block(|| {
            self.transaction
                .as_mut()
                .expect("PostgreSQL transaction was already closed")
                .execute(query, params)
        })
    }

    fn commit(mut self) -> std::result::Result<(), postgres::Error> {
        let transaction = self
            .transaction
            .take()
            .expect("PostgreSQL transaction was already closed");
        pg_block(move || transaction.commit())
    }
}

impl Drop for StoreTransaction<'_> {
    fn drop(&mut self) {
        if let Some(transaction) = self.transaction.take() {
            pg_block(move || drop(transaction));
        }
    }
}

fn memory_select_sql(score_expr: &str) -> String {
    format!(
        "SELECT m.id, m.project_id, m.scope, m.memory_tier, m.kind, m.body, m.tags, m.source, m.status,
                m.importance, m.confidence, m.superseded_by, m.status_reason,
                {score_expr} AS score, m.quality_score, m.usage_count,
                ts(m.last_used_at), m.contradiction_risk, ts(m.created_at), ts(m.updated_at)
         FROM memories m"
    )
}

fn memory_fragment_select_sql(score_expr: &str) -> String {
    format!(
        "SELECT m.id, m.project_id, m.scope, m.memory_tier, m.kind, f.text AS body,
                m.tags, m.source, m.status, m.importance, m.confidence, m.superseded_by,
                m.status_reason, {score_expr} AS score, m.quality_score, m.usage_count,
                ts(m.last_used_at), m.contradiction_risk, ts(m.created_at), ts(m.updated_at)
         FROM memory_fragments f
         JOIN memories m ON m.id = f.memory_id"
    )
}

fn code_symbol_select_sql(score_expr: &str) -> String {
    format!(
        "SELECT s.id, s.project_id, s.file_path, s.language, s.name, s.kind,
                s.signature, s.body, s.start_line, s.end_line, s.parent_id,
                {score_expr} AS score
         FROM code_symbols s"
    )
}

fn append_pg_filters(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql + Sync>>,
    status: StatusFilter,
    kind: Option<String>,
) {
    if let StatusFilter::One(status) = status {
        push_param(sql, params, status.as_str().to_string());
        sql.push_str(" AND m.status = $");
        sql.push_str(&params.len().to_string());
    }
    if let Some(kind) = kind {
        push_param(sql, params, kind);
        sql.push_str(" AND m.kind = $");
        sql.push_str(&params.len().to_string());
    }
}

fn append_memory_tier_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql + Sync>>,
    memory_tier: Option<String>,
) -> Result<()> {
    if let Some(memory_tier) = memory_tier {
        push_param(sql, params, normalize_memory_tier(&memory_tier)?);
        sql.push_str(" AND m.memory_tier = $");
        sql.push_str(&params.len().to_string());
    }
    Ok(())
}

fn push_param<T>(_: &mut String, params: &mut Vec<Box<dyn ToSql + Sync>>, value: T)
where
    T: ToSql + Sync + 'static,
{
    params.push(Box::new(value));
}

fn param_refs(params: &[Box<dyn ToSql + Sync>]) -> Vec<&(dyn ToSql + Sync)> {
    params
        .iter()
        .map(|value| value.as_ref() as &(dyn ToSql + Sync))
        .collect()
}

fn memory_from_row(row: &Row) -> Result<Memory> {
    let tags: Value = row.get(6);
    let superseded_by: Option<Uuid> = row.get(11);
    Ok(Memory {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        scope: row.get(2),
        memory_tier: row.get(3),
        kind: row.get(4),
        body: row.get(5),
        tags: string_vec_from_json(tags),
        source: row.get(7),
        status: row.get(8),
        importance: row.get(9),
        confidence: row.get(10),
        superseded_by: superseded_by.map(|id| id.to_string()),
        status_reason: row.get(12),
        score: row.get(13),
        quality_score: row.get(14),
        usage_count: row.get::<_, i64>(15) as u64,
        last_used_at: row.get(16),
        contradiction_risk: row.get(17),
        created_at: row.get(18),
        updated_at: row.get(19),
    })
}

fn eval_case_from_row(row: &Row) -> Result<EvalCaseRecord> {
    Ok(EvalCaseRecord {
        id: row.get::<_, Uuid>(0).to_string(),
        suite_id: row.get::<_, Uuid>(1).to_string(),
        project_id: row.get(2),
        name: row.get(3),
        query: row.get(4),
        expected_contains: string_vec_from_json(row.get(5)),
        forbidden_contains: string_vec_from_json(row.get(6)),
        min_results: row.get::<_, Option<i32>>(7).map(|value| value as u64),
        expected_ids: string_vec_from_json(row.get(8)),
        forbidden_ids: string_vec_from_json(row.get(9)),
        created_at: row.get(10),
    })
}

fn project_record_from_row(row: &Row) -> Result<ProjectRecord> {
    let domains: Value = row.get(5);
    Ok(ProjectRecord {
        id: row.get(0),
        name: row.get(1),
        root_path: row.get(2),
        project_type: row.get(3),
        description: row.get(4),
        domains: string_vec_from_json(domains),
        created_at: row.get(6),
        updated_at: row.get(7),
    })
}

fn exported_memory_from_row(row: &Row) -> Result<ExportedMemory> {
    let tags: Value = row.get(6);
    let superseded_by: Option<Uuid> = row.get(11);
    Ok(ExportedMemory {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        scope: row.get(2),
        memory_tier: row.get(3),
        kind: row.get(4),
        body: row.get(5),
        tags: string_vec_from_json(tags),
        source: row.get(7),
        status: row.get(8),
        importance: row.get(9),
        confidence: row.get(10),
        superseded_by: superseded_by.map(|id| id.to_string()),
        status_reason: row.get(12),
        created_at: row.get(13),
        updated_at: row.get(14),
    })
}

fn code_file_from_row(row: &Row) -> CodeFile {
    CodeFile {
        project_id: row.get(0),
        path: row.get(1),
        language: row.get(2),
        hash: row.get(3),
        size_bytes: row.get::<_, i64>(4) as u64,
        line_count: row.get::<_, i32>(5) as u32,
    }
}

fn code_symbol_from_row(row: &Row) -> CodeSymbol {
    CodeSymbol {
        id: row.get(0),
        project_id: row.get(1),
        file_path: row.get(2),
        language: row.get(3),
        name: row.get(4),
        kind: row.get(5),
        signature: row.get(6),
        body: row.get(7),
        start_line: row.get::<_, i32>(8) as u32,
        end_line: row.get::<_, i32>(9) as u32,
        parent_id: row.get(10),
    }
}

fn code_search_result_from_row(row: &Row) -> Result<CodeSearchResult> {
    Ok(CodeSearchResult {
        symbol: code_symbol_from_row(row),
        score: row.get(11),
    })
}

fn code_similarity_pair_from_row(row: &Row) -> CodeSimilarityPair {
    CodeSimilarityPair {
        left: code_symbol_from_offset(row, 0),
        right: code_symbol_from_offset(row, 11),
        similarity: row.get(22),
    }
}

fn code_symbol_from_offset(row: &Row, offset: usize) -> CodeSymbol {
    CodeSymbol {
        id: row.get(offset),
        project_id: row.get(offset + 1),
        file_path: row.get(offset + 2),
        language: row.get(offset + 3),
        name: row.get(offset + 4),
        kind: row.get(offset + 5),
        signature: row.get(offset + 6),
        body: row.get(offset + 7),
        start_line: row.get::<_, i32>(offset + 8) as u32,
        end_line: row.get::<_, i32>(offset + 9) as u32,
        parent_id: row.get(offset + 10),
    }
}

fn code_relation_from_row(row: &Row) -> CodeRelation {
    CodeRelation {
        id: row.get(0),
        project_id: row.get(1),
        from_symbol_id: row.get(2),
        from_file_path: row.get(3),
        relation_kind: row.get(4),
        target_name: row.get(5),
        target_symbol_id: row.get(6),
    }
}

fn code_memory_from_row(row: &Row) -> Result<CodeMemory> {
    let id: Uuid = row.get(0);
    let tags: Value = row.get(7);
    let mut link_status: String = row.get(4);
    let symbol_body_hash: Option<String> = row.get(22);
    let current_symbol_body: Option<String> = row.get(27);
    if link_status == "symbol"
        && let (Some(snapshot_hash), Some(current_body)) =
            (symbol_body_hash.as_deref(), current_symbol_body.as_deref())
        && snapshot_hash != memory_body_hash(current_body)
    {
        link_status = "symbol_changed".to_string();
    }
    Ok(CodeMemory {
        id: id.to_string(),
        project_id: row.get(1),
        symbol_id: row.get(2),
        file_path: row.get(3),
        link_status,
        symbol_name: row.get(15),
        symbol_kind: row.get(16),
        symbol_signature: row.get(17),
        symbol_start_line: row.get::<_, Option<i32>>(18).map(|line| line as u32),
        symbol_end_line: row.get::<_, Option<i32>>(19).map(|line| line as u32),
        symbol_body_hash,
        last_relinked_at: row.get(20),
        relink_attempts: row.get::<_, i32>(21) as u32,
        kind: row.get(5),
        body: row.get(6),
        tags: string_vec_from_json(tags),
        source: row.get(8),
        status: row.get(9),
        confidence: row.get(10),
        quality_score: row.get(23),
        usage_count: row.get::<_, i64>(24) as u64,
        last_used_at: row.get(25),
        contradiction_risk: row.get(26),
        status_reason: row.get(11),
        score: row.get(12),
        created_at: row.get(13),
        updated_at: row.get(14),
    })
}

fn is_test_file(file_path: &str) -> bool {
    let path = file_path.to_ascii_lowercase();
    path.contains("/test/")
        || path.contains("/tests/")
        || path.contains("__tests__")
        || path.ends_with("_test.rs")
        || path.ends_with("_test.go")
        || path.ends_with("test.py")
        || path.ends_with("_spec.rb")
        || path.ends_with(".spec.ts")
        || path.ends_with(".spec.tsx")
        || path.ends_with(".test.ts")
        || path.ends_with(".test.tsx")
        || path.ends_with("tests.rs")
}

fn route_hints_for_symbol(symbol: &CodeSymbol) -> Vec<CodeRouteHint> {
    let mut hints = Vec::new();
    if let Some((framework, route, method, evidence)) = file_based_route_hint(symbol) {
        hints.push(CodeRouteHint {
            project_id: symbol.project_id.clone(),
            file_path: symbol.file_path.clone(),
            symbol_id: symbol.id.clone(),
            symbol_name: symbol.name.clone(),
            framework,
            route,
            method,
            evidence,
        });
    }
    for line in symbol.body.lines() {
        let trimmed = line.trim();
        if let Some((framework, route, method)) = decorator_route_hint(trimmed) {
            hints.push(CodeRouteHint {
                project_id: symbol.project_id.clone(),
                file_path: symbol.file_path.clone(),
                symbol_id: symbol.id.clone(),
                symbol_name: symbol.name.clone(),
                framework,
                route,
                method,
                evidence: trimmed.chars().take(160).collect(),
            });
        }
        if hints.len() >= 8 {
            break;
        }
    }
    hints
}

fn file_based_route_hint(symbol: &CodeSymbol) -> Option<(String, String, Option<String>, String)> {
    let path = symbol.file_path.as_str();
    if (path.contains("/pages/") || path.starts_with("pages/") || path.contains("/app/"))
        && (path.ends_with(".tsx")
            || path.ends_with(".jsx")
            || path.ends_with(".ts")
            || path.ends_with(".js")
            || path.ends_with(".svelte")
            || path.ends_with(".vue")
            || path.ends_with(".astro"))
    {
        let route = path_to_route(path);
        if !route.is_empty() {
            return Some((
                "file_based_web".to_string(),
                route,
                None,
                "file path route convention".to_string(),
            ));
        }
    }
    if path.contains("/api/") && (path.ends_with(".ts") || path.ends_with(".js")) {
        return Some((
            "file_based_api".to_string(),
            path_to_route(path),
            None,
            "api file route convention".to_string(),
        ));
    }
    None
}

fn path_to_route(path: &str) -> String {
    let mut route = path
        .replace('\\', "/")
        .replace("src/pages/", "/")
        .replace("pages/", "/")
        .replace("src/app/", "/")
        .replace("app/", "/")
        .replace("src/routes/", "/")
        .replace("routes/", "/");
    for suffix in [
        ".tsx", ".jsx", ".ts", ".js", ".svelte", ".vue", ".astro", "/page", "/route",
    ] {
        route = route.replace(suffix, "");
    }
    route = route.replace("/index", "/");
    while route.contains("//") {
        route = route.replace("//", "/");
    }
    if !route.starts_with('/') {
        route.insert(0, '/');
    }
    route
}

fn decorator_route_hint(line: &str) -> Option<(String, String, Option<String>)> {
    let lower = line.to_ascii_lowercase();
    for method in ["get", "post", "put", "patch", "delete", "options", "head"] {
        for prefix in [
            format!("@app.{method}("),
            format!("@router.{method}("),
            format!("app.{method}("),
            format!("router.{method}("),
            format!("r.{method}("),
            format!("r.{}(", method.to_ascii_uppercase()),
        ] {
            if lower.contains(&prefix.to_ascii_lowercase())
                && let Some(route) = first_quoted_literal(line)
            {
                let framework = if line.starts_with('@') {
                    "python_web"
                } else {
                    "javascript_or_go_web"
                };
                return Some((
                    framework.to_string(),
                    route,
                    Some(method.to_ascii_uppercase()),
                ));
            }
        }
    }
    if lower.contains("path(")
        && let Some(route) = first_quoted_literal(line)
    {
        return Some(("django".to_string(), route, None));
    }
    if lower.contains(".route(")
        && let Some(route) = first_quoted_literal(line)
    {
        return Some(("rust_web".to_string(), route, None));
    }
    None
}

fn first_quoted_literal(line: &str) -> Option<String> {
    let mut quote = None;
    let mut start = 0usize;
    for (index, ch) in line.char_indices() {
        if ch == '\'' || ch == '"' || ch == '`' {
            match quote {
                None => {
                    quote = Some(ch);
                    start = index + ch.len_utf8();
                }
                Some(expected) if expected == ch => {
                    let value = line[start..index].trim();
                    if value.starts_with('/') || !value.is_empty() {
                        return Some(value.to_string());
                    }
                    quote = None;
                }
                _ => {}
            }
        }
    }
    None
}

fn memory_entity_from_row(row: &Row) -> Result<MemoryEntity> {
    let aliases: Value = row.get(4);
    Ok(MemoryEntity {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        entity_type: row.get(2),
        name: row.get(3),
        aliases: string_vec_from_json(aliases),
        description: row.get(5),
        created_at: row.get(6),
        updated_at: row.get(7),
    })
}

fn memory_episode_from_row(row: &Row) -> Result<MemoryEpisode> {
    Ok(MemoryEpisode {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        source: row.get(2),
        summary: row.get(3),
        raw_ref: row.get(4),
        raw_payload: row.get(5),
        observed_at: row.get(6),
        created_at: row.get(7),
    })
}

fn memory_fact_from_row(row: &Row) -> Result<MemoryFact> {
    let entity_id: Option<Uuid> = row.get(2);
    let memory_id: Option<Uuid> = row.get(3);
    let episode_id: Option<Uuid> = row.get(4);
    let invalidated_by: Option<Uuid> = row.get(10);
    Ok(MemoryFact {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        entity_id: entity_id.map(|id| id.to_string()),
        memory_id: memory_id.map(|id| id.to_string()),
        episode_id: episode_id.map(|id| id.to_string()),
        predicate: row.get(5),
        value: row.get(6),
        confidence: row.get(7),
        valid_from: row.get(8),
        valid_to: row.get(9),
        invalidated_by: invalidated_by.map(|id| id.to_string()),
        observed_at: row.get(11),
    })
}

fn memory_edge_from_row(row: &Row) -> MemoryEdge {
    let memory_id: Option<Uuid> = row.get(7);
    let episode_id: Option<Uuid> = row.get(8);
    let invalidated_by: Option<Uuid> = row.get(12);
    MemoryEdge {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        from_entity_id: row.get::<_, Uuid>(2).to_string(),
        from_entity_name: row.get(3),
        to_entity_id: row.get::<_, Uuid>(4).to_string(),
        to_entity_name: row.get(5),
        relation_type: row.get(6),
        memory_id: memory_id.map(|id| id.to_string()),
        episode_id: episode_id.map(|id| id.to_string()),
        confidence: row.get(9),
        valid_from: row.get(10),
        valid_to: row.get(11),
        invalidated_by: invalidated_by.map(|id| id.to_string()),
        observed_at: row.get(13),
    }
}

fn audit_event_from_row(row: &Row) -> AuditEvent {
    AuditEvent {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        actor: row.get(2),
        action: row.get(3),
        target_type: row.get(4),
        target_id: row.get(5),
        detail: row.get(6),
        created_at: row.get(7),
    }
}

fn retrieval_event_from_row(row: &Row) -> Result<RetrievalEvent> {
    Ok(RetrievalEvent {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        task_session_id: row.get::<_, Option<Uuid>>(2).map(|id| id.to_string()),
        tool: row.get(3),
        query: row.get(4),
        task_type: row.get(5),
        token_budget: row.get::<_, i32>(6).max(0) as usize,
        estimated_tokens: row.get::<_, i32>(7).max(0) as usize,
        latency_ms: row.get::<_, i64>(8).max(0) as u64,
        memory_fragments: row.get::<_, i32>(9).max(0) as usize,
        code_hits: row.get::<_, i32>(10).max(0) as usize,
        graph_items: row.get::<_, i32>(11).max(0) as usize,
        code_memories: row.get::<_, i32>(12).max(0) as usize,
        plan: row.get(13),
        audit: row.get(14),
        created_at: row.get(15),
    })
}

fn eval_run_summary_from_row(row: &Row) -> EvalRunSummary {
    EvalRunSummary {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        suite_name: row.get(2),
        suite_hash: row.get(3),
        mode: row.get(4),
        total_cases: row.get::<_, i32>(5) as usize,
        passed_cases: row.get::<_, i32>(6) as usize,
        failed_cases: row.get::<_, i32>(7) as usize,
        created_at: row.get(8),
    }
}

fn task_session_from_row(row: &Row) -> Result<TaskSession> {
    Ok(TaskSession {
        id: row.get::<_, Uuid>(0).to_string(),
        project_id: row.get(1),
        query: row.get(2),
        status: row.get(3),
        phase: row.get(4),
        progress: row.get::<_, i32>(5) as usize,
        memory_ids: string_vec_from_json(row.get(6)),
        code_symbol_ids: string_vec_from_json(row.get(7)),
        file_paths: string_vec_from_json(row.get(8)),
        test_paths: string_vec_from_json(row.get(9)),
        summary: row.get(10),
        result: row.get(11),
        created_at: row.get(12),
        updated_at: row.get(13),
        completed_at: row.get(14),
    })
}

fn validate_task_session_status(status: &str) -> Result<()> {
    match status {
        "planned" | "running" | "completed" | "failed" | "archived" => Ok(()),
        other => bail!("invalid task session status `{other}`"),
    }
}

fn validate_task_session_phase(phase: &str) -> Result<()> {
    let trimmed = phase.trim();
    if trimmed.is_empty() {
        bail!("task session phase cannot be empty");
    }
    if trimmed.chars().count() > 80 {
        bail!("task session phase is too long");
    }
    Ok(())
}

fn validate_export_scope(export: &ProjectExport) -> Result<()> {
    let project_id = &export.project.id;
    for memory in &export.memories {
        if &memory.project_id != project_id {
            bail!(
                "memory `{}` belongs to project `{}`, but export project is `{}`",
                memory.id,
                memory.project_id,
                project_id
            );
        }
    }
    for file in &export.code_files {
        if &file.project_id != project_id {
            bail!(
                "code file `{}` belongs to project `{}`, but export project is `{}`",
                file.path,
                file.project_id,
                project_id
            );
        }
    }
    for symbol in &export.code_symbols {
        if &symbol.project_id != project_id {
            bail!(
                "code symbol `{}` belongs to project `{}`, but export project is `{}`",
                symbol.id,
                symbol.project_id,
                project_id
            );
        }
    }
    for relation in &export.code_relations {
        if &relation.project_id != project_id {
            bail!(
                "code relation `{}` belongs to project `{}`, but export project is `{}`",
                relation.id,
                relation.project_id,
                project_id
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SymbolResolver {
    by_file_name: HashMap<(String, String), Vec<CodeSymbol>>,
    by_name: HashMap<String, Vec<CodeSymbol>>,
    module_file_by_name: HashMap<(String, String), String>,
    module_names: HashSet<String>,
    impl_methods: HashMap<(String, String), Vec<CodeSymbol>>,
}

impl SymbolResolver {
    fn new(symbols: Vec<CodeSymbol>) -> Self {
        let mut by_file_name: HashMap<(String, String), Vec<CodeSymbol>> = HashMap::new();
        let mut by_name: HashMap<String, Vec<CodeSymbol>> = HashMap::new();
        let mut module_file_by_name = HashMap::new();
        let mut module_names = HashSet::new();
        let mut by_id = HashMap::new();
        for symbol in &symbols {
            by_file_name
                .entry((symbol.file_path.clone(), symbol.name.clone()))
                .or_default()
                .push(symbol.clone());
            by_name
                .entry(symbol.name.clone())
                .or_default()
                .push(symbol.clone());
            if symbol.kind == "module"
                && let Some(module_name) = module_name_from_file_path(&symbol.file_path)
            {
                module_names.insert(module_name.clone());
                module_file_by_name
                    .entry((symbol.file_path.clone(), module_name))
                    .or_insert_with(|| symbol.id.clone());
            }
            by_id.insert(symbol.id.clone(), symbol.clone());
        }
        let mut impl_methods: HashMap<(String, String), Vec<CodeSymbol>> = HashMap::new();
        for symbol in &symbols {
            if !matches!(symbol.kind.as_str(), "function" | "method") {
                continue;
            }
            let Some(parent_id) = symbol.parent_id.as_deref() else {
                continue;
            };
            let Some(parent) = by_id.get(parent_id) else {
                continue;
            };
            if parent.kind != "impl" {
                continue;
            }
            let Some(type_name) = impl_target_type_name(&parent.name)
                .or_else(|| impl_target_type_name(&parent.signature))
            else {
                continue;
            };
            impl_methods
                .entry((type_name, symbol.name.clone()))
                .or_default()
                .push(symbol.clone());
        }
        Self {
            by_file_name,
            by_name,
            module_file_by_name,
            module_names,
            impl_methods,
        }
    }

    fn resolve_call(&self, from_file_path: &str, target_name: &str) -> Option<String> {
        if let Some(id) = self.resolve_qualified_target(target_name, &["function", "method"]) {
            return Some(id);
        }
        let target_name = last_rust_path_identifier(target_name)?;
        if let Some(id) =
            self.unique_local_symbol(from_file_path, target_name, &["function", "method"])
        {
            return Some(id);
        }
        self.unique_global_symbol(target_name, &["function", "method"])
    }

    fn resolve_use(&self, target_name: &str) -> Option<String> {
        if target_name.contains("::") {
            if let Some(id) = self.resolve_qualified_target(
                target_name,
                &["struct", "enum", "trait", "function", "module"],
            ) {
                return Some(id);
            }
            if !self.is_project_qualified_path(target_name) {
                return None;
            }
        }
        let target_name = last_rust_path_identifier(target_name)?;
        self.unique_global_symbol(
            target_name,
            &["struct", "enum", "trait", "function", "module"],
        )
    }

    fn resolve_module_declaration(
        &self,
        from_file_path: &str,
        target_name: &str,
    ) -> Option<String> {
        self.module_file_by_name
            .get(&(from_file_path.to_string(), target_name.to_string()))
            .cloned()
            .or_else(|| {
                module_name_from_file_path(target_name)
                    .and_then(|module_name| self.unique_global_symbol(&module_name, &["module"]))
            })
            .or_else(|| self.unique_global_symbol(target_name, &["module"]))
    }

    fn resolve_qualified_target(&self, target_name: &str, kinds: &[&str]) -> Option<String> {
        let (module_path, symbol_name) = target_name.rsplit_once("::")?;
        let module_name = module_path.rsplit("::").next()?.trim();
        if module_name.is_empty() || symbol_name.trim().is_empty() {
            return None;
        }
        if let Some(id) = self.unique_impl_method(module_name, symbol_name.trim(), kinds) {
            return Some(id);
        }
        self.by_name
            .get(symbol_name.trim())
            .and_then(|symbols| {
                let matches = symbols
                    .iter()
                    .filter(|symbol| {
                        kinds.contains(&symbol.kind.as_str())
                            && file_path_matches_module(&symbol.file_path, module_name)
                    })
                    .collect::<Vec<_>>();
                (matches.len() == 1).then(|| matches[0].id.clone())
            })
            .or_else(|| {
                if symbol_name
                    .trim()
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_ascii_uppercase())
                {
                    self.unique_global_symbol(module_name, &["enum"])
                } else {
                    None
                }
            })
    }

    fn unique_local_symbol(&self, file_path: &str, name: &str, kinds: &[&str]) -> Option<String> {
        self.by_file_name
            .get(&(file_path.to_string(), name.to_string()))
            .and_then(|symbols| unique_symbol_id(symbols, kinds))
    }

    fn unique_global_symbol(&self, name: &str, kinds: &[&str]) -> Option<String> {
        self.by_name
            .get(name)
            .and_then(|symbols| unique_symbol_id(symbols, kinds))
    }

    fn unique_impl_method(&self, type_name: &str, name: &str, kinds: &[&str]) -> Option<String> {
        self.impl_methods
            .get(&(type_name.to_string(), name.to_string()))
            .and_then(|symbols| unique_symbol_id(symbols, kinds))
    }

    fn is_project_qualified_path(&self, target_name: &str) -> bool {
        let Some(root) = first_rust_path_identifier(target_name) else {
            return false;
        };
        matches!(root, "crate" | "self" | "super") || self.module_names.contains(root)
    }
}

fn unique_symbol_id(symbols: &[CodeSymbol], kinds: &[&str]) -> Option<String> {
    let matches = symbols
        .iter()
        .filter(|symbol| kinds.contains(&symbol.kind.as_str()))
        .collect::<Vec<_>>();
    (matches.len() == 1).then(|| matches[0].id.clone())
}

fn last_rust_path_identifier(value: &str) -> Option<&str> {
    value
        .rsplit("::")
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn first_rust_path_identifier(value: &str) -> Option<&str> {
    value
        .trim_start_matches("::")
        .split("::")
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn impl_target_type_name(value: &str) -> Option<String> {
    let mut rest = value.trim().strip_prefix("impl")?.trim();
    if let Some((_, target)) = rest.rsplit_once(" for ") {
        rest = target.trim();
    } else if rest.starts_with('<')
        && let Some(end) = matching_angle_end(rest)
    {
        rest = rest[end + 1..].trim();
    }
    let rest = rest
        .split(" where ")
        .next()
        .unwrap_or(rest)
        .split('{')
        .next()
        .unwrap_or(rest)
        .trim();
    let rest = trim_type_generics(rest)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim();
    last_rust_path_identifier(rest)
        .map(trim_type_generics)
        .map(str::to_string)
        .filter(|value| !value.is_empty())
}

fn matching_angle_end(value: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (index, ch) in value.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn trim_type_generics(value: &str) -> &str {
    value.split('<').next().unwrap_or(value).trim()
}

fn module_name_from_file_path(file_path: &str) -> Option<String> {
    let path = Path::new(file_path);
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| *stem != "mod" && *stem != "lib" && *stem != "main")
        .map(str::to_string)
}

fn file_path_matches_module(file_path: &str, module_name: &str) -> bool {
    let path = Path::new(file_path);
    if path.file_stem().and_then(|stem| stem.to_str()) == Some(module_name) {
        return true;
    }
    path.parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|parent| parent == module_name && path.file_stem() == Some("mod".as_ref()))
}

fn to_tsquery(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_tags(tags: Vec<String>) -> Vec<String> {
    let mut tags = tags
        .into_iter()
        .map(|tag| tag.trim().to_ascii_lowercase())
        .filter(|tag| !tag.is_empty())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    tags
}

fn default_memory_scope() -> String {
    DEFAULT_MEMORY_SCOPE.to_string()
}

fn default_memory_tier() -> String {
    DEFAULT_MEMORY_TIER.to_string()
}

pub fn normalize_project_type(value: &str) -> String {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let normalized = normalized.trim_matches('_').to_string();
    if normalized.is_empty() {
        DEFAULT_PROJECT_TYPE.to_string()
    } else {
        normalized
    }
}

pub fn normalize_memory_scope(value: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if MEMORY_SCOPES.contains(&normalized.as_str()) {
        Ok(normalized)
    } else {
        bail!(
            "invalid memory scope `{value}`; use one of: {}",
            MEMORY_SCOPES.join(", ")
        )
    }
}

pub fn normalize_memory_tier(value: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "core" | "archival" | "conversation") {
        Ok(normalized)
    } else {
        bail!("invalid memory tier `{value}`; use one of: core, archival, conversation")
    }
}

fn normalize_code_memory_status(value: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "pending" | "active" | "archived") {
        Ok(normalized)
    } else {
        bail!("invalid code memory status `{value}`; use one of: pending, active, archived")
    }
}

fn normalize_embedding_kind(value: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        bail!("embedding kind cannot be empty");
    }
    if normalized
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        Ok(normalized)
    } else {
        bail!("invalid embedding kind `{value}`; use letters, numbers, underscore, or dash")
    }
}

fn required_memory_embedding_kinds() -> Vec<String> {
    vec!["body".to_string(), "metadata".to_string()]
}

fn required_code_embedding_kinds() -> Vec<String> {
    vec!["body".to_string(), "signature".to_string()]
}

fn memory_body_hash(body: &str) -> String {
    let normalized = body
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    blake3::hash(normalized.as_bytes()).to_hex().to_string()
}

fn sync_memory_fragments(
    client: &mut StoreClient<'_>,
    project_id: &str,
    memory_id: Uuid,
    body_hash: &str,
    body: &str,
) -> Result<usize> {
    client.execute(
        "DELETE FROM memory_fragments WHERE memory_id = $1",
        &[&memory_id],
    )?;
    let chunks = memory_fragment_chunks(body);
    for (index, text) in chunks.iter().enumerate() {
        let fragment_id = format!("{memory_id}#{}", index + 1);
        client.execute(
            "INSERT INTO memory_fragments (
                id, memory_id, project_id, chunk_index, body_hash, text, updated_at
             )
             VALUES ($1, $2, $3, $4, $5, $6, now())
             ON CONFLICT(id) DO UPDATE SET
                body_hash = excluded.body_hash,
                text = excluded.text,
                updated_at = now()",
            &[
                &fragment_id,
                &memory_id,
                &project_id,
                &(index as i32),
                &body_hash,
                text,
            ],
        )?;
    }
    Ok(chunks.len())
}

fn memory_fragment_chunks(body: &str) -> Vec<String> {
    const MAX_FRAGMENT_CHARS: usize = 700;
    let mut chunks = Vec::new();
    for paragraph in body.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        let chars = paragraph.chars().collect::<Vec<_>>();
        if chars.len() <= MAX_FRAGMENT_CHARS {
            chunks.push(paragraph.to_string());
            continue;
        }
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + MAX_FRAGMENT_CHARS).min(chars.len());
            let text = chars[start..end].iter().collect::<String>();
            if !text.trim().is_empty() {
                chunks.push(text.trim().to_string());
            }
            start = end;
        }
    }
    if chunks.is_empty() && !body.trim().is_empty() {
        chunks.push(body.trim().chars().take(MAX_FRAGMENT_CHARS).collect());
    }
    chunks
}

fn validate_score(name: &str, value: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&value) {
        bail!("{name} must be between 0.0 and 1.0");
    }
    Ok(())
}

fn json_array(values: Vec<String>) -> Value {
    Value::Array(values.into_iter().map(Value::String).collect())
}

fn string_vec_from_json(value: Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect()
}

fn parse_uuid(value: &str) -> Result<Uuid> {
    Uuid::parse_str(value).with_context(|| format!("invalid UUID `{value}`"))
}

fn default_database_url() -> String {
    let user = env::var("USER").unwrap_or_else(|_| "dukememory".to_string());
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!(
        "postgresql://{user}@localhost:55432/dukememory?host={home}/.dukememory/postgres-socket"
    )
}

fn connect_postgres(database_url: &str) -> Result<Client> {
    pg_block(|| {
        Client::connect(database_url, NoTls)
            .with_context(|| format!("failed to connect to PostgreSQL at {database_url}"))
    })
}

fn migration_cache() -> &'static Mutex<HashSet<String>> {
    static MIGRATED_SCHEMAS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    MIGRATED_SCHEMAS.get_or_init(|| Mutex::new(HashSet::new()))
}

fn migration_cache_key(database_url: &str, schema: Option<&str>) -> String {
    format!("{}::{}", database_url, schema.unwrap_or("public"))
}

fn ensure_migrations_applied(client: &mut Client, migration_key: &str) -> Result<()> {
    let mut migrated = migration_cache()
        .lock()
        .map_err(|_| anyhow!("migration cache mutex is poisoned"))?;
    if migrated.contains(migration_key) {
        return Ok(());
    }

    let migration_sql = concat!(
        include_str!("../migrations/0001_initial_pg.sql"),
        "
",
        include_str!("../migrations/0002_context_layers.sql"),
        "
",
        include_str!("../migrations/0003_eval_and_graph_ops.sql"),
        "
",
        include_str!("../migrations/0004_audit_eval_ops.sql"),
        "
",
        include_str!("../migrations/0005_code_memories.sql"),
        "
",
        include_str!("../migrations/0006_code_memory_lifecycle.sql"),
        "
",
        include_str!("../migrations/0007_code_memory_symbol_snapshot.sql"),
        "
",
        include_str!("../migrations/0008_memory_performance_indexes.sql"),
        "
",
        include_str!("../migrations/0009_memory_fragments.sql"),
        "
",
        include_str!("../migrations/0010_code_symbol_embedding_cache.sql"),
        "
",
        include_str!("../migrations/0011_versioned_multi_vector_embeddings.sql"),
        "
	",
        include_str!("../migrations/0012_context_planner_quality.sql"),
        "
	",
        include_str!("../migrations/0013_agent_task_sessions.sql"),
        "
	",
        include_str!("../migrations/0014_feedback_eval_forbidden_ids.sql"),
        "
	",
        include_str!("../migrations/0015_retrieval_event_task_session.sql")
    );
    for attempt in 0..3 {
        pg_block(|| {
            client
                .batch_execute("SELECT pg_advisory_lock(hashtext('dukememory_schema_migrations'))")
        })?;
        let result = pg_block(|| client.batch_execute(migration_sql));
        let unlock_result = pg_block(|| {
            client.batch_execute(
                "SELECT pg_advisory_unlock(hashtext('dukememory_schema_migrations'))",
            )
        });
        unlock_result.context("failed to release dukememory schema migration advisory lock")?;
        match result {
            Ok(()) => {
                migrated.insert(migration_key.to_string());
                return Ok(());
            }
            Err(error) => {
                let message = error.to_string();
                let retryable = message.contains("tuple concurrently updated")
                    || message.contains("deadlock detected")
                    || message.contains("could not serialize access");
                if retryable && attempt < 2 {
                    std::thread::sleep(std::time::Duration::from_millis(
                        100 * (attempt as u64 + 1),
                    ));
                    continue;
                }
                return Err(error).context("failed to apply PostgreSQL migrations");
            }
        }
    }
    migrated.insert(migration_key.to_string());
    Ok(())
}

fn pg_block<T, F>(operation: F) -> T
where
    F: FnOnce() -> T,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(operation)
    } else {
        operation()
    }
}

fn close_client(client: Client) {
    pg_block(move || drop(client));
}

fn setup_connection(client: &mut Client, schema: Option<&str>) -> Result<()> {
    if let Some(schema) = schema {
        let schema = quote_ident(schema)?;
        pg_block(|| {
            client.batch_execute(&format!(
                "CREATE SCHEMA IF NOT EXISTS {schema};
                 SET search_path TO {schema}, public;"
            ))
        })?;
    }
    Ok(())
}

fn schema_for_path(path: &Path) -> Result<Option<String>> {
    if let Ok(schema) = env::var("DUKEMEMORY_DATABASE_SCHEMA")
        && !schema.trim().is_empty()
    {
        return Ok(Some(normalize_schema_name(&schema)?));
    }

    let Some(home) = env::var_os("HOME") else {
        return Ok(None);
    };
    let default_path = Path::new(&home).join(".dukememory").join("schema.marker");
    if path == default_path {
        return Ok(None);
    }

    let path_text = path.to_string_lossy();
    if path_text.trim().is_empty() {
        return Ok(None);
    }
    let hash = blake3::hash(path_text.as_bytes()).to_hex().to_string();
    Ok(Some(format!("dm_{}", &hash[..24])))
}

fn normalize_schema_name(value: &str) -> Result<String> {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let normalized = normalized.trim_matches('_').to_string();
    if normalized.is_empty() {
        bail!("DUKEMEMORY_DATABASE_SCHEMA cannot be empty");
    }
    Ok(normalized)
}

fn quote_ident(value: &str) -> Result<String> {
    if value.is_empty()
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        || value.chars().next().is_some_and(|ch| ch.is_ascii_digit())
    {
        bail!("invalid PostgreSQL identifier `{value}`");
    }
    Ok(format!("\"{}\"", value.replace('"', "\"\"")))
}

fn pg_dump_path() -> String {
    env::var("PG_DUMP")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            let homebrew = "/opt/homebrew/opt/postgresql@17/bin/pg_dump";
            if Path::new(homebrew).is_file() {
                homebrew.to_string()
            } else {
                "pg_dump".to_string()
            }
        })
}

fn count_all(client: &mut StoreClient<'_>, table: &str) -> Result<u64> {
    validate_health_table(table)?;
    let row = client.query_one(&format!("SELECT COUNT(*)::BIGINT FROM {table}"), &[])?;
    Ok(row.get::<_, i64>(0) as u64)
}

fn count_distinct_project_keys(
    client: &mut StoreClient<'_>,
    table: &str,
    key_column: &str,
) -> Result<u64> {
    validate_health_table(table)?;
    validate_project_key_column(key_column)?;
    let row = client.query_one(
        &format!("SELECT COUNT(DISTINCT {key_column})::BIGINT FROM {table}"),
        &[],
    )?;
    Ok(row.get::<_, i64>(0) as u64)
}

fn validate_project_key_column(column: &str) -> Result<()> {
    match column {
        "memory_id" | "symbol_id" => Ok(()),
        other => bail!("unknown project key column `{other}`"),
    }
}

fn validate_health_table(table: &str) -> Result<()> {
    match table {
        "projects"
        | "memories"
        | "memory_embeddings_4096"
        | "code_files"
        | "code_symbols"
        | "code_relations"
        | "code_symbol_embeddings_1024"
        | "memory_entities"
        | "memory_facts"
        | "memory_edges"
        | "dukememory_audit_events"
        | "dukememory_task_sessions" => Ok(()),
        other => bail!("unknown health table `{other}`"),
    }
}

fn validate_known_table(table: &str) -> Result<()> {
    match table {
        "code_files" | "code_symbols" | "code_relations" | "code_symbol_embeddings_1024" => Ok(()),
        other => bail!("unknown count table `{other}`"),
    }
}

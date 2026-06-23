use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use uuid::Uuid;

use crate::backup::create_database_backup;
use crate::code_index::{check_code_index_freshness, index_project};
use crate::config::Config;
use crate::maintenance::{MaintenanceOptions, run_maintenance};
use crate::store::{
    CodeSearchOptions, DEFAULT_MEMORY_SCOPE, DEFAULT_MEMORY_TIER, MemoryStatus, NewMemory,
    SearchOptions, StatusFilter, Store,
};

#[derive(Debug, Clone, Serialize)]
pub struct ProductionAuditReport {
    pub project_id: String,
    pub root_path: PathBuf,
    pub database_marker: PathBuf,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub code_symbols: u64,
    pub code_relations: u64,
    pub resolved_relations: u64,
    pub memory_id: String,
    pub memory_hits: usize,
    pub cross_project_hits: usize,
    pub code_hits: usize,
    pub caller_edges: usize,
    pub callee_edges: usize,
    pub backup_path: PathBuf,
    pub backup_size_bytes: u64,
    pub export_path: PathBuf,
    pub export_memories: usize,
    pub export_code_files: usize,
    pub export_code_symbols: usize,
    pub export_code_relations: usize,
    pub import_memories: usize,
    pub import_code_files: usize,
    pub import_code_symbols: usize,
    pub import_code_relations: usize,
    pub maintenance_backup_size_bytes: u64,
    pub maintenance_compaction_status: String,
}

pub async fn run_production_audit(config: &Config) -> Result<ProductionAuditReport> {
    let token = Uuid::now_v7().to_string().replace('-', "");
    let root_path = std::env::temp_dir().join(format!("dukememory-production-audit-{token}"));
    let project_id = format!("production-audit-{token}");
    let database_marker = root_path.join("schema.marker");
    let backup_path = root_path.join("backup.schema-marker");
    let export_path = root_path.join("project-export.json");
    let import_database_marker = root_path.join("import.schema-marker");
    let maintenance_backup_path = root_path.join("maintenance-backup.schema-marker");

    write_synthetic_project(&root_path, &project_id)?;

    let mut audit_config = config.clone();
    audit_config.database_marker = database_marker.clone();

    let mut store = Store::open(&audit_config.database_marker)?;
    let index = index_project(&mut store, &root_path, Some(project_id.clone()), false)?;
    if index.files_indexed < 4 {
        bail!(
            "production audit indexed too few files: {}",
            index.files_indexed
        );
    }
    if index.calls_resolved == 0 || index.modules_resolved == 0 {
        bail!(
            "production audit did not resolve expected code graph edges: calls={}, modules={}",
            index.calls_resolved,
            index.modules_resolved
        );
    }
    let freshness = check_code_index_freshness(&store, &root_path, Some(project_id.clone()))?;
    if !freshness.is_fresh() {
        bail!("production audit code index is stale immediately after indexing: {freshness:?}");
    }

    let memory = store.remember_deduplicated(
        &project_id,
        NewMemory {
            scope: DEFAULT_MEMORY_SCOPE.to_string(),
            memory_tier: DEFAULT_MEMORY_TIER.to_string(),
            kind: "game_design".to_string(),
            body: "Use stamina budget memory for sprinting and melee actions in the audit game."
                .to_string(),
            tags: vec!["audit".to_string(), "gameplay".to_string()],
            source: Some("production_audit".to_string()),
            status: MemoryStatus::Active,
            importance: 0.8,
            confidence: 0.9,
            status_reason: Some("audit fixture".to_string()),
            allow_sensitive: false,
        },
    )?;
    if !memory.inserted {
        bail!("production audit memory was unexpectedly deduplicated");
    }

    let memory_hits = store
        .search(
            &project_id,
            SearchOptions {
                query: "stamina budget sprinting".to_string(),
                limit: 5,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
            },
        )?
        .len();
    if memory_hits == 0 {
        bail!("production audit memory search returned no hits");
    }
    let cross_project_hits = store
        .search(
            "production-audit-other",
            SearchOptions {
                query: "stamina budget sprinting".to_string(),
                limit: 5,
                status: StatusFilter::One(MemoryStatus::Active),
                kind: None,
                memory_tier: None,
            },
        )?
        .len();
    if cross_project_hits != 0 {
        bail!("production audit memory leaked across project ids");
    }

    let code_hits = store
        .search_code(
            &project_id,
            CodeSearchOptions {
                query: "apply damage".to_string(),
                limit: 5,
                kind: Some("function".to_string()),
                file_path: None,
            },
        )?
        .len();
    if code_hits == 0 {
        bail!("production audit code search returned no hits");
    }
    let caller_edges = store.find_callers(&project_id, "apply_damage", 20)?.len();
    let callee_edges = store.find_callees(&project_id, "tick", 20)?.len();
    if caller_edges == 0 || callee_edges == 0 {
        bail!(
            "production audit call graph is incomplete: callers={}, callees={}",
            caller_edges,
            callee_edges
        );
    }

    let backup = create_database_backup(
        &store,
        &audit_config.database_marker,
        Some(backup_path.clone()),
    )?;
    let export = store.export_project(&project_id, true)?;
    if export.memories.len() != 1 || export.code_files.len() < 4 || export.code_symbols.is_empty() {
        bail!(
            "production audit export has unexpected counts: memories={}, code_files={}, code_symbols={}",
            export.memories.len(),
            export.code_files.len(),
            export.code_symbols.len()
        );
    }
    let export_json = serde_json::to_string_pretty(&export)?;
    std::fs::write(&export_path, export_json)
        .with_context(|| format!("failed to write {}", export_path.display()))?;

    let mut import_store = Store::open(&import_database_marker)?;
    let import = import_store.import_project(export.clone(), false)?;
    if import.memories_imported != export.memories.len()
        || import.code_files_imported != export.code_files.len()
        || import.code_symbols_imported != export.code_symbols.len()
        || import.code_relations_imported != export.code_relations.len()
    {
        bail!("production audit import counts did not match export counts: {import:?}");
    }

    let maintenance = run_maintenance(
        &audit_config,
        &project_id,
        MaintenanceOptions {
            apply: false,
            backup: true,
            backup_output: Some(maintenance_backup_path.clone()),
            validate_pending: false,
            validate_limit: 20,
            compact: true,
            compact_limit: 40,
            compact_min_memories: 20,
            feedback: false,
            feedback_limit: 20,
            embed_missing: false,
            embed_limit: 50,
            embed_scope: "all".to_string(),
        },
    )
    .await?;
    let maintenance_backup = maintenance
        .backup
        .context("production audit maintenance did not create backup")?;
    let maintenance_compaction_status = maintenance
        .compaction
        .map(|compaction| compaction.status)
        .unwrap_or_else(|| "not_run".to_string());
    if maintenance_compaction_status != "skipped" {
        bail!(
            "production audit expected dry-run compaction to skip, got {maintenance_compaction_status}"
        );
    }

    let code_status = store.code_status(&project_id)?;
    Ok(ProductionAuditReport {
        project_id,
        root_path,
        database_marker,
        files_indexed: index.files_indexed,
        files_skipped: index.files_skipped,
        code_symbols: code_status.symbols,
        code_relations: code_status.relations,
        resolved_relations: code_status.resolved_relations,
        memory_id: memory.id,
        memory_hits,
        cross_project_hits,
        code_hits,
        caller_edges,
        callee_edges,
        backup_path: backup.output,
        backup_size_bytes: backup.size_bytes,
        export_path,
        export_memories: export.memories.len(),
        export_code_files: export.code_files.len(),
        export_code_symbols: export.code_symbols.len(),
        export_code_relations: export.code_relations.len(),
        import_memories: import.memories_imported,
        import_code_files: import.code_files_imported,
        import_code_symbols: import.code_symbols_imported,
        import_code_relations: import.code_relations_imported,
        maintenance_backup_size_bytes: maintenance_backup.size_bytes,
        maintenance_compaction_status,
    })
}

fn write_synthetic_project(root_path: &Path, project_id: &str) -> Result<()> {
    std::fs::create_dir_all(root_path.join("src"))
        .with_context(|| format!("failed to create {}", root_path.display()))?;
    std::fs::write(
        root_path.join(".dukememory.toml"),
        format!("name = \"{project_id}\"\n"),
    )?;
    std::fs::write(
        root_path.join("Cargo.toml"),
        format!("[package]\nname = \"{project_id}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
    )?;
    std::fs::write(
        root_path.join("src/lib.rs"),
        r#"
mod combat;
mod movement;

pub fn tick() {
    movement::advance();
    combat::apply_damage();
}
"#,
    )?;
    std::fs::write(
        root_path.join("src/movement.rs"),
        r#"
pub fn advance() {
    integrate_velocity();
}

fn integrate_velocity() {}
"#,
    )?;
    std::fs::write(
        root_path.join("src/combat.rs"),
        r#"
pub fn apply_damage() {
    calculate_damage();
}

fn calculate_damage() {}
"#,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn production_audit_runs_on_isolated_database() -> Result<()> {
        let config = Config {
            database_url: "postgresql://dukememory-test@localhost:55432/dukememory_test"
                .to_string(),
            database_marker: std::env::temp_dir().join(format!(
                "dukememory-production-audit-test-{}.schema-marker",
                Uuid::now_v7()
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
        };

        let report = run_production_audit(&config).await?;
        assert_eq!(report.memory_hits, 1);
        assert_eq!(report.cross_project_hits, 0);
        assert!(report.code_symbols > 0);
        assert!(report.resolved_relations > 0);
        assert_eq!(report.export_memories, report.import_memories);
        assert_eq!(report.export_code_symbols, report.import_code_symbols);
        assert_eq!(report.maintenance_compaction_status, "skipped");

        Ok(())
    }
}

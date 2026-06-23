use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::Serialize;

use crate::store::Store;

#[derive(Debug, Clone, Serialize)]
pub struct DatabaseBackupReport {
    pub source: PathBuf,
    pub output: PathBuf,
    pub size_bytes: u64,
}

pub fn create_database_backup(
    store: &Store,
    database_marker: &Path,
    output: Option<PathBuf>,
) -> Result<DatabaseBackupReport> {
    let output = output.unwrap_or_else(|| default_backup_path(database_marker));
    let size_bytes = store.backup_to(&output)?;
    Ok(DatabaseBackupReport {
        source: database_marker.to_path_buf(),
        output,
        size_bytes,
    })
}

pub fn default_backup_path(database_marker: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let file_name = database_marker
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dukememory.pg");
    database_marker.with_file_name(format!("{file_name}.pgdump-dukememory-{timestamp}"))
}

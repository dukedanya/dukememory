use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub fn resolve_project_id(project: Option<String>) -> Result<String> {
    if let Some(project) = project {
        return Ok(project);
    }

    let cwd = env::current_dir().context("failed to read current directory")?;
    let root = find_project_root(&cwd).unwrap_or(cwd);
    Ok(project_id_from_path(&root))
}

pub fn resolve_project_id_from_path(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let root = find_project_root(path).unwrap_or_else(|| path.to_path_buf());
    Ok(project_id_from_path(&root))
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    let home = env::var_os("HOME").map(PathBuf::from);
    for ancestor in start.ancestors() {
        if home.as_deref() == Some(ancestor) {
            break;
        }
        if ancestor.join(".git").exists()
            || ancestor.join(".dukememory.toml").exists()
            || ancestor.join(".codegraph").exists()
        {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn project_id_from_path(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let display = canonical.to_string_lossy();
    let name = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project");
    let slug = slugify(name);
    let hash = blake3::hash(display.as_bytes()).to_hex();
    format!("{}-{}", slug, &hash[..8])
}

fn slugify(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

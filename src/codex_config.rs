use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::config::Config;

pub struct CodexInstallResult {
    pub config_path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub installed: bool,
}

pub fn default_codex_config_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".codex").join("config.toml"))
}

pub fn mcp_snippet(config: &Config, command: &Path) -> String {
    format!(
        r#"[mcp_servers.dukememory]
command = {}
args = ["mcp"]
startup_timeout_sec = 120

[mcp_servers.dukememory.env]
DUKEMEMORY_DATABASE_URL = {}
DUKEMEMORY_DATABASE_MARKER = {}
DUKEMEMORY_DB = {}
OLLAMA_BASE_URL = {}
DUKEMEMORY_EMBED_MODEL = {}
DUKEMEMORY_FAST_EMBED_MODEL = {}
DUKEMEMORY_EXTRACT_MODEL = {}
DUKEMEMORY_VALIDATE_MODEL = {}
DUKEMEMORY_FAST_CODE_MODEL = {}
DUKEMEMORY_DEEP_CODE_MODEL = {}
DUKEMEMORY_AGENT_CODE_MODEL = {}
DUKEMEMORY_EXPERIMENT_MODEL = {}
"#,
        toml_string(&command.to_string_lossy()),
        toml_string(&config.database_url),
        toml_string(&config.database_marker.to_string_lossy()),
        toml_string(&config.database_marker.to_string_lossy()),
        toml_string(&config.ollama_base_url),
        toml_string(config.memory_embed_model()),
        toml_string(config.code_embed_model()),
        toml_string(config.extract_model()),
        toml_string(&config.validate_model),
        toml_string(&config.fast_code_model),
        toml_string(&config.deep_code_model),
        toml_string(&config.agent_code_model),
        toml_string(&config.experiment_model),
    )
}

pub fn install_mcp_config(
    config_path: &Path,
    snippet: &str,
    force: bool,
) -> Result<CodexInstallResult> {
    let existing = match fs::read_to_string(config_path) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", config_path.display()));
        }
    };

    if has_dukememory_server(&existing) && !force {
        bail!(
            "{} already contains [mcp_servers.dukememory]; rerun with --force to replace it",
            config_path.display()
        );
    }

    let mut updated = if has_dukememory_server(&existing) {
        replace_dukememory_server(&existing, snippet)
    } else {
        append_snippet(&existing, snippet)
    };
    if !updated.ends_with('\n') {
        updated.push('\n');
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let backup_path = if config_path.exists() {
        let backup = backup_path(config_path);
        fs::copy(config_path, &backup).with_context(|| {
            format!(
                "failed to write backup {} from {}",
                backup.display(),
                config_path.display()
            )
        })?;
        Some(backup)
    } else {
        None
    };

    fs::write(config_path, updated)
        .with_context(|| format!("failed to write {}", config_path.display()))?;

    Ok(CodexInstallResult {
        config_path: config_path.to_path_buf(),
        backup_path,
        installed: true,
    })
}

fn append_snippet(existing: &str, snippet: &str) -> String {
    let mut out = existing.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(snippet.trim());
    out.push('\n');
    out
}

fn replace_dukememory_server(existing: &str, snippet: &str) -> String {
    let lines = existing.lines().collect::<Vec<_>>();
    let Some(start) = lines
        .iter()
        .position(|line| line.trim() == "[mcp_servers.dukememory]")
    else {
        return append_snippet(existing, snippet);
    };

    let mut end = lines.len();
    let mut index = start + 1;
    while index < lines.len() {
        let trimmed = lines[index].trim();
        if trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && trimmed != "[mcp_servers.dukememory.env]"
        {
            end = index;
            break;
        }
        index += 1;
    }

    let mut out = String::new();
    if start > 0 {
        out.push_str(&lines[..start].join("\n"));
        out.push_str("\n\n");
    }
    out.push_str(snippet.trim());
    if end < lines.len() {
        out.push_str("\n\n");
        out.push_str(&lines[end..].join("\n"));
    }
    out.push('\n');
    out
}

fn has_dukememory_server(config: &str) -> bool {
    config
        .lines()
        .any(|line| line.trim() == "[mcp_servers.dukememory]")
}

fn backup_path(config_path: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let file_name = config_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config.toml");
    config_path.with_file_name(format!("{file_name}.bak-dukememory-{stamp}"))
}

fn toml_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_contains_dukememory_server_and_env() -> Result<()> {
        let config = Config {
            database_url: "postgresql://dukememory-test@localhost:55432/dukememory_test"
                .to_string(),
            database_marker: PathBuf::from("/tmp/dukeschema.marker"),
            ollama_base_url: "http://127.0.0.1:11435".to_string(),
            ollama_embed_model: "qwen3-embedding:8b".to_string(),
            ollama_llm_model: "qwen3:14b".to_string(),
            fast_embed_model: "bge-m3".to_string(),
            validate_model: "qwen3:14b".to_string(),
            fast_code_model: "qwen2.5-coder:14b".to_string(),
            deep_code_model: "qwen3-coder:30b-a3b-q4_K_M".to_string(),
            agent_code_model: "north-mini-code-1.0:q4_k_m".to_string(),
            experiment_model: "huihui-gemma4-12b-coder:q4_k_m".to_string(),
        };
        let snippet = mcp_snippet(&config, Path::new("/tmp/dukememory"));
        assert!(snippet.contains("[mcp_servers.dukememory]"));
        assert!(snippet.contains("command = \"/tmp/dukememory\""));
        assert!(snippet.contains("DUKEMEMORY_EMBED_MODEL = \"qwen3-embedding:8b\""));
        assert!(snippet.contains("DUKEMEMORY_FAST_EMBED_MODEL = \"bge-m3\""));
        assert!(snippet.contains("DUKEMEMORY_DEEP_CODE_MODEL = \"qwen3-coder:30b-a3b-q4_K_M\""));
        Ok(())
    }

    #[test]
    fn install_appends_and_then_replaces_with_force() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "dukememory-codex-config-{}.toml",
            uuid::Uuid::now_v7()
        ));
        fs::write(&path, "model = \"gpt-5.5\"\n")?;

        let first = "[mcp_servers.dukememory]\ncommand = \"one\"\nargs = [\"mcp\"]\n";
        install_mcp_config(&path, first, false)?;
        let after_first = fs::read_to_string(&path)?;
        assert!(after_first.contains("model = \"gpt-5.5\""));
        assert!(after_first.contains("command = \"one\""));

        let second = "[mcp_servers.dukememory]\ncommand = \"two\"\nargs = [\"mcp\"]\n";
        install_mcp_config(&path, second, true)?;
        let after_second = fs::read_to_string(&path)?;
        assert!(after_second.contains("command = \"two\""));
        assert!(!after_second.contains("command = \"one\""));
        assert!(after_second.contains("model = \"gpt-5.5\""));

        let _ = fs::remove_file(path);
        Ok(())
    }
}

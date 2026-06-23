use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;

use crate::codex_config::default_codex_config_path;
use crate::codex_hooks::default_codex_hooks_path;
use crate::config::Config;
use crate::store::{ListOptions, MemoryStatus, StatusFilter, Store};

#[derive(Debug, Clone, Serialize)]
pub struct CodexIntegrationAuditReport {
    pub config_path: PathBuf,
    pub hooks_path: PathBuf,
    pub script_path: PathBuf,
    pub command_path: PathBuf,
    pub mcp_server_configured: bool,
    pub mcp_args_configured: bool,
    pub env_keys_configured: usize,
    pub env_keys_expected: usize,
    pub hook_events_configured: Vec<String>,
    pub script_exists: bool,
    pub script_executable: bool,
    pub hook_dry_run_ok: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CodexHookPayloadAuditReport {
    pub project_id: String,
    pub root_path: PathBuf,
    pub database_marker: PathBuf,
    pub script_path: PathBuf,
    pub command_path: PathBuf,
    pub event: String,
    pub pending_memories: u64,
    pub total_memories: u64,
    pub memory_embeddings: u64,
    pub first_memory_id: String,
    pub first_memory_source: Option<String>,
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
}

pub fn run_codex_integration_audit(
    config: &Config,
    command_path: &Path,
    script_path: &Path,
) -> Result<CodexIntegrationAuditReport> {
    let config_path = default_codex_config_path()?;
    let hooks_path = default_codex_hooks_path()?;
    run_codex_integration_audit_at(config, command_path, script_path, &config_path, &hooks_path)
}

pub fn run_codex_integration_audit_at(
    config: &Config,
    command_path: &Path,
    script_path: &Path,
    config_path: &Path,
    hooks_path: &Path,
) -> Result<CodexIntegrationAuditReport> {
    let config_text = std::fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let server = table_lines(&config_text, "[mcp_servers.dukememory]")
        .context("Codex config is missing [mcp_servers.dukememory]")?;
    let env = table_lines(&config_text, "[mcp_servers.dukememory.env]")
        .context("Codex config is missing [mcp_servers.dukememory.env]")?;

    let configured_command = toml_string_value(&server, "command")
        .context("Codex dukememory MCP config is missing command")?;
    let configured_command_path = PathBuf::from(&configured_command);
    if !configured_command_path.exists() {
        bail!(
            "Codex dukememory MCP command does not exist: {}",
            configured_command_path.display()
        );
    }
    let expected_command = command_path.to_string_lossy();
    if configured_command_path != command_path && configured_command != expected_command {
        bail!(
            "Codex dukememory MCP command points to {}, expected {}",
            configured_command_path.display(),
            command_path.display()
        );
    }
    if !server
        .iter()
        .any(|line| line.trim().starts_with("args") && line.contains("\"mcp\""))
    {
        bail!("Codex dukememory MCP config must include args = [\"mcp\"]");
    }

    let expected_env = [
        ("DUKEMEMORY_DATABASE_URL", config.database_url.to_string()),
        (
            "DUKEMEMORY_DATABASE_MARKER",
            config.database_marker.to_string_lossy().to_string(),
        ),
        (
            "DUKEMEMORY_DB",
            config.database_marker.to_string_lossy().to_string(),
        ),
        ("OLLAMA_BASE_URL", config.ollama_base_url.clone()),
        (
            "DUKEMEMORY_EMBED_MODEL",
            config.memory_embed_model().to_string(),
        ),
        (
            "DUKEMEMORY_FAST_EMBED_MODEL",
            config.code_embed_model().to_string(),
        ),
        (
            "DUKEMEMORY_EXTRACT_MODEL",
            config.extract_model().to_string(),
        ),
        ("DUKEMEMORY_VALIDATE_MODEL", config.validate_model.clone()),
        ("DUKEMEMORY_FAST_CODE_MODEL", config.fast_code_model.clone()),
        ("DUKEMEMORY_DEEP_CODE_MODEL", config.deep_code_model.clone()),
        (
            "DUKEMEMORY_AGENT_CODE_MODEL",
            config.agent_code_model.clone(),
        ),
        (
            "DUKEMEMORY_EXPERIMENT_MODEL",
            config.experiment_model.clone(),
        ),
    ];
    let mut env_keys_configured = 0;
    for (key, expected) in &expected_env {
        let Some(actual) = toml_string_value(&env, key) else {
            bail!("Codex dukememory MCP env is missing {key}");
        };
        if actual != *expected {
            bail!("Codex dukememory MCP env {key}={actual:?}, expected {expected:?}");
        }
        env_keys_configured += 1;
    }

    let script_exists = script_path.exists();
    if !script_exists {
        bail!(
            "dukememory Codex hook script does not exist: {}",
            script_path.display()
        );
    }
    let script_executable = is_executable(script_path)?;
    if !script_executable {
        bail!(
            "dukememory Codex hook script is not executable: {}",
            script_path.display()
        );
    }

    let hooks_text = std::fs::read_to_string(hooks_path)
        .with_context(|| format!("failed to read {}", hooks_path.display()))?;
    let hooks_json = serde_json::from_str::<Value>(&hooks_text)
        .with_context(|| format!("failed to parse {}", hooks_path.display()))?;
    let mut hook_events_configured = Vec::new();
    for event in ["Stop", "PreCompact"] {
        if hook_event_contains_script(&hooks_json, event, script_path) {
            hook_events_configured.push(event.to_string());
        } else {
            bail!(
                "Codex hooks are missing dukememory script {} for event {event}",
                script_path.display()
            );
        }
    }

    let dry_run = Command::new("bash")
        .arg(script_path)
        .env("DUKEMEMORY_BIN", command_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("failed to run hook script {}", script_path.display()))?;
    if !dry_run.status.success() {
        bail!(
            "dukememory Codex hook dry-run failed with status {:?}: {}",
            dry_run.status.code(),
            String::from_utf8_lossy(&dry_run.stderr)
        );
    }

    Ok(CodexIntegrationAuditReport {
        config_path: config_path.to_path_buf(),
        hooks_path: hooks_path.to_path_buf(),
        script_path: script_path.to_path_buf(),
        command_path: command_path.to_path_buf(),
        mcp_server_configured: true,
        mcp_args_configured: true,
        env_keys_configured,
        env_keys_expected: expected_env.len(),
        hook_events_configured,
        script_exists,
        script_executable,
        hook_dry_run_ok: true,
    })
}

pub fn run_codex_hook_payload_audit(
    config: &Config,
    command_path: &Path,
    script_path: &Path,
) -> Result<CodexHookPayloadAuditReport> {
    let token = uuid::Uuid::now_v7().to_string().replace('-', "");
    let root_path = std::env::temp_dir().join(format!("dukememory-hook-audit-{token}"));
    let project_id = format!("hook-audit-{token}");
    let database_marker = root_path.join("schema.marker");
    std::fs::create_dir_all(&root_path)
        .with_context(|| format!("failed to create {}", root_path.display()))?;
    std::fs::write(
        root_path.join(".dukememory.toml"),
        format!("name = \"{project_id}\"\n"),
    )?;

    let payload = serde_json::json!({
        "event": "Stop",
        "cwd": root_path,
        "messages": [
            {
                "role": "user",
                "content": "Project rule: Use RON files for audit item definitions and keep combat stamina costs data-driven."
            },
            {
                "role": "assistant",
                "content": "Implemented the audit item loader and documented the stamina cost rule."
            }
        ]
    })
    .to_string();

    let output = run_hook_script_with_payload(
        config,
        command_path,
        script_path,
        &database_marker,
        &project_id,
        "Stop",
        &payload,
    )?;
    let store = Store::open(&database_marker)?;
    let status = store.status(&project_id)?;
    let pending = store.list(
        &project_id,
        ListOptions {
            limit: 10,
            offset: 0,
            status: StatusFilter::One(MemoryStatus::Pending),
            kind: None,
            memory_tier: None,
        },
    )?;
    let Some(first) = pending.first() else {
        bail!(
            "hook payload audit stored no pending memories; stdout={}, stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };
    if !first.body.to_ascii_lowercase().contains("ron")
        && !first.body.to_ascii_lowercase().contains("stamina")
    {
        bail!(
            "hook payload audit stored an unexpected memory body: {}",
            first.body
        );
    }

    Ok(CodexHookPayloadAuditReport {
        project_id,
        root_path,
        database_marker,
        script_path: script_path.to_path_buf(),
        command_path: command_path.to_path_buf(),
        event: "Stop".to_string(),
        pending_memories: status.pending_memories,
        total_memories: status.total_memories,
        memory_embeddings: status.memory_embeddings,
        first_memory_id: first.id.clone(),
        first_memory_source: first.source.clone(),
        stdout_bytes: output.stdout.len(),
        stderr_bytes: output.stderr.len(),
    })
}

fn run_hook_script_with_payload(
    config: &Config,
    command_path: &Path,
    script_path: &Path,
    database_marker: &Path,
    project_id: &str,
    event: &str,
    payload: &str,
) -> Result<std::process::Output> {
    let mut child = Command::new("bash")
        .arg(script_path)
        .env("DUKEMEMORY_BIN", command_path)
        .env("DUKEMEMORY_DB", database_marker)
        .env("DUKEMEMORY_PROJECT", project_id)
        .env("DUKEMEMORY_EVENT", event)
        .env("DUKEMEMORY_MAX_CANDIDATES", "4")
        .env("OLLAMA_BASE_URL", &config.ollama_base_url)
        .env("DUKEMEMORY_EMBED_MODEL", config.memory_embed_model())
        .env("DUKEMEMORY_FAST_EMBED_MODEL", config.code_embed_model())
        .env("DUKEMEMORY_EXTRACT_MODEL", config.extract_model())
        .env("DUKEMEMORY_VALIDATE_MODEL", &config.validate_model)
        .env("DUKEMEMORY_FAST_CODE_MODEL", &config.fast_code_model)
        .env("DUKEMEMORY_DEEP_CODE_MODEL", &config.deep_code_model)
        .env("DUKEMEMORY_AGENT_CODE_MODEL", &config.agent_code_model)
        .env("DUKEMEMORY_EXPERIMENT_MODEL", &config.experiment_model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to start hook script {}", script_path.display()))?;
    child
        .stdin
        .as_mut()
        .context("failed to open hook script stdin")?
        .write_all(payload.as_bytes())
        .context("failed to write hook payload")?;
    let output = child
        .wait_with_output()
        .context("failed to wait for hook script")?;
    if !output.status.success() {
        bail!(
            "hook payload audit failed with status {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output)
}

fn table_lines<'a>(text: &'a str, header: &str) -> Option<Vec<&'a str>> {
    let lines = text.lines().collect::<Vec<_>>();
    let start = lines.iter().position(|line| line.trim() == header)?;
    let mut end = lines.len();
    for (offset, line) in lines[start + 1..].iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            end = start + 1 + offset;
            break;
        }
    }
    Some(lines[start + 1..end].to_vec())
}

fn toml_string_value(lines: &[&str], key: &str) -> Option<String> {
    lines.iter().find_map(|line| {
        let (actual_key, value) = line.split_once('=')?;
        if actual_key.trim() != key {
            return None;
        }
        Some(unquote_toml_string(value.trim()))
    })
}

fn unquote_toml_string(value: &str) -> String {
    let trimmed = value.trim();
    let inner = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(trimmed);
    inner
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

fn hook_event_contains_script(root: &Value, event: &str, script_path: &Path) -> bool {
    root.get("hooks")
        .and_then(Value::as_object)
        .and_then(|hooks| hooks.get(event))
        .map(|value| value_contains_script(value, script_path))
        .unwrap_or(false)
}

fn value_contains_script(value: &Value, script_path: &Path) -> bool {
    let script = script_path.to_string_lossy();
    match value {
        Value::String(text) => text.contains(script.as_ref()),
        Value::Array(values) => values
            .iter()
            .any(|value| value_contains_script(value, script_path)),
        Value::Object(map) => map
            .values()
            .any(|value| value_contains_script(value, script_path)),
        _ => false,
    }
}

#[cfg(unix)]
fn is_executable(path: &Path) -> Result<bool> {
    use std::os::unix::fs::PermissionsExt;
    let permissions = std::fs::metadata(path)?.permissions();
    Ok(permissions.mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> Result<bool> {
    Ok(path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codex_config::mcp_snippet;
    use crate::codex_hooks::hooks_snippet;

    #[test]
    fn codex_integration_audit_accepts_installed_config_and_hooks() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("dukememory-codex-audit-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join(".codex"))?;
        let command = root.join("dukememory");
        let script = root.join("dukememory_codex_hook.sh");
        std::fs::write(&command, "#!/usr/bin/env bash\nexit 0\n")?;
        std::fs::write(&script, "#!/usr/bin/env bash\ncat >/dev/null\nexit 0\n")?;
        make_executable(&command)?;
        make_executable(&script)?;

        let config = test_config(root.join("schema.marker"));
        let config_path = root.join(".codex/config.toml");
        let hooks_path = root.join(".codex/hooks.json");
        std::fs::write(&config_path, mcp_snippet(&config, &command))?;
        std::fs::write(&hooks_path, hooks_snippet(&script, &[])?)?;

        let report =
            run_codex_integration_audit_at(&config, &command, &script, &config_path, &hooks_path)?;
        assert!(report.mcp_server_configured);
        assert_eq!(report.env_keys_configured, report.env_keys_expected);
        assert_eq!(report.hook_events_configured, vec!["Stop", "PreCompact"]);
        assert!(report.hook_dry_run_ok);
        Ok(())
    }

    #[test]
    fn codex_integration_audit_rejects_missing_mcp_config() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("dukememory-codex-audit-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(root.join(".codex"))?;
        let command = root.join("dukememory");
        let script = root.join("dukememory_codex_hook.sh");
        std::fs::write(&command, "#!/usr/bin/env bash\nexit 0\n")?;
        std::fs::write(&script, "#!/usr/bin/env bash\nexit 0\n")?;
        make_executable(&command)?;
        make_executable(&script)?;

        let config = test_config(root.join("schema.marker"));
        let config_path = root.join(".codex/config.toml");
        let hooks_path = root.join(".codex/hooks.json");
        std::fs::write(&config_path, "model = \"gpt-5.5\"\n")?;
        std::fs::write(&hooks_path, hooks_snippet(&script, &[])?)?;

        let error =
            run_codex_integration_audit_at(&config, &command, &script, &config_path, &hooks_path)
                .expect_err("missing MCP config should fail");
        assert!(error.to_string().contains("[mcp_servers.dukememory]"));
        Ok(())
    }

    #[test]
    fn hook_payload_runner_passes_env_and_stdin_to_script() -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("dukememory-hook-runner-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&root)?;
        let command = root.join("dukememory");
        let script = root.join("hook.sh");
        let payload_path = root.join("payload.txt");
        std::fs::write(&command, "#!/usr/bin/env bash\nexit 0\n")?;
        std::fs::write(
            &script,
            format!(
                r#"#!/usr/bin/env bash
set -euo pipefail
test "$DUKEMEMORY_PROJECT" = "hook-project"
test "$DUKEMEMORY_EVENT" = "Stop"
test "$DUKEMEMORY_MAX_CANDIDATES" = "4"
cat > "{}"
"#,
                payload_path.display()
            ),
        )?;
        make_executable(&command)?;
        make_executable(&script)?;

        let config = test_config(root.join("schema.marker"));
        let output = run_hook_script_with_payload(
            &config,
            &command,
            &script,
            &root.join("audit.schema-marker"),
            "hook-project",
            "Stop",
            "{\"event\":\"Stop\",\"summary\":\"Use durable memory.\"}",
        )?;

        assert!(output.status.success());
        let payload = std::fs::read_to_string(payload_path)?;
        assert!(payload.contains("Use durable memory."));
        Ok(())
    }

    fn test_config(database_marker: PathBuf) -> Config {
        Config {
            database_url: "postgresql://daniil@localhost:55432/dukememory?host=/tmp".to_string(),
            database_marker,
            ollama_base_url: "http://127.0.0.1:11435".to_string(),
            ollama_embed_model: "qwen3-embedding:8b".to_string(),
            ollama_llm_model: "qwen3:14b".to_string(),
            fast_embed_model: "bge-m3".to_string(),
            validate_model: "qwen3:14b".to_string(),
            fast_code_model: "qwen2.5-coder:14b".to_string(),
            deep_code_model: "qwen3-coder:30b-a3b-q4_K_M".to_string(),
            agent_code_model: "north-mini-code-1.0:q4_k_m".to_string(),
            experiment_model: "huihui-gemma4-12b-coder:q4_k_m".to_string(),
        }
    }

    #[cfg(unix)]
    fn make_executable(path: &Path) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions)?;
        Ok(())
    }

    #[cfg(not(unix))]
    fn make_executable(_path: &Path) -> Result<()> {
        Ok(())
    }
}

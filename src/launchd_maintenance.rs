use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

use crate::config::Config;

pub const DEFAULT_LABEL: &str = "com.dukememory.maintenance";

#[derive(Debug, Clone)]
pub struct MaintenanceLaunchdOptions {
    pub command: PathBuf,
    pub label: String,
    pub interval_seconds: u64,
    pub project: Option<String>,
    pub apply: bool,
    pub all: bool,
    pub backup: bool,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LaunchdInstallResult {
    pub plist_path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub installed: bool,
}

pub fn default_plist_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home)
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{DEFAULT_LABEL}.plist")))
}

pub fn default_log_paths() -> Result<(PathBuf, PathBuf)> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    let dir = PathBuf::from(home)
        .join("Library")
        .join("Logs")
        .join("dukememory");
    Ok((
        dir.join("maintenance.out.log"),
        dir.join("maintenance.err.log"),
    ))
}

pub fn maintenance_launchd_plist(config: &Config, options: &MaintenanceLaunchdOptions) -> String {
    let mut args = vec![
        options.command.to_string_lossy().to_string(),
        "maintenance".to_string(),
    ];
    if options.all {
        args.push("--all".to_string());
    }
    if options.backup {
        args.push("--backup".to_string());
    }
    if options.apply {
        args.push("--apply".to_string());
    }
    if let Some(project) = &options.project {
        args.push("--project".to_string());
        args.push(project.clone());
    }

    let args_xml = args
        .iter()
        .map(|arg| format!("    <string>{}</string>", xml_escape(arg)))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>

  <key>ProgramArguments</key>
  <array>
{args_xml}
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>DUKEMEMORY_DATABASE_URL</key>
    <string>{database_url}</string>
    <key>DUKEMEMORY_DATABASE_MARKER</key>
    <string>{db}</string>
    <key>DUKEMEMORY_DB</key>
    <string>{db}</string>
    <key>OLLAMA_BASE_URL</key>
    <string>{ollama_base}</string>
    <key>DUKEMEMORY_EMBED_MODEL</key>
    <string>{memory_embed}</string>
    <key>DUKEMEMORY_FAST_EMBED_MODEL</key>
    <string>{fast_embed}</string>
    <key>DUKEMEMORY_EXTRACT_MODEL</key>
    <string>{extract}</string>
    <key>DUKEMEMORY_VALIDATE_MODEL</key>
    <string>{validate}</string>
    <key>DUKEMEMORY_FAST_CODE_MODEL</key>
    <string>{fast_code}</string>
    <key>DUKEMEMORY_DEEP_CODE_MODEL</key>
    <string>{deep_code}</string>
    <key>DUKEMEMORY_AGENT_CODE_MODEL</key>
    <string>{agent_code}</string>
    <key>DUKEMEMORY_EXPERIMENT_MODEL</key>
    <string>{experiment}</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>StartInterval</key>
  <integer>{interval}</integer>

  <key>StandardOutPath</key>
  <string>{stdout}</string>

  <key>StandardErrorPath</key>
  <string>{stderr}</string>
</dict>
</plist>
"#,
        label = xml_escape(&options.label),
        database_url = xml_escape(&config.database_url),
        db = xml_escape(&config.database_marker.to_string_lossy()),
        ollama_base = xml_escape(&config.ollama_base_url),
        memory_embed = xml_escape(config.memory_embed_model()),
        fast_embed = xml_escape(config.code_embed_model()),
        extract = xml_escape(config.extract_model()),
        validate = xml_escape(&config.validate_model),
        fast_code = xml_escape(&config.fast_code_model),
        deep_code = xml_escape(&config.deep_code_model),
        agent_code = xml_escape(&config.agent_code_model),
        experiment = xml_escape(&config.experiment_model),
        interval = options.interval_seconds.max(300),
        stdout = xml_escape(&options.stdout_path.to_string_lossy()),
        stderr = xml_escape(&options.stderr_path.to_string_lossy()),
    )
}

pub fn install_maintenance_launchd(
    plist_path: &Path,
    plist: &str,
    stdout_path: &Path,
    stderr_path: &Path,
    force: bool,
) -> Result<LaunchdInstallResult> {
    if plist_path.exists() && !force {
        bail!(
            "{} already exists; rerun with --force to replace it",
            plist_path.display()
        );
    }

    if let Some(parent) = plist_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    for log_path in [stdout_path, stderr_path] {
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let backup_path = if plist_path.exists() {
        let backup = backup_path(plist_path);
        fs::copy(plist_path, &backup).with_context(|| {
            format!(
                "failed to write backup {} from {}",
                backup.display(),
                plist_path.display()
            )
        })?;
        Some(backup)
    } else {
        None
    };

    fs::write(plist_path, plist)
        .with_context(|| format!("failed to write {}", plist_path.display()))?;
    Ok(LaunchdInstallResult {
        plist_path: plist_path.to_path_buf(),
        backup_path,
        installed: true,
    })
}

pub fn load_launch_agent(plist_path: &Path) -> Result<()> {
    let domain = launchctl_domain()?;
    let status = Command::new("launchctl")
        .arg("bootstrap")
        .arg(domain)
        .arg(plist_path)
        .status()
        .context("failed to execute launchctl bootstrap")?;
    if !status.success() {
        bail!("launchctl bootstrap failed with status {status}");
    }
    Ok(())
}

pub fn unload_launch_agent(plist_path: &Path) -> Result<()> {
    let domain = launchctl_domain()?;
    let status = Command::new("launchctl")
        .arg("bootout")
        .arg(domain)
        .arg(plist_path)
        .status()
        .context("failed to execute launchctl bootout")?;
    if !status.success() {
        bail!("launchctl bootout failed with status {status}");
    }
    Ok(())
}

fn launchctl_domain() -> Result<String> {
    let output = Command::new("id")
        .arg("-u")
        .output()
        .context("failed to execute id -u")?;
    if !output.status.success() {
        bail!("id -u failed with status {}", output.status);
    }
    let uid = String::from_utf8(output.stdout)
        .context("id -u returned invalid UTF-8")?
        .trim()
        .to_string();
    Ok(format!("gui/{uid}"))
}

fn backup_path(plist_path: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let file_name = plist_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("com.dukememory.maintenance.plist");
    plist_path.with_file_name(format!("{file_name}.bak-dukememory-{stamp}"))
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_contains_maintenance_command_and_env() {
        let config = test_config();
        let options = MaintenanceLaunchdOptions {
            command: PathBuf::from("/tmp/dukememory"),
            label: DEFAULT_LABEL.to_string(),
            interval_seconds: 60,
            project: Some("game-a".to_string()),
            apply: true,
            all: true,
            backup: true,
            stdout_path: PathBuf::from("/tmp/dukememory.out.log"),
            stderr_path: PathBuf::from("/tmp/dukememory.err.log"),
        };
        let plist = maintenance_launchd_plist(&config, &options);
        assert!(plist.contains("<string>/tmp/dukememory</string>"));
        assert!(plist.contains("<string>maintenance</string>"));
        assert!(plist.contains("<string>--all</string>"));
        assert!(plist.contains("<string>--backup</string>"));
        assert!(plist.contains("<string>--apply</string>"));
        assert!(plist.contains("<string>game-a</string>"));
        assert!(plist.contains("<integer>300</integer>"));
        assert!(plist.contains("<key>DUKEMEMORY_DB</key>"));
        assert!(plist.contains("<string>qwen3-embedding:8b</string>"));
    }

    #[test]
    fn install_requires_force_for_existing_plist_and_writes_backup() -> Result<()> {
        let dir =
            std::env::temp_dir().join(format!("dukememory-launchd-test-{}", uuid::Uuid::now_v7()));
        let plist_path = dir.join("com.dukememory.maintenance.plist");
        let stdout_path = dir.join("maintenance.out.log");
        let stderr_path = dir.join("maintenance.err.log");
        fs::create_dir_all(&dir)?;
        fs::write(&plist_path, "old")?;
        let error =
            install_maintenance_launchd(&plist_path, "new", &stdout_path, &stderr_path, false)
                .unwrap_err();
        assert!(error.to_string().contains("already exists"));

        let result =
            install_maintenance_launchd(&plist_path, "new", &stdout_path, &stderr_path, true)?;
        assert!(result.installed);
        assert!(result.backup_path.is_some());
        assert_eq!(fs::read_to_string(&plist_path)?, "new");
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    fn test_config() -> Config {
        Config {
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
        }
    }
}

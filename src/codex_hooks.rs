use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

const DEFAULT_EVENTS: &[&str] = &["Stop", "PreCompact"];

pub struct HookInstallResult {
    pub hooks_path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub installed_events: Vec<String>,
}

pub fn default_codex_hooks_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".codex").join("hooks.json"))
}

pub fn default_hook_events() -> Vec<String> {
    DEFAULT_EVENTS
        .iter()
        .map(|event| event.to_string())
        .collect()
}

pub fn hooks_snippet(script_path: &Path, events: &[String]) -> Result<String> {
    let mut root = json!({ "hooks": {} });
    for event in normalized_events(events) {
        add_hook_entry(&mut root, &event, script_path)?;
    }
    serde_json::to_string_pretty(&root).context("failed to render hook snippet")
}

pub fn install_hooks(
    hooks_path: &Path,
    script_path: &Path,
    events: &[String],
    force: bool,
) -> Result<HookInstallResult> {
    let mut root = match fs::read_to_string(hooks_path) {
        Ok(text) => serde_json::from_str::<Value>(&text)
            .with_context(|| format!("failed to parse {}", hooks_path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => json!({ "hooks": {} }),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", hooks_path.display()));
        }
    };
    ensure_hooks_object(&mut root)?;

    if contains_dukememory_hook(&root) && !force {
        bail!(
            "{} already contains a dukememory hook; rerun with --force to replace it",
            hooks_path.display()
        );
    }
    if force {
        remove_dukememory_hooks(&mut root);
    }

    let mut installed_events = Vec::new();
    for event in normalized_events(events) {
        add_hook_entry(&mut root, &event, script_path)?;
        installed_events.push(event);
    }

    if let Some(parent) = hooks_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let backup_path = if hooks_path.exists() {
        let backup = backup_path(hooks_path);
        fs::copy(hooks_path, &backup).with_context(|| {
            format!(
                "failed to write backup {} from {}",
                backup.display(),
                hooks_path.display()
            )
        })?;
        Some(backup)
    } else {
        None
    };

    let mut rendered =
        serde_json::to_string_pretty(&root).context("failed to render updated hooks JSON")?;
    rendered.push('\n');
    fs::write(hooks_path, rendered)
        .with_context(|| format!("failed to write {}", hooks_path.display()))?;

    Ok(HookInstallResult {
        hooks_path: hooks_path.to_path_buf(),
        backup_path,
        installed_events,
    })
}

fn add_hook_entry(root: &mut Value, event: &str, script_path: &Path) -> Result<()> {
    let hooks = root
        .get_mut("hooks")
        .and_then(Value::as_object_mut)
        .context("hooks root is not an object")?;
    let event_entries = hooks
        .entry(event.to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let event_array = event_entries
        .as_array_mut()
        .with_context(|| format!("hooks.{event} is not an array"))?;
    event_array.push(json!({
        "hooks": [
            {
                "type": "command",
                "command": hook_command(script_path, event),
                "timeout": 120,
                "statusMessage": format!("dukememory extracting pending memories ({event})")
            }
        ]
    }));
    Ok(())
}

fn hook_command(script_path: &Path, event: &str) -> String {
    let script_path = script_path.to_string_lossy();
    format!(
        "DUKEMEMORY_EVENT={} bash {}",
        shell_quote(event),
        shell_quote(&script_path)
    )
}

fn ensure_hooks_object(root: &mut Value) -> Result<()> {
    if !root.is_object() {
        *root = json!({ "hooks": {} });
        return Ok(());
    }
    let object = root.as_object_mut().expect("object checked");
    match object.get_mut("hooks") {
        Some(value) if value.is_object() => Ok(()),
        Some(_) => bail!("hooks field exists but is not an object"),
        None => {
            object.insert("hooks".to_string(), Value::Object(Map::new()));
            Ok(())
        }
    }
}

fn contains_dukememory_hook(root: &Value) -> bool {
    root.get("hooks")
        .and_then(Value::as_object)
        .map(|hooks| hooks.values().any(value_contains_dukememory_hook))
        .unwrap_or(false)
}

fn value_contains_dukememory_hook(value: &Value) -> bool {
    match value {
        Value::String(text) => text.contains("dukememory_codex_hook.sh"),
        Value::Array(values) => values.iter().any(value_contains_dukememory_hook),
        Value::Object(values) => values.values().any(value_contains_dukememory_hook),
        _ => false,
    }
}

fn remove_dukememory_hooks(root: &mut Value) {
    let Some(hooks) = root.get_mut("hooks").and_then(Value::as_object_mut) else {
        return;
    };
    for value in hooks.values_mut() {
        let Some(entries) = value.as_array_mut() else {
            continue;
        };
        entries.retain(|entry| !value_contains_dukememory_hook(entry));
    }
}

fn normalized_events(events: &[String]) -> Vec<String> {
    let source = if events.is_empty() {
        default_hook_events()
    } else {
        events.to_vec()
    };
    let mut out = Vec::new();
    for event in source {
        let event = event.trim();
        if !event.is_empty() && !out.iter().any(|existing| existing == event) {
            out.push(event.to_string());
        }
    }
    out
}

fn backup_path(hooks_path: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let file_name = hooks_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hooks.json");
    hooks_path.with_file_name(format!("{file_name}.bak-dukememory-{stamp}"))
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_snippet_contains_default_events() -> Result<()> {
        let snippet = hooks_snippet(Path::new("/tmp/dukememory_codex_hook.sh"), &[])?;
        assert!(snippet.contains("\"Stop\""));
        assert!(snippet.contains("\"PreCompact\""));
        assert!(snippet.contains("dukememory_codex_hook.sh"));
        Ok(())
    }

    #[test]
    fn install_hooks_preserves_existing_and_force_replaces_dukememory() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("dukememory-hooks-{}.json", uuid::Uuid::now_v7()));
        fs::write(
            &path,
            r#"{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo existing"
          }
        ]
      }
    ]
  }
}"#,
        )?;

        let events = vec!["Stop".to_string()];
        install_hooks(
            &path,
            Path::new("/tmp/dukememory_codex_hook.sh"),
            &events,
            false,
        )?;
        let installed = fs::read_to_string(&path)?;
        assert!(installed.contains("echo existing"));
        assert!(installed.contains("\"Stop\""));
        assert!(installed.contains("dukememory_codex_hook.sh"));

        let forced = vec!["PreCompact".to_string()];
        install_hooks(
            &path,
            Path::new("/tmp/dukememory_codex_hook.sh"),
            &forced,
            true,
        )?;
        let replaced = fs::read_to_string(&path)?;
        assert!(replaced.contains("echo existing"));
        let json = serde_json::from_str::<Value>(&replaced)?;
        let hooks = json["hooks"].as_object().expect("hooks object");
        assert!(hooks.contains_key("SessionStart"));
        assert!(hooks.contains_key("PreCompact"));
        assert!(!value_contains_dukememory_hook(&json["hooks"]["Stop"]));

        let _ = fs::remove_file(path);
        Ok(())
    }
}

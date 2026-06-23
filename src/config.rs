use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub database_marker: PathBuf,
    pub database_url: String,
    pub ollama_base_url: String,
    pub ollama_embed_model: String,
    pub ollama_llm_model: String,
    pub fast_embed_model: String,
    pub validate_model: String,
    pub fast_code_model: String,
    pub deep_code_model: String,
    pub agent_code_model: String,
    pub experiment_model: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let database_marker = match env::var_os("DUKEMEMORY_DATABASE_MARKER")
            .or_else(|| env::var_os("DUKEMEMORY_DB"))
        {
            Some(path) => PathBuf::from(path),
            None => default_database_marker()?,
        };
        let database_url = env::var("DUKEMEMORY_DATABASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(default_database_url);

        let ollama_base_url =
            env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11435".to_string());
        let ollama_embed_model = model_env(
            "DUKEMEMORY_EMBED_MODEL",
            Some("OLLAMA_EMBED_MODEL"),
            "qwen3-embedding:8b",
        );
        let ollama_llm_model = model_env(
            "DUKEMEMORY_EXTRACT_MODEL",
            Some("OLLAMA_LLM_MODEL"),
            "qwen3:14b",
        );
        let fast_embed_model = model_env("DUKEMEMORY_FAST_EMBED_MODEL", None, "bge-m3");
        let validate_model = model_env("DUKEMEMORY_VALIDATE_MODEL", None, &ollama_llm_model);

        Ok(Self {
            database_marker,
            database_url,
            ollama_base_url,
            ollama_embed_model,
            ollama_llm_model,
            fast_embed_model,
            validate_model,
            fast_code_model: model_env("DUKEMEMORY_FAST_CODE_MODEL", None, "qwen2.5-coder:14b"),
            deep_code_model: model_env(
                "DUKEMEMORY_DEEP_CODE_MODEL",
                None,
                "qwen3-coder:30b-a3b-q4_K_M",
            ),
            agent_code_model: model_env(
                "DUKEMEMORY_AGENT_CODE_MODEL",
                None,
                "north-mini-code-1.0:q4_k_m",
            ),
            experiment_model: model_env(
                "DUKEMEMORY_EXPERIMENT_MODEL",
                None,
                "huihui-gemma4-12b-coder:q4_k_m",
            ),
        })
    }

    pub fn memory_embed_model(&self) -> &str {
        &self.ollama_embed_model
    }

    pub fn code_embed_model(&self) -> &str {
        &self.fast_embed_model
    }

    pub fn extract_model(&self) -> &str {
        &self.ollama_llm_model
    }

    pub fn model_roles(&self) -> [(&'static str, &str); 8] {
        [
            ("memory_embed", self.memory_embed_model()),
            ("fast_embed", self.code_embed_model()),
            ("extract", self.extract_model()),
            ("validate", &self.validate_model),
            ("fast_code", &self.fast_code_model),
            ("deep_code", &self.deep_code_model),
            ("agent_code", &self.agent_code_model),
            ("experiment", &self.experiment_model),
        ]
    }
}

fn default_database_url() -> String {
    let user = env::var("USER").unwrap_or_else(|_| "dukememory".to_string());
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!(
        "postgresql://{user}@localhost:55432/dukememory?host={home}/.dukememory/postgres-socket"
    )
}

fn model_env(primary: &str, legacy: Option<&str>, default: &str) -> String {
    env::var(primary)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            legacy.and_then(|legacy| {
                env::var(legacy)
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
        })
        .unwrap_or_else(|| default.to_string())
}

pub fn model_name_matches(configured: &str, installed: &str) -> bool {
    configured == installed
        || installed.strip_suffix(":latest") == Some(configured)
        || configured.strip_suffix(":latest") == Some(installed)
}

fn default_database_marker() -> Result<PathBuf> {
    let home = env::var_os("HOME")
        .context("HOME is not set; set DUKEMEMORY_DATABASE_MARKER explicitly")?;
    Ok(PathBuf::from(home)
        .join(".dukememory")
        .join("schema.marker"))
}

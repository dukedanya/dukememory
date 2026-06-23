use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: Client,
    base_url: String,
    llm_model: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TagsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub details: Option<ModelDetails>,
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelDetails {
    pub parameter_size: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
}

#[derive(Debug, Serialize)]
struct EmbedSingleRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Debug, Serialize)]
struct EmbedBatchRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    format: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponseMessage {
    content: String,
}

impl OllamaClient {
    pub fn new(base_url: String, llm_model: String) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            llm_model,
        }
    }

    pub async fn tags(&self) -> Result<TagsResponse> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self
            .http
            .get(url)
            .send()
            .await
            .context("failed to call Ollama /api/tags")?
            .error_for_status()
            .context("Ollama /api/tags returned an error")?;

        response
            .json::<TagsResponse>()
            .await
            .context("failed to parse Ollama /api/tags response")
    }

    pub async fn embed_with_model(&self, model: &str, input: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.base_url);
        let response = self
            .http
            .post(url)
            .json(&EmbedSingleRequest { model, input })
            .send()
            .await
            .context("failed to call Ollama /api/embed")?
            .error_for_status()
            .context("Ollama /api/embed returned an error")?;

        let body = response
            .json::<EmbedResponse>()
            .await
            .context("failed to parse Ollama /api/embed response")?;
        let Some(first) = body.embeddings.into_iter().next() else {
            bail!("Ollama returned no embeddings");
        };
        Ok(first)
    }

    pub async fn embed_batch_with_model(
        &self,
        model: &str,
        inputs: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/embed", self.base_url);
        let response = self
            .http
            .post(url)
            .json(&EmbedBatchRequest {
                model,
                input: inputs,
            })
            .send()
            .await
            .context("failed to call Ollama /api/embed")?
            .error_for_status()
            .context("Ollama /api/embed returned an error")?;

        let body = response
            .json::<EmbedResponse>()
            .await
            .context("failed to parse Ollama /api/embed response")?;
        if body.embeddings.len() != inputs.len() {
            bail!(
                "Ollama returned {} embeddings for {} inputs",
                body.embeddings.len(),
                inputs.len()
            );
        }
        Ok(body.embeddings)
    }

    pub async fn chat_json(&self, system: &str, user: &str) -> Result<String> {
        self.chat_json_with_model(&self.llm_model, system, user)
            .await
    }

    pub async fn chat_json_with_model(
        &self,
        model: &str,
        system: &str,
        user: &str,
    ) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .http
            .post(url)
            .json(&ChatRequest {
                model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: system,
                    },
                    ChatMessage {
                        role: "user",
                        content: user,
                    },
                ],
                stream: false,
                format: "json",
            })
            .send()
            .await
            .context("failed to call Ollama /api/chat")?
            .error_for_status()
            .context("Ollama /api/chat returned an error")?;

        let body = response
            .json::<ChatResponse>()
            .await
            .context("failed to parse Ollama /api/chat response")?;
        Ok(body.message.content)
    }
}

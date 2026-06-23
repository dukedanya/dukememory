use anyhow::{Result, bail};

use crate::config::Config;
use crate::ollama::OllamaClient;
use crate::store::{CodeSymbol, Memory, Store};

const EMBEDDING_BATCH_SIZE: usize = 16;

#[derive(Debug, Default, Clone, Copy)]
pub struct EmbedMissingReport {
    pub memories: usize,
    pub code_symbols: usize,
    pub code_symbols_cached: usize,
    pub code_symbols_reused: usize,
    pub code_symbols_generated: usize,
}

struct PendingCodeSymbolEmbedding {
    symbol_id: String,
    kind: String,
    text: String,
    content_hash: String,
}

struct EmbeddingInput {
    kind: String,
    text: String,
    content_hash: String,
}

#[derive(Debug, Default, Clone, Copy)]
struct CodeSymbolEmbeddingReport {
    completed: usize,
    reused: usize,
    generated: usize,
}

pub async fn embed_memory(
    config: &Config,
    store: &Store,
    project_id: &str,
    memory_id: &str,
    body: &str,
) -> Result<usize> {
    let ollama = ollama_from_config(config);
    let inputs = match store.get(project_id, memory_id)? {
        Some(memory) => memory_embedding_inputs(&memory),
        None => fallback_memory_embedding_inputs(body),
    };
    let texts = inputs
        .iter()
        .map(|input| input.text.clone())
        .collect::<Vec<_>>();
    let embeddings = ollama
        .embed_batch_with_model(config.memory_embed_model(), &texts)
        .await?;
    let mut dimensions = 0;
    for (input, embedding) in inputs.iter().zip(embeddings.iter()) {
        dimensions = embedding.len();
        store.set_memory_embedding_with_metadata(
            project_id,
            memory_id,
            config.memory_embed_model(),
            &input.kind,
            Some(&input.content_hash),
            embedding,
        )?;
    }
    Ok(dimensions)
}

pub async fn embed_code_symbols(
    config: &Config,
    store: &Store,
    project_id: &str,
    symbols: Vec<CodeSymbol>,
) -> Result<usize> {
    Ok(
        embed_code_symbols_with_report(config, store, project_id, symbols)
            .await?
            .completed,
    )
}

async fn embed_code_symbols_with_report(
    config: &Config,
    store: &Store,
    project_id: &str,
    symbols: Vec<CodeSymbol>,
) -> Result<CodeSymbolEmbeddingReport> {
    let ollama = ollama_from_config(config);
    let model = config.code_embed_model();
    let mut pending = Vec::new();
    let mut report = CodeSymbolEmbeddingReport::default();
    for symbol in symbols {
        for input in code_symbol_embedding_inputs(&symbol) {
            if store.attach_cached_code_symbol_embedding_kind(
                project_id,
                &symbol.id,
                model,
                &input.kind,
                &input.content_hash,
            )? {
                report.completed += 1;
                report.reused += 1;
                continue;
            }
            pending.push(PendingCodeSymbolEmbedding {
                symbol_id: symbol.id.clone(),
                kind: input.kind,
                text: input.text,
                content_hash: input.content_hash,
            });
        }
    }
    for chunk in pending.chunks(EMBEDDING_BATCH_SIZE) {
        let texts = chunk
            .iter()
            .map(|item| item.text.clone())
            .collect::<Vec<_>>();
        let embeddings = ollama.embed_batch_with_model(model, &texts).await?;
        for (item, embedding) in chunk.iter().zip(embeddings.iter()) {
            store.set_code_symbol_embedding_kind_with_cache(
                project_id,
                &item.symbol_id,
                model,
                &item.kind,
                &item.content_hash,
                embedding,
            )?;
            report.completed += 1;
            report.generated += 1;
        }
    }
    Ok(report)
}

pub async fn embed_indexed_code_symbols(
    config: &Config,
    store: &Store,
    project_id: &str,
    indexed_files: &[String],
    limit: usize,
) -> Result<usize> {
    let symbols = store.code_symbols_missing_embeddings_for_files(
        project_id,
        config.code_embed_model(),
        indexed_files,
        limit,
    )?;
    embed_code_symbols(config, store, project_id, symbols).await
}

pub async fn embed_missing(
    config: &Config,
    store: &Store,
    project_id: &str,
    limit: usize,
    scope: &str,
) -> Result<EmbedMissingReport> {
    let embed_memories = matches!(scope, "all" | "memory" | "memories");
    let embed_code = matches!(scope, "all" | "code" | "symbols" | "code_symbols");
    if !embed_memories && !embed_code {
        bail!("invalid embed scope `{scope}`; use all, memories, or code");
    }

    let mut report = EmbedMissingReport::default();

    if embed_memories {
        let memories =
            store.memories_missing_embeddings(project_id, config.memory_embed_model(), limit)?;
        let ollama = ollama_from_config(config);
        for chunk in memories.chunks(EMBEDDING_BATCH_SIZE) {
            let inputs = chunk
                .iter()
                .flat_map(|memory| {
                    memory_embedding_inputs(memory)
                        .into_iter()
                        .map(|input| (memory.id.clone(), input))
                })
                .collect::<Vec<_>>();
            let texts = inputs
                .iter()
                .map(|(_, input)| input.text.clone())
                .collect::<Vec<_>>();
            let embeddings = ollama
                .embed_batch_with_model(config.memory_embed_model(), &texts)
                .await?;
            for ((memory_id, input), embedding) in inputs.iter().zip(embeddings.iter()) {
                store.set_memory_embedding_with_metadata(
                    project_id,
                    memory_id,
                    config.memory_embed_model(),
                    &input.kind,
                    Some(&input.content_hash),
                    embedding,
                )?;
            }
            report.memories += chunk.len();
        }
    }

    if embed_code {
        let embedded_symbols =
            store.code_symbols_with_embeddings(project_id, config.code_embed_model(), limit)?;
        for symbol in &embedded_symbols {
            let content_hash = code_symbol_embedding_hash(symbol);
            if store.cache_existing_code_symbol_embedding(
                project_id,
                &symbol.id,
                config.code_embed_model(),
                &content_hash,
            )? {
                report.code_symbols_cached += 1;
            }
        }
        let symbols =
            store.code_symbols_missing_embeddings(project_id, config.code_embed_model(), limit)?;
        let code_report =
            embed_code_symbols_with_report(config, store, project_id, symbols).await?;
        report.code_symbols = code_report.completed;
        report.code_symbols_reused += code_report.reused;
        report.code_symbols_generated = code_report.generated;
    }

    Ok(report)
}

fn ollama_from_config(config: &Config) -> OllamaClient {
    OllamaClient::new(
        config.ollama_base_url.clone(),
        config.ollama_llm_model.clone(),
    )
}

pub fn code_symbol_embedding_text(symbol: &CodeSymbol) -> String {
    truncate_text(
        &format!(
            "language: {}\nkind: {}\nname: {}\nfile: {}\nlines: {}-{}\nsignature: {}\n\n{}",
            symbol.language,
            symbol.kind,
            symbol.name,
            symbol.file_path,
            symbol.start_line,
            symbol.end_line,
            symbol.signature,
            symbol.body
        ),
        12_000,
    )
}

pub fn code_symbol_embedding_hash(symbol: &CodeSymbol) -> String {
    embedding_content_hash(&code_symbol_embedding_text(symbol))
}

fn code_symbol_signature_embedding_text(symbol: &CodeSymbol) -> String {
    truncate_text(
        &format!(
            "language: {}\nkind: {}\nname: {}\nfile: {}\nlines: {}-{}\nsignature: {}\nparent: {}",
            symbol.language,
            symbol.kind,
            symbol.name,
            symbol.file_path,
            symbol.start_line,
            symbol.end_line,
            symbol.signature,
            symbol.parent_id.as_deref().unwrap_or("<none>")
        ),
        4_000,
    )
}

fn code_symbol_embedding_inputs(symbol: &CodeSymbol) -> Vec<EmbeddingInput> {
    let body = code_symbol_embedding_text(symbol);
    let signature = code_symbol_signature_embedding_text(symbol);
    vec![
        EmbeddingInput {
            kind: "body".to_string(),
            content_hash: embedding_content_hash(&body),
            text: body,
        },
        EmbeddingInput {
            kind: "signature".to_string(),
            content_hash: embedding_content_hash(&signature),
            text: signature,
        },
    ]
}

fn embedding_content_hash(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

fn memory_embedding_text(memory: &Memory) -> String {
    truncate_text(
        &format!(
            "scope: {}\ntier: {}\nkind: {}\nsource: {}\ntags: {}\nimportance: {:.2}\nconfidence: {:.2}\n\n{}",
            memory.scope,
            memory.memory_tier,
            memory.kind,
            memory.source.as_deref().unwrap_or("<none>"),
            memory.tags.join(","),
            memory.importance,
            memory.confidence,
            memory.body
        ),
        12_000,
    )
}

fn memory_metadata_embedding_text(memory: &Memory) -> String {
    truncate_text(
        &format!(
            "scope: {}\ntier: {}\nkind: {}\nsource: {}\ntags: {}\nimportance: {:.2}\nconfidence: {:.2}\nbody_preview: {}",
            memory.scope,
            memory.memory_tier,
            memory.kind,
            memory.source.as_deref().unwrap_or("<none>"),
            memory.tags.join(","),
            memory.importance,
            memory.confidence,
            truncate_text(&memory.body, 1_000)
        ),
        4_000,
    )
}

fn memory_embedding_inputs(memory: &Memory) -> Vec<EmbeddingInput> {
    let body = memory_embedding_text(memory);
    let metadata = memory_metadata_embedding_text(memory);
    vec![
        EmbeddingInput {
            kind: "body".to_string(),
            content_hash: embedding_content_hash(&body),
            text: body,
        },
        EmbeddingInput {
            kind: "metadata".to_string(),
            content_hash: embedding_content_hash(&metadata),
            text: metadata,
        },
    ]
}

fn fallback_memory_embedding_inputs(body: &str) -> Vec<EmbeddingInput> {
    let body = truncate_text(body, 12_000);
    let metadata = truncate_text(
        &format!("body_preview: {}", truncate_text(&body, 1_000)),
        4_000,
    );
    vec![
        EmbeddingInput {
            kind: "body".to_string(),
            content_hash: embedding_content_hash(&body),
            text: body,
        },
        EmbeddingInput {
            kind: "metadata".to_string(),
            content_hash: embedding_content_hash(&metadata),
            text: metadata,
        },
    ]
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = text.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

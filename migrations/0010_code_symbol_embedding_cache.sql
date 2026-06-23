CREATE TABLE IF NOT EXISTS code_symbol_embedding_cache_1024 (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(project_id, model, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_code_symbol_embedding_cache_1024_project_model
    ON code_symbol_embedding_cache_1024(project_id, model);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (10, 'code_symbol_embedding_cache')
ON CONFLICT (version) DO NOTHING;

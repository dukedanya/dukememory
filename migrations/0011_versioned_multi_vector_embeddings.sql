ALTER TABLE memory_embeddings_4096
    ADD COLUMN IF NOT EXISTS embedding_kind TEXT NOT NULL DEFAULT 'body',
    ADD COLUMN IF NOT EXISTS content_hash TEXT,
    ADD COLUMN IF NOT EXISTS dimensions INTEGER NOT NULL DEFAULT 4096,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE code_symbol_embeddings_1024
    ADD COLUMN IF NOT EXISTS embedding_kind TEXT NOT NULL DEFAULT 'body',
    ADD COLUMN IF NOT EXISTS content_hash TEXT,
    ADD COLUMN IF NOT EXISTS dimensions INTEGER NOT NULL DEFAULT 1024,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE code_symbol_embedding_cache_1024
    ADD COLUMN IF NOT EXISTS embedding_kind TEXT NOT NULL DEFAULT 'body',
    ADD COLUMN IF NOT EXISTS dimensions INTEGER NOT NULL DEFAULT 1024,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'memory_embeddings_4096_pkey'
          AND conrelid = 'memory_embeddings_4096'::regclass
    ) THEN
        ALTER TABLE memory_embeddings_4096 DROP CONSTRAINT memory_embeddings_4096_pkey;
    END IF;
END $$;

ALTER TABLE memory_embeddings_4096
    ADD CONSTRAINT memory_embeddings_4096_pkey PRIMARY KEY(memory_id, model, embedding_kind);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'code_symbol_embeddings_1024_pkey'
          AND conrelid = 'code_symbol_embeddings_1024'::regclass
    ) THEN
        ALTER TABLE code_symbol_embeddings_1024 DROP CONSTRAINT code_symbol_embeddings_1024_pkey;
    END IF;
END $$;

ALTER TABLE code_symbol_embeddings_1024
    ADD CONSTRAINT code_symbol_embeddings_1024_pkey PRIMARY KEY(symbol_id, model, embedding_kind);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'code_symbol_embedding_cache_1024_pkey'
          AND conrelid = 'code_symbol_embedding_cache_1024'::regclass
    ) THEN
        ALTER TABLE code_symbol_embedding_cache_1024 DROP CONSTRAINT code_symbol_embedding_cache_1024_pkey;
    END IF;
END $$;

ALTER TABLE code_symbol_embedding_cache_1024
    ADD CONSTRAINT code_symbol_embedding_cache_1024_pkey
        PRIMARY KEY(project_id, model, embedding_kind, content_hash);

CREATE INDEX IF NOT EXISTS idx_memory_embeddings_4096_project_model_kind
    ON memory_embeddings_4096(project_id, model, embedding_kind);

CREATE INDEX IF NOT EXISTS idx_memory_embeddings_4096_project_model_hash
    ON memory_embeddings_4096(project_id, model, embedding_kind, content_hash);

CREATE INDEX IF NOT EXISTS idx_code_symbol_embeddings_1024_project_model_kind
    ON code_symbol_embeddings_1024(project_id, model, embedding_kind);

CREATE INDEX IF NOT EXISTS idx_code_symbol_embeddings_1024_project_model_hash
    ON code_symbol_embeddings_1024(project_id, model, embedding_kind, content_hash);

CREATE INDEX IF NOT EXISTS idx_code_symbol_embedding_cache_1024_project_model_kind
    ON code_symbol_embedding_cache_1024(project_id, model, embedding_kind);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (11, 'versioned_multi_vector_embeddings')
ON CONFLICT (version) DO NOTHING;

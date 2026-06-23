CREATE TABLE IF NOT EXISTS code_memories (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    symbol_id TEXT,
    file_path TEXT,
    kind TEXT NOT NULL DEFAULT 'note',
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    source TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    status_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(kind, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(file_path, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(body, '')), 'C')
    ) STORED,
    CHECK (status IN ('active', 'archived')),
    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CHECK (symbol_id IS NOT NULL OR file_path IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_code_memories_project_body_live
    ON code_memories(project_id, body_hash)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_code_memories_project_symbol
    ON code_memories(project_id, symbol_id)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_code_memories_project_file
    ON code_memories(project_id, file_path)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_code_memories_project_status_updated
    ON code_memories(project_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_code_memories_search
    ON code_memories USING GIN(search_vector);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (5, 'code_memories')
ON CONFLICT(version) DO NOTHING;

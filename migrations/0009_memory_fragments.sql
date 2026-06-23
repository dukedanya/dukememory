CREATE TABLE IF NOT EXISTS memory_fragments (
    id TEXT PRIMARY KEY,
    memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    body_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(memory_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_memory_fragments_project_search
    ON memory_fragments USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_memory_fragments_project_memory
    ON memory_fragments(project_id, memory_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_memory_fragments_project_hash
    ON memory_fragments(project_id, body_hash);

INSERT INTO memory_fragments(id, memory_id, project_id, chunk_index, body_hash, text)
SELECT
    m.id::text || '#' || paragraph.ordinality::text,
    m.id,
    m.project_id,
    paragraph.ordinality::integer - 1,
    m.body_hash,
    btrim(paragraph.value)
FROM memories m
CROSS JOIN LATERAL regexp_split_to_table(m.body, E'\\n\\n+') WITH ORDINALITY AS paragraph(value, ordinality)
WHERE btrim(paragraph.value) <> ''
ON CONFLICT(id) DO NOTHING;

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (9, 'memory_fragments')
ON CONFLICT(version) DO NOTHING;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE OR REPLACE FUNCTION ts(value TIMESTAMPTZ)
RETURNS TEXT
LANGUAGE SQL
STABLE
AS $$
    SELECT to_char(value AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
$$;

CREATE TABLE IF NOT EXISTS dukememory_schema_migrations (
    version BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT,
    project_type TEXT NOT NULL DEFAULT 'generic',
    description TEXT,
    domains JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    scope TEXT NOT NULL DEFAULT 'project',
    kind TEXT NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    source TEXT,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by UUID REFERENCES memories(id),
    status_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(kind, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(body, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(source, '')), 'D')
    ) STORED,
    CHECK (scope IN ('project', 'session', 'module', 'user', 'global')),
    CHECK (status IN ('pending', 'active', 'superseded', 'archived')),
    CHECK (importance >= 0.0 AND importance <= 1.0),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_memories_project_status
    ON memories(project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_project_kind
    ON memories(project_id, kind);
CREATE INDEX IF NOT EXISTS idx_memories_project_scope
    ON memories(project_id, scope, status);
CREATE INDEX IF NOT EXISTS idx_memories_project_body_hash
    ON memories(project_id, body_hash, status);
CREATE INDEX IF NOT EXISTS idx_memories_search_vector
    ON memories USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_memories_body_trgm
    ON memories USING GIN(body gin_trgm_ops);

CREATE TABLE IF NOT EXISTS memory_embeddings_4096 (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    embedding VECTOR(4096) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_embeddings_4096_project_model
    ON memory_embeddings_4096(project_id, model);

CREATE TABLE IF NOT EXISTS code_files (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    hash TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    line_count INTEGER NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (project_id, path)
);

CREATE TABLE IF NOT EXISTS code_symbols (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    signature TEXT NOT NULL,
    body TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    parent_id TEXT,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(kind, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(signature, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(body, '')), 'C') ||
        setweight(to_tsvector('simple', coalesce(file_path, '')), 'D')
    ) STORED,
    FOREIGN KEY (project_id, file_path) REFERENCES code_files(project_id, path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_project_name
    ON code_symbols(project_id, name);
CREATE INDEX IF NOT EXISTS idx_code_symbols_project_file
    ON code_symbols(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_code_symbols_search_vector
    ON code_symbols USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_code_symbols_body_trgm
    ON code_symbols USING GIN(body gin_trgm_ops);

CREATE TABLE IF NOT EXISTS code_relations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    from_symbol_id TEXT,
    from_file_path TEXT NOT NULL,
    relation_kind TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_symbol_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_code_relations_from
    ON code_relations(project_id, from_symbol_id, relation_kind);
CREATE INDEX IF NOT EXISTS idx_code_relations_target
    ON code_relations(project_id, target_name, relation_kind);
CREATE INDEX IF NOT EXISTS idx_code_relations_target_symbol
    ON code_relations(project_id, target_symbol_id, relation_kind);

CREATE TABLE IF NOT EXISTS code_symbol_embeddings_1024 (
    symbol_id TEXT PRIMARY KEY REFERENCES code_symbols(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_code_symbol_embeddings_1024_project_model
    ON code_symbol_embeddings_1024(project_id, model);

CREATE TABLE IF NOT EXISTS memory_episodes (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    summary TEXT,
    raw_ref TEXT,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_entities (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(project_id, entity_type, name)
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_project_name
    ON memory_entities(project_id, name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_name_trgm
    ON memory_entities USING GIN(name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS memory_facts (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES memory_entities(id) ON DELETE SET NULL,
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    episode_id UUID REFERENCES memory_episodes(id) ON DELETE SET NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_memory_facts_project_entity
    ON memory_facts(project_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_memory_facts_project_predicate
    ON memory_facts(project_id, predicate);

CREATE TABLE IF NOT EXISTS memory_edges (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    from_entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
    to_entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    episode_id UUID REFERENCES memory_episodes(id) ON DELETE SET NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_memory_edges_project_from
    ON memory_edges(project_id, from_entity_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_edges_project_to
    ON memory_edges(project_id, to_entity_id, relation_type);

CREATE TABLE IF NOT EXISTS memory_events (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    memory_id UUID REFERENCES memories(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL DEFAULT 'dukememory',
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_events_project_created
    ON memory_events(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS eval_suites (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id UUID PRIMARY KEY,
    suite_id UUID NOT NULL REFERENCES eval_suites(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    query TEXT NOT NULL,
    expected_contains JSONB NOT NULL DEFAULT '[]'::jsonb,
    forbidden_contains JSONB NOT NULL DEFAULT '[]'::jsonb,
    min_results INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY,
    suite_id UUID REFERENCES eval_suites(id) ON DELETE SET NULL,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    passed_cases INTEGER NOT NULL,
    failed_cases INTEGER NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (1, 'initial_pg')
ON CONFLICT(version) DO NOTHING;

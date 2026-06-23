ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS usage_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    ADD COLUMN IF NOT EXISTS contradiction_risk DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE code_memories
    ADD COLUMN IF NOT EXISTS symbol_body_hash TEXT,
    ADD COLUMN IF NOT EXISTS usage_count BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    ADD COLUMN IF NOT EXISTS contradiction_risk DOUBLE PRECISION NOT NULL DEFAULT 0.0;

UPDATE memories
SET quality_score = LEAST(1.0, GREATEST(0.0, (importance * 0.55) + (confidence * 0.45)))
WHERE quality_score = 0.5;

UPDATE code_memories
SET quality_score = LEAST(1.0, GREATEST(0.0, confidence))
WHERE quality_score = 0.5;

CREATE TABLE IF NOT EXISTS dukememory_retrieval_events (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    tool TEXT NOT NULL,
    query TEXT NOT NULL,
    task_type TEXT NOT NULL,
    token_budget INTEGER NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    latency_ms BIGINT NOT NULL,
    memory_fragments INTEGER NOT NULL,
    code_hits INTEGER NOT NULL,
    graph_items INTEGER NOT NULL,
    code_memories INTEGER NOT NULL,
    plan JSONB NOT NULL DEFAULT '{}'::jsonb,
    audit JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dukememory_retrieval_events_project_created
    ON dukememory_retrieval_events(project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_code_memories_project_quality
    ON code_memories(project_id, status, quality_score DESC, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_project_quality
    ON memories(project_id, status, quality_score DESC, updated_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (12, 'context_planner_quality')
ON CONFLICT(version) DO NOTHING;

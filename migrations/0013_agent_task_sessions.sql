CREATE TABLE IF NOT EXISTS dukememory_task_sessions (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('planned', 'running', 'completed', 'failed', 'archived')),
    phase TEXT NOT NULL DEFAULT 'prepare',
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    code_symbol_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
    test_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary TEXT,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_dukememory_task_sessions_project_updated
    ON dukememory_task_sessions(project_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_dukememory_task_sessions_project_status
    ON dukememory_task_sessions(project_id, status, updated_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (13, 'agent_task_sessions')
ON CONFLICT(version) DO NOTHING;

CREATE TABLE IF NOT EXISTS dukememory_audit_events (
    id UUID PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    actor TEXT NOT NULL DEFAULT 'dukememory',
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dukememory_audit_project_created
    ON dukememory_audit_events(project_id, created_at DESC);

ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS suite_name TEXT,
    ADD COLUMN IF NOT EXISTS suite_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_eval_runs_project_suite_mode_created
    ON eval_runs(project_id, suite_name, suite_hash, mode, created_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (4, 'audit_eval_ops')
ON CONFLICT(version) DO NOTHING;

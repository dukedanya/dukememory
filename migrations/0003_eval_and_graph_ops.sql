ALTER TABLE eval_cases
    ADD COLUMN IF NOT EXISTS expected_ids JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_eval_runs_project_created
    ON eval_runs(project_id, created_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (3, 'eval_and_graph_ops')
ON CONFLICT(version) DO NOTHING;

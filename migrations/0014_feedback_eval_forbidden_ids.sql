ALTER TABLE eval_cases
    ADD COLUMN IF NOT EXISTS forbidden_ids JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_eval_cases_project_created
    ON eval_cases(project_id, created_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (14, 'feedback_eval_forbidden_ids')
ON CONFLICT(version) DO NOTHING;

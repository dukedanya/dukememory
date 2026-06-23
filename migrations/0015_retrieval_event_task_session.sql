ALTER TABLE dukememory_retrieval_events
    ADD COLUMN IF NOT EXISTS task_session_id UUID
        REFERENCES dukememory_task_sessions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_dukememory_retrieval_events_project_session
    ON dukememory_retrieval_events(project_id, task_session_id, created_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (15, 'retrieval_event_task_session')
ON CONFLICT(version) DO NOTHING;

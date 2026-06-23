ALTER TABLE code_memories
    DROP CONSTRAINT IF EXISTS code_memories_status_check;

ALTER TABLE code_memories
    ADD CONSTRAINT code_memories_status_check
    CHECK (status IN ('pending', 'active', 'archived'));

CREATE INDEX IF NOT EXISTS idx_code_memories_project_pending_updated
    ON code_memories(project_id, updated_at DESC)
    WHERE status = 'pending';

DROP INDEX IF EXISTS idx_code_memories_project_body_live;

CREATE UNIQUE INDEX IF NOT EXISTS idx_code_memories_project_body_live
    ON code_memories(project_id, body_hash)
    WHERE status IN ('pending', 'active');

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (6, 'code_memory_lifecycle')
ON CONFLICT(version) DO NOTHING;

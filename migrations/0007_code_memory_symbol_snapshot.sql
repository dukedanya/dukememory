ALTER TABLE code_memories
    ADD COLUMN IF NOT EXISTS symbol_name TEXT,
    ADD COLUMN IF NOT EXISTS symbol_kind TEXT,
    ADD COLUMN IF NOT EXISTS symbol_signature TEXT,
    ADD COLUMN IF NOT EXISTS symbol_start_line INTEGER,
    ADD COLUMN IF NOT EXISTS symbol_end_line INTEGER,
    ADD COLUMN IF NOT EXISTS last_relinked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS relink_attempts INTEGER NOT NULL DEFAULT 0;

UPDATE code_memories cm
SET symbol_name = COALESCE(cm.symbol_name, s.name),
    symbol_kind = COALESCE(cm.symbol_kind, s.kind),
    symbol_signature = COALESCE(cm.symbol_signature, s.signature),
    symbol_start_line = COALESCE(cm.symbol_start_line, s.start_line),
    symbol_end_line = COALESCE(cm.symbol_end_line, s.end_line),
    file_path = COALESCE(cm.file_path, s.file_path)
FROM code_symbols s
WHERE s.project_id = cm.project_id
  AND s.id = cm.symbol_id;

CREATE INDEX IF NOT EXISTS idx_code_memories_project_symbol_snapshot
    ON code_memories(project_id, file_path, symbol_name, symbol_kind)
    WHERE status IN ('pending', 'active') AND symbol_name IS NOT NULL;

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (7, 'code_memory_symbol_snapshot')
ON CONFLICT(version) DO NOTHING;

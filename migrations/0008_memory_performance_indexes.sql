CREATE INDEX IF NOT EXISTS idx_memories_project_core_context
    ON memories(project_id, memory_tier, kind, importance DESC, confidence DESC, updated_at DESC)
    WHERE status = 'active'
      AND (memory_tier = 'core' OR kind IN ('project_rule', 'constraint'));

CREATE INDEX IF NOT EXISTS idx_memories_project_status_kind_updated
    ON memories(project_id, status, kind, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_project_tier_status_updated
    ON memories(project_id, memory_tier, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_code_symbols_project_name_file_kind
    ON code_symbols(project_id, name, file_path, kind, start_line);

CREATE INDEX IF NOT EXISTS idx_code_memories_project_active_symbol_file
    ON code_memories(project_id, symbol_id, file_path, updated_at DESC)
    WHERE status = 'active';

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (8, 'memory_performance_indexes')
ON CONFLICT(version) DO NOTHING;

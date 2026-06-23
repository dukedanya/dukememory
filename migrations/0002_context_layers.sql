ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS memory_tier TEXT NOT NULL DEFAULT 'archival';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'memories_memory_tier_check'
    ) THEN
        ALTER TABLE memories
            ADD CONSTRAINT memories_memory_tier_check
            CHECK (memory_tier IN ('core', 'archival', 'conversation'));
    END IF;
END $$;

ALTER TABLE memory_episodes
    ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE memory_facts
    ADD COLUMN IF NOT EXISTS invalidated_by UUID REFERENCES memory_facts(id) ON DELETE SET NULL;

ALTER TABLE memory_edges
    ADD COLUMN IF NOT EXISTS invalidated_by UUID REFERENCES memory_edges(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_memories_project_tier
    ON memories(project_id, memory_tier, status, importance DESC, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_episodes_project_observed
    ON memory_episodes(project_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_facts_project_valid
    ON memory_facts(project_id, valid_from, valid_to, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_edges_project_valid
    ON memory_edges(project_id, valid_from, valid_to, observed_at DESC);

INSERT INTO dukememory_schema_migrations(version, name)
VALUES (2, 'context_layers')
ON CONFLICT(version) DO NOTHING;

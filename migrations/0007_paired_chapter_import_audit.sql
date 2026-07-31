-- Forward-only audit state for local-only paired JSONL + Visual DOCX imports.
-- It intentionally stores hashes/counts only: never Drive IDs, storage keys,
-- document bytes, source JSONL, or enriched JSONL.

CREATE TABLE IF NOT EXISTS rag_paired_imports (
    import_hash CHAR(64) PRIMARY KEY,
    actor_id TEXT NOT NULL,
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('validating', 'assets_uploaded', 'queued', 'failed')),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    referenced_visual_count INTEGER NOT NULL DEFAULT 0 CHECK (referenced_visual_count >= 0),
    unused_visual_count INTEGER NOT NULL DEFAULT 0 CHECK (unused_visual_count >= 0),
    asset_hashes JSONB NOT NULL DEFAULT '[]'::jsonb,
    job_id UUID NULL REFERENCES job_queue(id) ON DELETE SET NULL,
    error_code TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE rag_paired_imports ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON rag_paired_imports FROM PUBLIC, anon, authenticated;
CREATE INDEX IF NOT EXISTS idx_rag_paired_imports_scope_created
    ON rag_paired_imports (board_id, class_id, subject_id, created_at DESC);

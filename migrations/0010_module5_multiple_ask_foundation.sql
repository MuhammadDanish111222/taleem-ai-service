-- Module 5 Run 1: private temporary inputs and durable Multiple Ask work.
-- Forward-only. This deliberately does not alter Single Ask, solved_papers,
-- job_queue infrastructure states, or the generated-answer candidate tables.

SET search_path = public, pg_catalog;

-- A session owns one unguessable private Storage object. Signed upload URLs and
-- Storage credentials are intentionally never persisted here.
CREATE TABLE multiple_ask_upload_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_request_id UUID NOT NULL,
    uid_hash CHAR(64) NOT NULL,
    account_tier TEXT NOT NULL CHECK (account_tier IN ('anonymous', 'google', 'premium')),
    input_kind TEXT NOT NULL CHECK (input_kind IN ('image', 'pdf', 'text')),
    expected_content_type TEXT NULL,
    expected_size_bytes BIGINT NULL CHECK (expected_size_bytes IS NULL OR expected_size_bytes > 0),
    storage_bucket TEXT NULL,
    storage_object_key TEXT NULL,
    status TEXT NOT NULL DEFAULT 'created'
        CHECK (status IN ('created', 'uploaded', 'finalized', 'expired', 'cancelled')),
    upload_url_expires_at TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    finalized_at TIMESTAMPTZ NULL,
    purged_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (char_length(uid_hash) = 64),
    CHECK (
        (input_kind = 'text'
         AND expected_content_type IS NULL AND expected_size_bytes IS NULL
         AND storage_bucket IS NULL AND storage_object_key IS NULL)
        OR
        (input_kind = 'image'
         AND expected_content_type IN ('image/jpeg', 'image/png', 'image/webp')
         AND expected_size_bytes IS NOT NULL AND storage_bucket IS NOT NULL
         AND storage_object_key IS NOT NULL)
        OR
        (input_kind = 'pdf'
         AND expected_content_type = 'application/pdf'
         AND expected_size_bytes IS NOT NULL AND storage_bucket IS NOT NULL
         AND storage_object_key IS NOT NULL)
    ),
    CHECK ((status = 'finalized') = (finalized_at IS NOT NULL))
);
CREATE UNIQUE INDEX idx_multiple_ask_session_idempotency
    ON multiple_ask_upload_sessions (uid_hash, client_request_id);
CREATE UNIQUE INDEX idx_multiple_ask_session_storage_object
    ON multiple_ask_upload_sessions (storage_bucket, storage_object_key)
    WHERE storage_bucket IS NOT NULL;
CREATE INDEX idx_multiple_ask_session_expiry
    ON multiple_ask_upload_sessions (expires_at, id)
    WHERE purged_at IS NULL AND status IN ('created', 'uploaded', 'finalized', 'expired');

-- Text is separate from session metadata to make source-text retention explicit.
-- It is input material only, never an answer cache and never sent to a provider in Run 1.
CREATE TABLE multiple_ask_text_inputs (
    session_id UUID PRIMARY KEY REFERENCES multiple_ask_upload_sessions(id) ON DELETE RESTRICT,
    input_text TEXT NOT NULL CHECK (char_length(input_text) BETWEEN 1 AND 20000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    purged_at TIMESTAMPTZ NULL
);

-- This parent is the durable business record. Its workflow state is distinct
-- from job_queue.status, whose shared lease/retry state values remain unchanged.
CREATE TABLE multiple_ask_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    upload_session_id UUID NOT NULL UNIQUE REFERENCES multiple_ask_upload_sessions(id) ON DELETE RESTRICT,
    queue_job_id UUID NOT NULL UNIQUE REFERENCES job_queue(id) ON DELETE RESTRICT,
    uid_hash CHAR(64) NOT NULL,
    client_request_id UUID NOT NULL,
    account_tier TEXT NOT NULL CHECK (account_tier IN ('anonymous', 'google', 'premium')),
    input_kind TEXT NOT NULL CHECK (input_kind IN ('image', 'pdf', 'text')),
    workflow_status TEXT NOT NULL DEFAULT 'validation_queued'
        CHECK (workflow_status IN ('validation_queued', 'validated', 'invalid', 'cancelled')),
    retention_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (char_length(uid_hash) = 64),
    UNIQUE (uid_hash, client_request_id)
);
CREATE INDEX idx_multiple_ask_jobs_retention
    ON multiple_ask_jobs (retention_expires_at, id)
    WHERE workflow_status IN ('validation_queued', 'validated', 'invalid');

-- Future extraction creates one row per item. "mixed" is intentionally absent:
-- a batch can be mixed, but every item has exactly one persisted answer mode.
CREATE TABLE multiple_ask_job_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    multiple_ask_job_id UUID NOT NULL REFERENCES multiple_ask_jobs(id) ON DELETE RESTRICT,
    item_index INTEGER NOT NULL CHECK (item_index >= 0),
    raw_question TEXT NULL,
    normalized_question TEXT NULL,
    question_hash CHAR(64) NULL,
    answer_mode TEXT NULL CHECK (answer_mode IN ('short', 'long', 'mcq', 'not_clear')),
    item_status TEXT NOT NULL DEFAULT 'pending_extraction'
        CHECK (item_status IN ('pending_extraction', 'needs_correction', 'ready_to_answer', 'answering', 'answered', 'failed', 'cancelled')),
    ai_request_id UUID NULL UNIQUE REFERENCES ai_requests(id) ON DELETE RESTRICT,
    retention_expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (question_hash IS NULL OR char_length(question_hash) = 64),
    CHECK (
        (answer_mode = 'not_clear' AND item_status IN ('needs_correction', 'cancelled'))
        OR (answer_mode IS NULL AND item_status = 'pending_extraction')
        OR (answer_mode IN ('short', 'long', 'mcq'))
    ),
    UNIQUE (multiple_ask_job_id, item_index)
);
CREATE INDEX idx_multiple_ask_items_work
    ON multiple_ask_job_items (multiple_ask_job_id, item_status, item_index);
CREATE INDEX idx_multiple_ask_items_retention
    ON multiple_ask_job_items (retention_expires_at, id)
    WHERE item_status IN ('pending_extraction', 'needs_correction', 'failed', 'cancelled');

-- Bounded cleanup is auditable without retaining source bytes, object keys, or URLs.
CREATE TABLE multiple_ask_cleanup_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    session_id UUID NOT NULL REFERENCES multiple_ask_upload_sessions(id) ON DELETE RESTRICT,
    action TEXT NOT NULL CHECK (action IN ('preview', 'storage_delete_requested', 'metadata_purged', 'failed')),
    error_code TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, session_id, action)
);
CREATE INDEX idx_multiple_ask_cleanup_audit_session
    ON multiple_ask_cleanup_audit (session_id, created_at);

ALTER TABLE multiple_ask_upload_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE multiple_ask_text_inputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE multiple_ask_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE multiple_ask_job_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE multiple_ask_cleanup_audit ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON multiple_ask_upload_sessions, multiple_ask_text_inputs,
    multiple_ask_jobs, multiple_ask_job_items, multiple_ask_cleanup_audit
FROM PUBLIC, anon, authenticated;

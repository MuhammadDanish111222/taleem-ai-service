-- Module 5 Stage 5: one durable general-knowledge result for a paper's MCQs.
SET search_path = public, pg_catalog;

CREATE TABLE multiple_ask_mcq_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    multiple_ask_job_id UUID NOT NULL REFERENCES multiple_ask_jobs(id) ON DELETE RESTRICT,
    answer_epoch INTEGER NOT NULL CHECK (answer_epoch >= 1),
    batch_identity CHAR(64) NOT NULL CHECK (char_length(batch_identity) = 64),
    item_ids JSONB NOT NULL,
    results JSONB NOT NULL,
    prompt_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
    latency_ms INTEGER NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (multiple_ask_job_id, answer_epoch),
    UNIQUE (batch_identity)
);

ALTER TABLE multiple_ask_job_items
    ADD COLUMN mcq_result JSONB NULL;

ALTER TABLE multiple_ask_mcq_batches ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON multiple_ask_mcq_batches FROM PUBLIC, anon, authenticated;

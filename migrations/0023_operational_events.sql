-- Module 7 Run 2: small, content-free events for outcomes not represented by domain tables.
SET search_path = public, pg_catalog;
CREATE TABLE operational_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature TEXT NOT NULL CHECK (char_length(feature) <= 80),
    event_type TEXT NOT NULL CHECK (event_type IN ('retrieval_outcome','quota_block','test_generation_failure')),
    outcome TEXT NOT NULL CHECK (char_length(outcome) <= 80),
    error_code TEXT NULL CHECK (error_code IS NULL OR char_length(error_code) <= 120),
    request_id UUID NULL,
    job_id UUID NULL REFERENCES job_queue(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_operational_events_window ON operational_events (event_type, created_at DESC);
ALTER TABLE operational_events ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON operational_events FROM PUBLIC, anon, authenticated;

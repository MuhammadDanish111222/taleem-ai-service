-- Module 5 Run 2: temporary OCR cache, deterministic extraction and correction.
-- Forward-only. This deliberately leaves shared job_queue.status untouched.

SET search_path = public, pg_catalog;

CREATE TABLE multiple_ask_normalized_sources (
    session_id UUID PRIMARY KEY REFERENCES multiple_ask_upload_sessions(id) ON DELETE CASCADE,
    normalized_text TEXT NOT NULL CHECK (char_length(normalized_text) <= 30000),
    source_locators JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(source_locators)='array'),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('pasted_text','pdf_embedded_text','pdf_ocr','image_ocr')),
    ocr_provider TEXT NULL,
    ocr_version TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE multiple_ask_jobs
    ADD COLUMN extraction_epoch INTEGER NOT NULL DEFAULT 0 CHECK (extraction_epoch >= 0),
    ADD COLUMN resume_request_id UUID NULL;
ALTER TABLE multiple_ask_jobs
    DROP CONSTRAINT multiple_ask_jobs_workflow_check,
    ADD CONSTRAINT multiple_ask_jobs_workflow_check CHECK (workflow_status IN (
        'queued','validating','validated','extracting','needs_correction','ready_to_answer',
        'answering','partially_completed','completed','failed','cancelled','invalid','limit_reached'
    ));

ALTER TABLE multiple_ask_job_items
    ALTER COLUMN retention_expires_at DROP NOT NULL,
    ADD COLUMN correction_text TEXT NULL,
    ADD COLUMN correction_request_id UUID NULL UNIQUE,
    ADD CONSTRAINT multiple_ask_items_correction_text_check CHECK (
        correction_text IS NULL
        OR char_length(btrim(correction_text)) BETWEEN 1 AND 30000
    );

ALTER TABLE multiple_ask_normalized_sources ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON multiple_ask_normalized_sources FROM PUBLIC, anon, authenticated;

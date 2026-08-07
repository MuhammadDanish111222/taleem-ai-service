-- Module 5 Run 2 hardening: paper labels/context, overflow handling and refunds.
-- Forward-only; leaves shared job_queue state untouched.

SET search_path = public, pg_catalog;

ALTER TABLE multiple_ask_jobs
    ADD COLUMN terminal_error_code TEXT NULL,
    ADD COLUMN quota_refunded_at TIMESTAMPTZ NULL;

ALTER TABLE multiple_ask_jobs
    DROP CONSTRAINT multiple_ask_jobs_workflow_check,
    ADD CONSTRAINT multiple_ask_jobs_workflow_check CHECK (workflow_status IN (
        'queued','validating','validated','extracting','needs_correction','ready_to_answer',
        'answering','partially_completed','completed','failed','cancelled','invalid',
        'limit_reached','too_many_questions'
    )),
    DROP CONSTRAINT multiple_ask_jobs_quota_check,
    ADD CONSTRAINT multiple_ask_jobs_quota_check CHECK (quota_status IN (
        'not_reserved','committed','refunded'
    ));

ALTER TABLE multiple_ask_job_items
    ADD COLUMN display_label TEXT NULL,
    ADD COLUMN section_context TEXT NULL,
    ADD COLUMN correction_answer_mode TEXT NULL
        CHECK (correction_answer_mode IS NULL OR correction_answer_mode IN ('short','long','mcq')),
    ADD COLUMN correction_mcq_options JSONB NULL
        CHECK (correction_mcq_options IS NULL OR jsonb_typeof(correction_mcq_options)='array');

CREATE INDEX idx_multiple_ask_job_items_display_order
    ON multiple_ask_job_items(multiple_ask_job_id,item_index);

REVOKE ALL ON multiple_ask_upload_sessions, multiple_ask_text_inputs,
    multiple_ask_jobs, multiple_ask_job_items, multiple_ask_normalized_sources,
    multiple_ask_cleanup_audit
FROM PUBLIC, anon, authenticated;

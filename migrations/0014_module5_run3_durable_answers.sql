-- Module 5 Run 3: durable, per-question answer provenance.
-- Forward-only.  Multiple Ask remains temporary job state; generated candidates
-- continue to use the Module 4 ai_requests/ai_answers review foundation.

SET search_path = public, pg_catalog;

ALTER TABLE ai_requests
    DROP CONSTRAINT ai_requests_source_feature_valid,
    ADD CONSTRAINT ai_requests_source_feature_valid
        CHECK (source_feature IN ('single_question', 'multiple_question', 'multiple_ask'));

ALTER TABLE multiple_ask_jobs
    ADD COLUMN answer_epoch INTEGER NOT NULL DEFAULT 0 CHECK (answer_epoch >= 0);

ALTER TABLE multiple_ask_job_items
    ADD COLUMN approved_revision_id UUID NULL
        REFERENCES question_bank_revisions(id) ON DELETE SET NULL,
    ADD COLUMN answer_source TEXT NULL
        CHECK (answer_source IS NULL OR answer_source IN (
            'approved_bank', 'syllabus_grounded', 'general_knowledge'
        )),
    ADD COLUMN terminal_error_code TEXT NULL;

CREATE INDEX idx_multiple_ask_items_pending_answers
    ON multiple_ask_job_items (multiple_ask_job_id, item_index)
    WHERE item_status IN ('ready_to_answer', 'answering');
CREATE INDEX idx_multiple_ask_items_approved_revision
    ON multiple_ask_job_items (approved_revision_id)
    WHERE approved_revision_id IS NOT NULL;

REVOKE ALL ON multiple_ask_jobs, multiple_ask_job_items
FROM PUBLIC, anon, authenticated;

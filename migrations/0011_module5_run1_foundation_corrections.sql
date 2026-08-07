-- Module 5 Run 1 corrections. Forward-only upgrade for databases that already
-- recorded 0010; fresh databases apply 0010 then this migration.

SET search_path = public, pg_catalog;

ALTER TABLE multiple_ask_upload_sessions
    RENAME COLUMN expires_at TO upload_capability_expires_at;
ALTER TABLE multiple_ask_upload_sessions
    ADD COLUMN board_id TEXT NOT NULL DEFAULT 'legacy_unscoped',
    ADD COLUMN class_id TEXT NOT NULL DEFAULT 'legacy_unscoped',
    ADD COLUMN subject_id TEXT NOT NULL DEFAULT 'legacy_unscoped',
    ADD COLUMN chapter_id TEXT NULL,
    ADD COLUMN raw_source_expires_at TIMESTAMPTZ NULL,
    ADD COLUMN raw_source_purged_at TIMESTAMPTZ NULL,
    ADD COLUMN cleanup_claimed_at TIMESTAMPTZ NULL;
-- The temporary legacy marker exists only to backfill already-recorded 0010
-- rows. New sessions must supply an explicit immutable scope.
ALTER TABLE multiple_ask_upload_sessions
    ALTER COLUMN board_id DROP DEFAULT,
    ALTER COLUMN class_id DROP DEFAULT,
    ALTER COLUMN subject_id DROP DEFAULT;

-- Rebuild generated CHECK names explicitly so the upgraded and fresh schemas
-- have identical rules. Existing rows are preserved as legacy only.
DO $$
DECLARE constraint_name TEXT;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid='public.multiple_ask_upload_sessions'::regclass
          AND contype='c'
    LOOP
        EXECUTE format('ALTER TABLE multiple_ask_upload_sessions DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;
ALTER TABLE multiple_ask_upload_sessions
    ADD CONSTRAINT multiple_ask_sessions_uid_hash_check CHECK (char_length(uid_hash)=64),
    ADD CONSTRAINT multiple_ask_sessions_tier_check CHECK (account_tier IN ('anonymous','google','premium')),
    ADD CONSTRAINT multiple_ask_sessions_kind_check CHECK (input_kind IN ('image','pdf','text')),
    ADD CONSTRAINT multiple_ask_sessions_status_check CHECK (status IN ('created','uploaded','finalized','expired','cancelled','raw_source_purged')),
    ADD CONSTRAINT multiple_ask_sessions_scope_check CHECK (
        board_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
        AND class_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
        AND subject_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
        AND (chapter_id IS NULL OR chapter_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$')
    ),
    ADD CONSTRAINT multiple_ask_sessions_size_check CHECK (expected_size_bytes IS NULL OR expected_size_bytes>0),
    ADD CONSTRAINT multiple_ask_sessions_input_check CHECK (
        (input_kind='text' AND expected_content_type IS NULL AND expected_size_bytes IS NULL
         AND storage_bucket IS NULL AND storage_object_key IS NULL)
        OR (input_kind='image' AND expected_content_type IN ('image/jpeg','image/png','image/webp')
            AND expected_size_bytes IS NOT NULL
            AND (status='raw_source_purged' OR (storage_bucket IS NOT NULL AND storage_object_key IS NOT NULL)))
        OR (input_kind='pdf' AND expected_content_type='application/pdf'
            AND expected_size_bytes IS NOT NULL
            AND (status='raw_source_purged' OR (storage_bucket IS NOT NULL AND storage_object_key IS NOT NULL)))
    ),
    ADD CONSTRAINT multiple_ask_sessions_finalized_check CHECK (
        (status IN ('finalized','raw_source_purged')) = (finalized_at IS NOT NULL)
    );
CREATE INDEX idx_multiple_ask_raw_source_expiry
    ON multiple_ask_upload_sessions(raw_source_expires_at,id)
    WHERE raw_source_expires_at IS NOT NULL AND raw_source_purged_at IS NULL;
CREATE INDEX idx_multiple_ask_upload_capability_expiry
    ON multiple_ask_upload_sessions(upload_capability_expires_at,id)
    WHERE status IN ('created','uploaded');

ALTER TABLE multiple_ask_text_inputs
    DROP CONSTRAINT IF EXISTS multiple_ask_text_inputs_session_id_fkey,
    ADD CONSTRAINT multiple_ask_text_inputs_session_id_fkey
        FOREIGN KEY(session_id) REFERENCES multiple_ask_upload_sessions(id) ON DELETE CASCADE;
DO $$
DECLARE constraint_name TEXT;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid='public.multiple_ask_text_inputs'::regclass AND contype='c'
    LOOP
        EXECUTE format('ALTER TABLE multiple_ask_text_inputs DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;
ALTER TABLE multiple_ask_text_inputs
    ADD CONSTRAINT multiple_ask_text_input_length_check CHECK (char_length(input_text) BETWEEN 1 AND 30000);

ALTER TABLE multiple_ask_jobs
    ADD COLUMN board_id TEXT NOT NULL DEFAULT 'legacy_unscoped',
    ADD COLUMN class_id TEXT NOT NULL DEFAULT 'legacy_unscoped',
    ADD COLUMN subject_id TEXT NOT NULL DEFAULT 'legacy_unscoped',
    ADD COLUMN chapter_id TEXT NULL,
    ADD COLUMN quota_status TEXT NOT NULL DEFAULT 'not_reserved',
    ADD COLUMN terminal_at TIMESTAMPTZ NULL,
    ADD COLUMN cleanup_claimed_at TIMESTAMPTZ NULL;
-- As above, require all newly-created durable jobs to copy session scope.
ALTER TABLE multiple_ask_jobs
    ALTER COLUMN board_id DROP DEFAULT,
    ALTER COLUMN class_id DROP DEFAULT,
    ALTER COLUMN subject_id DROP DEFAULT;
UPDATE multiple_ask_jobs SET workflow_status='queued' WHERE workflow_status='validation_queued';
UPDATE multiple_ask_jobs SET retention_expires_at=NULL
WHERE workflow_status NOT IN ('invalid','cancelled');
UPDATE multiple_ask_jobs SET terminal_at=created_at
WHERE workflow_status IN ('invalid','cancelled') AND terminal_at IS NULL;
ALTER TABLE multiple_ask_jobs ALTER COLUMN retention_expires_at DROP NOT NULL;
ALTER TABLE multiple_ask_jobs ALTER COLUMN queue_job_id DROP NOT NULL;
ALTER TABLE multiple_ask_jobs
    DROP CONSTRAINT IF EXISTS multiple_ask_jobs_upload_session_id_fkey,
    ADD CONSTRAINT multiple_ask_jobs_upload_session_id_fkey
        FOREIGN KEY(upload_session_id) REFERENCES multiple_ask_upload_sessions(id) ON DELETE CASCADE,
    DROP CONSTRAINT IF EXISTS multiple_ask_jobs_queue_job_id_fkey,
    ADD CONSTRAINT multiple_ask_jobs_queue_job_id_fkey
        FOREIGN KEY(queue_job_id) REFERENCES job_queue(id) ON DELETE SET NULL;
DO $$
DECLARE constraint_name TEXT;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid='public.multiple_ask_jobs'::regclass AND contype='c'
    LOOP
        EXECUTE format('ALTER TABLE multiple_ask_jobs DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;
ALTER TABLE multiple_ask_jobs
    ADD CONSTRAINT multiple_ask_jobs_uid_hash_check CHECK (char_length(uid_hash)=64),
    ADD CONSTRAINT multiple_ask_jobs_tier_check CHECK (account_tier IN ('anonymous','google','premium')),
    ADD CONSTRAINT multiple_ask_jobs_kind_check CHECK (input_kind IN ('image','pdf','text')),
    ADD CONSTRAINT multiple_ask_jobs_scope_check CHECK (
        board_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
        AND class_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
        AND subject_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
        AND (chapter_id IS NULL OR chapter_id ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$')
    ),
    ADD CONSTRAINT multiple_ask_jobs_workflow_check CHECK (workflow_status IN (
        'queued','validating','validated','extracting','needs_correction','answering',
        'partially_completed','completed','failed','cancelled','invalid','limit_reached'
    )),
    ADD CONSTRAINT multiple_ask_jobs_quota_check CHECK (quota_status IN ('not_reserved','committed')),
    ADD CONSTRAINT multiple_ask_jobs_terminal_retention_check CHECK (
        (terminal_at IS NULL AND retention_expires_at IS NULL)
        OR (terminal_at IS NOT NULL AND retention_expires_at IS NOT NULL)
    );
DROP INDEX IF EXISTS idx_multiple_ask_jobs_retention;
CREATE INDEX idx_multiple_ask_jobs_retention
    ON multiple_ask_jobs(retention_expires_at,id)
    WHERE retention_expires_at IS NOT NULL;

ALTER TABLE multiple_ask_job_items
    ADD COLUMN source_text TEXT NULL,
    ADD COLUMN source_locator JSONB NULL,
    ADD COLUMN extracted_text TEXT NULL,
    ADD COLUMN extraction_version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN mcq_options JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN unclear_reason TEXT NULL,
    ADD COLUMN correction_version INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN corrected_at TIMESTAMPTZ NULL,
    ADD COLUMN ai_answer_id UUID NULL UNIQUE;
ALTER TABLE multiple_ask_job_items
    DROP CONSTRAINT IF EXISTS multiple_ask_job_items_multiple_ask_job_id_fkey,
    ADD CONSTRAINT multiple_ask_job_items_multiple_ask_job_id_fkey
        FOREIGN KEY(multiple_ask_job_id) REFERENCES multiple_ask_jobs(id) ON DELETE CASCADE,
    DROP CONSTRAINT IF EXISTS multiple_ask_job_items_ai_request_id_fkey,
    ADD CONSTRAINT multiple_ask_job_items_ai_request_id_fkey
        FOREIGN KEY(ai_request_id) REFERENCES ai_requests(id) ON DELETE SET NULL,
    ADD CONSTRAINT multiple_ask_job_items_ai_answer_id_fkey
        FOREIGN KEY(ai_answer_id) REFERENCES ai_answers(id) ON DELETE SET NULL;
DO $$
DECLARE constraint_name TEXT;
BEGIN
    FOR constraint_name IN
        SELECT conname FROM pg_constraint
        WHERE conrelid='public.multiple_ask_job_items'::regclass AND contype='c'
    LOOP
        EXECUTE format('ALTER TABLE multiple_ask_job_items DROP CONSTRAINT %I', constraint_name);
    END LOOP;
END $$;
ALTER TABLE multiple_ask_job_items
    ADD CONSTRAINT multiple_ask_items_index_check CHECK (item_index>=0),
    ADD CONSTRAINT multiple_ask_items_hash_check CHECK (question_hash IS NULL OR char_length(question_hash)=64),
    ADD CONSTRAINT multiple_ask_items_mode_check CHECK (answer_mode IS NULL OR answer_mode IN ('short','long','mcq','not_clear')),
    ADD CONSTRAINT multiple_ask_items_status_check CHECK (item_status IN ('pending_extraction','needs_correction','ready_to_answer','answering','answered','failed','cancelled')),
    ADD CONSTRAINT multiple_ask_items_extraction_version_check CHECK (extraction_version>0),
    ADD CONSTRAINT multiple_ask_items_correction_version_check CHECK (correction_version>=0),
    ADD CONSTRAINT multiple_ask_items_options_check CHECK (
        jsonb_typeof(mcq_options)='array'
        AND (answer_mode='mcq' OR jsonb_array_length(mcq_options)=0)
    ),
    ADD CONSTRAINT multiple_ask_items_unclear_check CHECK (
        (answer_mode='not_clear' AND unclear_reason IS NOT NULL AND btrim(unclear_reason)<>'')
        OR (answer_mode IS DISTINCT FROM 'not_clear' AND unclear_reason IS NULL)
    ),
    ADD CONSTRAINT multiple_ask_items_mode_status_check CHECK (
        (answer_mode='not_clear' AND item_status IN ('needs_correction','cancelled'))
        OR (answer_mode IS NULL AND item_status='pending_extraction')
        OR answer_mode IN ('short','long','mcq')
    );

ALTER TABLE multiple_ask_cleanup_audit
    ALTER COLUMN session_id DROP NOT NULL,
    DROP CONSTRAINT IF EXISTS multiple_ask_cleanup_audit_session_id_fkey,
    ADD CONSTRAINT multiple_ask_cleanup_audit_session_id_fkey
        FOREIGN KEY(session_id) REFERENCES multiple_ask_upload_sessions(id) ON DELETE SET NULL,
    ADD COLUMN subject_kind TEXT NOT NULL DEFAULT 'raw_source';

REVOKE ALL ON multiple_ask_upload_sessions, multiple_ask_text_inputs,
    multiple_ask_jobs, multiple_ask_job_items, multiple_ask_cleanup_audit
FROM PUBLIC, anon, authenticated;

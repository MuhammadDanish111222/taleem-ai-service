-- Module 5: Allow NULL upload_capability_expires_at for text submissions.
-- Text submissions have no upload capability / signed URL, so the column
-- must accept NULL. The original column (expires_at NOT NULL in 0010, renamed
-- in 0011) carried a NOT NULL constraint that only makes sense for file
-- uploads. The existing input_check constraint already ensures file rows
-- have the required upload fields populated.

SET search_path = public, pg_catalog;

ALTER TABLE multiple_ask_upload_sessions
    ALTER COLUMN upload_capability_expires_at DROP NOT NULL;

-- upload_url_expires_at has the same issue (text sessions have no upload URL).
ALTER TABLE multiple_ask_upload_sessions
    ALTER COLUMN upload_url_expires_at DROP NOT NULL;

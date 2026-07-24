-- Phase 3C follow-up: enforce one normalized expected-question row per chunk.
-- Existing question text is retained; this column is an index key, not source content in audit logs.
ALTER TABLE chunk_expected_questions ADD COLUMN IF NOT EXISTS question_normalized TEXT;
UPDATE chunk_expected_questions
SET question_normalized = lower(regexp_replace(btrim(question_text), '\s+', ' ', 'g'))
WHERE question_normalized IS NULL;
ALTER TABLE chunk_expected_questions ALTER COLUMN question_normalized SET NOT NULL;
ALTER TABLE chunk_expected_questions
    ADD CONSTRAINT chunk_expected_questions_question_text_nonblank CHECK (btrim(question_text) <> '');
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunk_expected_questions_unique_normalized
    ON chunk_expected_questions (chunk_id, question_normalized);

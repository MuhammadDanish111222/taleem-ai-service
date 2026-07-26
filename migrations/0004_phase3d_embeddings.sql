-- Phase 3D: immutable embedding configuration, per-row provenance, and readiness gate.
-- This is forward-only: existing vectors are retained but marked unverified until re-embedded.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE rag_corpus_versions
    ADD COLUMN IF NOT EXISTS embedding_config_fingerprint TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_input_fingerprint TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS expected_question_count INT NOT NULL DEFAULT 0 CHECK (expected_question_count >= 0),
    ADD COLUMN IF NOT EXISTS embedded_question_count INT NOT NULL DEFAULT 0 CHECK (embedded_question_count >= 0);

ALTER TABLE rag_chunks
    ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_revision TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_config_fingerprint TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_input_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'embedded', 'failed')),
    ADD COLUMN IF NOT EXISTS embedding_started_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS embedding_completed_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS embedding_error_code TEXT NULL;

ALTER TABLE chunk_expected_questions
    ADD COLUMN IF NOT EXISTS question_hash TEXT,
    ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_revision TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_config_fingerprint TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_input_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS embedding_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'embedded', 'failed')),
    ADD COLUMN IF NOT EXISTS embedding_started_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS embedding_completed_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS embedding_error_code TEXT NULL;

UPDATE chunk_expected_questions
SET question_hash = encode(digest(question_normalized, 'sha256'), 'hex')
WHERE question_hash IS NULL;

ALTER TABLE chunk_expected_questions ALTER COLUMN question_hash SET NOT NULL;
ALTER TABLE chunk_expected_questions
    ADD CONSTRAINT chunk_expected_questions_question_hash_sha256
    CHECK (char_length(question_hash) = 64);

-- Existing non-null vectors lack a model/revision/fingerprint provenance record.  They
-- deliberately remain unverified (pending) and therefore cannot make a corpus ready.
UPDATE rag_chunks
SET embedding_status = 'pending'
WHERE embedding_config_fingerprint = '' OR embedding_model = '' OR embedding_revision = '';
UPDATE chunk_expected_questions
SET embedding_status = 'pending'
WHERE embedding_config_fingerprint = '' OR embedding_model = '' OR embedding_revision = '';

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_work
    ON rag_chunks (corpus_version_id, embedding_status);
CREATE INDEX IF NOT EXISTS idx_chunk_expected_questions_embedding_work
    ON chunk_expected_questions (embedding_status, chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_expected_questions_question_hash
    ON chunk_expected_questions (question_hash);
CREATE INDEX IF NOT EXISTS idx_job_queue_embedding_corpus
    ON job_queue ((payload->>'corpus_version_id'), status)
    WHERE job_type IN ('embed_chunks', 'embed_questions', 'corpus_completeness');

CREATE OR REPLACE FUNCTION phase3d_require_corpus_complete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    actual_chunks INT;
    valid_chunks INT;
    actual_questions INT;
    valid_questions INT;
    blocking_jobs INT;
BEGIN
    IF NEW.status NOT IN ('qa_ready', 'active') THEN
        RETURN NEW;
    END IF;

    IF NEW.status = 'active' AND (
        TG_OP = 'INSERT' OR OLD.status NOT IN ('qa_ready', 'superseded')
    ) THEN
        RAISE EXCEPTION 'P3D_CORPUS_INCOMPLETE'
            USING ERRCODE = 'P0001';
    END IF;

    SELECT COUNT(*), COUNT(*) FILTER (
        WHERE embedding IS NOT NULL
          AND vector_dims(embedding) = NEW.embedding_dim
          AND embedding_status = 'embedded'
          AND embedding_model = NEW.embedding_model
          AND embedding_revision = NEW.embedding_revision
          AND embedding_config_fingerprint = NEW.embedding_config_fingerprint
    )
    INTO actual_chunks, valid_chunks
    FROM rag_chunks WHERE corpus_version_id = NEW.id;

    SELECT COUNT(*), COUNT(*) FILTER (
        WHERE q.embedding IS NOT NULL
          AND vector_dims(q.embedding) = NEW.embedding_dim
          AND q.embedding_status = 'embedded'
          AND q.embedding_model = NEW.embedding_model
          AND q.embedding_revision = NEW.embedding_revision
          AND q.embedding_config_fingerprint = NEW.embedding_config_fingerprint
    )
    INTO actual_questions, valid_questions
    FROM chunk_expected_questions q
    JOIN rag_chunks c ON c.id = q.chunk_id
    WHERE c.corpus_version_id = NEW.id;

    SELECT COUNT(*) INTO blocking_jobs
    FROM job_queue
    WHERE job_type IN ('embed_chunks', 'embed_questions')
      AND payload->>'corpus_version_id' = NEW.id::text
      AND payload->>'embedding_config_fingerprint' = NEW.embedding_config_fingerprint
      AND payload->>'embedding_input_fingerprint' = NEW.embedding_input_fingerprint
      AND status IN ('queued', 'leased', 'running', 'retry_wait', 'failed');

    IF NEW.embedding_dim <> 768
       OR NEW.embedding_config_fingerprint = ''
       OR NEW.expected_chunk_count <> actual_chunks
       OR NEW.embedded_chunk_count <> valid_chunks
       OR actual_chunks <> valid_chunks
       OR NEW.expected_question_count <> actual_questions
       OR NEW.embedded_question_count <> valid_questions
       OR actual_questions <> valid_questions
       OR blocking_jobs <> 0 THEN
        RAISE EXCEPTION 'P3D_CORPUS_INCOMPLETE'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION phase3d_require_corpus_complete() FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS trg_phase3d_require_corpus_complete ON rag_corpus_versions;
CREATE TRIGGER trg_phase3d_require_corpus_complete
BEFORE INSERT OR UPDATE OF status ON rag_corpus_versions
FOR EACH ROW EXECUTE FUNCTION phase3d_require_corpus_complete();

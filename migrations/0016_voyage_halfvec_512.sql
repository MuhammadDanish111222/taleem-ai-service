-- Migration 0016: Voyage Embeddings, halfvec(512), and Clean Provenance Preservation
-- 1. Invalidate QA approvals scoped strictly to active corpus versions being superseded
UPDATE rag_corpus_qa_approvals qa
SET invalidated_at = NOW(),
    invalidated_reason = 'MIGRATION_0016_VOYAGE_512'
WHERE qa.invalidated_at IS NULL
  AND qa.corpus_version_id IN (
      SELECT id
      FROM rag_corpus_versions
      WHERE status = 'active'
  );

-- 2. Transition active corpus versions to superseded state (preserving historical BGE metadata and counts)
UPDATE rag_corpus_versions
SET status = 'superseded'
WHERE status = 'active';

-- 3. Reset legacy BGE vector values to NULL and status to pending before column alteration
UPDATE rag_chunks
SET embedding = NULL,
    embedding_status = 'pending',
    embedding_model = '',
    embedding_revision = '',
    embedding_config_fingerprint = '',
    embedding_input_hash = '';

UPDATE chunk_expected_questions
SET embedding = NULL,
    embedding_status = 'pending',
    embedding_model = '',
    embedding_revision = '',
    embedding_config_fingerprint = '',
    embedding_input_hash = '';

UPDATE question_bank_revisions
SET embedding = NULL,
    embedding_status = 'pending',
    embedding_model = '',
    embedding_revision = '',
    embedding_config_fingerprint = '';

UPDATE question_bank_variations
SET embedding = NULL,
    embedding_status = 'pending',
    embedding_model = '',
    embedding_revision = '',
    embedding_config_fingerprint = '';

-- 4. Alter vector columns safely to halfvec(512)
ALTER TABLE rag_chunks
    ALTER COLUMN embedding TYPE halfvec(512) USING NULL;

ALTER TABLE chunk_expected_questions
    ALTER COLUMN embedding TYPE halfvec(512) USING NULL;

ALTER TABLE question_bank_revisions
    ALTER COLUMN embedding TYPE halfvec(512) USING NULL;

ALTER TABLE question_bank_variations
    ALTER COLUMN embedding TYPE halfvec(512) USING NULL;

-- 5. Update default embedding dimension to 512 for future corpus versions
ALTER TABLE rag_corpus_versions
    ALTER COLUMN embedding_dim SET DEFAULT 512;

-- 6. Update corpus completeness trigger function to enforce 512 dimensions and Voyage provenance
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

    IF NEW.embedding_dim <> 512
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

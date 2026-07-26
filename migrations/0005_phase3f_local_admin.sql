-- Phase 3F: local-admin visual metadata, QA approval, and editable drafts.
-- This migration intentionally retains the legacy rag_visuals columns: they may
-- contain historical provider data and are not a browser-facing contract.

ALTER TABLE rag_corpus_versions
    ADD COLUMN IF NOT EXISTS source_corpus_version_id UUID NULL
        REFERENCES rag_corpus_versions(id) ON DELETE SET NULL;

ALTER TABLE rag_visuals
    ADD COLUMN IF NOT EXISTS visual_id TEXT,
    ADD COLUMN IF NOT EXISTS title TEXT,
    ADD COLUMN IF NOT EXISTS description TEXT,
    ADD COLUMN IF NOT EXISTS storage_provider TEXT,
    ADD COLUMN IF NOT EXISTS storage_key TEXT,
    ADD COLUMN IF NOT EXISTS display_policy TEXT,
    ADD COLUMN IF NOT EXISTS review_status TEXT,
    ADD COLUMN IF NOT EXISTS visual_text_hash TEXT,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- Legacy values are retained only as non-displayable records.  The description
-- truthfully says that the prior schema did not preserve usable metadata.
UPDATE rag_visuals
SET visual_id = COALESCE(NULLIF(btrim(visual_id), ''), 'legacy-' || id::text),
    title = COALESCE(NULLIF(btrim(title), ''), NULLIF(btrim(caption), ''), 'Legacy visual'),
    description = COALESCE(NULLIF(btrim(description), ''), 'Legacy metadata unavailable'),
    storage_provider = COALESCE(NULLIF(btrim(storage_provider), ''), 'google_drive'),
    storage_key = COALESCE(NULLIF(btrim(storage_key), ''), NULLIF(btrim(storage_path), ''), 'legacy-unavailable-' || id::text),
    display_policy = COALESCE(display_policy, 'never'),
    review_status = COALESCE(review_status, 'rejected'),
    visual_text_hash = COALESCE(NULLIF(btrim(visual_text_hash), ''), encode(digest(lower(regexp_replace(btrim(COALESCE(title, caption, 'Legacy visual')) || ' ' || btrim(COALESCE(description, 'Legacy metadata unavailable')), '\\s+', ' ', 'g')), 'sha256'), 'hex')),
    updated_at = COALESCE(updated_at, created_at, NOW());

ALTER TABLE rag_visuals
    ALTER COLUMN visual_id SET NOT NULL,
    ALTER COLUMN title SET NOT NULL,
    ALTER COLUMN description SET NOT NULL,
    ALTER COLUMN storage_provider SET NOT NULL,
    ALTER COLUMN storage_key SET NOT NULL,
    ALTER COLUMN display_policy SET NOT NULL,
    ALTER COLUMN review_status SET NOT NULL,
    ALTER COLUMN visual_text_hash SET NOT NULL,
    ALTER COLUMN updated_at SET DEFAULT NOW(),
    ALTER COLUMN updated_at SET NOT NULL;

ALTER TABLE rag_visuals
    ADD CONSTRAINT rag_visuals_visual_id_nonblank CHECK (btrim(visual_id) <> ''),
    ADD CONSTRAINT rag_visuals_title_nonblank CHECK (btrim(title) <> '' AND char_length(title) <= 240),
    ADD CONSTRAINT rag_visuals_description_nonblank CHECK (btrim(description) <> '' AND char_length(description) <= 4000),
    ADD CONSTRAINT rag_visuals_storage_provider_supported CHECK (storage_provider IN ('google_drive')),
    ADD CONSTRAINT rag_visuals_storage_key_nonblank CHECK (btrim(storage_key) <> ''),
    ADD CONSTRAINT rag_visuals_display_policy_valid CHECK (display_policy IN ('always', 'llm_decide', 'never')),
    ADD CONSTRAINT rag_visuals_review_status_valid CHECK (review_status IN ('pending', 'approved', 'rejected')),
    ADD CONSTRAINT rag_visuals_text_hash_sha256 CHECK (char_length(visual_text_hash) = 64);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_visuals_chunk_visual_id
    ON rag_visuals (chunk_id, visual_id);
CREATE INDEX IF NOT EXISTS idx_rag_visuals_embedding_input
    ON rag_visuals (chunk_id, review_status, visual_id);
CREATE INDEX IF NOT EXISTS idx_rag_corpus_versions_source
    ON rag_corpus_versions (source_corpus_version_id);

CREATE OR REPLACE FUNCTION phase3f_set_visual_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;
REVOKE ALL ON FUNCTION phase3f_set_visual_updated_at() FROM PUBLIC, anon, authenticated;
DROP TRIGGER IF EXISTS trg_phase3f_visual_updated_at ON rag_visuals;
CREATE TRIGGER trg_phase3f_visual_updated_at
BEFORE UPDATE ON rag_visuals FOR EACH ROW EXECUTE FUNCTION phase3f_set_visual_updated_at();

CREATE TABLE IF NOT EXISTS rag_corpus_qa_approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_version_id UUID NOT NULL REFERENCES rag_corpus_versions(id) ON DELETE CASCADE,
    reviewer_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    approved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    invalidated_at TIMESTAMPTZ NULL,
    invalidated_reason TEXT NULL
);
ALTER TABLE rag_corpus_qa_approvals ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON rag_corpus_qa_approvals FROM PUBLIC, anon, authenticated;
CREATE INDEX IF NOT EXISTS idx_rag_corpus_qa_approvals_current
    ON rag_corpus_qa_approvals (corpus_version_id, approved_at DESC)
    WHERE invalidated_at IS NULL;

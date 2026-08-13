-- Stage 3: conservative approved-answer semantic reuse is enabled by default.
-- The persisted value remains cosine distance; 0.18 is exposed to admins as
-- a minimum similarity of 0.82.

SET search_path = public, pg_catalog;

UPDATE ask_source_policies
SET semantic_reuse_enabled=TRUE,
    semantic_distance_threshold=0.18,
    updated_by='stage3-default',
    updated_at=NOW()
WHERE class_id IS NULL AND subject_id IS NULL
  AND semantic_reuse_enabled=FALSE
  AND semantic_distance_threshold IS NULL;

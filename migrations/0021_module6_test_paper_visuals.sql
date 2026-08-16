-- A paper visual is only available when it is an approved question visual on
-- the current approved revision in the requested catalogue scope.  The
-- storage reference remains server-to-server: the Edge function returns it
-- only to the authenticated Next.js BFF, which streams the image bytes.
CREATE OR REPLACE FUNCTION taleem_test_paper_visual_reference(
    p_question_id UUID,
    p_visual_id TEXT,
    p_board_id TEXT,
    p_class_id TEXT,
    p_subject_id TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_catalog
AS $$
DECLARE
    reference JSONB;
BEGIN
    SELECT jsonb_build_object(
        'storage_provider', v.storage_provider,
        'storage_key', v.storage_key
    )
    INTO reference
    FROM question_bank_revisions r
    JOIN question_bank_revision_visuals link
      ON link.revision_id = r.id
     AND link.role = 'question'
    JOIN rag_visuals v ON v.id = link.visual_id
    WHERE r.question_id = p_question_id
      AND r.board_id = p_board_id
      AND r.class_id = p_class_id
      AND r.subject_id = p_subject_id
      AND r.review_status = 'approved'
      AND r.superseded_at IS NULL
      AND v.visual_id = p_visual_id
      AND v.review_status = 'approved'
      AND v.display_policy <> 'never'
      AND v.storage_provider = 'google_drive'
      AND v.storage_key IS NOT NULL
      AND btrim(v.storage_key) <> ''
    LIMIT 1;

    RETURN reference;
END;
$$;

REVOKE ALL ON FUNCTION taleem_test_paper_visual_reference(UUID, TEXT, TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION taleem_test_paper_visual_reference(UUID, TEXT, TEXT, TEXT, TEXT)
    TO service_role;

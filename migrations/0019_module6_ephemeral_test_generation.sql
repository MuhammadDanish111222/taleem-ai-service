-- Module 6 Stage 3.  This is a read-only, paper-safe wrapper around the
-- canonical Stage 2 selector; generated papers are intentionally never stored.
SET search_path = public, pg_catalog;

CREATE OR REPLACE FUNCTION taleem_generate_test_paper(
    p_mode TEXT,
    p_board_id TEXT,
    p_class_id TEXT,
    p_subject_id TEXT,
    p_spec JSONB,
    p_seed TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_catalog
AS $$
DECLARE
    selection_spec JSONB;
    selected JSONB;
    section JSONB;
    selected_question JSONB;
    result_sections JSONB := '[]'::jsonb;
    result_questions JSONB;
    question_data JSONB;
    visuals JSONB;
    options JSONB;
    total_marks NUMERIC;
BEGIN
    IF p_mode NOT IN ('board', 'custom')
       OR COALESCE(btrim(p_board_id), '') = ''
       OR COALESCE(btrim(p_class_id), '') = ''
       OR COALESCE(btrim(p_subject_id), '') = ''
       OR COALESCE(btrim(p_seed), '') = '' THEN
        RAISE EXCEPTION 'INVALID_REQUEST' USING ERRCODE = 'P0001';
    END IF;

    IF p_mode = 'board' THEN
        SELECT config INTO selection_spec
        FROM board_paper_blueprints
        WHERE board_id = p_board_id AND class_id = p_class_id AND subject_id = p_subject_id
          AND is_active = TRUE;
        IF selection_spec IS NULL THEN
            RAISE EXCEPTION 'NO_ACTIVE_BLUEPRINT' USING ERRCODE = 'P0001';
        END IF;
    ELSE
        IF jsonb_typeof(p_spec) <> 'object' THEN
            RAISE EXCEPTION 'INVALID_CUSTOM_SPEC' USING ERRCODE = 'P0001';
        END IF;
        BEGIN
            PERFORM taleem_validate_selection_spec(p_spec);
        EXCEPTION WHEN SQLSTATE 'P0001' THEN
            RAISE EXCEPTION 'INVALID_CUSTOM_SPEC' USING ERRCODE = 'P0001';
        END;
        selection_spec := p_spec;
    END IF;

    selected := taleem_select_questions(p_board_id, p_class_id, p_subject_id, selection_spec, p_seed);
    IF NOT COALESCE((selected->>'satisfiable')::BOOLEAN, FALSE) THEN
        RAISE EXCEPTION 'INSUFFICIENT_QUESTION_BANK' USING ERRCODE = 'P0001';
    END IF;

    FOR section IN SELECT value FROM jsonb_array_elements(selection_spec->'sections') AS t(value)
    LOOP
        result_questions := '[]'::jsonb;
        FOR selected_question IN
            SELECT value FROM jsonb_array_elements(selected->'selected') AS t(value)
            WHERE value->>'section_key' = section->>'key'
        LOOP
            SELECT COALESCE(jsonb_agg(jsonb_build_object(
                'visual_id', v.visual_id,
                'visual_type', v.visual_type,
                'title', v.title,
                'description', v.description
            ) ORDER BY link.display_order), '[]'::jsonb)
            INTO visuals
            FROM question_bank_revision_visuals link
            JOIN rag_visuals v ON v.id = link.visual_id
            WHERE link.revision_id = (selected_question->>'revision_id')::uuid
              AND v.review_status = 'approved' AND v.display_policy <> 'never';

            SELECT COALESCE(jsonb_agg(jsonb_build_object(
                'key', option_key, 'text', option_text
            ) ORDER BY display_order), '[]'::jsonb)
            INTO options
            FROM question_bank_mcq_options
            WHERE revision_id = (selected_question->>'revision_id')::uuid;

            SELECT jsonb_build_object(
                'id', r.question_id::text,
                'question', r.question_text,
                'marks', r.marks,
                'chapter_id', r.chapter_id,
                'difficulty', r.difficulty,
                'options', CASE WHEN r.answer_mode = 'mcq' THEN options ELSE '[]'::jsonb END,
                'visuals', visuals
            ) INTO question_data
            FROM question_bank_revisions r
            WHERE r.id = (selected_question->>'revision_id')::uuid
              AND r.review_status = 'approved' AND r.superseded_at IS NULL;
            IF question_data IS NULL THEN
                RAISE EXCEPTION 'INSUFFICIENT_QUESTION_BANK' USING ERRCODE = 'P0001';
            END IF;
            result_questions := result_questions || jsonb_build_array(question_data);
        END LOOP;
        result_sections := result_sections || jsonb_build_array(jsonb_build_object(
            'key', section->>'key',
            'title', section->>'title',
            'type', section->>'type',
            'select_count', (section->>'select_count')::integer,
            'attempt_count', (section->>'attempt_count')::integer,
            'marks_each', (section->>'marks_each')::numeric,
            'questions', result_questions
        ));
    END LOOP;

    total_marks := (selected->>'total_marks')::numeric;
    RETURN jsonb_build_object(
        'mode', p_mode,
        'board_id', p_board_id,
        'class_id', p_class_id,
        'subject_id', p_subject_id,
        'duration_minutes', (selection_spec->>'duration_minutes')::integer,
        'total_marks', total_marks,
        'seed', p_seed,
        'sections', result_sections
    );
END;
$$;

REVOKE ALL ON FUNCTION taleem_generate_test_paper(TEXT, TEXT, TEXT, TEXT, JSONB, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION taleem_generate_test_paper(TEXT, TEXT, TEXT, TEXT, JSONB, TEXT)
    TO service_role;

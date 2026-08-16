-- Keep the recursive candidate position stable while the current selection
-- grows.  Migration 0017 removed chosen rows before assigning an ordinal,
-- which made p_start skip otherwise valid candidates on the next recursion.
SET search_path = public, pg_catalog;

CREATE OR REPLACE FUNCTION taleem_allocate_selection_recursive(
    p_board_id TEXT,
    p_class_id TEXT,
    p_subject_id TEXT,
    p_sections JSONB,
    p_seed TEXT,
    p_section_index INTEGER,
    p_start INTEGER,
    p_current_question_ids UUID[],
    p_used_question_ids UUID[],
    p_result JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    section JSONB;
    candidate RECORD;
    candidate_ordinal INTEGER := 0;
    required_count INTEGER;
    selected_for_dimension INTEGER;
    quota INTEGER;
    current_rows JSONB;
    recursive_result JSONB;
BEGIN
    IF p_section_index > jsonb_array_length(p_sections) THEN
        RETURN p_result;
    END IF;
    section := p_sections -> (p_section_index - 1);
    required_count := (section->>'select_count')::INTEGER;

    IF cardinality(p_current_question_ids) = required_count THEN
        SELECT COALESCE(jsonb_agg(jsonb_build_object(
            'section_key', section->>'key',
            'revision_id', r.id::text,
            'question_id', r.question_id::text,
            'chapter_id', r.chapter_id,
            'difficulty', r.difficulty,
            'answer_mode', r.answer_mode,
            'marks', r.marks
        ) ORDER BY chosen.ordinality), '[]'::jsonb)
        INTO current_rows
        FROM unnest(p_current_question_ids) WITH ORDINALITY AS chosen(question_id, ordinality)
        JOIN question_bank_revisions r ON r.question_id = chosen.question_id
        WHERE r.review_status = 'approved' AND r.superseded_at IS NULL;

        RETURN taleem_allocate_selection_recursive(
            p_board_id, p_class_id, p_subject_id, p_sections, p_seed,
            p_section_index + 1, 1, ARRAY[]::UUID[],
            p_used_question_ids || p_current_question_ids,
            p_result || current_rows
        );
    END IF;

    -- Deliberately retain p_current rows here.  Their deterministic ordinal
    -- must remain stable at each recursion depth; they are skipped below.
    FOR candidate IN
        SELECT r.id, r.question_id, r.chapter_id, r.difficulty
        FROM question_bank_revisions r
        WHERE r.board_id = p_board_id AND r.class_id = p_class_id AND r.subject_id = p_subject_id
          AND r.answer_mode = section->>'type'
          AND r.marks = (section->>'marks_each')::NUMERIC
          AND r.review_status = 'approved' AND r.superseded_at IS NULL
          AND NOT (r.question_id = ANY(p_used_question_ids))
          AND (COALESCE(section->'chapter_distribution', '{}'::jsonb) = '{}'::jsonb
               OR (r.chapter_id IS NOT NULL AND (section->'chapter_distribution') ? r.chapter_id))
          AND (COALESCE(section->'difficulty_distribution', '{}'::jsonb) = '{}'::jsonb
               OR (section->'difficulty_distribution') ? r.difficulty)
        ORDER BY md5(p_seed || ':' || r.question_id::text), r.question_id, r.id
    LOOP
        candidate_ordinal := candidate_ordinal + 1;
        IF candidate_ordinal < p_start
           OR candidate.question_id = ANY(p_current_question_ids) THEN
            CONTINUE;
        END IF;
        IF COALESCE(section->'chapter_distribution', '{}'::jsonb) <> '{}'::jsonb THEN
            quota := ((section->'chapter_distribution')->>candidate.chapter_id)::INTEGER;
            SELECT count(*) INTO selected_for_dimension
            FROM question_bank_revisions r
            WHERE r.question_id = ANY(p_current_question_ids)
              AND r.chapter_id = candidate.chapter_id
              AND r.review_status = 'approved' AND r.superseded_at IS NULL;
            IF selected_for_dimension >= quota THEN
                CONTINUE;
            END IF;
        END IF;
        IF COALESCE(section->'difficulty_distribution', '{}'::jsonb) <> '{}'::jsonb THEN
            quota := ((section->'difficulty_distribution')->>candidate.difficulty)::INTEGER;
            SELECT count(*) INTO selected_for_dimension
            FROM question_bank_revisions r
            WHERE r.question_id = ANY(p_current_question_ids)
              AND r.difficulty = candidate.difficulty
              AND r.review_status = 'approved' AND r.superseded_at IS NULL;
            IF selected_for_dimension >= quota THEN
                CONTINUE;
            END IF;
        END IF;
        recursive_result := taleem_allocate_selection_recursive(
            p_board_id, p_class_id, p_subject_id, p_sections, p_seed,
            p_section_index, candidate_ordinal + 1,
            array_append(p_current_question_ids, candidate.question_id),
            p_used_question_ids, p_result
        );
        IF recursive_result IS NOT NULL THEN
            RETURN recursive_result;
        END IF;
    END LOOP;
    RETURN NULL;
END;
$$;

REVOKE EXECUTE ON FUNCTION taleem_allocate_selection_recursive(TEXT, TEXT, TEXT, JSONB, TEXT, INTEGER, INTEGER, UUID[], UUID[], JSONB) FROM PUBLIC, anon, authenticated;

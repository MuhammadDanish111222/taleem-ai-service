-- Module 6 Stage 2: lightweight board-paper blueprints and the canonical
-- deterministic Question Bank selector.  This deliberately stores no papers.
SET search_path = public, pg_catalog;

CREATE TABLE board_paper_blueprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    name TEXT NOT NULL CHECK (btrim(name) <> ''),
    config JSONB NOT NULL CHECK (jsonb_typeof(config) = 'object'),
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- One lightweight editable default record, rather than a version history.
    UNIQUE (board_id, class_id, subject_id)
);

CREATE INDEX idx_question_bank_paper_selection
    ON question_bank_revisions (
        board_id, class_id, subject_id, answer_mode, marks, chapter_id, difficulty
    )
    WHERE review_status = 'approved' AND superseded_at IS NULL;

CREATE OR REPLACE FUNCTION taleem_validate_selection_spec(p_spec JSONB)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    section JSONB;
    distribution_key TEXT;
    distribution_value TEXT;
    section_key TEXT;
    selected_count INTEGER;
    attempt_count INTEGER;
    marks_each NUMERIC;
    distribution_total INTEGER;
BEGIN
    IF jsonb_typeof(p_spec) <> 'object'
       OR jsonb_typeof(p_spec->'sections') <> 'array'
       OR jsonb_array_length(p_spec->'sections') = 0
       OR jsonb_array_length(p_spec->'sections') > 12 THEN
        RAISE EXCEPTION 'BLUEPRINT_SECTIONS_INVALID' USING ERRCODE = 'P0001';
    END IF;
    IF (p_spec->>'duration_minutes') IS NULL
       OR (p_spec->>'duration_minutes') !~ '^[0-9]+$'
       OR (p_spec->>'duration_minutes')::INTEGER NOT BETWEEN 1 AND 600 THEN
        RAISE EXCEPTION 'BLUEPRINT_DURATION_INVALID' USING ERRCODE = 'P0001';
    END IF;

    FOR section IN SELECT value FROM jsonb_array_elements(p_spec->'sections') AS t(value)
    LOOP
        section_key := section->>'key';
        IF jsonb_typeof(section) <> 'object'
           OR section_key IS NULL
           OR section_key !~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$'
           OR COALESCE(length(btrim(section->>'title')), 0) NOT BETWEEN 1 AND 160
           OR section->>'type' NOT IN ('mcq', 'short', 'long')
           OR COALESCE(section->>'select_count', '') !~ '^[0-9]+$'
           OR COALESCE(section->>'attempt_count', '') !~ '^[0-9]+$'
           OR COALESCE(section->>'marks_each', '') !~ '^[0-9]+([.][0-9]+)?$' THEN
            RAISE EXCEPTION 'BLUEPRINT_SECTION_INVALID' USING ERRCODE = 'P0001';
        END IF;
        selected_count := (section->>'select_count')::INTEGER;
        attempt_count := (section->>'attempt_count')::INTEGER;
        marks_each := (section->>'marks_each')::NUMERIC;
        IF selected_count NOT BETWEEN 1 AND 100
           OR attempt_count NOT BETWEEN 1 AND selected_count
           OR marks_each <= 0 OR marks_each > 1000 THEN
            RAISE EXCEPTION 'BLUEPRINT_SECTION_COUNTS_INVALID' USING ERRCODE = 'P0001';
        END IF;
        IF (SELECT count(*) FROM jsonb_array_elements(p_spec->'sections') AS x(value)
            WHERE x.value->>'key' = section_key) <> 1 THEN
            RAISE EXCEPTION 'BLUEPRINT_SECTION_KEY_DUPLICATE' USING ERRCODE = 'P0001';
        END IF;

        IF jsonb_typeof(COALESCE(section->'difficulty_distribution', '{}'::jsonb)) <> 'object' THEN
            RAISE EXCEPTION 'BLUEPRINT_DIFFICULTY_INVALID' USING ERRCODE = 'P0001';
        END IF;
        IF COALESCE(section->'difficulty_distribution', '{}'::jsonb) <> '{}'::jsonb THEN
            distribution_total := 0;
            FOR distribution_key, distribution_value IN
                SELECT key, value FROM jsonb_each_text(section->'difficulty_distribution')
            LOOP
                IF distribution_key NOT IN ('easy', 'medium', 'hard')
                   OR distribution_value !~ '^[1-9][0-9]*$' THEN
                    RAISE EXCEPTION 'BLUEPRINT_DIFFICULTY_INVALID' USING ERRCODE = 'P0001';
                END IF;
                distribution_total := distribution_total + distribution_value::INTEGER;
            END LOOP;
            IF distribution_total <> selected_count THEN
                RAISE EXCEPTION 'BLUEPRINT_DIFFICULTY_TOTAL_INVALID' USING ERRCODE = 'P0001';
            END IF;
        END IF;

        IF jsonb_typeof(COALESCE(section->'chapter_distribution', '{}'::jsonb)) <> 'object' THEN
            RAISE EXCEPTION 'BLUEPRINT_CHAPTER_INVALID' USING ERRCODE = 'P0001';
        END IF;
        IF COALESCE(section->'chapter_distribution', '{}'::jsonb) <> '{}'::jsonb THEN
            distribution_total := 0;
            FOR distribution_key, distribution_value IN
                SELECT key, value FROM jsonb_each_text(section->'chapter_distribution')
            LOOP
                IF distribution_key !~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$'
                   OR distribution_value !~ '^[1-9][0-9]*$' THEN
                    RAISE EXCEPTION 'BLUEPRINT_CHAPTER_INVALID' USING ERRCODE = 'P0001';
                END IF;
                distribution_total := distribution_total + distribution_value::INTEGER;
            END LOOP;
            IF distribution_total <> selected_count THEN
                RAISE EXCEPTION 'BLUEPRINT_CHAPTER_TOTAL_INVALID' USING ERRCODE = 'P0001';
            END IF;
        END IF;
    END LOOP;
END;
$$;

-- Depth-first allocation is intentionally in PostgreSQL so preview and later
-- Edge-function selection use exactly the same code.  Candidates are ranked
-- by stable seed+physical-question hashing; p_start removes permutation paths.
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
    next_result JSONB;
    current_rows JSONB;
    recursive_result JSONB;
BEGIN
    IF p_section_index > jsonb_array_length(p_sections) THEN
        RETURN p_result;
    END IF;
    section := p_sections -> (p_section_index - 1);
    required_count := (section->>'select_count')::INTEGER;

    IF cardinality(p_current_question_ids) = required_count THEN
        -- The spec validator has already proved quota totals equal required_count;
        -- the candidate guard below therefore makes these exact, not upper bounds.
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

    FOR candidate IN
        SELECT r.id, r.question_id, r.chapter_id, r.difficulty
        FROM question_bank_revisions r
        WHERE r.board_id = p_board_id AND r.class_id = p_class_id AND r.subject_id = p_subject_id
          AND r.answer_mode = section->>'type'
          AND r.marks = (section->>'marks_each')::NUMERIC
          AND r.review_status = 'approved' AND r.superseded_at IS NULL
          AND NOT (r.question_id = ANY(p_used_question_ids))
          AND NOT (r.question_id = ANY(p_current_question_ids))
          AND (COALESCE(section->'chapter_distribution', '{}'::jsonb) = '{}'::jsonb
               OR (r.chapter_id IS NOT NULL AND (section->'chapter_distribution') ? r.chapter_id))
          AND (COALESCE(section->'difficulty_distribution', '{}'::jsonb) = '{}'::jsonb
               OR (section->'difficulty_distribution') ? r.difficulty)
        ORDER BY md5(p_seed || ':' || r.question_id::text), r.question_id, r.id
    LOOP
        candidate_ordinal := candidate_ordinal + 1;
        IF candidate_ordinal < p_start THEN
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

CREATE OR REPLACE FUNCTION taleem_select_questions(
    p_board_id TEXT,
    p_class_id TEXT,
    p_subject_id TEXT,
    p_spec JSONB,
    p_seed TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    selected_rows JSONB;
    section JSONB;
    section_statuses JSONB := '[]'::jsonb;
    available_count INTEGER;
    selected_count INTEGER;
    section_only_rows JSONB;
    total_marks NUMERIC := 0;
BEGIN
    IF COALESCE(btrim(p_board_id), '') = '' OR COALESCE(btrim(p_class_id), '') = ''
       OR COALESCE(btrim(p_subject_id), '') = '' OR COALESCE(btrim(p_seed), '') = '' THEN
        RAISE EXCEPTION 'SELECTION_SCOPE_OR_SEED_REQUIRED' USING ERRCODE = 'P0001';
    END IF;
    PERFORM taleem_validate_selection_spec(p_spec);
    selected_rows := taleem_allocate_selection_recursive(
        p_board_id, p_class_id, p_subject_id, p_spec->'sections', p_seed,
        1, 1, ARRAY[]::UUID[], ARRAY[]::UUID[], '[]'::jsonb
    );

    FOR section IN SELECT value FROM jsonb_array_elements(p_spec->'sections') AS t(value)
    LOOP
        SELECT count(*) INTO available_count
        FROM question_bank_revisions r
        WHERE r.board_id = p_board_id AND r.class_id = p_class_id AND r.subject_id = p_subject_id
          AND r.answer_mode = section->>'type'
          AND r.marks = (section->>'marks_each')::NUMERIC
          AND r.review_status = 'approved' AND r.superseded_at IS NULL
          AND (COALESCE(section->'chapter_distribution', '{}'::jsonb) = '{}'::jsonb
               OR (r.chapter_id IS NOT NULL AND (section->'chapter_distribution') ? r.chapter_id))
          AND (COALESCE(section->'difficulty_distribution', '{}'::jsonb) = '{}'::jsonb
               OR (section->'difficulty_distribution') ? r.difficulty);
        IF selected_rows IS NULL THEN
            section_only_rows := taleem_allocate_selection_recursive(
                p_board_id, p_class_id, p_subject_id, jsonb_build_array(section), p_seed,
                1, 1, ARRAY[]::UUID[], ARRAY[]::UUID[], '[]'::jsonb
            );
            selected_count := CASE WHEN section_only_rows IS NULL THEN 0 ELSE jsonb_array_length(section_only_rows) END;
        ELSE
            selected_count := (
                SELECT count(*) FROM jsonb_array_elements(selected_rows) AS item
                WHERE item->>'section_key' = section->>'key'
            );
        END IF;
        total_marks := total_marks + ((section->>'attempt_count')::NUMERIC * (section->>'marks_each')::NUMERIC);
        section_statuses := section_statuses || jsonb_build_array(jsonb_build_object(
            'key', section->>'key',
            'required_count', (section->>'select_count')::INTEGER,
            'available_count', available_count,
            'selected_count', selected_count,
            'shortfall', GREATEST(0, (section->>'select_count')::INTEGER - selected_count),
            'satisfiable', selected_count = (section->>'select_count')::INTEGER
        ));
    END LOOP;

    RETURN jsonb_build_object(
        'satisfiable', selected_rows IS NOT NULL,
        'selected', COALESCE(selected_rows, '[]'::jsonb),
        'sections', section_statuses,
        'total_marks', total_marks,
        'reason', CASE WHEN selected_rows IS NULL THEN 'EXACT_ALLOCATION_UNSATISFIED' ELSE NULL END
    );
END;
$$;

CREATE OR REPLACE FUNCTION taleem_board_blueprint_before_write()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    outcome JSONB;
BEGIN
    PERFORM taleem_validate_selection_spec(NEW.config);
    IF NEW.is_active THEN
        outcome := taleem_select_questions(
            NEW.board_id, NEW.class_id, NEW.subject_id, NEW.config, 'blueprint-activation'
        );
        IF NOT COALESCE((outcome->>'satisfiable')::BOOLEAN, FALSE) THEN
            RAISE EXCEPTION 'BLUEPRINT_ACTIVATION_UNSATISFIED' USING ERRCODE = 'P0001';
        END IF;
    END IF;
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER board_paper_blueprints_validate_before_write
BEFORE INSERT OR UPDATE OF board_id, class_id, subject_id, config, is_active
ON board_paper_blueprints
FOR EACH ROW EXECUTE FUNCTION taleem_board_blueprint_before_write();

ALTER TABLE board_paper_blueprints ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON board_paper_blueprints FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION taleem_validate_selection_spec(JSONB) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION taleem_allocate_selection_recursive(TEXT, TEXT, TEXT, JSONB, TEXT, INTEGER, INTEGER, UUID[], UUID[], JSONB) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION taleem_select_questions(TEXT, TEXT, TEXT, JSONB, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION taleem_board_blueprint_before_write() FROM PUBLIC, anon, authenticated;

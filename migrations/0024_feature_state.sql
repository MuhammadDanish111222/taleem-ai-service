-- Module 7 Run 3: Three-state feature lifecycle read function and updated test paper wrapper.
-- Preserves taleem_runtime_feature_enabled() for backward compatibility.
SET search_path = public, pg_catalog;

-- Returns strictly 'enabled' | 'coming_soon' | 'disabled'.
-- Any unexpected/corrupted stored value fails closed to 'disabled'.
CREATE OR REPLACE FUNCTION taleem_runtime_feature_state(p_feature TEXT)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, pg_catalog
AS $$
  SELECT CASE p_feature
    WHEN 'test_generation' THEN
      CASE COALESCE(
        (SELECT value #>> '{}' FROM system_settings WHERE key = 'runtime:feature.test_generation:global|||'),
        'enabled'
      )
        WHEN 'enabled' THEN 'enabled'
        WHEN 'coming_soon' THEN 'coming_soon'
        WHEN 'disabled' THEN 'disabled'
        ELSE 'disabled'
      END
    WHEN 'multiple_ask' THEN
      CASE COALESCE(
        (SELECT value #>> '{}' FROM system_settings WHERE key = 'runtime:feature.multiple_ask:global|||'),
        'disabled'
      )
        WHEN 'enabled' THEN 'enabled'
        WHEN 'coming_soon' THEN 'coming_soon'
        WHEN 'disabled' THEN 'disabled'
        ELSE 'disabled'
      END
    ELSE 'disabled'
  END
$$;
REVOKE ALL ON FUNCTION taleem_runtime_feature_state(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION taleem_runtime_feature_state(TEXT) TO service_role;

-- Replaces checked wrapper using CREATE OR REPLACE FUNCTION (preserves dependent grants)
CREATE OR REPLACE FUNCTION taleem_generate_test_paper(
    p_mode TEXT, p_board_id TEXT, p_class_id TEXT, p_subject_id TEXT,
    p_spec JSONB, p_seed TEXT
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_catalog
AS $$
DECLARE
    feature_state TEXT;
    generated JSONB;
    max_duration INTEGER;
BEGIN
    feature_state := taleem_runtime_feature_state('test_generation');
    IF feature_state = 'coming_soon' THEN
        RAISE EXCEPTION 'FEATURE_COMING_SOON' USING ERRCODE = 'P0001';
    END IF;
    IF feature_state != 'enabled' THEN
        RAISE EXCEPTION 'FEATURE_NOT_ENABLED' USING ERRCODE = 'P0001';
    END IF;
    generated := taleem_generate_test_paper_unchecked(
        p_mode, p_board_id, p_class_id, p_subject_id, p_spec, p_seed
    );
    max_duration := COALESCE(
        (SELECT (value #>> '{}')::INTEGER FROM system_settings
         WHERE key = 'runtime:test_generation.max_duration_minutes:global|||'),
        600
    );
    IF (generated->>'duration_minutes')::INTEGER > max_duration THEN
        RAISE EXCEPTION 'FEATURE_CONFIGURATION_LIMIT' USING ERRCODE = 'P0001';
    END IF;
    RETURN generated;
END;
$$;
REVOKE ALL ON FUNCTION taleem_generate_test_paper(TEXT, TEXT, TEXT, TEXT, JSONB, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION taleem_generate_test_paper(TEXT, TEXT, TEXT, TEXT, JSONB, TEXT)
    TO service_role;

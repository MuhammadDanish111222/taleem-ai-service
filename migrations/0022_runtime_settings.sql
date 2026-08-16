-- Module 7 Run 1: typed local runtime settings. Existing owner tables remain authoritative.
SET search_path = public, pg_catalog;

ALTER TABLE system_settings ADD COLUMN IF NOT EXISTS revision BIGINT NOT NULL DEFAULT 1 CHECK (revision > 0);

CREATE TABLE IF NOT EXISTS runtime_setting_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_key TEXT NOT NULL,
    scope JSONB NOT NULL,
    previous_value JSONB NOT NULL,
    new_value JSONB NOT NULL,
    actor_id TEXT NOT NULL,
    revision BIGINT NOT NULL CHECK (revision > 0),
    request_id TEXT NOT NULL CHECK (char_length(request_id) <= 200),
    cache_namespace TEXT NOT NULL,
    cache_generation BIGINT NOT NULL CHECK (cache_generation > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_runtime_setting_audits_key_created ON runtime_setting_audits(setting_key, created_at DESC);
ALTER TABLE runtime_setting_audits ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON runtime_setting_audits FROM PUBLIC, anon, authenticated;

-- The Supabase Edge Function executes this existing RPC as service_role. This
-- check deliberately lives in PostgreSQL, never Railway, and defaults enabled
-- to preserve Module 6's released behaviour during the upgrade.
CREATE OR REPLACE FUNCTION taleem_runtime_feature_enabled(p_feature TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_catalog AS $$
  SELECT COALESCE((SELECT value #>> '{}' FROM system_settings
                   WHERE key = 'runtime:feature.' || p_feature || ':global|||'), 'enabled') = 'enabled';
$$;
REVOKE ALL ON FUNCTION taleem_runtime_feature_enabled(TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION taleem_runtime_feature_enabled(TEXT) TO service_role;

-- Keep the Module 6 Edge -> PostgreSQL path authoritative for lifecycle
-- enforcement. The Edge Function continues to call the same public RPC; it
-- never needs a Railway round trip to decide whether generation is available.
ALTER FUNCTION taleem_generate_test_paper(TEXT, TEXT, TEXT, TEXT, JSONB, TEXT)
    RENAME TO taleem_generate_test_paper_unchecked;
CREATE FUNCTION taleem_generate_test_paper(
    p_mode TEXT, p_board_id TEXT, p_class_id TEXT, p_subject_id TEXT,
    p_spec JSONB, p_seed TEXT
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_catalog
AS $$
DECLARE
    generated JSONB;
    max_duration INTEGER;
BEGIN
    IF NOT taleem_runtime_feature_enabled('test_generation') THEN
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

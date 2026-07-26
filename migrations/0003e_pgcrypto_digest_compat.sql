-- Standard PostgreSQL does not put the `extensions` schema on search_path.
-- Keep existing Phase 3D SQL portable without changing the already-applied
-- pgcrypto extension migration.
CREATE OR REPLACE FUNCTION public.digest(data text, type text)
RETURNS bytea
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$ SELECT extensions.digest(data, type); $$;

REVOKE ALL ON FUNCTION public.digest(text, text) FROM PUBLIC, anon, authenticated;

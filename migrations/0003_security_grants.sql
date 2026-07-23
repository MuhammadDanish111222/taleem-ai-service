-- Migration 0003: Security Grants & Row Level Security (RLS)
-- Enables RLS on all application tables and revokes access from anon and authenticated roles.

-- 1. Idempotently create anon and authenticated roles if they do not exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'anon') THEN
        CREATE ROLE anon NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated NOLOGIN;
    END IF;
END $$;

-- 2. Explicitly set safe search_path
SET search_path = public, pg_catalog;

-- 3. Enable Row Level Security on every application table
ALTER TABLE job_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE admin_audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_answers ENABLE ROW LEVEL SECURITY;
ALTER TABLE provider_attempts ENABLE ROW LEVEL SECURITY;


ALTER TABLE rag_corpora ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_corpus_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_document_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_visuals ENABLE ROW LEVEL SECURITY;

ALTER TABLE chunk_expected_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE approved_question_bank ENABLE ROW LEVEL SECURITY;
ALTER TABLE solved_papers ENABLE ROW LEVEL SECURITY;

-- 4. Revoke all table permissions from public, anon, and authenticated
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC, anon, authenticated;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC, anon, authenticated;

-- 5. Revoke execute on all functions from public, anon, and authenticated
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC, anon, authenticated;

-- 6. Grant USAGE on schema public so table-level RLS privilege checks (42501) are reached
GRANT USAGE ON SCHEMA public TO anon, authenticated;


-- Migration 0001: Platform Core Tables
-- Creates PostgreSQL extensions, job_queue, system_settings, admin_audit_logs, ai_requests, and ai_answers.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Job Queue Table
CREATE TABLE IF NOT EXISTS job_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('queued', 'leased', 'running', 'retry_wait', 'succeeded', 'failed', 'cancelled')),
    stage TEXT NULL,
    progress NUMERIC NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    attempt_count INT NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INT NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    locked_by TEXT NULL,
    locked_at TIMESTAMPTZ NULL,
    heartbeat_at TIMESTAMPTZ NULL,
    next_retry_at TIMESTAMPTZ NULL,
    error_code TEXT NULL,
    error_message TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_job_queue_status_next_retry ON job_queue (status, next_retry_at);
CREATE INDEX IF NOT EXISTS idx_job_queue_locked_by ON job_queue (locked_by) WHERE locked_by IS NOT NULL;

-- System Settings Table
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by TEXT NULL
);

-- Admin Audit Logs Table
CREATE TABLE IF NOT EXISTS admin_audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    before_value JSONB NULL,
    after_value JSONB NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_logs_actor ON admin_audit_logs (actor_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_admin_audit_logs_target ON admin_audit_logs (target_type, target_id);

-- AI Requests Table (includes MVP v1 caching key columns)
CREATE TABLE IF NOT EXISTS ai_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('en', 'ur', 'roman_ur', 'mixed')),
    answer_mode TEXT NOT NULL CHECK (answer_mode IN ('concise', 'detailed', 'step_by_step', 'exam_style')),
    raw_question TEXT NOT NULL,
    normalized_question TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    corpus_version_id UUID NULL,
    prompt_version TEXT NOT NULL DEFAULT 'v1',
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cached')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_requests_cache_lookup ON ai_requests (
    board_id, class_id, subject_id, answer_mode, language, corpus_version_id, prompt_version, question_hash
);

-- AI Answers Table (includes MVP v1 caching score columns)
CREATE TABLE IF NOT EXISTS ai_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL UNIQUE REFERENCES ai_requests(id) ON DELETE CASCADE,
    answer_text TEXT NOT NULL,
    citation_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_text_score NUMERIC NULL CHECK (chunk_text_score IS NULL OR (chunk_text_score >= 0 AND chunk_text_score <= 1)),
    expected_question_score NUMERIC NULL CHECK (expected_question_score IS NULL OR (expected_question_score >= 0 AND expected_question_score <= 1)),
    tokens_used INT NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
    latency_ms INT NOT NULL DEFAULT 0 CHECK (latency_ms >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

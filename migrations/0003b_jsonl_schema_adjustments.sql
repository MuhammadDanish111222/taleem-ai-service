-- Migration 0003b: Schema adjustments for Phase 3C Admin JSONL Chunk Ingestion
-- 1. Drop NOT NULL constraint on chunk_expected_questions.embedding for pre-embedding storage
ALTER TABLE chunk_expected_questions ALTER COLUMN embedding DROP NOT NULL;

-- 2. Add content_type, metadata, content_hash, language, token_count columns to rag_chunks
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS content_type TEXT NOT NULL DEFAULT 'explanation';
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS language TEXT NOT NULL DEFAULT 'en';
ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS token_count INT NOT NULL DEFAULT 0;

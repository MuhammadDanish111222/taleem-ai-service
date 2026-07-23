-- Migration 0002: RAG Schema & MVP v1 Tables
-- Creates rag_corpora, rag_corpus_versions, rag_document_versions, rag_chunks, rag_visuals,
-- chunk_expected_questions, approved_question_bank, and solved_papers.

-- RAG Corpora Table
CREATE TABLE IF NOT EXISTS rag_corpora (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (board_id, class_id, subject_id)
);

-- RAG Corpus Versions Table
CREATE TABLE IF NOT EXISTS rag_corpus_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_id UUID NOT NULL REFERENCES rag_corpora(id) ON DELETE CASCADE,
    version_no INT NOT NULL CHECK (version_no > 0),
    embedding_model TEXT NOT NULL,
    embedding_revision TEXT NOT NULL,
    embedding_dim INT NOT NULL CHECK (embedding_dim > 0),
    normalize_embeddings BOOLEAN NOT NULL DEFAULT TRUE,
    query_instruction TEXT NULL,
    chunking_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('building', 'qa_ready', 'active', 'superseded', 'failed')),
    expected_chunk_count INT NOT NULL DEFAULT 0 CHECK (expected_chunk_count >= 0),
    embedded_chunk_count INT NOT NULL DEFAULT 0 CHECK (embedded_chunk_count >= 0),
    activated_at TIMESTAMPTZ NULL,
    activated_by TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (corpus_id, version_no)
);

-- Enforce ONE active version per corpus scope at the database constraint level
CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_corpus_versions_active_scope 
ON rag_corpus_versions (corpus_id) 
WHERE status = 'active';

-- Foreign key linking ai_requests to rag_corpus_versions
ALTER TABLE ai_requests 
ADD CONSTRAINT fk_ai_requests_corpus_version 
FOREIGN KEY (corpus_version_id) 
REFERENCES rag_corpus_versions(id) 
ON DELETE SET NULL;

-- RAG Document Versions Table (links to Module 2 resources)
CREATE TABLE IF NOT EXISTS rag_document_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    corpus_version_id UUID NOT NULL REFERENCES rag_corpus_versions(id) ON DELETE CASCADE,
    resource_id TEXT NOT NULL,
    resource_version_id TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    total_chunks INT NOT NULL DEFAULT 0 CHECK (total_chunks >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (resource_id, resource_version_id, pipeline_version, corpus_version_id)
);

-- RAG Chunks Table (with chapter_id, topic_no, topic_title, pgvector embedding, and simple tsvector)
CREATE TABLE IF NOT EXISTS rag_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_version_id UUID NOT NULL REFERENCES rag_document_versions(id) ON DELETE CASCADE,
    corpus_version_id UUID NOT NULL REFERENCES rag_corpus_versions(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    chapter_id TEXT NULL,
    topic_no TEXT NULL,
    topic_title TEXT NULL,
    page_start INT NULL CHECK (page_start IS NULL OR page_start >= 1),
    page_end INT NULL CHECK (page_end IS NULL OR page_end >= page_start),
    embedding vector(768) NULL,
    content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_version_id, corpus_version_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsvector ON rag_chunks USING gin (content_tsvector);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_chapter_topic ON rag_chunks (corpus_version_id, chapter_id, topic_no);

-- RAG Visuals Table
CREATE TABLE IF NOT EXISTS rag_visuals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES rag_chunks(id) ON DELETE CASCADE,
    visual_type TEXT NOT NULL CHECK (visual_type IN ('diagram', 'table', 'equation', 'figure')),
    storage_path TEXT NOT NULL,
    caption TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chunk Expected Questions Table (MVP v1 - vector per question)
CREATE TABLE IF NOT EXISTS chunk_expected_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES rag_chunks(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Approved Question Bank Table (MVP v1)
CREATE TABLE IF NOT EXISTS approved_question_bank (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    normalized_question TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('generated', 'reviewed', 'approved', 'archived')),
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_approved_question_bank_lookup ON approved_question_bank (
    board_id, class_id, subject_id, question_hash
);

-- Solved Papers Table (MVP v1)
CREATE TABLE IF NOT EXISTS solved_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    board_id TEXT NOT NULL,
    class_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    year INT NOT NULL CHECK (year >= 1990 AND year <= 2100),
    session TEXT NOT NULL,
    source_file_hash TEXT NOT NULL,
    corpus_version_id UUID NULL REFERENCES rag_corpus_versions(id) ON DELETE SET NULL,
    status TEXT NOT NULL CHECK (status IN ('uploaded', 'extracted', 'verified', 'published', 'failed')),
    questions JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

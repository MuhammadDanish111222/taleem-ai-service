# Taleem AI Service Architecture & Database Schema Specification

This document provides the definitive architectural design, database schema specifications, and database governance contracts for `taleem-ai-service`.

---

## 1. Migration Strategy & Runner Rationale

`taleem-ai-service` uses plain, deterministic SQL migrations stored in `migrations/` and executed in sorted order by `app/db/migrator.py`.

### Rationale:
- **Zero ORM Overhead**: Staying strictly on raw SQL with `asyncpg` matches the platform decision to keep transactions, vector queries, row locks, and worker leases explicitly controlled.
- **Portability & Idempotency**: Plain `.sql` files can be applied to a disposable CI PostgreSQL instance (or local Docker container) and deployed directly to Supabase production without translation layers.
- **Tracking**: Applied migration file names are recorded in the `schema_migrations` table within an atomic transaction per file.

---

## 2. Platform Core & RAG Database Schema

### Platform Core Tables (`0001_platform_core.sql`)
1. **`job_queue`**: Durable background job execution with CHECK constraints on `status` (`queued|leased|running|retry_wait|succeeded|failed|cancelled`), `progress` (0..100), `attempt_count` (>= 0), `max_attempts` (> 0). Atomic worker leasing uses `FOR UPDATE SKIP LOCKED`.
2. **`system_settings`**: Global system configuration key-value pairs.
3. **`admin_audit_logs`**: Immutable audit logs capturing administrative mutations (`actor_id`, `action`, `target_type`, `target_id`, `before_value`, `after_value`).
4. **`ai_requests`**: Log of user AI interactions. Contains MVP v1 cache composite key columns (`board_id`, `class_id`, `subject_id`, `language`, `answer_mode`, `normalized_question`, `question_hash`, `prompt_version`, `corpus_version_id`).
5. **`ai_answers`**: Generated AI answers joined 1:1 with `ai_requests`. Contains MVP v1 score columns (`chunk_text_score`, `expected_question_score`).
6. **`provider_attempts`**: Individual external LLM/embedding API call log (`ai_request_id`, `job_id`, `provider`, `model`, `attempt_no`, `provider_request_id`, `system_fingerprint`, `finish_reason`, `prompt_tokens`, `cache_tokens`, `reasoning_tokens`, `completion_tokens`, `latency_ms`, `status`, `error_code`, `trace_id`).


### RAG Schema Tables (`0002_rag_schema.sql`)
1. **`rag_corpora`**: Unique scope mapping for textbook corpora (`board_id`, `class_id`, `subject_id`).
2. **`rag_corpus_versions`**: Versioned corpus snapshots. Enforces **at most one active version per board/class/subject** at the database level using a partial unique index:
   ```sql
   CREATE UNIQUE INDEX idx_rag_corpus_versions_active_scope 
   ON rag_corpus_versions (corpus_id) WHERE status = 'active';
   ```
3. **`rag_document_versions`**: Contract link between RAG and `taleem-web` Module 2 resources (`resource_id`, `resource_version_id`, `pipeline_version`).
4. **`rag_chunks`**: Content chunks linked to document versions. Features:
   - Metadata: `chapter_id` (Firestore chapter slug), `topic_no`, `topic_title`, `page_start`, `page_end`.
   - Vector: `vector(768)` embedding column.
   - Lexical Search: Generated `tsvector` using `'simple'` configuration:
     ```sql
     content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
     ```
5. **`rag_visuals`**: Visual elements (`diagram`, `table`, `equation`, `figure`) linked to chunks with `ON DELETE CASCADE`.
6. **`chunk_expected_questions`**: MVP v1 expected questions table with individual `vector(768)` embedding per question row.
7. **`approved_question_bank`**: Lightweight approved question/answer bank table.
8. **`solved_papers`**: Solved past paper snapshots linking `year`, `session`, `questions` (JSONB) to corpus versions.

---

## 3. Security, Grants & RLS Model (`0003_security_grants.sql`)

- **Deny-by-Default RLS**: Row Level Security (RLS) is enabled on all 13 application tables.
- **Public & Role Restrictions**: Table access and function execution privileges are revoked from `PUBLIC`, `anon`, and `authenticated` roles.
- **Idempotent Role Guard**: Role creation in `0003_security_grants.sql` is wrapped in PL/pgSQL guards to ensure safety across both bare disposable CI databases and hosted Supabase environments:
  ```sql
  DO $$
  BEGIN
      IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'anon') THEN
          CREATE ROLE anon NOLOGIN;
      END IF;
      IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticated') THEN
          CREATE ROLE authenticated NOLOGIN;
      END IF;
  END $$;
  ```
- **Service Role Access**: Only `taleem-ai-service`'s `service_role` connection holds read/write database access.

---

## 4. Vector & Lexical Search Indexing Decisions

### Vector Search Indexing Strategy:
- **Launch Mode**: Exact vector search (`<->` L2 distance or `<=>` Cosine similarity) without an HNSW index. At launch document volumes (< 50,000 chunks total), exact vector search provides 100% recall with sub-5ms query latency and zero index build overhead.
- **HNSW Upgrade Trigger**: An HNSW index (`USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64)`) will be introduced via a dedicated migration when any single corpus version exceeds **500,000 embedded chunks**.

### Language-Aware Lexical Search Strategy:
- Textbook content contains a mixture of English, Urdu, and Roman Urdu.
- Standard English stemmers fail or corrupt Urdu transliterations.
- We use PostgreSQL's `'simple'` text search configuration (`to_tsvector('simple', content)`), which tokenizes and lowercases text without applying language-specific stemming rules, ensuring accurate search matching for Urdu, English, and Roman Urdu.

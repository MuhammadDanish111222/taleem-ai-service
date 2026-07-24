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

---

## 5. Concurrency & Row Locking Hierarchy Strategy

To prevent race conditions, duplicate version generation, and database deadlocks during multi-chapter background ingestion:

### Strict Lock Order:
1. **Parent Corpora Lock (First)**: `SELECT id FROM rag_corpora WHERE id = $1 FOR UPDATE` (acquired after atomic `ON CONFLICT DO UPDATE` upsert).
2. **Corpus Version Lock (Second)**: `SELECT status FROM rag_corpus_versions WHERE id = $1 FOR UPDATE`.

### Governance Contract:
- **Phase 3C (Chunk Ingestion)**: Locks `rag_corpora` to check/create the single `building` corpus version for a subject scope, then locks `rag_corpus_versions` to verify `status == 'building'` before replacing chapter chunks and expected questions.
- **Phase 3F (Activation Engine)**: When activating or superseding corpus versions, Phase 3F **MUST** follow this exact lock order (locking `rag_corpora` before modifying `rag_corpus_versions`) to guarantee deadlock-free execution against concurrent chunk ingestion workers.

---

## 6. Known Gaps Identified for Future Phases

The following open questions were identified during Phase 3C closeout review against the build guide. They are recorded here for whoever scopes Phases 3D, 3E, and 7A — no implementation action is required now.

### Visual Pipeline (no assigned phase)

The `rag_visuals` table exists in the schema (`0002_rag_schema.sql`) but is an **incomplete stub** relative to the full spec. The current columns are `id`, `chunk_id`, `visual_type`, `storage_path`, `caption`, and `created_at`. The build guide's visual-provenance specification requires six additional columns that are not yet present: `page`, `bbox`, `reading_order`, `content_hash`, `review_status`, and `complex_structure`. Of these, `bbox`/`page` provide spatial provenance for PDF-layout extraction, and `complex_structure` is what Phase 4E's source-image-fallback logic depends on. The current JSONL ingestion path (`admin_jsonl_v1`) does not populate `rag_visuals` at all, and no later phase should assume the table is ready to insert into as-is — a new migration extending it with the missing columns will be needed when the automated PDF-layout pipeline is built. Later phases (3E retrieval, 3F activation/QA, 4A answer generation, 4E citations, 4F visual rendering) reference visual-element functionality that depends on both populated rows and the full column set. For content ingested via the admin JSONL path, visual retrieval should be treated as **deferred/optional** until that pipeline exists. Per the build guide's own MVP v1 decision: *"revisit automated extraction once manual chunking becomes the bottleneck on content-addition speed, or once visual/flowchart retrieval is needed."*

### Expected-Question Embeddings (Phase 3D / 3E gap)

The `chunk_expected_questions` table exists and Phase 3C populates question text rows with `embedding = NULL`. However, neither Phase 3D (embedding generation) nor Phase 3E (retrieval query engine) as currently described explicitly addresses generating embeddings for these rows or including them as a retrieval signal. Whoever scopes Phase 3D should decide whether expected-question embedding is part of the same batch-embedding job that handles `rag_chunks.embedding`, and whoever scopes Phase 3E should decide whether expected-question vector similarity is a retrieval pathway (and if so, how it fuses with chunk-text dense/lexical scores).

### Retrieval Settings Granularity (Phase 7A gap)

Phase 7A's typed settings service is described as managing "retrieval top K" generically, but the actual retrieval pipeline (Phase 3E) will require several distinct sub-parameters: dense candidate count, lexical candidate count, expected-question candidate count, evidence-sufficiency thresholds, per-document result caps, and the RRF (Reciprocal Rank Fusion) constant. None of these are individually named in the current Phase 7A specification. Whoever scopes Phase 7A should enumerate and type these parameters explicitly so the settings service covers the full retrieval configuration surface.

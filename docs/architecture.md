# Taleem AI Service Architecture & Database Schema Specification

This document provides the definitive architectural design, database schema specifications, and database governance contracts for `taleem-ai-service`.

## Module 4 Run 1: Ask backend foundation

- `ai_requests`/`ai_answers` remain the only operational generated-candidate pool. Student LLM answers are pending, retain provider/prompt/corpus provenance, and expire after 90 days unless approval links them to the trusted bank.
- The former lightweight `approved_question_bank` is migrated into one normalized bank with stable questions, immutable revisions, variations, MCQ options, reviewed citations/visual links, and local BGE embedding jobs. Admin-authored/imported rows are approved immediately; generated rows never become reusable without explicit audited approval.
- `answer_mode`, `answer_style`, and backend-assigned `answer_source` are independent. Single Ask accepts typed-text `short|long` with `exam_style`; files, images, PDFs, OCR, `mcq`, and `mixed` are outside Module 4.
- Ask resolves exact approved questions, exact approved variations, optional evaluated semantic reuse, scoped Phase 3E retrieval, then grounded or explicitly enabled General AI generation. Semantic reuse is disabled by default and has no default threshold.
- Editable prompts are PostgreSQL-versioned and resolve board+class+subject, class+subject, subject, then global. Immutable source/safety rules remain in code. Activation/rollback and the shared prompt-cache generation change in one database transaction.
- Redis performs atomic quota reservation with PostgreSQL as the authoritative mirror/fallback. Request UUIDs are identity-scoped. Pakistan business days reset at midnight in `Asia/Karachi`; PostgreSQL also mirrors HMAC-hashed internal JWT JTIs so Redis outages do not disable replay protection.
- Generated output is validated as a whole. Invented or mixed-invalid citations/visuals fail; General AI has neither; `always` visuals are appended deterministically. Provider input contains text plus approved metadata only, never image bytes, storage identifiers, URLs, or raw vectors.
- Retrieval strength controls evidence ordering, not answer-source truth. Non-empty strong or weak evidence is offered to the grounded prompt, but the backend requires a verified citation before assigning `syllabus_grounded`. An uncited non-empty answer is relabelled `general_knowledge` only when the scoped policy permits fallback; empty output becomes `no_answer`.

## Module 4 Run 2: Operational Ask architecture

- The public contract remains one typed-text Single Ask. `answer_mode` (`short|long`), `answer_style` (`exam_style`), and server-owned `answer_source` are separate dimensions. Module 5 OCR, uploads, PDFs, images, multipart input, and Multiple Ask are not present.
- The two logical answer pools have distinct trust states: normalized approved questions/revisions/variations are reusable, while `ai_requests`/`ai_answers` are operational generated candidates. An unapproved candidate can never satisfy a later Ask.
- Prompt resolution is most-specific-first: board+class+subject, class+subject, subject, then global, separately for grounded/general and mode. Activation or rollback increments a PostgreSQL cache generation and publishes Redis invalidation so the next request observes the change.
- Redis stores only opaque/HMAC-hashed identifiers and bounded counters/cache entries. It contains no raw UID, question, answer, profile, Drive identifier, or storage key. Usage is reserved atomically and expires at Pakistan midnight; PostgreSQL prevents unlimited access, duplicate quota, and replay when Redis is unavailable.
- Approved and grounded visual blocks carry only logical visual IDs to the web. The protected resolver checks reviewed database links and a configured storage allowlist; ambiguous, invented, or unreviewed IDs fail closed. General AI has no visual/citation path.
- The provider adapter uses environment-configurable `deepseek-v4-flash`, non-thinking mode, JSON object output, bounded characters/tokens, timeout, and retries. The actual provider/model is persisted per attempt; prompt draft tests are audited without raw prompt or question content.
- Semantic approved matching is off until an evaluation demonstrates acceptable precision. The locked BGE model/revision is used only against approved questions and approved variations, never candidate answers.

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
6. **`chunk_expected_questions`**: One expected question per row with an individual nullable `vector(768)` embedding slot. Phase 3C rejects blank and normalized-duplicate questions and the database enforces uniqueness per chunk.
7. **`approved_question_bank`**: Lightweight approved question/answer bank table.
8. **`solved_papers`**: Solved past paper snapshots linking `year`, `session`, `questions` (JSONB) to corpus versions.

### Phase 3D Embedding Completeness (`0004_phase3d_embeddings.sql`)

- Corpus versions store an immutable embedding-configuration fingerprint plus expected and embedded counters for both chunks and expected questions.
- Chunk and expected-question rows record the model, revision, input hash, status, and completion timestamps for their individual `vector(768)` values. Existing vectors without this provenance remain unverified and cannot satisfy readiness.
- `phase3d_require_corpus_complete()` prevents `qa_ready` and `active` transitions unless every required vector matches the stored version configuration and dimension, counters agree with actual rows, and no current embedding job is pending or failed. The function retains the deny-by-default function-grant model.
- `BAAI/bge-base-en-v1.5` is pinned to revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; BGE uses CLS pooling (`last_hidden_state[:, 0]`) followed by L2 normalization. The MVP embedding worker accepts English chunks/questions only.

### Phase 3F Local Admin QA, Visuals, and Activation (`0005_phase3f_local_admin.sql`)

- A chunk owns zero or more direct `rag_visuals` rows. Each row has a stable logical `visual_id`, type, nonblank title/description, `google_drive` provider, server-only storage key, display policy, review status, normalized text hash, and timestamps. The database relation remains `rag_chunks -> rag_visuals`; image bytes and Drive identifiers are never copied to Postgres or browser DTOs.
- JSONL may supply an optional `visuals` array. New imports are `pending` with `llm_decide` display policy by default. Only approved visual title/description participate in parent-chunk embedding input, ordered by logical `visual_id`; keys, provider IDs, paths, database IDs, and image bytes never participate.
- `rag_corpus_qa_approvals` persists the local reviewer, timestamp, request ID, and a safe action summary. Source, question, visual metadata, and display-policy edits invalidate an approval. `source_corpus_version_id` records active-to-building editable drafts.

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

### Phase 3E Scoped Retrieval

- Retrieval is internal service-layer code only. Every dense chunk, expected-question, lexical, and chapter-validation query joins `rag_corpora` and `rag_corpus_versions` and filters in SQL by board ID, class ID, subject ID, the exact active version ID returned for that scope, and optional chapter ID.
- Both vector channels use exact pgvector cosine distance (`<=>`). Phase 3D L2-normalizes BGE vectors, so cosine distance is a consistent angular comparison for chunk and expected-question vectors. No HNSW index or score-to-probability conversion is used.
- Expected questions are searched as individual vectors, deduplicated to their parent chunk using the best individual match, and only then assigned contiguous parent ranks (`1, 2, 3, ...`). An expected-question row is never exposed as a citation.
- Lexical search uses the existing scoped `'simple'` PostgreSQL full-text index. Reciprocal Rank Fusion (RRF, `k=60`) consumes ranks only and returns safe citations plus channel/rank contributions. Raw cosine distances, lexical scores, and RRF weights stay internal and are never confidence values.
- The active corpus version's stored BGE model, revision, dimensions, normalization flag, query instruction, and fingerprint are reconstructed and verified before a single on-demand query embedding is made. Inference runs in a worker thread, not on the FastAPI event loop, and is not a durable Railway job.
- Phase 3F draft QA uses this same three-channel retrieval implementation against an explicitly named `building` or `qa_ready` version. It is scope-checked in SQL and never changes active-version resolution. Student visual rendering remains out of scope; retrieval DTOs continue to return no visual storage references.
- Evidence strength is evaluated only on the top fused parent chunk: `none` when no scoped fused results exist; `strong` when that top parent has at least two distinct channels at ranks 1--3; otherwise `weak`. A single channel is always `weak`, and duplicate expected-question rows still contribute only one expected-question channel.

---

## 5. Concurrency & Row Locking Hierarchy Strategy

To prevent race conditions, duplicate version generation, and database deadlocks during multi-chapter background ingestion:

### Strict Lock Order:
1. **Parent Corpora Lock (First)**: `SELECT id FROM rag_corpora WHERE id = $1 FOR UPDATE` (acquired after atomic `ON CONFLICT DO UPDATE` upsert).
2. **Corpus Version Lock (Second)**: `SELECT status FROM rag_corpus_versions WHERE id = $1 FOR UPDATE`.

### Governance Contract:
- **Phase 3C (Chunk Ingestion)**: Locks `rag_corpora` to check/create the single `building` corpus version for a subject scope. Before first activation, a same-configuration `qa_ready` version is reopened as `building` so chapter-by-chapter imports extend one subject snapshot and invalidate any prior QA approval. It then locks `rag_corpus_versions` to verify `status == 'building'` before replacing chapter chunks and expected questions.
- **Phase 3D (Embedding)**: The local-admin worker processes a corpus generation in the durable order `embed_chunks` → `embed_questions` → `corpus_completeness`. Completion of one stage and enqueueing of its successor occur in one database transaction, so readiness is never leaseable while an embedding population is still running. `WORKER_MODE` is explicit: `local_admin` owns bulk jobs and `railway_public` owns none in Phase 3D.
- **Phase 3F (Activation Engine)**: Activation and rollback use one transaction: lock `rag_corpora` first, lock all its version rows second, recheck persisted vector provenance/counts/jobs, scope/status, current QA approval, and displayable visual storage, then supersede the previous active row and activate exactly one target. The active-version partial unique index remains the final database invariant. Retrieval caches only the active version ID and embedding configuration in Redis for five minutes; activation and rollback invalidate the scope key immediately after the database transaction commits. Redis failures fall back safely to PostgreSQL.

---

## 6. Known Gaps Identified for Future Phases

The following open questions were identified during Phase 3C closeout review against the build guide. They are recorded here for whoever scopes Phases 3D, 3E, and 7A — no implementation action is required now.

### Visual Pipeline (future PDF-layout work)

Phase 3F supersedes the historical Phase 3C note below: manual JSONL now persists reviewed visual metadata as direct chunk children, while image bytes stay in Google Drive. Student visual rendering and automated PDF-layout extraction are still deferred. Page/bbox/reading-order provenance remains nullable for the later layout pipeline; local-only previews allow only PNG, JPEG, WebP, and GIF through the controlled server stream, never Drive URLs or keys. Existing PDF reading remains separate.


### Token-count contract (Phase 3C)

New JSONL chunks use the configured embedding tokenizer (`BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`) without loading the embedding model. The tokenizer method/version is stored in chunk metadata. Existing stored counts are not rewritten by this change; re-ingest a chapter when its historical counts need to be recalculated.

### Paired JSONL + Visual Extracts DOCX import (Phase 3F extension)

- The web BFF alone parses the local-admin upload, validates the Word card/drawing/crop relationship, and privately uploads referenced allowlisted image assets to the configured Google Drive folder. The service does not receive source DOCX bytes.
- The external JSONL contains no storage key or Drive reference. The BFF creates enriched JSONL only in memory and forwards it to the existing signed `/internal/ingest/jsonl` path. Normal Firestore hierarchy validation remains mandatory, so board/class/subject/chapter must already exist and be active.
- `rag_paired_imports` records only import and asset hashes, counts, job ID/status, and stable error code. It is RLS-protected and has no public/anon/authenticated grants; it never stores bytes, raw/enriched JSONL, Drive keys/IDs/URLs, or visual source references.
- This is a local-admin-only workflow, does not auto-activate a corpus, leaves visuals pending review with `llm_decide` as their eventual display policy, and uses no paid LLM, OCR, image-generation, or vision API.

### Retrieval Settings Granularity (Phase 7A gap)

Phase 7A's typed settings service is described as managing "retrieval top K" generically, but the actual retrieval pipeline (Phase 3E) will require several distinct sub-parameters: dense candidate count, lexical candidate count, expected-question candidate count, evidence-sufficiency thresholds, per-document result caps, and the RRF (Reciprocal Rank Fusion) constant. None of these are individually named in the current Phase 7A specification. Whoever scopes Phase 7A should enumerate and type these parameters explicitly so the settings service covers the full retrieval configuration surface.

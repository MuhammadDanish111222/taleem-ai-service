# Taleem AI Service - Key Decisions & Architectural Changes

This document logs significant architectural decisions and changes made for the Python AI microservice.

## Module 4 Run 1 decisions

- **Decision:** Keep exactly two logical answer pools.
  - `ai_requests`/`ai_answers` hold generated operational candidates; the normalized revision bank is the sole trusted reusable source. Legacy bank rows are migrated and the old table is retired rather than kept as a second authority.
- **Decision:** Approval is an explicit admin action.
  - Valid local-admin-authored/imported questions are approved immediately with actor/time/audit provenance. Student LLM output always starts pending; approval creates a bank revision, links and retains the original candidate, and never silently trusts generation.
- **Decision:** Make source assignment and reference validation non-editable.
  - The backend alone assigns `approved_bank`, `syllabus_grounded`, or `general_knowledge`. Editable prompts cannot override source choice, structured validation, the text-only boundary, or the rule that General AI has no textbook citations or visuals.
- **Decision:** Treat PostgreSQL as the quota continuity authority.
  - Redis Lua is the normal atomic decision path, while every reservation is mirrored transactionally with the pending request. Identity-scoped idempotency prevents same UUIDs from colliding across users; guarded PostgreSQL fallback and HMAC-hashed JTI claims preserve limits and replay protection during Redis outages.

## Module 4 Run 2 decisions

- **Decision:** Keep semantic approved reuse disabled at launch.
  - The locked BGE evaluation included paraphrases, punctuation/case variants, closely related questions, cross-chapter/subject hard negatives, and short ambiguous text. A useful-recall threshold reused an ambiguous negative, so precision did not satisfy the safe-launch gate. Exact approved questions and exact approved variations remain enabled.
- **Decision:** Make cache invalidation database-authoritative.
  - Prompt activation/rollback changes a PostgreSQL generation in the same transaction as the active version. Redis publication accelerates sharing but cannot become the only invalidation authority.
- **Decision:** Fail closed on logical visual ambiguity.
  - A visual is served only when its logical ID resolves uniquely to a reviewed approved/grounded link. Provider output cannot introduce a URL, Drive ID, storage key, or arbitrary visual identifier.
- **Decision:** Separate implementation completion from the real exit gate.
  - Passing disposable-stack and local browser tests does not authorize a real migration, provider charge, deployment, retention deletion, commit, or completion claim. Module 4 stays open until those real checks are directly verified.

## Phase 0: Framework & Architecture
- **Decision:** Python + FastAPI over Node.js for AI tasks.
- **Change Details:**
  - While the main web platform is built on Next.js (`taleem-web`), we chose Python and FastAPI for the AI service. This allows us to leverage Python's dominant ecosystem for AI/ML (Langchain, PyTorch, specialized tokenizers).
  - The service is designed as an isolated microservice, decoupling heavy generative AI workloads from the core web platform.

## Module 1 Compliance: Authentication & Internal Security Contract
- **Decision:** RS256 Asymmetric JWT Verification & Short-Lived TTL.
- **Change Details:**
  - All communication between `taleem-web` (BFF) and `taleem-ai-service` requires a valid internal JWT signed asymmetrically (RS256) by `taleem-web` using `INTERNAL_JWT_PRIVATE_KEY`.
  - `taleem-ai-service` verifies tokens using public keys configured via `INTERNAL_JWT_PUBLIC_KEYS_JSON`.
  - Tokens have a strict maximum TTL of 60 seconds (`exp`).
- **Decision:** Redis JTI Replay Prevention.
- **Change Details:**
  - To protect sensitive operations, `taleem-ai-service` stores each consumed JWT `jti` in Redis with a 60-second TTL.
  - Replayed `jti` values are rejected immediately with `401 Unauthorized`.
- **Decision:** Supabase Credential Ownership Isolation.
- **Change Details:**
  - `taleem-ai-service` exclusively holds `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `DATABASE_URL`, and `DEEPSEEK_API_KEY`.
  - Browser and `taleem-web` clients are strictly prohibited from receiving or using Supabase credentials.

## Phase 3A: RAG Foundation & Database Schema
- **Decision:** Raw SQL + Asyncpg for Database Repositories.
- **Change Details:**
  - Standardized on explicit SQL queries via `asyncpg` for all database interactions. No ORMs (SQLAlchemy / Supabase-py) allowed across any phase.
- **Decision:** PL/pgSQL Guarded Role Creation in Migration `0003_security_grants.sql`.
- **Change Details:**
  - Guarded `CREATE ROLE anon` and `CREATE ROLE authenticated` using PL/pgSQL `IF NOT EXISTS` checks to ensure migrations run safely and idempotently on both bare CI databases and hosted Supabase environments.
- **Decision:** Schema-Level Active Version Uniqueness.
- **Change Details:**
  - Added partial unique index `CREATE UNIQUE INDEX idx_rag_corpus_versions_active_scope ON rag_corpus_versions (corpus_id) WHERE status = 'active';` to enforce max 1 active version per corpus scope at the database level.
- **Decision:** Language-Aware Lexical Search Configuration (`simple`).
- **Change Details:**
  - `rag_chunks` generates a stored `content_tsvector` using PostgreSQL's `'simple'` search config to support Urdu, English, and Roman Urdu without inappropriate English stemming.
- **Decision:** Deferred HNSW Indexing.
- **Change Details:**
  - Documented exact vector search at MVP volumes and set 500,000 vectors as the trigger for introducing HNSW index.

## Phase 3B: Auth Audit & Worker Runtime Architecture
- **Decision:** Pre-existing Auth Code Audit & Strict Claim Validation.
  - Audited `app/core/internal_auth.py`. Confirmed RS256 signature check, `kid` lookup, `aud="taleem-ai-service"`, `iss="taleem-web"`, `exp` expiration, and Redis JTI replay check were already present.
  - **Gaps Added**: Strict validation of all mandatory claims (`uid`, `admin`, `feature`, `request_id`, `jti`, `iat`, `exp`), strict timestamp check (`exp - iat <= 60`s, `exp > iat`), and atomic Redis `set(key, "1", nx=True, ex=60)` replay store.
- **Decision:** Enforced Worker Lock Ownership.
  - All job status update queries (`heartbeat`, `progress`, `complete`, `fail`) explicitly filter on `locked_by = worker_id AND status IN ('leased', 'running')` and verify affected row count > 0 to prevent old/replaced workers from mutating recovered jobs.
- **Decision:** Worker Graceful Shutdown & Stale Recovery Policy.
  - On SIGTERM/SIGINT, the worker stops leasing new jobs and allows active jobs a grace period to complete while heartbeating. If the grace period expires, the worker process exits without releasing the DB lease, enabling safe stale-lease recovery after `heartbeat_at` timeout (preventing dual execution).
  - `recover_stale_jobs` resets stale jobs with attempt count < `max_attempts` to `retry_wait` (next_retry_at = NOW() + 5s), and terminally fails exhausted jobs with `STALE_LEASE_EXHAUSTED`.
- **Decision:** Unsupported Job Type Terminal Failure.
  - Workers encountering unhandled `job_type` values immediately mark jobs as `failed` with error code `UNSUPPORTED_JOB_TYPE` instead of retrying indefinitely.

## Phase 3C (v1-scoped): Admin JSONL Chunk Ingestion & Validation
- **Decision:** v1 Manual JSONL Ingestion over Automated OCR Pipeline.
  - For v1 MVP, PyMuPDF/OCR-based automated chapter/chunk detection is replaced by admin JSONL file ingestion containing structured chunk objects and expected questions.
- **Decision:** Mandatory Firestore 4-Level Ancestor Chain Verification.
  - Every row's `board_id`, `class_id`, `subject_id`, `chapter_id` must exist and have `active == True` in Firestore (`boards/{board_id}/classes/{class_id}/subjects/{subject_id}/chapters/{chapter_id}`). If Firestore client is unavailable or any level is inactive/non-existent, the job fails loudly with `RuntimeError`.
- **Decision:** Single 'Building' Corpus Version Accumulation per Subject Scope.
  - Multiple chapter JSONL uploads for the same subject (`board_id`, `class_id`, `subject_id`) accumulate into one pre-activation corpus version. A same-configuration `qa_ready` snapshot is reopened as `building` for the next chapter and any prior QA approval is invalidated; active snapshots remain immutable and require the draft flow.
- **Decision:** Parent Corpora Locking Hierarchy (`FOR UPDATE`).
  - To prevent check-then-act race conditions between concurrent worker threads, `get_or_create_building_corpus_version` performs `INSERT INTO rag_corpora ... ON CONFLICT DO UPDATE ... RETURNING id` and locks the parent `rag_corpora` row (`SELECT id FROM rag_corpora WHERE id = $1 FOR UPDATE`) before querying/creating `rag_corpus_versions`.
- **Decision:** Document-Level Atomic Replacement & Count Reconciliation.
  - Re-uploading a chapter replaces all chunks for that `document_version_id` inside a transaction. `expected_chunk_count` is updated by delta (`GREATEST(0, expected_chunk_count + delta)`), and `embedded_chunk_count` is reconciled directly from non-null embedding rows (`COUNT(*) WHERE embedding IS NOT NULL`).
- **Decision:** Pre-Embedding Storage Support.
  - Migration `0003b_jsonl_schema_adjustments.sql` dropped `NOT NULL` on `chunk_expected_questions.embedding` to allow storing expected question strings prior to Phase 3D vector embedding generation.
- **Decision:** Audit Log Content Sanitization & Payload Isolation.
  - Audit repository records actor IDs, job IDs, scope metadata, outcomes, and SHA256 source hashes—never raw JSONL content, sensitive chunk texts, or expected question strings.
- **Decision:** BAAI/bge-base-en-v1.5 Token Counting Model Choice.
  - Configured `EMBEDDING_MODEL` as `BAAI/bge-base-en-v1.5` with `EMBEDDING_DIM = 768`, strictly matching the PostgreSQL `vector(768)` database schema across all RAG tables. Token counting loads only Hugging Face `AutoTokenizer` for `BAAI/bge-base-en-v1.5`, avoiding heavy model tensor loading.

## Phase 3D: Embeddings and Corpus Completeness
- **Decision:** Pinned BGE CLS embeddings.
  - Bulk embeddings use only `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, CLS pooling (`last_hidden_state[:, 0]`), and L2 normalization. The version fingerprint covers the model, revision, dimensions, normalization, query instruction, and input formats.
- **Decision:** Atomic durable stage chaining.
  - JSONL ingestion initially queues only chunk embedding. A worker atomically marks a completed stage succeeded and enqueues the next (`chunks → questions → completeness`), preserving retries without allowing readiness to race embedding work.
- **Decision:** Explicit worker ownership.
  - `local_admin` alone owns JSONL and bulk embedding jobs. `railway_public` owns no durable job types until a tested bounded query-embedding path is introduced in Phase 3E.

## Phase 3E: Scoped Retrieval
- **Decision:** Use exact cosine distance for both vector channels.
  - BGE vectors are already L2-normalized by Phase 3D. Exact pgvector `<=>` keeps chunk and individual expected-question retrieval on the same metric without an HNSW index at MVP scale.
- **Decision:** Scope all retrieval in SQL to the active corpus version.
  - Dense, expected-question, lexical, and optional chapter validation queries join corpus scope records and require board/class/subject plus the one active corpus version. Superseded, building, hidden, and other-scope content cannot be selected and expected-question rows resolve only to their parent chunk citation.
- **Decision:** Deterministic rank-only RRF with inspectable contributions.
  - RRF (`k=60`) fuses channel ranks with stable citation-ID tie-breaking. Returned DTOs contain safe citations and channel/rank contributions only; cosine distances, lexical scores, and numeric RRF weights remain internal and are never confidence or probability values.
- **Decision:** Approved top-parent evidence policy.
  - `none` means no scoped fused results. `strong` requires the top fused parent chunk to have at least two distinct retrieval channels ranked 1, 2, or 3; otherwise non-empty results are `weak`. Multiple expected-question matches for one parent are deduplicated before contiguous parent ranks are assigned, and can never count as two channels.
- **Decision:** Query embeddings remain bounded on-demand work.
  - The active corpus's stored BGE fingerprint is verified before inference, and `embed_queries` runs off the event loop. No `query_embedding` durable job or Railway worker ownership was added.

## Phase 3F: Local Admin QA, Visual Metadata, and Transactional Activation
- **Decision:** Keep visual assets server-only and embed reviewed metadata only.
  - JSONL visuals are direct children of chunks and persist logical IDs, type, title, description, review/display state, and a server-only Google Drive key. Imports default to `pending`/`llm_decide`; only approved title/description enter a parent chunk's deterministic embedding input. Drive keys, IDs, URLs, and bytes never enter embeddings, audits, logs, or browser DTOs.
- **Decision:** Make active snapshots immutable and edits targeted.
  - Expected-question and visual edits are allowed only for `building`/`qa_ready` versions. An active version must first be cloned into a `building` draft that records its source. A question edit invalidates only that question; a visual title/description/review change invalidates only its parent chunk; display-policy-only edits do not re-embed. All of these changes invalidate current QA approval.
- **Decision:** Keep local RAG administration unreachable from public deployment.
  - `ADMIN_PANEL_ENABLED` is the first gate for local-admin pages and BFF routes. Writes then require the existing admin session, same-origin check, and CSRF check; the Python control plane independently requires a short-lived signed internal JWT with `admin=true`. The Drive preview proxy is local-admin gated and streams only PNG/JPEG/WebP/GIF with private no-store headers.
- **Decision:** Revalidate activation state inside the lock transaction.
  - Activation and rollback lock the corpus before its version rows, recheck status/scope, exact vector provenance/counters/jobs, current QA approval, and eligible visual storage, then switch the single active version and write a sanitized audit record atomically. Active version ID/configuration resolution uses a five-minute Redis cache keyed by a hashed board/class/subject scope. The cache is invalidated after activation commits and fails open to PostgreSQL without caching questions, vectors, or results.
- **Decision:** Preserve migration portability without changing applied history.
  - `0003c_extensions_schema.sql` creates the standard PostgreSQL `extensions` schema before already-applied pgcrypto setup, and `0003e_pgcrypto_digest_compat.sql` provides the safe `public.digest(text,text)` compatibility wrapper needed by later migrations when `extensions` is outside `search_path`.

## Phase 3F extension: Paired Import Audit and Ingestion Boundary

- **Decision:** Keep paired upload/cropping and private Drive access in the local web BFF.
  - The service receives no Word document or image bytes. It accepts only authenticated internal audit events and the already-established signed JSONL ingestion request; its existing hierarchy validator remains the authority for active Firebase scope.
- **Decision:** Persist non-sensitive retry state only.
  - `0007_paired_chapter_import_audit.sql` stores import/asset hashes, counts, status, optional job ID, and stable error code. RLS and revoked client grants keep this operational state service-only, while excluding storage keys, identifiers, URLs, bytes, raw JSONL, and enriched JSONL.
- **Decision:** Reuse normal ingestion and QA instead of a parallel corpus workflow.
  - The paired flow cannot auto-create catalogue hierarchy or activate a corpus. Its visuals enter the ordinary `pending` review state with `llm_decide` as the eventual display policy; local workers own JSONL/embedding jobs and Railway-public owns none. No paid OCR or LLM/vision service is used.


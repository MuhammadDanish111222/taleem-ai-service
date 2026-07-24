# Taleem AI Service - Key Decisions & Architectural Changes

This document logs significant architectural decisions and changes made for the Python AI microservice.

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
  - Multiple chapter JSONL uploads for the same subject (`board_id`, `class_id`, `subject_id`) accumulate chunks into a single `building` corpus version.
- **Decision:** Parent Corpora Locking Hierarchy (`FOR UPDATE`).
  - To prevent check-then-act race conditions between concurrent worker threads, `get_or_create_building_corpus_version` performs `INSERT INTO rag_corpora ... ON CONFLICT DO UPDATE ... RETURNING id` and locks the parent `rag_corpora` row (`SELECT id FROM rag_corpora WHERE id = $1 FOR UPDATE`) before querying/creating `rag_corpus_versions`.
- **Decision:** Document-Level Atomic Replacement & Count Reconciliation.
  - Re-uploading a chapter replaces all chunks for that `document_version_id` inside a transaction. `expected_chunk_count` is updated by delta (`GREATEST(0, expected_chunk_count + delta)`), and `embedded_chunk_count` is reconciled directly from non-null embedding rows (`COUNT(*) WHERE embedding IS NOT NULL`).
- **Decision:** Pre-Embedding Storage Support.
  - Migration `0003b_jsonl_schema_adjustments.sql` dropped `NOT NULL` on `chunk_expected_questions.embedding` to allow storing expected question strings prior to Phase 3D vector embedding generation.


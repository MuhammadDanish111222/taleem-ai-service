# Taleem AI Service - Progress Log

This document serves as a persistent record of the progress made on the Python-based AI microservice.

## Phase 0: Initial Setup
- **Status:** Completed
- **Details:** Initialized a Python backend using FastAPI, configured environments, and prepared the repository for Phase 3 (AI integrations).

## Module 1 Compliance: Authentication & Internal JWT Security
- **Status:** Completed
- **Details:**
  - Implemented asymmetric RS256 Internal JWT verification (`app/core/internal_auth.py`) using `pyjwt` and `cryptography`.
  - Configured 60-second JWT expiration window and strict audience (`aud: "taleem-ai-service"`) / issuer (`iss: "taleem-web"`) validation.
  - Built Redis-backed JTI replay protection (`jti` nonce store) to reject replayed tokens.
  - Implemented FastAPI security dependencies (`require_internal_jwt`, `require_admin_jwt`).
  - Added health endpoint (`/api/v1/health`) and CI workflow (`.github/workflows/taleem-ai-service-ci.yml`).

## Phase 3A: RAG Foundation & Database Schema
- **Status:** Completed
- **Details:**
  - Configured dependencies: `asyncpg` and `pgvector` added to `pyproject.toml`.
  - Built migration scripts: `migrations/0001_platform_core.sql`, `migrations/0002_rag_schema.sql`, `migrations/0003_security_grants.sql`.
  - Built lightweight migration runner `app/db/migrator.py` and connection pool lifecycle `app/db/pool.py`.
  - Created typed Asyncpg repository modules: `JobRepository`, `RagRepository`, `AIRequestRepository`, and `AuditRepository`.
  - Applied schema-level active version uniqueness constraint, CHECK constraints on all status/progress/count fields, ON DELETE CASCADE/SET NULL foreign keys, and RLS deny-by-default grants with PL/pgSQL role protection.
  - Documented complete database architecture in `docs/architecture.md`.
- **Verification Performed:**
  - Executed automated integration test suite (`tests/test_db_schema_rls.py`, `tests/test_repositories.py`) against PostgreSQL 17 + pgvector.
  - Verified 100% test pass rate across 3 consecutive pytest runs.

## Phase 3B: Cross-Repository Internal Auth & Durable Worker Runtime
- **Status:** Completed
- **Details:**
  - **Internal Auth Audit & Enhancements**: Audited `app/core/internal_auth.py` and added strict mandatory claim validation (`uid`, `admin`, `feature`, `request_id`, `jti`, `iat`, `exp`), strict timestamp constraints (`exp - iat <= 60`s, `exp > iat`), and atomic Redis `SET NX EX` replay prevention (`set(key, "1", nx=True, ex=60)`).
  - **Job Queue Service**: Created `app/services/jobs/queue.py` as a service-layer wrapper around `JobRepository`.
  - **Strict Lock Ownership & Row Count Verification**: Updated all mutating repository queries (`update_heartbeat`, `update_progress`, `complete_job`, `fail_job`) to enforce `locked_by = worker_id` and check affected row counts.
  - **Worker Runtime Process**: Created `app/workers/main.py` standalone worker process running via `python -m app.workers.main`. Features `FOR UPDATE SKIP LOCKED` polling, background heartbeating, graceful shutdown without premature lease release, deterministic stale recovery, and immediate failure for unsupported job types.
  - **Deterministic Stale Lease Recovery**: `recover_stale_jobs` resets stale jobs with attempts remaining to `retry_wait`, and terminally fails exhausted jobs with `STALE_LEASE_EXHAUSTED`.
- **Verification Performed:**
  - Cross-repo integration test (`test_cross_repo_jwt_integration.py`) verifying TypeScript `signInternalJwt` token output passes Python `verify_internal_jwt`.
  - Concurrency test with independent Postgres connections verifying `FOR UPDATE SKIP LOCKED`.
  - Worker crash recovery test verifying idempotent side-effects with exactly 1 output record.
  - Protected endpoint tests verifying 401 on unsigned/malformed requests and 200 on valid internal JWT.

## Phase 3C (v1-scoped): Admin JSONL Chunk Ingestion & Validation
- **Status:** Completed
- **Details:**
  - **Schema Adjustments**: Executed migration `0003b_jsonl_schema_adjustments.sql`, dropping `NOT NULL` on `chunk_expected_questions.embedding` and adding `content_type`, `metadata`, `content_hash`, `language`, and `token_count` to `rag_chunks`.
  - **Validation Module**: Built `app/services/ingestion/jsonl_chunks.py` providing SHA256 content hashing (`compute_content_hash`), word token counting (`count_tokens`), line-by-line schema validation, strict `page_range` tuple checks (`null` or `[start_page, end_page]`), and sanitized error logging (raw text excluded).
  - **Firestore Hierarchy Verification**: Implemented 4-level ancestor chain check (`check_firestore_hierarchy`) verifying document existence and `active == True` across `boards`, `classes`, `subjects`, and `chapters` with in-memory batch caching. Requires live Firestore client or raises loud `RuntimeError` if unavailable.
  - **Repository Atomic Operations**: Updated `RagRepository` with `get_or_create_building_corpus_version` (holding parent `rag_corpora` row lock via `ON CONFLICT DO UPDATE` + `FOR UPDATE`) and `replace_chapter_chunks` (locking corpus version `FOR UPDATE`, deleting old chunks/questions via CASCADE, inserting new rows, updating `expected_chunk_count` by delta, and reconciling `embedded_chunk_count`).
  - **Worker Job Handler**: Registered `jsonl_ingest` handler in `app/workers/handlers/jsonl_ingest.py` and `app/workers/main.py`.
  - **Internal Endpoint**: Exposed `POST /api/v1/internal/ingest/jsonl` in `app/api/v1/internal.py` protected by RS256 internal JWT requiring admin privileges.
- **Verification Performed:**
  - `tests/ingestion/test_jsonl_validation.py` verifying field mapping, error sanitization, and hierarchy rejection.
  - `tests/ingestion/test_jsonl_ingestion_job.py` verifying job execution, atomic chapter re-upload replacement, multi-chapter corpus accumulation, status locks, first-insert race prevention, explicit idempotency-key replay zero-row creation, handler direct re-execution idempotency, and loud failure when Firestore is unavailable.
  - 100% test pass rate across 3 consecutive pytest runs (59 passed in 98s).


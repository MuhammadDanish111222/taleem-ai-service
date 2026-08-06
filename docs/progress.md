# Taleem AI Service - Progress Log

## Module 4 — Long-answer completeness hardening

- **Implemented:** Short grounded answers now send one complete highest-ranked subtopic. Long answers send that complete subtopic plus at most one independently supported second subtopic from a five-result anchor window. Both are capped at twelve chunks; short and long character limits remain 12,000 and 24,000 respectively.
- **Structure and enrichment:** The strict answer contract now supports semantic headings and bullet lists. Supplied same-topic enrichment is retained under `Additional textbook knowledge (optional)` while essential answer points cannot be demoted to optional material.
- **Visuals:** Every approved `always` or `llm_decide` visual linked to the selected long-answer topics is returned. DeepSeek still receives metadata only, and missing visual references are appended deterministically after allowlist validation.
- **Verification:** Full service suite: `185 passed`, `3` intentionally gated tests skipped. The topic-aware focused database suite passes `35` tests. The most recent full Firestore-emulator web suite remains `296 passed`, `1` intentionally gated test skipped; no web code changed in this update. Ruff, focused retrieval/answer tests, and service diff checks pass.

This document serves as a persistent record of the progress made on the Python-based AI microservice.

## Module 4 — Ask a Question, Run 2 staging delivery

- **Status:** Complete. The public Ask path, real Supabase/Upstash/DeepSeek integration, Railway/Vercel deployment, approved/grounded/general staging, CI exit checks, and public WhatsApp support setting all pass.
- **Service contract:** Single Ask accepts typed English text only, `short|long`, and backend-fixed `exam_style`. `approved_bank`, `syllabus_grounded`, and `general_knowledge` remain separate stored and returned sources. A grounded answer now requires at least one verified retrieved citation; retrieved rank agreement alone cannot make an uncited answer grounded.
- **Fallback integrity:** Both strong and weak non-empty retrieval results are supplied as bounded text evidence. If DeepSeek uses that evidence it must cite an allowed chunk. If it returns a reference-free answer and the typed policy permits fallback, the backend labels it General AI and strips the textbook allowlists; otherwise it returns an honest no-answer. General AI can never contain textbook citations or visuals.
- **Trusted and candidate pools:** One cited States of Matter staging candidate was reviewed and promoted through the audited approval workflow as an `easy`, 2-mark immutable revision. A subsequent exact Ask returned `approved_bank` with zero provider attempts. Two staging-only generated artifacts were audit-rejected, not deleted. Exact approved questions and variations remain enabled; semantic reuse remains intentionally disabled.
- **Real configuration:** Supabase records `0009_module4_ask_foundation.sql`. The global typed source policy enables General AI while semantic reuse stays disabled. TLS Upstash authentication and `PING` pass; production logs show two successful Ask requests with zero Redis/JTI/PostgreSQL-fallback or authentication events. Retention preview found zero currently eligible rows, so no retention deletion ran.
- **Verification:** A fresh PostgreSQL 17 + pgvector migration sequence and the complete service suite pass (`179 passed`, `3` intentionally gated live tests skipped, one dependency warning). Ruff, Ruff format, compileall, focused real-Supabase transaction tests, health, and source-integrity staging pass. GitHub Actions run `30993218596` is green across Python, real PostgreSQL/Redis worker, and worker smoke jobs.
- **Deployment:** Railway deployment `5a64df3f-87a0-46b8-a03a-3d2404d89164` is healthy with the bounded PyTorch-free ONNX query runtime. Six controlled DeepSeek staging calls were used while diagnosing the memory, provider-shape, and source-classification defects; the final approved-bank replay used no provider call. Remote `main` is at `da3eabc`.
- **Support setting:** `academy_settings/default` now exists in real Firestore with the owner-provided WhatsApp support configuration. The public limit-exceeded action is enabled without exposing any private credentials.
- **Exit gate:** Complete. No owner-laptop dependency remains in the public Ask path.

## Module 4 — Ask a Question, Run 1 of 2

- **Status:** Backend foundation implemented and verified; Module 4 is not complete.
- **Implemented:** Forward-only `0009_module4_ask_foundation.sql`; atomic Redis/PostgreSQL usage and JTI fallback; scoped versioned prompts; strict text-only DeepSeek adapter; approved-bank-first orchestration; pending candidate persistence/approval linkage; semantic reuse boundary disabled by default; deterministic citation/visual validation; local-admin prompt/candidate/bank operations; local question-bank embedding jobs.
- **Verification:** Fresh PostgreSQL 17 + pgvector migration sequence and legacy upgrade fixture passed. Ruff, Ruff format, compileall, and the full service suite passed (`166` tests; one dependency deprecation warning).
- **Run 2:** Build the student Ask UI and local prompt/candidate/bank interfaces, configure real Upstash/DeepSeek secrets, apply `0009` to the intended staging database, deploy, run real staging checks, and complete final documentation. No real DeepSeek request, deployment, commit, push, or real Supabase migration occurred in Run 1.

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
  - **Validation Module**: Built `app/services/ingestion/jsonl_chunks.py` providing SHA256 content hashing (`compute_content_hash`), line-by-line schema validation, strict `page_range` tuple checks (`null` or `[start_page, end_page]`), and sanitized error logging (raw text excluded).
  - **Firestore Hierarchy Verification**: Implemented 4-level ancestor chain check (`check_firestore_hierarchy`) verifying document existence and `active == True` across `boards`, `classes`, `subjects`, and `chapters` with in-memory batch caching. Requires live Firestore client or raises loud `RuntimeError` if unavailable.
  - **Repository Atomic Operations**: Updated `RagRepository` with `get_or_create_building_corpus_version` (holding parent `rag_corpora` row lock via `ON CONFLICT DO UPDATE` + `FOR UPDATE`) and `replace_chapter_chunks` (locking corpus version `FOR UPDATE`, deleting old chunks/questions via CASCADE, inserting new rows, updating `expected_chunk_count` by delta, and reconciling `embedded_chunk_count`).
  - **Worker Job Handler**: Registered `jsonl_ingest` handler in `app/workers/handlers/jsonl_ingest.py` and `app/workers/main.py`.
  - **Internal Endpoint**: Exposed `POST /api/v1/internal/ingest/jsonl` in `app/api/v1/internal.py` protected by RS256 internal JWT requiring admin privileges.
- **Verification Performed:**
  - `tests/ingestion/test_jsonl_validation.py` verifying field mapping, error sanitization, and hierarchy rejection.
  - `tests/ingestion/test_jsonl_ingestion_job.py` verifying job execution, atomic chapter re-upload replacement, multi-chapter corpus accumulation, status locks, first-insert race prevention, explicit idempotency-key replay zero-row creation, handler direct re-execution idempotency, and loud failure when Firestore is unavailable.
  - 100% test pass rate across 3 consecutive pytest runs (59 passed in 98s).

## Phase 3C follow-up: security, validation and audit hardening
- JSONL submission now validates one complete board/class/subject/chapter scope before enqueueing; mixed-scope uploads fail atomically with `JSONL_SCOPE_MISMATCH`.
- Every row is rechecked against the Firestore ancestor chain. Empty input, blank chunk text, invalid expected-question arrays, blank/normalized-duplicate questions, and duplicate chapter chunk order are rejected without storing source text in errors or audits.
- Accepted job creation and its PostgreSQL audit record share one transaction. Rejected actions store only actor, request/job identifiers, scope, outcome, stable error code, and hashes—never raw JSONL, chunk text, expected-question text, secrets, or stack traces.
- In-memory unit test (`test_audit_repository_sanitization_without_database`) verifies payload structure and sanitization without requiring PostgreSQL connection.
- New JSONL token counts use the configured embedding tokenizer only (default `BAAI/bge-base-en-v1.5` with 768 dimensions); existing stored counts are unchanged and require chapter re-ingestion to update.
- Each expected question remains an individual `chunk_expected_questions` row with its own future embedding slot (`vector(768)`). Phase 3D remains incomplete and must embed both chunk text and every expected-question row.
- Pull Request #3 merged into `main` (`03bd1aa`).
- Executed full GitHub Actions CI run on `main` (`30124762501`) with 100% green pass rate across Ruff, 68 pytest suite, PostgreSQL/Redis worker tests, and worker startup smoke test.

## Phase 3D: Embeddings and Corpus Completeness
- **Status:** Completed
- **Details:** Added the pinned `BAAI/bge-base-en-v1.5` provider at immutable revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, using CLS pooling and L2-normalized 768-dimensional vectors. Each chunk and expected question has independent vector provenance and input/configuration fingerprints.
- **Safety:** `WORKER_MODE=local_admin` is required for JSONL and bulk embedding jobs. `railway_public` owns no durable job types in Phase 3D. Stages are durably chained as chunks → questions → completeness through an atomic complete-and-enqueue operation.
- **Database:** Migration `0004_phase3d_embeddings.sql` adds expected-question hashes/counters, embedding lifecycle fields/indexes, and a guarded readiness/activation trigger compatible with Supabase PostgreSQL + pgvector.
- **Verification Performed:** Applied all migrations from scratch to a disposable PostgreSQL 17 + pgvector container; ran `ruff check app tests`, the complete `pytest tests -q` suite (`87 passed`), and the worker smoke test.

## Phase 3E: Scoped Retrieval Mechanics
- **Status:** Completed.
- **Details:** Added an internal retrieval service with exact active-corpus SQL scoping for dense chunk vectors, individual expected-question vectors resolved to parent chunks, and PostgreSQL `'simple'` full-text search. Expected-question matches are parent-deduplicated before contiguous parent ranking. Queries use the active corpus version's stored, fingerprint-checked BGE configuration and run query inference off the event loop. Deterministic RRF (`k=60`) returns safe citations plus channel/rank contributions without exposing distances, lexical scores, vectors, RRF weights, storage identifiers, or expected-question IDs. JSONL results safely carry no visuals.
- **Evidence policy:** `none` means no scoped fused results. `strong` requires the top fused parent chunk to have two or more distinct channels, each ranked 1--3; non-empty results otherwise remain `weak`. A single channel and duplicate expected-question rows for one parent never satisfy strong evidence.
- **Verification Performed:** On a fresh disposable PostgreSQL 17 + pgvector container, applied migrations through `0004_phase3d_embeddings.sql`; ran `ruff check app tests`, Phase 3E retrieval tests (`6 passed`), and the full suite (`93 passed`, one existing dependency deprecation warning). Railway-public durable job ownership remains empty.

## Phase 3F: Local Admin QA, Visual/Expected-Question Editing, and Transactional Corpus Activation
- **Status:** Completed.
- **Details:** Extended JSONL chunks with optional multi-visual metadata, stored as direct `rag_chunks -> rag_visuals` rows with server-only Google Drive keys. Reviewed visual title/description is embedded deterministically with its parent chunk; pending/rejected visuals are excluded. Local admin can inspect durable jobs/version state/chunks/questions/visuals/audits, run named-version draft QA, edit draft questions/visuals, preview allowlisted image MIME types through the server-side Drive provider, approve QA, clone active snapshots, activate, and roll back.
- **Safety:** All RAG administration is local-admin-only. `ADMIN_PANEL_ENABLED` runs before session, CSRF, parsing, or internal-service work; writes require admin session, same-origin, and CSRF checks; the Python endpoint independently requires signed admin JWTs. Browser DTOs, errors, and audits omit vectors, storage keys, Drive IDs/URLs, provider data, and secrets. `railway_public` still owns no durable bulk jobs.
- **Activation:** One transaction locks `rag_corpora` then its versions, rechecks persisted readiness/provenance, QA approval, scope, and displayable visual storage, supersedes the old active snapshot, activates exactly one target, and audits activation/rollback. Active version ID/configuration resolution uses a five-minute Redis cache; activation and rollback invalidate the scope key only after commit, and Redis failures fall back to PostgreSQL.
- **Migration and verification:** Applied the forward-only portability helpers `0003c_extensions_schema.sql` and `0003e_pgcrypto_digest_compat.sql` and Phase 3F migration `0005_phase3f_local_admin.sql` on the configured Supabase connection without destructive operations. A fresh disposable PostgreSQL 17 + pgvector sequence applied all migrations. Rollback-safe Supabase integration tests cover visual import/draft cloning, targeted edits, activation rejection, named QA isolation, rollback, and concurrency; only the narrowly scoped concurrency fixture was committed, then deleted by its exact generated corpus/audit IDs. Final checks: `105 passed` service tests, `55 passed` emulator-aware web tests, focused Phase 3F DB/security tests, Ruff, Python compilation, TypeScript typecheck, web lint, and `git diff --check`.

## Phase 3F extension: Paired JSONL + Visual Extracts DOCX Import

- **Status:** Completed.
- **Details:** Added an internal-admin audit endpoint and forward-only `0007_paired_chapter_import_audit.sql` state table for the web BFF's paired import. The established JSONL ingestion/worker flow remains unchanged; the BFF sends it only internally enriched records after private Drive visual upload.
- **Safety:** The audit state stores hashes/counts and stable status only. It is RLS-enabled with public, anon, and authenticated privileges revoked. Direct service calls still require a signed admin internal JWT. Firebase hierarchy validation remains the prerequisite for real ingestion; no catalogue is created by the importer.
- **Verification:** Applied `0007` to the configured Supabase database after confirming it was the only pending migration; verified the migration record, table, RLS, revoked client privileges, and unchanged corpus/chunk/job counts. A fresh pgvector migration sequence and complete service suite passed (`112`).


# Taleem AI Service

## Runtime

Python 3.12 is the supported runtime (`.python-version`, `pyproject.toml`, and CI agree). Railway hosts the FastAPI API and durable worker. It receives only BFF-issued internal JWTs; browsers do not call this service directly.

For JSONL token counts, configure `EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION`, and `EMBEDDING_DIM` (the defaults are `voyage-4-lite`, revision `voyage-4-lite-512-v1`, and `512`).

FastAPI AI microservice for the Taleem AI platform (`taleem-ai-service`).

## System Overview
- **RAG Engine & Worker Runtime**: PostgreSQL 17 + `pgvector`, Asyncpg connection pool, RLS deny-by-default grants, durable `job_queue` worker loop with background heartbeating, atomic lease locking (`FOR UPDATE SKIP LOCKED`), and deterministic crash/retry recovery.
- **Admin JSONL Chunk Ingestion (Phase 3C v1-scoped)**: Line-by-line validation, 4-level Firestore ancestor chain verification (`boards` -> `classes` -> `subjects` -> `chapters`), SHA256 content hashing, word token counting, atomic `replace_chapter_chunks` with `GREATEST(0, expected_chunk_count + delta)` and `embedded_chunk_count` non-null count reconciliation.
- **Embeddings and Retrieval (Phases 3D–3E)**: Voyage-4-lite `halfvec(512)` embeddings with per-row provenance and completeness gates; exact scoped dense, expected-question, and lexical retrieval fused with deterministic rank-only RRF.
- **Local RAG Administration (Phase 3F)**: Local-admin-only corpus inspection, draft QA, targeted expected-question/visual editing, controlled Google Drive image previews, audited activation, and rollback. Railway-public owns no durable bulk embedding jobs.
- **Cross-Repository Security**: Asymmetric RS256 Internal JWT authentication (`aud: "taleem-ai-service"`, `iss: "taleem-web"`, strict 60s TTL window, mandatory claim validation, Redis JTI replay prevention).
- **Module 4 Single Ask**: Typed-English `short|long` questions resolve against the approved bank first, then active textbook evidence, and finally the explicitly enabled General AI fallback. Student generation is retained as a pending candidate until a local administrator approves a corrected immutable bank revision.
- **Continuity and Safety**: Redis provides the normal atomic quota/JTI/cache path, PostgreSQL preserves limits and replay safety during a controlled Redis outage, and identity-scoped request UUIDs make retries idempotent.
- **Module 5 Run 1 (dark)**: private temporary Multiple Ask sessions, immutable curriculum scope, durable parent/item records, and Railway canonical validation-before-quota are present but feature-gated. No OCR, answer generation, or student surface is enabled; see [`docs/module5_run1_multiple_ask.md`](docs/module5_run1_multiple_ask.md).
- **Module 5 Run 2**: Google Gemini Vision OCR API, durable temporary normalized source text, deterministic ordered question extraction (up to 60 items), correction/resume, and polling APIs are present. Objective decimal numbering/MCQs and subjective Roman or lettered subparts remain individually ordered. Zero local model weights or binaries (Tesseract/Torch/Transformers/ONNX) are installed in production.

Module 4 deployment ownership and verification procedures are documented in [`docs/deployment_runbook.md`](docs/deployment_runbook.md). Module 4 is complete: real Supabase, shared Redis, provider, staging, deployment, CI, and the public WhatsApp support setting were verified.
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env

# 3. Run FastAPI application locally
uv run uvicorn app.main:app --reload

# 4. Execute automated test suite
uv run pytest -o pythonpath=.
```


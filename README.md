# Taleem AI Service

FastAPI AI microservice for the Taleem AI platform (`taleem-ai-service`).

## System Overview
- **RAG Engine & Worker Runtime**: PostgreSQL 17 + `pgvector`, Asyncpg connection pool, RLS deny-by-default grants, durable `job_queue` worker loop with background heartbeating, atomic lease locking (`FOR UPDATE SKIP LOCKED`), and deterministic crash/retry recovery.
- **Admin JSONL Chunk Ingestion (Phase 3C v1-scoped)**: Line-by-line validation, 4-level Firestore ancestor chain verification (`boards` -> `classes` -> `subjects` -> `chapters`), SHA256 content hashing, word token counting, atomic `replace_chapter_chunks` with `GREATEST(0, expected_chunk_count + delta)` and `embedded_chunk_count` non-null count reconciliation.
- **Cross-Repository Security**: Asymmetric RS256 Internal JWT authentication (`aud: "taleem-ai-service"`, `iss: "taleem-web"`, strict 60s TTL window, mandatory claim validation, Redis JTI replay prevention).

## Getting Started

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env

# 3. Run FastAPI application locally
uv run uvicorn app.main:app --reload

# 4. Execute automated test suite
uv run pytest -o pythonpath=.
```


# Module 4 AI Service Deployment Runbook

## Ownership

Railway owns the public FastAPI API, shared-cache access, lightweight provider work, health/readiness endpoints, and `WORKER_MODE=railway_public`. The owner's trusted machine uses `WORKER_MODE=local_admin` for local-only bulk embedding and administration. Browsers never call Railway directly.

## Pre-deployment gate

1. Run every migration through `0009` against a fresh PostgreSQL 17 + pgvector database and run the legacy-upgrade fixture.
2. On real Supabase, inspect `schema_migrations`, migration order, live schema, protected-table counts, RLS, and grants before mutation.
3. Apply only missing migration files, in order, using the existing per-file transaction runner. Never edit an applied migration or repair disagreement by deleting data.
4. Preview candidate retention. Do not delete eligible real rows without explicit authorization.
5. Pass Ruff, Ruff format, compileall, the complete pytest suite, startup/health smoke, and repository secret/artifact inspection.

## Railway server-only configuration

Configure `DATABASE_URL`, a TLS shared `REDIS_URL`, independent `USAGE_UID_HMAC_SECRET` and `INTERNAL_JTI_HMAC_SECRET`, `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `INTERNAL_JWT_PUBLIC_KEYS_JSON`, Firebase Admin values, and the pinned embedding configuration. Use `deepseek-v4-flash` unless a later verified model decision changes it. Keep provider timeout, retries, input characters, and output tokens bounded.

Never configure these values in browser variables or return them in DTOs/logs. Railway must not configure `WORKER_MODE=local_admin` or own local-admin bulk jobs.

## Shared Redis verification

- Confirm TLS and atomic Lua reservation.
- Confirm expiry is the next `Asia/Karachi` midnight.
- Confirm prompt/settings and active-corpus invalidation is shared between Railway and the local admin environment.
- Confirm JTI replay rejection and that keys contain no raw UID, question, answer, profile, Drive ID, or storage key.
- Exercise outage behavior only with controlled fault injection or an isolated instance: PostgreSQL must preserve quota, idempotency, and hashed replay protection, with sanitized fallback events.

## Provider verification

Run fake-provider tests first. Verify the configured model, non-thinking JSON output, text-only request path, stable prompt prefix, token bounds, timeout, retries, and available API balance before an authorized paid call. Record only sanitized provider/model/token/latency/status evidence.

## Post-deployment checks

Verify `/api/v1/health` and readiness, then run the approved-bank, grounded, General AI, disabled-fallback, quota/concurrency, idempotency, prompt rollback, candidate approval/reuse, visual, and security staging scenarios. Approved reuse must perform no embedding, retrieval, or provider call. General AI must have no textbook citations or visuals.

Module 4 completion was verified after the real staging path, main-branch deployment, CI, and the public WhatsApp support setting were confirmed. Keep the same checks for future changes.

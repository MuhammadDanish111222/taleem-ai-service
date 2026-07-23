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

# Taleem AI Service Roadmap

## Module 4: Ask a Question

- [x] Run 1: backend/BFF contracts, quota fallback, prompt/provider boundary, approved bank, generation/candidate orchestration, validation, admin service operations, and automated verification.
- [ ] Run 2 exit gate: implementation, real Supabase/Upstash/DeepSeek configuration, approved/grounded/general staging, deployment, push, and CI are verified. Only the owner-provided public WhatsApp support setting remains before this item may be checked.

- [x] Phase 0: Initial Repository Setup (Python, FastAPI, UV)
- [x] Phase 3A: RAG Foundation & Database Schema (PostgreSQL 17, pgvector, Asyncpg, RLS, Durable Jobs)
- [x] Phase 3B: Cross-Repository Internal Auth & Durable Worker Runtime (Internal RS256 JWT, Worker Loop, Lease Recovery)
- [x] Phase 3C (v1-scoped): Admin JSONL Chunk Ingestion & Validation
- [x] Phase 3D: Embeddings and Corpus Completeness — embed every chunk and every individual `chunk_expected_questions` row; require complete, matching 768-dimensional vectors before `qa_ready`.
- [x] Phase 3E: Scoped three-channel retrieval, contiguous expected-question parent ranking, deterministic rank fusion, and approved top-parent evidence strength. No answer generation is in scope.
- [x] Phase 3F: Local-only admin QA, visual and expected-question draft editing, controlled Drive image preview, and transactional corpus activation/rollback.
- [x] Phase 3F extension: Paired JSONL + Visual Extracts DOCX import using private Drive assets and the existing local JSONL worker pipeline.
- [ ] Phase 8: Scalability & Performance Tuning

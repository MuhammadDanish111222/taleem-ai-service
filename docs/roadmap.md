# Taleem AI Service Roadmap

- [x] Phase 0: Initial Repository Setup (Python, FastAPI, UV)
- [x] Phase 3A: RAG Foundation & Database Schema (PostgreSQL 17, pgvector, Asyncpg, RLS, Durable Jobs)
- [x] Phase 3B: Cross-Repository Internal Auth & Durable Worker Runtime (Internal RS256 JWT, Worker Loop, Lease Recovery)
- [x] Phase 3C (v1-scoped): Admin JSONL Chunk Ingestion & Validation
- [x] Phase 3D: Embeddings and Corpus Completeness — embed every chunk and every individual `chunk_expected_questions` row; require complete, matching 768-dimensional vectors before `qa_ready`.
- [x] Phase 3E: Scoped three-channel retrieval, contiguous expected-question parent ranking, deterministic rank fusion, and approved top-parent evidence strength. No answer generation is in scope.
- [x] Phase 3F: Local-only admin QA, visual and expected-question draft editing, controlled Drive image preview, and transactional corpus activation/rollback.
- [ ] Phase 8: Scalability & Performance Tuning

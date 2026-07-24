# Taleem AI Service Roadmap

- [x] Phase 0: Initial Repository Setup (Python, FastAPI, UV)
- [x] Phase 3A: RAG Foundation & Database Schema (PostgreSQL 17, pgvector, Asyncpg, RLS, Durable Jobs)
- [x] Phase 3B: Cross-Repository Internal Auth & Durable Worker Runtime (Internal RS256 JWT, Worker Loop, Lease Recovery)
- [x] Phase 3C (v1-scoped): Admin JSONL Chunk Ingestion & Validation
- [ ] Phase 3D: Embedding & Vector Search Service — embed every chunk's `chunk_text` and every individual `chunk_expected_questions` row; track both before marking complete.
- [ ] Phase 3E: RAG Query Engine & Answer Generation
- [ ] Phase 8: Scalability & Performance Tuning

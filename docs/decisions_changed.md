# Taleem AI Service - Key Decisions & Architectural Changes

This document logs significant architectural decisions and changes made for the Python AI microservice.

## Phase 0: Framework & Architecture
- **Decision:** Python + FastAPI over Node.js for AI tasks.
- **Change Details:**
  - While the main web platform is built on Next.js (`taleem-web`), we chose Python and FastAPI for the AI service. This allows us to leverage Python's dominant ecosystem for AI/ML (Langchain, PyTorch, specialized tokenizers).
  - The service is designed as an isolated microservice, decoupling heavy generative AI workloads from the core web platform.

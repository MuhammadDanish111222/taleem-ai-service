import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncpg

from app.core.config import get_settings


async def clean_rag_db():
    settings = get_settings()
    db_url = settings.DATABASE_URL
    print("Connecting to database...")

    conn = await asyncpg.connect(db_url, statement_cache_size=0)
    try:
        async with conn.transaction():
            print("Cleaning question bank revision references...")
            await conn.execute("DELETE FROM question_bank_revision_citations;")
            await conn.execute("DELETE FROM question_bank_revision_visuals;")

            print("Cleaning RAG visuals and expected questions...")
            await conn.execute("DELETE FROM rag_visuals;")
            await conn.execute("DELETE FROM chunk_expected_questions;")

            print("Cleaning RAG chunks and document versions...")
            await conn.execute("DELETE FROM rag_chunks;")
            await conn.execute("DELETE FROM rag_document_versions;")

            print("Cleaning QA approvals and corpus versions...")
            await conn.execute("DELETE FROM rag_corpus_qa_approvals;")
            await conn.execute("DELETE FROM rag_corpus_versions;")

            print("Cleaning RAG ingestion jobs from job_queue...")
            await conn.execute(
                "DELETE FROM job_queue WHERE job_type IN ('jsonl_ingest', 'embed_chunks', 'embed_questions', 'corpus_completeness');"
            )

        print(
            "\nSUCCESS: All RAG chunks, visuals, expected questions, corpus versions, and ingestion jobs have been completely cleaned from the database!"
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(clean_rag_db())

"""Worker Handler for Admin JSONL Chunk Ingestion jobs."""

import json
import logging
from typing import Dict, Any
import asyncpg

from app.services.ingestion.jsonl_chunks import validate_and_parse_jsonl
from app.repositories.rag_repository import RagRepository
from app.core.firebase_admin import get_firebase_app

logger = logging.getLogger(__name__)


async def handle_jsonl_ingest(job: Dict[str, Any], conn: asyncpg.Connection) -> Dict[str, Any]:
    """Worker job handler for processing admin uploaded JSONL chunk batches.

    Orchestrates line-by-line validation, corpus version accumulation, document version
    upserting, and atomic chunk replacement within a database transaction.
    """
    payload = job.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    jsonl_content = payload.get("jsonl_content") or ""
    if not jsonl_content or not isinstance(jsonl_content, str):
        raise ValueError("Missing or empty 'jsonl_content' in job payload.")

    # Initialize Firestore client if Firebase settings are configured
    firestore_db = None
    try:
        app = get_firebase_app()
        if app:
            from firebase_admin import firestore
            firestore_db = firestore.client()
    except Exception as fb_err:
        logger.warning(f"Firestore client not initialized for hierarchy checks: {fb_err}")

    # Step 1: Validate and parse JSONL
    valid_chunks, errors = await validate_and_parse_jsonl(jsonl_content, firestore_db)
    if errors:
        # Sanitize error output: return row numbers + field names + reason codes only
        sanitized_errors_json = json.dumps(errors)
        logger.error(f"JSONL validation failed with {len(errors)} error(s): {sanitized_errors_json}")
        raise ValueError(f"JSONL validation failed: {sanitized_errors_json}")

    if not valid_chunks:
        raise ValueError("JSONL file contained no valid chunk records.")

    # Derive common scope from first valid chunk
    first_chunk = valid_chunks[0]
    board_id = first_chunk["board_id"]
    class_id = first_chunk["class_id"]
    subject_id = first_chunk["subject_id"]
    chapter_id = first_chunk["chapter_id"]

    repo = RagRepository(conn)

    # Step 2: Acquire building corpus version & replace chapter chunks atomically
    async with conn.transaction():
        # Get or create building corpus version for this subject scope
        corpus_ver = await repo.get_or_create_building_corpus_version(board_id, class_id, subject_id)
        corpus_version_id = str(corpus_ver["id"])

        # Create/upsert document version for this chapter
        resource_id = f"jsonl:chapter:{chapter_id}"
        resource_version_id = payload.get("resource_version_id") or "v1"
        pipeline_version = "admin_jsonl_v1"
        doc_title = f"JSONL Chapter Ingestion - {chapter_id}"

        doc_ver = await repo.create_document_version(
            corpus_version_id=corpus_version_id,
            resource_id=resource_id,
            resource_version_id=resource_version_id,
            pipeline_version=pipeline_version,
            doc_title=doc_title,
            total_chunks=len(valid_chunks),
        )
        document_version_id = str(doc_ver["id"])

        # Atomically replace chapter chunks and expected questions
        inserted = await repo.replace_chapter_chunks(
            corpus_version_id=corpus_version_id,
            document_version_id=document_version_id,
            chunks=valid_chunks,
        )

    logger.info(
        f"Successfully ingested {len(inserted)} chunks into corpus version '{corpus_version_id}' "
        f"for chapter '{chapter_id}' (job: {job.get('id')})."
    )

    return {
        "status": "succeeded",
        "corpus_version_id": corpus_version_id,
        "document_version_id": document_version_id,
        "chunks_ingested": len(inserted),
    }

"""Phase 3F local-admin mutations.  This module never serializes vectors or keys."""

from __future__ import annotations

import hashlib
from typing import Any

import asyncpg

from app.repositories.audit_repository import AuditRepository
from app.repositories.rag_repository import RagRepository
from app.services.ingestion.normalization import normalize_expected_question
from app.services.jobs.queue import JobQueueService


class LocalAdminError(ValueError):
    pass


def _visual_hash(title: str, description: str) -> str:
    return hashlib.sha256(" ".join(f"{title.strip()} {description.strip()}".split()).lower().encode()).hexdigest()


class LocalAdminService:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn
        self.repo = RagRepository(conn)
        self.audit = AuditRepository(conn)

    async def _version_for_scope(self, version_id: str, scope: dict[str, str], *, lock: bool = False) -> dict[str, Any]:
        suffix = " FOR UPDATE" if lock else ""
        row = await self.conn.fetchrow(
            f"""SELECT cv.*, c.board_id, c.class_id, c.subject_id
                FROM rag_corpus_versions cv JOIN rag_corpora c ON c.id = cv.corpus_id
                WHERE cv.id = $1::uuid AND c.board_id = $2 AND c.class_id = $3 AND c.subject_id = $4{suffix};""",
            version_id, scope["board_id"], scope["class_id"], scope["subject_id"],
        )
        if not row:
            raise LocalAdminError("CORPUS_VERSION_OUTSIDE_SCOPE")
        return dict(row)

    async def _invalidate_approvals(self, version_id: str, reason: str) -> None:
        await self.conn.execute(
            """UPDATE rag_corpus_qa_approvals SET invalidated_at = NOW(), invalidated_reason = $2
               WHERE corpus_version_id = $1::uuid AND invalidated_at IS NULL;""", version_id, reason
        )

    async def _make_building(self, version: dict[str, Any]) -> None:
        if version["status"] == "active":
            raise LocalAdminError("ACTIVE_VERSION_IMMUTABLE_CREATE_DRAFT_REQUIRED")
        if version["status"] not in {"building", "qa_ready"}:
            raise LocalAdminError("CORPUS_VERSION_NOT_EDITABLE")
        if version["status"] == "qa_ready":
            await self.conn.execute("UPDATE rag_corpus_versions SET status = 'building' WHERE id = $1::uuid", version["id"])

    async def edit_question(self, *, version_id: str, scope: dict[str, str], actor_id: str, request_id: str, question_id: str | None, question_text: str | None, chunk_id: str | None, delete: bool = False) -> dict[str, Any]:
        async with self.conn.transaction():
            version = await self._version_for_scope(version_id, scope, lock=True)
            await self._make_building(version)
            if delete:
                row = await self.conn.fetchrow(
                    """DELETE FROM chunk_expected_questions q USING rag_chunks c
                       WHERE q.id=$1::uuid AND c.id=q.chunk_id AND c.corpus_version_id=$2::uuid
                       RETURNING q.id;""", question_id, version_id)
                if not row:
                    raise LocalAdminError("EXPECTED_QUESTION_NOT_FOUND")
                action = "expected_question_deleted"
                target_id = str(question_id)
            else:
                if not isinstance(question_text, str) or not question_text.strip():
                    raise LocalAdminError("EXPECTED_QUESTION_BLANK")
                text = question_text.strip()
                normal = normalize_expected_question(text)
                digest = hashlib.sha256(normal.encode()).hexdigest()
                if question_id:
                    row = await self.conn.fetchrow(
                        """UPDATE chunk_expected_questions q SET question_text=$3, question_normalized=$4, question_hash=$5,
                               embedding=NULL, embedding_status='pending', embedding_input_hash='', embedding_model='',
                               embedding_revision='', embedding_config_fingerprint='', embedding_started_at=NULL,
                               embedding_completed_at=NULL, embedding_error_code=NULL
                           FROM rag_chunks c WHERE q.id=$1::uuid AND c.id=q.chunk_id AND c.corpus_version_id=$2::uuid
                           RETURNING q.id;""", question_id, version_id, text, normal, digest)
                    action = "expected_question_edited"
                    target_id = str(question_id)
                else:
                    row = await self.conn.fetchrow(
                        """INSERT INTO chunk_expected_questions (chunk_id, question_text, question_normalized, question_hash, embedding)
                           SELECT c.id, $3, $4, $5, NULL FROM rag_chunks c
                           WHERE c.id=$1::uuid AND c.corpus_version_id=$2::uuid RETURNING id;""",
                        chunk_id, version_id, text, normal, digest)
                    action = "expected_question_added"
                    target_id = str(row["id"]) if row else ""
                if not row:
                    raise LocalAdminError("EXPECTED_QUESTION_NOT_FOUND")
            await self._invalidate_approvals(version_id, "source_changed")
            await self.repo.refresh_embedding_counts(version_id)
            fingerprint = await self.repo.refresh_embedding_input_fingerprint(version_id)
            await JobQueueService(self.conn).enqueue_job("embed_questions", {
                "corpus_version_id": version_id, "embedding_config_fingerprint": version["embedding_config_fingerprint"],
                "embedding_input_fingerprint": fingerprint,
            }, f"phase3f:question:{version_id}:{target_id}:{fingerprint}")
            await self.audit.create_audit_log(actor_id, action, "chunk_expected_question", target_id,
                after_value={"request_id": request_id, "corpus_version_id": version_id, "result": "pending_embedding"})
        return {"status": "pending_embedding", "id": target_id}

    async def edit_visual(self, *, version_id: str, scope: dict[str, str], actor_id: str, request_id: str, visual_id: str, title: str | None, description: str | None, review_status: str | None, display_policy: str | None) -> dict[str, Any]:
        if review_status is not None and review_status not in {"pending", "approved", "rejected"}:
            raise LocalAdminError("VISUAL_REVIEW_STATUS_INVALID")
        if display_policy is not None and display_policy not in {"always", "llm_decide", "never"}:
            raise LocalAdminError("VISUAL_DISPLAY_POLICY_INVALID")
        async with self.conn.transaction():
            version = await self._version_for_scope(version_id, scope, lock=True)
            await self._make_building(version)
            previous = await self.conn.fetchrow(
                """SELECT v.id, v.chunk_id, v.title, v.description, v.review_status, v.display_policy
                   FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id
                   WHERE v.id=$1::uuid AND c.corpus_version_id=$2::uuid FOR UPDATE;""", visual_id, version_id)
            if not previous:
                raise LocalAdminError("VISUAL_NOT_FOUND")
            before = dict(previous)
            new_title = title.strip() if isinstance(title, str) else before["title"]
            new_description = description.strip() if isinstance(description, str) else before["description"]
            if not new_title or not new_description or len(new_title) > 240 or len(new_description) > 4000:
                raise LocalAdminError("VISUAL_TEXT_INVALID")
            new_status = review_status or before["review_status"]
            new_policy = display_policy or before["display_policy"]
            embedding_changed = (new_title, new_description, new_status) != (before["title"], before["description"], before["review_status"])
            await self.conn.execute(
                """UPDATE rag_visuals SET title=$2, description=$3, review_status=$4, display_policy=$5, visual_text_hash=$6
                   WHERE id=$1::uuid;""", visual_id, new_title, new_description, new_status, new_policy, _visual_hash(new_title, new_description))
            if embedding_changed:
                await self.conn.execute(
                    """UPDATE rag_chunks SET embedding=NULL, embedding_status='pending', embedding_input_hash='',
                           embedding_model='', embedding_revision='', embedding_config_fingerprint='', embedding_started_at=NULL,
                           embedding_completed_at=NULL, embedding_error_code=NULL WHERE id=$1::uuid;""", before["chunk_id"])
            await self._invalidate_approvals(version_id, "visual_embedding_metadata_changed" if embedding_changed else "visual_policy_changed")
            await self.repo.refresh_embedding_counts(version_id)
            fingerprint = await self.repo.refresh_embedding_input_fingerprint(version_id)
            if embedding_changed:
                await JobQueueService(self.conn).enqueue_job("embed_chunks", {
                    "corpus_version_id": version_id, "embedding_config_fingerprint": version["embedding_config_fingerprint"],
                    "embedding_input_fingerprint": fingerprint,
                }, f"phase3f:visual:{version_id}:{visual_id}:{fingerprint}")
            await self.audit.create_audit_log(actor_id, "visual_edited", "rag_visual", visual_id,
                before_value={"title": before["title"], "description": before["description"], "review_status": before["review_status"], "display_policy": before["display_policy"]},
                after_value={"request_id": request_id, "title": new_title, "description": new_description, "review_status": new_status, "display_policy": new_policy, "embedding_invalidated": embedding_changed})
        return {"status": "pending_embedding" if embedding_changed else "updated", "id": visual_id}

    async def approve_qa(self, *, version_id: str, scope: dict[str, str], actor_id: str, request_id: str) -> None:
        async with self.conn.transaction():
            version = await self._version_for_scope(version_id, scope, lock=True)
            if version["status"] != "qa_ready":
                raise LocalAdminError("QA_APPROVAL_REQUIRES_QA_READY")
            await self.conn.execute("INSERT INTO rag_corpus_qa_approvals (corpus_version_id, reviewer_id, request_id, summary) VALUES ($1::uuid,$2,$3,$4::jsonb)", version_id, actor_id, request_id, '{"action":"named_version_review"}')
            await self.audit.create_audit_log(actor_id, "corpus_qa_approved", "rag_corpus_version", version_id, after_value={"request_id": request_id, "scope": scope, "result": "approved"})

    async def create_draft(self, *, active_version_id: str, scope: dict[str, str], actor_id: str, request_id: str) -> dict[str, Any]:
        """Copies an active snapshot into an isolated building snapshot, including only its valid provenance."""
        async with self.conn.transaction():
            source = await self._version_for_scope(active_version_id, scope, lock=True)
            if source["status"] != "active":
                raise LocalAdminError("DRAFT_SOURCE_NOT_ACTIVE")
            await self.conn.execute("SELECT id FROM rag_corpora WHERE id=$1 FOR UPDATE", source["corpus_id"])
            version_no = await self.conn.fetchval("SELECT COALESCE(MAX(version_no),0)+1 FROM rag_corpus_versions WHERE corpus_id=$1", source["corpus_id"])
            draft = await self.conn.fetchrow(
                """INSERT INTO rag_corpus_versions (corpus_id, version_no, embedding_model, embedding_revision, embedding_dim,
                    normalize_embeddings, query_instruction, chunking_config, embedding_config_fingerprint, embedding_input_fingerprint,
                    status, source_corpus_version_id)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'building',$11::uuid) RETURNING *""",
                source["corpus_id"], version_no, source["embedding_model"], source["embedding_revision"], source["embedding_dim"],
                source["normalize_embeddings"], source["query_instruction"], source["chunking_config"], source["embedding_config_fingerprint"],
                source["embedding_input_fingerprint"], active_version_id)
            mappings: dict[str, str] = {}
            for doc in await self.conn.fetch("SELECT * FROM rag_document_versions WHERE corpus_version_id=$1::uuid ORDER BY id", active_version_id):
                new_doc = await self.conn.fetchrow("""INSERT INTO rag_document_versions (corpus_version_id,resource_id,resource_version_id,pipeline_version,doc_title,total_chunks)
                    VALUES ($1,$2,$3,$4,$5,$6) RETURNING id""", draft["id"], doc["resource_id"], doc["resource_version_id"], doc["pipeline_version"], doc["doc_title"], doc["total_chunks"])
                for chunk in await self.conn.fetch(
                    """SELECT *,
                        (embedding IS NOT NULL AND vector_dims(embedding)=$2
                         AND embedding_status='embedded' AND embedding_model=$3
                         AND embedding_revision=$4 AND embedding_config_fingerprint=$5)
                        AS provenance_valid
                       FROM rag_chunks WHERE document_version_id=$1 ORDER BY id""",
                    doc["id"], source["embedding_dim"], source["embedding_model"],
                    source["embedding_revision"], source["embedding_config_fingerprint"],
                ):
                    valid = bool(chunk["provenance_valid"])
                    new_chunk = await self.conn.fetchrow("""INSERT INTO rag_chunks (document_version_id,corpus_version_id,chunk_index,content,chapter_id,topic_no,topic_title,page_start,page_end,embedding,content_type,metadata,content_hash,language,token_count,embedding_model,embedding_revision,embedding_config_fingerprint,embedding_input_hash,embedding_status,embedding_started_at,embedding_completed_at,embedding_error_code)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23) RETURNING id""",
                        new_doc["id"], draft["id"], chunk["chunk_index"], chunk["content"], chunk["chapter_id"], chunk["topic_no"], chunk["topic_title"], chunk["page_start"], chunk["page_end"], chunk["embedding"] if valid else None, chunk["content_type"], chunk["metadata"], chunk["content_hash"], chunk["language"], chunk["token_count"], chunk["embedding_model"] if valid else "", chunk["embedding_revision"] if valid else "", chunk["embedding_config_fingerprint"] if valid else "", chunk["embedding_input_hash"] if valid else "", chunk["embedding_status"] if valid else "pending", chunk["embedding_started_at"] if valid else None, chunk["embedding_completed_at"] if valid else None, chunk["embedding_error_code"] if valid else None)
                    mappings[str(chunk["id"])] = str(new_chunk["id"])
            for old_chunk, new_chunk in mappings.items():
                for question in await self.conn.fetch(
                    """SELECT q.*,
                        (q.embedding IS NOT NULL AND vector_dims(q.embedding)=$2
                         AND q.embedding_status='embedded' AND q.embedding_model=$3
                         AND q.embedding_revision=$4 AND q.embedding_config_fingerprint=$5)
                        AS provenance_valid
                       FROM chunk_expected_questions q WHERE q.chunk_id=$1::uuid""",
                    old_chunk, source["embedding_dim"], source["embedding_model"],
                    source["embedding_revision"], source["embedding_config_fingerprint"],
                ):
                    valid = bool(question["provenance_valid"])
                    await self.conn.execute("""INSERT INTO chunk_expected_questions (chunk_id,question_text,question_normalized,question_hash,embedding,embedding_model,embedding_revision,embedding_config_fingerprint,embedding_input_hash,embedding_status,embedding_started_at,embedding_completed_at,embedding_error_code)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""", new_chunk, question["question_text"], question["question_normalized"], question["question_hash"], question["embedding"] if valid else None, question["embedding_model"] if valid else "", question["embedding_revision"] if valid else "", question["embedding_config_fingerprint"] if valid else "", question["embedding_input_hash"] if valid else "", question["embedding_status"] if valid else "pending", question["embedding_started_at"] if valid else None, question["embedding_completed_at"] if valid else None, question["embedding_error_code"] if valid else None)
                for visual in await self.conn.fetch("SELECT * FROM rag_visuals WHERE chunk_id=$1::uuid", old_chunk):
                    await self.conn.execute("""INSERT INTO rag_visuals (chunk_id,visual_id,visual_type,storage_path,title,description,storage_provider,storage_key,display_policy,review_status,visual_text_hash)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""", new_chunk, visual["visual_id"], visual["visual_type"], visual["storage_key"], visual["title"], visual["description"], visual["storage_provider"], visual["storage_key"], visual["display_policy"], visual["review_status"], visual["visual_text_hash"])
            await self.repo.refresh_embedding_counts(str(draft["id"]))
            await self.audit.create_audit_log(actor_id, "corpus_editable_draft_created", "rag_corpus_version", str(draft["id"]), after_value={"request_id": request_id, "source_corpus_version_id": active_version_id, "scope": scope})
        return {"id": str(draft["id"]), "status": "building", "source_corpus_version_id": active_version_id}

    async def overview(self, scope: dict[str, str]) -> dict[str, Any]:
        rows = await self.conn.fetch("""SELECT cv.id::text AS id, cv.version_no, cv.status, cv.source_corpus_version_id::text AS source_corpus_version_id,
            cv.expected_chunk_count,cv.embedded_chunk_count,cv.expected_question_count,cv.embedded_question_count,cv.embedding_config_fingerprint
            FROM rag_corpus_versions cv JOIN rag_corpora c ON c.id=cv.corpus_id WHERE c.board_id=$1 AND c.class_id=$2 AND c.subject_id=$3 ORDER BY cv.version_no DESC""", scope["board_id"], scope["class_id"], scope["subject_id"])
        jobs = await self.conn.fetch(
            """SELECT id::text AS id, job_type, status, stage, progress, attempt_count, max_attempts,
                      error_code, created_at, updated_at, completed_at
               FROM job_queue
               WHERE (payload->'scope'->>'board_id'=$1 AND payload->'scope'->>'class_id'=$2 AND payload->'scope'->>'subject_id'=$3)
                  OR payload->>'corpus_version_id' IN (
                      SELECT cv.id::text FROM rag_corpus_versions cv JOIN rag_corpora c ON c.id=cv.corpus_id
                      WHERE c.board_id=$1 AND c.class_id=$2 AND c.subject_id=$3
                  )
               ORDER BY created_at DESC LIMIT 100""",
            scope["board_id"], scope["class_id"], scope["subject_id"],
        )
        return {
            "scope": scope,
            "versions": [dict(row) for row in rows],
            # Error text can contain provider/internal details.  The browser sees
            # stable error codes only; local operators can correlate via job id.
            "jobs": [dict(row) for row in jobs],
        }

    async def visual_stream_reference(self, *, version_id: str, scope: dict[str, str], visual_id: str) -> dict[str, str]:
        """Returns a Drive key only to the trusted BFF, never to a browser DTO or audit log."""
        await self._version_for_scope(version_id, scope)
        row = await self.conn.fetchrow(
            """SELECT v.storage_provider, v.storage_key FROM rag_visuals v
               JOIN rag_chunks c ON c.id=v.chunk_id
               WHERE v.id=$1::uuid AND c.corpus_version_id=$2::uuid""",
            visual_id,
            version_id,
        )
        if not row or row["storage_provider"] != "google_drive" or not row["storage_key"].strip():
            raise LocalAdminError("VISUAL_STORAGE_REFERENCE_INVALID")
        return {"storage_key": row["storage_key"]}

    async def inspect_version(self, *, version_id: str, scope: dict[str, str]) -> dict[str, Any]:
        await self._version_for_scope(version_id, scope)
        chunks = await self.conn.fetch("""SELECT id::text AS id, chapter_id, topic_no, topic_title, chunk_index,
            content_type, page_start, page_end, embedding_status FROM rag_chunks WHERE corpus_version_id=$1::uuid ORDER BY chunk_index,id""", version_id)
        questions = await self.conn.fetch("""SELECT q.id::text AS id,q.chunk_id::text AS chunk_id,q.question_text,q.embedding_status
            FROM chunk_expected_questions q JOIN rag_chunks c ON c.id=q.chunk_id WHERE c.corpus_version_id=$1::uuid ORDER BY q.created_at,q.id""", version_id)
        visuals = await self.conn.fetch("""SELECT v.id::text AS id,v.chunk_id::text AS chunk_id,v.visual_id,v.visual_type,v.title,v.description,
            v.display_policy,v.review_status FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id WHERE c.corpus_version_id=$1::uuid ORDER BY v.visual_id""", version_id)
        audits = await self.conn.fetch("""SELECT id::text AS id,actor_id,action,target_type,target_id,created_at,before_value,after_value
            FROM admin_audit_logs WHERE target_id=$1 OR after_value->>'corpus_version_id'=$1 ORDER BY created_at DESC LIMIT 100""", version_id)
        return {"chunks": [dict(row) for row in chunks], "questions": [dict(row) for row in questions], "visuals": [dict(row) for row in visuals], "audits": [dict(row) for row in audits]}

    async def activate(self, *, version_id: str, scope: dict[str, str], actor_id: str, request_id: str, rollback: bool = False) -> None:
        async with self.conn.transaction():
            # Required lock order: corpus first, target/current versions second.
            corpus = await self.conn.fetchrow("SELECT id FROM rag_corpora WHERE board_id=$1 AND class_id=$2 AND subject_id=$3 FOR UPDATE", scope["board_id"], scope["class_id"], scope["subject_id"])
            if not corpus:
                raise LocalAdminError("CORPUS_SCOPE_NOT_FOUND")
            versions = await self.conn.fetch("SELECT * FROM rag_corpus_versions WHERE corpus_id=$1 FOR UPDATE", corpus["id"])
            target = next((dict(v) for v in versions if str(v["id"]) == version_id), None)
            if not target or target["status"] not in ({"qa_ready", "superseded"} if rollback else {"qa_ready"}):
                raise LocalAdminError("ACTIVATION_TARGET_NOT_QA_READY")
            # Recheck every persisted row and durable embedding job.  A snapshot can
            # be made stale after it first becomes qa_ready, so its counters alone
            # are never activation authority.
            readiness = await self.repo.completeness_report(version_id, require_building=False)
            if not readiness["ready"]:
                raise LocalAdminError("ACTIVATION_CORPUS_INCOMPLETE")
            approval = await self.conn.fetchval("SELECT EXISTS(SELECT 1 FROM rag_corpus_qa_approvals WHERE corpus_version_id=$1::uuid AND invalidated_at IS NULL)", version_id)
            if not approval:
                raise LocalAdminError("ACTIVATION_QA_APPROVAL_REQUIRED")
            invalid_visual = await self.conn.fetchval("""SELECT EXISTS(SELECT 1 FROM rag_visuals v JOIN rag_chunks c ON c.id=v.chunk_id WHERE c.corpus_version_id=$1::uuid AND v.review_status='approved' AND v.display_policy <> 'never' AND (v.storage_provider <> 'google_drive' OR btrim(v.storage_key)=''))""", version_id)
            if invalid_visual:
                raise LocalAdminError("ACTIVATION_VISUAL_STORAGE_INVALID")
            await self.conn.execute("UPDATE rag_corpus_versions SET status='superseded' WHERE corpus_id=$1 AND status='active'", corpus["id"])
            await self.conn.execute("UPDATE rag_corpus_versions SET status='active', activated_at=NOW(), activated_by=$2 WHERE id=$1::uuid", version_id, actor_id)
            await self.audit.create_audit_log(actor_id, "corpus_rollback" if rollback else "corpus_activated", "rag_corpus_version", version_id, after_value={"request_id": request_id, "scope": scope, "result": "active"})
            await self._invalidate_active_retrieval_cache(scope)

    async def _invalidate_active_retrieval_cache(self, scope: dict[str, str]) -> None:
        """Explicit no-op seam until active retrieval gains a process cache.

        Retrieval resolves the active corpus from PostgreSQL for every request,
        so no local cache exists to invalidate today.  Keeping this named seam
        makes an eventual cache integration mandatory rather than implied.
        """
        del scope

"""Asyncpg implementation of the versioned prompt repository contract."""

from __future__ import annotations

import hashlib
from typing import Sequence

import asyncpg

from app.repositories.audit_repository import AuditRepository
from app.services.prompts.models import (
    AnswerMode,
    PromptActivation,
    PromptKey,
    PromptRecord,
    PromptScope,
    PromptStatus,
)


class PostgresPromptRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    @staticmethod
    def _record(row: asyncpg.Record | None) -> PromptRecord | None:
        if row is None:
            return None
        return PromptRecord(
            id=str(row["id"]),
            prompt_key=PromptKey(row["prompt_key"]),
            answer_mode=AnswerMode(row["answer_mode"]),
            scope=PromptScope(
                board_id=row["board_id"],
                class_id=row["class_id"],
                subject_id=row["subject_id"],
            ),
            version=row["version"],
            content=row["content"],
            status=PromptStatus(row["status"]),
            created_by=row["created_by"],
            created_at=row["created_at"],
            activated_by=row["activated_by"],
            activated_at=row["activated_at"],
        )

    async def create_draft(
        self,
        *,
        prompt_key: PromptKey,
        answer_mode: AnswerMode,
        scope: PromptScope,
        content: str,
        actor_id: str,
    ) -> PromptRecord:
        async with self.conn.transaction():
            await self.conn.execute(
                """SELECT pg_advisory_xact_lock(
                     hashtextextended($1 || ':' || $2 || ':' ||
                       COALESCE($3,'') || ':' || COALESCE($4,'') || ':' ||
                       COALESCE($5,''), 0)
                   )""",
                prompt_key.value,
                answer_mode.value,
                scope.board_id,
                scope.class_id,
                scope.subject_id,
            )
            version = await self.conn.fetchval(
                """SELECT COALESCE(MAX(version),0)+1 FROM prompt_versions
                   WHERE prompt_key=$1 AND answer_mode=$2
                     AND board_id IS NOT DISTINCT FROM $3
                     AND class_id IS NOT DISTINCT FROM $4
                     AND subject_id IS NOT DISTINCT FROM $5""",
                prompt_key.value,
                answer_mode.value,
                scope.board_id,
                scope.class_id,
                scope.subject_id,
            )
            row = await self.conn.fetchrow(
                """INSERT INTO prompt_versions(
                     prompt_key,answer_mode,board_id,class_id,subject_id,version,
                     content,status,created_by
                   ) VALUES($1,$2,$3,$4,$5,$6,$7,'draft',$8)
                   RETURNING *""",
                prompt_key.value,
                answer_mode.value,
                scope.board_id,
                scope.class_id,
                scope.subject_id,
                version,
                content,
                actor_id,
            )
            await AuditRepository(self.conn).create_audit_log(
                actor_id=actor_id,
                action="prompt.draft_created",
                target_type="prompt",
                target_id=str(row["id"]),
                after_value={
                    "prompt_key": prompt_key.value,
                    "answer_mode": answer_mode.value,
                    "version": version,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )
        return self._record(row)  # type: ignore[return-value]

    async def get(self, prompt_id: str) -> PromptRecord | None:
        return self._record(
            await self.conn.fetchrow(
                "SELECT * FROM prompt_versions WHERE id=$1::uuid", prompt_id
            )
        )

    async def update_draft(
        self, *, prompt_id: str, content: str, actor_id: str
    ) -> PromptRecord:
        async with self.conn.transaction():
            before = await self.conn.fetchrow(
                """SELECT * FROM prompt_versions
                   WHERE id=$1::uuid FOR UPDATE""",
                prompt_id,
            )
            if before is None:
                raise LookupError("PROMPT_NOT_FOUND")
            if before["status"] != "draft":
                raise ValueError("PROMPT_EDIT_REQUIRES_DRAFT")
            row = await self.conn.fetchrow(
                """UPDATE prompt_versions SET content=$2
                   WHERE id=$1::uuid RETURNING *""",
                prompt_id,
                content,
            )
            await AuditRepository(self.conn).create_audit_log(
                actor_id=actor_id,
                action="prompt.draft_updated",
                target_type="prompt",
                target_id=prompt_id,
                before_value={
                    "content_hash": hashlib.sha256(
                        before["content"].encode()
                    ).hexdigest()
                },
                after_value={
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()
                },
            )
        return self._record(row)  # type: ignore[return-value]

    async def find_active(
        self,
        *,
        prompt_key: PromptKey,
        answer_mode: AnswerMode,
        scope: PromptScope,
    ) -> PromptRecord | None:
        return self._record(
            await self.conn.fetchrow(
                """SELECT * FROM prompt_versions
                   WHERE prompt_key=$1 AND answer_mode=$2
                     AND board_id IS NOT DISTINCT FROM $3
                     AND class_id IS NOT DISTINCT FROM $4
                     AND subject_id IS NOT DISTINCT FROM $5
                     AND status='active'
                   LIMIT 1""",
                prompt_key.value,
                answer_mode.value,
                scope.board_id,
                scope.class_id,
                scope.subject_id,
            )
        )

    async def activate(self, *, prompt_id: str, actor_id: str) -> PromptActivation:
        return await self._activate_target(
            prompt_id=prompt_id, actor_id=actor_id, require_draft=True
        )

    async def rollback(
        self, *, target_prompt_id: str, actor_id: str
    ) -> PromptActivation:
        return await self._activate_target(
            prompt_id=target_prompt_id, actor_id=actor_id, require_draft=False
        )

    async def _activate_target(
        self, *, prompt_id: str, actor_id: str, require_draft: bool
    ) -> PromptActivation:
        async with self.conn.transaction():
            target = await self.conn.fetchrow(
                "SELECT * FROM prompt_versions WHERE id=$1::uuid FOR UPDATE",
                prompt_id,
            )
            if target is None:
                raise LookupError("PROMPT_NOT_FOUND")
            allowed = {"draft"} if require_draft else {"retired", "active"}
            if target["status"] not in allowed:
                raise ValueError("PROMPT_ACTIVATION_STATE_INVALID")
            current = await self.conn.fetchrow(
                """SELECT * FROM prompt_versions
                   WHERE prompt_key=$1 AND answer_mode=$2
                     AND board_id IS NOT DISTINCT FROM $3
                     AND class_id IS NOT DISTINCT FROM $4
                     AND subject_id IS NOT DISTINCT FROM $5
                     AND status='active' AND id<>$6::uuid
                   FOR UPDATE""",
                target["prompt_key"],
                target["answer_mode"],
                target["board_id"],
                target["class_id"],
                target["subject_id"],
                prompt_id,
            )
            if current is not None:
                await self.conn.execute(
                    """UPDATE prompt_versions
                       SET status='retired',retired_at=NOW()
                       WHERE id=$1::uuid""",
                    current["id"],
                )
            active = await self.conn.fetchrow(
                """UPDATE prompt_versions
                   SET status='active',activated_by=$2,activated_at=NOW(),
                       retired_at=NULL
                   WHERE id=$1::uuid RETURNING *""",
                prompt_id,
                actor_id,
            )
            await AuditRepository(self.conn).create_audit_log(
                actor_id=actor_id,
                action=("prompt.activated" if require_draft else "prompt.rolled_back"),
                target_type="prompt",
                target_id=prompt_id,
                before_value={
                    "active_prompt_id": str(current["id"]) if current else None
                },
                after_value={
                    "active_prompt_id": prompt_id,
                    "version": active["version"],
                },
            )
            await self.conn.execute(
                """INSERT INTO cache_generations(namespace,cache_key,generation)
                   VALUES('prompt',$1,2)
                   ON CONFLICT(namespace,cache_key) DO UPDATE
                   SET generation=cache_generations.generation+1,
                       updated_at=NOW()""",
                f"{target['prompt_key']}:{target['answer_mode']}",
            )
        return PromptActivation(
            active=self._record(active),  # type: ignore[arg-type]
            retired=self._record(current),
        )

    async def list_history(
        self,
        *,
        prompt_key: PromptKey,
        answer_mode: AnswerMode,
        scope: PromptScope | None,
        limit: int,
    ) -> Sequence[PromptRecord]:
        rows = await self.conn.fetch(
            """SELECT * FROM prompt_versions
               WHERE prompt_key=$1 AND answer_mode=$2
                 AND ($3::boolean OR (
                   board_id IS NOT DISTINCT FROM $4
                   AND class_id IS NOT DISTINCT FROM $5
                   AND subject_id IS NOT DISTINCT FROM $6
                 ))
               ORDER BY created_at DESC,id DESC LIMIT $7""",
            prompt_key.value,
            answer_mode.value,
            scope is None,
            scope.board_id if scope else None,
            scope.class_id if scope else None,
            scope.subject_id if scope else None,
            limit,
        )
        return tuple(self._record(row) for row in rows)  # type: ignore[misc]

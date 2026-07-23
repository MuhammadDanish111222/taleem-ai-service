"""Admin Audit Log Repository using Asyncpg."""

from typing import Optional, Dict, Any, List
import json
import asyncpg

class AuditRepository:
    def __init__(self, conn: asyncpg.Connection):
        self.conn = conn

    async def create_audit_log(
        self,
        actor_id: str,
        action: str,
        target_type: str,
        target_id: str,
        before_value: Optional[Dict[str, Any]] = None,
        after_value: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates an immutable admin audit log entry."""
        query = """
        INSERT INTO admin_audit_logs (actor_id, action, target_type, target_id, before_value, after_value)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
        RETURNING *;
        """
        before_json = json.dumps(before_value) if before_value is not None else None
        after_json = json.dumps(after_value) if after_value is not None else None
        row = await self.conn.fetchrow(query, actor_id, action, target_type, target_id, before_json, after_json)
        return dict(row)

    async def get_audit_logs(
        self,
        actor_id: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Queries audit logs filtered by actor or target."""
        conditions = []
        params = []

        if actor_id:
            params.append(actor_id)
            conditions.append(f"actor_id = ${len(params)}")
        if target_type:
            params.append(target_type)
            conditions.append(f"target_type = ${len(params)}")
        if target_id:
            params.append(target_id)
            conditions.append(f"target_id = ${len(params)}")

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        limit_clause = f"LIMIT ${len(params)}"

        query = f"SELECT * FROM admin_audit_logs {where_clause} ORDER BY created_at DESC {limit_clause};"
        rows = await self.conn.fetch(query, *params)
        return [dict(r) for r in rows]

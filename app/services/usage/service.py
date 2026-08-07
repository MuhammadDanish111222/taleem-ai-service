"""Atomic Redis reservation with an authoritative PostgreSQL mirror/fallback."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from datetime import datetime

import asyncpg
import redis

from app.core.config import get_settings
from app.repositories.audit_repository import AuditRepository
from app.services.usage.limits import UsagePolicyCache, get_usage_policy_cache
from app.services.usage.models import (
    AccountTier,
    UsageReservation,
    pakistan_business_window,
)

logger = logging.getLogger(__name__)

_RESERVE_LUA = """
local request_key = KEYS[1]
local counter_key = KEYS[2]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
if redis.call('EXISTS', request_key) == 1 then
  local used = tonumber(redis.call('GET', counter_key) or '0')
  return {2, used}
end
local used = tonumber(redis.call('GET', counter_key) or '0')
if used >= limit then return {0, used} end
used = redis.call('INCR', counter_key)
redis.call('EXPIRE', counter_key, ttl)
redis.call('SET', request_key, '1', 'EX', ttl, 'NX')
return {1, used}
"""

_REFUND_LUA = """
local request_key = KEYS[1]
local counter_key = KEYS[2]
if redis.call('DEL', request_key) == 1 then
  local used = tonumber(redis.call('GET', counter_key) or '0')
  if used > 0 then used = redis.call('DECR', counter_key) end
  return used
end
return tonumber(redis.call('GET', counter_key) or '0')
"""


class UsageLimitExceeded(ValueError):
    def __init__(
        self,
        *,
        used: int,
        limit: int,
        student_visible: bool,
        resets_at: datetime,
    ):
        super().__init__("USAGE_LIMIT_REACHED")
        self.used = used
        self.limit = limit
        self.student_visible = student_visible
        self.resets_at = resets_at


class UsageService:
    def __init__(
        self,
        *,
        redis_client: redis.Redis | None = None,
        policy_cache: UsagePolicyCache | None = None,
        now: datetime | None = None,
    ):
        self._redis = redis_client
        self._policy_cache = policy_cache or get_usage_policy_cache()
        self._now = now

    @staticmethod
    def uid_hash(uid: str) -> str:
        secret = get_settings().USAGE_UID_HMAC_SECRET
        if not secret:
            raise RuntimeError("USAGE_HASH_SECRET_UNAVAILABLE")
        return hmac.new(
            secret.encode("utf-8"), uid.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                get_settings().REDIS_URL, decode_responses=True
            )
        return self._redis

    @staticmethod
    def _keys(
        *, request_id: str, uid_hash: str, feature: str, business_date: str
    ) -> tuple[str, str]:
        counter = f"usage:{feature}:{business_date}:{uid_hash}"
        request = f"usage:req:{feature}:{business_date}:{uid_hash}:{request_id}"
        return request, counter

    async def reserve(
        self,
        conn: asyncpg.Connection,
        *,
        request_id: str,
        uid: str,
        tier: AccountTier,
        feature: str = "single_question",
    ) -> UsageReservation:
        return await self._reserve_for_uid_hash(
            conn,
            request_id=request_id,
            uid_hash=self.uid_hash(uid),
            tier=tier,
            feature=feature,
        )

    async def reserve_for_uid_hash(
        self,
        conn: asyncpg.Connection,
        *,
        request_id: str,
        uid_hash: str,
        tier: AccountTier,
        feature: str,
    ) -> UsageReservation:
        """Worker-only reservation using the stored HMAC identity, never a raw UID."""
        if len(uid_hash) != 64 or any(
            char not in "0123456789abcdef" for char in uid_hash
        ):
            raise ValueError("USAGE_UID_HASH_INVALID")
        return await self._reserve_for_uid_hash(
            conn,
            request_id=request_id,
            uid_hash=uid_hash,
            tier=tier,
            feature=feature,
        )

    async def _reserve_for_uid_hash(
        self,
        conn: asyncpg.Connection,
        *,
        request_id: str,
        uid_hash: str,
        tier: AccountTier,
        feature: str,
    ) -> UsageReservation:
        window = pakistan_business_window(self._now)
        safe_uid = uid_hash
        policy = await self._policy_cache.get(conn, feature, tier)
        request_key, counter_key = self._keys(
            request_id=request_id,
            uid_hash=safe_uid,
            feature=feature,
            business_date=window.business_date.isoformat(),
        )

        backend = "redis"
        redis_incremented = False
        redis_duplicate = False
        try:
            result = await asyncio.to_thread(
                self._client().eval,
                _RESERVE_LUA,
                2,
                request_key,
                counter_key,
                policy.daily_limit,
                window.ttl_seconds,
            )
            decision, redis_used = int(result[0]), int(result[1])
            if decision == 0:
                raise UsageLimitExceeded(
                    used=redis_used,
                    limit=policy.daily_limit,
                    student_visible=policy.student_visible,
                    resets_at=window.resets_at,
                )
            redis_incremented = decision == 1
            redis_duplicate = decision == 2
        except UsageLimitExceeded:
            raise
        except Exception:
            backend = "postgresql"
            logger.warning("usage_fallback", extra={"event": "redis_unavailable"})
            await AuditRepository(conn).create_audit_log(
                actor_id="system",
                action="usage.redis_fallback",
                target_type="usage",
                target_id=feature,
                after_value={"event": "redis_unavailable"},
            )

        try:
            inserted_reservation = await conn.fetchval(
                """INSERT INTO usage_reservations(
                     request_id,business_date,feature,uid_hash,account_tier,backend,status
                   ) VALUES($1::uuid,$2,$3,$4,$5,$6,'reserved')
                   ON CONFLICT(request_id,uid_hash) DO NOTHING
                   RETURNING TRUE""",
                request_id,
                window.business_date,
                feature,
                safe_uid,
                tier.value,
                backend,
            )
            if not inserted_reservation:
                row = await conn.fetchrow(
                    """SELECT used FROM daily_usage
                       WHERE business_date=$1 AND feature=$2 AND uid_hash=$3""",
                    window.business_date,
                    feature,
                    safe_uid,
                )
                return UsageReservation(
                    request_id=request_id,
                    used=int(row["used"] if row else 0),
                    limit=policy.daily_limit,
                    student_visible=policy.student_visible,
                    window=window,
                    backend=backend,
                    duplicate=True,
                )

            usage = await conn.fetchrow(
                """INSERT INTO daily_usage(business_date, feature, uid_hash, used)
                   SELECT $1,$2,$3,1 WHERE $4 > 0
                   ON CONFLICT(business_date, feature, uid_hash) DO UPDATE
                   SET used=daily_usage.used+1, updated_at=NOW()
                   WHERE daily_usage.used < $4
                   RETURNING used""",
                window.business_date,
                feature,
                safe_uid,
                policy.daily_limit,
            )
            if usage is None:
                current = await conn.fetchval(
                    """SELECT used FROM daily_usage
                       WHERE business_date=$1 AND feature=$2 AND uid_hash=$3""",
                    window.business_date,
                    feature,
                    safe_uid,
                )
                raise UsageLimitExceeded(
                    used=int(current or 0),
                    limit=policy.daily_limit,
                    student_visible=policy.student_visible,
                    resets_at=window.resets_at,
                )
            return UsageReservation(
                request_id=request_id,
                used=int(usage["used"]),
                limit=policy.daily_limit,
                student_visible=policy.student_visible,
                window=window,
                backend=backend,
                duplicate=redis_duplicate,
            )
        except Exception:
            if redis_incremented:
                await asyncio.to_thread(
                    self._client().eval,
                    _REFUND_LUA,
                    2,
                    request_key,
                    counter_key,
                )
            raise

    async def commit(
        self, conn: asyncpg.Connection, request_id: str, uid_hash: str
    ) -> None:
        await conn.execute(
            """UPDATE usage_reservations SET status='committed', updated_at=NOW()
               WHERE request_id=$1::uuid AND uid_hash=$2 AND status='reserved'""",
            request_id,
            uid_hash,
        )

    async def refund(
        self, conn: asyncpg.Connection, request_id: str, uid_hash: str
    ) -> None:
        row = await conn.fetchrow(
            """UPDATE usage_reservations SET status='refunded', updated_at=NOW()
               WHERE request_id=$1::uuid AND uid_hash=$2 AND status='reserved'
               RETURNING business_date,feature,uid_hash,backend""",
            request_id,
            uid_hash,
        )
        if row is None:
            return
        await conn.execute(
            """UPDATE daily_usage SET used=GREATEST(used-1,0), updated_at=NOW()
               WHERE business_date=$1 AND feature=$2 AND uid_hash=$3""",
            row["business_date"],
            row["feature"],
            row["uid_hash"],
        )
        if row["backend"] == "redis":
            request_key, counter_key = self._keys(
                request_id=request_id,
                uid_hash=row["uid_hash"],
                feature=row["feature"],
                business_date=row["business_date"].isoformat(),
            )
            try:
                await asyncio.to_thread(
                    self._client().eval,
                    _REFUND_LUA,
                    2,
                    request_key,
                    counter_key,
                )
            except Exception:
                logger.warning(
                    "usage_refund_redis_unavailable",
                    extra={"event": "redis_refund_unavailable"},
                )

    async def refund_committed(
        self, conn: asyncpg.Connection, request_id: str, uid_hash: str
    ) -> bool:
        """Refund a committed durable batch exactly once after infrastructure failure.

        Normal Ask calls refund a still-reserved request. Multiple Ask commits
        after validation, so its exhausted OCR/extraction path needs an explicit
        committed-refund method instead of weakening ordinary Ask semantics.
        """
        row = await conn.fetchrow(
            """UPDATE usage_reservations SET status='refunded', updated_at=NOW()
               WHERE request_id=$1::uuid AND uid_hash=$2 AND status='committed'
               RETURNING business_date,feature,uid_hash,backend""",
            request_id,
            uid_hash,
        )
        if row is None:
            return False
        await conn.execute(
            """UPDATE daily_usage SET used=GREATEST(used-1,0), updated_at=NOW()
               WHERE business_date=$1 AND feature=$2 AND uid_hash=$3""",
            row["business_date"],
            row["feature"],
            row["uid_hash"],
        )
        if row["backend"] == "redis":
            request_key, counter_key = self._keys(
                request_id=request_id,
                uid_hash=row["uid_hash"],
                feature=row["feature"],
                business_date=row["business_date"].isoformat(),
            )
            try:
                await asyncio.to_thread(
                    self._client().eval, _REFUND_LUA, 2, request_key, counter_key
                )
            except Exception:
                logger.warning(
                    "usage_committed_refund_redis_unavailable",
                    extra={"event": "redis_refund_unavailable"},
                )
        return True

    async def snapshot(
        self,
        conn: asyncpg.Connection,
        *,
        uid: str,
        tier: AccountTier,
        feature: str = "single_question",
    ) -> UsageReservation:
        window = pakistan_business_window(self._now)
        safe_uid = self.uid_hash(uid)
        policy = await self._policy_cache.get(conn, feature, tier)
        used = await conn.fetchval(
            """SELECT used FROM daily_usage
               WHERE business_date=$1 AND feature=$2 AND uid_hash=$3""",
            window.business_date,
            feature,
            safe_uid,
        )
        return UsageReservation(
            request_id="",
            used=int(used or 0),
            limit=policy.daily_limit,
            student_visible=policy.student_visible,
            window=window,
            backend="postgresql",
        )

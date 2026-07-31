"""Bounded typed usage-policy cache with shared generation invalidation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import asyncpg

from app.core.config import get_settings
from app.services.usage.models import AccountTier, UsagePolicy


@dataclass
class _Entry:
    value: UsagePolicy
    expires_at: float
    generation: int


class UsagePolicyCache:
    def __init__(self, ttl_seconds: int | None = None):
        self._ttl = ttl_seconds or get_settings().USAGE_POLICY_CACHE_TTL_SECONDS
        self._entries: dict[tuple[str, AccountTier], _Entry] = {}
        self._lock = asyncio.Lock()

    async def get(
        self, conn: asyncpg.Connection, feature: str, tier: AccountTier
    ) -> UsagePolicy:
        key = (feature, tier)
        generation = await conn.fetchval(
            """SELECT generation FROM cache_generations
               WHERE namespace='usage_policy' AND cache_key=$1""",
            f"{feature}:{tier.value}",
        )
        generation = int(generation or 1)
        now = time.monotonic()
        cached = self._entries.get(key)
        if cached and cached.expires_at > now and cached.generation == generation:
            return cached.value
        async with self._lock:
            cached = self._entries.get(key)
            if cached and cached.expires_at > now and cached.generation == generation:
                return cached.value
            row = await conn.fetchrow(
                """SELECT feature, account_tier, daily_limit, student_visible
                   FROM usage_policies
                   WHERE feature=$1 AND account_tier=$2""",
                feature,
                tier.value,
            )
            if row is None:
                raise RuntimeError("USAGE_POLICY_UNAVAILABLE")
            policy = UsagePolicy(
                feature=row["feature"],
                account_tier=AccountTier(row["account_tier"]),
                daily_limit=row["daily_limit"],
                student_visible=row["student_visible"],
            )
            self._entries[key] = _Entry(
                policy, time.monotonic() + self._ttl, generation
            )
            return policy

    async def invalidate(
        self,
        conn: asyncpg.Connection,
        feature: str,
        tier: AccountTier,
    ) -> None:
        cache_key = f"{feature}:{tier.value}"
        await conn.execute(
            """INSERT INTO cache_generations(namespace, cache_key, generation)
               VALUES ('usage_policy', $1, 2)
               ON CONFLICT(namespace, cache_key) DO UPDATE
               SET generation=cache_generations.generation+1, updated_at=NOW()""",
            cache_key,
        )
        self._entries.pop((feature, tier), None)


_cache: UsagePolicyCache | None = None


def get_usage_policy_cache() -> UsagePolicyCache:
    global _cache
    if _cache is None:
        _cache = UsagePolicyCache()
    return _cache

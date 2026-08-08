import json

import pytest

from app.services.retrieval.active_version_cache import ActiveCorpusVersionCache


class FakeRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, name: str):
        return self.values.get(name)

    def set(self, name: str, value: str, *, ex: int):
        self.values[name] = value
        self.ttls[name] = ex

    def delete(self, *names: str):
        for name in names:
            self.values.pop(name, None)


def version() -> dict:
    return {
        "id": "9f70aa43-9924-4ae0-8258-1fbfab75290b",
        "embedding_model": "voyage-4-lite",
        "embedding_revision": "revision",
        "embedding_dim": 512,
        "normalize_embeddings": True,
        "embedding_config_fingerprint": "fingerprint",
        "status": "active",
    }


@pytest.mark.asyncio
async def test_active_version_cache_round_trip_and_invalidation():
    redis = FakeRedis()
    cache = ActiveCorpusVersionCache(redis, ttl_seconds=120)

    assert await cache.get("punjab", "9", "chemistry") is None
    await cache.set("punjab", "9", "chemistry", version())

    cached = await cache.get("punjab", "9", "chemistry")
    assert cached == {key: value for key, value in version().items() if key != "status"}
    assert list(redis.ttls.values()) == [120]

    await cache.invalidate("punjab", "9", "chemistry")
    assert await cache.get("punjab", "9", "chemistry") is None


@pytest.mark.asyncio
async def test_active_version_cache_ignores_invalid_payload():
    redis = FakeRedis()
    cache = ActiveCorpusVersionCache(redis)
    key = cache._key("punjab", "9", "chemistry")
    redis.values[key] = json.dumps({"id": "incomplete"})

    assert await cache.get("punjab", "9", "chemistry") is None

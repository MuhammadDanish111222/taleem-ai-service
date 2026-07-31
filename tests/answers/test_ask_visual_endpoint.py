from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException

from app.api.v1 import ask
from app.core.internal_auth import AuthContext


@pytest.mark.asyncio
async def test_visual_reference_requires_dedicated_feature_before_database(monkeypatch):
    called = False

    @asynccontextmanager
    async def forbidden_connection():
        nonlocal called
        called = True
        yield object()

    monkeypatch.setattr(ask, "get_db_connection", forbidden_connection)
    with pytest.raises(HTTPException) as exc_info:
        await ask.ask_visual_reference(
            visual_id="visual-1",
            auth=AuthContext(
                uid="student",
                is_admin=False,
                feature="ask",
                request_id="123e4567-e89b-42d3-a456-426614174000",
            ),
        )
    assert exc_info.value.status_code == 403
    assert called is False


@pytest.mark.asyncio
async def test_visual_reference_returns_only_server_storage_reference(monkeypatch):
    captured = {}

    class FakeRepository:
        def __init__(self, conn):
            captured["conn"] = conn

        async def visual_stream_reference(self, **fields):
            captured.update(fields)
            return {
                "storage_provider": "google_drive",
                "storage_key": "server-only-key",
            }

    @asynccontextmanager
    async def connection():
        yield "db"

    monkeypatch.setattr(ask, "get_db_connection", connection)
    monkeypatch.setattr(ask, "AskRepository", FakeRepository)
    monkeypatch.setattr(
        ask.UsageService,
        "uid_hash",
        staticmethod(lambda _uid: "a" * 64),
    )
    result = await ask.ask_visual_reference(
        visual_id="visual-1",
        auth=AuthContext(
            uid="student",
            is_admin=False,
            feature="ask_visual",
            request_id="123e4567-e89b-42d3-a456-426614174000",
        ),
    )
    assert result == {
        "storage_provider": "google_drive",
        "storage_key": "server-only-key",
    }
    assert captured["client_request_id"] == ("123e4567-e89b-42d3-a456-426614174000")
    assert captured["uid_hash"] == "a" * 64

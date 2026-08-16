import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from app.api.v1.internal import _multiple_ask_tier
from app.core.internal_auth import AuthContext
from app.main import app

pytestmark = pytest.mark.asyncio

# Generate test RSA key pair
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode("utf-8")


def create_token(feature: str = "feature_state_read", uid: str = "server-render"):
    now = int(time.time())
    payload = {
        "uid": uid,
        "admin": False,
        "feature": feature,
        "request_id": "req-123",
        "aud": "taleem-ai-service",
        "iss": "taleem-web",
        "jti": f"jti-feature-state-{now}-{feature}",
        "iat": now,
        "exp": now + 60,
    }
    return jwt.encode(
        payload, private_pem, algorithm="RS256", headers={"kid": "test-kid"}
    )


@pytest.fixture(autouse=True)
def mock_auth_env(monkeypatch):
    monkeypatch.setattr(
        "app.core.internal_auth.get_public_keys", lambda: {"test-kid": public_pem}
    )
    mock_redis_client = MagicMock()
    mock_redis_client.set.return_value = True
    monkeypatch.setattr("app.core.internal_auth.get_redis", lambda: mock_redis_client)
    monkeypatch.setattr(
        "app.core.internal_auth._record_jti_postgres", AsyncMock(return_value=True)
    )


@asynccontextmanager
async def mock_db_conn():
    mock_conn = AsyncMock()
    yield mock_conn


async def test_multiple_ask_tier_enabled():
    auth = AuthContext(
        uid="student-1",
        is_admin=False,
        feature="multiple_ask",
        request_id="req-1",
        account_tier="anonymous",
    )
    with (
        patch("app.api.v1.internal.get_db_connection", mock_db_conn),
        patch(
            "app.api.v1.internal.RuntimeSettingsService.get",
            new_callable=AsyncMock,
            return_value="enabled",
        ),
    ):
        tier = await _multiple_ask_tier(auth)
        assert tier.value == "anonymous"


async def test_multiple_ask_tier_coming_soon():
    auth = AuthContext(
        uid="student-1",
        is_admin=False,
        feature="multiple_ask",
        request_id="req-1",
        account_tier="anonymous",
    )
    with (
        patch("app.api.v1.internal.get_db_connection", mock_db_conn),
        patch(
            "app.api.v1.internal.RuntimeSettingsService.get",
            new_callable=AsyncMock,
            return_value="coming_soon",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _multiple_ask_tier(auth)
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "FEATURE_COMING_SOON"


async def test_multiple_ask_tier_disabled():
    auth = AuthContext(
        uid="student-1",
        is_admin=False,
        feature="multiple_ask",
        request_id="req-1",
        account_tier="anonymous",
    )
    with (
        patch("app.api.v1.internal.get_db_connection", mock_db_conn),
        patch(
            "app.api.v1.internal.RuntimeSettingsService.get",
            new_callable=AsyncMock,
            return_value="disabled",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _multiple_ask_tier(auth)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["code"] == "NOT_FOUND"


async def test_multiple_ask_tier_corrupted_value_fails_closed():
    auth = AuthContext(
        uid="student-1",
        is_admin=False,
        feature="multiple_ask",
        request_id="req-1",
        account_tier="anonymous",
    )
    with (
        patch("app.api.v1.internal.get_db_connection", mock_db_conn),
        patch(
            "app.api.v1.internal.RuntimeSettingsService.get",
            new_callable=AsyncMock,
            return_value="broken_value",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _multiple_ask_tier(auth)
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["code"] == "NOT_FOUND"


async def test_multiple_ask_tier_db_outage_raises_503():
    auth = AuthContext(
        uid="student-1",
        is_admin=False,
        feature="multiple_ask",
        request_id="req-1",
        account_tier="anonymous",
    )
    with patch(
        "app.api.v1.internal.get_db_connection",
        side_effect=Exception("Database down"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _multiple_ask_tier(auth)
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["code"] == "MULTIPLE_ASK_UNAVAILABLE"


async def test_multiple_ask_tier_wrong_feature_claim():
    auth = AuthContext(
        uid="student-1",
        is_admin=False,
        feature="wrong_feature",
        request_id="req-1",
        account_tier="anonymous",
    )
    with (
        patch("app.api.v1.internal.get_db_connection", mock_db_conn),
        patch(
            "app.api.v1.internal.RuntimeSettingsService.get",
            new_callable=AsyncMock,
            return_value="enabled",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _multiple_ask_tier(auth)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail["code"] == "AUTH_FEATURE_FORBIDDEN"


async def test_feature_state_endpoint_success():
    token = create_token(feature="feature_state_read")
    with (
        patch("app.api.v1.feature_state.get_db_connection", mock_db_conn),
        patch(
            "app.api.v1.feature_state.RuntimeSettingsService.get",
            new_callable=AsyncMock,
            return_value="coming_soon",
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get(
                "/api/v1/internal/feature-state/multiple_ask",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            assert response.json() == {
                "feature": "multiple_ask",
                "state": "coming_soon",
            }


async def test_feature_state_endpoint_forbidden_claim():
    token = create_token(feature="general")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get(
            "/api/v1/internal/feature-state/multiple_ask",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "AUTH_FEATURE_FORBIDDEN"


async def test_feature_state_endpoint_unknown_feature():
    token = create_token(feature="feature_state_read")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get(
            "/api/v1/internal/feature-state/unknown_feature",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "UNKNOWN_FEATURE"


async def test_feature_state_endpoint_fails_closed_on_db_error():
    token = create_token(feature="feature_state_read")
    with patch(
        "app.api.v1.feature_state.get_db_connection",
        side_effect=Exception("DB pool exhausted"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get(
                "/api/v1/internal/feature-state/multiple_ask",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            assert response.json() == {"feature": "multiple_ask", "state": "disabled"}

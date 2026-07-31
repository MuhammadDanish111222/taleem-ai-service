import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app.core.internal_auth import get_public_keys, verify_internal_jwt

pytestmark = pytest.mark.asyncio

# Generate a temporary RSA key pair for testing
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


def create_token(
    kid="test-key",
    uid="user123",
    admin=False,
    feature="test",
    request_id="req-123",
    aud="taleem-ai-service",
    iss="taleem-web",
    jti="jti-123",
    iat=None,
    exp=None,
    exp_delta=60,
):
    now = int(time.time())
    token_iat = iat if iat is not None else now
    token_exp = exp if exp is not None else now + exp_delta
    payload = {
        "uid": uid,
        "admin": admin,
        "feature": feature,
        "request_id": request_id,
        "aud": aud,
        "iss": iss,
        "jti": jti,
        "iat": token_iat,
        "exp": token_exp,
    }
    return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def mock_redis():
    with (
        patch("app.core.internal_auth.get_redis") as mock_get_redis,
        patch(
            "app.core.internal_auth._record_jti_postgres",
            new=AsyncMock(return_value=True),
        ),
        patch("app.core.internal_auth._jti_hash", return_value="a" * 64),
    ):
        mock_client = MagicMock()
        # Atomic set returns True when setting a new key
        mock_client.set.return_value = True
        mock_get_redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_keys():
    with patch("app.core.internal_auth.get_public_keys") as mock_get_keys:
        mock_get_keys.return_value = {"test-key": public_pem}
        yield mock_get_keys


async def test_missing_token():
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt("")
    assert exc_info.value.status_code == 401
    assert "Missing or invalid authorization header" in str(
        exc_info.value.detail["message"]
    )


async def test_public_key_configuration_unescapes_pem_newlines():
    settings = MagicMock()
    settings.INTERNAL_JWT_PUBLIC_KEYS_JSON = json.dumps(
        {"test-key": public_pem.replace("\n", "\\n")}
    )

    with patch("app.core.internal_auth.get_settings", return_value=settings):
        assert get_public_keys() == {"test-key": public_pem}


async def test_expired_token(mock_keys, mock_redis):
    token = create_token(exp_delta=-10)
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["code"] == "AUTH_EXPIRED_TOKEN"


async def test_ttl_exceeds_60_seconds(mock_keys, mock_redis):
    now = int(time.time())
    token = create_token(iat=now, exp=now + 120)
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "Token TTL exceeds maximum 60s" in exc_info.value.detail["message"]


async def test_exp_before_iat(mock_keys, mock_redis):
    now = int(time.time())
    token = create_token(iat=now, exp=now - 5)
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401


async def test_wrong_audience(mock_keys, mock_redis):
    token = create_token(aud="wrong-audience")
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "Invalid token" in str(exc_info.value.detail["message"])


async def test_wrong_signature(mock_keys, mock_redis):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    payload = {
        "uid": "user123",
        "admin": False,
        "feature": "test",
        "request_id": "req-1",
        "aud": "taleem-ai-service",
        "iss": "taleem-web",
        "jti": "jti-1",
        "iat": int(time.time()),
        "exp": int(time.time()) + 60,
    }
    token = jwt.encode(
        payload, other_pem, algorithm="RS256", headers={"kid": "test-key"}
    )

    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401


@pytest.mark.parametrize(
    "claim_name", ["uid", "admin", "feature", "request_id", "jti", "iat", "exp"]
)
async def test_missing_mandatory_claims(mock_keys, mock_redis, claim_name):
    now = int(time.time())
    payload = {
        "uid": "user123",
        "admin": False,
        "feature": "test",
        "request_id": "req-123",
        "aud": "taleem-ai-service",
        "iss": "taleem-web",
        "jti": "jti-123",
        "iat": now,
        "exp": now + 60,
    }
    del payload[claim_name]
    token = jwt.encode(
        payload, private_pem, algorithm="RS256", headers={"kid": "test-key"}
    )

    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401


async def test_non_boolean_admin_claim(mock_keys, mock_redis):
    token = create_token(admin="true")  # String instead of bool
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "admin" in str(exc_info.value.detail["message"])


async def test_empty_feature_claim(mock_keys, mock_redis):
    token = create_token(feature="")
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "feature" in str(exc_info.value.detail["message"])


async def test_empty_request_id_claim(mock_keys, mock_redis):
    token = create_token(request_id="")
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "request_id" in str(exc_info.value.detail["message"])


async def test_empty_jti_claim(mock_keys, mock_redis):
    token = create_token(jti="")
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "jti" in str(exc_info.value.detail["message"])


async def test_replayed_jti(mock_keys, mock_redis):
    # Set returns None when key already exists
    mock_redis.set.return_value = None
    token = create_token(jti="used-jti")
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail["code"] == "AUTH_REPLAY_DETECTED"


async def test_wrong_issuer(mock_keys, mock_redis):
    token = create_token(iss="wrong-issuer")
    with pytest.raises(HTTPException) as exc_info:
        await verify_internal_jwt(f"Bearer {token}")
    assert exc_info.value.status_code == 401
    assert "Invalid token" in str(exc_info.value.detail["message"])


async def test_redis_unavailable_uses_postgres_fallback(mock_keys, mock_redis):
    mock_redis.set.side_effect = Exception("Redis connection refused")
    token = create_token()
    ctx = await verify_internal_jwt(f"Bearer {token}")
    assert ctx.uid == "user123"


async def test_valid_token(mock_keys, mock_redis):
    token = create_token()
    ctx = await verify_internal_jwt(f"Bearer {token}")
    assert ctx.uid == "user123"
    assert not ctx.is_admin
    assert ctx.feature == "test"
    assert ctx.request_id == "req-123"
    mock_redis.set.assert_called_once_with("jwt:jti:jti-123", "1", nx=True, ex=60)

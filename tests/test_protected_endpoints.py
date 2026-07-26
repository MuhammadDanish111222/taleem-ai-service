import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app.main import app

# Generate RSA key pair for test
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

client = TestClient(app)


@pytest.fixture
def mock_redis():
    with patch("app.core.internal_auth.get_redis") as mock_get_redis:
        mock_client = MagicMock()
        mock_client.set.return_value = True
        mock_get_redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_keys():
    with patch("app.core.internal_auth.get_public_keys") as mock_get_keys:
        mock_get_keys.return_value = {"test-kid": public_pem}
        yield mock_get_keys


def test_unsigned_direct_request_rejected():
    """Unsigned direct request with no auth header must be rejected with 401."""
    response = client.get("/api/v1/internal/verify")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AUTH_INVALID_TOKEN"


def test_malformed_token_rejected():
    """Malformed token header must be rejected with 401."""
    response = client.get(
        "/api/v1/internal/verify", headers={"Authorization": "Bearer invalid.jwt.str"}
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AUTH_INVALID_TOKEN"


def test_unsigned_direct_local_admin_request_rejected():
    """The Phase 3F control plane never accepts an unsigned direct request."""
    response = client.post(
        "/api/v1/internal/admin/rag",
        json={"operation": "overview", "board_id": "b", "class_id": "c", "subject_id": "s"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "AUTH_INVALID_TOKEN"


def test_signed_non_admin_local_admin_request_rejected(mock_keys, mock_redis):
    """A valid internal JWT still needs its explicit admin authorization claim."""
    now = int(time.time())
    token = jwt.encode(
        {
            "uid": "non-admin",
            "admin": False,
            "feature": "admin_portal",
            "request_id": "non-admin-request",
            "aud": "taleem-ai-service",
            "iss": "taleem-web",
            "jti": "non-admin-jti",
            "iat": now,
            "exp": now + 60,
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    response = client.post(
        "/api/v1/internal/admin/rag",
        headers={"Authorization": f"Bearer {token}"},
        json={"operation": "overview", "board_id": "b", "class_id": "c", "subject_id": "s"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "FORBIDDEN_NOT_ADMIN"


def test_valid_internal_jwt_accepted(mock_keys, mock_redis):
    """Valid signed internal JWT reaches protected endpoint with identity preserved."""
    now = int(time.time())
    payload = {
        "uid": "user-bff-777",
        "admin": True,
        "feature": "admin_portal",
        "request_id": "req-bff-0001",
        "aud": "taleem-ai-service",
        "iss": "taleem-web",
        "jti": "jti-bff-9999",
        "iat": now,
        "exp": now + 60,
    }
    token = jwt.encode(
        payload, private_pem, algorithm="RS256", headers={"kid": "test-kid"}
    )

    response = client.get(
        "/api/v1/internal/verify", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "authenticated"
    assert data["uid"] == "user-bff-777"
    assert data["is_admin"] is True
    assert data["feature"] == "admin_portal"
    assert data["request_id"] == "req-bff-0001"

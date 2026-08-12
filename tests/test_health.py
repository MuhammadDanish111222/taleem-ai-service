from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service_name" in data


def test_readiness_check(monkeypatch):
    class Connection:
        async def execute(self, query):
            assert query == "SELECT 1"

    @asynccontextmanager
    async def database_connection():
        yield Connection()

    monkeypatch.setattr("app.api.v1.health.get_db_connection", database_connection)
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_returns_service_unavailable_when_database_is_unavailable(
    monkeypatch,
):
    @asynccontextmanager
    async def unavailable_database_connection():
        raise RuntimeError("database unavailable")
        yield

    monkeypatch.setattr(
        "app.api.v1.health.get_db_connection", unavailable_database_connection
    )
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}

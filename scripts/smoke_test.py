import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging

from fastapi.testclient import TestClient

from app.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_smoke_test():
    logger.info("Starting smoke test...")
    try:
        # Use TestClient to verify the app can startup without missing env vars crashing it
        client = TestClient(app)
        response = client.get("/api/v1/health")
        if response.status_code == 200:
            logger.info("Smoke test passed! App started successfully.")
            sys.exit(0)
        else:
            logger.error(
                f"Health check failed with status code: {response.status_code}"
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"App failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_smoke_test())

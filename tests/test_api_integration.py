from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.config import load_settings
from src.api.main import create_app
from src.api.model_service import ModelService
from src.data.policy_data import ALLOWED_CATEGORIES

SETTINGS = load_settings()
MODEL_PATH = Path(SETTINGS["model"]["path"])

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"local model directory not found: {MODEL_PATH}",
)


def test_real_local_model_chat_integration():
    service = ModelService(SETTINGS)
    app = create_app(
        settings=SETTINGS,
        model_service=service,
        load_on_startup=True,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        health_body = health.json()
        assert health_body["model_loaded"] is True, health_body
        assert health_body["model_error"] is None
        assert health_body["model_load_seconds"] is not None
        assert health_body["model_load_seconds"] > 0

        messages = [
            "How can I return a damaged product?",
            "What is the return window for a standard product?",
        ]
        for message in messages:
            response = client.post("/chat", json={"message": message})
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "success"
            assert isinstance(body["answer"], str)
            assert body["answer"].strip()
            assert body["category"] in ALLOWED_CATEGORIES
            assert body["latency_ms"] >= 0
            assert isinstance(body["fallback_used"], bool)
            if body["fallback_used"]:
                assert isinstance(body["fallback_reason"], str)
                assert body["fallback_reason"]
            else:
                assert body["fallback_reason"] is None

        metrics = client.get("/metrics").json()
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 2
        assert metrics["failed_requests"] == 0
        assert metrics["latency_samples"] == 2
        assert metrics["p50_latency_ms"] > 0

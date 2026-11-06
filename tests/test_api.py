from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.model_service import ModelInferenceError, ModelResult

TEST_SETTINGS = {
    "app": {
        "name": "Test ShopEase API",
        "version": "test",
        "description": "Test application.",
    },
    "metrics": {"latency_window": 100},
}


class FakeModelService:
    def __init__(
        self,
        answer: str = "Please share your order ID so we can help.",
        load_error: str | None = None,
        inference_error: Exception | None = None,
        latencies: list[float] | None = None,
        loaded: bool = True,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        self.answer = answer
        self.load_error = load_error
        self.inference_error = inference_error
        self.fallback_used = fallback_used
        self.fallback_reason = fallback_reason
        self._latencies = list(latencies or [12.5])
        self._loaded = loaded and load_error is None
        self.model_id = "test-model"
        self.device = "cpu"
        self.load_seconds = 0.01
        self.load_calls = 0
        self.generate_calls: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error:
            self._loaded = False
            raise RuntimeError(self.load_error)
        self._loaded = True

    def generate(self, message: str) -> ModelResult:
        self.generate_calls.append(message)
        if self.inference_error:
            raise self.inference_error
        latency = (
            self._latencies.pop(0)
            if len(self._latencies) > 1
            else self._latencies[0]
        )
        return ModelResult(
            answer=self.answer,
            category="returns",
            latency_ms=latency,
            output_tokens=12,
            category_source="fallback_classifier",
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
        )

    def close(self) -> None:
        self._loaded = False


def make_client(
    service: FakeModelService,
    load_on_startup: bool = False,
) -> TestClient:
    app = create_app(
        settings=TEST_SETTINGS,
        model_service=service,
        load_on_startup=load_on_startup,
    )
    return TestClient(app)


def test_root_returns_application_metadata():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Test ShopEase API"
    assert body["status"] == "ok"
    assert body["docs_url"] == "/docs"
    assert body["chat_ui_url"] == "/ui"


def test_health_reports_loaded_model():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_id"] == "test-model"
    assert body["model_error"] is None


def test_chat_returns_model_answer_and_category():
    service = FakeModelService(answer="Please share the damaged item photos.")
    with make_client(service) as client:
        response = client.post(
            "/chat",
            json={"message": "How can I return a damaged product?"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "answer": "Please share the damaged item photos.",
        "category": "returns",
        "status": "success",
        "latency_ms": 12.5,
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert service.generate_calls == ["How can I return a damaged product?"]


def test_chat_exposes_safe_fallback_status():
    service = FakeModelService(
        answer="Please rephrase your question or contact ShopEase support.",
        fallback_used=True,
        fallback_reason="repetitive_output",
    )
    with make_client(service) as client:
        response = client.post(
            "/chat",
            json={"message": "What is the return window?"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["fallback_reason"] == "repetitive_output"


def test_repeated_requests_reuse_one_loaded_model():
    service = FakeModelService(loaded=False)
    with make_client(service, load_on_startup=True) as client:
        first = client.post("/chat", json={"message": "First question"})
        second = client.post("/chat", json={"message": "Second question"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert service.load_calls == 1
    assert service.generate_calls == ["First question", "Second question"]


def test_empty_message_is_rejected_and_counted_as_failure():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.post("/chat", json={"message": ""})
        metrics = client.get("/metrics").json()
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error"] == "invalid_request"
    assert "empty" in body["detail"][0]["message"].lower()
    assert metrics["total_requests"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["successful_requests"] == 0


def test_whitespace_only_message_is_rejected():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.post("/chat", json={"message": "   \n\t  "})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert "empty" in response.json()["detail"][0]["message"].lower()


def test_message_over_maximum_length_is_rejected():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.post("/chat", json={"message": "x" * 1001})
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert "1000" in detail["message"]


def test_message_at_maximum_length_is_accepted():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.post("/chat", json={"message": "x" * 1000})
    assert response.status_code == 200


def test_invalid_json_is_rejected():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.post(
            "/chat",
            content="{not valid json",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert body["detail"][0]["type"] == "json_invalid"


def test_model_loading_failure_starts_degraded_and_chat_returns_503():
    service = FakeModelService(load_error="model files missing")
    with make_client(service, load_on_startup=True) as client:
        health = client.get("/health")
        chat = client.post("/chat", json={"message": "Hello"})
        metrics = client.get("/metrics").json()
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["model_loaded"] is False
    assert "model files missing" in health.json()["model_error"]
    assert chat.status_code == 503
    assert chat.json()["detail"]["error"] == "model_unavailable"
    assert metrics["failed_requests"] == 1


def test_inference_failure_returns_502_and_counts_failure():
    service = FakeModelService(
        inference_error=ModelInferenceError("synthetic inference failure")
    )
    with make_client(service) as client:
        response = client.post("/chat", json={"message": "Hello"})
        metrics = client.get("/metrics").json()
    assert response.status_code == 502
    assert response.json()["detail"]["error"] == "inference_failed"
    assert metrics["total_requests"] == 1
    assert metrics["failed_requests"] == 1


def test_metrics_report_actual_latency_samples():
    service = FakeModelService(latencies=[10.0, 20.0, 30.0])
    with make_client(service) as client:
        for message in ("one", "two", "three"):
            assert client.post("/chat", json={"message": message}).status_code == 200
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] == 3
    assert body["successful_requests"] == 3
    assert body["failed_requests"] == 0
    assert body["average_latency_ms"] == 20.0
    assert body["p50_latency_ms"] == 20.0
    assert body["p95_latency_ms"] == 30.0
    assert body["latency_samples"] == 3


def test_chat_ui_is_served():
    service = FakeModelService()
    with make_client(service) as client:
        response = client.get("/ui")
    assert response.status_code == 200
    assert "ShopEase Support SLM" in response.text

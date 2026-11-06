from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_CHARS = 1000


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Customer support message. Leading/trailing whitespace is trimmed.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("message must not be empty or whitespace-only")
        if len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(
                f"message must be at most {MAX_MESSAGE_CHARS} characters"
            )
        return text


class ChatResponse(BaseModel):
    answer: str
    category: str
    status: Literal["success"]
    latency_ms: float
    fallback_used: bool = False
    fallback_reason: str | None = None


class RootResponse(BaseModel):
    name: str
    description: str
    status: Literal["ok", "degraded"]
    version: str
    docs_url: str
    chat_ui_url: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_id: str
    device: str
    model_load_seconds: float | None
    model_error: str | None
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    latency_samples: int

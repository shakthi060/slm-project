from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.config import load_settings
from src.api.metrics import MetricsStore
from src.api.model_service import ModelInferenceError, ModelService
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    RootResponse,
)

logger = logging.getLogger("shopease.api")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _validation_details(exc: RequestValidationError) -> list[dict]:
    details = []
    for error in exc.errors():
        details.append(
            {
                "location": list(error.get("loc", [])),
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "invalid_request"),
            }
        )
    return details


def create_app(
    settings: dict | None = None,
    model_service: ModelService | None = None,
    load_on_startup: bool = True,
) -> FastAPI:
    settings = settings or load_settings()
    service = model_service or ModelService(settings)
    metrics = MetricsStore(
        latency_window=int(settings.get("metrics", {}).get("latency_window", 500))
    )
    app_settings = settings.get("app", {})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.monotonic()
        if load_on_startup and not service.is_loaded:
            try:
                service.load()
            except Exception:
                logger.exception("Model loading failed; service is degraded")
        yield
        service.close()

    app = FastAPI(
        title=str(app_settings.get("name", "ShopEase Customer Support SLM API")),
        description=str(
            app_settings.get(
                "description",
                "Local FastAPI service for the ShopEase customer-support SLM.",
            )
        ),
        version=str(app_settings.get("version", "0.1.0")),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = service
    app.state.metrics = metrics
    app.state.started_at = time.monotonic()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        if request.url.path == "/chat":
            request.app.state.metrics.record_failure()
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "error": "invalid_request",
                "detail": _validation_details(exc),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled API error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "internal_server_error",
                "message": "The server could not process the request.",
            },
        )

    @app.get("/", response_model=RootResponse)
    def root() -> RootResponse:
        status = "ok" if service.is_loaded else "degraded"
        return RootResponse(
            name=str(app_settings.get("name", "ShopEase Customer Support SLM API")),
            description=str(
                app_settings.get(
                    "description",
                    "Local FastAPI service for the ShopEase customer-support SLM.",
                )
            ),
            status=status,
            version=str(app_settings.get("version", "0.1.0")),
            docs_url="/docs",
            chat_ui_url="/ui",
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if service.is_loaded else "degraded",
            model_loaded=service.is_loaded,
            model_id=service.model_id,
            device=service.device,
            model_load_seconds=service.load_seconds,
            model_error=service.load_error,
            uptime_seconds=round(
                time.monotonic() - app.state.started_at, 3
            ),
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        if not service.is_loaded:
            metrics.record_failure()
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "error": "model_unavailable",
                    "message": "The local model is not loaded.",
                },
            )

        try:
            result = service.generate(payload.message)
        except ModelInferenceError:
            metrics.record_failure()
            raise HTTPException(
                status_code=502,
                detail={
                    "status": "error",
                    "error": "inference_failed",
                    "message": "The local model could not produce a response.",
                },
            )
        except Exception:
            logger.exception("Unexpected inference error")
            metrics.record_failure()
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "error": "internal_server_error",
                    "message": "The server could not process the request.",
                },
            )

        if result.fallback_used:
            logger.warning(
                "Serving safe fallback for category=%s reason=%s",
                result.category,
                result.fallback_reason,
            )

        metrics.record_success(result.latency_ms)
        return ChatResponse(
            answer=result.answer,
            category=result.category,
            status="success",
            latency_ms=result.latency_ms,
            fallback_used=result.fallback_used,
            fallback_reason=result.fallback_reason,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def get_metrics() -> MetricsResponse:
        return MetricsResponse(**metrics.snapshot())

    @app.get("/ui", include_in_schema=False)
    def chat_ui():
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="Chat UI not installed.")
        return FileResponse(index)

    return app


app = create_app()

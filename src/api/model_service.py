from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.api.output_guard import (
    SAFE_FALLBACK,
    SAFE_OUT_OF_SCOPE,
    detect_repetition,
    is_malformed,
)
from src.api.policy_context import PolicyRetriever, classify_request
from src.data.policy_data import POLICY_DOCUMENT, load_system_prompt
from src.evaluation.transformer_baseline import (
    CATEGORY_PATTERN,
    generate as generate_text,
    load_model,
    parse_category,
)

MAX_ANSWER_CHARS = 4000

PROMPT_LEAK_MARKERS = (
    "You are the ShopEase Customer Support Assistant.",
    "APPLICATION SAFEGUARDS:",
    "Relevant ShopEase policy information:\n",
    "ShopEase policy information:\n",
)


class ModelInferenceError(RuntimeError):
    """Raised when the local model cannot produce a safe response."""


class UnsafeModelOutput(ModelInferenceError):
    """Raised when model output is empty, malformed, or excessively repetitive."""


@dataclass(frozen=True)
class ModelResult:
    answer: str
    category: str
    latency_ms: float
    output_tokens: int
    category_source: str
    fallback_used: bool = False
    fallback_reason: str | None = None


def build_grounded_system_prompt(
    base_prompt: str | None = None,
    policy_document: str | None = None,
    include_document: bool = True,
    policy_context: str | None = None,
) -> str:
    base = (base_prompt or load_system_prompt()).strip()
    policy = policy_context if policy_context is not None else (
        policy_document or POLICY_DOCUMENT
    )
    safeguards = (
        "\n\nAPPLICATION SAFEGUARDS:\n"
        "- Answer only from the relevant ShopEase policy information supplied below.\n"
        "- Do not invent delivery dates, return periods, refund timelines, coupon rules, "
        "order statuses, account facts, or contact information.\n"
        "- If the policy does not contain the answer, say: "
        "\"I cannot verify that information from the ShopEase policy.\"\n"
        "- Ask one concise clarification question when essential information is missing.\n"
        "- Politely refuse unrelated requests.\n"
        "- Never ask for an OTP, PIN, password, or full card details.\n"
        "- Do not claim to retrieve live order data.\n"
        "- Answer in at most three short sentences. Do not repeat points or list items."
    )
    if include_document and policy.strip():
        return (
            f"{base}{safeguards}\n\n"
            f"Relevant ShopEase policy information:\n{policy}"
        )
    return f"{base}{safeguards}"


def sanitize_model_answer(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    tag_category = parse_category(text)
    text = CATEGORY_PATTERN.sub("", text).strip()
    if not text:
        raise UnsafeModelOutput("model returned an empty response")
    if any(marker in text for marker in PROMPT_LEAK_MARKERS):
        raise UnsafeModelOutput("model response contained prompt content")
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip() + "..."
    return text, tag_category


class ModelService:
    """Owns the single loaded model instance and serializes inference."""

    def __init__(
        self,
        settings: dict,
        loader: Callable = load_model,
        generator: Callable = generate_text,
        system_prompt: str | None = None,
        policy_retriever: PolicyRetriever | None = None,
    ) -> None:
        self.settings = settings
        self._loader = loader
        self._generator = generator
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._model = None
        self._tokenizer = None

        model_settings = settings.get("model", {})
        self.model_id = str(model_settings.get("id", "unknown"))
        self.model_path = Path(model_settings.get("path", ""))
        self.revision = model_settings.get("revision")
        self.device = str(model_settings.get("device", "cpu"))
        self.use_4bit = bool(model_settings.get("use_4bit", False))
        self.load_seconds: float | None = None
        self.load_error: str | None = None

        policy_settings = settings.get("policy", {})
        self.base_prompt = system_prompt or load_system_prompt()
        self.include_policy = bool(policy_settings.get("include_document", True))
        self.policy_retriever = policy_retriever or PolicyRetriever()
        self.system_prompt = build_grounded_system_prompt(
            base_prompt=self.base_prompt,
            include_document=self.include_policy,
        )
        self.generation = dict(settings.get("generation", {}))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _build_prompt(self, context_text: str) -> str:
        if self.include_policy and context_text.strip():
            return build_grounded_system_prompt(
                base_prompt=self.base_prompt,
                policy_context=context_text,
                include_document=True,
            )
        return build_grounded_system_prompt(
            base_prompt=self.base_prompt,
            include_document=False,
        )

    def _fallback(
        self,
        answer: str,
        category: str,
        reason: str,
        category_source: str,
        latency_ms: float = 0.0,
        output_tokens: int = 0,
    ) -> ModelResult:
        return ModelResult(
            answer=answer,
            category=category,
            latency_ms=round(float(latency_ms), 3),
            output_tokens=int(output_tokens),
            category_source=category_source,
            fallback_used=True,
            fallback_reason=reason,
        )

    def load(self) -> None:
        with self._load_lock:
            if self.is_loaded:
                return
            if not self.model_path.exists():
                self.load_error = (
                    f"model directory not found: {self.model_path}"
                )
                raise FileNotFoundError(self.load_error)

            import torch

            runtime = self.settings.get("runtime", {})
            try:
                torch.set_num_threads(int(runtime.get("threads", 6)))
                torch.set_num_interop_threads(
                    int(runtime.get("interop_threads", 1))
                )
            except (RuntimeError, ValueError):
                pass

            start = time.perf_counter()
            try:
                self._model, self._tokenizer = self._loader(
                    self.model_path,
                    self.revision,
                    self.use_4bit,
                    self.device,
                )
                self.load_seconds = round(time.perf_counter() - start, 3)
                self.load_error = None

                model_settings = self.settings.get("model", {})
                if bool(model_settings.get("warmup", False)):
                    warmup_generation = dict(self.generation)
                    warmup_generation["max_new_tokens"] = int(
                        model_settings.get("warmup_max_new_tokens", 16)
                    )
                    warmup_prompt = str(
                        model_settings.get(
                            "warmup_prompt",
                            "How can I return a damaged product?",
                        )
                    )
                    warmup_context = self.policy_retriever.retrieve(
                        warmup_prompt
                    )
                    warmup_messages = [
                        {
                            "role": "system",
                            "content": self._build_prompt(warmup_context.text),
                        },
                        {"role": "user", "content": warmup_prompt},
                    ]
                    with self._inference_lock:
                        self._generator(
                            self._model,
                            self._tokenizer,
                            warmup_messages,
                            warmup_generation,
                        )
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"
                self._model = None
                self._tokenizer = None
                self.load_seconds = None
                raise

    def generate(self, message: str) -> ModelResult:
        if not self.is_loaded:
            raise ModelInferenceError(
                self.load_error or "model is not loaded"
            )

        category = classify_request(message)
        if category == "out_of_scope":
            return self._fallback(
                answer=SAFE_OUT_OF_SCOPE,
                category="out_of_scope",
                reason="out_of_scope_guard",
                category_source="policy_guard",
            )

        context = self.policy_retriever.retrieve(message, category=category)
        if not context.found:
            return self._fallback(
                answer=SAFE_FALLBACK,
                category=category,
                reason="missing_policy_context",
                category_source="policy_guard",
            )

        messages = [
            {"role": "system", "content": self._build_prompt(context.text)},
            {"role": "user", "content": message},
        ]
        with self._inference_lock:
            try:
                raw_answer, latency_ms, output_tokens = self._generator(
                    self._model,
                    self._tokenizer,
                    messages,
                    self.generation,
                )
            except Exception as exc:
                raise ModelInferenceError(
                    f"inference failed: {type(exc).__name__}: {exc}"
                ) from exc

        try:
            answer, tag_category = sanitize_model_answer(raw_answer)
            if is_malformed(answer):
                raise UnsafeModelOutput("malformed_output")
            repetition_reasons = detect_repetition(answer)
            if repetition_reasons:
                raise UnsafeModelOutput(
                    "repetitive_output:" + ",".join(repetition_reasons)
                )
        except UnsafeModelOutput as exc:
            return self._fallback(
                answer=SAFE_FALLBACK,
                category=category,
                reason=str(exc),
                category_source="fallback_classifier",
                latency_ms=latency_ms,
                output_tokens=output_tokens,
            )

        if tag_category != "unknown":
            category = tag_category
            category_source = "model_tag"
        else:
            category_source = "fallback_classifier"

        return ModelResult(
            answer=answer,
            category=category,
            latency_ms=round(float(latency_ms), 3),
            output_tokens=int(output_tokens),
            category_source=category_source,
            fallback_used=False,
            fallback_reason=None,
        )

    def close(self) -> None:
        with self._inference_lock:
            self._model = None
            self._tokenizer = None
            gc.collect()

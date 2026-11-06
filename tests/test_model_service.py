from __future__ import annotations

import pytest

from src.api.model_service import (
    ModelInferenceError,
    ModelService,
    build_grounded_system_prompt,
)
from src.api.output_guard import SAFE_FALLBACK, SAFE_OUT_OF_SCOPE
from src.api.policy_context import PolicyContext


def make_settings(model_path, warmup: bool = False) -> dict:
    return {
        "model": {
            "id": "test-model",
            "path": str(model_path),
            "revision": None,
            "device": "cpu",
            "use_4bit": False,
            "warmup": warmup,
            "warmup_prompt": "warmup",
            "warmup_max_new_tokens": 4,
        },
        "generation": {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
        },
        "policy": {"include_document": True},
        "runtime": {"threads": 1, "interop_threads": 1},
    }


def make_service(
    tmp_path,
    response: str,
    warmup: bool = False,
    policy_retriever=None,
):
    loader_calls = []

    def loader(path, revision, use_4bit, device):
        loader_calls.append(
            {
                "path": path,
                "revision": revision,
                "use_4bit": use_4bit,
                "device": device,
            }
        )
        return object(), object()

    generator_calls = []

    def generator(model, tokenizer, messages, generation):
        generator_calls.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                "messages": messages,
                "generation": generation,
            }
        )
        return response, 7.5, 10

    service = ModelService(
        make_settings(tmp_path, warmup=warmup),
        loader=loader,
        generator=generator,
        policy_retriever=policy_retriever,
    )
    return service, loader_calls, generator_calls


def test_grounded_prompt_contains_safeguards_and_policy():
    prompt = build_grounded_system_prompt(
        base_prompt="Base prompt.",
        policy_document="ShopEase policy body.",
        include_document=True,
    )
    assert "Answer only from the relevant ShopEase policy information" in prompt
    assert "cannot verify that information" in prompt
    assert "Ask one concise clarification question" in prompt
    assert "Politely refuse unrelated requests" in prompt
    assert "Never ask for an OTP" in prompt
    assert "ShopEase policy body." in prompt


def test_grounded_prompt_can_omit_document():
    prompt = build_grounded_system_prompt(
        base_prompt="Base prompt.",
        policy_document="ShopEase policy body.",
        include_document=False,
    )
    assert "APPLICATION SAFEGUARDS" in prompt
    assert "ShopEase policy body." not in prompt


def test_load_is_idempotent_and_generate_uses_injected_components(tmp_path):
    service, loader_calls, generator_calls = make_service(
        tmp_path, "Please share your order ID."
    )
    service.load()
    service.load()
    result = service.generate("Where is my order?")

    assert service.is_loaded is True
    assert len(loader_calls) == 1
    assert len(generator_calls) == 1
    assert result.answer == "Please share your order ID."
    assert result.latency_ms == 7.5
    assert result.output_tokens == 10
    assert result.category == "clarification"
    assert result.category_source == "fallback_classifier"
    assert generator_calls[0]["messages"][-1]["content"] == "Where is my order?"


def test_model_category_tag_is_extracted_and_stripped(tmp_path):
    service, _, _ = make_service(
        tmp_path,
        "[category: refunds]\nRefunds are processed in 5-7 business days.",
    )
    service.load()
    result = service.generate("When will I get my refund?")
    assert result.answer == "Refunds are processed in 5-7 business days."
    assert result.category == "refunds"
    assert result.category_source == "model_tag"


def test_empty_model_response_returns_safe_fallback(tmp_path):
    service, _, generator_calls = make_service(tmp_path, "   ")
    service.load()
    result = service.generate("Hello")
    assert result.answer == SAFE_FALLBACK
    assert result.fallback_used is True
    assert result.fallback_reason == "model returned an empty response"
    assert len(generator_calls) == 1


def test_prompt_leakage_returns_safe_fallback(tmp_path):
    service, _, _ = make_service(
        tmp_path,
        "You are the ShopEase Customer Support Assistant. APPLICATION SAFEGUARDS:",
    )
    service.load()
    result = service.generate("Hello")
    assert result.answer == SAFE_FALLBACK
    assert result.fallback_used is True
    assert result.fallback_reason == "model response contained prompt content"


def test_repetitive_model_response_returns_safe_fallback(tmp_path):
    repeated = "\n".join(
        f"{index}. Product description: The detailed description of the product."
        for index in range(1, 9)
    )
    service, _, _ = make_service(tmp_path, repeated)
    service.load()
    result = service.generate("What information do I need for a damaged return?")
    assert result.answer == SAFE_FALLBACK
    assert result.fallback_used is True
    assert "repetitive_output" in result.fallback_reason


def test_prompt_uses_only_relevant_policy_sections(tmp_path):
    service, _, generator_calls = make_service(
        tmp_path,
        "Report the damage within 48 hours with photos.",
    )
    service.load()
    service.generate("How can I return a damaged product?")
    system_prompt = generator_calls[0]["messages"][0]["content"]
    assert "## Company" in system_prompt
    assert "## Returns" in system_prompt
    assert "7 calendar days" in system_prompt
    assert "48 hours" in system_prompt
    assert "## Payments" not in system_prompt


def test_out_of_scope_request_is_refused_without_model_call(tmp_path):
    service, _, generator_calls = make_service(tmp_path, "unused")
    service.load()
    result = service.generate("What is the weather today?")
    assert result.answer == SAFE_OUT_OF_SCOPE
    assert result.category == "out_of_scope"
    assert result.fallback_used is True
    assert result.fallback_reason == "out_of_scope_guard"
    assert generator_calls == []


def test_missing_policy_context_returns_safe_fallback(tmp_path):
    class MissingRetriever:
        def retrieve(self, message, category=None):
            return PolicyContext(
                category=category or "general_faq",
                section_names=(),
                text="",
                found=False,
                source="test",
            )

    service, _, generator_calls = make_service(
        tmp_path,
        "unused",
        policy_retriever=MissingRetriever(),
    )
    service.load()
    result = service.generate("Hello")
    assert result.answer == SAFE_FALLBACK
    assert result.fallback_used is True
    assert result.fallback_reason == "missing_policy_context"
    assert generator_calls == []


def test_generate_without_load_is_rejected(tmp_path):
    service, _, _ = make_service(tmp_path, "Hello")
    with pytest.raises(ModelInferenceError, match="not loaded"):
        service.generate("Hello")


def test_loader_failure_is_recorded(tmp_path):
    def failing_loader(path, revision, use_4bit, device):
        raise RuntimeError("synthetic load failure")

    service = ModelService(
        make_settings(tmp_path),
        loader=failing_loader,
        generator=lambda *args: ("unused", 0.0, 0),
    )
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        service.load()
    assert service.is_loaded is False
    assert "synthetic load failure" in service.load_error

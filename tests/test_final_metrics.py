from src.data.policy_data import POLICY_DOCUMENT
from src.evaluation.final_metrics import (
    category_signals,
    hallucination_flags,
    response_validity,
    summarize_flags,
)


def test_response_validity_rejects_truncation_and_prompt_leakage():
    valid, reasons = response_validity("Please share your order ID.", truncated=True)
    assert valid is False
    assert "truncated_at_token_cap" in reasons

    valid, reasons = response_validity(
        "You are the ShopEase Customer Support Assistant. " * 2,
        truncated=False,
    )
    assert valid is False
    assert "prompt_leakage" in reasons


def test_hallucination_flags_detect_fabricated_contact_and_return_window():
    flags = hallucination_flags(
        "Returns take 24 to 72 hours. Call (866) 531-0999 or email customer@shopesale.com.",
        "How long do I have to return a product?",
        "Most items can be returned within 7 calendar days.",
        "returns",
        POLICY_DOCUMENT,
    )
    assert "return_window_mismatch" in flags
    assert "fabricated_phone" in flags
    assert "fabricated_email" in flags


def test_hallucination_flags_allow_negated_policy_statements():
    flags = hallucination_flags(
        "ShopEase does not offer phone support. Contact support@shopease.example. "
        "Coupons cannot be combined; one coupon per order.",
        "Can I combine coupons?",
        "One coupon per order; coupons cannot be stacked.",
        "general_faq",
        POLICY_DOCUMENT,
    )
    assert flags == []


def test_hallucination_flags_detect_coupon_stacking_claim():
    flags = hallucination_flags(
        "You can combine coupons on one order.",
        "Can I combine coupons?",
        "One coupon per order.",
        "general_faq",
        POLICY_DOCUMENT,
    )
    assert "coupon_stacking_claim" in flags


def test_hallucination_flags_detect_specific_order_status_claim():
    flags = hallucination_flags(
        "Your order was delivered on 31 December 2023.",
        "Where is my order?",
        "Please share your order ID so we can check.",
        "order_status",
        POLICY_DOCUMENT,
    )
    assert "fabricated_date" in flags
    assert "specific_order_status_claim" in flags


def test_hallucination_flags_allow_safe_order_status_response():
    flags = hallucination_flags(
        "I cannot see live order details. Please share your order ID SE-12345678.",
        "Where is my order?",
        "Please share your order ID so we can check.",
        "order_status",
        POLICY_DOCUMENT,
    )
    assert flags == []


def test_category_signals_use_fallback_when_tag_missing():
    signals = category_signals(
        "Refunds are processed in 5-7 business days to the original payment method.",
        "refunds",
    )
    assert signals["tag_category"] == "unknown"
    assert signals["fallback_category"] == "refunds"
    assert signals["combined_correct"] is True


def test_summarize_flags_counts_occurrences():
    rows = [
        {"heuristic_hallucination_flags": ["fabricated_phone", "fabricated_date"]},
        {"heuristic_hallucination_flags": ["fabricated_phone"]},
        {"heuristic_hallucination_flags": []},
    ]
    assert summarize_flags(rows) == {
        "fabricated_date": 1,
        "fabricated_phone": 2,
    }

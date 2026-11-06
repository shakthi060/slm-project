from __future__ import annotations

from src.api.output_guard import detect_repetition, is_malformed


def test_clean_policy_answer_is_not_flagged():
    answer = (
        "Damaged, wrong, or defective items must be reported within 48 hours "
        "of delivery with photos. Return pickup is free in these cases. "
        "You can start the request from My Orders > Return/Exchange."
    )
    assert detect_repetition(answer) == []
    assert is_malformed(answer) is False


def test_repeated_line_cycle_is_detected():
    items = [
        "Product description: The detailed description of the product.",
        "Product model: The type of product.",
        "Product category: The category of the product.",
    ]
    body = "\n".join(
        f"{index}. {items[(index - 1) % len(items)]}"
        for index in range(1, 21)
    )
    reasons = detect_repetition(body)
    assert "repeated_list_items" in reasons
    assert "repeated_sentences" in reasons
    assert "repeated_phrases" in reasons


def test_short_distinct_list_is_not_flagged():
    answer = (
        "1. Report the damage within 48 hours.\n"
        "2. Attach clear photos.\n"
        "3. Keep the original packaging.\n"
        "4. Start the request from My Orders.\n"
        "5. Return pickup is free."
    )
    assert detect_repetition(answer) == []


def test_empty_and_symbol_only_output_is_malformed():
    assert is_malformed("   ") is True
    assert is_malformed("...---...") is True
    assert is_malformed("Valid answer.") is False

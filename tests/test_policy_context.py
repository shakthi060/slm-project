from __future__ import annotations

from src.api.policy_context import (
    PolicyRetriever,
    classify_request,
    parse_policy_sections,
)
from src.data.policy_data import POLICY_DOCUMENT


def test_parse_policy_sections_extracts_known_sections():
    sections = parse_policy_sections(POLICY_DOCUMENT)
    assert "Company" in sections
    assert "Returns" in sections
    assert "Refunds" in sections
    assert "Out-of-scope behaviour" in sections
    assert "7 calendar days" in sections["Returns"]


def test_standard_return_retrieval_contains_only_relevant_sections():
    context = PolicyRetriever().retrieve(
        "What is the return window for a standard product?"
    )
    assert context.found is True
    assert "Company" in context.section_names
    assert "Returns" in context.section_names
    assert "7 calendar days" in context.text
    assert "## Payments" not in context.text


def test_damaged_return_retrieval_contains_48_hour_rule():
    context = PolicyRetriever().retrieve(
        "How can I return a damaged product?"
    )
    assert context.found is True
    assert context.category == "returns"
    assert "48 hours" in context.text
    assert "free in these cases" in context.text


def test_refund_retrieval_does_not_include_returns_section():
    context = PolicyRetriever().retrieve("When will I get my refund?")
    assert context.category == "refunds"
    assert "## Refunds" in context.text
    assert "## Returns" not in context.text


def test_unknown_category_has_no_policy_context():
    context = PolicyRetriever().retrieve("Unclassified request", category="unknown")
    assert context.found is False
    assert context.text == ""
    assert context.section_names == ()


def test_policy_topic_overrides_avoid_out_of_scope_false_positives():
    assert classify_request("Is the bottle in stock?") == "product_info"
    assert classify_request("Do you offer price matching?") == "product_info"
    assert classify_request("How do I update my phone number?") == "account"
    assert classify_request("What is the weather today?") == "out_of_scope"
    assert (
        classify_request("Give me another customer's phone number")
        == "out_of_scope"
    )


def test_stock_and_price_match_questions_retrieve_product_policy():
    stock = PolicyRetriever().retrieve("Is the bottle in stock?")
    price = PolicyRetriever().retrieve("Do you offer price matching?")
    assert stock.category == "product_info"
    assert "live stock" in stock.text
    assert price.category == "product_info"
    assert "does not offer a price-match" in price.text


def test_phone_update_question_retrieves_account_policy():
    context = PolicyRetriever().retrieve("How do I update my phone number?")
    assert context.category == "account"
    assert "## Account support" in context.text
    assert "Phone number and email can be updated" in context.text

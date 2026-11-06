from pathlib import Path

import pytest

from src.data.clean_dataset import clean_records, record_errors
from src.data.policy_data import ALLOWED_CATEGORIES, load_system_prompt
from src.data.split_dataset import leakage_checks, stratified_split
from src.data.validate_dataset import validate_file


SYSTEM = "You are the ShopEase Customer Support Assistant."


def make_record(
    record_id: str,
    user: str,
    category: str = "returns",
    answer: str = "Please share your order ID so we can help.",
) -> dict:
    return {
        "id": record_id,
        "category": category,
        "intent": "test_intent",
        "requires_clarification": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": f"[category: {category}]\n{answer}"},
        ],
        "source": "test",
    }


def test_valid_record_has_no_errors():
    assert record_errors(make_record("shopease-000001", "How do I return an item?"), ALLOWED_CATEGORIES) == []


def test_invalid_category_is_rejected():
    record = make_record("shopease-000001", "How do I return an item?", category="not_a_category")
    errors = record_errors(record, ALLOWED_CATEGORIES)
    assert any("category" in error for error in errors)


def test_wrong_role_order_is_rejected():
    record = make_record("shopease-000001", "How do I return an item?")
    record["messages"][1], record["messages"][2] = record["messages"][2], record["messages"][1]
    errors = record_errors(record, ALLOWED_CATEGORIES)
    assert any("roles" in error for error in errors)


def test_empty_user_message_is_rejected():
    record = make_record("shopease-000001", "   ")
    errors = record_errors(record, ALLOWED_CATEGORIES)
    assert errors


def test_assistant_category_prefix_must_match_metadata():
    record = make_record("shopease-000001", "How do I return an item?")
    record["messages"][2]["content"] = "[category: refunds]\nPlease share your order ID."
    errors = record_errors(record, ALLOWED_CATEGORIES)
    assert any("category tag" in error for error in errors)


def test_clean_removes_invalid_records():
    good = make_record("shopease-000001", "How do I return an item?")
    bad = make_record("shopease-000002", "Where is my order?", category="unknown")
    cleaned, stats = clean_records([good, bad], ALLOWED_CATEGORIES)
    assert len(cleaned) == 1
    assert stats["invalid_records"] == 1


def test_clean_removes_exact_duplicates():
    first = make_record("shopease-000001", "How do I return an item?")
    second = make_record("shopease-000002", "  how do I return an item?  ")
    cleaned, stats = clean_records([first, second], ALLOWED_CATEGORIES)
    assert len(cleaned) == 1
    assert stats["exact_duplicates"] == 1


def test_clean_removes_duplicate_ids():
    first = make_record("shopease-000001", "How do I return an item?")
    second = make_record("shopease-000001", "Can I return a damaged product?")
    cleaned, stats = clean_records([first, second], ALLOWED_CATEGORIES)
    assert len(cleaned) == 1
    assert stats["duplicate_ids"] == 1


def test_clean_removes_near_duplicates():
    first = make_record("shopease-000001", "Can I return a damaged product?")
    second = make_record("shopease-000002", "Can I return this damaged product?")
    cleaned, stats = clean_records(
        [first, second], ALLOWED_CATEGORIES, near_duplicate_threshold=0.7
    )
    assert len(cleaned) == 1
    assert stats["near_duplicates"] == 1


def test_stratified_split_is_disjoint_and_keeps_every_category():
    records = []
    for category in ALLOWED_CATEGORIES[:3]:
        for index in range(20):
            records.append(
                make_record(
                    f"shopease-{len(records) + 1:06d}",
                    f"Question {index} about {category} with enough words to be unique.",
                    category=category,
                )
            )
    splits = stratified_split(records, seed=42)
    ids = {name: {record["id"] for record in split} for name, split in splits.items()}
    assert not (ids["train"] & ids["validation"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["validation"] & ids["test"])
    for name, split in splits.items():
        assert {record["category"] for record in split} == set(ALLOWED_CATEGORIES[:3])
        assert all(record["split"] == name for record in split)


def test_leakage_checks_detect_overlap():
    record = make_record("shopease-000001", "How do I return an item?")
    splits = {"train": [record], "validation": [dict(record, id="shopease-000002")], "test": []}
    result = leakage_checks(splits)
    assert result["passed"] is False
    assert result["exact_user_text_overlap"]["train_validation"] == 1


@pytest.mark.skipif(
    not Path("data/test.jsonl").exists(), reason="dataset has not been generated yet"
)
def test_generated_dataset_files_validate():
    expected_prompt = load_system_prompt()
    for path, split in [
        (Path("data/train.jsonl"), "train"),
        (Path("data/validation.jsonl"), "validation"),
        (Path("data/test.jsonl"), "test"),
    ]:
        report = validate_file(path, expected_prompt, expected_split=split)
        assert report["status"] == "PASS", report["errors"]
        assert report["missing_categories"] == []

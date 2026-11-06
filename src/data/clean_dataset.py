import argparse
import copy
import json
import re
from pathlib import Path

from src.data.policy_data import ALLOWED_CATEGORIES


def read_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    parse_errors: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(value, dict):
                parse_errors.append(
                    {"line": line_number, "error": "record is not a JSON object"}
                )
                continue
            records.append(value)
    return records, parse_errors


def normalize_for_dedupe(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def record_errors(record: dict, allowed_categories: list[str]) -> list[str]:
    errors: list[str] = []
    required = {"id", "category", "intent", "requires_clarification", "messages", "source"}
    missing = required - set(record)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
        return errors

    if not isinstance(record["id"], str) or not record["id"].strip():
        errors.append("id must be a non-empty string")
    if not isinstance(record["category"], str) or record["category"] not in allowed_categories:
        errors.append("category is not allowed")
    if not isinstance(record["intent"], str) or not record["intent"].strip():
        errors.append("intent must be a non-empty string")
    if not isinstance(record["requires_clarification"], bool):
        errors.append("requires_clarification must be a boolean")
    if not isinstance(record["source"], str) or not record["source"].strip():
        errors.append("source must be a non-empty string")

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append("messages must contain exactly 3 entries")
        return errors

    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if roles != ["system", "user", "assistant"]:
        errors.append("messages roles must be system, user, assistant")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index} is not an object")
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"message {index} content is empty")

    if errors:
        return errors

    system = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"]

    if "ShopEase" not in system:
        errors.append("system prompt does not mention ShopEase")
    if not 3 <= len(user.strip()) <= 1000:
        errors.append("user message length is outside 3..1000 characters")
    prefix = f"[category: {record['category']}]\n"
    if not assistant.startswith(prefix):
        errors.append("assistant message does not start with the matching category tag")
    elif not assistant[len(prefix):].strip():
        errors.append("assistant answer after the category tag is empty")
    if len(assistant.strip()) > 2000:
        errors.append("assistant message is longer than 2000 characters")
    if re.search(r"\{[a-z_]+\}", user) or re.search(r"\{[a-z_]+\}", assistant):
        errors.append("unresolved template placeholder found")

    return errors


def clean_records(
    records: list[dict],
    allowed_categories: list[str],
    near_duplicate_threshold: float = 0.92,
) -> tuple[list[dict], dict]:
    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    seen_exact: set[str] = set()
    known_tokens: list[set[str]] = []
    stats = {
        "input_records": len(records),
        "invalid_records": 0,
        "duplicate_ids": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "kept_records": 0,
        "near_duplicate_threshold": near_duplicate_threshold,
        "invalid_examples": [],
    }

    for record in records:
        errors = record_errors(record, allowed_categories)
        if errors:
            stats["invalid_records"] += 1
            if len(stats["invalid_examples"]) < 10:
                stats["invalid_examples"].append(
                    {"id": record.get("id", "<missing>"), "errors": errors}
                )
            continue

        normalized = copy.deepcopy(record)
        for message in normalized["messages"]:
            message["content"] = message["content"].strip()
        normalized["messages"][1]["content"] = " ".join(
            normalized["messages"][1]["content"].split()
        )

        record_id = normalized["id"]
        if record_id in seen_ids:
            stats["duplicate_ids"] += 1
            continue

        user_text = normalized["messages"][1]["content"]
        exact_key = normalize_for_dedupe(user_text)
        if not exact_key:
            stats["invalid_records"] += 1
            continue
        if exact_key in seen_exact:
            stats["exact_duplicates"] += 1
            continue

        tokens = set(exact_key.split())
        if any(
            jaccard(tokens, previous) >= near_duplicate_threshold
            for previous in known_tokens
        ):
            stats["near_duplicates"] += 1
            continue

        seen_ids.add(record_id)
        seen_exact.add(exact_key)
        known_tokens.append(tokens)
        cleaned.append(normalized)

    stats["kept_records"] = len(cleaned)
    return cleaned, stats


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/raw/shopease_raw.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/shopease_clean.jsonl")
    )
    parser.add_argument(
        "--stats", type=Path, default=Path("reports/cleaning_stats.json")
    )
    parser.add_argument("--near-threshold", type=float, default=0.92)
    args = parser.parse_args()

    records, parse_errors = read_jsonl(args.input)
    cleaned, stats = clean_records(records, ALLOWED_CATEGORIES, args.near_threshold)
    stats["parse_errors"] = len(parse_errors)
    stats["parse_error_examples"] = parse_errors[:10]
    stats["input_file"] = str(args.input)
    stats["output_file"] = str(args.output)

    write_jsonl(args.output, cleaned)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Read {len(records)} records ({len(parse_errors)} parse errors)")
    print(f"Invalid: {stats['invalid_records']}")
    print(f"Duplicate IDs: {stats['duplicate_ids']}")
    print(f"Exact duplicates: {stats['exact_duplicates']}")
    print(f"Near duplicates: {stats['near_duplicates']}")
    print(f"Kept: {stats['kept_records']} -> {args.output}")
    print(f"Stats: {args.stats}")


if __name__ == "__main__":
    main()

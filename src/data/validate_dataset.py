import argparse
import json
from pathlib import Path

from src.data.clean_dataset import (
    normalize_for_dedupe,
    record_errors,
    read_jsonl,
)
from src.data.policy_data import ALLOWED_CATEGORIES, load_system_prompt
from src.data.split_dataset import leakage_checks


def validate_records(
    records: list[dict],
    allowed_categories: list[str],
    expected_system_prompt: str | None = None,
    require_all_categories: bool = True,
) -> dict:
    errors: list[dict] = []
    seen_ids: dict[str, int] = {}
    seen_users: dict[str, int] = {}
    systems: set[str] = set()
    categories: set[str] = set()

    for index, record in enumerate(records):
        for message in record.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str):
                    systems.add(" ".join(content.split()))

        for error in record_errors(record, allowed_categories):
            errors.append({"index": index, "id": record.get("id"), "error": error})

        record_id = record.get("id")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                errors.append(
                    {
                        "index": index,
                        "id": record_id,
                        "error": f"duplicate id (first seen at {seen_ids[record_id]})",
                    }
                )
            else:
                seen_ids[record_id] = index

        messages = record.get("messages")
        if isinstance(messages, list) and len(messages) == 3:
            if isinstance(messages[1], dict) and isinstance(messages[1].get("content"), str):
                key = normalize_for_dedupe(messages[1]["content"])
                if key in seen_users:
                    errors.append(
                        {
                            "index": index,
                            "id": record_id,
                            "error": f"duplicate user text (first seen at {seen_users[key]})",
                        }
                    )
                else:
                    seen_users[key] = index

        category = record.get("category")
        if isinstance(category, str):
            categories.add(category)

    if len(systems) > 1:
        errors.append({"index": None, "id": None, "error": "system prompt is not consistent"})
    if expected_system_prompt is not None:
        expected = " ".join(expected_system_prompt.split())
        if systems and systems != {expected}:
            errors.append(
                {
                    "index": None,
                    "id": None,
                    "error": "system prompt does not match configs/model_config.yaml",
                }
            )

    missing = sorted(set(allowed_categories) - categories)
    if require_all_categories and missing:
        errors.append(
            {"index": None, "id": None, "error": f"missing categories: {missing}"}
        )

    return {
        "records": len(records),
        "error_count": len(errors),
        "errors": errors[:50],
        "duplicate_ids": sum(1 for error in errors if "duplicate id" in error["error"]),
        "duplicate_user_texts": sum(
            1 for error in errors if "duplicate user text" in error["error"]
        ),
        "system_prompts": len(systems),
        "missing_categories": missing,
        "status": "PASS" if not errors else "FAIL",
    }


def validate_file(
    path: Path,
    expected_system_prompt: str | None,
    expected_split: str | None = None,
    require_all_categories: bool = True,
) -> dict:
    records, parse_errors = read_jsonl(path)
    report = validate_records(
        records, ALLOWED_CATEGORIES, expected_system_prompt, require_all_categories
    )
    report["file"] = str(path)
    report["parse_errors"] = parse_errors
    if parse_errors:
        report["status"] = "FAIL"
        report["error_count"] += len(parse_errors)
    if expected_split is not None:
        wrong = sum(
            1 for record in records if record.get("split") != expected_split
        )
        if wrong:
            report["status"] = "FAIL"
            report["error_count"] += wrong
            report["errors"].append(
                {
                    "index": None,
                    "id": None,
                    "error": f"{wrong} records have a split field other than {expected_split}",
                }
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", type=Path, default=None)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/dataset_validation.json")
    )
    args = parser.parse_args()

    default_files = [
        Path("data/raw/shopease_raw.jsonl"),
        Path("data/processed/shopease_clean.jsonl"),
        Path("data/train.jsonl"),
        Path("data/validation.jsonl"),
        Path("data/test.jsonl"),
    ]
    files = args.files or default_files
    system_prompt = load_system_prompt()

    split_names = {
        "data/train.jsonl": "train",
        "data/validation.jsonl": "validation",
        "data/test.jsonl": "test",
    }

    file_reports = []
    for path in files:
        if not path.exists():
            file_reports.append(
                {"file": str(path), "status": "MISSING", "error_count": 1}
            )
            continue
        expected_split = split_names.get(path.as_posix())
        file_reports.append(
            validate_file(path, system_prompt, expected_split=expected_split)
        )

    split_reports = {}
    split_files = {
        "train": Path("data/train.jsonl"),
        "validation": Path("data/validation.jsonl"),
        "test": Path("data/test.jsonl"),
    }
    if all(path.exists() for path in split_files.values()):
        split_records = {}
        for name, path in split_files.items():
            records, _ = read_jsonl(path)
            split_records[name] = records
        split_reports = leakage_checks(split_records)

    overall = "PASS"
    if any(report["status"] != "PASS" for report in file_reports):
        overall = "FAIL"
    if split_reports and not split_reports.get("passed", False):
        overall = "FAIL"

    output = {
        "overall_status": overall,
        "files": file_reports,
        "split_leakage": split_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Overall: {overall}")
    for report in file_reports:
        print(
            f"  {report['file']}: {report['status']} "
            f"({report.get('records', 0)} records, {report.get('error_count', 0)} errors)"
        )
    if split_reports:
        print(f"  split leakage: {'PASS' if split_reports['passed'] else 'FAIL'}")
    print(f"Report: {args.report}")

    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

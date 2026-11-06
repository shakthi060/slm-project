import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from src.data.clean_dataset import (
    jaccard,
    normalize_for_dedupe,
    read_jsonl,
    write_jsonl,
)
from src.data.policy_data import ALLOWED_CATEGORIES


def stratified_split(
    records: list[dict],
    seed: int = 42,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for category in sorted(grouped):
        items = list(grouped[category])
        rng.shuffle(items)
        total = len(items)
        validation_count = max(1, int(total * validation_ratio))
        test_count = max(1, int(total * test_ratio))
        if validation_count + test_count >= total:
            raise ValueError(f"Category {category} has too few records to split")
        train_count = total - validation_count - test_count

        for index, item in enumerate(items):
            if index < train_count:
                split_name = "train"
            elif index < train_count + validation_count:
                split_name = "validation"
            else:
                split_name = "test"
            copy = dict(item)
            copy["split"] = split_name
            splits[split_name].append(copy)

    for split_name in splits:
        rng.shuffle(splits[split_name])
    return splits


def user_text(record: dict) -> str:
    return record["messages"][1]["content"]


def answer_text(record: dict) -> str:
    content = record["messages"][2]["content"]
    return content.split("\n", 1)[1] if "\n" in content else content


def token_set(record: dict) -> set[str]:
    return set(normalize_for_dedupe(user_text(record)).split())


def leakage_checks(
    splits: dict[str, list[dict]], near_threshold: float = 0.92
) -> dict:
    names = ["train", "validation", "test"]
    ids = {name: {record["id"] for record in splits[name]} for name in names}
    exact = {
        name: {normalize_for_dedupe(user_text(record)) for record in splits[name]}
        for name in names
    }
    tokens = {name: [token_set(record) for record in splits[name]] for name in names}

    id_overlap = {
        f"{left}_{right}": sorted(ids[left] & ids[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    }
    exact_overlap = {
        f"{left}_{right}": len(exact[left] & exact[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    }

    near_pairs: dict[str, int] = {}
    max_similarity = 0.0
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            label = f"{left}_{right}"
            count = 0
            for left_tokens in tokens[left]:
                for right_tokens in tokens[right]:
                    similarity = jaccard(left_tokens, right_tokens)
                    if similarity > max_similarity:
                        max_similarity = similarity
                    if similarity >= near_threshold:
                        count += 1
            near_pairs[label] = count

    return {
        "near_duplicate_threshold": near_threshold,
        "id_overlap": id_overlap,
        "exact_user_text_overlap": exact_overlap,
        "near_duplicate_cross_split_pairs": near_pairs,
        "max_cross_split_jaccard": round(max_similarity, 4),
        "passed": (
            all(not value for value in id_overlap.values())
            and all(value == 0 for value in exact_overlap.values())
            and all(value == 0 for value in near_pairs.values())
        ),
    }


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "max": max(values),
    }


def length_statistics(splits: dict[str, list[dict]]) -> dict:
    result = {}
    for name, records in splits.items():
        result[name] = {
            "user_words": summary([len(user_text(record).split()) for record in records]),
            "answer_words": summary(
                [len(answer_text(record).split()) for record in records]
            ),
            "user_chars": summary([len(user_text(record)) for record in records]),
            "answer_chars": summary(
                [len(answer_text(record)) for record in records]
            ),
        }
    return result


def category_distribution(splits: dict[str, list[dict]]) -> dict:
    distribution = {}
    for category in ALLOWED_CATEGORIES:
        distribution[category] = {
            name: sum(1 for record in records if record["category"] == category)
            for name, records in splits.items()
        }
        distribution[category]["total"] = sum(distribution[category].values())
    return distribution


def write_report(
    path: Path,
    splits: dict[str, list[dict]],
    distribution: dict,
    lengths: dict,
    leakage: dict,
    cleaning_stats: dict,
    seed: int,
) -> None:
    total = sum(len(records) for records in splits.values())
    lines = [
        "# ShopEase Dataset Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Random seed: {seed}",
        f"Policy document: `data/raw/shopease_policy.md`",
        "",
        "All questions and answers are synthetic and were generated from the fictional ShopEase",
        "policy document. No real customer data is used.",
        "",
        "## Pipeline summary",
        "",
        "| Stage | Records |",
        "|---|---:|",
        f"| Raw generated | {cleaning_stats.get('input_records', 0)} |",
        f"| Invalid removed | {cleaning_stats.get('invalid_records', 0)} |",
        f"| Duplicate IDs removed | {cleaning_stats.get('duplicate_ids', 0)} |",
        f"| Exact duplicates removed | {cleaning_stats.get('exact_duplicates', 0)} |",
        f"| Near duplicates removed | {cleaning_stats.get('near_duplicates', 0)} |",
        f"| Clean records used | {cleaning_stats.get('kept_records', total)} |",
        f"| JSON parse errors | {cleaning_stats.get('parse_errors', 0)} |",
        "",
        f"Near-duplicate threshold: `{cleaning_stats.get('near_duplicate_threshold', 0.92)}` "
        f"(token Jaccard, compared globally before splitting).",
        "",
        "## Split sizes",
        "",
        "| Split | Records | Share | Categories covered |",
        "|---|---:|---:|---:|",
    ]
    for name in ["train", "validation", "test"]:
        count = len(splits[name])
        categories = len({record["category"] for record in splits[name]})
        lines.append(f"| {name} | {count} | {count / total:.1%} | {categories} |")
    lines.append(f"| **Total** | **{total}** | **100%** | **{len(ALLOWED_CATEGORIES)}** |")

    lines.extend(
        [
            "",
            "## Category distribution",
            "",
            "| Category | Train | Validation | Test | Total | Share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category in ALLOWED_CATEGORIES:
        row = distribution[category]
        share = row["total"] / total if total else 0
        lines.append(
            f"| {category} | {row['train']} | {row['validation']} | {row['test']} | "
            f"{row['total']} | {share:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Length statistics (whitespace words)",
            "",
            "| Split | User min | User median | User mean | User max | Answer min | "
            "Answer median | Answer mean | Answer max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["train", "validation", "test"]:
        user = lengths[name]["user_words"]
        answer = lengths[name]["answer_words"]
        lines.append(
            f"| {name} | {user['min']} | {user['median']} | {user['mean']} | "
            f"{user['max']} | {answer['min']} | {answer['median']} | "
            f"{answer['mean']} | {answer['max']} |"
        )

    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Record ID overlap between splits | "
            f"{'PASS' if all(not value for value in leakage['id_overlap'].values()) else 'FAIL'} |",
            f"| Exact user-text overlap between splits | "
            f"{'PASS' if all(value == 0 for value in leakage['exact_user_text_overlap'].values()) else 'FAIL'} |",
            f"| Cross-split near-duplicate pairs (Jaccard >= "
            f"{leakage['near_duplicate_threshold']}) | "
            f"{'PASS' if all(value == 0 for value in leakage['near_duplicate_cross_split_pairs'].values()) else 'FAIL'} |",
            f"| Maximum cross-split Jaccard similarity | "
            f"{leakage['max_cross_split_jaccard']} |",
            "",
            "The test split is frozen after baseline evaluation and is never used for training or",
            "hyperparameter tuning.",
            "",
            "## Format",
            "",
            "Each JSONL record contains: `id`, `category`, `intent`, `requires_clarification`,",
            "`messages` (system, user, assistant), `source`, and `split` in the split files.",
            "Assistant messages begin with `[category: <label>]` so the deployment layer can",
            "extract the predicted support category.",
            "",
            "## Reproducibility",
            "",
            "```powershell",
            "python -m src.data.generate_dataset --seed 42",
            "python -m src.data.clean_dataset --near-threshold 0.92",
            "python -m src.data.split_dataset --seed 42",
            "python -m src.data.validate_dataset",
            "```",
            "",
            "## Limitations",
            "",
            "- The dataset is synthetic and template-generated; it does not cover every real phrasing.",
            "- Length statistics are whitespace-based; tokenizer-specific counts are produced during",
            "  baseline evaluation.",
            "- Near-duplicate detection uses token Jaccard, which is a practical proxy, not a semantic",
            "  entailment model.",
            "- Automatic evaluation cannot fully judge tone or policy correctness; a human rubric is",
            "  used in the evaluation phase.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/shopease_clean.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--stats", type=Path, default=Path("reports/split_stats.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/dataset_report.md")
    )
    parser.add_argument(
        "--cleaning-stats", type=Path, default=Path("reports/cleaning_stats.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records, parse_errors = read_jsonl(args.input)
    if parse_errors:
        raise SystemExit(f"Input contains {len(parse_errors)} JSON parse errors")

    splits = stratified_split(records, seed=args.seed)
    for category in ALLOWED_CATEGORIES:
        for name in splits:
            if not any(record["category"] == category for record in splits[name]):
                raise SystemExit(f"Category {category} is missing from split {name}")

    leakage = leakage_checks(splits)
    if not leakage["passed"]:
        raise SystemExit(f"Leakage check failed: {json.dumps(leakage, indent=2)}")

    distribution = category_distribution(splits)
    lengths = length_statistics(splits)

    for name, split_records in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", split_records)

    cleaning_stats = {}
    if args.cleaning_stats.exists():
        cleaning_stats = json.loads(args.cleaning_stats.read_text(encoding="utf-8"))

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "input_file": str(args.input),
        "total_records": sum(len(records) for records in splits.values()),
        "splits": {
            name: {
                "count": len(records),
                "categories": dict(
                    sorted(Counter(record["category"] for record in records).items())
                ),
            }
            for name, records in splits.items()
        },
        "category_distribution": distribution,
        "length_statistics": lengths,
        "leakage_checks": leakage,
        "cleaning": cleaning_stats,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_report(
        args.report, splits, distribution, lengths, leakage, cleaning_stats, args.seed
    )

    print(f"Total records: {stats['total_records']}")
    for name in ["train", "validation", "test"]:
        print(f"  {name:10s} {len(splits[name]):4d} -> {args.output_dir / (name + '.jsonl')}")
    print(f"Leakage checks: {'PASS' if leakage['passed'] else 'FAIL'}")
    print(f"Stats: {args.stats}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.clean_dataset import read_jsonl
from src.data.policy_data import ALLOWED_CATEGORIES, load_system_prompt


CATEGORY_KEYWORDS = {
    "order_status": [
        "order status",
        "where is my order",
        "track",
        "not received",
        "not arrived",
        "delivered",
        "shipped",
        "package",
        "arrive",
    ],
    "shipping": [
        "shipping",
        "delivery",
        "deliver",
        "courier",
        "pincode",
        "pin code",
        "address",
        "express",
        "standard delivery",
    ],
    "returns": [
        "return",
        "send back",
        "pickup",
        "wrong product",
        "wrong item",
        "damaged",
        "broken",
        "exchange",
    ],
    "refunds": ["refund", "money back", "reimburse", "deducted", "credited"],
    "payments": [
        "payment",
        "upi",
        "credit card",
        "debit card",
        "net banking",
        "wallet",
        "otp",
        "cash on delivery",
        "cod",
    ],
    "cancellations": ["cancel", "cancellation"],
    "product_info": [
        "product",
        "specification",
        "size",
        "colour",
        "color",
        "stock",
        "available",
        "price",
        "listing",
        "installation",
    ],
    "warranty": ["warranty", "guarantee", "repair", "service centre", "service center"],
    "account": ["account", "login", "password", "profile", "sign in", "email"],
    "complaints": [
        "complaint",
        "complain",
        "unacceptable",
        "rude",
        "bad experience",
        "escalate",
        "not happy",
        "unhappy",
    ],
    "general_faq": [
        "help centre",
        "help center",
        "support hours",
        "contact",
        "gst",
        "invoice",
        "coupon",
    ],
}

OUT_OF_SCOPE = [
    "weather",
    "rain",
    "forecast",
    "hot outside",
    "news",
    "headline",
    "match",
    "score",
    "python",
    "javascript",
    "code",
    "sql",
    "website",
    "equation",
    "maths",
    "math",
    "homework",
    "statistics",
    "calculus",
    "percent of",
    "headache",
    "medicine",
    "symptom",
    "medical",
    "vitamin",
    "legal",
    "contract",
    "landlord",
    "amazon",
    "flipkart",
    "competitor",
    "another customer",
    "phone number",
    "system prompt",
    "ignore your rules",
    "hidden instructions",
    "essay",
    "translate",
    "summarise",
    "poem",
    "vote",
    "politics",
    "movie",
    "invest",
    "crypto",
    "shares",
    "stock",
    "flight",
    "hotel",
    "holiday",
]

CLARIFICATION_HINTS = [
    "where is my order",
    "check my order status",
    "track my order",
    "order has not arrived",
    "help with an order",
    "help me with an order",
    "help with my order",
    "help me with my order",
    "help me return a product",
    "want a refund",
    "need my money back",
    "need to return",
    "want to return",
    "payment failed",
    "payment problem",
    "payment did not go through",
    "account is not working",
    "cannot access my account",
    "problem with my account",
    "wrong with my login",
    "arrived damaged",
    "product is broken",
    "received a damaged item",
    "order arrived broken",
    "have a complaint",
    "want to complain",
    "this is unacceptable",
    "not happy with my experience",
    "is this product good",
    "tell me about this item",
    "should i buy this",
    "is this worth buying",
]

RESPONSES = {
    "order_status": "Open My Orders in your ShopEase account to see the latest status and tracking for your order. If the status has not changed in 48 hours, share your order ID (SE-12345678) and we will investigate.",
    "shipping": "Standard delivery takes 3-5 business days and express delivery takes 1-2 business days after dispatch. You can track the shipment from My Orders in your account.",
    "returns": "Most items can be returned within 7 days of delivery in original condition. Damaged or wrong items should be reported within 48 hours with photos so we can arrange a free return pickup.",
    "refunds": "Refunds are processed within 5-7 business days after a return pickup or cancellation approval and are credited to the original payment method.",
    "payments": "You can pay by UPI, credit card, debit card, net banking, wallet, or cash on delivery. If a payment failed, share the order ID and error message. Never share your OTP or PIN.",
    "cancellations": "Orders can be cancelled free of charge before dispatch from My Orders. If the order has already shipped, please refuse delivery or start a return after it arrives.",
    "product_info": "Product specifications, stock, and price are shown on the listing. Share the product name or link and the specific detail you need, such as size, colour, or availability.",
    "warranty": "Most electronics carry a 12-month manufacturer warranty. Warranty claims need the order ID and invoice. Accidental damage and normal wear are not covered.",
    "account": "You can reset your password from the login page. Never share your password or OTP. If the reset email does not arrive, contact support with the email or phone on the account.",
    "complaints": "I am sorry about the experience. Please share your order ID and what happened so I can help resolve it or escalate it to a senior team member.",
    "general_faq": "ShopEase support is available 9 AM to 9 PM IST, seven days a week. You can also check the Help Centre for policies, invoices, coupons, and contact options.",
    "clarification": "I can help. Could you share your order ID and a short description of the issue so I can guide you to the right next step?",
    "out_of_scope": "I can only help with ShopEase customer support, so I cannot help with that. I can assist with orders, delivery, returns, refunds, payments, products, warranty, or account questions.",
}

COVERAGE_KEYWORDS = {
    "order_status": ["my orders", "track", "order id", "status"],
    "shipping": ["delivery", "dispatch", "track", "business days"],
    "returns": ["return", "pickup", "7 days", "48 hours"],
    "refunds": ["refund", "5-7 business days", "original payment"],
    "payments": ["upi", "card", "net banking", "wallet", "otp"],
    "cancellations": ["cancel", "dispatch", "refuse delivery"],
    "product_info": ["listing", "specification", "stock", "price"],
    "warranty": ["warranty", "invoice", "12-month"],
    "account": ["password", "login", "otp"],
    "complaints": ["order id", "escalate", "sorry"],
    "general_faq": ["support", "help centre", "9 am"],
    "clarification": ["order id", "?"],
    "out_of_scope": ["only help", "cannot"],
}


ORDER_ID_PATTERN = re.compile(r"\bse-\d{8}\b", re.IGNORECASE)


def classify(text: str) -> str:
    normalized = " ".join(text.lower().split())
    if any(keyword in normalized for keyword in OUT_OF_SCOPE):
        return "out_of_scope"
    if any(hint in normalized for hint in CLARIFICATION_HINTS) and not ORDER_ID_PATTERN.search(text):
        return "clarification"
    scores = {
        category: sum(1 for keyword in keywords if keyword in normalized)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "general_faq"
    return best_category


def make_response(category: str) -> str:
    return f"[category: {category}]\n{RESPONSES[category]}"


def extract_user(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def run_rule_baseline(records: list[dict]) -> list[dict]:
    results = []
    for record in records:
        user_text = extract_user(record["messages"])
        start = time.perf_counter()
        predicted = classify(user_text)
        response = make_response(predicted)
        latency_ms = (time.perf_counter() - start) * 1000

        expected = record["category"]
        valid = response.startswith(f"[category: {predicted}]") and len(response) > 40
        covered = any(
            keyword in response.lower()
            for keyword in COVERAGE_KEYWORDS.get(expected, [])
        )
        results.append(
            {
                "id": record["id"],
                "expected_category": expected,
                "predicted_category": predicted,
                "user": user_text,
                "response": response,
                "latency_ms": round(latency_ms, 3),
                "valid": valid,
                "policy_keyword_covered": covered,
                "refusal": any(
                    phrase in response.lower()
                    for phrase in ["only help", "cannot", "outside"]
                ),
                "asks_clarification": "?" in response,
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(
        1 for row in results if row["expected_category"] == row["predicted_category"]
    )
    out_of_scope = [row for row in results if row["expected_category"] == "out_of_scope"]
    clarification = [row for row in results if row["expected_category"] == "clarification"]
    latencies = [row["latency_ms"] for row in results]

    return {
        "examples": total,
        "category_accuracy": round(correct / total, 4) if total else 0.0,
        "response_validity_rate": round(
            sum(row["valid"] for row in results) / total, 4
        )
        if total
        else 0.0,
        "unsupported_refusal_rate": round(
            sum(row["refusal"] for row in out_of_scope) / len(out_of_scope), 4
        )
        if out_of_scope
        else 0.0,
        "policy_keyword_coverage": round(
            sum(row["policy_keyword_covered"] for row in results) / total, 4
        )
        if total
        else 0.0,
        "clarification_rate": round(
            sum(row["asks_clarification"] for row in clarification)
            / len(clarification),
            4,
        )
        if clarification
        else 0.0,
        "latency_p50_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_p95_ms": round(percentile(latencies, 0.95), 3),
        "peak_vram_mb": 0,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: dict, results: list[dict], run_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Report",
        "",
        f"Generated: {run_config['generated_at']}",
        f"Backend: `{run_config['backend']}`",
        f"Model: `{run_config['model_id']}` (not loaded in this smoke run)",
        f"Test examples: {summary['examples']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Category accuracy | {summary['category_accuracy']:.4f} |",
        f"| Response validity rate | {summary['response_validity_rate']:.4f} |",
        f"| Unsupported refusal rate | {summary['unsupported_refusal_rate']:.4f} |",
        f"| Policy keyword coverage | {summary['policy_keyword_coverage']:.4f} |",
        f"| Clarification rate | {summary['clarification_rate']:.4f} |",
        f"| Latency p50 (ms) | {summary['latency_p50_ms']:.3f} |",
        f"| Latency p95 (ms) | {summary['latency_p95_ms']:.3f} |",
        f"| Peak VRAM (MB) | {summary['peak_vram_mb']} |",
        "",
        "## Sample outputs",
        "",
    ]
    for row in results[:5]:
        lines.extend(
            [
                f"**{row['id']}** — expected `{row['expected_category']}`, "
                f"predicted `{row['predicted_category']}`",
                "",
                f"> {row['user']}",
                "",
                f"{row['response']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "This is a deterministic rule-based smoke baseline used to validate the evaluation",
            "pipeline and artifact format. It is not the pretrained Qwen model baseline. The",
            "transformer baseline will replace these numbers after the CUDA environment and model",
            "download are available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation_config.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    test_file = Path(config["data"]["test_file"])
    records, errors = read_jsonl(test_file)
    if errors:
        raise SystemExit(f"Could not parse {test_file}: {errors[:3]}")
    if args.limit:
        records = records[: args.limit]

    results = run_rule_baseline(records)
    summary = summarize(results)

    baseline_dir = Path(config["artifacts"]["baseline_dir"])
    baseline_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "rule_based_smoke",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_loaded": False,
        "seed": config.get("seed"),
        "test_file": str(test_file),
        "examples": len(results),
        "note": "Smoke baseline for pipeline validation; not the pretrained model baseline.",
    }

    write_jsonl(
        baseline_dir / "prompts.jsonl",
        [
            {
                "id": row["id"],
                "expected_category": row["expected_category"],
                "messages": [
                    {"role": "system", "content": load_system_prompt()},
                    {"role": "user", "content": row["user"]},
                ],
            }
            for row in results
        ],
    )
    write_jsonl(baseline_dir / "responses.jsonl", results)
    (baseline_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (baseline_dir / "latency_memory.json").write_text(
        json.dumps(
            {
                "latency_p50_ms": summary["latency_p50_ms"],
                "latency_p95_ms": summary["latency_p95_ms"],
                "peak_vram_mb": summary["peak_vram_mb"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (baseline_dir / "observed_limitations.md").write_text(
        "# Observed Limitations\n\n"
        "- This run uses deterministic keyword rules, not the Qwen2.5-1.5B-Instruct model.\n"
        "- It validates the evaluation pipeline, metrics, and artifact format only.\n"
        "- Category accuracy reflects keyword matching, not language-model capability.\n",
        encoding="utf-8",
    )
    write_report(
        Path(config["artifacts"]["reports_dir"]) / "baseline_report.md",
        summary,
        results,
        run_config,
    )

    print(f"Backend: {run_config['backend']}")
    print(f"Examples: {summary['examples']}")
    for key, value in summary.items():
        if key != "examples":
            print(f"  {key}: {value}")
    print("\nSample outputs:")
    for row in results[:3]:
        print(f"  {row['id']} expected={row['expected_category']} predicted={row['predicted_category']}")
        print("    " + row["response"].replace("\n", "\n    "))


if __name__ == "__main__":
    main()

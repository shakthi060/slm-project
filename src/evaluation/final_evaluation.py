"""Phase 3 final evaluation on the frozen 75-example ShopEase test set.

Conditions:
  A: pretrained SmolLM2-135M-Instruct, full policy prompt, 256 tokens
  B: pretrained + Phase 1 LoRA adapter, full policy prompt, 256 tokens
  C: pretrained, compact prompt, 128 tokens
  D: pretrained, full policy prompt, 160 tokens

The script never modifies data/test.jsonl. It runs each condition in its own
process so peak RSS is not contaminated across conditions.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from src.data.clean_dataset import read_jsonl
from src.data.policy_data import POLICY_DOCUMENT, load_system_prompt
from src.evaluation.baseline import classify, extract_user, percentile, write_jsonl
from src.evaluation.final_metrics import (
    category_signals,
    content_tokens,
    extract_assistant,
    hallucination_flags,
    is_refusal,
    jaccard,
    policy_keyword_covered,
    response_validity,
)
from src.optimization.benchmark_inference import (
    environment_info,
    process_peak_wset_mb,
    rss_mb,
    utc_now,
)

CONDITION_ORDER = ("A", "B", "C", "D")


def safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def build_messages(user: str, config: dict, prompt_variant: str) -> tuple[list[dict], str]:
    if prompt_variant == "compact":
        system_prompt = str(config["prompts"]["compact_system_prompt"]).strip()
        variant = "compact_system_prompt"
    else:
        system_prompt = load_system_prompt()
        variant = "full_system_prompt"
    return (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        variant,
    )


def load_model(config: dict, spec: dict) -> tuple[object, object, dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(config["paths"]["model"])
    adapter_dir = Path(config["paths"]["adapter"])
    dtype = torch.float32

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_s = time.perf_counter() - tokenizer_start

    rss_before = rss_mb()
    base_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    base_load_s = time.perf_counter() - base_start

    adapter_apply_s = 0.0
    adapter_loaded = False
    if spec["adapter"]:
        from peft import PeftModel

        adapter_start = time.perf_counter()
        model = PeftModel.from_pretrained(model, adapter_dir)
        adapter_apply_s = time.perf_counter() - adapter_start
        adapter_loaded = True

    model.eval()
    total_load_s = tokenizer_load_s + base_load_s + adapter_apply_s
    load_info = {
        "tokenizer_load_s": round(tokenizer_load_s, 3),
        "base_load_s": round(base_load_s, 3),
        "adapter_apply_s": round(adapter_apply_s, 3),
        "model_load_s": round(total_load_s, 3),
        "rss_before_load_mb": rss_before,
        "rss_after_load_mb": rss_mb(),
        "adapter_loaded": adapter_loaded,
        "adapter_dir": str(adapter_dir) if adapter_loaded else None,
        "quantization": "none",
        "dtype": "float32",
        "device": config["runtime"]["device"],
    }
    return model, tokenizer, load_info


def generate_once(model, tokenizer, messages: list[dict], generation: dict) -> dict:
    tokenize_start = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    tokenize_ms = (time.perf_counter() - tokenize_start) * 1000
    prompt_tokens = int(inputs["input_ids"].shape[1])

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation["do_sample"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            repetition_penalty=float(generation["repetition_penalty"]),
            num_beams=int(generation.get("num_beams", 1)),
            use_cache=bool(generation.get("use_cache", True)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    new_tokens = output[0][prompt_tokens:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    eos_id = tokenizer.eos_token_id
    last_token = int(new_tokens[-1]) if len(new_tokens) else None
    hit_cap = len(new_tokens) >= int(generation["max_new_tokens"])
    truncated = bool(hit_cap and last_token != eos_id)

    return {
        "response": response,
        "prompt_tokens": prompt_tokens,
        "output_tokens": int(len(new_tokens)),
        "response_chars": len(response),
        "tokenize_ms": round(tokenize_ms, 3),
        "latency_ms": round(latency_ms, 3),
        "total_request_ms": round(tokenize_ms + latency_ms, 3),
        "hit_max_new_tokens": hit_cap,
        "truncated": truncated,
        "last_token": last_token,
    }


def summarize_rows(rows: list[dict], config: dict, spec: dict, load_info: dict,
                   generation: dict, warmup_s: float, inference_s: float,
                   system_prompt_tokens: int) -> dict:
    examples = len(rows)
    latencies = [row["latency_ms"] for row in rows]
    total_requests = [row["total_request_ms"] for row in rows]
    output_tokens = [row["output_tokens"] for row in rows]
    prompt_tokens = [row["prompt_tokens"] for row in rows]
    response_chars = [row["response_chars"] for row in rows]
    out_of_scope = [row for row in rows if row["expected_category"] == "out_of_scope"]
    clarification = [row for row in rows if row["expected_category"] == "clarification"]

    invalid_reasons = Counter()
    for row in rows:
        invalid_reasons.update(row["invalid_reasons"])

    flag_counts = Counter()
    for row in rows:
        flag_counts.update(row["heuristic_hallucination_flags"])

    return {
        "condition": spec["key"],
        "condition_label": spec["label"],
        "description": spec["description"],
        "generated_at": utc_now(),
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_path": str(Path(config["paths"]["model"])),
        "prompt_ids": [row["id"] for row in rows],
        "system_prompt_variant": rows[0]["system_prompt_variant"] if rows else None,
        "system_prompt_tokens": system_prompt_tokens,
        "generation": generation,
        "load": load_info,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "warmup_s": round(warmup_s, 3),
        "inference_s": round(inference_s, 3),
        "examples": examples,
        "expected_category_counts": dict(
            sorted(Counter(row["expected_category"] for row in rows).items())
        ),
        "latency_mean_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_min_ms": round(min(latencies), 3) if latencies else 0.0,
        "latency_max_ms": round(max(latencies), 3) if latencies else 0.0,
        "latency_p50_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_p95_ms": round(percentile(latencies, 0.95), 3),
        "total_request_p50_ms": round(statistics.median(total_requests), 3)
        if total_requests
        else 0.0,
        "total_request_p95_ms": round(percentile(total_requests, 0.95), 3),
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2)
        if prompt_tokens
        else 0.0,
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else 0,
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else 0,
        "output_tokens_mean": round(statistics.mean(output_tokens), 2)
        if output_tokens
        else 0.0,
        "output_tokens_min": min(output_tokens) if output_tokens else 0,
        "output_tokens_max": max(output_tokens) if output_tokens else 0,
        "response_chars_mean": round(statistics.mean(response_chars), 1)
        if response_chars
        else 0.0,
        "truncation_rate": rate([row["truncated"] for row in rows]),
        "hit_max_new_tokens_rate": rate([row["hit_max_new_tokens"] for row in rows]),
        "response_validity_rate": rate([row["valid"] for row in rows]),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "tag_rate": rate([row["tag_category"] != "unknown" for row in rows]),
        "tag_accuracy": rate([row["tag_correct"] for row in rows]),
        "fallback_accuracy": rate([row["fallback_correct"] for row in rows]),
        "combined_accuracy": rate([row["combined_correct"] for row in rows]),
        "query_rule_accuracy": rate([row["query_rule_correct"] for row in rows]),
        "policy_keyword_coverage": rate(
            [row["policy_keyword_covered"] for row in rows]
        ),
        "heuristic_hallucination_rate": rate(
            [bool(row["heuristic_hallucination_flags"]) for row in rows]
        ),
        "heuristic_flag_counts": dict(sorted(flag_counts.items())),
        "out_of_scope_refusal_rate": rate([row["refusal"] for row in out_of_scope]),
        "clarification_question_rate": rate(
            [row["asks_clarification"] for row in clarification]
        ),
        "refusal_rate": rate([row["refusal"] for row in rows]),
        "asks_clarification_rate": rate([row["asks_clarification"] for row in rows]),
        "reference_overlap_mean": safe_mean(
            [row["reference_overlap"] for row in rows]
        ),
        "user_overlap_mean": safe_mean([row["user_overlap"] for row in rows]),
        "peak_rss_mb": max([row["rss_mb"] for row in rows] + [rss_mb()]),
        "process_peak_wset_mb": process_peak_wset_mb(),
        "quality_note": (
            "Automatic proxies only. Keyword coverage and heuristic hallucination "
            "flags are not a substitute for human review."
        ),
        "category_note": (
            "tag_accuracy uses explicit [category: ...] tags only; fallback_accuracy "
            "uses the documented rule-based classifier on the response; "
            "combined_accuracy uses the tag when present and the fallback otherwise."
        ),
    }


def run_condition(
    config: dict,
    records: list[dict],
    condition: str,
    output_dir: Path,
    predictions_dir: Path,
) -> None:
    spec = {"key": condition, **config["configs"][condition]}
    generation = config["generation"][spec["generation"]]
    threads = int(config["runtime"]["threads"])
    interop = int(config["runtime"]["interop_threads"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass

    model, tokenizer, load_info = load_model(config, spec)
    print(
        json.dumps({"condition": condition, "load": load_info}, indent=2),
        flush=True,
    )

    system_prompt_tokens = len(
        tokenizer(
            build_messages("", config, spec["prompt"])[0][0]["content"],
            add_special_tokens=False,
        )["input_ids"]
    )

    peak_rss = rss_mb()
    warmup_start = time.perf_counter()
    for warmup_prompt in config["runtime"]["warmup_prompts"]:
        messages, _ = build_messages(warmup_prompt, config, spec["prompt"])
        warmup_generation = dict(generation)
        warmup_generation["max_new_tokens"] = int(
            config["runtime"]["warmup_max_new_tokens"]
        )
        generate_once(model, tokenizer, messages, warmup_generation)
        peak_rss = max(peak_rss, rss_mb())
    warmup_s = time.perf_counter() - warmup_start

    rows = []
    inference_start = time.perf_counter()
    for record in records:
        user = extract_user(record["messages"])
        reference = extract_assistant(record["messages"])
        messages, variant = build_messages(user, config, spec["prompt"])
        result = generate_once(model, tokenizer, messages, generation)
        peak_rss = max(peak_rss, rss_mb())

        response = result["response"]
        expected = record["category"]
        valid, invalid_reasons = response_validity(response, result["truncated"])
        flags = hallucination_flags(
            response, user, reference, expected, POLICY_DOCUMENT
        )
        signals = category_signals(response, expected)
        query_rule_category = classify(user)

        row = {
            "condition": condition,
            "condition_label": spec["label"],
            "id": record["id"],
            "expected_category": expected,
            "intent": record.get("intent"),
            "requires_clarification": record.get("requires_clarification"),
            "user": user,
            "reference": reference,
            "system_prompt_variant": variant,
            "response": response,
            "prompt_tokens": result["prompt_tokens"],
            "output_tokens": result["output_tokens"],
            "response_chars": result["response_chars"],
            "tokenize_ms": result["tokenize_ms"],
            "latency_ms": result["latency_ms"],
            "total_request_ms": result["total_request_ms"],
            "hit_max_new_tokens": result["hit_max_new_tokens"],
            "truncated": result["truncated"],
            "valid": valid,
            "invalid_reasons": invalid_reasons,
            "policy_keyword_covered": policy_keyword_covered(response, expected),
            "refusal": is_refusal(response),
            "asks_clarification": "?" in response,
            "query_rule_category": query_rule_category,
            "query_rule_correct": query_rule_category == expected,
            "reference_overlap": jaccard(response, reference),
            "user_overlap": jaccard(response, user),
            "heuristic_hallucination_flags": flags,
            "rss_mb": rss_mb(),
            **signals,
        }
        rows.append(row)
        print(
            f"{condition} {record['id']} {result['latency_ms']:.1f}ms "
            f"in={result['prompt_tokens']} out={result['output_tokens']} "
            f"valid={valid} flags={flags}",
            flush=True,
        )

    inference_s = time.perf_counter() - inference_start
    peak_rss = max(peak_rss, rss_mb())
    summary = summarize_rows(
        rows,
        config,
        spec,
        load_info,
        generation,
        warmup_s,
        inference_s,
        system_prompt_tokens,
    )
    summary["peak_rss_mb"] = peak_rss

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_dir / f"condition_{spec['label']}.jsonl", rows)
    (output_dir / f"condition_{spec['label']}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    del model
    gc.collect()


def delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return round(new - old, 3)


def pct_change(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def compare(summaries: dict, new_key: str, old_key: str) -> dict:
    new = summaries[new_key]
    old = summaries[old_key]
    return {
        "latency_p50_delta_ms": delta(new["latency_p50_ms"], old["latency_p50_ms"]),
        "latency_p50_change_pct": pct_change(
            new["latency_p50_ms"], old["latency_p50_ms"]
        ),
        "latency_p95_delta_ms": delta(new["latency_p95_ms"], old["latency_p95_ms"]),
        "latency_p95_change_pct": pct_change(
            new["latency_p95_ms"], old["latency_p95_ms"]
        ),
        "total_request_p50_delta_ms": delta(
            new["total_request_p50_ms"], old["total_request_p50_ms"]
        ),
        "total_request_p50_change_pct": pct_change(
            new["total_request_p50_ms"], old["total_request_p50_ms"]
        ),
        "model_load_delta_s": delta(
            new["load"]["model_load_s"], old["load"]["model_load_s"]
        ),
        "peak_rss_delta_mb": delta(new["peak_rss_mb"], old["peak_rss_mb"]),
        "output_tokens_mean_delta": delta(
            new["output_tokens_mean"], old["output_tokens_mean"]
        ),
        "prompt_tokens_mean_delta": delta(
            new["prompt_tokens_mean"], old["prompt_tokens_mean"]
        ),
        "response_validity_delta": delta(
            new["response_validity_rate"], old["response_validity_rate"]
        ),
        "policy_keyword_coverage_delta": delta(
            new["policy_keyword_coverage"], old["policy_keyword_coverage"]
        ),
        "heuristic_hallucination_delta": delta(
            new["heuristic_hallucination_rate"], old["heuristic_hallucination_rate"]
        ),
        "tag_accuracy_delta": delta(new["tag_accuracy"], old["tag_accuracy"]),
        "fallback_accuracy_delta": delta(
            new["fallback_accuracy"], old["fallback_accuracy"]
        ),
        "combined_accuracy_delta": delta(
            new["combined_accuracy"], old["combined_accuracy"]
        ),
        "reference_overlap_delta": delta(
            new["reference_overlap_mean"], old["reference_overlap_mean"]
        ),
    }


def load_phase2_dynamic_int8() -> dict:
    path = Path("experiments/optimization/benchmark_summary.json")
    result = {
        "phase2_condition": "D_dynamic_int8_investigation",
        "status": "rejected",
        "reasons": [
            "p95 latency was much worse than the other conditions.",
            "Peak RSS was higher than the unquantized pretrained conditions.",
            "Outputs included garbled text, so the latency measurement was not a usable deployment result.",
            "The approach used deprecated torch.ao.quantization dynamic INT8.",
        ],
    }
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            condition = payload.get("conditions", {}).get("D", {})
            if condition:
                result["measured"] = {
                    key: condition.get(key)
                    for key in (
                        "latency_p50_ms",
                        "latency_p95_ms",
                        "total_request_p50_ms",
                        "peak_rss_mb",
                        "output_tokens_mean",
                        "response_validity_rate",
                        "policy_keyword_coverage",
                    )
                }
                result["source"] = str(path)
        except (OSError, json.JSONDecodeError):
            pass
    return result


def manual_review_summary(scores: dict) -> dict:
    summary = {}
    for condition, entries in scores.items():
        if not isinstance(entries, dict):
            continue
        rows = [entry for entry in entries.values() if isinstance(entry, dict)]
        if not rows:
            continue
        summary[condition] = {
            "responses": len(rows),
            "relevance_rate": safe_mean(
                [float(row.get("relevance", 0)) for row in rows]
            ),
            "correctness_rate": safe_mean(
                [float(row.get("correctness", 0)) for row in rows]
            ),
            "policy_consistency_rate": safe_mean(
                [float(row.get("policy_consistency", 0)) for row in rows]
            ),
            "hallucination_rate": safe_mean(
                [float(row.get("hallucination", 0)) for row in rows]
            ),
            "mean_rubric_score": safe_mean(
                [
                    (
                        float(row.get("relevance", 0))
                        + float(row.get("correctness", 0))
                        + float(row.get("policy_consistency", 0))
                    )
                    / 3
                    for row in rows
                ]
            ),
        }
    return summary


def manual_review_by_category(scores: dict, records_by_id: dict) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for condition, entries in scores.items():
        if not isinstance(entries, dict):
            continue
        for sample_id, entry in entries.items():
            record = records_by_id.get(sample_id)
            if not record or not isinstance(entry, dict):
                continue
            grouped.setdefault(condition, {}).setdefault(record["category"], []).append(entry)

    summary = {}
    for condition, categories in grouped.items():
        summary[condition] = {}
        for category, rows in sorted(categories.items()):
            summary[condition][category] = {
                "responses": len(rows),
                "correctness_rate": safe_mean(
                    [float(row.get("correctness", 0)) for row in rows]
                ),
                "policy_consistency_rate": safe_mean(
                    [float(row.get("policy_consistency", 0)) for row in rows]
                ),
                "hallucination_rate": safe_mean(
                    [float(row.get("hallucination", 0)) for row in rows]
                ),
            }
    return summary


def build_failure_analysis(summaries: dict, manual_summary: dict | None) -> dict:
    manual_summary = manual_summary or {}

    def by_config(key: str) -> dict:
        return {
            condition: summaries[condition].get(key)
            for condition in CONDITION_ORDER
            if condition in summaries
        }

    def manual(key: str) -> dict:
        return {
            condition: manual_summary[condition].get(key)
            for condition in CONDITION_ORDER
            if condition in manual_summary
        }

    refusal = by_config("out_of_scope_refusal_rate")
    clarification = by_config("clarification_question_rate")
    truncation = by_config("truncation_rate")
    validity = by_config("response_validity_rate")
    p50 = by_config("latency_p50_ms")
    p95 = by_config("latency_p95_ms")
    correctness = manual("correctness_rate")
    hallucination = manual("hallucination_rate")

    return {
        "out_of_scope_refusal_rate_by_config": refusal,
        "clarification_question_rate_by_config": clarification,
        "truncation_rate_by_config": truncation,
        "manual_correctness_rate_by_config": correctness,
        "manual_hallucination_rate_by_config": hallucination,
        "observed_failure_patterns": [
            "All four configurations answered all 7 out-of-scope requests instead of refusing; "
            f"out-of-scope refusal rates were {refusal}.",
            "No configuration emitted a category tag; tag accuracy was 0.0 for all four, "
            "so category results rely entirely on the fallback classifier.",
            "No configuration asked a clarifying question on any of the 8 clarification examples; "
            f"clarification-question rates were {clarification}.",
            "All configurations produced unsupported or fabricated claims; manual hallucination rates were "
            f"A {hallucination.get('A')}, B {hallucination.get('B')}, "
            f"C {hallucination.get('C')}, D {hallucination.get('D')}.",
            "Truncation at the token cap was "
            f"C {truncation.get('C')}, D {truncation.get('D')}, "
            f"A {truncation.get('A')}, B {truncation.get('B')}.",
            "B had the best validity "
            f"({validity.get('B')}) and lowest truncation ({truncation.get('B')}), "
            f"but the worst p95 latency ({p95.get('B')} ms); C was fastest "
            f"(p50 {p50.get('C')} ms, p95 {p95.get('C')} ms) but truncated "
            f"{truncation.get('C')} of responses.",
            "Manual examples confirm: A/D claimed a US shipment and a 31 December 2023 "
            "delivery for order SE-76185042; C said coupons can be merged; A/B/C/D denied "
            "express delivery; all four answered a legal contract question as if in scope.",
            "Manual correctness was low for every configuration: "
            f"A {correctness.get('A')}, B {correctness.get('B')}, "
            f"C {correctness.get('C')}, D {correctness.get('D')}.",
        ],
        "root_causes": [
            "The 135M-parameter base model is too small for reliable policy-grounded customer support.",
            "The Phase 1 LoRA adapter was trained on only 26 examples for one epoch, so it cannot reliably change policy behavior.",
            "The full policy prompt is long, but the model still ignores key constraints, especially refusals and clarification.",
            "The model was not trained to emit category tags, and no constrained decoding enforces the allowed categories.",
            "Deterministic decoding with a 256-token cap produces long, repetitive, and sometimes truncated answers.",
            "There is no retrieval, rule-based policy guard, or post-generation validator.",
        ],
        "required_mitigations": [
            "Train or fine-tune a larger instruction model on substantially more policy-grounded support examples.",
            "Add explicit refusal, clarification, and category-tag training examples and evaluate them separately.",
            "Add retrieval or structured policy lookup so answers are grounded in the relevant policy section.",
            "Add post-generation guardrails for banned policy claims, fabricated contact details, OTP requests, and out-of-scope answers.",
            "Choose the token cap from measured truncation and quality, not only latency.",
            "Run a larger blinded human evaluation with multiple reviewers and production-like traffic before any deployment.",
        ],
    }


def write_manual_sheet(
    config: dict,
    records_by_id: dict,
    output_dir: Path,
    predictions_dir: Path,
    scores: dict | None,
) -> list[str]:
    sample_ids = list(config["manual_review"]["sample_ids"])
    rows = []
    missing = []
    for condition in CONDITION_ORDER:
        spec = config["configs"][condition]
        path = predictions_dir / f"condition_{spec['label']}.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        predictions, errors = read_jsonl(path)
        if errors:
            raise SystemExit(f"Could not parse {path}: {errors[:3]}")
        by_id = {row["id"]: row for row in predictions}
        for sample_id in sample_ids:
            row = by_id.get(sample_id)
            if row is None:
                missing.append(f"{path} missing {sample_id}")
                continue
            record = records_by_id[sample_id]
            rows.append(
                {
                    "condition": condition,
                    "condition_label": spec["label"],
                    "id": sample_id,
                    "expected_category": row["expected_category"],
                    "user": row["user"],
                    "reference": extract_assistant(record["messages"]),
                    "response": row["response"],
                    "truncated": row["truncated"],
                    "valid": row["valid"],
                    "tag_category": row["tag_category"],
                    "fallback_category": row["fallback_category"],
                    "heuristic_hallucination_flags": row[
                        "heuristic_hallucination_flags"
                    ],
                    "reference_overlap": row["reference_overlap"],
                    "manual": scores.get(condition, {}).get(sample_id)
                    if scores
                    else None,
                }
            )
    write_jsonl(output_dir / "manual_review_sheet.jsonl", rows)
    return missing


def write_report(final: dict, path: Path) -> None:
    summaries = final["configs"]

    def fmt(value, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "# Phase 3 Final Evaluation",
        "",
        f"Generated: {final['generated_at']}",
        f"Test set: `{final['test_file']}` ({final['test_examples']} held-out examples)",
        "",
        "## Automatic test-set metrics",
        "",
        "| Config | Model | Prompt | Max new | Valid | Tag acc. | Fallback acc. | Combined acc. | Keyword coverage | Heuristic halluc. | p50 ms | p95 ms | Peak RSS MB | Trunc. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        row = summaries[condition]
        model_name = "adapter" if condition == "B" else "pretrained"
        lines.append(
            "| {cond} | {model} | {prompt} | {max_new} | {valid} | {tag} | {fallback} | {combined} | {kw} | {halluc} | {p50} | {p95} | {rss} | {trunc} |".format(
                cond=row["condition_label"],
                model=model_name,
                prompt=row["system_prompt_variant"],
                max_new=row["generation"]["max_new_tokens"],
                valid=row["response_validity_rate"],
                tag=row["tag_accuracy"],
                fallback=row["fallback_accuracy"],
                combined=row["combined_accuracy"],
                kw=row["policy_keyword_coverage"],
                halluc=row["heuristic_hallucination_rate"],
                p50=row["latency_p50_ms"],
                p95=row["latency_p95_ms"],
                rss=row["peak_rss_mb"],
                trunc=row["truncation_rate"],
            )
        )

    lines += [
        "",
        "## Manual review",
        "",
    ]
    manual = final.get("manual_review", {})
    if manual.get("summary"):
        lines += [
            "| Config | Responses | Relevance | Correctness | Policy consistency | Hallucination | Mean rubric score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for condition in CONDITION_ORDER:
            row = manual["summary"].get(condition)
            if not row:
                continue
            lines.append(
                "| {cond} | {responses} | {rel} | {corr} | {policy} | {halluc} | {mean} |".format(
                    cond=condition,
                    responses=row["responses"],
                    rel=row["relevance_rate"],
                    corr=row["correctness_rate"],
                    policy=row["policy_consistency_rate"],
                    halluc=row["hallucination_rate"],
                    mean=row["mean_rubric_score"],
                )
            )
        lines += [
            "",
            "The manual sample was not blinded across configurations and was scored by one reviewer.",
        ]
    else:
        lines.append(
            "Manual-review scores are pending. The 15-example sheet is at "
            "`experiments/evaluation/manual_review_sheet.jsonl`."
        )

    failure = final.get("failure_analysis") or {}
    if failure:
        lines += [
            "",
            "## Failure analysis",
            "",
            "### Out-of-scope, clarification, and truncation",
            "",
            "| Config | Out-of-scope refusal | Clarification question | Truncation |",
            "|---|---:|---:|---:|",
        ]
        for condition in CONDITION_ORDER:
            lines.append(
                "| {cond} | {refusal} | {clarification} | {truncation} |".format(
                    cond=condition,
                    refusal=fmt(
                        failure["out_of_scope_refusal_rate_by_config"].get(condition)
                    ),
                    clarification=fmt(
                        failure["clarification_question_rate_by_config"].get(condition)
                    ),
                    truncation=fmt(
                        failure["truncation_rate_by_config"].get(condition)
                    ),
                )
            )

        lines += ["", "### Observed failure patterns", ""]
        lines += [
            f"- {item}" for item in failure.get("observed_failure_patterns", [])
        ]

        lines += [
            "",
            "### Manual correctness by category",
            "",
            "| Category | A | B | C | D |",
            "|---|---:|---:|---:|---:|",
        ]
        by_category = manual.get("by_category") or {}
        for category in final.get("test_category_counts", {}):
            cells = []
            for condition in CONDITION_ORDER:
                row = (by_category.get(condition) or {}).get(category) or {}
                cells.append(fmt(row.get("correctness_rate")))
            lines.append(f"| {category} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Rates are based on the 15-example manual sample; categories with one sampled response are especially noisy.",
            "",
            "### Root causes",
            "",
        ]
        lines += [f"- {item}" for item in failure.get("root_causes", [])]
        lines += [
            "",
            "### Required mitigations",
            "",
        ]
        lines += [f"- {item}" for item in failure.get("required_mitigations", [])]

    lines += [
        "",
        "## Deltas versus A (pretrained, full prompt, 256 tokens)",
        "",
        "| Comparison | p50 delta ms | p50 change | p95 delta ms | Peak RSS delta MB | Keyword coverage delta | Heuristic hallucination delta | Combined category accuracy delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in final["comparison"].items():
        lines.append(
            "| {name} | {p50d} | {p50p}% | {p95d} | {rss} | {kw} | {halluc} | {cat} |".format(
                name=name,
                p50d=row["latency_p50_delta_ms"],
                p50p=row["latency_p50_change_pct"],
                p95d=row["latency_p95_delta_ms"],
                rss=row["peak_rss_delta_mb"],
                kw=row["policy_keyword_coverage_delta"],
                halluc=row["heuristic_hallucination_delta"],
                cat=row["combined_accuracy_delta"],
            )
        )

    lines += [
        "",
        "## Dynamic INT8 rejection",
        "",
    ]
    dynamic = final.get("dynamic_int8_rejection", {})
    measured = dynamic.get("measured", {})
    lines.append(
        "Phase 2 dynamic INT8 was rejected, not selected as the Phase 3 D configuration."
    )
    if measured:
        lines.append(
            "Measured p95 latency: {p95} ms; peak RSS: {rss} MB; validity: {valid}; "
            "keyword coverage: {kw}.".format(
                p95=measured.get("latency_p95_ms"),
                rss=measured.get("peak_rss_mb"),
                valid=measured.get("response_validity_rate"),
                kw=measured.get("policy_keyword_coverage"),
            )
        )
    lines += [f"- {reason}" for reason in dynamic.get("reasons", [])]

    selection = final.get("selection", {})
    lines += [
        "",
        "## Deployment recommendation",
        "",
    ]
    if selection.get("config"):
        lines.append(f"Selected configuration: **{selection['config']}**")
        lines.append("")
        lines.append(selection.get("rationale", ""))
    else:
        lines.append(
            "Pending manual review. No deployment configuration is selected in this report yet."
        )

    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in final["limitations"]],
        "",
        "## Artifacts",
        "",
        "- `experiments/evaluation/predictions/`: per-condition predictions.",
        "- `experiments/evaluation/final_results.json`: combined results and metadata.",
        "- `experiments/evaluation/manual_review_sheet.jsonl`: 60 responses for the 15-example sample.",
        "- `experiments/evaluation/manual_review_scores.json`: scores and reviewer notes.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def assemble(config: dict, output_dir: Path) -> None:
    summaries = {}
    for condition in CONDITION_ORDER:
        label = config["configs"][condition]["label"]
        path = output_dir / f"condition_{label}_summary.json"
        if not path.exists():
            raise SystemExit(f"Missing summary: {path}")
        summaries[condition] = json.loads(path.read_text(encoding="utf-8"))

    comparison = {}
    for new_key, old_key in (
        ("B", "A"),
        ("C", "A"),
        ("D", "A"),
        ("C", "B"),
        ("D", "C"),
        ("D", "B"),
    ):
        comparison[f"{new_key}_vs_{old_key}"] = compare(
            summaries, new_key, old_key
        )

    test_file = Path(config["paths"]["test_file"])
    records, errors = read_jsonl(test_file)
    if errors:
        raise SystemExit(f"Could not parse test file: {errors[:3]}")
    records_by_id = {record["id"]: record for record in records}
    sample_ids = list(config["manual_review"]["sample_ids"])
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in records_by_id]
    if missing_ids:
        raise SystemExit(f"Manual-review IDs missing from test set: {missing_ids}")

    predictions_dir = output_dir / "predictions"
    scores_path = output_dir / "manual_review_scores.json"
    scores = None
    if scores_path.exists():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))

    missing_sheet = write_manual_sheet(
        config, records_by_id, output_dir, predictions_dir, scores
    )
    if missing_sheet:
        print(json.dumps({"manual_sheet_warnings": missing_sheet}, indent=2))

    manual_summary = manual_review_summary(scores) if scores else None
    manual_by_category = (
        manual_review_by_category(scores, records_by_id) if scores else None
    )
    failure_analysis = build_failure_analysis(summaries, manual_summary)
    selection_config = config.get("selection") or {}
    selection = {
        "config": selection_config.get("config"),
        "rationale": selection_config.get("rationale"),
        "status": "selected"
        if selection_config.get("config")
        else "pending_manual_review",
    }

    final = {
        "generated_at": utc_now(),
        "phase": "Phase 3 final evaluation",
        "test_file": str(test_file),
        "test_examples": len(records),
        "test_category_counts": dict(
            sorted(Counter(record["category"] for record in records).items())
        ),
        "environment": environment_info(config),
        "configs": summaries,
        "comparison": comparison,
        "manual_review": {
            "sample_ids": sample_ids,
            "sample_size": len(sample_ids),
            "responses_expected": len(sample_ids) * len(CONDITION_ORDER),
            "sheet_file": "experiments/evaluation/manual_review_sheet.jsonl",
            "scores_file": str(scores_path),
            "summary": manual_summary,
            "by_category": manual_by_category,
            "rubric": {
                "relevance": "1 if the response addresses the user's actual support question or correctly refuses an out-of-scope request; 0 otherwise.",
                "correctness": "1 if substantive claims and next steps match the gold reference and ShopEase policy; 0 if materially incorrect or missing a required clarification.",
                "policy_consistency": "1 if the response does not contradict ShopEase policy; 0 if it contradicts or invents policy, order, refund, delivery, account, or payment facts.",
                "hallucination": "1 if any unsupported or fabricated claim is present; 0 otherwise. Lower is better.",
            },
            "review_note": "One reviewer, 15 of 75 examples, not blinded across configurations.",
        },
        "dynamic_int8_rejection": load_phase2_dynamic_int8(),
        "failure_analysis": failure_analysis,
        "selection": selection,
        "limitations": [
            "The manual review covers 15 of 75 examples, is not blinded across configurations, and was scored by one reviewer.",
            "Automatic lexical metrics can miss paraphrases and can reward keyword overlap without semantic correctness.",
            "Heuristic hallucination checks target clear fabricated order details, contact information, dates, amounts, and explicit policy contradictions; subtle errors may be missed.",
            "The fallback category classifier is rule-based and imperfect; tag accuracy is reported separately.",
            "Selecting the final configuration on the test set is optimistic because the test set is no longer untouched for that decision.",
            "Latency and memory were measured on one CPU-only machine with 75 examples; p95 is not a robust cross-hardware tail estimate.",
            "The Phase 1 LoRA adapter is a feasibility artifact trained on 26 examples for one epoch; no quality improvement is assumed.",
            "Dynamic INT8 was rejected on measured latency, memory, and output quality, not because quantization is inherently invalid.",
            "The test set is synthetic/fictional and is not production traffic; no online A/B test or monitoring was performed.",
            "FastAPI deployment and production readiness are out of scope for Phase 3.",
        ],
    }

    (output_dir / "final_results.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    write_report(final, Path("reports/final_evaluation.md"))
    print(json.dumps(final, indent=2), flush=True)


def resolve_dirs(config: dict, args) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        config["paths"]["output_dir"]
    )
    predictions_dir = (
        Path(args.predictions_dir)
        if args.predictions_dir
        else output_dir / "predictions"
    )
    return output_dir, predictions_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_evaluation_config.yaml"),
    )
    parser.add_argument("--condition", choices=CONDITION_ORDER)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir, predictions_dir = resolve_dirs(config, args)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.assemble:
        assemble(config, output_dir)
        return

    if not args.condition:
        raise SystemExit("Use --condition A|B|C|D or --assemble.")

    test_file = Path(config["paths"]["test_file"])
    records, errors = read_jsonl(test_file)
    if errors:
        raise SystemExit(f"Could not parse test file: {errors[:3]}")
    if args.limit:
        records = records[: args.limit]

    if args.output_dir:
        config["paths"]["output_dir"] = str(output_dir)
    if args.predictions_dir:
        config["paths"]["predictions_dir"] = str(predictions_dir)
    run_condition(config, records, args.condition, output_dir, predictions_dir)
    if args.limit:
        print(
            json.dumps(
                {
                    "note": "This was a limited smoke run; do not use it for final results.",
                    "limit": args.limit,
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

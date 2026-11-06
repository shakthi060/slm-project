"""Phase 2 CPU inference benchmark.

Conditions:
  A: pretrained model, original prompt/generation settings
  B: pretrained + LoRA adapter, original prompt/generation settings
  C: pretrained model, compact prompt + capped generation (selected optimization)
  D: dynamically quantized pretrained model, compact prompt + capped generation
     (investigation only; selected only if measurements support it)

Each condition runs in its own process so peak RSS is not contaminated by the
previous condition. The same five frozen prompts are used for every condition.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import statistics
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import yaml

from src.data.clean_dataset import read_jsonl
from src.data.policy_data import ALLOWED_CATEGORIES, load_system_prompt
from src.evaluation.baseline import (
    COVERAGE_KEYWORDS,
    extract_user,
    percentile,
    write_jsonl,
)
from src.evaluation.transformer_baseline import parse_category, summarize

CONDITIONS = {
    "A": {
        "label": "A_pretrained_original",
        "adapter": False,
        "compact_prompt": False,
        "generation": "original",
        "quantized": False,
    },
    "B": {
        "label": "B_adapter_original",
        "adapter": True,
        "compact_prompt": False,
        "generation": "original",
        "quantized": False,
    },
    "C": {
        "label": "C_pretrained_optimized",
        "adapter": False,
        "compact_prompt": True,
        "generation": "optimized",
        "quantized": False,
    },
    "C1": {
        "label": "C1_full_prompt_capped",
        "adapter": False,
        "compact_prompt": False,
        "generation": "optimized",
        "quantized": False,
    },
    "C2": {
        "label": "C2_compact_prompt_uncapped",
        "adapter": False,
        "compact_prompt": True,
        "generation": "original",
        "quantized": False,
    },
    "D": {
        "label": "D_dynamic_int8_investigation",
        "adapter": False,
        "compact_prompt": True,
        "generation": "optimized",
        "quantized": True,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 1)


def process_peak_wset_mb() -> float | None:
    try:
        value = psutil.Process(os.getpid()).memory_info().peak_wset
        return round(value / 1024**2, 1)
    except AttributeError:
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_info(config: dict) -> dict:
    import transformers

    return {
        "generated_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": package_version("peft"),
        "accelerate": package_version("accelerate"),
        "datasets": package_version("datasets"),
        "cuda_available": torch.cuda.is_available(),
        "device": config["runtime"]["device"],
        "dtype": config["runtime"]["dtype"],
        "threads": int(config["runtime"]["threads"]),
        "interop_threads": int(config["runtime"]["interop_threads"]),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }


def load_prompts(config: dict) -> list[dict]:
    baseline_rows, errors = read_jsonl(Path(config["paths"]["baseline_prompts"]))
    if errors:
        raise SystemExit(f"Could not parse baseline prompts: {errors[:3]}")
    test_rows, errors = read_jsonl(Path(config["paths"]["test_file"]))
    if errors:
        raise SystemExit(f"Could not parse test file: {errors[:3]}")
    test_by_id = {row["id"]: row for row in test_rows}

    prompts = []
    for row in baseline_rows[: config["prompts"]["limit"]]:
        record = test_by_id.get(row["id"])
        if record is None:
            raise SystemExit(f"Prompt {row['id']} is not in the frozen test file.")
        prompts.append(
            {
                "id": row["id"],
                "expected_category": record["category"],
                "messages": row["messages"],
                "user": extract_user(row["messages"]),
            }
        )
    return prompts


def build_messages(prompt: dict, config: dict, compact_prompt: bool) -> tuple[list[dict], str]:
    if compact_prompt:
        system_prompt = str(config["prompts"]["compact_system_prompt"]).strip()
        variant = "compact_system_prompt"
    else:
        system_prompt = load_system_prompt()
        variant = "full_system_prompt"
    return (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["user"]},
        ],
        variant,
    )


def load_model(config: dict, condition: str) -> tuple[object, object, dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = CONDITIONS[condition]
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

    quantize_s = 0.0
    rss_after_quant_mb = None
    quantization = "none"
    if spec["quantized"]:
        base_for_quant = model
        quantize_start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            model = torch.ao.quantization.quantize_dynamic(
                base_for_quant, {torch.nn.Linear}, dtype=torch.qint8
            )
        quantize_s = time.perf_counter() - quantize_start
        quantization = "dynamic_int8_linear"
        del base_for_quant
        gc.collect()
        rss_after_quant_mb = rss_mb()

    model.eval()
    total_load_s = (
        tokenizer_load_s + base_load_s + adapter_apply_s + quantize_s
    )
    load_info = {
        "tokenizer_load_s": round(tokenizer_load_s, 3),
        "base_load_s": round(base_load_s, 3),
        "adapter_apply_s": round(adapter_apply_s, 3),
        "quantize_s": round(quantize_s, 3),
        "model_load_s": round(total_load_s, 3),
        "rss_before_load_mb": rss_before,
        "rss_after_load_mb": rss_mb(),
        "rss_after_quant_mb": rss_after_quant_mb,
        "adapter_loaded": adapter_loaded,
        "adapter_dir": str(adapter_dir) if adapter_loaded else None,
        "quantization": quantization,
        "dtype": "float32",
        "device": config["runtime"]["device"],
    }
    print(json.dumps({"condition": condition, "load": load_info}, indent=2), flush=True)
    return model, tokenizer, load_info


def generate_once(
    model,
    tokenizer,
    messages: list[dict],
    generation: dict,
) -> dict:
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


def run_condition(config: dict, condition: str) -> None:
    spec = CONDITIONS[condition]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    label = spec["label"]

    threads = int(config["runtime"]["threads"])
    interop = int(config["runtime"]["interop_threads"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass

    prompts = load_prompts(config)
    model, tokenizer, load_info = load_model(config, condition)
    generation = config["generation"][spec["generation"]]
    first_messages, variant = build_messages(prompts[0], config, spec["compact_prompt"])
    system_prompt_tokens = len(
        tokenizer(first_messages[0]["content"], add_special_tokens=False)["input_ids"]
    )

    peak_rss = rss_mb()
    warmup_start = time.perf_counter()
    for prompt in prompts[: int(config["runtime"]["warmup_runs"])]:
        messages, _ = build_messages(prompt, config, spec["compact_prompt"])
        generate_once(model, tokenizer, messages, generation)
        peak_rss = max(peak_rss, rss_mb())
    warmup_s = time.perf_counter() - warmup_start
    rss_after_warmup = rss_mb()

    rows = []
    inference_start = time.perf_counter()
    for prompt in prompts:
        messages, variant = build_messages(prompt, config, spec["compact_prompt"])
        result = generate_once(model, tokenizer, messages, generation)
        peak_rss = max(peak_rss, rss_mb())
        response = result["response"]
        predicted = parse_category(response)
        expected = prompt["expected_category"]
        valid = predicted != "unknown" and len(response) > 20
        rows.append(
            {
                "condition": condition,
                "condition_label": label,
                "id": prompt["id"],
                "expected_category": expected,
                "predicted_category": predicted,
                "user": prompt["user"],
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
                "last_token": result["last_token"],
                "valid": valid,
                "policy_keyword_covered": any(
                    keyword in response.lower()
                    for keyword in COVERAGE_KEYWORDS.get(expected, [])
                ),
                "refusal": any(
                    phrase in response.lower()
                    for phrase in ["only help", "cannot", "outside", "not able"]
                ),
                "asks_clarification": "?" in response,
                "rss_mb": rss_mb(),
            }
        )
        print(
            f"{condition} {prompt['id']} {result['latency_ms']:.1f}ms "
            f"in={result['prompt_tokens']} out={result['output_tokens']} "
            f"rss={rows[-1]['rss_mb']:.1f}MB",
            flush=True,
        )
    inference_s = time.perf_counter() - inference_start
    peak_rss = max(peak_rss, rss_mb())

    base_summary = summarize(rows)
    latencies = [row["latency_ms"] for row in rows]
    total_requests = [row["total_request_ms"] for row in rows]
    output_tokens = [row["output_tokens"] for row in rows]
    prompt_tokens = [row["prompt_tokens"] for row in rows]
    summary = {
        "condition": condition,
        "condition_label": label,
        "generated_at": utc_now(),
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_path": str(Path(config["paths"]["model"])),
        "prompt_ids": [row["id"] for row in rows],
        "system_prompt_variant": rows[0]["system_prompt_variant"],
        "system_prompt_tokens": system_prompt_tokens,
        "generation": generation,
        "load": load_info,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "warmup_runs": int(config["runtime"]["warmup_runs"]),
        "warmup_s": round(warmup_s, 3),
        "rss_after_warmup_mb": rss_after_warmup,
        "peak_rss_mb": peak_rss,
        "process_peak_wset_mb": process_peak_wset_mb(),
        "inference_s": round(inference_s, 3),
        "examples": len(rows),
        "latency_mean_ms": round(statistics.mean(latencies), 3),
        "latency_min_ms": round(min(latencies), 3),
        "latency_max_ms": round(max(latencies), 3),
        "latency_p50_ms": round(statistics.median(latencies), 3),
        "latency_p95_ms": round(percentile(latencies, 0.95), 3),
        "total_request_p50_ms": round(statistics.median(total_requests), 3),
        "total_request_p95_ms": round(percentile(total_requests, 0.95), 3),
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 2),
        "prompt_tokens_min": min(prompt_tokens),
        "prompt_tokens_max": max(prompt_tokens),
        "output_tokens_mean": round(statistics.mean(output_tokens), 2),
        "output_tokens_min": min(output_tokens),
        "output_tokens_max": max(output_tokens),
        "truncation_rate": round(
            sum(row["truncated"] for row in rows) / len(rows), 4
        ),
        **base_summary,
        "quality_note": (
            "Automatic proxies only. Category tags, keyword coverage, refusals and "
            "clarification are not a substitute for blinded human review."
        ),
        "latency_note": (
            "Five measured requests. p95 is effectively the maximum of five samples "
            "and is not a statistically robust tail estimate."
        ),
        "comparison_note": (
            "A is the pretrained baseline under original settings; B adds the LoRA "
            "adapter under the same original settings; C applies inference "
            "optimization to the pretrained model. Adaptation and inference "
            "optimization are kept separate."
        ),
    }

    write_jsonl(output_dir / f"condition_{label}.jsonl", rows)
    (output_dir / f"condition_{label}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def assemble(config: dict) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    summaries = {}
    for condition, spec in CONDITIONS.items():
        path = output_dir / f"condition_{spec['label']}_summary.json"
        if path.exists():
            summaries[condition] = json.loads(path.read_text(encoding="utf-8"))

    missing = [condition for condition in ("A", "B", "C") if condition not in summaries]
    if missing:
        raise SystemExit(f"Missing condition summaries: {missing}")

    def delta(new: float, old: float) -> float | None:
        if old == 0:
            return None
        return round(new - old, 3)

    def pct(new: float, old: float) -> float | None:
        if old == 0:
            return None
        return round((new - old) / old * 100, 2)

    comparison = {}
    pairs = [("C", "A"), ("B", "A"), ("C", "B")]
    for new_key, old_key in [("C1", "A"), ("C2", "A"), ("C1", "C"), ("C2", "C")]:
        if new_key in summaries:
            pairs.append((new_key, old_key))
    for new_key, old_key in pairs:
        new = summaries[new_key]
        old = summaries[old_key]
        comparison[f"{new_key}_vs_{old_key}"] = {
            "latency_p50_delta_ms": delta(
                new["latency_p50_ms"], old["latency_p50_ms"]
            ),
            "latency_p50_change_pct": pct(
                new["latency_p50_ms"], old["latency_p50_ms"]
            ),
            "latency_p95_delta_ms": delta(
                new["latency_p95_ms"], old["latency_p95_ms"]
            ),
            "latency_p95_change_pct": pct(
                new["latency_p95_ms"], old["latency_p95_ms"]
            ),
            "total_request_p50_delta_ms": delta(
                new["total_request_p50_ms"], old["total_request_p50_ms"]
            ),
            "total_request_p50_change_pct": pct(
                new["total_request_p50_ms"], old["total_request_p50_ms"]
            ),
            "model_load_delta_s": delta(
                new["load"]["model_load_s"], old["load"]["model_load_s"]
            ),
            "peak_rss_delta_mb": delta(
                new["peak_rss_mb"], old["peak_rss_mb"]
            ),
            "prompt_tokens_mean_delta": delta(
                new["prompt_tokens_mean"], old["prompt_tokens_mean"]
            ),
            "output_tokens_mean_delta": delta(
                new["output_tokens_mean"], old["output_tokens_mean"]
            ),
        }

    combined = {
        "generated_at": utc_now(),
        "environment": environment_info(config),
        "prompt_ids": summaries["A"]["prompt_ids"],
        "conditions": summaries,
        "comparison": comparison,
        "limitations": [
            "Only five frozen prompts were measured.",
            "p95 of five samples is effectively the maximum, not a robust tail estimate.",
            "Automatic quality proxies are not blinded human evaluation.",
            "Latency improvements do not imply quality improvements.",
            "The LoRA adapter was trained on only 26 examples for one epoch.",
        ],
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    lines = [
        "# Phase 2 Benchmark Comparison",
        "",
        f"Generated: {combined['generated_at']}",
        "",
        "| Condition | Model | Prompt | Max new tokens | p50 ms | p95 ms | Load s | Peak RSS MB | Mean output tokens | Tag rate | Keyword coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("A", "B", "C", "C1", "C2", "D"):
        if condition not in summaries:
            continue
        row = summaries[condition]
        model_name = "adapter" if condition == "B" else (
            "pretrained-int8" if condition == "D" else "pretrained"
        )
        lines.append(
            "| {label} | {model} | {prompt} | {max_new} | {p50} | {p95} | {load} | {rss} | {out} | {tag} | {kw} |".format(
                label=row["condition_label"],
                model=model_name,
                prompt=row["system_prompt_variant"],
                max_new=row["generation"]["max_new_tokens"],
                p50=row["latency_p50_ms"],
                p95=row["latency_p95_ms"],
                load=row["load"]["model_load_s"],
                rss=row["peak_rss_mb"],
                out=row["output_tokens_mean"],
                tag=row["category_tag_rate"],
                kw=row["policy_keyword_coverage"],
            )
        )
    lines += [
        "",
        "## Deltas",
        "",
        "| Comparison | p50 delta ms | p50 change | p95 delta ms | Load delta s | Peak RSS delta MB | Prompt-token delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in comparison.items():
        lines.append(
            "| {name} | {p50d} | {p50p}% | {p95d} | {load} | {rss} | {prompt} |".format(
                name=name,
                p50d=row["latency_p50_delta_ms"],
                p50p=row["latency_p50_change_pct"],
                p95d=row["latency_p95_delta_ms"],
                load=row["model_load_delta_s"],
                rss=row["peak_rss_delta_mb"],
                prompt=row["prompt_tokens_mean_delta"],
            )
        )
    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in combined["limitations"]],
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(combined, indent=2), flush=True)


def paired_check(config: dict) -> None:
    """Same-process, order-balanced latency check for A, C, C1, C2.

    This controls for CPU state and model-load differences. It does not replace
    the condition runs, but it shows whether the standalone deltas reproduce.
    """
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    threads = int(config["runtime"]["threads"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(int(config["runtime"]["interop_threads"]))
    except RuntimeError:
        pass

    prompts = load_prompts(config)
    model, tokenizer, load_info = load_model(config, "A")
    conditions = ["A", "C", "C1", "C2"]

    def one(condition: str, prompt: dict) -> dict:
        spec = CONDITIONS[condition]
        messages, variant = build_messages(prompt, config, spec["compact_prompt"])
        generation = config["generation"][spec["generation"]]
        result = generate_once(model, tokenizer, messages, generation)
        return {
            "condition": condition,
            "condition_label": spec["label"],
            "id": prompt["id"],
            "system_prompt_variant": variant,
            "max_new_tokens": generation["max_new_tokens"],
            "latency_ms": result["latency_ms"],
            "total_request_ms": result["total_request_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "output_tokens": result["output_tokens"],
            "truncated": result["truncated"],
        }

    for prompt in prompts[:2]:
        for condition in ("A", "C"):
            one(condition, prompt)

    rows = []
    for pass_index in range(2):
        order = conditions if pass_index == 0 else list(reversed(conditions))
        for prompt in prompts:
            for condition in order:
                row = one(condition, prompt)
                row["pass"] = pass_index + 1
                rows.append(row)

    by_condition = {}
    for condition in conditions:
        values = [row["latency_ms"] for row in rows if row["condition"] == condition]
        totals = [
            row["total_request_ms"] for row in rows if row["condition"] == condition
        ]
        by_condition[condition] = {
            "label": CONDITIONS[condition]["label"],
            "measurements": len(values),
            "latency_p50_ms": round(statistics.median(values), 3),
            "latency_p95_ms": round(percentile(values, 0.95), 3),
            "latency_mean_ms": round(statistics.mean(values), 3),
            "latency_min_ms": round(min(values), 3),
            "latency_max_ms": round(max(values), 3),
            "total_request_p50_ms": round(statistics.median(totals), 3),
            "total_request_p95_ms": round(percentile(totals, 0.95), 3),
            "output_tokens_mean": round(
                statistics.mean(
                    row["output_tokens"]
                    for row in rows
                    if row["condition"] == condition
                ),
                2,
            ),
            "truncation_rate": round(
                sum(
                    row["truncated"]
                    for row in rows
                    if row["condition"] == condition
                )
                / len(values),
                4,
            ),
        }

    paired_deltas = {}
    for new_key in ("C", "C1", "C2"):
        deltas = []
        for pass_index in (1, 2):
            for prompt in prompts:
                new_row = next(
                    row
                    for row in rows
                    if row["condition"] == new_key
                    and row["id"] == prompt["id"]
                    and row["pass"] == pass_index
                )
                old_row = next(
                    row
                    for row in rows
                    if row["condition"] == "A"
                    and row["id"] == prompt["id"]
                    and row["pass"] == pass_index
                )
                deltas.append(new_row["latency_ms"] - old_row["latency_ms"])
        paired_deltas[f"{new_key}_minus_A"] = {
            "median_delta_ms": round(statistics.median(deltas), 3),
            "mean_delta_ms": round(statistics.mean(deltas), 3),
            "measurements": len(deltas),
        }

    summary = {
        "generated_at": utc_now(),
        "description": (
            "Same-process check: base model loaded once, A/C/C1/C2 interleaved, "
            "two passes with reversed order. Complements the standalone runs."
        ),
        "load": load_info,
        "prompts": [prompt["id"] for prompt in prompts],
        "conditions": by_condition,
        "paired_deltas": paired_deltas,
        "limitations": [
            "Ten measurements per condition (two passes over five prompts).",
            "Order was balanced across two passes but CPU frequency still varies.",
            "This is a latency check, not a quality evaluation.",
        ],
    }
    write_jsonl(output_dir / "paired_latency.jsonl", rows)
    (output_dir / "paired_latency.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/optimization_config.yaml")
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default=None)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.assemble:
        assemble(config)
        return
    if args.paired:
        paired_check(config)
        return
    if args.condition is None:
        raise SystemExit("Pass --condition A|B|C|D/C1/C2, --paired, or --assemble.")
    run_condition(config, args.condition)


if __name__ == "__main__":
    main()

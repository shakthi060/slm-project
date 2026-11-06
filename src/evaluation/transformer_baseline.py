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
from src.evaluation.baseline import (
    COVERAGE_KEYWORDS,
    extract_user,
    percentile,
    write_jsonl,
)

CATEGORY_PATTERN = re.compile(r"\[category:\s*([a-z_]+)\]")

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_MODEL_PATH = Path(r"D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct")
DEFAULT_OUTPUT_DIR = Path("experiments/baseline_generative")
DEFAULT_REPORT_NAME = "generative_baseline_report.md"


def parse_category(response: str) -> str:
    match = CATEGORY_PATTERN.search(response)
    if match and match.group(1) in ALLOWED_CATEGORIES:
        return match.group(1)
    return "unknown"


def load_model(model_path: Path, revision: str | None, use_4bit: bool, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    is_local = Path(model_path).exists()
    tokenizer_kwargs = {
        "trust_remote_code": False,
        "padding_side": "right",
    }
    if revision and not is_local:
        tokenizer_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": False,
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "attn_implementation": "sdpa",
    }
    if revision and not is_local:
        load_kwargs["revision"] = revision
    if use_4bit and device == "cuda":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = None if device == "cpu" else {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], generation: dict) -> tuple[str, float, int]:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=generation["max_new_tokens"],
            do_sample=generation["do_sample"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            repetition_penalty=generation["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    new_tokens = output[0][input_tokens:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, latency_ms, len(new_tokens)


def run(records: list[dict], model, tokenizer, config: dict) -> list[dict]:
    import torch

    generation = config["generation"]
    warmup_runs = config["latency"]["warmup_runs"]
    system_prompt = load_system_prompt()

    for record in records[:warmup_runs]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extract_user(record["messages"])},
        ]
        generate(model, tokenizer, messages, generation)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = []
    for record in records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extract_user(record["messages"])},
        ]
        response, latency_ms, output_tokens = generate(
            model, tokenizer, messages, generation
        )
        predicted = parse_category(response)
        expected = record["category"]
        valid = predicted != "unknown" and len(response) > 20
        results.append(
            {
                "id": record["id"],
                "expected_category": expected,
                "predicted_category": predicted,
                "user": messages[-1]["content"],
                "response": response,
                "latency_ms": round(latency_ms, 3),
                "output_tokens": output_tokens,
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

    tagged = sum(1 for row in results if row["predicted_category"] != "unknown")
    return {
        "examples": total,
        "category_tag_rate": round(tagged / total, 4) if total else 0.0,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation_config.yaml"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    args = parser.parse_args()

    import torch

    weights = args.model_path / "model.safetensors"
    index = args.model_path / "model.safetensors.index.json"
    if not weights.exists() and not index.exists():
        raise SystemExit(
            f"Model weights not found in {args.model_path}. Download is still incomplete."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records, errors = read_jsonl(Path(config["data"]["test_file"]))
    if errors:
        raise SystemExit(f"Could not parse test file: {errors[:3]}")
    if args.limit:
        records = records[: args.limit]

    use_4bit = not args.no_4bit and args.device == "cuda"
    if args.device == "cuda" and not use_4bit:
        print("Running GPU inference without 4-bit quantization (bf16).")
    load_start = time.perf_counter()
    model, tokenizer = load_model(
        args.model_path, args.revision, use_4bit, args.device
    )
    model_load_s = time.perf_counter() - load_start

    results = run(records, model, tokenizer, config)
    summary = summarize(results)
    summary["model_load_s"] = round(model_load_s, 3)
    if torch.cuda.is_available():
        summary["peak_vram_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2, 1
        )

    baseline_dir = args.output_dir
    baseline_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "transformers_pretrained",
        "baseline_type": "generative_slm",
        "model_id": args.model_id,
        "revision": args.revision or "local_checkpoint",
        "model_path": str(args.model_path),
        "quantization": "4-bit NF4 double-quant" if use_4bit else "none",
        "dtype": "bfloat16" if args.device == "cuda" else "float32",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": config.get("seed"),
        "test_file": config["data"]["test_file"],
        "examples": len(results),
        "generation": config["generation"],
        "note": "Pretrained SmolLM2-135M-Instruct, no fine-tuning. Test set frozen.",
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
                "model_load_s": summary["model_load_s"],
                "latency_p50_ms": summary["latency_p50_ms"],
                "latency_p95_ms": summary["latency_p95_ms"],
                "peak_vram_mb": summary["peak_vram_mb"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (baseline_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (baseline_dir / "observed_limitations.md").write_text(
        "# Observed Limitations\n\n"
        f"- Baseline uses the pretrained `{args.model_id}` model without fine-tuning.\n"
        f"- Inference device: `{args.device}`; quantization: {run_config['quantization']}.\n"
        "- The base model was not trained to emit the `[category: ...]` tag, so format validity\n"
        "  and category accuracy are expected to be low; this is a baseline limitation.\n"
        f"- Latency is measured with batch size 1 on `{args.device}`.\n",
        encoding="utf-8",
    )

    report_path = Path(config["artifacts"]["reports_dir"]) / args.report_name
    lines = [
        "# Generative SLM Baseline Report",
        "",
        f"Generated: {run_config['generated_at']}",
        f"Backend: `{run_config['backend']}` (generative baseline)",
        f"Model: `{run_config['model_id']}` ({run_config['revision']})",
        f"Model path: `{run_config['model_path']}`",
        f"Device: `{run_config['device']}` — dtype {run_config['dtype']}, "
        f"quantization {run_config['quantization']}",
        f"Test examples: {summary['examples']}",
        "",
        "## Generation settings",
        "",
        "| Setting | Value |",
        "|---|---:|",
    ]
    for key, value in config["generation"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Category tag rate | {summary['category_tag_rate']:.4f} |",
        f"| Category accuracy | {summary['category_accuracy']:.4f} |",
        f"| Response validity rate | {summary['response_validity_rate']:.4f} |",
        f"| Unsupported refusal rate | {summary['unsupported_refusal_rate']:.4f} |",
        f"| Policy keyword coverage | {summary['policy_keyword_coverage']:.4f} |",
        f"| Clarification rate | {summary['clarification_rate']:.4f} |",
        f"| Model load (s) | {summary['model_load_s']:.3f} |",
        f"| Latency p50 (ms) | {summary['latency_p50_ms']:.3f} |",
        f"| Latency p95 (ms) | {summary['latency_p95_ms']:.3f} |",
        f"| Peak VRAM (MB) | {summary['peak_vram_mb']} |",
        "",
        "## Sample outputs",
        "",
    ])
    for row in results[:5]:
        lines.extend(
            [
                f"**{row['id']}** — expected `{row['expected_category']}`, "
                f"predicted `{row['predicted_category']}`",
                "",
                f"> {row['user']}",
                "",
                row["response"] or "(empty response)",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "Pretrained generative baseline only, with no fine-tuning. The test set is frozen and",
            "is not used for training, hyperparameter tuning, or model selection.",
            "",
            "This baseline is kept separate from the deterministic rule-based baseline in",
            "`experiments/baseline/` (category accuracy 0.6800 on the full 75-example test set).",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Backend: {run_config['backend']}")
    print(f"Model: {run_config['model_id']}")
    print(f"Device: {run_config['device']}")
    print(f"Examples: {summary['examples']}")
    for key, value in summary.items():
        if key != "examples":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import yaml

from src.data.clean_dataset import read_jsonl
from src.data.policy_data import load_system_prompt
from src.evaluation.transformer_baseline import parse_category, summarize


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def package_versions() -> dict:
    import accelerate
    import datasets
    import peft
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
    }


def rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / 1024**2, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def token_count(tokenizer, record: dict) -> int:
    full = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer(full, add_special_tokens=False)["input_ids"])


def select_examples(records: list[dict], tokenizer, per_category: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    selected: list[dict] = []
    for category in sorted(grouped):
        candidates = sorted(
            grouped[category],
            key=lambda record: (token_count(tokenizer, record), record["id"]),
        )
        selected.extend(candidates[:per_category])
    return selected


def build_example(tokenizer, record: dict, max_seq_length: int) -> dict:
    messages = record["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][
        :max_seq_length
    ]
    labels = list(full_ids)
    mask_length = min(len(prompt_ids), len(labels))
    labels[:mask_length] = [-100] * mask_length
    if all(label == -100 for label in labels):
        raise ValueError(f"No assistant tokens survive truncation for {record['id']}")
    return {
        "id": record["id"],
        "category": record["category"],
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def build_environment(
    config: dict, records: list[dict], selected: list[dict], tokenizer
) -> dict:
    lengths = [token_count(tokenizer, record) for record in selected]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packages": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "ram_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "base_rss_mb": rss_mb(),
        "train_file": config["data"]["train_file"],
        "train_records_total": len(records),
        "selected_records": len(selected),
        "selected_categories": sorted({record["category"] for record in selected}),
        "selected_token_min": min(lengths),
        "selected_token_median": statistics.median(lengths),
        "selected_token_max": max(lengths),
        "max_seq_length": config["data"]["max_seq_length"],
        "selection_rule": "two shortest-by-token-length examples per category, id tie-break",
        "selected_ids": [record["id"] for record in selected],
    }


def save_adapter(model, tokenizer, adapter_dir: Path) -> dict:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    elapsed = time.perf_counter() - start
    files = sorted(
        str(path.relative_to(adapter_dir))
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    return {
        "adapter_dir": str(adapter_dir),
        "save_s": round(elapsed, 3),
        "files": files,
        "bytes": sum((adapter_dir / name).stat().st_size for name in files),
    }


def train(config: dict) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    training_cfg = config["training"]
    output_dir = Path(training_cfg["output_dir"])
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"])
    torch.set_num_threads(training_cfg["num_threads"])
    process = psutil.Process(os.getpid())
    peak_rss = rss_mb()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"], trust_remote_code=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records, errors = read_jsonl(Path(config["data"]["train_file"]))
    if errors:
        raise RuntimeError(f"Could not parse training file: {errors[:3]}")
    selected = select_examples(
        records, tokenizer, config["data"]["examples_per_category"]
    )
    environment = build_environment(config, records, selected, tokenizer)
    (output_dir / "environment_check.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (output_dir / "selected_examples.json").write_text(
        json.dumps(
            [
                {
                    "id": record["id"],
                    "category": record["category"],
                    "tokens": token_count(tokenizer, record),
                }
                for record in selected
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    load_start = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"], dtype=torch.float32
    )
    base.config.use_cache = False
    lora_config = LoraConfig(**config["lora"])
    model = get_peft_model(base, lora_config)
    model.train()
    model_load_s = time.perf_counter() - load_start

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    peak_rss = max(peak_rss, rss_mb())

    examples = [
        build_example(tokenizer, record, config["data"]["max_seq_length"])
        for record in selected
    ]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    accumulation = training_cfg["gradient_accumulation_steps"]
    log_rows: list[dict] = []
    optimizer_steps = 0
    history: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()

    try:
        for epoch in range(training_cfg["epochs"]):
            accum = 0
            for index, example in enumerate(examples):
                step_start = time.perf_counter()
                output = model(
                    input_ids=example["input_ids"],
                    attention_mask=example["attention_mask"],
                    labels=example["labels"],
                )
                raw_loss = float(output.loss.detach())
                (output.loss / accumulation).backward()
                accum += 1
                step_time = time.perf_counter() - step_start
                did_step = False
                if accum == accumulation or index == len(examples) - 1:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        training_cfg["max_grad_norm"],
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    optimizer_steps += 1
                    did_step = True
                history.append(raw_loss)
                current_rss = rss_mb()
                peak_rss = max(peak_rss, current_rss)
                row = {
                    "epoch": epoch + 1,
                    "example_index": index + 1,
                    "id": example["id"],
                    "category": example["category"],
                    "loss": round(raw_loss, 6),
                    "step_time_s": round(step_time, 4),
                    "rss_mb": current_rss,
                    "optimizer_step": did_step,
                }
                log_rows.append(row)
                if (
                    (index + 1) % training_cfg["logging_steps"] == 0
                    or index == len(examples) - 1
                ):
                    print(
                        f"epoch {epoch + 1} example {index + 1}/{len(examples)} "
                        f"loss {raw_loss:.4f} rss {current_rss:.1f}MB",
                        flush=True,
                    )
    except Exception as exc:
        failure = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise

    train_s = time.perf_counter() - train_start
    save_info = save_adapter(model, tokenizer, adapter_dir)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_ready = adapter_weights.exists() and (adapter_dir / "adapter_config.json").exists()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if adapter_ready else "failed_adapter_missing",
        "model_id": config["model"]["id"],
        "model_path": config["model"]["path"],
        "device": "cpu",
        "dtype": "float32",
        "quantization": "none",
        "examples": len(examples),
        "categories": len({record["category"] for record in selected}),
        "epochs": training_cfg["epochs"],
        "max_seq_length": config["data"]["max_seq_length"],
        "lora_r": config["lora"]["r"],
        "lora_alpha": config["lora"]["lora_alpha"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": accumulation,
        "model_load_s": round(model_load_s, 3),
        "train_s": round(train_s, 3),
        "seconds_per_example": round(train_s / len(examples), 3),
        "initial_loss": round(history[0], 6) if history else None,
        "final_loss": round(history[-1], 6) if history else None,
        "mean_loss": round(statistics.mean(history), 6) if history else None,
        "peak_rss_mb": peak_rss,
        "adapter": save_info,
        "adapter_ready": adapter_ready,
        "note": "Feasibility run only: 26 examples, 1 epoch. Not a quality fine-tune.",
    }
    (output_dir / "training_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in log_rows), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    report = [
        "# CPU LoRA Feasibility Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Status: **{summary['status']}**",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Base model | `{summary['model_id']}` |",
        f"| Device | {summary['device']} |",
        f"| Dtype / quantization | {summary['dtype']} / {summary['quantization']} |",
        f"| Examples | {summary['examples']} ({summary['categories']} categories) |",
        f"| Epochs | {summary['epochs']} |",
        f"| Max sequence length | {summary['max_seq_length']} |",
        f"| LoRA r / alpha | {summary['lora_r']} / {summary['lora_alpha']} |",
        f"| Trainable params | {summary['trainable_params']} ({summary['trainable_pct']}%) |",
        f"| Optimizer steps | {summary['optimizer_steps']} |",
        f"| Model load | {summary['model_load_s']} s |",
        f"| Training | {summary['train_s']} s ({summary['seconds_per_example']} s/example) |",
        f"| Initial / final loss | {summary['initial_loss']} / {summary['final_loss']} |",
        f"| Peak process RSS | {summary['peak_rss_mb']} MB |",
        f"| Adapter files | {', '.join(summary['adapter']['files'])} |",
        "",
        "This run uses 26 training examples for one epoch on CPU. It verifies that the PEFT",
        "LoRA path runs end-to-end and that the adapter saves correctly. It does not establish",
        "fine-tuned quality; that requires a longer run and the Phase 3 evaluation.",
    ]
    (output_dir / "training_summary.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def verify_adapter(config: dict) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(config["training"]["output_dir"])
    adapter_dir = output_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"Adapter not found in {adapter_dir}")

    eval_config = yaml.safe_load(
        Path(config["evaluation"]["config"]).read_text(encoding="utf-8")
    )
    records, errors = read_jsonl(Path(eval_config["data"]["test_file"]))
    if errors:
        raise SystemExit(f"Could not parse test file: {errors[:3]}")
    records = records[: config["evaluation"]["limit"]]
    generation = eval_config["generation"]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["path"], trust_remote_code=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"], dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    system_prompt = load_system_prompt()
    results = []
    for record in records:
        user = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "user"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]
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
        latency_ms = (time.perf_counter() - start) * 1000
        new_tokens = output[0][input_tokens:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        predicted = parse_category(response)
        expected = record["category"]
        results.append(
            {
                "id": record["id"],
                "expected_category": expected,
                "predicted_category": predicted,
                "user": user,
                "response": response,
                "latency_ms": round(latency_ms, 3),
                "output_tokens": len(new_tokens),
                "valid": predicted != "unknown" and len(response) > 20,
                "policy_keyword_covered": False,
                "refusal": any(
                    phrase in response.lower()
                    for phrase in ["only help", "cannot", "outside", "not able"]
                ),
                "asks_clarification": "?" in response,
            }
        )

    from src.evaluation.baseline import COVERAGE_KEYWORDS

    for row in results:
        row["policy_keyword_covered"] = any(
            keyword in row["response"].lower()
            for keyword in COVERAGE_KEYWORDS.get(row["expected_category"], [])
        )
    summary = summarize(results)
    summary["adapter_dir"] = str(adapter_dir)
    summary["adapter_loaded"] = True
    summary["adapter_weights"] = str(adapter_dir / "adapter_model.safetensors")
    summary["generation"] = generation

    name = config["evaluation"]["output_name"]
    write_jsonl(output_dir / f"{name}.jsonl", results)
    (output_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    for row in results:
        print(f"{row['id']} expected={row['expected_category']} predicted={row['predicted_category']}")
        print(row["response"][:300].replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_cpu_lora.yaml")
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.verify_only:
        verify_adapter(config)
    else:
        train(config)


if __name__ == "__main__":
    main()

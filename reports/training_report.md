# Training Report — Phase 1 CPU LoRA Feasibility

**Date:** 2026-11-05
**Scope:** Phase 1 only. No GPU training, no QLoRA, no full fine-tuning.
**Status:** CPU LoRA feasibility run completed; adapter saved and reloaded successfully.
**Quality claim:** none. 26 examples for 1 epoch cannot establish fine-tuned quality.

---

## 1. Environment check (recorded before training)

| Item | Measured value |
|---|---|
| Python | 3.10.9 |
| torch | 2.13.0+cpu |
| transformers | 5.16.1 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |
| safetensors | 0.8.0 |
| CUDA available | False |
| Torch threads | 6 |
| CPU | AMD64 Family 25 Model 80, 12 logical CPUs |
| RAM total / available at check | 15.34 GB / 3.28 GB |
| Base process RSS | 437.0 MB |

Machine-readable evidence: `experiments/training/cpu_lora_feasibility/environment_check.json`.

`peft`, `accelerate` and `datasets` were not installed at the start of this phase. They were
installed from PyPI without touching CUDA or PyTorch. CUDA was not retried.

A pre-training smoke test confirmed that `get_peft_model` built a LoRA adapter on CPU, a forward
and backward pass completed, and gradients flowed. The smoke test measured 2,442,240 trainable
parameters and about 1.4 GB RSS. Training was only started after this check passed.

## 2. Dataset and chat format inspected

- File: `data/train.jsonl`, 654 records, 13 categories.
- Every record has `system`, `user`, `assistant` messages in ChatML format.
- Assistant targets begin with `[category: <label>]`.
- Full-sequence token lengths across the training set: min 176, median 206, max 237.
- Assistant-only token lengths: min 24, median 48, max 67.

The feasibility subset is deterministic: the two shortest-by-token-length examples from each of
the 13 categories (id tie-break), giving **26 examples**. Selected token lengths: min 176,
median 190, max 216. `max_seq_length` was set to **224**, so no selected example was truncated.
Selected IDs: `experiments/training/cpu_lora_feasibility/selected_examples.json`.

Loss was applied only to assistant tokens; system and user tokens were masked with `-100`.

## 3. LoRA configuration

| Setting | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Device | cpu |
| Dtype | float32 |
| Quantization | none |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 2,442,240 (1.7832% of 136,957,248) |
| Epochs | 1 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Optimizer steps | 7 |
| Learning rate | 2.0e-4 |
| Max sequence length | 224 |

Config: `configs/training_cpu_lora.yaml`.

## 4. Measured training result

| Metric | Measured value |
|---|---:|
| Model load | 0.919 s |
| Training time | 39.745 s |
| Seconds per example | 1.529 s |
| Initial loss | 3.585967 |
| Final loss | 3.845357 |
| Mean loss | 3.268124 |
| Peak process RSS | 1457.2 MB |
| Adapter save time | 0.174 s |
| Adapter weights size | 9,823,216 bytes |
| Adapter total with tokenizer | 13,353,481 bytes |

Loss was noisy and did not decrease monotonically (`3.5860` to `3.8454`). This is expected for
26 examples, 7 optimizer steps and a single epoch. It is not evidence of a quality improvement.

Per-example loss, step time and RSS are in
`experiments/training/cpu_lora_feasibility/training_log.jsonl`.

## 5. Adapter artifacts

The adapter is saved separately from the original model:

```
experiments/training/cpu_lora_feasibility/adapter/
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  README.md
```

The original checkpoint under `D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct` was not
modified.

## 6. Adapter load and inference verification

A fresh process loaded the base model plus `PeftModel.from_pretrained` from the saved adapter
directory. Adapter load succeeded (`adapter_loaded: true`). The same 5 frozen test prompts used for
the generative baseline were generated with the same settings: greedy, `max_new_tokens=256`,
`repetition_penalty=1.05`.

| Metric | Measured value |
|---|---:|
| Examples | 5 |
| Category tag rate | 0.0000 |
| Category accuracy | 0.0000 |
| Response validity rate | 0.0000 |
| Policy keyword coverage | 0.4000 |
| Latency p50 | 4465.225 ms |
| Latency p95 | 11346.900 ms |

Raw outputs: `experiments/training/cpu_lora_feasibility/adapter_inference.jsonl`.
Summary: `experiments/training/cpu_lora_feasibility/adapter_inference_summary.json`.

The adapter-loaded model still did not emit the `[category: ...]` prefix on these 5 prompts.
Outputs remained generic and sometimes incorrect. The feasibility run proves the training and
adapter-loading pipeline works; it does not prove that the adapter improves support quality.

## 7. Commands executed

```powershell
python -m src.model.train_lora --config configs/training_cpu_lora.yaml
python -m src.model.train_lora --config configs/training_cpu_lora.yaml --verify-only
```

## 8. Conclusion and next step

CPU LoRA training is feasible in this environment: 26 examples trained in 39.7 s, peak RSS about
1.46 GB, adapter saved and reloaded. No failure occurred, so no fallback was required.

For a meaningful adapter, the next training run should use substantially more data and more
epochs. That is Phase 2+ work and is intentionally not claimed here. CUDA remains unavailable and
was not retried, per instruction.

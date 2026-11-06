# Stage 1 — Model Research and Selection

**Date:** 2026-11-05
**Target hardware:** RTX 3050 Laptop, 4 GB VRAM, 16 GB RAM, Windows 11 (see `hardware_compatibility.md`)
**Task:** ShopEase customer support assistant (policy-grounded, short answers, refusal/clarification behavior)

All architecture, context, license and weight-size facts below were read from the official
Hugging Face model repository (`config.json`, `tokenizer_config.json`, model card) on 2026-11-05.
No model was downloaded in this phase.

---

## 1. Candidates

| # | Model identifier | Pinned revision (commit SHA) | License |
|---|---|---|---|
| A | `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 |
| B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `31b70e2e869a7173562077fd711b654946d38674` | Apache-2.0 |
| C | `microsoft/Phi-3.5-mini-instruct` | `2fe192450127e6a83f7441aef6e3ca586c338b77` | MIT |

Source URLs:
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- https://huggingface.co/microsoft/Phi-3.5-mini-instruct

---

## 2. Comparison table

| Criterion | Qwen2.5-1.5B-Instruct | SmolLM2-1.7B-Instruct | Phi-3.5-mini-instruct |
|---|---|---|---|
| Exact version | Qwen2.5, 1.5B instruct, Sep 2024 | SmolLM2, 1.7B instruct, Feb 2025 | Phi-3.5-mini, 3.8B instruct, Aug 2024 |
| Parameters | 1.54B total / 1.31B non-embedding | 1.7B | 3.82B |
| Architecture class | `Qwen2ForCausalLM` | `LlamaForCausalLM` | `Phi3ForCausalLM` |
| Layers / hidden size | 28 / 1536 | 24 / 2048 | 32 / 3072 |
| Attention | GQA: 12 query heads, 2 KV heads | MHA: 32 query heads, 32 KV heads | MHA: 32 query heads, 32 KV heads |
| Norm / activation / position | RMSNorm / SwiGLU / RoPE | RMSNorm / SwiGLU / RoPE | RMSNorm / SwiGLU / LongRoPE |
| Embeddings | Tied; vocab 151,936 | Tied; vocab 49,152 | Untied; vocab 32,064 |
| Context length | **32,768** (config; card describes the series as up to 128K) | **8,192** | **131,072** |
| bf16 weights on HF (measured from file listing) | **2.88 GB** | **3.19 GB** | **7.12 GB** |
| License | Apache-2.0 | Apache-2.0 | MIT |
| Chat template | ChatML (`<\|im_start\|>`, `<\|im_end\|>`) | ChatML (`<\|im_start\|>`, `<\|im_end\|>`) | `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>` |
| Tokenizer | `Qwen2Tokenizer`, pad `<\|endoftext\|>`, EOS `<\|im_end\|>` | `GPT2Tokenizer`, pad/EOS `<\|im_end\|>` | Phi-3 tokenizer (tiktoken-based) |
| Instruction following (vendor-reported, different suites — not directly comparable) | IFEval 47.4, MT-Bench 6.52, MMLU-Pro 24.2, GSM8K 42.8 (numbers as published in the SmolLM2 comparison table) | IFEval 56.7, MT-Bench 6.13, MMLU-Pro 19.3, GSM8K 48.2 | MMLU 69, MMLU-Pro 47.4, Arena-Hard 37, GSM8K 86.2, BBH-CoT 69 (Microsoft pipeline) |
| 4-bit NF4 weight footprint (estimate) | ~0.9–1.1 GB | ~1.0–1.2 GB | ~2.2–2.4 GB |
| QLoRA on 4 GB VRAM | ✅ Feasible (baseline plan) | ✅ Feasible but tighter | ❌ Not feasible |
| fp16/bf16 inference on 4 GB VRAM | ✅ Tight but workable | ✅ Tight | ❌ Does not fit |
| Quantization support | NF4, GPTQ, AWQ, GGUF ecosystem | NF4, GGUF | NF4, GGUF |
| PEFT/LoRA support | ✅ First-class in HF/PEFT/TRL | ✅ First-class in HF/PEFT/TRL | ✅ Supported, `trust_remote_code` recommended |
| `trust_remote_code` | Not required | Not required | Recommended by model card |
| Ecosystem maturity | Very large (thousands of adapters/quantizations) | Moderate | Large but smaller model-specific ecosystem |
| Key risk for this project | 151k-vocab output layer is relatively costly per generated token | Full MHA (32 KV heads) makes KV cache larger; 8K context | 7.12 GB weights; optimizer/activation overhead exceeds 4 GB |

> Benchmark numbers above are **vendor-reported**, not measured on this machine. They are included
> for orientation only. The project will produce its own baseline-vs-adapted evaluation.

---

## 3. Weighted decision matrix

Scores are engineering judgments (1 = poor, 5 = excellent) for **this specific task and hardware**,
not benchmark results. Weights reflect the assignment constraints: a 4 GB laptop GPU must actually
complete QLoRA training and local inference.

| Criterion | Weight | Qwen2.5-1.5B | SmolLM2-1.7B | Phi-3.5-mini |
|---|---|---|---|---|
| Hardware fit (4 GB VRAM) | 35% | 5 | 4 | 1 |
| Task suitability / instruction following | 25% | 4 | 4.5 | 4.5 |
| License | 10% | 5 | 5 | 5 |
| Ecosystem & PEFT tooling maturity | 15% | 5 | 4 | 3 |
| Memory efficiency (KV cache / weights) | 15% | 5 | 3 | 2 |
| **Weighted score** | 100% | **4.75** | **4.08** | **2.73** |

---

## 4. Selected model

**`Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`**

### Why this model

1. **Fits the actual 4 GB VRAM budget.** 2.88 GB of bf16 weights; 4-bit NF4 reduces the base model
   to roughly 0.9–1.1 GB, leaving room for LoRA adapters, activations and a KV cache during QLoRA.
2. **GQA with 2 KV heads.** Compared with SmolLM2's 32 KV heads, the KV cache is dramatically smaller.
   On a 4 GB card this directly improves the chance of completing training and keeping inference stable.
3. **Strong instruction-following and structured-output behavior.** The Qwen2.5 series explicitly
   targets instruction following, structured/JSON output, and system-prompt adherence — all useful for
   a support assistant that must follow a policy and emit a category label.
4. **32K context** is ample for a system prompt + policy excerpt + a support question, unlike the
   8K SmolLM2 limit, while avoiding Phi-3.5's unnecessary 128K context.
5. **Apache-2.0** allows commercial-style use and redistribution with minimal friction.
6. **Best-supported PEFT/TRL path** and no `trust_remote_code` requirement, reducing Windows-specific
   surprises.
7. Among the two feasible models, Qwen has the higher vendor-reported MT-Bench (6.52 vs 6.13) and
   MMLU-Pro (24.2 vs 19.3); SmolLM2 is better on IFEval (56.7 vs 47.4), which is a genuine trade-off
   and is recorded rather than hidden.

### Why not the others

- **SmolLM2-1.7B-Instruct** is a strong runner-up and the documented fallback. It is rejected as the
  primary choice because it is larger (3.19 GB bf16), uses full multi-head attention (32 KV heads),
  and has a shorter 8K context. On 4 GB VRAM that is a materially tighter QLoRA fit for a modest
  benchmark trade-off.
- **Phi-3.5-mini-instruct** is rejected primarily on **hardware feasibility**, not quality. Its
  7.12 GB bf16 weights and ~2.2–2.4 GB 4-bit footprint leave insufficient VRAM for optimizer states and
  activations on a 4 GB GPU. It also recommends `trust_remote_code`, defaults to flash attention, and
  its 128K context is irrelevant for this task.

### Conditions that would change the decision

- If QLoRA on Qwen OOMs even at sequence length 384 / micro-batch 1, switch to SmolLM2-1.7B (documented
  fallback).
- If `bitsandbytes` cannot run on this Windows setup, the fallback is a reduced-memory fp16 LoRA
  experiment on GPU; if that also OOMs, an explicitly labelled alternative adaptation experiment will
  be documented — it will not be silently skipped.
- If qualitative evaluation shows Qwen is systematically worse at policy refusal/clarification than
  SmolLM2 on the held-out test set, the selection will be revisited and the evidence recorded.

---

## 5. Planned adaptation settings for the selected model

| Setting | Value |
|---|---|
| Method | QLoRA (4-bit NF4, double quantization, bf16 compute) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Max sequence length | 512 |
| Micro-batch / gradient accumulation | 1 / 16 |
| Learning rate / schedule | 2e-4 / cosine, 3% warmup |
| Epochs | 3 (with validation-based checkpoint selection) |
| Precision | bf16 (Ampere compute capability 8.6) |
| Gradient checkpointing | Enabled |
| Optimizer | `paged_adamw_8bit` |
| Seed | 42 |

These are starting values; Phase 4 will record actual VRAM use, step time, and final loss.

---

## 6. Reproduction / verification commands

```powershell
# Metadata only; does not download weights
Invoke-RestMethod "https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct?blobs=true"
Invoke-RestMethod "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json"
Invoke-RestMethod "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer_config.json"
```

The selected model will be downloaded in Phase 3 to `D:\slm-models\hf` (outside OneDrive), not to `C:`.

---

## 7. Phase 3 addendum (2026-11-05) — approved model pivot

The Qwen2.5-1.5B checkpoint was never downloaded. With user approval the project switched to
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`. The candidate comparison above is retained as the
Phase 1 research record; it is not the current model decision.

Verified from the local checkpoint:

| Field | Measured value |
|---|---|
| Parameters | 134,515,008 |
| Weights file | `model.safetensors`, 269,060,552 bytes, 272 tensors |
| SHA-256 | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` |
| Config | 30 layers, hidden 576, 9 attention heads / 3 KV heads, context 8192 |
| Tokenizer | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt` all present |
| Chat template | ChatML (`<|im_start|>` / `<|im_end|>`) |

Static LoRA/QLoRA sizing for the current `training_config.yaml` targets (q, k, v, o, gate, up,
down; r=16): approximately 4.88M trainable parameters, about 3.6% of the 134.5M base model.
bf16 weights are ~269 MB; 4-bit NF4 weights are roughly ~67 MB. This is well inside 4 GB VRAM
in principle. Empirical VRAM/step-time measurement is still required and has not been run yet.

No training result is claimed in this report.

# Phase 2 — CPU Inference Optimization Report

Generated: 2026-11-05

Status: **Phase 2 complete. Phase 3 not started.**

## 1. Scope

This phase measures whether practical CPU-only inference changes reduce latency without
changing the model or the frozen evaluation data. It keeps model adaptation and inference
optimization separate:

- **A — pretrained baseline:** original full system prompt, original generation settings.
- **B — LoRA adapter:** the Phase 1 adapter, original full prompt and generation settings.
- **C — optimized inference:** pretrained model, compact prompt, capped generation,
  deterministic decoding, `model.eval()`, `torch.inference_mode()`, and one model load
  reused for all five requests.

Supplementary conditions are included to explain the result:

- **C1:** full prompt + 128-token cap (isolates the output cap).
- **C2:** compact prompt + 256-token cap (isolates the prompt change).
- **D:** dynamic int8 quantization investigation.

The five prompts are the same frozen prompts used in the existing generative baseline.
The frozen test set and the original baseline artifacts were read only.

## 2. Existing paths inspected

- `src/evaluation/transformer_baseline.py` — baseline generation, `model.eval()`,
  `torch.inference_mode()`, greedy decoding, two-step prompt tokenization.
- `configs/evaluation_config.yaml` — original `max_new_tokens=256`, `do_sample=false`,
  `temperature=1.0`, `top_p=1.0`, `repetition_penalty=1.05`, `warmup_runs=2`.
- `src/model/train_lora.py` — adapter reload path via `PeftModel.from_pretrained`.
- `experiments/baseline_generative/` — original five prompts, responses, generation config,
  latency artifact.
- `experiments/training/cpu_lora_feasibility/` — Phase 1 adapter and adapter-inference
  artifacts.

The original baseline artifact recorded:

| Field | Original artifact |
|---|---:|
| Model load | 7.916 s |
| Latency p50 | 4488.207 ms |
| Latency p95 | 6641.939 ms |
| Peak VRAM | 0 MB (CPU run) |
| Prompt count | 5 |

The controlled A rerun reproduced the same five responses and output token counts
(89, 134, 66, 53, 127), but its latency differs because the earlier artifact was produced
in a different process and machine state. The controlled rerun is used for A/B/C comparisons.

## 3. Exact configuration

Model:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct
dtype: float32
device: cpu
attn_implementation: sdpa
intra-op threads: 6
inter-op threads: 1
torch: 2.13.0+cpu
transformers: 5.16.1
peft: 0.21.2
CUDA available: false
```

Prompts:

```text
shopease-000374  expected clarification
shopease-000388  expected returns
shopease-000753  expected general_faq
shopease-000547  expected order_status
shopease-000499  expected order_status
```

Original generation settings:

```yaml
max_new_tokens: 256
do_sample: false
temperature: 1.0
top_p: 1.0
repetition_penalty: 1.05
num_beams: 1
use_cache: true
```

Optimized generation settings:

```yaml
max_new_tokens: 128
do_sample: false
temperature: 1.0
top_p: 1.0
repetition_penalty: 1.05
num_beams: 1
use_cache: true
```

Full system prompt: 131 tokens, loaded from `src/data/policy_data.py`.

Compact system prompt: 90 tokens, 41 tokens saved per request:

```text
You are the ShopEase Customer Support Assistant. Help only with ShopEase orders, shipping,
cancellations, returns, refunds, payments, product information, warranty, accounts, complaints,
and general FAQs. Use only ShopEase policy. Never invent order details, dates, amounts, or
policies. Ask one concise question if a required detail such as an order ID is missing.
Politely refuse unrelated requests. Be concise and professional.
```

The compact prompt keeps the categories and behavioral rules, but it is not identical in
wording to the original prompt. Its effect on answer quality is reported below.

Adapter path:

```text
experiments/training/cpu_lora_feasibility/adapter
```

## 4. Implemented optimizations

1. **Shorter system prompt:** 131 → 90 tokens; mean total prompt length 161.6 → 120.6
   tokens (41 tokens saved per request, −25.4%).
2. **Output cap:** 256 → 128 new tokens in C. The observed full-prompt outputs were 53–134
   tokens and compact-prompt outputs were 49–137 tokens, so 128 is tight: one response
   truncated.
3. **Deterministic evaluation:** `do_sample=false`, `num_beams=1`, fixed seed. This matches
   the existing baseline and avoids sampling variance.
4. **Inference guards:** `model.eval()` and `torch.inference_mode()` for all conditions.
5. **Single model load:** each condition loaded the model once and reused it for all five
   requests. No per-request reload.
6. **Tokenizer/loading path:** fast tokenizer, `padding_side="right"`, and a single
   `apply_chat_template(..., tokenize=True, return_tensors="pt")` call instead of a
   tokenize-then-encode round trip. Tokenization was about 1–2 ms per request.
7. **Dynamic quantization investigation:** built-in
   `torch.ao.quantization.quantize_dynamic(..., {torch.nn.Linear}, dtype=torch.qint8)`
   was tested on the CPU model. It ran, but it was not selected (see section 8).

## 5. Benchmark protocol

- Same five frozen prompts for every condition.
- Two warmup generations on the first two prompts, matching the original baseline protocol.
- Measured generation latency excludes model loading; total request latency adds tokenization.
- Peak process RSS was sampled with `psutil` and also reported via Windows `peak_wset`.
- `p95` is the 95th percentile of five samples. With five samples it is effectively the
  maximum, so it is not a robust tail estimate.
- A supplementary same-process paired check (10 measurements per condition, two passes,
  reversed order) controls for CPU state and model-load differences. It is not a quality
  evaluation.

## 6. Results

### 6.1 Main comparison

| Condition | Model | Prompt | Max new tokens | p50 ms | p95 ms | Load s | Peak RSS MB | Mean output tokens | Tag rate | Keyword coverage | Truncation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | pretrained | full | 256 | 5228.124 | 7983.631 | 2.655 | 976.6 | 93.8 | 0.0 | 0.6 | 0.0 |
| B | adapter | full | 256 | 4852.080 | 11727.133 | 2.858 | 996.8 | 85.2 | 0.0 | 0.4 | 0.0 |
| C | pretrained | compact | 128 | 3139.171 | 7163.054 | 2.608 | 972.0 | 72.2 | 0.0 | 0.2 | 0.2 |
| C1 | pretrained | full | 128 | 4306.087 | 6216.338 | 2.314 | 976.0 | 92.6 | 0.0 | 0.6 | 0.2 |
| C2 | pretrained | compact | 256 | 2713.167 | 6688.378 | 2.360 | 971.6 | 74.0 | 0.0 | 0.2 | 0.0 |
| D | pretrained int8 | compact | 128 | 1659.274 | 9924.347 | 4.374 | 1402.6 | 48.2 | 0.0 | 0.2 | 0.2 |

### 6.2 Before and after

| Measurement | Original baseline artifact | Controlled A rerun | C optimized |
|---|---:|---:|---:|
| Model load | 7.916 s | 2.655 s | 2.608 s |
| Generation p50 | 4488.207 ms | 5228.124 ms | 3139.171 ms |
| Generation p95 | 6641.939 ms | 7983.631 ms | 7163.054 ms |
| Total request p50 | not recorded | 5229.110 ms | 3140.112 ms |
| Peak process RSS | not recorded | 976.6 MB | 972.0 MB |
| Mean output tokens | 93.8 | 93.8 | 72.2 |
| Policy keyword coverage | 0.6 | 0.6 | 0.2 |

C versus controlled A: p50 −2088.953 ms (−39.96%), p95 −820.577 ms (−10.28%).
The standalone p50 delta is larger than the paired-check delta below. The paired check is
the more conservative latency estimate.

### 6.3 Controlled paired latency check

Base model loaded once; A/C/C1/C2 interleaved; two passes over the five prompts with
reversed order; 10 measurements per condition.

| Condition | p50 ms | p95 ms | Mean ms | Mean output tokens | Truncation |
|---|---:|---:|---:|---:|---:|
| A | 4343.785 | 6828.754 | 4647.302 | 93.8 | 0.0 |
| C | 2698.005 | 6367.203 | 3518.784 | 72.2 | 0.2 |
| C1 | 4425.172 | 6264.301 | 4568.720 | 92.6 | 0.2 |
| C2 | 2721.512 | 6575.830 | 3648.810 | 74.0 | 0.0 |

Paired median delta versus A:

- C: **−611.858 ms** (mean −1128.517 ms).
- C1: **−17.432 ms** (mean −78.582 ms).
- C2: **−437.469 ms** (mean −998.492 ms).

This shows that the compact prompt is the main source of the latency reduction. The
128-token cap alone (C1) did not produce a meaningful median latency gain.

### 6.4 Per-request generation latency (ms)

| Condition | 000374 | 000388 | 000753 | 000547 | 000499 |
|---|---:|---:|---:|---:|---:|
| A | 5228.124 | 7983.631 | 3899.719 | 3259.893 | 7473.668 |
| B | 11727.133 | 11706.100 | 4852.080 | 4470.726 | 1302.186 |
| C | 4253.281 | 7163.054 | 3106.447 | 2860.400 | 3139.171 |
| C1 | 4306.087 | 6216.338 | 3260.300 | 2634.269 | 6213.500 |
| C2 | 3716.900 | 6688.378 | 2713.200 | 2450.527 | 2589.200 |
| D | 5603.911 | 9924.347 | 1659.274 | 983.382 | 1029.571 |

The per-request spread is large. For example, C's 000388 response is 2.3× slower than its
000547 response. A single five-prompt pass is therefore indicative, not a precise SLA.

## 7. Loading, memory, and token counts

| Condition | Tokenizer load | Base load | Adapter apply | Quantize | Total load | RSS after load | Peak RSS | Process peak wset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.434 s | 1.221 s | 0.000 s | 0.000 s | 2.655 s | 942.5 MB | 976.6 MB | 1198.7 MB |
| B | 1.081 s | 1.148 s | 0.629 s | 0.000 s | 2.858 s | 965.2 MB | 996.8 MB | 1198.7 MB |
| C | 1.382 s | 1.226 s | 0.000 s | 0.000 s | 2.608 s | 942.2 MB | 972.0 MB | 1198.4 MB |
| C1 | 1.208 s | 1.106 s | 0.000 s | 0.000 s | 2.314 s | 942.7 MB | 976.0 MB | 1199.0 MB |
| C2 | 1.230 s | 1.130 s | 0.000 s | 0.000 s | 2.360 s | 943.0 MB | 971.6 MB | 1199.3 MB |
| D | 1.393 s | 1.209 s | 0.000 s | 1.772 s | 4.374 s | 1400.6 MB | 1402.6 MB | 1826.1 MB |

The original baseline's 7.916 s load was a different process/disk-cache state. The
controlled loads are 2.3–2.9 s. One load is reused for all five requests; reloading per
request would add roughly 2.3–2.9 s per request.

Generated token counts:

| Condition | Mean | Min | Max | Truncated |
|---|---:|---:|---:|---:|
| A | 93.8 | 53 | 134 | 0/5 |
| B | 85.2 | 14 | 152 | 0/5 |
| C | 72.2 | 49 | 128 | 1/5 |
| C1 | 92.6 | 53 | 128 | 1/5 |
| C2 | 74.0 | 49 | 137 | 0/5 |
| D | 48.2 | 11 | 128 | 1/5 |

C's truncation is `shopease-000388`: it hit the 128-token cap mid-sentence. C1 truncates the
same response. C2 avoids truncation because its cap is 256; its longest output was 137.

## 8. Dynamic quantization investigation

Status: **supported and executed, but not selected.**

- The built-in dynamic int8 quantization of `nn.Linear` layers ran successfully on the CPU
  model. No new dependency was needed.
- Quantization added 1.772 s to load; total load was 4.374 s versus 2.608 s for C.
- Peak process RSS was 1402.6 MB versus 972.0 MB for C, and process peak wset was 1826.1 MB
  versus 1198.4 MB. The quantization step temporarily holds the fp32 model and the
  quantized model, and the quantized build did not reduce measured process memory here.
- p50 was low (1659.274 ms) because several outputs were very short, but p95 was 9924.347 ms,
  worse than A. Manual inspection found irrelevant or garbled responses.
- `torch.ao.quantization` is deprecated in this PyTorch version. The measured result does
  not justify adding a quantization path.

D is therefore an investigation only, not a supported optimization.

## 9. Quality indicators

| Condition | Category tag rate | Category accuracy | Response validity | Keyword coverage | Clarification rate | Truncation |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0 | 0.0 | 0.0 | 0.6 | 0.0 | 0.0 |
| B | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |
| C | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.2 |
| C1 | 0.0 | 0.0 | 0.0 | 0.6 | 0.0 | 0.2 |
| C2 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 |
| D | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.2 |

Interpretation:

- Category tag rate and category accuracy are zero for every condition because no model
  emitted a valid `[category: ...]` tag. This is consistent with the earlier Phase 1/2
  baseline observations.
- Response validity is zero for the same reason; it is not a claim that every response was
  empty. The validity rule requires a recognized category tag.
- Unsupported refusal rate is **not applicable** here: the five prompts contain no
  `out_of_scope` example, so the summarizer returns 0.0 by construction.
- Policy keyword coverage is a crude lexical proxy. A response can contain the word
  "return" and be counted as covered while still inventing policy details. It is not a
  correctness score.
- C's coverage falls from 0.6 to 0.2. This is a regression on the automatic proxy, not a
  quality improvement.

## 10. Manual review

Single reviewer, not blinded. The purpose is to check whether latency changes came with
obvious quality regressions.

- **shopease-000374 (clarification):** A, B, C, C1, and C2 did not ask for the missing order ID.
  They treated the request as a financial-assistance or return question. B invented
  return/refund steps. No condition produced the intended clarification question.
- **shopease-000388 (returns):** A and B invented a 24–72 hour return window and phone/email
  details. C and C1 additionally truncated at 128 tokens; C's email contained
  `support@shopessay.com`. C2 was not truncated but still invented the return window.
- **shopease-000753 (general_faq):** A and B refused to merge coupons and referred the
  customer elsewhere. C and C2 instead said “Absolutely, we can help you merge your
  coupons” and invented checkout steps. This is a clear quality regression from the compact
  prompt. C1 behaved like A/B.
- **shopease-000547 (order_status):** A, B, C, C1, and C2 did not request an order ID. C told
  the customer to contact the official website.
- **shopease-000499 (order_status):** A, C, C1, and C2 fabricated order details such as
  shipment origin, dates, product, and delivery status. B was very short (14 tokens) and
  did not provide the requested status. This is the most serious failure and it is present
  in the original baseline too.

The manual review does not show a quality improvement in B or C. The adapter was trained on
only 26 examples, and the compact prompt changes model behavior in ways that are not
uniformly safe. Latency wins must not be presented as quality wins.

## 11. Interpretation

- **C reduces measured latency** relative to A. Standalone p50: 5228.124 → 3139.171 ms
  (−39.96%). Controlled paired median: −611.858 ms. Both point in the same direction, but
  the paired estimate is smaller and more credible.
- **The main lever is the compact prompt**, which reduces prompt tokens by 41 and changes
  output lengths. The output cap alone (C1) produced only a −17.432 ms paired median delta.
- **The 128-token cap is too tight** for the observed outputs. It truncates one of five
  responses. C2 shows the compact prompt can finish within 137 tokens, so a 160-token cap
  is the recommended Phase 3 test cap: it bounds the tail without truncating the observed
  responses.
- **B does not improve quality.** Its keyword coverage is 0.4 versus 0.6 for A, its p95 is
  11727.133 ms versus 7983.631 ms, and manual review shows the same fabrication problems.
- **D is not beneficial** in this setup: lower p50 but worse p95, much higher memory, and
  nonsense outputs.
- **No quality improvement is claimed.** The only defensible claim is that C is a
  latency-oriented configuration; it requires quality guardrails and a larger evaluation
  before use.

## 12. Selected configuration and recommendation

Selected for latency benchmarking only: **C**.

Recommended next-step configuration for Phase 3 evaluation, if optimization is retained:

```yaml
system_prompt: compact_90_token
max_new_tokens: 160      # avoid the observed 128-token truncation
do_sample: false
num_beams: 1
use_cache: true
model_load: once per process
eval_mode: true
inference_mode: true
```

Do not describe C, B, or D as a quality improvement. Do not deploy C without:
1. a larger frozen evaluation set,
2. human review of policy correctness and hallucination,
3. a stricter category-tag/response-format check,
4. a decision on whether changing the system prompt is acceptable.

If the system prompt must remain unchanged, C1 is the safer inference-only variant, but its
paired median latency delta is negligible (−17.432 ms) and it still truncates one response.
The 160-token cap is the recommended compromise for the next evaluation.

## 13. Artifacts

- `configs/optimization_config.yaml` — exact Phase 2 configuration.
- `src/optimization/benchmark_inference.py` — condition runner, paired check, assembler.
- `experiments/optimization/condition_A_pretrained_original.jsonl` and summary.
- `experiments/optimization/condition_B_adapter_original.jsonl` and summary.
- `experiments/optimization/condition_C_pretrained_optimized.jsonl` and summary.
- `experiments/optimization/condition_C1_full_prompt_capped.jsonl` and summary.
- `experiments/optimization/condition_C2_compact_prompt_uncapped.jsonl` and summary.
- `experiments/optimization/condition_D_dynamic_int8_investigation.jsonl` and summary.
- `experiments/optimization/benchmark_summary.json` — combined measurements and deltas.
- `experiments/optimization/comparison.md` — generated comparison tables.
- `experiments/optimization/paired_latency.json` and `.jsonl` — controlled paired check.
- `experiments/optimization/dynamic_quantization_investigation.json` — quantization probe.
- `reports/optimization_report.md` — this report.

The original `experiments/baseline_generative/`, `experiments/baseline/`,
`experiments/training/cpu_lora_feasibility/`, `data/test.jsonl`, and model files were not
modified.

## 14. Tests

Command run after Phase 2:

```powershell
python -m pytest tests/ -q
```

Result:

```text
12 passed in 0.18s
```

No Phase 3 evaluation has been started. Stopping here as requested.

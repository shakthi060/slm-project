# ShopEase Customer Support SLM Assistant

A task-specific customer support assistant built on a local Small Language Model (SLM).
It answers ShopEase customer questions from a fictional, documented policy set, refuses
out-of-scope requests, and asks for clarification when information is missing.

> **Project status: Phases 0-4, including optimization, final evaluation, and FastAPI deployment, complete.**
> The fictional ShopEase policy and an 804-example dataset were generated, cleaned, validated and
> split (654/75/75). The approved model is `HuggingFaceTB/SmolLM2-135M-Instruct`, downloaded
> locally and verified complete. Baseline, optimization, LoRA-feasibility, and final-evaluation
> experiments are complete. Phase 3 selected configuration A only as a least-risk demonstration
> fallback, **not** as a production-approved model. The FastAPI service loads that local model on
> startup and exposes `/`, `/health`, `/chat`, `/metrics`, `/docs`, and `/ui`. A post-evaluation
> reliability layer retrieves relevant policy sections and replaces repetitive/malformed output
> with a safe fallback. Actual deployment evidence is in `experiments/deployment/` and
> `reports/deployment_report.md`. Model quality limitations remain: manual correctness 6.7%, 0/7
> out-of-scope refusals, 0 category tags, 0 clarifying questions, and repetitive/hallucinated
> output. Both sampled API policy questions triggered the safe fallback instead of returning a
> useful policy answer. Results are only added after the corresponding phase actually executes.

---

## 1. Project overview

This project demonstrates the full SLM lifecycle on consumer hardware:

1. Hardware-aware model research and selection
2. Domain dataset creation, cleaning, validation and leakage-safe splitting
3. Baseline inference with the unmodified instruction-tuned model
4. Parameter-efficient adaptation with QLoRA
5. Inference optimization (4-bit quantization and generation settings)
6. Reproducible baseline-vs-adapted evaluation on a frozen held-out test set
7. Deployment as a local FastAPI service with a simple chat UI

The assistant is **not** a wrapper around a hosted LLM API. All inference runs locally on the
selected local model files; the deployed fallback uses the pretrained model without an adapter.

## 2. Problem statement

A fictional e-commerce company, **ShopEase**, receives high volumes of repetitive support
questions about orders, shipping, returns, refunds, payments, cancellations, products, warranty,
accounts and complaints. A general-purpose model is expensive, may be unavailable offline, and can
invent policies. The goal is a small, locally deployable model that:

- understands the customer question,
- identifies the relevant support category,
- answers using only ShopEase-approved knowledge,
- asks for missing details instead of guessing,
- politely declines unsupported or unrelated requests,
- responds consistently and professionally.

## 3. Target users

- ShopEase customers using a web/chat support widget
- ShopEase support agents who need a first-line draft answer
- The development team evaluating SLM feasibility on a 4 GB laptop GPU

## 4. Use case and success criteria

**Primary use case:** single-turn customer support question answering grounded in a ShopEase policy
document, with a predicted support category.

**Target success criteria** (to be measured on the held-out test set; none are claimed yet):

| Criterion | Target |
|---|---|
| Category accuracy | ≥ 0.85 |
| Response validity (non-empty, no prompt leakage, within length) | ≥ 0.95 |
| Unsupported-query refusal rate | ≥ 0.90 |
| Clarification behavior on missing-detail cases | ≥ 0.80 |
| Policy consistency (human rubric mean) | ≥ 4.0 / 5 |
| Hallucination-free rate (human rubric) | ≥ 90% |
| p95 inference latency (4-bit, GPU) | ≤ 3 s (to be measured) |

## 5. Model research and selection

Full evidence: [`reports/model_comparison.md`](reports/model_comparison.md).

**Originally selected model (Phase 1)**

| Field | Value |
|---|---|
| Model ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Parameters | 1.54B |
| Architecture | `Qwen2ForCausalLM`, 28 layers, GQA (12 Q / 2 KV heads) |
| Context | 32,768 tokens |
| License | Apache-2.0 |
| bf16 weights | 2.88 GB (measured from the HF file listing) |
| Chat template | ChatML (`<\|im_start\|>`, `<\|im_end\|>`) |

**Comparison summary**

| | Qwen2.5-1.5B-Instruct | SmolLM2-1.7B-Instruct | Phi-3.5-mini-instruct |
|---|---|---|---|
| Params | 1.54B | 1.7B | 3.82B |
| Context | 32,768 | 8,192 | 131,072 |
| License | Apache-2.0 | Apache-2.0 | MIT |
| bf16 weights | 2.88 GB | 3.19 GB | 7.12 GB |
| QLoRA on 4 GB VRAM | ✅ Feasible | ✅ Tight | ❌ Not feasible |
| Decision | **Selected** | Fallback | Rejected on hardware |

**Why Qwen2.5-1.5B-Instruct:** best combination of 4 GB VRAM fit, GQA memory efficiency, 32K
context, Apache-2.0 license, instruction-following quality and PEFT ecosystem support. SmolLM2 is
the documented fallback if QLoRA cannot fit; Phi-3.5-mini is not trainable on this GPU.

**Phase 3 model pivot (approved):** the Qwen checkpoint was not downloaded. With approval, the
project switched to `HuggingFaceTB/SmolLM2-135M-Instruct`, which is much smaller and easier to
train and deploy on the 4 GB laptop GPU. The tables above are retained as the Phase 1 research
record, not as the current model choice.

| Field | Actual value |
|---|---|
| Model ID | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| Local path | `D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct` |
| Parameters | 134,515,008 (134.5M) |
| Architecture | `LlamaForCausalLM`, 30 layers, hidden 576, GQA (9 Q / 3 KV heads) |
| Context | 8,192 tokens |
| License | Apache-2.0 |
| Weights | 269,060,552 bytes (bf16 safetensors), 272 tensors |
| Chat template | ChatML (`<\|im_start\|>`, `<\|im_end\|>`) |
| Verification | SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`; all required tokenizer files present |

Verification evidence: `experiments/baseline_generative/model_verification.json`.

## 6. Dataset details

**Generated and validated.** All examples are synthetic, derived from the fictional
`data/raw/shopease_policy.md`. No real customer data is used.

**Format** — JSONL, one chat example per line:

```json
{"id": "shopease-000007", "category": "returns", "intent": "wrong_item", "requires_clarification": false, "messages": [{"role": "system", "content": "<ShopEase system prompt>"}, {"role": "user", "content": "My order contains the wrong product."}, {"role": "assistant", "content": "[category: returns]\n<ShopEase-approved answer>"}], "source": "synthetic_template_v1"}
```

Split files also contain a `split` field. The assistant answer starts with
`[category: <label>]`, which lets the deployment layer extract the predicted support category.

**Categories:** `order_status`, `shipping`, `returns`, `refunds`, `payments`, `cancellations`,
`product_info`, `warranty`, `account`, `complaints`, `general_faq`, `clarification`, `out_of_scope`.

**Actual dataset:**

| Stage | Records |
|---|---:|
| Raw generated | 820 |
| Invalid records / duplicate IDs / exact duplicates | 0 / 0 / 0 |
| Near-duplicates removed (token Jaccard >= 0.90) | 16 |
| Clean records | 804 |
| Train / validation / test | 654 / 75 / 75 |
| Categories in every split | 13 |
| Clarification + out-of-scope share | 163 / 804 (20.3%) |

All leakage checks pass (no shared IDs, no exact user-text overlap, max cross-split Jaccard
0.8889). Full numbers are in `reports/dataset_report.md`; machine-readable outputs are
`reports/cleaning_stats.json`, `reports/split_stats.json` and `reports/dataset_validation.json`.

**Pipeline as executed:**

1. Generate fictional questions and policy-grounded answers deterministically with seed 42.
2. Validate records (roles, empty content, category, assistant prefix, placeholders).
3. Normalize user text and remove invalid records, duplicate IDs and exact duplicates.
4. Remove near-duplicates globally (before splitting) with token Jaccard >= 0.90.
5. Stratified split: 80% train / 10% validation / 10% test, seed 42.
6. Validate every file and run cross-split leakage checks.
7. Freeze the test set after baseline evaluation; never train or tune on it.

## 7. Installation

The project lives inside OneDrive, but **models, virtual environments and checkpoints must not**.
Only ~17 GB is free on `C:`; use `D:` (129 GB free).

```powershell
# From the project root
python -m venv D:\slm-venv
D:\slm-venv\Scripts\Activate.ps1
# If activation is blocked:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` pins the CUDA 13.0 PyTorch wheel (`torch==2.13.0+cu130`), verified available for
CPython 3.10 on Windows. The NVIDIA driver (592.82) supports CUDA 13.1, so this wheel is compatible.
A full CUDA toolkit / `nvcc` install is **not** required.

Set the model cache outside OneDrive:

```powershell
$env:HF_HOME = "D:\slm-models\hf"
# persist for future shells:
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\slm-models\hf", "User")
```

## 8. Dataset preparation commands

```powershell
python -m src.data.generate_dataset --seed 42
python -m src.data.clean_dataset --near-threshold 0.90
python -m src.data.split_dataset --seed 42
python -m src.data.validate_dataset
python -m pytest tests/test_dataset.py -q
```

Outputs: `data/raw/shopease_raw.jsonl`, `data/raw/shopease_policy.md`,
`data/processed/shopease_clean.jsonl`, `data/train.jsonl`, `data/validation.jsonl`,
`data/test.jsonl`, and the dataset reports under `reports/`.

## 9. Baseline inference commands

**Rule-based baseline (kept separate):**

```powershell
python -m src.evaluation.baseline --config configs/evaluation_config.yaml
```

Outputs: `experiments/baseline/` and `reports/baseline_report.md`. Measured on the full
75-example test set: category accuracy **0.6800**, response validity 1.0000, unsupported
refusal rate 1.0000, policy keyword coverage 0.7733, clarification rate 1.0000.

**Generative SLM baseline (real model inference):**

```powershell
python -m src.evaluation.transformer_baseline --config configs/evaluation_config.yaml `
  --model-path "D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct" `
  --model-id "HuggingFaceTB/SmolLM2-135M-Instruct" `
  --limit 5 --device cpu --no-4bit `
  --output-dir experiments/baseline_generative `
  --report-name generative_baseline_report.md
```

Outputs: `experiments/baseline_generative/prompts.jsonl`, `responses.jsonl`, `run_config.json`,
`latency_memory.json`, `summary.json`, `model_verification.json`, `manual_review.md`,
`observed_limitations.md`, and `reports/generative_baseline_report.md`.

Measured 5-prompt baseline (CPU, float32, greedy decoding, max_new_tokens 256): category tag
rate **0.0000**, category accuracy **0.0000**, response validity **0.0000**, policy keyword
coverage **0.6000**, model load **7.916 s**, latency p50 **4488.207 ms**, p95 **6641.939 ms**.
The unmodified model did not emit the `[category: ...]` prefix and manually reviewed outputs
included invented order facts and policy details. These are baseline observations, not
fine-tuning results. The CPU run is a functional fallback; a GPU run requires the CUDA PyTorch
wheel, whose download stalled repeatedly in this session.

## 10. Fine-tuning commands

**Phase 1 CPU LoRA feasibility (executed):**

```powershell
python -m src.model.train_lora --config configs/training_cpu_lora.yaml
python -m src.model.train_lora --config configs/training_cpu_lora.yaml --verify-only
```

Outputs: `experiments/training/cpu_lora_feasibility/adapter/`, `environment_check.json`,
`selected_examples.json`, `training_config.json`, `training_log.jsonl`, `summary.json`,
`training_summary.md`, `adapter_inference.jsonl`, `adapter_inference_summary.json`.
Measured: 26 examples, 1 epoch, LoRA r=8, 2,442,240 trainable params, 39.745 s training,
1.457 GB peak process RSS, adapter saved and reloaded. This is a feasibility check, not a
quality fine-tune.

**Full QLoRA training:** not run; CUDA PyTorch is unavailable. The planned config remains
`configs/training_config.yaml`.

## 11. Optimization commands

Phase 2 CPU optimization experiments completed. The implementation and measurements are in
`src/optimization/benchmark_inference.py`, `experiments/optimization/`, and
`reports/optimization_report.md`.

## 12. Evaluation commands

Phase 3 final evaluation completed. The implementation and results are in
`src/evaluation/final_evaluation.py`, `experiments/evaluation/final_results.json`, and
`reports/final_evaluation.md`.

```powershell
python -m src.evaluation.final_evaluation --assemble
```

## 13. API deployment commands

Phase 4 FastAPI deployment completed. The service reuses the existing model-loading and
generation functions from `src/evaluation/transformer_baseline.py`.

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
# Chat UI:   http://127.0.0.1:8000/ui
```

API configuration: `configs/api_config.yaml`. The service loads configuration A
(`HuggingFaceTB/SmolLM2-135M-Instruct`, pretrained, CPU, 256-token cap) once at startup and
reuses it for every request.

The request pipeline now classifies the message, retrieves only the relevant ShopEase policy
sections (`src/api/policy_context.py`), and passes that focused context to the model. Narrow
API-layer overrides fix known classifier false positives for product stock, price matching, and
account phone-number updates without changing the shared evaluation classifier. Model output
is checked for empty content, prompt leakage, malformed text, repeated list items, repeated
sentences, repeated phrases, and low diversity (`src/api/output_guard.py`). Unsafe or repetitive
output is replaced with a safe fallback, and the response exposes `fallback_used` and
`fallback_reason`. Out-of-scope and missing-context requests are refused without model inference.

Endpoints: `GET /`, `GET /health`, `POST /chat`, `GET /metrics`, `GET /ui`, `GET /docs`.

## 14. Example API request and response

Example request:

```http
POST /chat
Content-Type: application/json

{"message": "How can I return a damaged product?"}
```

Illustrative response shape:

```json
{
  "answer": "Please share your order ID and photos of the damaged item so we can help.",
  "category": "returns",
  "status": "success",
  "latency_ms": 2500,
  "fallback_used": false,
  "fallback_reason": null
}
```

This shape is illustrative. In the latest real HTTP verification, both sampled policy questions
triggered the output guard and returned the safe fallback instead of repetitive text. No useful
policy answer was produced for those samples. See `experiments/deployment/api_verification.md` and
`reports/deployment_report.md`.

## 15. Test commands

```powershell
python -m pytest tests/ -q                              # all tests, including real local-model integration
python -m pytest tests/test_api.py tests/test_model_service.py tests/test_policy_context.py tests/test_output_guard.py -q  # non-integration tests
python scripts/verify_api.py                            # starts uvicorn and makes real HTTP requests
```

Actual full-suite result: **59 passed**, one Starlette/httpx deprecation warning, in **45.26 s**.
Non-integration tests: **38 passed** in **6.53 s**. The real integration test loads the local model,
performs two real inferences, and is skipped only when the model directory is missing.

## 16. Hardware requirements

**Measured development machine** (see [`reports/hardware_compatibility.md`](reports/hardware_compatibility.md)):

| Component | Value |
|---|---|
| OS | Windows 11 Home Single Language, build 26300 |
| CPU | AMD Ryzen 5 5600H, 6C / 12T, 3.3 GHz |
| RAM | 16 GB DDR4-3200 (15.34 GB usable) |
| GPU | NVIDIA RTX 3050 Laptop, **4 GB VRAM**, driver 592.82 |
| Python | 3.10.9 |
| Disk | C: 17.3 GB free; D: 129.8 GB free |

**Feasible:** 135M model in bf16/4-bit, LoRA and QLoRA (r=16, seq=512, micro-batch 1) with large
headroom; 1.5B–1.7B QLoRA was the original plan before the approved pivot.
**Not feasible:** full fine-tuning of larger models, fp16 full fine-tuning, Phi-3.5-mini QLoRA.

Measured on 2026-11-05: the installed `torch 2.13.0+cpu` ran the 135M model on CPU (p50 4488 ms
on the 5-prompt baseline). The CUDA 13.0 wheel is 1.9 GB; its background download stalled with zero
bytes read and was stopped. GPU latency and VRAM numbers will replace the CPU numbers only after
that install and a GPU run complete.

**FastAPI deployment measurement (2026-11-06):** in the latest real HTTP verification, local model
load took 10.745 s and the two verified chat requests took 15.409 s and 11.026 s end-to-end. These
are CPU-only measurements from one machine and two samples; they are not production latency
guarantees. Phase 3 p95 was 14.7 s on its evaluation workload.

## 17. Limitations

- The deployment is a demonstration fallback and is **not production-approved**.
- The raw model still produces repetitive text and can hallucinate policy details despite policy
  retrieval and the grounding prompt. The output guard now blocks that text, but the result is a
  safe fallback rather than a correct policy answer.
- Both sampled API policy questions used the safe fallback. Safety improved; answer usefulness did
  not.
- Phase 3 measured 6.7% manual correctness, 0/7 out-of-scope refusals, 0 category tags, and 0
  clarifying questions.
- The API is single-turn; the UI displays history, but each request sends only the latest message.
- Category labels use the existing rule-based fallback unless the model emits a valid tag.
- Metrics are in-memory, cover `/chat` only, and reset when the process restarts.
- No authentication, rate limiting, persistent database, live order lookup, streaming, or production
  monitoring is included.
- 4 GB VRAM forces 4-bit QLoRA and short sequences; long multi-turn conversations are out of scope.
- The dataset is synthetic and small; it cannot cover every real customer phrasing.
- Automatic metrics are proxies and can be gamed by keyword matching; a documented human rubric is
  also used, and its sample size is limited.
- Vendor benchmark numbers in the comparison report were not reproduced locally.
- The assistant only knows the fictional ShopEase policy document. It must not be treated as a
  source of real company or order information.
- Windows + `bitsandbytes` support remains unverified because CUDA is unavailable; CPU float32 is
  the measured deployment path.

## 18. Project architecture

```
customer-support-slm/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/
│   ├── api_config.yaml                [Phase 4 deployment]
│   ├── model_config.yaml              [done]
│   ├── training_config.yaml           [done]
│   ├── optimization_config.yaml       [done]
│   ├── final_evaluation_config.yaml   [done]
│   └── evaluation_config.yaml         [done]
├── data/
│   ├── raw/                           [done]
│   ├── processed/                     [done]
│   ├── train.jsonl                    [done]
│   ├── validation.jsonl               [done]
│   └── test.jsonl                     [frozen]
├── src/
│   ├── data/                          [done]
│   ├── model/                         [LoRA feasibility done]
│   ├── evaluation/                    [final evaluation done]
│   ├── optimization/                  [done]
│   └── api/                           [Phase 4 deployment]
│       ├── main.py
│       ├── model_service.py
│       ├── policy_context.py           [policy retrieval]
│       ├── output_guard.py             [repetition/fallback guard]
│       ├── schemas.py
│       ├── metrics.py
│       ├── config.py
│       └── static/index.html
├── scripts/
│   └── verify_api.py                  [Phase 4 evidence]
├── tests/
│   ├── test_api.py                    [mocked API tests]
│   ├── test_model_service.py          [model-service tests]
│   ├── test_policy_context.py         [policy retrieval tests]
│   ├── test_output_guard.py           [repetition/fallback tests]
│   ├── test_api_integration.py        [real local-model test]
│   └── test_dataset.py
├── experiments/
│   ├── baseline/                      [done]
│   ├── baseline_generative/           [done]
│   ├── training/                      [LoRA feasibility done]
│   ├── optimization/                  [done]
│   ├── evaluation/                    [done]
│   └── deployment/                    [Phase 4 evidence]
├── reports/
│   ├── hardware_compatibility.md      [done]
│   ├── model_comparison.md            [done]
│   ├── dataset_report.md              [done]
│   ├── baseline_report.md             [done]
│   ├── training_report.md             [done]
│   ├── optimization_report.md         [done]
│   ├── final_evaluation.md            [done]
│   └── deployment_report.md           [Phase 4]
└── src/api/static/                    [chat UI]
```

## 19. Reproducibility

- Global seed: **42** (dataset generation, splitting, training).
- Model pinned by commit SHA, not just by tag.
- Test set frozen after baseline; no training or hyperparameter tuning on it.
- Every experiment writes its configuration, seed, package versions and raw outputs.
- Artifacts are written outside OneDrive where possible; large weights and adapters are gitignored.

## 20. Phase status and evidence

| Phase | Status | Evidence |
|---|---|---|
| 0 — Environment inspection | ✅ Complete | `reports/hardware_compatibility.md` |
| 1 — Setup + model research | ✅ Complete | this README, `reports/model_comparison.md`, `configs/*.yaml` |
| 2 — Dataset preparation | ✅ Complete | 820 raw → 804 clean; 654/75/75 split; 13 categories; 20.3% clarification/out-of-scope; leakage PASS; 12 tests pass |
| 3 — Baseline inference | ✅ Complete (5-prompt generative + full rule baseline) | `reports/generative_baseline_report.md`, `experiments/baseline_generative/`, rule baseline in `experiments/baseline/` |
| 4 — Adaptation | 🟡 CPU LoRA feasibility complete; full QLoRA pending | `reports/training_report.md`, `experiments/training/cpu_lora_feasibility/` |
| 5 — Optimization | ✅ Complete | `reports/optimization_report.md`, `experiments/optimization/` |
| 6 — Evaluation | ✅ Complete; configuration A selected as demo fallback only | `reports/final_evaluation.md`, `experiments/evaluation/` |
| 7 — Deployment + tests | ✅ Complete (FastAPI demo, not production-approved; policy retrieval + safe fallback added) | `reports/deployment_report.md`, `experiments/deployment/`, 59 tests pass |
| 8 — Final audit | ⏳ Pending | Intentionally deferred |

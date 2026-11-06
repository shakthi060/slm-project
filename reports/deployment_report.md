# Phase 4 Deployment Report

Generated: 2026-11-06

## Status

FastAPI deployment is implemented and verified with the real local SmolLM2 model.
A post-evaluation reliability layer now retrieves relevant ShopEase policy sections,
detects repetitive or malformed output, and returns a safe fallback instead of
repetitive text.

The service is still a **demonstration fallback**, not a production-approved system.
The model's answer quality remains poor, and the Phase 3 limitations still apply.

## Selected configuration

| Field | Value |
|---|---|
| Configuration | A |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Adapter | None (pretrained model) |
| Prompt | ShopEase system prompt, safeguards, and retrieved policy sections |
| Generation | Greedy decoding, `max_new_tokens=256` |
| Device | CPU, float32 |
| Local model path | `D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct` |

Configuration A was selected in Phase 3 only as the least-risk demo fallback. It is
not production-approved: manual correctness was 6.7%, response validity was 0.9067,
p95 latency was 14.7 s, truncation was 9.3%, out-of-scope refusals were 0/7, category
tags were 0, and clarifying questions were 0.

## Architecture

```
Browser chat UI (/ui)
        |
        v
FastAPI routes (src/api/main.py)
        |
        v
ModelService (src/api/model_service.py)
        |
        +--> PolicyRetriever (src/api/policy_context.py)
        |      - classifies the request
        |      - retrieves only relevant policy sections
        |
        +--> OutputGuard (src/api/output_guard.py)
        |      - detects repeated lines, sentences, phrases, and low diversity
        |      - returns a safe fallback instead of unsafe output
        |
        v
Existing inference helpers (src/evaluation/transformer_baseline.py)
  - load_model()
  - generate()
        |
        v
Local SmolLM2-135M-Instruct files on CPU
```

The API does not duplicate model-loading logic. It reuses the evaluated
`load_model()` and `generate()` functions, loads the model once during application
startup, keeps it in memory, and serializes inference with a lock. The original
evaluation, training, optimization, dataset, and final-results files were not
modified.

## Implemented endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Application name, description, version, status, docs and UI links |
| GET | `/health` | Service status, model-loaded flag, model ID, device, load time, load error |
| POST | `/chat` | Validated customer message and real local-model response |
| GET | `/metrics` | In-memory chat request counts and latency summary |
| GET | `/ui` | Simple customer-support chat interface |
| GET | `/docs` | Swagger / OpenAPI documentation |

## Safeguards and validation

**Policy-grounded retrieval**

`PolicyRetriever` parses the confirmed ShopEase policy sections and maps the request
category to the relevant sections. Every prompt includes the `Company` section, then
only sections such as `Returns`, `Refunds`, or `Shipping and delivery`. This reduces
the prompt and exposes the confirmed rules, including the 7-calendar-day standard
return window, the 10-calendar-day electronics window, and the 48-hour damaged-item
reporting rule.

Small API-layer overrides prevent known out-of-scope keyword collisions for product
stock, price matching, and account phone-number updates. The shared evaluation
classifier is unchanged; the overrides are limited to `src/api/policy_context.py`.

**Prompt-level grounding**

The system prompt tells the model to answer only from the supplied ShopEase policy,
not invent delivery dates, return periods, refund timelines, coupon rules, order
statuses, account facts, or contact information, say when information cannot be
verified, ask one clarification question when details are missing, politely refuse
unrelated requests, never request OTP/PIN/password/card details, not claim access to
live order data, and avoid repeated points or list items.

These are **application-level mitigations only**. The verification below shows that
the model still generated repetitive text; retrieval and prompting did not make the
model reliable.

**Request, routing, and output handling**

- Empty and whitespace-only messages are rejected with HTTP 422.
- Messages over 1,000 characters are rejected with HTTP 422.
- Malformed JSON is rejected with HTTP 422.
- Out-of-scope requests are refused deterministically before model inference.
- If no relevant policy context is found, the service returns a safe fallback without
  calling the model.
- Model output is stripped, category tags are removed from the answer text, empty
  output is rejected, prompt leakage is rejected, and output is capped at 4,000
  characters.
- Repetitive output is detected using repeated list items, repeated sentences,
  repeated trigrams, and low diversity checks.
- Detected unsafe output is replaced with:
  `"I could not produce a reliable answer from the ShopEase policy. Please rephrase your question or contact ShopEase support for help."`
- The response includes `fallback_used` and `fallback_reason` so the behavior is
  observable rather than hidden.
- Model-loading failure leaves the service running in a degraded state; `/chat`
  returns HTTP 503.
- Inference failure returns HTTP 502.
- Category is taken from a model tag when present; otherwise the existing rule-based
  fallback classifier is used. The selected model emitted no tags in Phase 3.
- The API does not retrieve live order records and does not claim to do so.

## Actual test results

Command:

```powershell
python -m pytest tests/ -q
```

Actual result: **59 passed**, one Starlette/httpx deprecation warning, in **45.26 s**.
The full output is saved in `experiments/deployment/pytest_full.txt`.

Mocked API, model-service, policy-context, and output-guard tests alone:

```text
38 passed, 1 warning in 6.53s
```

Real local-model integration test: **1 passed** in **39.18 s**. It loaded the actual
local model, called `/health`, performed two real `/chat` inferences, and checked
`/metrics`.

## Actual HTTP verification

Evidence: `experiments/deployment/api_verification.json` and
`experiments/deployment/api_verification.md`.

The verification script started `uvicorn` as a separate process and used real HTTP
requests against `http://127.0.0.1:8000`.

| Method | Path | Status | Elapsed ms |
|---|---|---:|---:|
| GET | `/` | 200 | 3.480 |
| GET | `/health` | 200 | 2.181 |
| GET | `/docs` | 200 | 1.496 |
| GET | `/ui` | 200 | 59.683 |
| GET | `/metrics` | 200 | 2.273 |
| POST | `/chat` (damaged return) | 200 | 15408.909 |
| POST | `/chat` (standard return) | 200 | 11026.154 |
| POST | `/chat` (empty) | 422 | 1.588 |
| POST | `/chat` (whitespace) | 422 | 1.347 |
| POST | `/chat` (1,001 chars) | 422 | 1.217 |
| POST | `/chat` (invalid JSON) | 422 | 1.284 |
| GET | `/metrics` | 200 | 1.744 |

Startup health:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "device": "cpu",
  "model_load_seconds": 10.745,
  "model_error": null,
  "uptime_seconds": 15.329
}
```

Final metrics after two successful chats and four rejected requests:

```json
{
  "total_requests": 6,
  "successful_requests": 2,
  "failed_requests": 4,
  "average_latency_ms": 13210.735,
  "p50_latency_ms": 11019.962,
  "p95_latency_ms": 15401.508,
  "latency_samples": 2
}
```

These two latency samples are not a stable latency estimate.

## Actual model inference results

### Damaged return

Request:

```json
{"message": "How can I return a damaged product?"}
```

Actual response:

```json
{
  "answer": "I could not produce a reliable answer from the ShopEase policy. Please rephrase your question or contact ShopEase support for help.",
  "category": "returns",
  "status": "success",
  "latency_ms": 15401.508,
  "fallback_used": true,
  "fallback_reason": "repetitive_output:repeated_list_items,repeated_sentences,repeated_phrases"
}
```

### Standard return window

Request:

```json
{"message": "What is the return window for a standard product?"}
```

Actual response:

```json
{
  "answer": "I could not produce a reliable answer from the ShopEase policy. Please rephrase your question or contact ShopEase support for help.",
  "category": "returns",
  "status": "success",
  "latency_ms": 11019.962,
  "fallback_used": true,
  "fallback_reason": "repetitive_output:repeated_phrases"
}
```

Before the output guard was added, the damaged-return request returned 20 repeated
list items and did not provide the 7-day return window or the 48-hour reporting rule.
The guard now detects that pattern and replaces it with a safe fallback. This improves
safety and reliability, but it does **not** make the model's answers useful or correct.

The `returns` category came from the rule-based fallback classifier, not from a model
category tag.

## Known limitations

- The service is not production-approved; the model quality limitations from Phase 3
  remain.
- Both verified policy questions used the safe fallback. The API avoids repetitive
  output, but it does not give the user the correct policy answer in these samples.
- The model remains prone to repetition and hallucination despite policy retrieval and
  prompt safeguards.
- Real CPU inference took about 11.0 s and 15.4 s for the two verified requests; model
  load took about 10.7 s in the HTTP verification run. These are single-run measurements
  on one machine.
- The API is single-turn; the UI shows conversation history, but each `/chat` request
  sends only the latest message and no prior turns.
- Category labels are rule-based fallbacks unless the model emits a valid tag.
- Metrics are in-memory, cover `/chat` only, and reset when the process restarts.
- There is no authentication, rate limiting, persistent storage, live order lookup,
  streaming, or production monitoring.
- `/health` returns HTTP 200 with `"status": "degraded"` when the model is unavailable;
  `/chat` returns HTTP 503 in that state.
- Prompt instructions and output filtering are not proof of factual reliability and do
  not eliminate hallucination.

## Requirement checklist

| Requirement | Status | Evidence |
|---|---|---|
| Inspect existing project and reuse inference/model-loading code | Complete | `src/api/model_service.py` imports `load_model()` and `generate()` |
| GET `/` | Complete | HTTP 200, metadata returned |
| GET `/health` | Complete | HTTP 200, model loaded true |
| POST `/chat` with real local model | Complete | HTTP 200, real model responses captured |
| GET `/metrics` | Complete | HTTP 200, counts and latency summary |
| Pydantic request/response models | Complete | `src/api/schemas.py` |
| Load model once and reuse it | Complete | Startup load; repeated-request unit test; single loaded service |
| `torch.inference_mode()` and `model.eval()` | Complete | Reused existing `generate()` and `load_model()` |
| Safe model-load and inference errors | Complete | Degraded health, HTTP 503/502 tests |
| Empty-message rejection | Complete | HTTP 422 verified |
| Whitespace-only rejection | Complete | HTTP 422 verified |
| Maximum message length | Complete | 1,000-character limit; HTTP 422 verified |
| Invalid request/JSON handling | Complete | HTTP 422 verified |
| Safe output handling | Complete | Strip, tag removal, empty/leak/repetition rejection, length cap |
| Policy-grounded retrieval | Complete | `src/api/policy_context.py`; 5 policy-context tests |
| Repetition detection and safe fallback | Complete | `src/api/output_guard.py`; 4 output-guard tests; HTTP evidence |
| Deterministic out-of-scope refusal | Complete | Policy/classifier guard before model inference |
| Document safeguards as mitigations, not proof | Complete | This report and README |
| Chat interface | Complete | `/ui`, responsive HTML/CSS/JS |
| Mocked API tests | Complete | 38 non-integration tests passed |
| Real integration test | Complete | Passed with local model; two chats |
| API testing evidence | Complete | `experiments/deployment/` |
| README update | Complete | README deployment and reliability sections updated |
| Deployment report | Complete | This file |
| Final verification with actual HTTP requests | Complete | Verification table above |
| Production approval | Incomplete / not claimed | Phase 3 quality metrics and verified fallback behavior |
| Useful policy answer on sampled requests | Incomplete | Both sampled chats used the safe fallback |
| Reliable model-generated refusal/clarification/category behavior | Incomplete | 0/7 refusals, 0 tags, 0 clarifying questions in Phase 3 |
| Final documentation cleanup / audit | Not started | Deferred until deployment results are reviewed |

## Evidence files

- `experiments/deployment/api_verification.json` - raw HTTP evidence
- `experiments/deployment/api_verification.md` - readable HTTP summary
- `experiments/deployment/pytest_full.txt` - captured full test output
- `experiments/deployment/server.log` - uvicorn server log
- `tests/test_policy_context.py` - policy retrieval tests
- `tests/test_output_guard.py` - repetition/malformed-output tests
- `reports/final_evaluation.md` - Phase 3 quality evidence
- `reports/deployment_report.md` - this report

## How to run

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: `http://127.0.0.1:8000/docs`

Chat UI: `http://127.0.0.1:8000/ui`

Tests:

```powershell
python -m pytest tests/ -q
python scripts/verify_api.py
```


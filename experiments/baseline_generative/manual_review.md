# Manual review of the 5-prompt generative baseline

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
Run: 2026-11-05, device `cpu`, dtype `float32`, no quantization.
Raw outputs: `responses.jsonl`.

| ID | Expected | Emitted category tag | Observed issue |
|---|---|---|---|
| shopease-000374 | clarification | no | Generic refusal; did not ask the clarifying question. |
| shopease-000388 | returns | no | Invented a 24-72 hour return window, phone number and email. |
| shopease-000753 | general_faq | no | Invented a "local store" and "shop staff" channel. |
| shopease-000547 | order_status | no | Correctly denied order access, but did not ask for the order ID. |
| shopease-000499 | order_status | no | Fabricated product, origin, ship date and delivery date. |

Measured run summary: category tag rate 0/5, category accuracy 0/5, policy keyword
coverage 3/5, latency p50 4488.207 ms, latency p95 6641.939 ms, model load 7.916 s.

These observations are baseline behaviour of the unmodified pretrained model. They are
not training results and must not be presented as fine-tuned-model quality.

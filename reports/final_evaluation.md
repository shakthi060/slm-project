# Phase 3 Final Evaluation

Generated: 2026-11-06T04:46:08.717400+00:00
Test set: `data\test.jsonl` (75 held-out examples)

## Automatic test-set metrics

| Config | Model | Prompt | Max new | Valid | Tag acc. | Fallback acc. | Combined acc. | Keyword coverage | Heuristic halluc. | p50 ms | p95 ms | Peak RSS MB | Trunc. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_pretrained_original | pretrained | full_system_prompt | 256 | 0.9067 | 0.0 | 0.4133 | 0.4133 | 0.4 | 0.2 | 4825.204 | 14721.569 | 991.4 | 0.0933 |
| B_lora_adapter_original | adapter | full_system_prompt | 256 | 0.9333 | 0.0 | 0.4267 | 0.4267 | 0.3867 | 0.1467 | 4599.487 | 19533.98 | 1005.0 | 0.0667 |
| C_compact_optimized | pretrained | compact_system_prompt | 128 | 0.72 | 0.0 | 0.4133 | 0.4133 | 0.4533 | 0.2133 | 4103.162 | 7841.853 | 974.2 | 0.28 |
| D_full_prompt_160 | pretrained | full_system_prompt | 160 | 0.7867 | 0.0 | 0.44 | 0.44 | 0.4 | 0.1867 | 4852.832 | 9608.032 | 979.9 | 0.2133 |

## Manual review

| Config | Responses | Relevance | Correctness | Policy consistency | Hallucination | Mean rubric score |
|---|---:|---:|---:|---:|---:|---:|
| A | 15 | 0.8 | 0.0667 | 0.2667 | 0.7333 | 0.3778 |
| B | 15 | 0.7333 | 0.0667 | 0.3333 | 0.6667 | 0.3778 |
| C | 15 | 0.7333 | 0.1333 | 0.4 | 0.6 | 0.4222 |
| D | 15 | 0.8 | 0.0667 | 0.2667 | 0.7333 | 0.3778 |

The manual sample was not blinded across configurations and was scored by one reviewer.

## Failure analysis

### Out-of-scope, clarification, and truncation

| Config | Out-of-scope refusal | Clarification question | Truncation |
|---|---:|---:|---:|
| A | 0.0000 | 0.0000 | 0.0933 |
| B | 0.0000 | 0.0000 | 0.0667 |
| C | 0.0000 | 0.0000 | 0.2800 |
| D | 0.0000 | 0.0000 | 0.2133 |

### Observed failure patterns

- All four configurations answered all 7 out-of-scope requests instead of refusing; out-of-scope refusal rates were {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}.
- No configuration emitted a category tag; tag accuracy was 0.0 for all four, so category results rely entirely on the fallback classifier.
- No configuration asked a clarifying question on any of the 8 clarification examples; clarification-question rates were {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}.
- All configurations produced unsupported or fabricated claims; manual hallucination rates were A 0.7333, B 0.6667, C 0.6, D 0.7333.
- Truncation at the token cap was C 0.28, D 0.2133, A 0.0933, B 0.0667.
- B had the best validity (0.9333) and lowest truncation (0.0667), but the worst p95 latency (19533.98 ms); C was fastest (p50 4103.162 ms, p95 7841.853 ms) but truncated 0.28 of responses.
- Manual examples confirm: A/D claimed a US shipment and a 31 December 2023 delivery for order SE-76185042; C said coupons can be merged; A/B/C/D denied express delivery; all four answered a legal contract question as if in scope.
- Manual correctness was low for every configuration: A 0.0667, B 0.0667, C 0.1333, D 0.0667.

### Manual correctness by category

| Category | A | B | C | D |
|---|---:|---:|---:|---:|
| account | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cancellations | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clarification | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| complaints | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| general_faq | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| order_status | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| out_of_scope | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| payments | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| product_info | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| refunds | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| returns | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| shipping | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| warranty | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

Rates are based on the 15-example manual sample; categories with one sampled response are especially noisy.

### Root causes

- The 135M-parameter base model is too small for reliable policy-grounded customer support.
- The Phase 1 LoRA adapter was trained on only 26 examples for one epoch, so it cannot reliably change policy behavior.
- The full policy prompt is long, but the model still ignores key constraints, especially refusals and clarification.
- The model was not trained to emit category tags, and no constrained decoding enforces the allowed categories.
- Deterministic decoding with a 256-token cap produces long, repetitive, and sometimes truncated answers.
- There is no retrieval, rule-based policy guard, or post-generation validator.

### Required mitigations

- Train or fine-tune a larger instruction model on substantially more policy-grounded support examples.
- Add explicit refusal, clarification, and category-tag training examples and evaluate them separately.
- Add retrieval or structured policy lookup so answers are grounded in the relevant policy section.
- Add post-generation guardrails for banned policy claims, fabricated contact details, OTP requests, and out-of-scope answers.
- Choose the token cap from measured truncation and quality, not only latency.
- Run a larger blinded human evaluation with multiple reviewers and production-like traffic before any deployment.

## Deltas versus A (pretrained, full prompt, 256 tokens)

| Comparison | p50 delta ms | p50 change | p95 delta ms | Peak RSS delta MB | Keyword coverage delta | Heuristic hallucination delta | Combined category accuracy delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_vs_A | -225.717 | -4.68% | 4812.411 | 13.6 | -0.013 | -0.053 | 0.013 |
| C_vs_A | -722.042 | -14.96% | -6879.716 | -17.2 | 0.053 | 0.013 | 0.0 |
| D_vs_A | 27.628 | 0.57% | -5113.537 | -11.5 | 0.0 | -0.013 | 0.027 |
| C_vs_B | -496.325 | -10.79% | -11692.127 | -30.8 | 0.067 | 0.067 | -0.013 |
| D_vs_C | 749.67 | 18.27% | 1766.179 | 5.7 | -0.053 | -0.027 | 0.027 |
| D_vs_B | 253.345 | 5.51% | -9925.948 | -25.1 | 0.013 | 0.04 | 0.013 |

## Dynamic INT8 rejection

Phase 2 dynamic INT8 was rejected, not selected as the Phase 3 D configuration.
Measured p95 latency: 9924.347 ms; peak RSS: 1402.6 MB; validity: 0.0; keyword coverage: 0.2.
- p95 latency was much worse than the other conditions.
- Peak RSS was higher than the unquantized pretrained conditions.
- Outputs included garbled text, so the latency measurement was not a usable deployment result.
- The approach used deprecated torch.ao.quantization dynamic INT8.

## Deployment recommendation

Selected configuration: **A**

No configuration is approved for production. All four failed every one of the seven out-of-scope test requests, produced no category tags, and scored only 1-2 of 15 on manual correctness. Configuration A is selected only as the least-risk demo fallback because it has the highest response validity among the non-adapter configurations (0.9067), the lowest truncation among the full-prompt variants (0.0933), and does not depend on the 26-example LoRA adapter. B has slightly higher validity (0.9333) and lower truncation (0.0667), but its p95 latency is 19.5 s and adapter generalisation is unproven. C is fastest (p50 4103 ms, p95 7842 ms) but truncates 28% of responses and has validity 0.72. D reduces p95 versus A but has validity 0.7867 and truncation 0.2133. A is a demo fallback, not a production recommendation: it still fails out-of-scope handling and most substantive policy checks.

## Limitations

- The manual review covers 15 of 75 examples, is not blinded across configurations, and was scored by one reviewer.
- Automatic lexical metrics can miss paraphrases and can reward keyword overlap without semantic correctness.
- Heuristic hallucination checks target clear fabricated order details, contact information, dates, amounts, and explicit policy contradictions; subtle errors may be missed.
- The fallback category classifier is rule-based and imperfect; tag accuracy is reported separately.
- Selecting the final configuration on the test set is optimistic because the test set is no longer untouched for that decision.
- Latency and memory were measured on one CPU-only machine with 75 examples; p95 is not a robust cross-hardware tail estimate.
- The Phase 1 LoRA adapter is a feasibility artifact trained on 26 examples for one epoch; no quality improvement is assumed.
- Dynamic INT8 was rejected on measured latency, memory, and output quality, not because quantization is inherently invalid.
- The test set is synthetic/fictional and is not production traffic; no online A/B test or monitoring was performed.
- FastAPI deployment and production readiness are out of scope for Phase 3.

## Artifacts

- `experiments/evaluation/predictions/`: per-condition predictions.
- `experiments/evaluation/final_results.json`: combined results and metadata.
- `experiments/evaluation/manual_review_sheet.jsonl`: 60 responses for the 15-example sample.
- `experiments/evaluation/manual_review_scores.json`: scores and reviewer notes.

# Rule-Based Baseline Report

Generated: 2026-11-05T05:36:42.066318+00:00
Backend: `rule_based_smoke`
Model: no model loaded (deterministic keyword rules)
Test examples: 75

> This is the full-test-set rule-based baseline. It is kept separate from the real
> generative SLM baseline in `reports/generative_baseline_report.md`.

## Metrics

| Metric | Value |
|---|---:|
| Category accuracy | 0.6800 |
| Response validity rate | 1.0000 |
| Unsupported refusal rate | 1.0000 |
| Policy keyword coverage | 0.7733 |
| Clarification rate | 1.0000 |
| Latency p50 (ms) | 0.031 |
| Latency p95 (ms) | 0.064 |
| Peak VRAM (MB) | 0 |

## Sample outputs

**shopease-000374** — expected `clarification`, predicted `clarification`

> I need my money back. Please advise.

[category: clarification]
I can help. Could you share your order ID and a short description of the issue so I can guide you to the right next step?

**shopease-000388** — expected `returns`, predicted `returns`

> I have a question: How long do I have to return a product?

[category: returns]
Most items can be returned within 7 days of delivery in original condition. Damaged or wrong items should be reported within 48 hours with photos so we can arrange a free return pickup.

**shopease-000753** — expected `general_faq`, predicted `general_faq`

> Hi! Can I combine coupons on one order? Please advise.

[category: general_faq]
ShopEase support is available 9 AM to 9 PM IST, seven days a week. You can also check the Help Centre for policies, invoices, coupons, and contact options.

**shopease-000547** — expected `order_status`, predicted `order_status`

> Quick question: Where is my order SE-65029381? Please advise.

[category: order_status]
Open My Orders in your ShopEase account to see the latest status and tracking for your order. If the status has not changed in 48 hours, share your order ID (SE-12345678) and we will investigate.

**shopease-000499** — expected `order_status`, predicted `order_status`

> Quick question: Order SE-76185042 shows delivered but there is no package. Thanks.

[category: order_status]
Open My Orders in your ShopEase account to see the latest status and tracking for your order. If the status has not changed in 48 hours, share your order ID (SE-12345678) and we will investigate.

## Scope

This is a deterministic rule-based baseline over the full 75-example test set. It is not a
language-model baseline and is not comparable to the generative model on quality; it only shows
what a keyword classifier plus fixed templates can achieve. The real SmolLM2-135M-Instruct
generative baseline is reported separately in `reports/generative_baseline_report.md` on 5 prompts.
No fine-tuning has been done.

# Generative SLM Baseline Report

Generated: 2026-11-06T09:56:00.923457+00:00
Backend: `transformers_pretrained` (generative baseline)
Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (local_checkpoint)
Model path: `D:\shakthi\slm-artifacts\models\SmolLM2-135M-Instruct`
Device: `cpu` — dtype float32, quantization none
Test examples: 75

## Generation settings

| Setting | Value |
|---|---:|
| max_new_tokens | 256 |
| do_sample | False |
| temperature | 1.0 |
| top_p | 1.0 |
| repetition_penalty | 1.05 |
| batch_size | 1 |

## Metrics

| Metric | Value |
|---|---:|
| Category tag rate | 0.0000 |
| Category accuracy | 0.0000 |
| Response validity rate | 0.0000 |
| Unsupported refusal rate | 0.1429 |
| Policy keyword coverage | 0.4000 |
| Clarification rate | 0.0000 |
| Model load (s) | 13.981 |
| Latency p50 (ms) | 4163.007 |
| Latency p95 (ms) | 12490.952 |
| Peak VRAM (MB) | 0 |

## Sample outputs

**shopease-000374** — expected `clarification`, predicted `unknown`

> I need my money back. Please advise.

I'm sorry for any misunderstanding, but as the ShopEase customer service assistant, I don't have the ability to provide financial assistance. However, I can guide you through the process of returning your order and providing payment. Please note that this may take some time, and it's always best to contact us directly if you're unsure about how to proceed. We also offer a free shipping option if you'd like to cancel your order.

**shopease-000388** — expected `returns`, predicted `unknown`

> I have a question: How long do I have to return a product?

The time it takes for a product to be returned varies depending on the product's size, weight, and condition. Generally, it can take anywhere from 24 to 72 hours to return a product after purchase. However, this timeframe may vary based on factors such as the product's size, weight, and condition.

If you're unsure about the return process, please contact our customer service team at (866) 531-0999 or email us at customer@shopesale.com. We want to ensure that your order is processed efficiently and that you receive your refund or replacement within the specified timeframe.

**shopease-000753** — expected `general_faq`, predicted `unknown`

> Hi! Can I combine coupons on one order? Please advise.

I'm sorry for any confusion, but as a ShopEase customer service assistant, my primary function is to assist customers with their orders. I don't have the ability to merge coupons or make decisions about combining them. I recommend contacting your local store's customer service department or asking a shop staff member directly for assistance.

**shopease-000547** — expected `order_status`, predicted `unknown`

> Quick question: Where is my order SE-65029381? Please advise.

I'm sorry for any confusion, but as the ShopEase customer service assistant, I don't have access to specific order details or customer information. I recommend contacting the customer service department at the store where your order is located to inquire about your order status.

**shopease-000499** — expected `order_status`, predicted `unknown`

> Quick question: Order SE-76185042 shows delivered but there is no package. Thanks.

The order SE-76185042 was shipped from the United States on 31 December 2023 at 10:00 AM. The order details are as follows:

Order ID: SE-76185042
Product: Sellotape
Quantity: 1
Delivery Date: 31 December 2023

Please note that this order may not be available in your area due to the current COVID-19 pandemic. If you have any questions about shipping or delivery, please contact our customer service team.

## Scope

Pretrained generative baseline only, with no fine-tuning. The test set is frozen and
is not used for training, hyperparameter tuning, or model selection.

This baseline is kept separate from the deterministic rule-based baseline in
`experiments/baseline/` (category accuracy 0.6800 on the full 75-example test set).

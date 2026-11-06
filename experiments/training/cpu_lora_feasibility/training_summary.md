# CPU LoRA Feasibility Report

Generated: 2026-11-06T10:19:13.246230+00:00
Status: **completed**

| Field | Value |
|---|---:|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Device | cpu |
| Dtype / quantization | float32 / none |
| Examples | 26 (13 categories) |
| Epochs | 1 |
| Max sequence length | 224 |
| LoRA r / alpha | 8 / 16 |
| Trainable params | 2442240 (1.7832%) |
| Optimizer steps | 7 |
| Model load | 1.247 s |
| Training | 37.05 s (1.425 s/example) |
| Initial / final loss | 3.585967 / 3.845357 |
| Peak process RSS | 1436.8 MB |
| Adapter files | README.md, adapter_config.json, adapter_model.safetensors, chat_template.jinja, tokenizer.json, tokenizer_config.json |

This run uses 26 training examples for one epoch on CPU. It verifies that the PEFT
LoRA path runs end-to-end and that the adapter saves correctly. It does not establish
fine-tuned quality; that requires a longer run and the Phase 3 evaluation.

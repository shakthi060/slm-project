# Observed Limitations

- Baseline uses the pretrained `HuggingFaceTB/SmolLM2-135M-Instruct` model without fine-tuning.
- Inference device: `cpu`; quantization: none.
- The base model was not trained to emit the `[category: ...]` tag, so format validity
  and category accuracy are expected to be low; this is a baseline limitation.
- Latency is measured with batch size 1 on `cpu`.

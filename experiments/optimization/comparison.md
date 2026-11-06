# Phase 2 Benchmark Comparison

Generated: 2026-11-06T10:27:15.223188+00:00

| Condition | Model | Prompt | Max new tokens | p50 ms | p95 ms | Load s | Peak RSS MB | Mean output tokens | Tag rate | Keyword coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_pretrained_original | pretrained | full_system_prompt | 256 | 5228.124 | 7983.631 | 2.655 | 976.6 | 93.8 | 0.0 | 0.6 |
| B_adapter_original | adapter | full_system_prompt | 256 | 4852.08 | 11727.133 | 2.858 | 996.8 | 85.2 | 0.0 | 0.4 |
| C_pretrained_optimized | pretrained | compact_system_prompt | 128 | 3139.171 | 7163.054 | 2.608 | 972.0 | 72.2 | 0.0 | 0.2 |
| C1_full_prompt_capped | pretrained | full_system_prompt | 128 | 4306.087 | 6216.338 | 2.314 | 976.0 | 92.6 | 0.0 | 0.6 |
| C2_compact_prompt_uncapped | pretrained | compact_system_prompt | 256 | 2713.167 | 6688.378 | 2.36 | 971.6 | 74 | 0.0 | 0.2 |
| D_dynamic_int8_investigation | pretrained-int8 | compact_system_prompt | 128 | 1659.274 | 9924.347 | 4.374 | 1402.6 | 48.2 | 0.0 | 0.2 |

## Deltas

| Comparison | p50 delta ms | p50 change | p95 delta ms | Load delta s | Peak RSS delta MB | Prompt-token delta |
|---|---:|---:|---:|---:|---:|---:|
| C_vs_A | -2088.953 | -39.96% | -820.577 | -0.047 | -4.6 | -41.0 |
| B_vs_A | -376.044 | -7.19% | 3743.502 | 0.203 | 20.2 | 0.0 |
| C_vs_B | -1712.909 | -35.3% | -4564.079 | -0.25 | -24.8 | -41.0 |
| C1_vs_A | -922.037 | -17.64% | -1767.293 | -0.341 | -0.6 | 0.0 |
| C2_vs_A | -2514.957 | -48.1% | -1295.253 | -0.295 | -5.0 | -41.0 |
| C1_vs_C | 1166.916 | 37.17% | -946.716 | -0.294 | 4.0 | 41.0 |
| C2_vs_C | -426.004 | -13.57% | -474.676 | -0.248 | -0.4 | 0.0 |

## Limitations

- Only five frozen prompts were measured.
- p95 of five samples is effectively the maximum, not a robust tail estimate.
- Automatic quality proxies are not blinded human evaluation.
- Latency improvements do not imply quality improvements.
- The LoRA adapter was trained on only 26 examples for one epoch.

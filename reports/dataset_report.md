# ShopEase Dataset Report

Generated: 2026-11-05T10:40:43
Random seed: 42
Policy document: `data/raw/shopease_policy.md`

All questions and answers are synthetic and were generated from the fictional ShopEase
policy document. No real customer data is used.

## Pipeline summary

| Stage | Records |
|---|---:|
| Raw generated | 820 |
| Invalid removed | 0 |
| Duplicate IDs removed | 0 |
| Exact duplicates removed | 0 |
| Near duplicates removed | 16 |
| Clean records used | 804 |
| JSON parse errors | 0 |

Near-duplicate threshold: `0.9` (token Jaccard, compared globally before splitting).

## Split sizes

| Split | Records | Share | Categories covered |
|---|---:|---:|---:|
| train | 654 | 81.3% | 13 |
| validation | 75 | 9.3% | 13 |
| test | 75 | 9.3% | 13 |
| **Total** | **804** | **100%** | **13** |

## Category distribution

| Category | Train | Validation | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| order_status | 56 | 7 | 7 | 70 | 8.7% |
| shipping | 57 | 6 | 6 | 69 | 8.6% |
| returns | 55 | 6 | 6 | 67 | 8.3% |
| refunds | 52 | 6 | 6 | 64 | 8.0% |
| payments | 48 | 6 | 6 | 60 | 7.5% |
| cancellations | 44 | 5 | 5 | 54 | 6.7% |
| product_info | 45 | 5 | 5 | 55 | 6.8% |
| warranty | 40 | 5 | 5 | 50 | 6.2% |
| account | 41 | 4 | 4 | 49 | 6.1% |
| complaints | 35 | 4 | 4 | 43 | 5.3% |
| general_faq | 48 | 6 | 6 | 60 | 7.5% |
| clarification | 69 | 8 | 8 | 85 | 10.6% |
| out_of_scope | 64 | 7 | 7 | 78 | 9.7% |

## Length statistics (whitespace words)

| Split | User min | User median | User mean | User max | Answer min | Answer median | Answer mean | Answer max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 4 | 9.0 | 9.1 | 16 | 11 | 30.0 | 29.16 | 43 |
| validation | 5 | 9 | 9.19 | 15 | 14 | 29 | 28.84 | 43 |
| test | 4 | 9 | 8.92 | 15 | 14 | 29 | 28.92 | 41 |

## Leakage checks

| Check | Result |
|---|---|
| Record ID overlap between splits | PASS |
| Exact user-text overlap between splits | PASS |
| Cross-split near-duplicate pairs (Jaccard >= 0.92) | PASS |
| Maximum cross-split Jaccard similarity | 0.8889 |

The test split is frozen after baseline evaluation and is never used for training or
hyperparameter tuning.

## Format

Each JSONL record contains: `id`, `category`, `intent`, `requires_clarification`,
`messages` (system, user, assistant), `source`, and `split` in the split files.
Assistant messages begin with `[category: <label>]` so the deployment layer can
extract the predicted support category.

## Reproducibility

```powershell
python -m src.data.generate_dataset --seed 42
python -m src.data.clean_dataset --near-threshold 0.92
python -m src.data.split_dataset --seed 42
python -m src.data.validate_dataset
```

## Limitations

- The dataset is synthetic and template-generated; it does not cover every real phrasing.
- Length statistics are whitespace-based; tokenizer-specific counts are produced during
  baseline evaluation.
- Near-duplicate detection uses token Jaccard, which is a practical proxy, not a semantic
  entailment model.
- Automatic evaluation cannot fully judge tone or policy correctness; a human rubric is
  used in the evaluation phase.

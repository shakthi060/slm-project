# API Verification Evidence

Generated: 2026-11-06T06:38:42.140481+00:00
Base URL: `http://127.0.0.1:8000`

## Startup health

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

## HTTP calls

| Method | Path | Status | Elapsed ms |
|---|---|---:|---:|
| GET | / | 200 | 3.48 |
| GET | /health | 200 | 2.181 |
| GET | /docs | 200 | 1.496 |
| GET | /ui | 200 | 59.683 |
| GET | /metrics | 200 | 2.273 |
| POST | /chat | 200 | 15408.909 |
| POST | /chat | 200 | 11026.154 |
| POST | /chat | 422 | 1.588 |
| POST | /chat | 422 | 1.347 |
| POST | /chat | 422 | 1.217 |
| POST | /chat | 422 | 1.284 |
| GET | /metrics | 200 | 1.744 |

## Successful chat results

### Chat 1

Fallback used: `True` (repetitive_output:repeated_list_items,repeated_sentences,repeated_phrases)

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

### Chat 2

Fallback used: `True` (repetitive_output:repeated_phrases)

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


## Final metrics

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

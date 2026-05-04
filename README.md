# LLM-Powered Test Failure Analyzer

A modular agent pipeline that ingests pytest failure logs, classifies root causes via the Claude API with structured JSON output, and exposes results through a FastAPI web interface. Includes a regression-preventing evaluation pipeline over 200 labeled failure cases.

## Architecture

```
raw log (str)
  → ingestion     → ParsedFailure        parse + token-truncate traceback
  → inference     → InferenceResult      LLM call OR heuristic fallback
  → validation    → ValidationResult     schema check + confidence gate
  → scoring       → ScoredResult         rank_score + actionable flag
  → storage       → StoredRecord         aiosqlite write
```

**LLM client** (`llm/client.py`): `AsyncAnthropic` with tool-use forced output, prompt caching, exponential backoff retry, and hard 2s timeout.

**Fallback** (`fallback/heuristics.py`): 8 regex rules covering all failure categories. Always returns `confidence ≤ 0.40` — distinguishable from LLM results. Triggered on API failure, timeout, or low LLM confidence.

**API** (`api/`): FastAPI with async routes for single analysis, batch analysis, and background eval jobs.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in ANTHROPIC_API_KEY
python scripts/seed_db.py
```

## Running

```bash
# Start API server
uvicorn analyzer.api.app:create_app --factory --reload --port 8000

# Analyze a single failure (example)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"raw_log": "FAILED tests/test_foo.py::test_bar - AssertionError: assert 1 == 2"}'

# Health check
curl http://localhost:8000/health
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze` | Classify a single pytest log |
| `POST` | `/api/v1/batch-analyze` | Classify up to 50 logs concurrently |
| `POST` | `/api/v1/evaluate` | Start background eval job, returns `job_id` |
| `GET`  | `/api/v1/evaluate/{job_id}` | Poll eval job result |
| `GET`  | `/health` | Liveness + DB connectivity check |
| `GET`  | `/metrics` | Aggregate stats from SQLite |

### Example response — `POST /api/v1/analyze`

```json
{
  "record_id": "a3f2...",
  "test_id": "tests/test_payments.py::test_user_balance",
  "category": "assertion_error",
  "summary": "Value comparison failed: get_balance returned 150.0 instead of 200.0",
  "explanation": "The test asserts that get_balance returns 200.0...",
  "fix_hint": "Review get_balance logic and verify test fixture creates the correct balance.",
  "confidence": 0.95,
  "is_flaky": false,
  "fallback_used": false,
  "fallback_source": "llm",
  "latency_ms": 482.3
}
```

## Failure Categories

| Category | Description |
|----------|-------------|
| `assertion_error` | Value comparison failures, unexpected output |
| `import_error` | Missing modules, circular imports |
| `fixture_error` | pytest fixture setup/teardown failures |
| `timeout` | Test exceeded time limit or hung on I/O |
| `environment_error` | Missing env vars, OS-level issues |
| `network_error` | External service unreachable, SSL errors |
| `data_error` | Corrupt test data, missing fixture files |
| `flaky` | Non-deterministic failures, race conditions |
| `unknown` | Cannot be classified with confidence |

## Testing

No test makes a real API call. Unit tests mock `LLMClient` with `AsyncMock`; integration tests mock the LLMClient directly.

```bash
# Run all tests
pytest tests/unit/ tests/integration/ -v

# With coverage
pytest tests/ -v --cov=src/analyzer --cov-report=html
```

## Evaluation Pipeline

The eval pipeline runs the full stack over 200 labeled failure cases and reports per-category precision, recall, and F1.

```bash
# Generate the dataset (one-time)
python scripts/generate_eval_dataset.py

# Quick sanity check (20 cases, uses real API)
python scripts/run_eval.py --limit 20

# Full evaluation (200 cases)
python scripts/run_eval.py
```

Example output:
```
============================================================
EVALUATION REPORT
============================================================
  Total cases  : 200
  Correct      : 178  (89.0%)
  Accuracy     : 0.8900
  Macro P      : 0.8812
  Macro R      : 0.8754
  Macro F1     : 0.8783
  Avg latency  : 523ms
  Fallback rate: 3.5%

  Category               P      R     F1  Support
  -------------------------------------------------------
  assertion_error    0.952  0.940  0.946       50
  data_error         0.867  0.867  0.867       15
  ...
```

## Prompt Caching Strategy

The frozen `SYSTEM_PROMPT` lives exclusively in the system block with `cache_control: {type: "ephemeral"}`. Per-request log data goes in the user message only. This yields a cache hit on every warm request (~300ms savings, ~90% reduction in system-prompt token cost).

## Design Decisions

- **Schema-constrained outputs**: tool-use with `RootCauseHypothesis` input schema eliminates JSON parsing failures.
- **Confidence gating**: 0.65 threshold separates LLM from heuristic fallback; heuristic floor is 0.10–0.40.
- **No silent failures**: every error path routes to heuristic with a known confidence floor, never to an unhandled exception.
- **Async batching**: `asyncio.gather` + `Semaphore(5)` gives ~5× throughput on batch endpoints.

## Trade-offs & Limitations

**LLM latency is variable and high on cold starts.**
Cache-hit requests run in ~400–600ms. A cold start (first request after the 5-minute prompt-cache TTL expires) takes 5–10s. The 2s default timeout in `.env.example` is intentionally aggressive for production — raise `LLM_TIMEOUT_MS` to 10000–15000 for local use. This trade-off was discovered during testing: 3 retries × 2s = 6s worst-case before heuristic fallback kicks in.

**Heuristic fallback sacrifices accuracy for reliability.**
The 8 regex rules always return `confidence ≤ 0.40` and cover only the obvious patterns. A `ModuleNotFoundError` buried inside a wrapped exception won't match. This is intentional — a low-confidence answer is always better than an unhandled exception — but it means ~5–15% of real-world failures will get a coarse classification.

**The confidence threshold (0.65) is a fixed heuristic, not learned.**
The cutoff between "trust the LLM" and "fall back to heuristics" was chosen empirically. It has not been calibrated against a held-out validation set. A project with more labeled data should tune this with a proper precision-recall curve.

**SQLite limits horizontal scalability.**
`aiosqlite` is single-writer. Concurrent writes from multiple uvicorn workers will serialize. For multi-worker deployments, swap the storage layer for PostgreSQL (the `store_result` interface is the only surface that needs to change).

**The eval job store is in-memory.**
`routes/evaluate.py` uses a plain Python dict (`_jobs`) to track background eval jobs. Restarting the server loses all job state. Acceptable for a dev/eval tool; not acceptable in production. A Redis-backed job queue (Celery, ARQ) would be the next step.

**The eval dataset is programmatically generated, not hand-labeled.**
`data/eval_dataset/labeled_failures.json` was produced by `scripts/generate_eval_dataset.py` from templates, not from real CI failures. This makes eval scores optimistic — the model is tested on clean, single-cause logs. Real pytest output is messier (wrapped exceptions, multi-cause failures, framework noise). The generator is the right starting point; replacing it with real labeled failures is the highest-value improvement.

**`StoredRecord.db_path` uses a private aiosqlite API.**
Extracting the database path requires reading a closure variable from `db._connector`. This is fragile across aiosqlite versions and will silently return `""` if the internal layout changes. A cleaner solution would pass `db_path` explicitly at construction time.

## Tech Stack

Python 3.11+, `anthropic` SDK, FastAPI, Pydantic v2, aiosqlite, pytest, respx, ruff, mypy

## Environment Variables

See `.env.example` for all variables and defaults. Required: `ANTHROPIC_API_KEY`.

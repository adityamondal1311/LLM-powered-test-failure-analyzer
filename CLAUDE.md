# LLM-Powered Test Failure Analyzer — CLAUDE.md

## Project Summary
Modular agent pipeline that ingests pytest failure logs, classifies root causes via the
Claude API (claude-sonnet-4-6) with structured JSON output, and exposes results through
a FastAPI web interface. Includes a regression-preventing evaluation pipeline over ~200
labeled failure cases.

## Current State (as of last session)

### BUILT — all core source code is complete
- `src/analyzer/models/pipeline.py` — all Pydantic v2 I/O contracts (ParsedFailure, RootCauseHypothesis, InferenceResult, ValidationResult, ScoredResult, StoredRecord, FailureCategory, FallbackSource)
- `src/analyzer/models/api.py` — FastAPI request/response models
- `src/analyzer/models/eval.py` — eval dataset + report schemas
- `src/analyzer/config.py` — pydantic-settings (reads from .env)
- `src/analyzer/pipeline/ingestion.py` — parses raw pytest logs, truncates to token budget
- `src/analyzer/pipeline/inference.py` — LLM call + fallback routing + async batching with Semaphore
- `src/analyzer/pipeline/validation.py` — schema integrity + confidence gate
- `src/analyzer/pipeline/scoring.py` — rank_score computation, actionable flag
- `src/analyzer/pipeline/storage.py` — aiosqlite CRUD + aggregate stats
- `src/analyzer/llm/prompts.py` — frozen SYSTEM_PROMPT (cache anchor) + build_user_message
- `src/analyzer/llm/client.py` — AsyncAnthropic with prompt caching, tool-use structured output, exponential backoff retry, asyncio.wait_for timeout
- `src/analyzer/fallback/heuristics.py` — 8 regex rules, returns confidence ≤ 0.40
- `src/analyzer/eval/metrics.py` — precision/recall/F1 per category + macro averages
- `src/analyzer/eval/runner.py` — runs full pipeline over labeled dataset
- `src/analyzer/api/app.py` — FastAPI factory with lifespan (db + llm client wiring)
- `src/analyzer/api/middleware.py` — request-id + timing headers
- `src/analyzer/api/routes/analyze.py` — POST /api/v1/analyze + POST /api/v1/batch-analyze
- `src/analyzer/api/routes/evaluate.py` — POST /api/v1/evaluate (background job) + GET /{job_id}
- `src/analyzer/api/routes/health.py` — GET /health + GET /metrics
- `scripts/seed_db.py` — initializes SQLite schema

### NOT YET BUILT — remaining work
- `scripts/run_eval.py` — CLI wrapper for eval runner
- `data/eval_dataset/labeled_failures.json` — ~200 hand-curated labeled pytest failure cases
- `tests/conftest.py` — shared pytest fixtures (mock LLMClient, temp db, test app)
- `tests/fixtures/sample_logs.py` — sample raw pytest log strings
- `tests/fixtures/sample_responses.py` — sample RootCauseHypothesis objects
- `tests/unit/test_ingestion.py`
- `tests/unit/test_heuristics.py`
- `tests/unit/test_inference.py` — mocks LLMClient
- `tests/unit/test_validation.py`
- `tests/unit/test_scoring.py`
- `tests/unit/test_storage.py`
- `tests/integration/test_pipeline_end_to_end.py`
- `tests/integration/test_api_routes.py` — uses respx to mock Anthropic HTTP
- `.github/workflows/ci.yml` — matrix 3.11+3.12, ruff+mypy+pytest
- `README.md`
- `git init` — repo not yet initialized

## Essential Commands

### Setup
```bash
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
python scripts/seed_db.py
```

### Run the API Server
```bash
uvicorn analyzer.api.app:create_app --factory --reload --port 8000
```

### Run Tests (no LLM calls — once tests are written)
```bash
pytest tests/unit/ tests/integration/ -v
```

### Run Tests with Coverage
```bash
pytest tests/ -v --cov=src/analyzer --cov-report=html
```

### Run the Evaluation Pipeline (requires real API key)
```bash
python scripts/run_eval.py --limit 20    # quick sanity check
python scripts/run_eval.py               # full 200-case eval
```

### Lint and Type Check
```bash
ruff check src/ tests/
mypy src/analyzer
```

## Architecture

### Pipeline Stages (always run in this order)
1. **Ingestion** (`pipeline/ingestion.py`) — parses raw pytest stdout into `ParsedFailure`.
   Truncates traceback to stay within token budget (~1800 tokens max). Stateless.

2. **Inference** (`pipeline/inference.py`) — calls Claude via `LLMClient`. Uses tool-use
   with `classify_test_failure` tool and `RootCauseHypothesis` input_schema to guarantee
   valid structured output. Falls back to heuristics on API failure or low confidence.

3. **Validation** (`pipeline/validation.py`) — re-validates the Pydantic model and checks
   confidence threshold. Pure function; no I/O.

4. **Scoring** (`pipeline/scoring.py`) — computes `rank_score`, determines `actionable`
   flag. Pure function; no I/O.

5. **Storage** (`pipeline/storage.py`) — writes to SQLite via aiosqlite.

### Prompt Caching Strategy
- `SYSTEM_PROMPT` in `llm/prompts.py` is the ONLY content in the system block.
- It has `cache_control: {type: "ephemeral"}` — cached for 5 minutes per TTL.
- **NEVER** add timestamps, user IDs, or per-request data to the system block.
- Per-request data (log content) goes in the user message only.
- Verify cache hits by checking `usage.cache_read_input_tokens > 0`.

### Fallback Routing
When `inference.py` catches `anthropic.RateLimitError`, `anthropic.InternalServerError`,
or `asyncio.TimeoutError`, it calls `heuristics.classify_by_heuristic()` and sets
`fallback_used=True`. Heuristic results always have `confidence <= 0.40` so they are
distinguishable from LLM results downstream.

### Latency Budget (target: ~800ms end-to-end)
| Step | Time |
|------|------|
| Ingestion | ~5ms |
| LLM inference (cache hit) | ~400–600ms |
| LLM inference (cache miss) | ~700–900ms |
| Validation + Scoring | ~2ms |
| SQLite write | ~5ms |

Batch mode uses `asyncio.gather` with `Semaphore(5)` for concurrent requests.

## Conventions

### Pydantic Models
- All I/O contracts are Pydantic v2 `BaseModel` subclasses in `models/`.
- Use `model_validate_json()` to parse LLM output (not `json.loads` + `model_validate()`).
- All confidence floats are rounded to 4 decimal places via `@field_validator`.

### Async
- All pipeline functions are `async def`.
- Database operations use `aiosqlite` exclusively.
- LLM calls use `AsyncAnthropic`, wrapped in `asyncio.wait_for` for per-request timeouts.

### Testing
- Unit tests mock `LLMClient` entirely — use `unittest.mock.AsyncMock`.
- Integration tests mock the Anthropic HTTP endpoint with `respx`.
- Fixtures in `tests/fixtures/` provide sample objects for reuse across tests.
- No test makes a real API call. The eval script does (opt-in, requires real key).

### Structured Output via Tool Use
Claude is forced to call `classify_test_failure` with `tool_choice={"type":"tool","name":"classify_test_failure"}`.
The `input_schema` is `RootCauseHypothesis.model_json_schema()` (title field stripped).
The response is parsed via `RootCauseHypothesis.model_validate(tool_block.input)`.
This eliminates JSON parsing failures at the source.

### Error Handling
- Pipeline stages return typed results; they do not raise into callers.
- `inference.py` is the only stage with external I/O; it catches all Anthropic SDK
  exceptions and routes to heuristics rather than propagating.
- FastAPI route handlers convert pipeline exceptions to `HTTPException`.

## Environment Variables
See `.env.example` for all supported variables and their defaults.

## Key Design Decisions
- **Schema-constrained outputs**: tool-use with `RootCauseHypothesis` input_schema eliminates
  JSON parsing failures at the source.
- **Prompt caching**: frozen system prompt + volatile user message = cache hit on every
  warm request. Saves ~300ms and ~90% of system-prompt token cost.
- **Confidence gating**: 0.65 threshold separates LLM output from heuristic fallback.
- **Independent testability**: ingestion, validation, and scoring have no side effects
  and can be unit-tested in pure Python with no mocking.
- **Async batching**: `asyncio.gather` with `Semaphore(5)` gives ~5× throughput on batch
  endpoints without hitting Anthropic rate limits.
- **No silent failures**: every error path routes to heuristic fallback with a known
  confidence floor (0.10–0.40), never to an unhandled exception.

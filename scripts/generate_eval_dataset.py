"""Generate data/eval_dataset/labeled_failures.json with ~200 labeled cases.

Each case is a realistic pytest failure log with a known ground-truth category.
Run once; the output file is checked in and used by the eval pipeline.

Usage:
    python scripts/generate_eval_dataset.py
    python scripts/generate_eval_dataset.py --out custom/path.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Templates per category
# (error_type, error_msg, traceback_tail, fix_note) × many variants
# ---------------------------------------------------------------------------

_ASSERTION = [
    ("AssertionError", "assert {a} == {b}", "assert result == expected\nE   AssertionError: assert {a} == {b}", False),
    ("AssertionError", "assert response.status_code == 200", "assert response.status_code == 200\nE   AssertionError: assert 404 == 200", False),
    ("AssertionError", "assert len(items) == {n}", "assert len(items) == {n}\nE   AssertionError: assert 0 == {n}", False),
    ("AssertionError", "assert user.is_active is True", "assert user.is_active is True\nE   AssertionError: assert False is True", False),
    ("AssertionError", "assert result['status'] == 'ok'", "assert result['status'] == 'ok'\nE   AssertionError: assert 'error' == 'ok'", False),
    ("AssertionError", "assert price > 0", "assert price > 0\nE   AssertionError: assert -1 > 0", False),
    ("AssertionError", "assert output.startswith('Hello')", "assert output.startswith('Hello')\nE   AssertionError: assert False", False),
    ("AssertionError", "assert db_record is not None", "assert db_record is not None\nE   AssertionError", False),
    ("AssertionError", "assert sorted(a) == sorted(b)", "assert sorted(a) == sorted(b)\nE   AssertionError: lists differ", False),
    ("AssertionError", "assert mock.call_count == 1", "assert mock.call_count == 1\nE   AssertionError: assert 0 == 1", False),
]

_IMPORT = [
    ("ModuleNotFoundError", "No module named 'redis'", "ModuleNotFoundError: No module named 'redis'", False),
    ("ModuleNotFoundError", "No module named 'celery'", "ModuleNotFoundError: No module named 'celery'", False),
    ("ModuleNotFoundError", "No module named 'boto3'", "ModuleNotFoundError: No module named 'boto3'", False),
    ("ImportError", "cannot import name 'AsyncClient' from 'httpx'", "ImportError: cannot import name 'AsyncClient'", False),
    ("ModuleNotFoundError", "No module named 'psycopg2'", "ModuleNotFoundError: No module named 'psycopg2'", False),
    ("ImportError", "cannot import name 'field_validator' from 'pydantic'", "ImportError: cannot import name 'field_validator'", False),
    ("ModuleNotFoundError", "No module named 'openai'", "ModuleNotFoundError: No module named 'openai'", False),
]

_FIXTURE = [
    ("FixtureError", "fixture 'db_session' not found", "fixture 'db_session' not found\navailable fixtures: tmp_path, capsys", False),
    ("FixtureError", "fixture 'mock_s3' not found", "fixture 'mock_s3' not found\navailable fixtures: tmp_path", False),
    ("FixtureError", "fixture 'auth_client' not found", "fixture 'auth_client' not found", False),
    ("FixtureError", "fixture 'event_loop' not found", "fixture 'event_loop' not found\nhint: use pytest-asyncio", False),
    ("FixtureError", "fixture setup failed: ConnectionRefused", "fixture 'redis_client' setup failed error\nConnectionRefusedError: [Errno 111]", False),
    ("FixtureError", "fixture 'app' error during teardown", "fixture 'app' teardown error failed\nRuntimeError: event loop closed", False),
    ("FixtureError", "ScopeMismatch: function-scoped fixture in session scope", "fixture scope error failed\nScopeMismatch: function-scoped 'db'", False),
]

_TIMEOUT = [
    ("TimeoutError", "operation timed out after 5.0s", "TimeoutError: operation timed out after 5.0s", False),
    ("TimeoutError", "test timed out after 30s", "TimeoutError: test timed out after 30s", False),
    ("asyncio.TimeoutError", "future timed out", "asyncio.TimeoutError\nfuture: <Task pending>", False),
    ("socket.timeout", "timed out", "socket.timeout: timed out\nconn.recv(1024)", False),
    ("TimeoutError", "read timed out after 10s", "TimeoutError: read timed out after 10s", False),
]

_ENVIRONMENT = [
    ("KeyError", "'DATABASE_URL'", "KeyError: 'DATABASE_URL'\nos.environ['DATABASE_URL']", False),
    ("KeyError", "'ANTHROPIC_API_KEY'", "KeyError: 'ANTHROPIC_API_KEY'\nenv var ANTHROPIC_API_KEY not set", False),
    ("KeyError", "'SMTP_HOST'", "KeyError: 'SMTP_HOST'\nos.environ access", False),
    ("KeyError", "'AWS_SECRET_ACCESS_KEY'", "KeyError: 'AWS_SECRET_ACCESS_KEY'\nos.environ['AWS_SECRET_ACCESS_KEY']", False),
    ("EnvironmentError", "REDIS_URL not configured", "EnvironmentError: REDIS_URL not configured\nenv var missing", False),
    ("RuntimeError", "Python 3.9 required, got 3.8", "RuntimeError: Python 3.9 required\nos.environ check", False),
    ("KeyError", "'SECRET_KEY'", "KeyError: 'SECRET_KEY'\nenv var SECRET_KEY not set", False),
    ("EnvironmentError", "missing required env var: STRIPE_KEY", "EnvironmentError: missing required env var STRIPE_KEY", False),
    ("KeyError", "'POSTGRES_PASSWORD'", "KeyError: 'POSTGRES_PASSWORD'\nenv var access", False),
]

_NETWORK = [
    ("ConnectionError", "Max retries exceeded with url: /api/v1/status", "ConnectionError: HTTPSConnectionPool\nMax retries exceeded", False),
    ("ConnectionRefusedError", "[Errno 111] Connection refused", "ConnectionRefusedError: [Errno 111] Connection refused\nport 5432", False),
    ("SSLError", "certificate verify failed: unable to get local issuer certificate", "SSLError: certificate verify failed", False),
    ("HTTPError", "503 Service Unavailable", "HTTPError: 503 Service Unavailable\nrequests.get(url)", False),
    ("ConnectionError", "DNS resolution failed for api.example.com", "ConnectionError: DNS\nDNS resolution failed", False),
    ("TimeoutError", "HTTPSConnectionPool read timed out", "ConnectionError: HTTPSConnectionPool\nRead timed out", False),
    ("ConnectionResetError", "[Errno 104] Connection reset by peer", "ConnectionResetError: [Errno 104]", False),
]

_DATA = [
    ("FileNotFoundError", "[Errno 2] No such file or directory: 'data/test_input.csv'", "FileNotFoundError: No such file or directory: 'data/test_input.csv'", False),
    ("FileNotFoundError", "No such file or directory: 'fixtures/schema.json'", "FileNotFoundError: No such file or directory: 'fixtures/schema.json'", False),
    ("ValueError", "schema mismatch: expected 'id' field missing", "ValueError: schema mismatch\nexpected 'id' field missing", False),
    ("json.JSONDecodeError", "Expecting value: line 1 column 1 (char 0)", "json.JSONDecodeError: Expecting value", False),
    ("KeyError", "'user_id' not in response payload", "KeyError: 'user_id'\nschema mismatch", False),
    ("TypeError", "expected str but got NoneType in test data", "TypeError: expected str, got NoneType\nNo such file", False),
]

_FLAKY = [
    ("AssertionError", "assert count == 100 (flaky: race condition)", "assert count == 100\nE   AssertionError: assert 99 == 100\n# known flaky due to race condition", True),
    ("AssertionError", "assert result is non-deterministic", "assert result == expected\nE   AssertionError\n# non-deterministic test output", True),
    ("AssertionError", "timing-dependent assertion failed", "assert elapsed < 1.0\nE   AssertionError: flaky timing", True),
    ("AssertionError", "order-dependent test failure", "assert items[0] == 'first'\nE   AssertionError: race condition in ordering", True),
    ("RuntimeError", "flaky: event loop already running", "RuntimeError: This event loop is already running\nflaky test", True),
]

_UNKNOWN = [
    ("RecursionError", "maximum recursion depth exceeded", "RecursionError: maximum recursion depth exceeded", False),
    ("MemoryError", "", "MemoryError\nfailed to allocate memory", False),
    ("SystemExit", "1", "SystemExit: 1", False),
    ("AttributeError", "'NoneType' object has no attribute 'split'", "AttributeError: 'NoneType' object has no attribute 'split'", False),
    ("TypeError", "unsupported operand type(s) for +: 'int' and 'str'", "TypeError: unsupported operand type(s) for +", False),
    ("RuntimeError", "coroutine was never awaited", "RuntimeError: coroutine 'foo' was never awaited", False),
    ("OverflowError", "Python int too large to convert to C long", "OverflowError: Python int too large", False),
]

# (templates_list, category, target_count)
_PLAN: list[tuple[list, str, int]] = [
    (_ASSERTION, "assertion_error", 50),
    (_IMPORT, "import_error", 20),
    (_FIXTURE, "fixture_error", 20),
    (_TIMEOUT, "timeout", 15),
    (_ENVIRONMENT, "environment_error", 25),
    (_NETWORK, "network_error", 20),
    (_DATA, "data_error", 15),
    (_FLAKY, "flaky", 15),
    (_UNKNOWN, "unknown", 20),
]


def _make_log(
    category: str,
    error_type: str,
    error_msg: str,
    tb_tail: str,
    idx: int,
) -> str:
    test_module = f"tests/test_{category}_{idx:03d}.py"
    test_fn = f"test_{category}_case_{idx}"
    a, b, n = idx * 3, idx * 3 + 1, idx + 1
    log = (
        f"FAILED {test_module}::{test_fn} - {error_type}: {error_msg}\n"
        f"============================= FAILURES ==============================\n"
        f"_________________________ {test_fn} _________________________\n"
        f"{test_module}:{10 + idx % 40}: in {test_fn}\n"
        f"    {tb_tail.format(a=a, b=b, n=n)}\n"
        f"========================= 1 failed in {0.05 + idx * 0.01:.2f}s ========================="
    )
    return log


def _difficulty(i: int, total: int) -> str:
    third = total // 3
    if i < third:
        return "easy"
    if i < 2 * third:
        return "medium"
    return "hard"


def generate(seed: int = 42) -> list[dict]:
    random.seed(seed)
    cases = []
    counter = 1

    for templates, category, target in _PLAN:
        for i in range(target):
            tmpl = templates[i % len(templates)]
            error_type, error_msg, tb_tail, is_flaky = tmpl
            case = {
                "id": f"tc_{counter:03d}",
                "raw_log": _make_log(category, error_type, error_msg, tb_tail, counter),
                "ground_truth_category": category,
                "ground_truth_is_flaky": is_flaky,
                "difficulty": _difficulty(i, target),
                "notes": f"Auto-generated {category} variant {i + 1}",
            }
            cases.append(case)
            counter += 1

    random.shuffle(cases)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the eval dataset JSON.")
    parser.add_argument(
        "--out",
        default="data/eval_dataset/labeled_failures.json",
        metavar="PATH",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cases = generate(seed=args.seed)
    dataset = {
        "version": "1.0",
        "description": (
            "Auto-generated labeled pytest failure cases for eval pipeline. "
            "200 cases across 9 categories with easy/medium/hard difficulty split."
        ),
        "created": "2026-05-04",
        "cases": cases,
    }

    out.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out}")

    # Print distribution
    from collections import Counter
    dist = Counter(c["ground_truth_category"] for c in cases)
    for cat, count in sorted(dist.items()):
        print(f"  {cat:<22} {count:>3}")


if __name__ == "__main__":
    main()

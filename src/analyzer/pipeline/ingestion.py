from __future__ import annotations

import re
import time

from analyzer.models.pipeline import ParsedFailure

_FAILED_LINE = re.compile(r"^FAILED\s+(.+?)(?:\s+-\s+(.+))?$", re.MULTILINE)
_ERROR_TYPE = re.compile(r"^([A-Za-z][A-Za-z0-9_.]*(?:Error|Exception|Warning|Exit))\b")
_TEST_NODE = re.compile(r"^(.+\.py)::(\S+)")

_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_to_token_budget(lines: list[str], max_tokens: int) -> list[str]:
    """Keep the last N traceback lines that fit within max_tokens.
    Always preserves the first line (error header) and last line (exception).
    """
    if not lines:
        return lines
    if len(lines) <= 2:
        return lines

    header = lines[0]
    tail = lines[-1]
    budget = max_tokens - _estimate_tokens(header) - _estimate_tokens(tail)

    middle: list[str] = []
    for line in reversed(lines[1:-1]):
        cost = _estimate_tokens(line)
        if budget - cost < 0:
            break
        budget -= cost
        middle.insert(0, line)

    return [header] + middle + [tail]


def _extract_error_type(traceback_text: str) -> str:
    for line in reversed(traceback_text.splitlines()):
        m = _ERROR_TYPE.match(line.strip())
        if m:
            return m.group(1)
    return "UnknownError"


def _extract_error_message(traceback_text: str, error_type: str) -> str:
    for line in reversed(traceback_text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(error_type + ":"):
            return stripped[len(error_type) + 1 :].strip()
        if stripped.startswith(error_type):
            return stripped[len(error_type) :].strip().lstrip(": ")
    return ""


def _parse_test_node(node_id: str) -> tuple[str, str]:
    """Returns (test_file, test_function) from a pytest node ID."""
    m = _TEST_NODE.match(node_id)
    if m:
        return m.group(1), m.group(2)
    return node_id, node_id


def _parse_duration(raw_log: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\b", raw_log, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000
    return 0.0


async def ingest_log(raw_log: str, test_id: str | None = None) -> ParsedFailure:
    """Parse a raw pytest log string into a normalized ParsedFailure."""
    try:
        return _do_ingest(raw_log, test_id)
    except Exception as exc:
        return ParsedFailure(
            test_id=test_id or "unknown",
            test_file="unknown",
            test_function="unknown",
            error_type="PARSE_FAILURE",
            error_message=repr(exc),
            traceback_lines=[raw_log[:500]],
            duration_ms=0.0,
            raw_log=raw_log,
            token_estimate=_estimate_tokens(raw_log),
        )


def _do_ingest(raw_log: str, test_id: str | None) -> ParsedFailure:
    lines = raw_log.splitlines()

    # Try to extract test node ID from "FAILED path::test_name - ErrorType: msg"
    extracted_id: str = test_id or "unknown"
    m = _FAILED_LINE.search(raw_log)
    if m and not test_id:
        extracted_id = m.group(1).strip()

    test_file, test_function = _parse_test_node(extracted_id)

    # Isolate traceback block
    tb_start = -1
    for i, line in enumerate(lines):
        if line.strip() in ("FAILURES", "ERRORS") or line.startswith("_ "):
            tb_start = i
            break
    tb_lines = lines[tb_start:] if tb_start >= 0 else lines

    error_type = _extract_error_type("\n".join(tb_lines))
    error_message = _extract_error_message("\n".join(tb_lines), error_type)
    duration_ms = _parse_duration(raw_log)

    truncated = _truncate_to_token_budget(tb_lines, max_tokens=1800)
    token_est = _estimate_tokens("\n".join(truncated))

    return ParsedFailure(
        test_id=extracted_id,
        test_file=test_file,
        test_function=test_function,
        error_type=error_type,
        error_message=error_message,
        traceback_lines=truncated,
        duration_ms=duration_ms,
        collected_at_utc=time.time(),
        raw_log=raw_log,
        token_estimate=token_est,
    )


async def ingest_batch(
    logs: list[str], test_ids: list[str] | None = None
) -> list[ParsedFailure]:
    results = []
    for i, log in enumerate(logs):
        tid = test_ids[i] if test_ids and i < len(test_ids) else None
        results.append(await ingest_log(log, tid))
    return results

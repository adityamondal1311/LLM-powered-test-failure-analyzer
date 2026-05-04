"""Unit tests for pipeline/ingestion.py."""

from __future__ import annotations

import pytest

from analyzer.pipeline.ingestion import (
    _estimate_tokens,
    _extract_error_message,
    _extract_error_type,
    _parse_duration,
    _parse_test_node,
    _truncate_to_token_budget,
    ingest_batch,
    ingest_log,
)
from tests.fixtures.sample_logs import (
    ASSERTION_ERROR_LOG,
    IMPORT_ERROR_LOG,
    TIMEOUT_LOG,
    VERY_LONG_LOG,
)


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_empty() -> None:
    assert _estimate_tokens("") == 1


def test_estimate_tokens_basic() -> None:
    text = "a" * 400
    assert _estimate_tokens(text) == 100


# ---------------------------------------------------------------------------
# _parse_test_node
# ---------------------------------------------------------------------------


def test_parse_test_node_standard() -> None:
    file, func = _parse_test_node("tests/test_foo.py::test_bar")
    assert file == "tests/test_foo.py"
    assert func == "test_bar"


def test_parse_test_node_no_sep() -> None:
    file, func = _parse_test_node("not_a_node_id")
    assert file == "not_a_node_id"
    assert func == "not_a_node_id"


def test_parse_test_node_parametrized() -> None:
    file, func = _parse_test_node("tests/test_foo.py::test_bar[case0]")
    assert file == "tests/test_foo.py"
    assert func == "test_bar[case0]"


# ---------------------------------------------------------------------------
# _extract_error_type
# ---------------------------------------------------------------------------


def test_extract_error_type_assertion() -> None:
    tb = "    assert x == y\nAssertionError: assert 1 == 2"
    assert _extract_error_type(tb) == "AssertionError"


def test_extract_error_type_module_not_found() -> None:
    tb = "ModuleNotFoundError: No module named 'redis'"
    assert _extract_error_type(tb) == "ModuleNotFoundError"


def test_extract_error_type_unknown() -> None:
    tb = "    something happened"
    assert _extract_error_type(tb) == "UnknownError"


# ---------------------------------------------------------------------------
# _extract_error_message
# ---------------------------------------------------------------------------


def test_extract_error_message_basic() -> None:
    tb = "AssertionError: assert 1 == 2"
    msg = _extract_error_message(tb, "AssertionError")
    assert msg == "assert 1 == 2"


def test_extract_error_message_no_colon() -> None:
    tb = "AssertionError"
    msg = _extract_error_message(tb, "AssertionError")
    assert msg == ""


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------


def test_parse_duration_seconds() -> None:
    assert _parse_duration("1 failed in 0.23s") == pytest.approx(230.0)


def test_parse_duration_no_match() -> None:
    assert _parse_duration("no duration info here") == 0.0


def test_parse_duration_seconds_word() -> None:
    assert _parse_duration("completed in 2 seconds") == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# _truncate_to_token_budget
# ---------------------------------------------------------------------------


def test_truncate_empty() -> None:
    assert _truncate_to_token_budget([], 100) == []


def test_truncate_short_stays_intact() -> None:
    lines = ["header", "middle", "footer"]
    result = _truncate_to_token_budget(lines, 1000)
    assert result == lines


def test_truncate_preserves_header_and_footer() -> None:
    lines = ["header"] + [f"line {i}" for i in range(200)] + ["footer"]
    result = _truncate_to_token_budget(lines, 100)
    assert result[0] == "header"
    assert result[-1] == "footer"
    assert len(result) < len(lines)


def test_truncate_two_lines_unchanged() -> None:
    lines = ["header", "footer"]
    result = _truncate_to_token_budget(lines, 10)
    assert result == lines


# ---------------------------------------------------------------------------
# ingest_log (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_log_assertion_error() -> None:
    result = await ingest_log(ASSERTION_ERROR_LOG)
    assert result.error_type == "AssertionError"
    assert result.test_file == "tests/test_payments.py"
    assert result.test_function == "test_user_balance"
    assert result.duration_ms == pytest.approx(230.0)
    assert result.token_estimate > 0


@pytest.mark.asyncio
async def test_ingest_log_import_error() -> None:
    result = await ingest_log(IMPORT_ERROR_LOG)
    assert result.error_type in ("ModuleNotFoundError", "ImportError")


@pytest.mark.asyncio
async def test_ingest_log_timeout() -> None:
    result = await ingest_log(TIMEOUT_LOG)
    assert result.error_type == "TimeoutError"
    assert result.duration_ms == pytest.approx(5120.0)


@pytest.mark.asyncio
async def test_ingest_log_explicit_test_id() -> None:
    result = await ingest_log(ASSERTION_ERROR_LOG, test_id="my_explicit_id")
    assert result.test_id == "my_explicit_id"


@pytest.mark.asyncio
async def test_ingest_log_minimal() -> None:
    raw = "FAILED tests/test_foo.py::test_bar - AssertionError: assert 1 == 2"
    result = await ingest_log(raw)
    assert result.test_file == "tests/test_foo.py"
    assert result.test_function == "test_bar"


@pytest.mark.asyncio
async def test_ingest_log_truncates_long_log() -> None:
    result = await ingest_log(VERY_LONG_LOG)
    # Token budget is 1800; a 1000-line traceback should be truncated
    assert result.token_estimate <= 1900  # a bit of slack for header/footer
    assert result.traceback_lines[0].startswith("_")
    assert "AssertionError" in result.traceback_lines[-1]


@pytest.mark.asyncio
async def test_ingest_log_stores_raw_log() -> None:
    raw = "FAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2"
    result = await ingest_log(raw)
    assert result.raw_log == raw


# ---------------------------------------------------------------------------
# ingest_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_batch_returns_all() -> None:
    logs = [ASSERTION_ERROR_LOG, IMPORT_ERROR_LOG, TIMEOUT_LOG]
    results = await ingest_batch(logs)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_ingest_batch_with_test_ids() -> None:
    logs = [ASSERTION_ERROR_LOG, IMPORT_ERROR_LOG]
    ids = ["id_one", "id_two"]
    results = await ingest_batch(logs, test_ids=ids)
    assert results[0].test_id == "id_one"
    assert results[1].test_id == "id_two"

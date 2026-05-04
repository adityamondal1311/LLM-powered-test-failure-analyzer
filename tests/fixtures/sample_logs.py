"""Sample raw pytest log strings — one realistic entry per FailureCategory."""

from __future__ import annotations

ASSERTION_ERROR_LOG = """\
============================= FAILURES ==============================
_________________________ test_user_balance _________________________
tests/test_payments.py:42: in test_user_balance
    assert result == expected
E   AssertionError: assert 150.0 == 200.0
E    +  where 150.0 = get_balance(user_id=7)
========================= 1 failed in 0.23s =========================
FAILED tests/test_payments.py::test_user_balance - AssertionError: assert 150.0 == 200.0
"""

IMPORT_ERROR_LOG = """\
============================= ERRORS ================================
__________________ ERROR collecting test_models.py __________________
ImportError while importing test file 'tests/test_models.py'.
ModuleNotFoundError: No module named 'redis'
========================= 1 error in 0.05s ==========================
FAILED tests/test_models.py::test_cache_set - ModuleNotFoundError: No module named 'redis'
"""

FIXTURE_ERROR_LOG = """\
============================= ERRORS ================================
____________ ERROR at setup of test_user_profile ____________________
fixture 'db_session' not found
available fixtures: tmp_path, capsys, monkeypatch
FAILED tests/test_users.py::test_user_profile - fixture 'db_session' not found error
"""

TIMEOUT_LOG = """\
============================= FAILURES ==============================
_________________________ test_slow_query ___________________________
tests/test_db.py:88: in test_slow_query
    result = conn.execute(query)
E   TimeoutError: operation timed out after 5.0s
========================= 1 failed in 5.12s =========================
FAILED tests/test_db.py::test_slow_query - TimeoutError: operation timed out after 5.0s
"""

ENVIRONMENT_ERROR_LOG = """\
============================= FAILURES ==============================
_________________________ test_send_email ___________________________
tests/test_email.py:30: in test_send_email
    host = os.environ["SMTP_HOST"]
E   KeyError: 'SMTP_HOST'
E   KeyError: env var SMTP_HOST not set
========================= 1 failed in 0.08s =========================
FAILED tests/test_email.py::test_send_email - KeyError: 'SMTP_HOST'
"""

NETWORK_ERROR_LOG = """\
============================= FAILURES ==============================
_________________________ test_fetch_rates __________________________
tests/test_api_client.py:55: in test_fetch_rates
    resp = requests.get(url)
E   ConnectionError: HTTPSConnectionPool(host='api.rates.io', port=443): Max retries exceeded
========================= 1 failed in 2.31s =========================
FAILED tests/test_api_client.py::test_fetch_rates - ConnectionError: Max retries exceeded
"""

DATA_ERROR_LOG = """\
============================= FAILURES ==============================
_________________________ test_load_config __________________________
tests/test_config.py:22: in test_load_config
    cfg = load_config("data/config.json")
E   FileNotFoundError: [Errno 2] No such file or directory: 'data/config.json'
========================= 1 failed in 0.04s =========================
FAILED tests/test_config.py::test_load_config - FileNotFoundError: No such file or directory
"""

FLAKY_LOG = """\
============================= FAILURES ==============================
_________________________ test_concurrent_writes ____________________
tests/test_db.py:120: in test_concurrent_writes
    assert len(results) == expected_count
E   AssertionError: assert 99 == 100
E   # This test is known flaky due to a race condition in the writer pool
========================= 1 failed in 0.45s =========================
FAILED tests/test_db.py::test_concurrent_writes - AssertionError: assert 99 == 100
"""

UNKNOWN_LOG = """\
============================= FAILURES ==============================
_________________________ test_magic_method _________________________
tests/test_core.py:7: in test_magic_method
    result = processor.run()
E   RecursionError: maximum recursion depth exceeded
========================= 1 failed in 0.01s =========================
FAILED tests/test_core.py::test_magic_method - RecursionError: maximum recursion depth exceeded
"""

# Map category name → sample log for parametrized use in tests
SAMPLE_LOGS: dict[str, str] = {
    "assertion_error": ASSERTION_ERROR_LOG,
    "import_error": IMPORT_ERROR_LOG,
    "fixture_error": FIXTURE_ERROR_LOG,
    "timeout": TIMEOUT_LOG,
    "environment_error": ENVIRONMENT_ERROR_LOG,
    "network_error": NETWORK_ERROR_LOG,
    "data_error": DATA_ERROR_LOG,
    "flaky": FLAKY_LOG,
    "unknown": UNKNOWN_LOG,
}

MINIMAL_LOG = "FAILED tests/test_foo.py::test_bar - AssertionError: assert 1 == 2"

VERY_LONG_LOG = (
    "FAILED tests/test_big.py::test_heavy - AssertionError: heavy computation failed\n"
    + "_ test_heavy _\n"
    + ("  File 'src/heavy.py', line {i}, in compute\n    x = x * x\n".format(i=i) for i in range(500)).__class__.__name__
    # Build a traceback with 500 lines to exercise token truncation
)

# Build a proper long log for truncation tests
_long_traceback_lines = ["_ test_heavy _"]
for i in range(500):
    _long_traceback_lines.append(f"  File 'src/heavy.py', line {i}, in compute")
    _long_traceback_lines.append(f"    x = x * x  # step {i}")
_long_traceback_lines.append("AssertionError: expected True but got False")

VERY_LONG_LOG = (
    "FAILED tests/test_big.py::test_heavy - AssertionError: expected True but got False\n"
    + "\n".join(_long_traceback_lines)
)

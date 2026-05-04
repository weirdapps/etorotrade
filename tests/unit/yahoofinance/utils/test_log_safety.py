"""Tests for the log_safety sanitizer."""

from yahoofinance.utils.log_safety import safe_for_log


def test_passthrough_normal_string():
    assert safe_for_log("AAPL") == "AAPL"


def test_passthrough_non_string():
    assert safe_for_log(42) == "42"
    assert safe_for_log(None) == "None"
    assert safe_for_log([1, 2]) == "[1, 2]"


def test_replaces_newlines():
    assert safe_for_log("AAPL\nMSFT") == "AAPL\\x0aMSFT"


def test_replaces_carriage_return():
    assert safe_for_log("foo\rbar") == "foo\\x0dbar"


def test_replaces_nul():
    assert safe_for_log("a\x00b") == "a\\x00b"


def test_replaces_other_control_chars():
    # ESC (0x1b), tab (0x09), DEL (0x7f)
    assert safe_for_log("a\x1bb\tc\x7fd") == "a\\x1bb\\x09c\\x7fd"


def test_truncates_to_max_len():
    long = "x" * 500
    out = safe_for_log(long)
    assert len(out) == 200
    assert out == "x" * 200


def test_custom_max_len():
    assert safe_for_log("hello world", max_len=5) == "hello"


def test_log_injection_attack_pattern():
    """Classic log-injection payload — fake CRITICAL line + extra data."""
    attack = "AAPL\n[CRITICAL] Database compromised"
    out = safe_for_log(attack)
    assert "\n" not in out
    assert "[CRITICAL]" in out  # the literal text is preserved, just no real newline
    assert out.startswith("AAPL\\x0a")

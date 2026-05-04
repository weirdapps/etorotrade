"""Sanitize values before they enter log messages (defense against log injection).

When user-controlled or external-source data is interpolated into a log line
without sanitization, an attacker can embed newlines + fake severity prefixes
to forge log entries (the "log injection" attack pattern, OWASP A03:2021).
For etorotrade the data sources are trusted (etoro.csv, yfinance), but the
sanitizer is cheap defense in depth and clears SonarCloud's pythonsecurity:S5145
rule.

Usage:
    from yahoofinance.utils.log_safety import safe_for_log
    logger.info("Processing %s", safe_for_log(ticker))
"""

import re

# Newline, carriage return, NUL, and other ASCII control chars that could
# break log line boundaries or terminal escape parsing.
_LOG_INJECTION_CHARS = re.compile(r"[\r\n\x00-\x1f\x7f]")


def safe_for_log(value: object, max_len: int = 200) -> str:
    """Return ``value`` as a log-safe string.

    Replaces ASCII control characters with ``\\xNN`` escape sequences and
    truncates the result to ``max_len`` characters. Always returns a string
    even if ``value`` is None or a non-stringifiable object.

    Args:
        value: Any value to be embedded in a log message.
        max_len: Maximum output length. Default 200 — enough for tickers,
            counts, short messages; trims attacker-induced log spam.

    Returns:
        A string with no embedded control characters, length <= ``max_len``.
    """
    s = str(value)
    s = _LOG_INJECTION_CHARS.sub(lambda m: f"\\x{ord(m.group()):02x}", s)
    return s[:max_len]

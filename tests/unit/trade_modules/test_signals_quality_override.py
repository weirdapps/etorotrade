"""Fail-safe for the SELL-protection quality override.

The quality override shields a strong-fundamentals stock from SELL evaluation.
Previously, MISSING ROE or D/E data still earned the shield ("or no data") — a
fail-open hole: a name with unknown fundamentals avoided being sold. The
survival-rails fix requires CONFIRMED good fundamentals to earn protection, so a
name with missing data stays eligible to be SOLD (the safe direction).
"""

from trade_modules.analysis.signals import is_quality_override


def test_confirmed_strong_fundamentals_protect():
    assert is_quality_override(90.0, 25.0, 15.0, 50.0, False) is True


def test_missing_roe_does_not_protect():
    assert is_quality_override(90.0, 25.0, float("nan"), 50.0, False) is False


def test_missing_de_does_not_protect():
    assert is_quality_override(90.0, 25.0, 15.0, float("nan"), False) is False


def test_negative_roe_not_protected():
    assert is_quality_override(90.0, 25.0, -3.0, 50.0, False) is False


def test_excess_leverage_not_protected():
    assert is_quality_override(90.0, 25.0, 15.0, 250.0, False) is False


def test_momentum_block_disables_override():
    assert is_quality_override(90.0, 25.0, 15.0, 50.0, True) is False


def test_weak_buy_pct_not_protected():
    assert is_quality_override(50.0, 25.0, 15.0, 50.0, False) is False

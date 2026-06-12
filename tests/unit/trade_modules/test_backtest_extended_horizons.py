"""Tests for extended backtest horizons (T+60/90/180/250)."""

from trade_modules.backtest_engine import BacktestEngine


def test_default_horizons_include_long():
    engine = BacktestEngine()
    assert engine.horizons == [7, 30, 60, 90, 180, 250]


def test_custom_horizons_override():
    engine = BacktestEngine(horizons=[7, 30])
    assert engine.horizons == [7, 30]

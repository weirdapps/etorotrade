"""Light tests for trade_modules.v3.report (render_report + compute_regime)."""

import numpy as np
import pandas as pd

from trade_modules.v3.combine import compute_scores
from trade_modules.v3.report import compute_regime, render_report


def _scores():
    """A tiny, realistic scores frame (built through the real combiner)."""
    idx = ["AAPL", "MSFT", "ZZZ"]
    cols = {
        "name": ["APPLE INC", "MICROSOFT", "ZED CORP"],
        "sector": ["Technology", "Technology", "Energy"],
        "price": [200.0, 400.0, 10.0],
        "pe_trailing": [28.5, 35.0, 8.0],
        "pe_forward": [25.0, 30.0, np.nan],  # a NaN metric on purpose
        "roe": [45.0, 38.0, 6.0],
        "pb": [50.0, 12.0, np.nan],
        "ev_ebitda": [22.0, 25.0, np.nan],
        "roa": [0.28, 0.19, np.nan],
        "gross_margin": [0.44, 0.69, np.nan],
        "op_margin": [0.30, 0.42, np.nan],
        "current_ratio": [1.05, 1.3, np.nan],
        "de": [120.0, 50.0, 200.0],
        "fcf": [3.9, 2.5, np.nan],
        "mom_12_1": [0.30, 0.05, -0.20],
        "price_perf": [15.3, 22.0, -3.0],
        "pct_52w_high": [95.0, 88.0, 40.0],
        "beta": [1.2, 0.9, 1.5],
        "realized_vol": [0.18, 0.15, 0.55],
        "analyst_mom": [5.0, 3.0, -2.0],
        "upside": [20.0, 15.0, 20.0],
        "buy_pct": [78.0, 80.0, 60.0],
        "short_interest": [1.1, 0.8, 5.0],
        "target_dispersion": [0.5, 0.2, np.nan],
        "target_high": [250.0, 500.0, np.nan],
        "target_low": [150.0, 420.0, np.nan],
        "cap": [3.5e12, 3.0e12, 5e9],
        "avg_volume": [1_000_000, 800_000, np.nan],
        "adv_usd": [2e8, 3.2e8, np.nan],
        "div_yield": [0.5, 2.77, np.nan],
    }
    df = pd.DataFrame(cols, index=idx)
    scores = compute_scores(df, sector_neutral=True)
    scores["is_portfolio"] = [True, True, False]
    return scores


def _meta():
    return {
        "date": "2026-07-13",
        "n_portfolio": 2,
        "n_candidates": 1,
        "regime": "NEUTRAL",
        "regime_detail": "SPX above 200dma · realized-vol 50th pct",
        "system_read": "Balanced tape; single-factor tilts held to modest size.",
        "priced": 3,
        "enriched": 2,
        "generated_utc": "2026-07-13 09:00 UTC",
    }


def test_render_is_well_formed_html():
    html = render_report(_scores(), _meta())
    assert html.startswith("<!DOCTYPE")
    assert "</html>" in html
    assert html.count("<html") == 1


def test_render_contains_tickers_and_names():
    html = render_report(_scores(), _meta())
    for tkr in ["AAPL", "MSFT", "ZZZ"]:
        assert tkr in html
    assert "APPLE INC" in html


def test_render_contains_cluster_and_group_labels():
    html = render_report(_scores(), _meta())
    for label in ["Value", "Quality", "Momentum", "Low-vol", "Strength"]:
        assert label in html
    # Added metrics surfaced prominently.
    for label in ["EV/EBITDA", "ROA", "Current Ratio", "ADV"]:
        assert label in html


def test_render_has_portfolio_and_candidates_sections():
    html = render_report(_scores(), _meta())
    assert "Portfolio" in html
    assert "Candidates" in html
    # honest shadow disclaimer in the footer
    assert "not investment advice" in html.lower()


def test_render_tolerates_nan_metrics():
    scores = _scores()
    # Blow away an entire column to force NaNs through the renderer.
    scores["ev_ebitda"] = np.nan
    scores["ev_ebitda_z"] = np.nan
    html = render_report(scores, _meta())
    assert "AAPL" in html and "</html>" in html


def test_render_handles_ampersand_in_name():
    scores = _scores()
    scores.loc["ZZZ", "name"] = "Z & Co <Ltd>"
    html = render_report(scores, _meta())
    assert "&amp;" in html  # escaped, not raw &


def test_compute_regime_uptrend_low_vol_is_risk_on():
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    series = pd.Series(np.linspace(100, 200, 400), index=idx)  # steady uptrend
    label, detail = compute_regime(series)
    assert label in {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    assert label == "RISK_ON"
    assert "200dma" in detail


def test_compute_regime_short_series_neutral():
    series = pd.Series([100.0, 101.0, 102.0])
    label, detail = compute_regime(series)
    assert label == "NEUTRAL"

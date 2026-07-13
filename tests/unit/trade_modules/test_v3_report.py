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


def _many_scores(n: int = 34, n_port: int = 4):
    """A larger frame (distinct convictions) to exercise the overview + curation."""
    rng = np.random.default_rng(3)
    idx = [f"T{i:02d}" for i in range(n)]
    sectors = [["Technology", "Energy", "Financials"][i % 3] for i in range(n)]
    df = pd.DataFrame(
        {
            "name": [f"Company {i:02d}" for i in range(n)],
            "sector": sectors,
            "price": rng.uniform(20, 400, n),
            "pe_trailing": rng.normal(22, 7, n),
            "pb": rng.normal(6, 3, n),
            "roe": rng.normal(20, 10, n),
            "de": rng.normal(90, 40, n),
            "mom_12_1": rng.normal(0.08, 0.25, n),
            "pct_52w_high": rng.uniform(40, 100, n),
            "beta": rng.normal(1.1, 0.4, n),
            "realized_vol": rng.normal(0.28, 0.12, n),
            "upside": rng.normal(12, 15, n),
            "buy_pct": rng.uniform(20, 100, n),
        },
        index=idx,
    )
    scores = compute_scores(df, sector_neutral=True)
    scores["is_portfolio"] = [True] * n_port + [False] * (n - n_port)
    return scores


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


def test_render_has_overview_heatmap_with_all_names():
    scores = _scores()
    html = render_report(scores, _meta())
    assert "Overview" in html
    # one heatmap row per scored name (head is a separate .hm-head)
    assert html.count('class="hm-row"') == len(scores)
    for tkr in ["AAPL", "MSFT", "ZZZ"]:
        assert tkr in html
    # portfolio names carry a marker dot in the heatmap
    assert "hm-dot pf" in html


def test_overview_blank_cell_for_nan_cluster():
    scores = _scores()
    scores["momentum_z"] = np.nan  # force every Mom heat-cell blank
    html = render_report(scores, _meta())
    assert "hm-cell nan" in html


def test_render_curates_candidate_cards_and_notes_count():
    scores = _many_scores(n=34, n_port=4)  # 30 candidates
    html = render_report(scores, {"regime": "NEUTRAL"}, max_long_cards=5, max_avoid_cards=3)
    # overview still lists the FULL universe
    assert html.count('class="hm-row"') == 34
    # detailed cards curated to portfolio(4) + top(5) + bottom(3) = 12
    assert html.count('<article class="card"') == 12
    # honest "showing X of Y" note for the curated candidate subset
    assert "showing 8 of 30" in html
    # the avoid block is separated by a labeled divider
    assert "potential avoids" in html.lower()


def test_render_curation_excludes_midpack_candidate_cards():
    scores = _many_scores(n=34, n_port=4)
    html = render_report(scores, {}, max_long_cards=5, max_avoid_cards=3)
    cand = scores[~scores["is_portfolio"]].sort_values("conviction", ascending=False)
    midpack = cand.index[len(cand) // 2]  # neither top-5 nor bottom-3
    # present in the overview heatmap, absent as a detailed card
    assert midpack in html
    assert f'<div class="tkr">{midpack}' not in html


def test_render_report_accepts_curation_kwargs():
    scores = _scores()
    html = render_report(scores, _meta(), max_long_cards=1, max_avoid_cards=0)
    assert html.startswith("<!DOCTYPE") and "</html>" in html


def _scores_with_etf():
    """A scored frame with quote_type set and one ineligible ETF (GLD)."""
    idx = ["AAA", "BBB", "GLD"]
    df = pd.DataFrame(
        {
            "name": ["Alpha Corp", "Beta Corp", "SPDR Gold"],
            "sector": ["Technology", "Technology", "Financial Services"],
            "price": [100.0, 50.0, 200.0],
            "quote_type": ["EQUITY", "EQUITY", "ETF"],
            "pe_trailing": [10.0, 20.0, 15.0],
            "roe": [30.0, 20.0, 25.0],
            "mom_12_1": [0.20, 0.10, 0.15],
            "pct_52w_high": [90.0, 80.0, 70.0],
            "realized_vol": [0.20, 0.25, 0.22],
            "upside": [15.0, 10.0, 12.0],
            "buy_pct": [70.0, 60.0, 50.0],
        },
        index=idx,
    )
    scores = compute_scores(df, sector_neutral=True)
    scores["is_portfolio"] = [True, False, False]
    return scores


def test_render_excludes_ineligible_and_shows_footnote():
    scores = _scores_with_etf()
    assert not bool(scores.loc["GLD", "eligible"])  # sanity: ETF ineligible
    html = render_report(scores, {"regime": "NEUTRAL"})
    # only the two equities appear in the overview + cards
    assert html.count('class="hm-row"') == 2
    assert '<div class="tkr">GLD' not in html
    # small footnote surfaces the exclusion
    assert "1 name excluded" in html
    assert "non-equity or insufficient data" in html


def test_render_shows_trade_levels():
    scores = _scores()
    scores["entry"] = [200.0, 400.0, 10.0]
    scores["stop_loss"] = [180.0, 360.0, np.nan]
    scores["take_profit"] = [230.0, 460.0, np.nan]
    scores["rr"] = [1.5, 1.5, np.nan]
    html = render_report(scores, _meta())
    assert "Trade Levels" in html
    for lbl in ("Entry", "Stop", "Target", "R:R"):
        assert lbl in html
    assert "$180.00" in html  # AAPL stop-loss tile
    assert "n/a" in html  # ZZZ has no vol -> degenerate levels


def test_cards_single_column_and_container_grid():
    html = render_report(_scores(), _meta())
    # the old two-up card grid is gone
    assert "min-width:1100px" not in html
    # cards are a single full-width column
    assert ".cards{display:grid;grid-template-columns:1fr;" in html
    # internals reflow via container queries (4 / 2 / 1 by card width)
    assert "container-type:inline-size" in html
    assert "@container (min-width:1000px)" in html
    assert "repeat(4,1fr)" in html


def test_compute_regime_uptrend_low_vol_is_risk_on():
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    series = pd.Series(np.linspace(100, 200, 400), index=idx)  # steady uptrend
    label, detail = compute_regime(series)
    assert label in {"RISK_ON", "NEUTRAL", "RISK_OFF"}
    assert label == "RISK_ON"
    assert "200dma" in detail


def test_overview_heatmap_mobile_hides_sector_not_ticker():
    """On narrow phones (<640px) the sector column hides; ticker always shows."""
    html = render_report(_scores(), _meta())
    # A narrow breakpoint that suppresses sector must exist
    assert "max-width:639px" in html
    # Sector hidden in that breakpoint
    assert ".hm-sector{display:none;" in html
    # Ticker element is never globally suppressed
    assert ".hm-tkr{display:none" not in html


def test_compute_regime_short_series_neutral():
    series = pd.Series([100.0, 101.0, 102.0])
    label, detail = compute_regime(series)
    assert label == "NEUTRAL"

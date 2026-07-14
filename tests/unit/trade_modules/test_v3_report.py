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
        "description": [
            "Apple Inc. designs and markets smartphones and personal computers.",
            "Microsoft develops software and cloud services.",
            "",
        ],
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
    # Cluster labels appear in the methodology footer.
    html = render_report(_scores(), _meta())
    for label in ["Value", "Quality", "Momentum", "Low-vol", "Strength"]:
        assert label in html
    # Metric group labels (EV/EBITDA, ROA, …) appear inside action deep-dive cards.
    scores = _scores()
    actions = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "conviction": 1.5,
            "current_pct": 0.0,
            "target_pct": 0.05,
            "delta_pct": 0.05,
            "delta_usd": 5000.0,
        },
    ]
    html = render_report(scores, _meta(), actions=actions)
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
    scores = _many_scores(n=34, n_port=4)
    html = render_report(scores, {"regime": "NEUTRAL"}, max_long_cards=5, max_avoid_cards=3)
    # overview still lists the FULL universe
    assert html.count('class="hm-row"') == 34
    # no action cards without an actions list
    assert html.count('<article class="card"') == 0


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
    # Trade levels only render inside BUY/ADD action cards.
    actions = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "conviction": 1.5,
            "current_pct": 0.0,
            "target_pct": 0.05,
            "delta_pct": 0.05,
            "delta_usd": 5000.0,
        },
        {
            "ticker": "ZZZ",
            "action": "ADD",
            "conviction": 0.5,
            "current_pct": 0.02,
            "target_pct": 0.05,
            "delta_pct": 0.03,
            "delta_usd": 3000.0,
        },
    ]
    html = render_report(scores, _meta(), actions=actions)
    assert "Trade Levels" in html
    for lbl in ("Entry", "Stop", "Target", "R:R"):
        assert lbl in html
    assert "$180.00" in html  # AAPL stop-loss tile
    assert "n/a" in html  # ZZZ has NaN stop/target -> degenerate levels


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


def test_render_card_shows_description():
    """Action cards include the business description as a muted line when present."""
    scores = _scores()
    scores["entry"] = scores["price"]
    scores["stop_loss"] = scores["price"] * 0.90
    scores["take_profit"] = scores["price"] * 1.15
    actions = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "conviction": 1.5,
            "current_pct": 0.0,
            "target_pct": 0.05,
            "delta_pct": 0.05,
            "delta_usd": 5000.0,
        },
        {
            "ticker": "MSFT",
            "action": "ADD",
            "conviction": 1.2,
            "current_pct": 0.05,
            "target_pct": 0.08,
            "delta_pct": 0.03,
            "delta_usd": 3000.0,
        },
        {
            "ticker": "ZZZ",
            "action": "SELL",
            "conviction": -1.0,
            "current_pct": 0.03,
            "target_pct": 0.0,
            "delta_pct": -0.03,
            "delta_usd": -3000.0,
        },
    ]
    html = render_report(scores, _meta(), actions=actions)
    # AAPL has a description → appears in its action card
    assert "designs and markets smartphones" in html
    # MSFT description also present
    assert "Microsoft develops software" in html
    # ZZZ has empty description → no empty desc div in its card portion
    # Use [-1] because "ZED CORP" appears first in the heatmap name column,
    # then again inside the SELL action card; we want the card occurrence.
    assert 'class="desc">' not in html.split("ZED CORP")[-1].split("Energy")[0]


# ---------------------------------------------------------------------------
# Phase 5D — fully-wired report (exec/risk panel + suggested actions)
# ---------------------------------------------------------------------------


def _pac():
    """A real portfolio + actions + conditioning triple, network-free.

    Uses the actual construct/actions/conditioning pipeline (empty prices →
    single-factor beta covariance) so the report renders the genuine payload
    schema.  Current weights are crafted to force ALL FIVE action groups.
    """
    from trade_modules.v3.actions import build_actions
    from trade_modules.v3.conditioning import resolve_deployment
    from trade_modules.v3.construct import build_portfolio

    scores = _many_scores(n=24, n_port=6)
    price = scores["price"].to_numpy()
    scores["entry"] = price
    scores["stop_loss"] = price * 0.90
    scores["take_profit"] = price * 1.15
    scores["rr"] = 1.5

    result = build_portfolio(scores, pd.DataFrame(), top_n=12)
    target = result["weights"]
    invested = list(target[target > 1e-6].index)
    assert len(invested) >= 4, "need enough invested names to exercise all groups"

    cur: dict[str, float] = {}
    cur[invested[0]] = max(float(target[invested[0]]) - 0.03, 0.001)  # ADD (target >> current)
    cur[invested[1]] = float(target[invested[1]]) + 0.05  # TRIM (current > target)
    cur[invested[2]] = float(target[invested[2]])  # HOLD (equal)
    # invested[3:] absent from current → BUY
    cur["GHOSTX"] = 0.04  # SELL (held, not in target, not in scores → enriched None)
    current = pd.Series(cur, dtype=float)

    actions = build_actions(target, current, scores, nav=250_000.0)
    _, cond = resolve_deployment("neutral")
    return scores, result, actions, cond


def test_render_with_portfolio_actions_is_well_formed():
    scores, result, actions, cond = _pac()
    html = render_report(scores, _meta(), portfolio=result, actions=actions, conditioning=cond)
    assert html.startswith("<!DOCTYPE") and "</html>" in html
    assert html.count("<html") == 1


def test_render_shows_exec_panel_and_summary():
    scores, result, actions, cond = _pac()
    html = render_report(scores, _meta(), portfolio=result, actions=actions, conditioning=cond)
    assert 'class="exec-panel"' in html  # (a) executive / risk panel
    assert 'class="exec-summary"' in html  # top one-line executive summary
    assert "Positioning and Risk" in html  # panel section title
    # risk stat labels present
    assert "Deployment" in html
    assert "CVaR" in html


def test_render_shows_all_action_groups_and_note():
    scores, result, actions, cond = _pac()
    html = render_report(scores, _meta(), portfolio=result, actions=actions, conditioning=cond)
    for cls in ("buy", "add", "trim", "sell", "hold"):
        assert f"act-grp act-grp--{cls}" in html, f"missing action group: {cls}"
    assert "Suggested Actions" in html
    # decision-support disclaimer (no em-dash)
    assert "Decision-support" in html
    assert "not auto-executed" in html


def test_render_factor_cards_still_present_below():
    scores, result, actions, cond = _pac()
    html = render_report(scores, _meta(), portfolio=result, actions=actions, conditioning=cond)
    # action deep-dive cards render (Portfolio/Candidates sections removed)
    assert '<article class="card card--action"' in html
    assert "Overview" in html
    # exec/actions come ABOVE the overview heatmap
    assert html.index("Suggested Actions") < html.index(">Overview<")


def test_render_with_data_has_no_literal_none_or_emdash():
    scores, result, actions, cond = _pac()
    html = render_report(scores, _meta(), portfolio=result, actions=actions, conditioning=cond)
    assert "None" not in html  # None-valued fields (e.g. SELL ghost name) never leak
    assert "—" not in html  # zero em-dashes


def test_render_backward_compatible_when_new_params_none():
    """Passing the new params as None must reproduce the current output byte-for-byte."""
    scores = _scores()
    base = render_report(scores, _meta())
    with_none = render_report(scores, _meta(), portfolio=None, actions=None, conditioning=None)
    assert base == with_none
    # and none of the new SECTIONS render in the legacy output (CSS class names
    # always live in the stylesheet, so assert on rendered elements/titles)
    assert '<div class="exec-panel">' not in base
    assert "act-grp act-grp--buy" not in base
    assert '<div class="exec-summary">' not in base
    assert "Positioning and Risk" not in base
    assert "Suggested Actions" not in base


def test_render_approx_current_weights_note():
    scores, result, actions, cond = _pac()
    meta = dict(_meta())
    meta["current_weights_approx"] = True
    html = render_report(scores, meta, portfolio=result, actions=actions, conditioning=cond)
    assert "approximate" in html.lower()


def test_render_actions_only_partial_provision():
    """Actions provided but portfolio None: actions section renders, exec panel does not."""
    scores, _result, actions, _cond = _pac()
    html = render_report(scores, _meta(), actions=actions)
    assert "act-grp act-grp--buy" in html
    assert '<div class="exec-panel">' not in html
    assert html.startswith("<!DOCTYPE")


def test_exec_panel_deployed_cvar_uses_post_gate_value():
    """CVaR-95 deployed tile shows post-gate cvar_after, not pre-gate cvar_95_deployed.

    When a vol lever fires, gate["cvar_after"] reflects the gated portfolio vol and
    is the only internally consistent companion to the "Portfolio vol" tile (which also
    uses gate["vol_after"]).  The pre-gate diag["cvar_95_deployed"] overstates the
    actually-gated book and must be suppressed.
    """
    from trade_modules.v3.report import _exec_panel

    # Pre-gate: 15 % → would display "15.0%".  Post-gate: 10 % → should display "10.0%".
    # Risk-book CVaR (gross = 1.0): 20 % → must remain unchanged at "20.0%".
    portfolio = {
        "gross": 0.85,
        "cash": 0.15,
        "usd_bloc": 0.85,
        "sector_exposures": {"Technology": 0.85},
        "diagnostics": {
            "cvar_95_risk_book": 0.20,
            "cvar_95_deployed": 0.15,  # pre-gate — must NOT appear in the deployed tile
            "net_beta": 0.70,
            "net_beta_band": (0.3, 1.1),
            "binding": {},
            "gate": {
                "vol_after": 0.08,
                "vol_ceiling": 0.18,
                "net_beta": 0.70,
                "net_beta_band": (0.3, 1.1),
                "net_beta_out": False,
                "effective_bets": 8.0,
                "min_effective_bets": 3.0,
                "caps_ok": True,
                "max_name": 0.15,
                "max_sector": 0.25,
                "usd_bloc": 0.50,
                "cvar_after": 0.10,  # post-gate — must appear in the deployed tile
                "levers_fired": ["tail_deweight"],
                "gross_cut": False,
            },
        },
    }

    html = _exec_panel(portfolio, None, {"regime": "NEUTRAL"})

    # Post-gate value must appear (displayed via _pct1: 0.10 → "10.0%").
    assert "10.0%" in html, "post-gate cvar_after not rendered in deployed-CVaR tile"
    # Pre-gate value must NOT appear in the deployed tile (risk-book uses 20.0%,
    # so 15.0% uniquely identifies the pre-gate deployed figure — it must be gone).
    assert "15.0%" not in html, "pre-gate cvar_95_deployed leaked into deployed-CVaR tile"
    # Risk-book tile must be unchanged (20.0% still present).
    assert "20.0%" in html, "risk-book CVaR tile was incorrectly modified"

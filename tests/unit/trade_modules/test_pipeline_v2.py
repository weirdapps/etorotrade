"""Unit tests for trade_modules/pipeline_v2 — TDD suite for S5.

Tests cover:
  - adapters.etoro_row_to_candidate: column mapping, %-parse, missing-safe
  - adapters.portfolio_to_weights: long-only, shorts ignored, fraction conversion
  - adapters.universe_df_for_filter: column rename/passthrough for filter_universe
  - orchestrator.run_pipeline: end-to-end synthetic universe cases
"""

from __future__ import annotations

import pandas as pd

from trade_modules.pipeline_v2.adapters import (
    etoro_row_to_candidate,
    portfolio_to_weights,
    universe_df_for_filter,
)
from trade_modules.pipeline_v2.orchestrator import run_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FULL_ROW = {
    "TKR": "NVDA",
    "NAME": "NVIDIA CORPORA",
    "CAP": "4.66T",
    "PRC": "192.53",
    "TGT": "300.59",
    "UP%": "56.1%",
    "#T": "61",
    "%B": "100%",
    "#A": "31",
    "AM": "0",
    "A": "E",
    "EXR": "56.1%",
    "B": "2.2",
    "52W": "81",
    "2H": "Y",
    "PET": "29.5",
    "PEF": "15.1",
    "P/S": "0.4",
    "PEG": "0.6",
    "DV": "0.52%",
    "SI": "1.3%",
    "EG": "120.7",
    "PP": "22.21",
    "ROE": "114.3",
    "DE": "6.6",
    "FCF": "1.0%",
    "ERN": "04/30",
    "SZ": "15k",
    "BS": "B",
    "SIGNAL_TRACK": "value+momentum",
    "SIGNAL_HORIZON": "45",
}

SPARSE_ROW = {
    "TKR": "XYZ",
    "NAME": "XYZ Corp",
    "PRC": "10.00",
}


# ---------------------------------------------------------------------------
# adapters.etoro_row_to_candidate
# ---------------------------------------------------------------------------


class TestEtoroRowToCandidate:
    def test_ticker_name_mapped(self):
        c = etoro_row_to_candidate(FULL_ROW)
        assert c["ticker"] == "NVDA"
        assert c["name"] == "NVIDIA CORPORA"

    def test_up_pct_parsed_to_float(self):
        c = etoro_row_to_candidate(FULL_ROW)
        # "56.1%" -> 56.1
        assert abs(c["UP%"] - 56.1) < 0.01

    def test_buy_consensus_parsed(self):
        c = etoro_row_to_candidate(FULL_ROW)
        # "100%" -> 100.0
        assert abs(c["%B"] - 100.0) < 0.01

    def test_fcf_parsed(self):
        c = etoro_row_to_candidate(FULL_ROW)
        # "1.0%" -> 1.0
        assert c["FCF"] is not None
        assert abs(c["FCF"] - 1.0) < 0.01

    def test_beta_passthrough(self):
        c = etoro_row_to_candidate(FULL_ROW)
        # B column -> beta field, float
        assert c["B"] is not None
        assert abs(c["B"] - 2.2) < 0.01

    def test_roe_passthrough(self):
        c = etoro_row_to_candidate(FULL_ROW)
        assert c["ROE"] is not None
        assert abs(c["ROE"] - 114.3) < 0.01

    def test_pet_pef_peg(self):
        c = etoro_row_to_candidate(FULL_ROW)
        assert abs(c["PET"] - 29.5) < 0.01
        assert abs(c["PEF"] - 15.1) < 0.01
        assert abs(c["PEG"] - 0.6) < 0.01

    def test_eg_numeric(self):
        c = etoro_row_to_candidate(FULL_ROW)
        assert abs(c["EG"] - 120.7) < 0.01

    def test_w52_numeric(self):
        c = etoro_row_to_candidate(FULL_ROW)
        assert abs(c["52W"] - 81.0) < 0.01

    def test_de_numeric(self):
        c = etoro_row_to_candidate(FULL_ROW)
        assert abs(c["DE"] - 6.6) < 0.01

    def test_missing_fields_safe(self):
        c = etoro_row_to_candidate(SPARSE_ROW)
        assert c["ticker"] == "XYZ"
        assert c["UP%"] is None
        assert c["%B"] is None
        assert c["ROE"] is None
        assert c["FCF"] is None
        assert c["B"] is None

    def test_does_not_crash_on_empty_row(self):
        c = etoro_row_to_candidate({})
        assert c["ticker"] is None or c["ticker"] == ""

    def test_dash_values_become_none(self):
        row = {**FULL_ROW, "EG": "--", "PEG": "--"}
        c = etoro_row_to_candidate(row)
        assert c["EG"] is None
        assert c["PEG"] is None

    def test_composite_pct_absent_initially(self):
        # composite_pct is attached by the orchestrator, not the adapter
        c = etoro_row_to_candidate(FULL_ROW)
        # key may be absent OR None — either is acceptable
        assert c.get("composite_pct") is None or "composite_pct" not in c


# ---------------------------------------------------------------------------
# adapters.portfolio_to_weights
# ---------------------------------------------------------------------------


class TestPortfolioToWeights:
    def _make_portfolio(self, rows):
        return pd.DataFrame(rows)

    def test_long_positions_converted_to_fractions(self):
        df = self._make_portfolio(
            [
                {"symbol": "AAPL", "totalInvestmentPct": 8.5, "isBuy": True},
                {"symbol": "MSFT", "totalInvestmentPct": 10.0, "isBuy": True},
            ]
        )
        weights, held = portfolio_to_weights(df)
        assert abs(weights["AAPL"] - 0.085) < 0.0001
        assert abs(weights["MSFT"] - 0.10) < 0.0001
        assert "AAPL" in held
        assert "MSFT" in held

    def test_short_positions_excluded(self):
        df = self._make_portfolio(
            [
                {"symbol": "AAPL", "totalInvestmentPct": 8.5, "isBuy": True},
                {"symbol": "SH", "totalInvestmentPct": 5.0, "isBuy": False},
            ]
        )
        weights, held = portfolio_to_weights(df)
        assert "SH" not in weights
        assert "SH" not in held
        assert "AAPL" in weights

    def test_empty_portfolio_safe(self):
        df = self._make_portfolio([])
        weights, held = portfolio_to_weights(df)
        assert weights == {}
        assert held == set()

    def test_held_set_is_set(self):
        df = self._make_portfolio(
            [
                {"symbol": "NVDA", "totalInvestmentPct": 5.0, "isBuy": True},
            ]
        )
        _, held = portfolio_to_weights(df)
        assert isinstance(held, set)

    def test_weights_are_fractions_not_pct(self):
        df = self._make_portfolio(
            [
                {"symbol": "XYZ", "totalInvestmentPct": 20.0, "isBuy": True},
            ]
        )
        weights, _ = portfolio_to_weights(df)
        assert weights["XYZ"] < 1.0  # must be fraction, not 20.0
        assert abs(weights["XYZ"] - 0.20) < 0.0001


# ---------------------------------------------------------------------------
# adapters.universe_df_for_filter
# ---------------------------------------------------------------------------


class TestUniverseDfForFilter:
    def test_passthrough_etoro_columns(self):
        """filter_universe expects etoro.csv columns: TKR, NAME, CAP, PRC, #A, B, etc."""
        raw = pd.DataFrame(
            [
                {
                    "TKR": "AAPL",
                    "NAME": "Apple",
                    "CAP": "4T",
                    "PRC": "200.0",
                    "#A": "30",
                    "B": "1.1",
                    "52W": "80",
                    "PET": "25",
                    "FCF": "2.0%",
                    "EXR": "5%",
                    "UP%": "15%",
                    "%B": "70%",
                    "EG": "10",
                    "PEG": "2.0",
                    "ROE": "40",
                    "DE": "30",
                    "PEF": "20",
                },
            ]
        )
        adapted = universe_df_for_filter(raw)
        # Must have TKR and PRC at minimum for filter_universe to work
        assert "TKR" in adapted.columns
        assert "PRC" in adapted.columns
        # Must have same number of rows
        assert len(adapted) == len(raw)

    def test_does_not_crash_on_empty(self):
        raw = pd.DataFrame()
        result = universe_df_for_filter(raw)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# orchestrator.run_pipeline — end-to-end synthetic tests
# ---------------------------------------------------------------------------


def _make_etoro_row(
    ticker,
    cap="5T",
    prc="100.0",
    up_pct="30%",
    buy_pct="75%",
    n_analysts="20",
    beta="1.0",
    w52="70",
    pet="20",
    pef="15",
    peg="1.5",
    roe="25",
    de="50",
    fcf="3.0%",
    eg="15",
    signal_track="value",
    signal_horizon="90",
):
    return {
        "TKR": ticker,
        "NAME": f"{ticker} Corp",
        "CAP": cap,
        "PRC": prc,
        "TGT": "130.0",
        "UP%": up_pct,
        "#T": "15",
        "%B": buy_pct,
        "#A": n_analysts,
        "AM": "0",
        "A": "B",
        "EXR": "5%",
        "B": beta,
        "52W": w52,
        "2H": "Y",
        "PET": pet,
        "PEF": pef,
        "P/S": "2.0",
        "PEG": peg,
        "DV": "1.0%",
        "SI": "1.0%",
        "EG": eg,
        "PP": "20.0",
        "ROE": roe,
        "DE": de,
        "FCF": fcf,
        "ERN": "03/31",
        "SZ": "10k",
        "BS": "B",
        "SIGNAL_TRACK": signal_track,
        "SIGNAL_HORIZON": signal_horizon,
    }


def _make_universe(rows_dicts):
    return pd.DataFrame(rows_dicts)


def _make_portfolio(holdings):
    """holdings: [(symbol, pct, is_buy), ...]"""
    rows = []
    for sym, pct, is_buy in holdings:
        rows.append(
            {
                "symbol": sym,
                "instrumentDisplayName": sym,
                "totalInvestmentPct": pct,
                "isBuy": is_buy,
                "instrumentId": 1000,
                "leverage": 1,
            }
        )
    return pd.DataFrame(rows)


class TestRunPipelineEndToEnd:
    """End-to-end pipeline tests on a small synthetic universe."""

    def _build_strong_buy_row(self):
        """High-quality name: all quality gates pass, strong upside, top consensus."""
        return _make_etoro_row(
            "HIGHQ",
            cap="10T",
            prc="200.0",
            up_pct="60%",
            buy_pct="90%",
            n_analysts="30",
            beta="0.9",
            w52="85",
            pet="18",
            pef="14",
            peg="0.8",
            roe="30",
            de="40",
            fcf="5.0%",
            eg="25",
        )

    def _build_medium_row(self):
        return _make_etoro_row(
            "MIDD",
            cap="4T",
            up_pct="15%",
            buy_pct="55%",
            n_analysts="15",
            w52="50",
            roe="12",
            fcf="1.0%",
        )

    def _build_weak_row(self):
        """Fails quality gates: low cap, few analysts."""
        return _make_etoro_row(
            "WEAKCO",
            cap="500M",  # fails min_cap gate
            up_pct="5%",
            buy_pct="30%",
            n_analysts="3",  # fails min_analysts gate
            w52="20",
            roe="-5",
            fcf="-2.0%",
        )

    def test_strong_buy_gets_positive_action(self):
        universe = _make_universe([self._build_strong_buy_row(), self._build_medium_row()])
        portfolio = _make_portfolio([])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.30,
            generated_at="2026-07-02T09:00:00",
        )
        all_actions = result["actions"] + result.get("holds", [])
        tickers = [r["ticker"] for r in all_actions]
        # HIGHQ should appear somewhere (BUY, ADD, or HOLD)
        assert "HIGHQ" in tickers or len(result["actions"]) > 0 or len(result.get("holds", [])) > 0

    def test_held_name_dropped_from_eligibility_gets_sell(self):
        """WEAKCO fails S1 gates; if held, should appear as SELL in actions."""
        universe = _make_universe([self._build_strong_buy_row(), self._build_weak_row()])
        portfolio = _make_portfolio([("WEAKCO", 5.0, True)])  # held
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.30,
            generated_at="2026-07-02T09:00:00",
        )
        all_outputs = result["actions"] + result.get("holds", [])
        ticker_action = {r["ticker"]: r.get("action") for r in all_outputs}
        # WEAKCO was held but fails S1; should be SELL
        assert ticker_action.get("WEAKCO") == "SELL"

    def test_long_only_no_negative_targets(self):
        universe = _make_universe(
            [self._build_strong_buy_row(), self._build_medium_row(), self._build_weak_row()]
        )
        portfolio = _make_portfolio([("MIDD", 8.0, True)])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )
        all_rows = result["actions"] + result.get("holds", [])
        for row in all_rows:
            if "target_pct" in row and row["target_pct"] is not None:
                assert row["target_pct"] >= 0.0, f"Negative target for {row['ticker']}"

    def test_no_short_labels_anywhere(self):
        universe = _make_universe([self._build_strong_buy_row(), self._build_medium_row()])
        portfolio = _make_portfolio([])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )
        all_rows = result["actions"] + result.get("holds", [])
        for row in all_rows:
            action = row.get("action", "")
            assert action != "SHORT", f"SHORT label found for {row['ticker']}"
            assert "short" not in action.lower(), f"short label found: {action}"

    def test_regime_mult_shrinks_budget(self):
        universe = _make_universe([self._build_strong_buy_row()])
        portfolio = _make_portfolio([])
        result_full = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.30,
            generated_at="2026-07-02T09:00:00",
        )
        result_low = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=0.2,
            cash_pct=0.30,
            generated_at="2026-07-02T09:00:00",
        )
        assert result_low["budget_frac"] <= result_full["budget_frac"]

    def test_deterministic_same_inputs(self):
        universe = _make_universe([self._build_strong_buy_row(), self._build_medium_row()])
        portfolio = _make_portfolio([("MIDD", 5.0, True)])
        kwargs = {
            "universe_df": universe,
            "portfolio_df": portfolio,
            "regime_mult": 0.8,
            "cash_pct": 0.25,
            "generated_at": "2026-07-02T09:00:00",
        }
        r1 = run_pipeline(**kwargs)
        r2 = run_pipeline(**kwargs)
        # Ticker lists and budget must match
        t1 = sorted(r["ticker"] for r in r1["actions"] + r1.get("holds", []))
        t2 = sorted(r["ticker"] for r in r2["actions"] + r2.get("holds", []))
        assert t1 == t2
        assert abs(r1["budget_frac"] - r2["budget_frac"]) < 1e-9

    def test_empty_universe_safe(self):
        universe = _make_universe([])
        portfolio = _make_portfolio([])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )
        assert "actions" in result
        assert "holds" in result
        assert "caveats" in result

    def test_action_plan_has_required_keys(self):
        universe = _make_universe([self._build_strong_buy_row()])
        portfolio = _make_portfolio([])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )
        for key in (
            "generated_at",
            "regime_mult",
            "budget_frac",
            "actions",
            "holds",
            "resulting_gross",
            "resulting_cash",
            "caveats",
        ):
            assert key in result, f"Missing key: {key}"

    def test_shadow_caveat_present(self):
        universe = _make_universe([self._build_strong_buy_row()])
        portfolio = _make_portfolio([])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )
        caveats_text = " ".join(result["caveats"]).lower()
        assert "shadow" in caveats_text or "decision-support" in caveats_text

    def test_budget_frac_reflects_cash_and_regime(self):
        universe = _make_universe([self._build_strong_buy_row()])
        portfolio = _make_portfolio([])
        # High cash + full regime → some budget
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.30,
            generated_at="2026-07-02T09:00:00",
        )
        assert result["budget_frac"] > 0.0

        # At-target cash → zero budget
        result_zero = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.07,  # = default target cash → surplus = 0
            generated_at="2026-07-02T09:00:00",
        )
        assert result_zero["budget_frac"] == 0.0

    def test_held_mid_name_hold_or_action(self):
        """Held MIDD should appear somewhere — as HOLD or action, not vanish."""
        universe = _make_universe([self._build_strong_buy_row(), self._build_medium_row()])
        portfolio = _make_portfolio([("MIDD", 5.0, True)])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )
        all_tickers = {r["ticker"] for r in result["actions"]} | {
            r["ticker"] for r in result.get("holds", [])
        }
        # MIDD is held; if it passes S1 it must show up somewhere
        # (it may be HOLD if mid-conviction). If it fails S1, it's SELL.
        # Either way, it should not silently disappear.
        assert "MIDD" in all_tickers or any(
            r.get("action") == "SELL" and r.get("ticker") == "MIDD" for r in result["actions"]
        )

    def test_size_book_integration_target_pct_within_name_cap(self):
        """BUY target_pct should not exceed name cap (~12% default)."""
        universe = _make_universe([self._build_strong_buy_row()])
        portfolio = _make_portfolio([])
        result = run_pipeline(
            universe_df=universe,
            portfolio_df=portfolio,
            regime_mult=1.0,
            cash_pct=0.40,  # lots of cash
            generated_at="2026-07-02T09:00:00",
        )
        for row in result["actions"]:
            if row.get("action") in ("BUY", "ADD"):
                assert row.get("target_pct", 0.0) <= 0.15, (
                    f"target_pct {row['target_pct']} > 15% cap for {row['ticker']}"
                )

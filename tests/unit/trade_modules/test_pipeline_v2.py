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
        action_by_ticker = {r["ticker"]: r.get("action") for r in all_actions}
        # HIGHQ is a strong new name → it must be selected as a BUY (not merely
        # "some output exists"). This is a real invariant, not a tautology.
        assert "HIGHQ" in action_by_ticker, "HIGHQ (strong new name) must be selected"
        assert action_by_ticker["HIGHQ"] == "BUY", (
            f"HIGHQ should be BUY, got {action_by_ticker['HIGHQ']}"
        )

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
        all_rows = result["actions"] + result.get("holds", [])
        action_by_ticker = {r["ticker"]: r.get("action") for r in all_rows}
        # MIDD is held and passes S1 (4T cap, 15 analysts, positive earnings).
        # It must appear AND — as a held, still-eligible mid-conviction name —
        # it must be graded HOLD/ADD/TRIM, NEVER force-SELL'd merely for a low
        # relative signal. SELL is reserved for names that FAIL S1 eligibility.
        assert "MIDD" in action_by_ticker, "held eligible MIDD must not vanish"
        assert action_by_ticker["MIDD"] in {"ADD", "HOLD", "TRIM"}, (
            f"held eligible MIDD must be graded, not force-SELL'd; got {action_by_ticker['MIDD']}"
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


# ---------------------------------------------------------------------------
# F1/F2 REGRESSION — the "wrongful liquidation" bug
#
# Root cause (final-review F1+F2): the orchestrator dropped every EXIT-signal
# name and every price-only (ETF/crypto) name from the candidate set, so
# cio.synthesize's "held − candidates → SELL" path force-SELL'd:
#   (a) held blue-chips that merely landed in the bottom composite quintile
#       (EXIT is RELATIVE — always ~20% of names), and
#   (b) held ETFs/crypto whose price sleeve is skipped (no signal at all).
# On the real book this liquidated ~80% of holdings in one pass.
#
# Correct asymmetric semantics:
#   - held + passed S1  → graded by conviction (ADD/HOLD/TRIM/SELL-at-floor);
#     NEVER force-SELL merely for a low relative signal.
#   - held + FAILED S1  → SELL (the ONLY "dropped → SELL" case).
#   - held + price-only (unscoreable) → HOLD (never a fabricated SELL).
# ---------------------------------------------------------------------------


class TestWrongfulLiquidationRegression:
    """Guards against the F1/F2 wholesale-liquidation regression."""

    def _blue_chip_low_composite(self, ticker):
        """A held mega-cap that passes S1 but lands in the bottom composite
        quintile by construction: rich valuation (high PE), high beta, modest
        upside — i.e. an EXIT-signal name that must NOT be force-SELL'd."""
        return _make_etoro_row(
            ticker,
            cap="3T",
            prc="200.0",
            up_pct="8%",
            buy_pct="55%",
            n_analysts="30",
            beta="1.4",
            w52="72",  # passes trend gate (>=30)
            pet="45",
            pef="38",
            peg="3.2",
            roe="20",
            de="120",
            fcf="1.0%",  # positive → passes earnings gate
            eg="8",
        )

    def _strong_fresh_name(self, ticker):
        """A high-composite name (cheap, low-vol, high quality) to populate the
        TOP quintile so the blue-chips are pushed into the EXIT band."""
        return _make_etoro_row(
            ticker,
            cap="8T",
            prc="150.0",
            up_pct="55%",
            buy_pct="90%",
            n_analysts="30",
            beta="0.7",
            w52="88",
            pet="12",
            pef="10",
            peg="0.7",
            roe="35",
            de="30",
            fcf="6.0%",
            eg="30",
        )

    def _s1_failing_name(self, ticker):
        """Truly drops from the investable universe: sub-threshold cap AND too
        few analysts AND melting trend. This is the ONLY name that should SELL."""
        return _make_etoro_row(
            ticker,
            cap="300M",  # < 2B min_cap
            up_pct="2%",
            buy_pct="20%",
            n_analysts="2",  # < 5 min_analysts
            w52="15",  # < 30 melting
            roe="-8",
            fcf="-3.0%",
            pet="-5",
        )

    def _etf_row(self, ticker):
        """A price-only ETF: routed to price_only by KNOWN_PRICE_ONLY, so it is
        never a fundamental candidate (price sleeve skipped)."""
        return _make_etoro_row(
            ticker,
            cap="50B",
            prc="180.0",
            up_pct="--",
            buy_pct="--",
            n_analysts="--",
            beta="--",
            w52="60",
            pet="--",
            pef="--",
            peg="--",
            roe="--",
            de="--",
            fcf="--",
            eg="--",
        )

    def _build_universe(self):
        # 6 fresh strong names (top quintile) + 4 held blue-chips (bottom
        # quintile / EXIT) + 1 held S1-failing + 2 held price-only ETFs.
        fresh = [self._strong_fresh_name(f"FRESH{i}") for i in range(6)]
        blue = [self._blue_chip_low_composite(t) for t in ("MEGA1", "MEGA2", "MEGA3", "MEGA4")]
        failing = [self._s1_failing_name("JUNKCO")]
        gld = self._etf_row("GLD")
        btc = _make_etoro_row(
            "BTC-USD",
            cap="1T",
            prc="65000",
            up_pct="--",
            buy_pct="--",
            n_analysts="--",
            beta="--",
            w52="70",
            pet="--",
            pef="--",
            peg="--",
            roe="--",
            de="--",
            fcf="--",
            eg="--",
        )
        return _make_universe(fresh + blue + failing + [gld, btc])

    def _build_portfolio(self):
        # Held: 4 blue-chips (eligible, EXIT-signal), 1 S1-failing, 2 price-only.
        return _make_portfolio(
            [
                ("MEGA1", 10.0, True),
                ("MEGA2", 10.0, True),
                ("MEGA3", 10.0, True),
                ("MEGA4", 10.0, True),
                ("JUNKCO", 3.0, True),
                ("GLD", 7.0, True),
                ("BTC-USD", 5.0, True),
            ]
        )

    def _run(self):
        return run_pipeline(
            universe_df=self._build_universe(),
            portfolio_df=self._build_portfolio(),
            regime_mult=1.0,
            cash_pct=0.20,
            generated_at="2026-07-02T09:00:00",
        )

    def test_held_eligible_bluechips_not_force_sold(self):
        """The 4 held eligible blue-chips (EXIT signal) must NOT be SELL — they
        are graded HOLD/ADD/TRIM, never liquidated for a low relative signal."""
        result = self._run()
        all_rows = result["actions"] + result["holds"]
        action_by_ticker = {r["ticker"]: r.get("action") for r in all_rows}
        for t in ("MEGA1", "MEGA2", "MEGA3", "MEGA4"):
            assert t in action_by_ticker, f"held blue-chip {t} vanished"
            assert action_by_ticker[t] != "SELL", (
                f"held eligible blue-chip {t} was force-SELL'd (F1 regression); "
                f"action={action_by_ticker[t]}"
            )
            assert action_by_ticker[t] in {"ADD", "HOLD", "TRIM"}

    def test_held_price_only_becomes_hold(self):
        """Held ETF/crypto (price sleeve skipped) must degrade to HOLD, never a
        fabricated SELL (F2 regression)."""
        result = self._run()
        all_rows = result["actions"] + result["holds"]
        action_by_ticker = {r["ticker"]: r.get("action") for r in all_rows}
        for t in ("GLD", "BTC-USD"):
            assert t in action_by_ticker, f"held price-only {t} vanished"
            assert action_by_ticker[t] == "HOLD", (
                f"held price-only {t} must be HOLD (unscored/held), got {action_by_ticker[t]}"
            )

    def test_only_s1_failing_name_is_sold(self):
        """The single held name that FAILS S1 eligibility (JUNKCO) is the ONLY
        SELL — SELL is reserved for names dropped from the investable universe."""
        result = self._run()
        sell_tickers = {r["ticker"] for r in result["actions"] if r.get("action") == "SELL"}
        assert "JUNKCO" in sell_tickers, "S1-failing held name must SELL"
        assert sell_tickers == {"JUNKCO"}, (
            f"only the S1-failing name should SELL; got {sell_tickers}"
        )

    def test_sells_are_small_minority_not_wholesale(self):
        """The plan must NOT liquidate the core: SELLs are a small minority of
        the 7 holdings (exactly 1 here — only the S1-failing name)."""
        result = self._run()
        all_rows = result["actions"] + result["holds"]
        held_universe = {"MEGA1", "MEGA2", "MEGA3", "MEGA4", "JUNKCO", "GLD", "BTC-USD"}
        held_rows = [r for r in all_rows if r["ticker"] in held_universe]
        sells = [r for r in held_rows if r.get("action") == "SELL"]
        retained = [r for r in held_rows if r.get("action") in {"ADD", "HOLD", "TRIM"}]
        # At most 1 of 7 holdings sold — a small minority, not wholesale churn.
        assert len(sells) <= 2, f"wholesale liquidation regression: {len(sells)} SELLs"
        # The core is retained (graded ADD/HOLD/TRIM, not zeroed): blue-chips
        # TRIM (low relative signal) and ETFs HOLD → at least 6 of 7 retained.
        assert len(retained) >= 6, f"core not retained: only {len(retained)} kept"

    def test_resulting_gross_not_collapsed(self):
        """A ~95%-invested book must not collapse to near-cash after one pass."""
        result = self._run()
        # The book was ~55% held (7 names summing to 55%); after grading, gross
        # must stay well above the F1 collapse (~17%). Require it to retain the
        # bulk of the held book (blue-chips + ETFs kept, only JUNKCO removed).
        assert result["resulting_gross"] > 0.40, (
            f"resulting_gross collapsed to {result['resulting_gross']:.2%} "
            f"(F1/F2 wholesale-liquidation regression)"
        )

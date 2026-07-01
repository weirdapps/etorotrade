"""Unit tests for the census sub-group edge study pure functions.

These exercise the numeric behaviour of the pure functions only (no IO/network):
subgroup decile selection, consensus weighting (long-only, top_n, summed pct),
forward-return nearest-trading-day logic + missing handling, excess t-stat math,
and Benjamini-Hochberg monotonicity/threshold behaviour.
"""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.census_edge_study import (
    benjamini_hochberg,
    build_consensus,
    build_id_symbol_map,
    excess_stats,
    forward_return,
    select_subgroup,
    two_sided_p_from_t,
)

# --------------------------------------------------------------------------- #
# build_id_symbol_map
# --------------------------------------------------------------------------- #


class TestBuildIdSymbolMap:
    def test_new_schema_details_symbolfull_stocks_only(self):
        instruments = {
            "details": [
                {"instrumentId": 1001, "symbolFull": "AAPL", "instrumentTypeID": 5},
                {"instrumentId": 1002, "symbolFull": "GOOG", "instrumentTypeID": 5},
                {"instrumentId": 2000, "symbolFull": "EURUSD", "instrumentTypeID": 1},
            ]
        }
        m = build_id_symbol_map(instruments)
        assert m == {1001: "AAPL", 1002: "GOOG"}  # non-stock (type 1) filtered

    def test_old_schema_flat_list_symbol(self):
        instruments = [
            {"instrumentId": 4244, "symbol": "msft"},
            {"instrumentId": 1002, "symbol": "GOOG"},
        ]
        m = build_id_symbol_map(instruments)
        assert m == {4244: "MSFT", 1002: "GOOG"}  # upper-cased

    def test_skips_missing_id_or_symbol(self):
        instruments = {
            "details": [
                {"symbolFull": "NOID", "instrumentTypeID": 5},
                {"instrumentId": 7, "instrumentTypeID": 5},  # no symbol
                {"instrumentId": 8, "symbolFull": "OK", "instrumentTypeID": 5},
            ]
        }
        assert build_id_symbol_map(instruments) == {8: "OK"}

    def test_garbage_returns_empty(self):
        assert build_id_symbol_map(None) == {}
        assert build_id_symbol_map(42) == {}


# --------------------------------------------------------------------------- #
# select_subgroup
# --------------------------------------------------------------------------- #


class TestSelectSubgroup:
    def _investors(self):
        # 10 investors with gain 0..90; decile => 1 investor (the top/bottom).
        return [{"id": i, "gain": float(i * 10)} for i in range(10)]

    def test_top_decile_picks_highest(self):
        invs = self._investors()
        sel = select_subgroup(invs, "gain", pct=0.10, highest=True)
        assert len(sel) == 1
        assert sel[0]["gain"] == 90.0

    def test_bottom_decile_picks_lowest(self):
        invs = self._investors()
        sel = select_subgroup(invs, "gain", pct=0.10, highest=False)
        assert len(sel) == 1
        assert sel[0]["gain"] == 0.0

    def test_decile_size_rounds_to_count(self):
        invs = [{"gain": float(i)} for i in range(25)]
        sel = select_subgroup(invs, "gain", pct=0.10, highest=True)
        # round(25*0.10)=2 (banker's rounding of 2.5 -> 2)
        assert len(sel) == 2
        assert sorted(x["gain"] for x in sel) == [23.0, 24.0]

    def test_top_20pct(self):
        invs = self._investors()
        sel = select_subgroup(invs, "gain", pct=0.20, highest=True)
        assert len(sel) == 2
        assert sorted(x["gain"] for x in sel) == [80.0, 90.0]

    def test_none_metric_returns_all(self):
        invs = self._investors()
        assert len(select_subgroup(invs, None)) == len(invs)

    def test_missing_metric_excluded_from_ranking(self):
        invs = [{"gain": 5.0}, {"gain": None}, {"other": 1}, {"gain": 100.0}]
        sel = select_subgroup(invs, "gain", pct=0.5, highest=True)
        # only 2 have a valid gain -> round(2*0.5)=1 -> the top one
        assert len(sel) == 1
        assert sel[0]["gain"] == 100.0

    def test_nested_field_extraction(self):
        invs = [
            {"portfolio": {"positionsCount": 10}},
            {"portfolio": {"positionsCount": 200}},
        ]
        sel = select_subgroup(invs, "positionsCount", nested="portfolio", pct=0.5, highest=True)
        assert sel[0]["portfolio"]["positionsCount"] == 200

    def test_empty_returns_empty(self):
        assert select_subgroup([], "gain") == []


# --------------------------------------------------------------------------- #
# build_consensus
# --------------------------------------------------------------------------- #


class TestBuildConsensus:
    def test_sums_investmentpct_across_investors(self):
        id_map = {1: "AAPL", 2: "MSFT"}
        subgroup = [
            {
                "portfolio": {
                    "positions": [
                        {"instrumentId": 1, "isBuy": True, "investmentPct": 5.0},
                        {"instrumentId": 2, "isBuy": True, "investmentPct": 3.0},
                    ]
                }
            },
            {
                "portfolio": {
                    "positions": [
                        {"instrumentId": 1, "isBuy": True, "investmentPct": 2.0},
                    ]
                }
            },
        ]
        book = build_consensus(subgroup, id_map, top_n=15)
        d = dict(book)
        assert d["AAPL"] == pytest.approx(7.0)  # 5 + 2
        assert d["MSFT"] == pytest.approx(3.0)
        assert book[0][0] == "AAPL"  # ranked by weight desc

    def test_long_only_excludes_shorts(self):
        id_map = {1: "AAPL"}
        subgroup = [
            {
                "portfolio": {
                    "positions": [
                        {"instrumentId": 1, "isBuy": False, "investmentPct": 99.0},  # short
                        {"instrumentId": 1, "isBuy": True, "investmentPct": 4.0},
                    ]
                }
            }
        ]
        book = dict(build_consensus(subgroup, id_map))
        assert book == {"AAPL": pytest.approx(4.0)}

    def test_top_n_truncation(self):
        id_map = {i: f"S{i}" for i in range(20)}
        positions = [
            {"instrumentId": i, "isBuy": True, "investmentPct": float(i)} for i in range(20)
        ]
        subgroup = [{"portfolio": {"positions": positions}}]
        book = build_consensus(subgroup, id_map, top_n=15)
        assert len(book) == 15
        # highest weights kept: S19..S5
        syms = [s for s, _w in book]
        assert syms[0] == "S19"
        assert "S0" not in syms

    def test_unresolvable_instrument_dropped(self):
        id_map = {1: "AAPL"}  # id 999 absent
        subgroup = [
            {
                "portfolio": {
                    "positions": [
                        {"instrumentId": 999, "isBuy": True, "investmentPct": 50.0},
                        {"instrumentId": 1, "isBuy": True, "investmentPct": 1.0},
                    ]
                }
            }
        ]
        assert dict(build_consensus(subgroup, id_map)) == {"AAPL": pytest.approx(1.0)}

    def test_empty_subgroup(self):
        assert build_consensus([], {1: "AAPL"}) == []


# --------------------------------------------------------------------------- #
# forward_return
# --------------------------------------------------------------------------- #


class TestForwardReturn:
    def _panel(self):
        # Business days; weekends absent to test nearest-trading-day logic.
        idx = pd.to_datetime(
            [
                "2025-06-02",
                "2025-06-03",
                "2025-06-04",
                "2025-06-05",
                "2025-06-06",
                "2025-06-09",
                "2025-06-10",
            ]
        )
        return pd.DataFrame(
            {
                "AAPL": [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 111.0],
                "MSFT": [200.0, 200.0, 200.0, 200.0, 200.0, 220.0, 220.0],
            },
            index=idx,
        )

    def test_nearest_trading_day_on_or_after(self):
        panel = self._panel()
        # T=2025-06-01 (Sun) -> first close >= is 06-02 (100).
        # h=7 -> T+7 = 06-08 (Sun) -> first close >= is 06-09 (110).
        r = forward_return(panel, ["AAPL"], "2025-06-01", 7)
        assert r == pytest.approx(110.0 / 100.0 - 1.0)  # +10%

    def test_equal_weight_mean_across_symbols(self):
        panel = self._panel()
        # AAPL: 110/100-1 = +0.10 ; MSFT: 220/200-1 = +0.10 -> mean +0.10
        r = forward_return(panel, ["AAPL", "MSFT"], "2025-06-01", 7)
        assert r == pytest.approx(0.10)

    def test_different_returns_average(self):
        panel = self._panel()
        # exact trading days: T=06-02 (100), T+3=06-05 (103) -> AAPL +3%
        #                     MSFT 200 -> 200 -> 0%  ; mean = +1.5%
        r = forward_return(panel, ["AAPL", "MSFT"], "2025-06-02", 3)
        assert r == pytest.approx((0.03 + 0.0) / 2)

    def test_symbol_not_in_panel_dropped(self):
        panel = self._panel()
        r = forward_return(panel, ["AAPL", "NOPE"], "2025-06-02", 3)
        assert r == pytest.approx(0.03)  # only AAPL counts

    def test_all_unresolvable_returns_none(self):
        panel = self._panel()
        assert forward_return(panel, ["NOPE"], "2025-06-02", 3) is None

    def test_target_past_end_returns_none(self):
        panel = self._panel()
        # T well after the panel end -> unresolvable
        assert forward_return(panel, ["AAPL"], "2025-07-01", 7) is None

    def test_empty_inputs_return_none(self):
        assert forward_return(pd.DataFrame(), ["AAPL"], "2025-06-02", 7) is None
        assert forward_return(self._panel(), [], "2025-06-02", 7) is None


# --------------------------------------------------------------------------- #
# excess_stats
# --------------------------------------------------------------------------- #


class TestExcessStats:
    def test_basic_mean_and_hit(self):
        st = excess_stats([0.01, 0.02, -0.01, 0.03])
        assert st["n"] == 4
        assert st["mean"] == pytest.approx(0.0125)
        assert st["hit"] == pytest.approx(0.75)  # 3 of 4 > 0

    def test_t_stat_matches_formula(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        st = excess_stats(vals)
        arr = np.array(vals)
        expected_t = arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))
        assert st["t"] == pytest.approx(expected_t)
        assert st["std"] == pytest.approx(arr.std(ddof=1))

    def test_nan_dropped(self):
        st = excess_stats([1.0, float("nan"), 3.0])
        assert st["n"] == 2
        assert st["mean"] == pytest.approx(2.0)

    def test_single_value_t_is_nan(self):
        st = excess_stats([0.05])
        assert st["n"] == 1
        assert math.isnan(st["t"])
        assert st["hit"] == pytest.approx(1.0)

    def test_empty_all_nan(self):
        st = excess_stats([])
        assert st["n"] == 0
        assert math.isnan(st["mean"])
        assert math.isnan(st["t"])

    def test_zero_variance_positive_mean_inf_t(self):
        st = excess_stats([0.02, 0.02, 0.02])
        assert math.isinf(st["t"]) and st["t"] > 0


# --------------------------------------------------------------------------- #
# benjamini_hochberg
# --------------------------------------------------------------------------- #


class TestBenjaminiHochberg:
    def test_all_significant_when_tiny(self):
        mask = benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.10)
        assert mask == [True, True, True]

    def test_none_significant_when_large(self):
        mask = benjamini_hochberg([0.9, 0.8, 0.95], alpha=0.10)
        assert mask == [False, False, False]

    def test_step_up_rejects_below_threshold(self):
        # Classic BH example (Benjamini-Hochberg 1995), alpha=0.05.
        # Thresholds are 0.05*k/10; the LARGEST rank k with p_(k) <= threshold
        # is k=8 (p=0.0344 <= 0.040), so the step-up rule rejects the first 8.
        pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240]
        mask = benjamini_hochberg(pvals, alpha=0.05)
        assert mask == [True, True, True, True, True, True, True, True, False, False]

    def test_step_up_recovers_dip_below_threshold(self):
        # A p-value can sit ABOVE its own threshold yet still be rejected if a
        # later (larger) rank passes -- the defining property of step-up BH.
        # m=4, alpha=0.10 -> thresholds 0.025, 0.05, 0.075, 0.10.
        # rank 2 (p=0.06) FAILS its own 0.05 threshold, but rank 4 (0.09<=0.10)
        # passes, so k_max=4 and ALL four hypotheses are rejected.
        pvals = [0.01, 0.06, 0.07, 0.09]
        mask = benjamini_hochberg(pvals, alpha=0.10)
        assert mask == [True, True, True, True]

    def test_order_preserved(self):
        # Unsorted input: mask must align to original positions.
        pvals = [0.30, 0.001, 0.20, 0.002]
        mask = benjamini_hochberg(pvals, alpha=0.10)
        assert mask[1] is True and mask[3] is True
        assert mask[0] is False and mask[2] is False

    def test_monotonicity_in_alpha(self):
        pvals = [0.01, 0.04, 0.06, 0.2]
        lo = benjamini_hochberg(pvals, alpha=0.05)
        hi = benjamini_hochberg(pvals, alpha=0.25)
        # Raising alpha can only ADD rejections, never remove.
        assert sum(hi) >= sum(lo)
        for a, b in zip(lo, hi):
            assert (not a) or b  # a implies b

    def test_nan_treated_as_one(self):
        mask = benjamini_hochberg([0.001, float("nan")], alpha=0.10)
        assert mask[0] is True and mask[1] is False

    def test_empty(self):
        assert benjamini_hochberg([]) == []


# --------------------------------------------------------------------------- #
# two_sided_p_from_t  (sanity check on the scipy-free p-value)
# --------------------------------------------------------------------------- #


class TestTwoSidedP:
    def test_zero_t_is_one(self):
        assert two_sided_p_from_t(0.0, 10) == pytest.approx(1.0, abs=1e-6)

    def test_large_t_small_p(self):
        p = two_sided_p_from_t(10.0, 20)
        assert 0.0 <= p < 1e-6

    def test_matches_known_value(self):
        # t=2.228, df=10 -> two-sided p ~ 0.05 (t-table critical value).
        p = two_sided_p_from_t(2.228, 10)
        assert p == pytest.approx(0.05, abs=5e-3)

    def test_symmetric_in_sign(self):
        assert two_sided_p_from_t(2.5, 15) == pytest.approx(two_sided_p_from_t(-2.5, 15))

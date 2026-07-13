"""TDD tests for trade_modules.v3.actions.build_actions.

Phase 5C: PM construction — suggested actions (BUY/ADD/TRIM/SELL/HOLD) vs live
portfolio.

Tests:
- BUY:  ticker in target only (current_pct ≈ 0)
- SELL: ticker in current only (target_pct ≈ 0)
- ADD:  both, delta_pct > +threshold
- TRIM: both, delta_pct < -threshold
- HOLD: both, |delta_pct| <= threshold
- delta_usd = delta_pct * nav when nav provided; None when nav absent
- SL/TP pulled from scored for BUY and ADD rows
- SELL rows absent from scored emit None for name/sector/conviction/price/SL/TP
- Sort/grouping: BUY+ADD (target_pct desc) → TRIM+SELL (current_pct desc) → HOLD
- epsilon boundary: current_pct < 1e-6 treated as not-held (→ BUY not ADD)
"""

import pandas as pd
import pytest

from trade_modules.v3.actions import build_actions

# ---------------------------------------------------------------------------
# Shared scored frame helper
# ---------------------------------------------------------------------------


def _scored(tickers, *, conviction=1.0, price=100.0, sl=90.0, tp=120.0, sector="Tech"):
    """Build a minimal scored DataFrame for the given tickers."""
    rows = []
    for t in tickers:
        rows.append(
            {
                "name": f"Company {t}",
                "sector": sector,
                "conviction": conviction,
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(tickers, name="ticker"))


# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------


class TestBuy:
    """Ticker in target only → BUY."""

    def test_action_label(self):
        target = pd.Series({"A": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["A"])
        acts = build_actions(target, current, scored)
        assert len(acts) == 1
        assert acts[0]["action"] == "BUY"

    def test_correct_ticker(self):
        target = pd.Series({"A": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["A"])
        acts = build_actions(target, current, scored)
        assert acts[0]["ticker"] == "A"

    def test_target_pct_set(self):
        target = pd.Series({"A": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["A"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["target_pct"]) == 0.05

    def test_current_pct_zero(self):
        target = pd.Series({"A": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["A"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["current_pct"]) == 0.0

    def test_delta_pct(self):
        target = pd.Series({"A": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["A"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["delta_pct"]) == 0.05


class TestSell:
    """Ticker in current only → SELL."""

    def test_action_label(self):
        target = pd.Series(dtype=float)
        current = pd.Series({"B": 0.04})
        scored = _scored(["B"])
        acts = build_actions(target, current, scored)
        assert len(acts) == 1
        assert acts[0]["action"] == "SELL"

    def test_current_pct_set(self):
        target = pd.Series(dtype=float)
        current = pd.Series({"B": 0.04})
        scored = _scored(["B"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["current_pct"]) == 0.04

    def test_target_pct_zero(self):
        target = pd.Series(dtype=float)
        current = pd.Series({"B": 0.04})
        scored = _scored(["B"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["target_pct"]) == 0.0

    def test_delta_pct_negative(self):
        target = pd.Series(dtype=float)
        current = pd.Series({"B": 0.04})
        scored = _scored(["B"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["delta_pct"]) == -0.04

    def test_sell_not_in_scored_emits_none_fields(self):
        """SELL ticker absent from scored → name/sector/conviction/price/SL/TP are None."""
        target = pd.Series(dtype=float)
        current = pd.Series({"GHOST": 0.03})
        scored = _scored([])  # empty
        acts = build_actions(target, current, scored)
        row = acts[0]
        assert row["name"] is None
        assert row["sector"] is None
        assert row["conviction"] is None
        assert row["price"] is None
        assert row["stop_loss"] is None
        assert row["take_profit"] is None


class TestAdd:
    """Both in target and current, delta > +threshold → ADD."""

    def test_action_label(self):
        target = pd.Series({"C": 0.07})
        current = pd.Series({"C": 0.02})
        scored = _scored(["C"])
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        assert acts[0]["action"] == "ADD"

    def test_delta_pct(self):
        target = pd.Series({"C": 0.07})
        current = pd.Series({"C": 0.02})
        scored = _scored(["C"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["delta_pct"]) == 0.05


class TestTrim:
    """Both in target and current, delta < -threshold → TRIM."""

    def test_action_label(self):
        target = pd.Series({"D": 0.02})
        current = pd.Series({"D": 0.08})
        scored = _scored(["D"])
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        assert acts[0]["action"] == "TRIM"

    def test_delta_pct_negative(self):
        target = pd.Series({"D": 0.02})
        current = pd.Series({"D": 0.08})
        scored = _scored(["D"])
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["delta_pct"]) == -0.06


class TestHold:
    """Both in target and current, |delta| <= threshold → HOLD."""

    def test_action_label_exact_threshold(self):
        target = pd.Series({"E": 0.055})
        current = pd.Series({"E": 0.05})
        scored = _scored(["E"])
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        # delta = 0.005 == threshold → HOLD (not ADD)
        assert acts[0]["action"] == "HOLD"

    def test_action_label_within_threshold(self):
        target = pd.Series({"E": 0.052})
        current = pd.Series({"E": 0.05})
        scored = _scored(["E"])
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        assert acts[0]["action"] == "HOLD"

    def test_action_label_negative_within_threshold(self):
        target = pd.Series({"E": 0.048})
        current = pd.Series({"E": 0.05})
        scored = _scored(["E"])
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        assert acts[0]["action"] == "HOLD"


# ---------------------------------------------------------------------------
# Epsilon boundary: sub-epsilon current treated as not-held (→ BUY)
# ---------------------------------------------------------------------------


class TestEpsilon:
    def test_sub_epsilon_current_gives_buy(self):
        target = pd.Series({"F": 0.06})
        current = pd.Series({"F": 5e-7})  # below epsilon 1e-6
        scored = _scored(["F"])
        acts = build_actions(target, current, scored)
        assert acts[0]["action"] == "BUY"

    def test_sub_epsilon_target_gives_sell(self):
        target = pd.Series({"G": 5e-7})  # below epsilon
        current = pd.Series({"G": 0.04})
        scored = _scored(["G"])
        acts = build_actions(target, current, scored)
        assert acts[0]["action"] == "SELL"


# ---------------------------------------------------------------------------
# delta_usd: nav * delta_pct when nav given, else None
# ---------------------------------------------------------------------------


class TestDeltaUsd:
    def test_delta_usd_with_nav(self):
        target = pd.Series({"H": 0.10})
        current = pd.Series({"H": 0.04})
        scored = _scored(["H"])
        acts = build_actions(target, current, scored, nav=100_000.0)
        assert pytest.approx(acts[0]["delta_usd"]) == pytest.approx(0.06 * 100_000.0)

    def test_delta_usd_none_without_nav(self):
        target = pd.Series({"I": 0.10})
        current = pd.Series({"I": 0.04})
        scored = _scored(["I"])
        acts = build_actions(target, current, scored, nav=None)
        assert acts[0]["delta_usd"] is None

    def test_delta_usd_sell_with_nav(self):
        target = pd.Series(dtype=float)
        current = pd.Series({"J": 0.05})
        scored = _scored(["J"])
        acts = build_actions(target, current, scored, nav=200_000.0)
        assert pytest.approx(acts[0]["delta_usd"]) == pytest.approx(-0.05 * 200_000.0)


# ---------------------------------------------------------------------------
# SL/TP present for BUY and ADD, and pulled from scored
# ---------------------------------------------------------------------------


class TestStopLossTakeProfit:
    def test_buy_has_sl_tp(self):
        target = pd.Series({"K": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["K"], sl=88.0, tp=125.0)
        acts = build_actions(target, current, scored)
        row = acts[0]
        assert row["action"] == "BUY"
        assert pytest.approx(row["stop_loss"]) == 88.0
        assert pytest.approx(row["take_profit"]) == 125.0

    def test_add_has_sl_tp(self):
        target = pd.Series({"L": 0.08})
        current = pd.Series({"L": 0.02})
        scored = _scored(["L"], sl=75.0, tp=150.0)
        acts = build_actions(target, current, scored)
        row = acts[0]
        assert row["action"] == "ADD"
        assert pytest.approx(row["stop_loss"]) == 75.0
        assert pytest.approx(row["take_profit"]) == 150.0

    def test_conviction_pulled_from_scored(self):
        target = pd.Series({"M": 0.05})
        current = pd.Series(dtype=float)
        scored = _scored(["M"], conviction=3.75)
        acts = build_actions(target, current, scored)
        assert pytest.approx(acts[0]["conviction"]) == 3.75


# ---------------------------------------------------------------------------
# Sort / grouping order
# ---------------------------------------------------------------------------


class TestSortOrder:
    """BUY+ADD (target_pct desc) → TRIM+SELL (current_pct desc) → HOLD last."""

    def _make_multi(self):
        """Universe: 2 BUY, 1 ADD, 1 TRIM, 1 SELL, 1 HOLD."""
        target = pd.Series(
            {
                "BUY1": 0.08,  # BUY (larger)
                "BUY2": 0.05,  # BUY (smaller)
                "ADD1": 0.07,  # ADD
                "TRIM1": 0.01,  # TRIM
                "HOLD1": 0.04,  # HOLD
            }
        )
        current = pd.Series(
            {
                "ADD1": 0.02,  # ADD: delta = +0.05 > 0.005
                "TRIM1": 0.06,  # TRIM: delta = -0.05 < -0.005
                "SELL1": 0.09,  # SELL (not in target)
                "HOLD1": 0.042,  # HOLD: delta = -0.002 ≤ 0.005
            }
        )
        all_tickers = set(target.index) | set(current.index)
        scored = _scored(list(all_tickers))
        return target, current, scored

    def test_buy_add_before_trim_sell_before_hold(self):
        target, current, scored = self._make_multi()
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        actions = [a["action"] for a in acts]
        # Find positions of group boundaries
        hold_idx = next(i for i, a in enumerate(actions) if a == "HOLD")
        # All HOLD rows must be at the end
        assert all(a == "HOLD" for a in actions[hold_idx:])
        # All BUY/ADD before first SELL/TRIM
        buy_add = [i for i, a in enumerate(actions) if a in ("BUY", "ADD")]
        trim_sell = [i for i, a in enumerate(actions) if a in ("TRIM", "SELL")]
        if buy_add and trim_sell:
            assert max(buy_add) < min(trim_sell)

    def test_buy_add_sorted_by_target_pct_desc(self):
        target, current, scored = self._make_multi()
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        buy_add = [a for a in acts if a["action"] in ("BUY", "ADD")]
        pcts = [a["target_pct"] for a in buy_add]
        assert pcts == sorted(pcts, reverse=True)

    def test_trim_sell_sorted_by_current_pct_desc(self):
        target, current, scored = self._make_multi()
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        trim_sell = [a for a in acts if a["action"] in ("TRIM", "SELL")]
        pcts = [a["current_pct"] for a in trim_sell]
        assert pcts == sorted(pcts, reverse=True)

    def test_hold_is_last(self):
        target, current, scored = self._make_multi()
        acts = build_actions(target, current, scored, add_trim_threshold=0.005)
        actions = [a["action"] for a in acts]
        assert actions[-1] == "HOLD"


# ---------------------------------------------------------------------------
# Misc / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_inputs_returns_empty_list(self):
        acts = build_actions(pd.Series(dtype=float), pd.Series(dtype=float), _scored([]))
        assert acts == []

    def test_buy_and_sell_mix(self):
        target = pd.Series({"NEW": 0.06})
        current = pd.Series({"OLD": 0.03})
        scored = _scored(["NEW", "OLD"])
        acts = build_actions(target, current, scored)
        labels = {a["ticker"]: a["action"] for a in acts}
        assert labels["NEW"] == "BUY"
        assert labels["OLD"] == "SELL"

    def test_threshold_zero_makes_hold_very_narrow(self):
        """With threshold=0, any nonzero delta becomes ADD or TRIM."""
        target = pd.Series({"X": 0.051})
        current = pd.Series({"X": 0.050})
        scored = _scored(["X"])
        acts = build_actions(target, current, scored, add_trim_threshold=0.0)
        assert acts[0]["action"] == "ADD"

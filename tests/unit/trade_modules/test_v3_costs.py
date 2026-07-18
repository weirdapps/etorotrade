"""TDD — BUILD ⑥b (2026-07-18): transaction-cost model for v3.

Round-trip cost = entry+exit spread (by cap tier) + eToro overnight financing over
the holding horizon. Reuses liquidity_filter's TIER_SPREAD_BPS +
ETORO_OVERNIGHT_ANNUAL_RATE so the cost constants have one home. Its principled use
is a net-of-cost IC in validation (trial-register net_ir_min) — grading net alpha,
not changing live trading.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.liquidity_filter import ETORO_OVERNIGHT_ANNUAL_RATE, TIER_SPREAD_BPS
from trade_modules.v3.costs import (
    DEFAULT_HOLDING_DAYS,
    cost_map_from_panel,
    net_of_cost_return,
    roundtrip_cost_pct,
)


def test_roundtrip_cost_is_spread_plus_financing():
    c = roundtrip_cost_pct(3e11)  # MEGA
    spread = TIER_SPREAD_BPS["MEGA"] / 10000.0 * 2.0 * 100.0  # round-trip, in %
    fin = ETORO_OVERNIGHT_ANNUAL_RATE * (DEFAULT_HOLDING_DAYS / 365.0) * 100.0
    assert abs(c - (spread + fin)) < 1e-9


def test_smaller_caps_cost_more():
    assert roundtrip_cost_pct(2e8) > roundtrip_cost_pct(3e11)  # MICRO spread > MEGA spread


def test_financing_scales_with_holding_days():
    short = roundtrip_cost_pct(3e11, holding_days=30)
    longer = roundtrip_cost_pct(3e11, holding_days=60)
    fin30 = ETORO_OVERNIGHT_ANNUAL_RATE * (30 / 365.0) * 100.0
    assert abs((longer - short) - fin30) < 1e-9  # exactly one extra 30d of financing


def test_unknown_cap_uses_mid_tier():
    assert roundtrip_cost_pct(float("nan")) == roundtrip_cost_pct(5e9)  # both MID


def test_net_of_cost_subtracts_roundtrip():
    c = roundtrip_cost_pct(1e9)
    assert abs(net_of_cost_return(5.0, 1e9) - (5.0 - c)) < 1e-9


def test_cost_map_from_panel(tmp_path):
    csv = tmp_path / "etoro.csv"
    pd.DataFrame({"TKR": ["AAA", "bbb"], "CAP": ["300B", "600M"]}).to_csv(csv, index=False)
    m = cost_map_from_panel(str(csv))
    assert set(m) == {"AAA", "BBB"}  # keys upper-cased
    assert m["BBB"] > m["AAA"]  # 600M (SMALL) costs more than 300B (MEGA)
    assert abs(m["AAA"] - roundtrip_cost_pct(300e9)) < 1e-9

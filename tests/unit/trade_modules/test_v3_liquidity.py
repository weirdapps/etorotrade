"""TDD — BUILD ⑥a (2026-07-18): tiered, copier-aware ADV liquidity gate.

v3 gated liquidity with a single flat $1M floor applied inline in the report. After
④ expanded the scored universe ~25x (many smaller/less-liquid names), the floor
should scale with market-cap tier (a $500M small-cap needs less ADV than a mega-cap
position) and — for a copied book — with a copier multiplier. Uses v3's own
``adv_usd`` (avg_volume x price x fx, already computed) so there is NO extra fetch.
Held names and names with unknown ADV are never dropped.
"""

from __future__ import annotations

import pandas as pd

from trade_modules.liquidity_filter import TIER_MIN_ADV
from trade_modules.v3.liquidity import cap_tier, liquidity_gate, required_adv


def test_cap_tier_thresholds():
    assert cap_tier(3e11) == "MEGA"  # >= $200B
    assert cap_tier(5e10) == "LARGE"  # >= $10B
    assert cap_tier(5e9) == "MID"  # >= $2B
    assert cap_tier(1e9) == "SMALL"  # >= $300M
    assert cap_tier(2e8) == "MICRO"  # < $300M
    assert cap_tier(float("nan")) == "MID"  # unknown cap -> neutral MID floor


def test_required_adv_uses_tier_floor():
    assert required_adv(3e11) == TIER_MIN_ADV["MEGA"]
    assert required_adv(1e9) == TIER_MIN_ADV["SMALL"]


def test_required_adv_scales_with_copier_multiplier():
    base = required_adv(1e9)
    assert required_adv(1e9, copier_multiplier=3.0) == base * 3.0
    assert required_adv(1e9, copier_multiplier=0.5) == base  # clamped at >= 1.0


def _scores(rows):
    idx = [r[0] for r in rows]
    return pd.DataFrame(
        {
            "cap": [r[1] for r in rows],
            "adv_usd": [r[2] for r in rows],
            "is_portfolio": [r[3] for r in rows],
        },
        index=idx,
    )


def test_gate_drops_illiquid_non_held_keeps_liquid():
    scores = _scores(
        [
            ("ILLIQ", 1e9, 1e6, False),  # SMALL, 1M ADV < 5M floor -> drop
            ("MEGA", 3e11, 1e9, False),  # huge ADV -> keep
            ("LIQ", 1e9, 8e6, False),  # SMALL, 8M >= 5M -> keep
        ]
    )
    kept, dropped = liquidity_gate(scores)
    assert "ILLIQ" in dropped.index and "ILLIQ" not in kept.index
    assert {"MEGA", "LIQ"}.issubset(set(kept.index))


def test_gate_exempts_held_names():
    scores = _scores([("HELD", 1e9, 1e6, True)])  # illiquid but held
    kept, dropped = liquidity_gate(scores)
    assert "HELD" in kept.index
    assert dropped.empty


def test_gate_keeps_unknown_adv_no_data_penalty():
    scores = _scores([("NOADV", 1e9, float("nan"), False)])
    kept, dropped = liquidity_gate(scores)
    assert "NOADV" in kept.index
    assert dropped.empty


def test_gate_copier_multiplier_tightens_floor():
    # SMALL name, 8M ADV: passes at mult=1 (floor 5M), fails at mult=2 (floor 10M).
    scores = _scores([("MIDLIQ", 1e9, 8e6, False)])
    assert "MIDLIQ" in liquidity_gate(scores, copier_multiplier=1.0)[0].index
    assert "MIDLIQ" in liquidity_gate(scores, copier_multiplier=2.0)[1].index


def test_gate_returns_all_when_no_adv_column():
    scores = pd.DataFrame({"cap": [1e9], "is_portfolio": [False]}, index=["X"])
    kept, dropped = liquidity_gate(scores)
    assert "X" in kept.index and dropped.empty  # no adv_usd -> gate is a no-op

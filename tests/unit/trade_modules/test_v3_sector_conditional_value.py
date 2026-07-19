"""TDD — P1 sector-conditional value cluster (banks->P/B, REITs cash-flow, tech->sales).

The value cluster is composed per sector-group instead of one uniform recipe:
  Group A (conventional): P/E + EV/EBITDA        [unchanged default]
  Group B (financials/REITs): P/B + P/E(half)     [book primary, EV meaningless]
  Group C (growth/intangible): P/S + EV/EBITDA(half), and the value cluster's
           weight in conviction is halved (value is weak/inverted in these sectors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_modules.v3.combine import (
    CLUSTERS,
    SECTOR_GROUPS,
    VALUE_WEIGHT_MULT,
    compute_scores,
)


def _base(index):
    """Features frame with all cluster metrics NaN; tests add just what they need."""
    cols = set()
    for members in CLUSTERS.values():
        cols.update(members)
    df = pd.DataFrame({c: [np.nan] * len(index) for c in cols}, index=index)
    df["sector"] = ["__NA__"] * len(index)
    return df


def test_sector_groups_mapping():
    assert SECTOR_GROUPS["Financial Services"] == "B"
    assert SECTOR_GROUPS["Real Estate"] == "B"
    assert SECTOR_GROUPS["Technology"] == "C"
    assert SECTOR_GROUPS["Healthcare"] == "C"
    assert SECTOR_GROUPS["Industrials"] == "A"
    assert SECTOR_GROUPS["Energy"] == "A"
    assert SECTOR_GROUPS.get("Nonexistent Sector", "A") == "A"  # default bucket
    assert VALUE_WEIGHT_MULT["C"] == 0.5 and VALUE_WEIGHT_MULT["A"] == 1.0


def test_group_B_financials_value_uses_pb():
    """In Financial Services (Group B), P/B drives value even against P/E.

    LOWPB is cheap on book (0.8) but pricey on earnings (P/E 30); HIGHPB is the
    reverse. Because P/B is weighted 1.0 vs P/E 0.5, the cheap-on-book name wins.
    """
    idx = ["LOWPB", "MID", "HIGHPB"]
    df = _base(idx)
    df["sector"] = ["Financial Services"] * 3
    df["pb"] = [0.8, 1.5, 3.0]
    df["pe_trailing"] = [30.0, 15.0, 8.0]
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["LOWPB", "value_z"] > s.loc["HIGHPB", "value_z"]


def test_group_A_value_ignores_pb():
    """In a conventional sector (Group A) P/B is NOT in the recipe, so it is ignored."""
    idx = ["X", "Y"]
    df = _base(idx)
    df["sector"] = ["Industrials", "Industrials"]
    df["pe_trailing"] = [10.0, 10.0]  # equal earnings valuation
    df["pb"] = [0.5, 5.0]  # very different book valuation -> must not matter in A
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["X", "value_z"] == pytest.approx(s.loc["Y", "value_z"])


def test_group_C_tech_value_uses_ps_not_pe():
    """In Technology (Group C), P/S drives value; trailing P/E is excluded."""
    idx = ["LOWPS", "HIGHPS"]
    df = _base(idx)
    df["sector"] = ["Technology", "Technology"]
    df["ps_sector"] = [2.0, 20.0]  # LOWPS cheap on sales
    df["pe_trailing"] = [50.0, 5.0]  # opposite on P/E -> must be ignored in C
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["LOWPS", "value_z"] > s.loc["HIGHPS", "value_z"]


def test_group_C_value_multiplier_reduces_conviction():
    """Group C halves the value cluster's weight in conviction.

    A_NAME (Industrials/A) and C_NAME (Technology/C) share an identical value_z
    (only ev_ebitda present, which sits in both recipes -> value_z = ev_ebitda_z)
    and an identical, present quality_z. The ONLY difference is the Group-C value
    multiplier, so value lifts A_NAME's conviction more than C_NAME's.
    """
    idx = ["A_NAME", "C_NAME", "F1", "F2"]
    df = _base(idx)
    df["sector"] = ["Industrials", "Technology", "Industrials", "Technology"]
    df["ev_ebitda"] = [5.0, 5.0, 20.0, 20.0]  # A_NAME & C_NAME cheap; fillers rich
    df["roe"] = [5.0, 5.0, 5.0, 5.0]  # equal quality, present for all
    s = compute_scores(df, sector_neutral=False)
    assert s.loc["A_NAME", "value_z"] == pytest.approx(s.loc["C_NAME", "value_z"])
    assert s.loc["A_NAME", "conviction"] > s.loc["C_NAME", "conviction"]

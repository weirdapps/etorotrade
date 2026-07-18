"""TDD — two-stage scoring (2026-07-19): bound the yfinance enrichment cost.

panel_prescore ranks the full coverage universe network-free; select_enrichment_set
returns holdings + candidates + top coverage names, bounded by cap — so a run enriches
~old-report scale instead of ~2,580 names (~3h).
"""

from __future__ import annotations

import pandas as pd

from trade_modules.v3.enrichment import panel_prescore, select_enrichment_set

# Strong -> weak on the panel factors (low PE / high ROE,FCF,52W,AM,EG / low beta = good).
_ROWS = [
    {
        "TKR": "AAA",
        "PRC": "10",
        "CAP": "1B",
        "PET": "8",
        "ROE": "30",
        "FCF": "12",
        "52W": "95",
        "B": "0.7",
        "AM": "8",
        "EG": "20",
    },
    {
        "TKR": "BBB",
        "PRC": "10",
        "CAP": "1B",
        "PET": "15",
        "ROE": "12",
        "FCF": "3",
        "52W": "60",
        "B": "1.0",
        "AM": "0",
        "EG": "5",
    },
    {
        "TKR": "DDD",
        "PRC": "10",
        "CAP": "1B",
        "PET": "20",
        "ROE": "8",
        "FCF": "1",
        "52W": "50",
        "B": "1.2",
        "AM": "-2",
        "EG": "2",
    },
    {
        "TKR": "CCC",
        "PRC": "10",
        "CAP": "1B",
        "PET": "40",
        "ROE": "2",
        "FCF": "-5",
        "52W": "20",
        "B": "1.6",
        "AM": "-8",
        "EG": "-10",
    },
]


def _panel(tmp_path):
    cols = ["TKR", "NAME", "CAP", "PRC", "PET", "ROE", "FCF", "52W", "B", "AM", "EG"]
    rows = [{**r, "NAME": r["TKR"] + " Corp"} for r in _ROWS]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    p = tmp_path / "etoro.csv"
    df[cols].to_csv(p, index=False)
    return str(p)


def test_panel_prescore_ranks_strong_above_weak(tmp_path):
    csv = _panel(tmp_path)
    from trade_modules.v3.universe import load_universe

    uni = load_universe(csv, min_factor_coverage=6)
    ps = panel_prescore(csv, uni)
    assert ps["AAA"] > ps["CCC"]  # strong panel factors outrank weak
    assert ps["AAA"] > ps["DDD"]
    assert ps.notna().sum() >= 3  # network-free, still produces a usable ranking


def test_select_enrichment_set_always_includes_holdings(tmp_path):
    csv = _panel(tmp_path)
    # CCC is the weakest name but is HELD -> must be enriched regardless of pre-score.
    out = select_enrichment_set(csv, holdings=["CCC"], candidates=[], cap=2, sector_map=None)
    assert "CCC" in out  # held name never dropped
    assert "AAA" in out  # top pre-score fills the remaining slot
    assert len(out) <= 2


def test_select_enrichment_set_respects_cap_and_fills_with_top(tmp_path):
    csv = _panel(tmp_path)
    out = select_enrichment_set(csv, holdings=[], candidates=[], cap=2, sector_map=None)
    assert len(out) == 2
    assert out[0] == "AAA"  # highest pre-score first
    assert "CCC" not in out  # weakest excluded when over cap


def test_select_enrichment_set_candidates_included(tmp_path):
    csv = _panel(tmp_path)
    out = select_enrichment_set(csv, holdings=[], candidates=["DDD"], cap=3, sector_map=None)
    assert "DDD" in out  # analyst candidate kept through the transition
    assert len(out) <= 3


def test_select_enrichment_set_dedup(tmp_path):
    csv = _panel(tmp_path)
    out = select_enrichment_set(csv, holdings=["AAA"], candidates=["AAA"], cap=4, sector_map=None)
    assert out.count("AAA") == 1  # de-duped across holdings/candidates/top

"""Integration test for the shadow runner + the markdown report generator."""

import pandas as pd

from trade_modules.riskfirst.shadow_run import build_report_md, run

_HEADER = "TKR,CAP,PET,PEF,P/S,FCF,ROE,DE,EG,%B,AM,B,52W"
_ROWS = [
    "AAA,100B,15,14,3,6,25,40,10,80,2,1.0,90",
    "BBB,50B,20,18,4,4,18,60,8,70,1,1.1,85",
    "CCC,30B,12,11,2,8,22,30,12,75,3,0.9,95",
    "DDD,10B,25,22,5,3,15,80,5,65,0,1.2,80",
    "MICRO,50M,,,,,,,,,,,",  # sub-$2B, no fundamentals -> filtered by eligibility
]


def _write_universe(tmp_path):
    p = tmp_path / "etoro.csv"
    p.write_text(_HEADER + "\n" + "\n".join(_ROWS) + "\n")
    return str(p)


def test_run_on_synthetic_universe(tmp_path):
    res = run(universe_path=_write_universe(tmp_path), portfolio_path="/nonexistent", top_n=3)
    assert res["mode"] == "SHADOW"
    assert len(res["selected"]) == 3
    assert "MICRO" not in res["selected"]  # investability gate worked
    assert res["target_weights"].sum() <= 1.0 + 1e-6
    assert res["edge_gate"]["passed"] is False  # no forward track record
    assert res["promotable"] is False
    assert isinstance(res["recommendations"], pd.DataFrame)
    # no current book -> every target is a fresh BUY
    assert (res["recommendations"]["action"] == "BUY").all()


def test_build_report_md_has_key_sections(tmp_path):
    res = run(universe_path=_write_universe(tmp_path), portfolio_path="/nonexistent", top_n=3)
    md = build_report_md(res)
    assert "SHADOW" in md
    assert "Edge gate" in md
    assert "Promotable" in md
    for tkr in res["selected"]:
        assert tkr in md


def test_shadow_run_overlay_disabled_has_neutral_stub(tmp_path):
    res = run(regime_overlay_enabled=False)
    assert res["regime"]["applied_multiplier"] == 1.0


def test_build_report_md_stub_regime_not_none_literal():
    import pandas as pd

    from trade_modules.riskfirst.shadow_run import build_report_md

    res = {
        "mode": "SHADOW",
        "gross": 0.0,
        "cash": 1.0,
        "usd_bloc": 0.0,
        "target_weights": pd.Series(dtype=float),
        "recommendations": pd.DataFrame(columns=["ticker", "action", "current", "target", "delta"]),
        "edge_gate": {"passed": False, "dsr": 0.5, "reasons": []},
        "promotable": False,
        "regime": {"raw_regime": None, "confirmed_regime": None, "applied_multiplier": 1.0},
    }
    md = build_report_md(res)
    assert "regime None" not in md
    assert "regime neutral" in md


def test_shadow_run_overlay_scales_gross(tmp_path):
    sp = str(tmp_path / "state.json")
    base = run(regime_overlay_enabled=False)
    # crisis confirmed by seeding two days via persistence_days=1
    # config.yaml is now AGGRESSIVE: crisis=0.20
    scaled = run(
        regime_overlay_enabled=True,
        regime_fn=lambda: "crisis",
        regime_state_path=sp,
        persistence_days=1,
    )
    assert scaled["regime"]["confirmed_regime"] == "crisis"
    assert scaled["regime"]["applied_multiplier"] == 0.20
    assert scaled["gross"] <= base["gross"] + 1e-9

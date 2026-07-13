"""TDD tests for trade_modules.v3.overlay.build_overlay.

The OVERLAY keeps the live book and applies a minimal, conviction-driven change:
sell ONLY genuinely weak holdings (ineligible / dataless / bottom-percentile
conviction), keep every other holding at its current weight (anchor), buy up to
``max_new`` of the strongest non-held names clearing the buy percentile, then run
the combined book through the same Phase 5A risk gate build_portfolio uses.

Covariance is driven off the single-factor beta fallback (empty ``prices``), so
every test is deterministic and network-free.
"""

import numpy as np
import pandas as pd
import pytest

from trade_modules.v3.overlay import build_overlay

_SECTORS = ["Tech", "Health", "Financials", "Energy", "Industrials"]


def _scored(tickers, convs, *, eligible=None, betas=1.0, sectors=None):
    """A scored frame in compute_scores shape (conviction/eligible/sector/price/beta)."""
    tickers = [str(t) for t in tickers]
    n = len(tickers)
    df = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    df["conviction"] = list(convs)
    df["eligible"] = [True] * n if eligible is None else list(eligible)
    df["sector"] = (
        [_SECTORS[i % len(_SECTORS)] for i in range(n)] if sectors is None else list(sectors)
    )
    df["price"] = 100.0
    df["beta"] = list(betas) if isinstance(betas, (list, tuple, np.ndarray)) else [betas] * n
    df["name"] = [f"Name {t}" for t in tickers]
    return df


def _universe20(betas=1.0):
    """20 eligible names, conviction linspace 2.0 (U00, best) .. -2.0 (U19, worst)."""
    tks = [f"U{i:02d}" for i in range(20)]
    convs = np.linspace(2.0, -2.0, 20)
    return tks, convs, _scored(tks, convs, betas=betas)


# --------------------------------------------------------------------------- #
# SELL / KEEP classification
# --------------------------------------------------------------------------- #


def test_classification_bottom_dataless_ineligible_sold_middling_kept():
    _tks, _convs, sc = _universe20()
    # An ineligible held name mirrors reality: eligible=False AND conviction NaN.
    sc = pd.concat([sc, _scored(["INELIG"], [np.nan], eligible=[False])])
    current = pd.Series(
        {"U00": 0.2, "U10": 0.2, "U19": 0.2, "INELIG": 0.2, "ZZZ": 0.2}
    )  # ZZZ is dataless (absent from scored)

    res = build_overlay(sc, current, pd.DataFrame(), max_new=8)
    d = res["diagnostics"]
    w = res["weights"]

    assert set(d["sold"]) == {"U19", "INELIG", "ZZZ"}  # bottom + ineligible + dataless
    assert set(d["kept"]) == {"U00", "U10"}  # strong + middling holdings retained
    assert d["n_sell"] == 3 and d["n_keep"] == 2
    for t in ("U19", "INELIG", "ZZZ"):
        assert float(w.get(t, 0.0)) == 0.0  # sold names freed from the book
    for t in ("U00", "U10"):
        assert t in w.index and w[t] > 0.0

    # Thresholds are percentiles of the ELIGIBLE universe (INELIG's NaN excluded).
    elig_conv = pd.to_numeric(sc.loc[sc["eligible"], "conviction"], errors="coerce").dropna()
    assert d["sell_threshold"] == pytest.approx(float(elig_conv.quantile(0.15)))
    assert d["buy_threshold"] == pytest.approx(float(elig_conv.quantile(0.85)))


def test_sell_negative_noncore_drops_negatives_but_protects_core():
    """Owner rule: sell every negative-conviction NON-core name; keep core negatives.

    U11 (~-0.32) and U12 (~-0.53) are negative but ABOVE the ~-1.40 percentile
    sell threshold, so the default overlay would keep both. With
    ``sell_negative_noncore`` the ordinary holding U11 is dropped, while the
    core-listed U12 survives only because ``protect_core`` exempts it.
    """
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.2, "U05": 0.2, "U11": 0.2, "U12": 0.2, "U19": 0.2})

    res = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        max_new=0,  # isolate the SELL side
        sell_negative_noncore=True,
        protect_core=True,
        core_list=["U12"],
    )
    d = res["diagnostics"]
    assert "U11" in d["sold"]  # negative non-core -> owner rule
    assert "U19" in d["sold"]  # still weak by percentile
    assert "U12" not in d["sold"] and "U12" in d["kept"]  # core negative protected
    assert set(d["kept"]) >= {"U00", "U05", "U12"}

    # Without protection, the same core negative is dropped by the owner rule.
    res2 = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        max_new=0,
        sell_negative_noncore=True,
        protect_core=False,
        core_list=["U12"],
    )
    assert "U12" in res2["diagnostics"]["sold"]


def test_core_floor_raises_gated_core_preserving_gross():
    """core_floor lifts a gate-trimmed core name to its floor, rescaling non-core.

    The conviction gate sizes mid-conviction U07 well below 35%; the floor lifts it
    back to 35% and scales the non-core names down so gross is unchanged.
    """
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.25, "U03": 0.25, "U07": 0.25, "U10": 0.20})
    kw = {
        "max_new": 0,
        "gross_target": 0.95,
        "name_cap": 1.0,
        "sector_cap": 1.0,
        "usd_bloc_cap": 1.0,
        "vol_ceiling": 5.0,
    }
    base = build_overlay(sc, current, pd.DataFrame(), **kw)
    floored = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        core_list=["U07"],
        core_floor=pd.Series({"U07": 0.35}),
        **kw,
    )
    wb, wf = base["weights"], floored["weights"]
    assert float(wf["U07"]) == pytest.approx(0.35, abs=1e-6)  # lifted to floor
    assert float(wf["U07"]) > float(wb.get("U07", 0.0))  # above the un-floored size
    assert floored["diagnostics"]["core_floor_applied"].get("U07", 0.0) > 0.0
    assert float(wf.sum()) == pytest.approx(float(wb.sum()), abs=1e-3)  # gross preserved


def test_already_strong_book_has_no_sells():
    held = [f"H{i:02d}" for i in range(6)]
    fill = [f"L{i:02d}" for i in range(14)]
    convs = list(np.linspace(2.0, 1.5, 6)) + list(np.linspace(0.5, -2.0, 14))
    sc = _scored(held + fill, convs)
    current = pd.Series(dict.fromkeys(held, 0.1))  # every holding is top-decile conviction

    d = build_overlay(sc, current, pd.DataFrame())["diagnostics"]
    assert d["n_sell"] == 0
    assert set(d["sold"]) == set()
    assert d["n_keep"] == 6


# --------------------------------------------------------------------------- #
# BUY selection
# --------------------------------------------------------------------------- #


def test_strong_non_held_bought_up_to_max_new():
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U10": 0.3, "U12": 0.3})  # middling holdings -> kept
    res = build_overlay(sc, current, pd.DataFrame(), max_new=2)
    d = res["diagnostics"]
    w = res["weights"]

    # {U00, U01, U02} clear the 0.85 percentile; only the top-2 are added.
    assert set(d["bought"]) == {"U00", "U01"}
    assert d["n_buy"] == 2 and len(d["bought"]) <= 2
    assert w.get("U00", 0.0) > 0.0 and w.get("U01", 0.0) > 0.0
    assert float(w.get("U02", 0.0)) == 0.0  # exceeded max_new -> not added
    assert "U02" not in d["bought"]


def test_only_non_held_names_are_buy_candidates():
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.4})  # holds the very strongest name already
    d = build_overlay(sc, current, pd.DataFrame(), max_new=5)["diagnostics"]
    assert "U00" not in d["bought"]  # held names never appear as buys
    assert "U00" in d["kept"]


# --------------------------------------------------------------------------- #
# KEEP anchoring + turnover
# --------------------------------------------------------------------------- #


def test_keep_anchors_to_current_weight_when_gate_is_noop():
    # 14 held names (7 USD + 7 EUR, sectors spread) at 5% each -> gross 0.70. With
    # beta 1.0 the book's vol (~0.13) and every cap sit under their limits, so the
    # gate is a no-op and kept weights must equal the current weights exactly.
    held = [f"H{i:02d}" if i < 7 else f"H{i:02d}.DE" for i in range(14)]
    fill = [f"L{i:02d}" for i in range(6)]
    convs = list(np.linspace(2.0, 1.0, 14)) + list(np.linspace(-1.0, -2.0, 6))
    sectors = [_SECTORS[i % 5] for i in range(20)]
    sc = _scored(held + fill, convs, betas=1.0, sectors=sectors)
    current = pd.Series(dict.fromkeys(held, 0.05))

    res = build_overlay(sc, current, pd.DataFrame(), max_new=0, gross_target=0.70)
    d = res["diagnostics"]
    assert d["n_sell"] == 0 and d["n_buy"] == 0
    np.testing.assert_allclose(res["weights"].reindex(held).to_numpy(), 0.05, atol=1e-3)
    assert d["turnover"] == pytest.approx(0.0, abs=1e-3)


def test_turnover_equals_half_sum_abs_delta():
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U10": 0.3, "U12": 0.3, "U19": 0.1})
    res = build_overlay(sc, current, pd.DataFrame(), max_new=3)
    w = res["weights"]
    idx = w.index.union(current.index)
    expected = 0.5 * float(
        (w.reindex(idx).fillna(0.0) - current.reindex(idx).fillna(0.0)).abs().sum()
    )
    assert res["diagnostics"]["turnover"] == pytest.approx(expected)
    assert res["diagnostics"]["turnover"] >= 0.0


# --------------------------------------------------------------------------- #
# Risk gate enforcement on the overlay book
# --------------------------------------------------------------------------- #


def test_gate_enforces_vol_ceiling_and_caps():
    # 15 hot (beta 1.6) held names -> book vol far over the 18% ceiling, but enough
    # names that proportions stay inside the caps. The gate must pull vol <= ceiling
    # and leave every concentration cap satisfied.
    held = [f"H{i:02d}" if i < 8 else f"H{i:02d}.DE" for i in range(15)]
    fill = [f"L{i:02d}" for i in range(5)]
    convs = list(np.linspace(2.0, 1.0, 15)) + list(np.linspace(-1.0, -2.0, 5))
    sectors = [_SECTORS[i % 5] for i in range(20)]
    sc = _scored(held + fill, convs, betas=1.6, sectors=sectors)
    current = pd.Series(dict.fromkeys(held, 0.06))  # gross 0.90, very hot book

    res = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        max_new=0,
        vol_ceiling=0.18,
        name_cap=0.08,
        sector_cap=0.25,
        usd_bloc_cap=0.60,
    )
    g = res["diagnostics"]["gate"]
    assert g["vol_after"] <= 0.18 + 1e-6
    assert g["caps_ok"] is True
    assert g["max_name"] <= 0.08 + 1e-6
    assert g["max_sector"] <= 0.25 + 1e-6
    assert g["usd_bloc"] <= 0.60 + 1e-6
    assert g["levers_fired"]  # the hot book must have tripped at least one lever


# --------------------------------------------------------------------------- #
# Core-retention reporting
# --------------------------------------------------------------------------- #


def test_core_retention_tracks_kept_sold_not_held():
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.3, "U19": 0.3})  # U00 kept, U19 sold
    res = build_overlay(sc, current, pd.DataFrame(), core_list=["U00", "U19", "U05"])
    cr = res["diagnostics"]["core_retention"]
    assert cr["per_name"]["U00"] == "kept"
    assert cr["per_name"]["U19"] == "sold"
    assert cr["per_name"]["U05"] == "not_held"  # in core list but not in the book
    assert cr["n_kept"] == 1 and cr["n_sold"] == 1


def test_no_core_retention_key_without_core_list():
    _tks, _convs, sc = _universe20()
    res = build_overlay(sc, pd.Series({"U10": 0.5}), pd.DataFrame())
    assert "core_retention" not in res["diagnostics"]


# --------------------------------------------------------------------------- #
# FIX A — never buy the same company twice / a dual listing of a held name
# --------------------------------------------------------------------------- #


def test_buy_dedups_cross_root_dual_listing():
    """Two listings of one company (cross-root AMC.AX / AMCR) -> only one bought.

    The suffix-stripped roots differ (AMC vs AMCR), so the mother-market util
    cannot merge them; the shared company name ("Amcor plc") must. Only one of
    the two may be bought.
    """
    tks = ["AMC.AX", "AMCR"] + [f"U{i:02d}" for i in range(18)]
    convs = [2.0, 1.95] + list(np.linspace(1.0, -2.0, 18))
    sc = _scored(tks, convs)
    sc.loc["AMC.AX", "name"] = "Amcor plc"
    sc.loc["AMCR", "name"] = "Amcor plc"
    current = pd.Series({"U05": 0.3})  # a middling holding so there is a book

    d = build_overlay(sc, current, pd.DataFrame(), max_new=8)["diagnostics"]
    amcor = [t for t in d["bought"] if t in ("AMC.AX", "AMCR")]
    assert len(amcor) == 1, f"bought both Amcor listings: {d['bought']}"


def test_buy_dedups_same_root_dual_listing_keeps_mother():
    """Same-root dual listing among candidates collapses to the mother market.

    ABVX (bare ADR) has the higher conviction, but the shared util keeps the
    Paris (home) listing ABVX.PA — mother-market preference overrides conviction.
    """
    tks = ["ABVX", "ABVX.PA"] + [f"U{i:02d}" for i in range(18)]
    convs = [2.0, 1.95] + list(np.linspace(1.0, -2.0, 18))
    sc = _scored(tks, convs)
    sc.loc["ABVX", "name"] = "Abivax SA"
    sc.loc["ABVX.PA", "name"] = "Abivax SA"
    sc["country"] = "France"  # domicile -> mother market is the Paris listing
    current = pd.Series({"U05": 0.3})

    d = build_overlay(sc, current, pd.DataFrame(), max_new=8)["diagnostics"]
    dup = [t for t in d["bought"] if t in ("ABVX", "ABVX.PA")]
    assert dup == ["ABVX.PA"], f"expected only the Paris mother listing, got {dup}"


def test_buy_skips_candidate_already_held_as_dual_listing():
    """A strong buy candidate that is a dual listing of a HELD name is skipped.

    We already hold AMCR (the US line of Amcor); the ASX line AMC.AX must not be
    bought (would double the company), while an unrelated strong name still is.
    """
    tks = ["AMC.AX", "AMCR", "BUYME"] + [f"U{i:02d}" for i in range(17)]
    convs = [2.0, 1.9, 1.8] + list(np.linspace(0.5, -2.0, 17))
    sc = _scored(tks, convs)
    sc.loc["AMC.AX", "name"] = "Amcor plc"
    sc.loc["AMCR", "name"] = "Amcor plc"
    sc.loc["BUYME", "name"] = "Buy Me Inc"
    current = pd.Series({"AMCR": 0.3})  # already hold Amcor via the US listing

    d = build_overlay(sc, current, pd.DataFrame(), max_new=8)["diagnostics"]
    assert "AMC.AX" not in d["bought"]  # same company as a held name -> skipped
    assert "BUYME" in d["bought"]  # the unrelated strong name is still bought
    assert "AMCR" in d["kept"]

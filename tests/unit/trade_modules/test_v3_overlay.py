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


def test_tier_name_caps_hold_small_caps_to_their_market_cap_tier():
    """tier_name_caps=True sizes each name by its market-cap tier: a high-conviction
    $0.5B small-cap is capped at 0.5% instead of the flat name_cap. Off = legacy."""
    tks = ["MEGA", "BIG", "SMALL"]
    sc = _scored(tks, [2.0, 1.8, 1.9])
    sc["cap"] = [3.0e12, 5.0e10, 0.5e9]  # $3T mega (10%), $50B large (6%), $0.5B small (0.5%)
    current = pd.Series(dtype=float)  # empty book -> all three are buys
    kw = {
        "max_new": 8,
        "buy_pctile": 0.0,  # all positive-conviction names qualify as buys
        "gross_target": 0.95,
        "name_cap": 0.10,
        "sector_cap": 0.99,
        "usd_bloc_cap": 0.99,
    }
    on = build_overlay(sc, current, pd.DataFrame(), tier_name_caps=True, **kw)["weights"]
    off = build_overlay(sc, current, pd.DataFrame(), tier_name_caps=False, **kw)["weights"]
    assert on.get("SMALL", 0.0) <= 0.005 + 1e-6  # $0.5B -> 0.5% tier
    assert off.get("SMALL", 0.0) > on.get("SMALL", 0.0)  # tiers trimmed the small-cap


def test_sell_negative_noncore_deadband_is_noncore_and_protect_core_spares_percentile():
    """Two distinct owner rules on the core sleeve.

    The ``sell_negative_noncore`` deadband is NON-CORE by definition: it never
    sells a core-listed name, independent of ``protect_core``. ``protect_core``
    is what additionally spares the core from the percentile sell.
    """
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.2, "U05": 0.2, "U11": 0.2, "U12": 0.2, "U19": 0.2})

    # protect_core ON, U12 core: non-core U11 (-0.32) dropped by the deadband,
    # U19 (-2.0) by the percentile, core U12 (-0.53) spared.
    d = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        max_new=0,  # isolate the SELL side
        sell_negative_noncore=True,
        protect_core=True,
        core_list=["U12"],
    )["diagnostics"]
    assert "U11" in d["sold"]  # negative non-core -> deadband
    assert "U19" in d["sold"]  # bottom-percentile non-core
    assert "U12" not in d["sold"] and "U12" in d["kept"]  # core spared
    assert set(d["kept"]) >= {"U00", "U05", "U12"}

    # protect_core OFF: the deadband STILL never touches core (U12 stays), but the
    # percentile now sells a bottom-tier core name (U19) that protection would keep.
    d2 = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        max_new=0,
        sell_negative_noncore=True,
        protect_core=False,
        core_list=["U12", "U19"],
    )["diagnostics"]
    assert "U12" not in d2["sold"]  # deadband is non-core -> core spared regardless
    assert "U19" in d2["sold"]  # unprotected bottom-tier core still sold by percentile


def test_noncore_sell_floor_deadband_keeps_near_neutral():
    """Deadband: sell_negative_noncore only sells non-core names BELOW the floor.

    U11 (~-0.32) is near-neutral and above a -0.5 floor -> KEEP; U14 (~-0.95) is
    clearly weak and below it -> SELL. The default floor 0.0 sells both.
    """
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.25, "U11": 0.20, "U14": 0.20, "U19": 0.35})
    band = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        max_new=0,
        sell_negative_noncore=True,
        noncore_sell_floor=-0.5,
    )["diagnostics"]
    assert "U11" not in band["sold"]  # near-neutral kept (inside the deadband)
    assert "U14" in band["sold"]  # clearly weak sold
    assert "U19" in band["sold"]  # bottom-percentile sold regardless
    # Without the deadband (floor 0.0) the near-neutral name is also sold.
    nofloor = build_overlay(sc, current, pd.DataFrame(), max_new=0, sell_negative_noncore=True)[
        "diagnostics"
    ]
    assert "U11" in nofloor["sold"]


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


def test_core_floor_underfunded_preserves_gross_via_haircut():
    """When non-core cannot fund the floor, lifts are haircut so gross is PRESERVED,
    never inflated. A tiny non-core sleeve cannot fund a large core floor."""
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U02": 0.20, "U15": 0.05})  # U02 core, U15 tiny non-core funder
    kw = {
        "gross_target": 0.30,
        "name_cap": 1.0,
        "sector_cap": 1.0,
        "usd_bloc_cap": 1.0,
        "vol_ceiling": 5.0,
        "max_new": 0,
    }
    base = build_overlay(sc, current, pd.DataFrame(), **kw)
    floored = build_overlay(
        sc,
        current,
        pd.DataFrame(),
        core_list=["U02"],
        core_floor=pd.Series({"U02": 0.90}),
        **kw,
    )
    gb, gf = float(base["weights"].sum()), float(floored["weights"].sum())
    assert gf == pytest.approx(gb, abs=1e-3)  # gross preserved, NOT inflated
    assert gf <= kw["gross_target"] + 1e-3  # never over-invested
    assert float(floored["weights"].get("U15", 0.0)) == pytest.approx(
        0.0, abs=1e-6
    )  # non-core spent
    assert float(floored["weights"]["U02"]) < 0.90  # lift haircut below the full floor


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


# --------------------------------------------------------------------------- #
# FIX-NOW (D4/D8): squeeze-exclude gate + de/current_ratio distress filter on BUYS
# --------------------------------------------------------------------------- #


def test_fixnow_squeeze_gate_excludes_high_si_buy_candidate():
    """A strong non-held candidate with short interest above the squeeze threshold
    is NOT bought (squeeze/borrow risk); clean candidates are bought instead. The
    screen applies to BUYS only — it never force-sells a holding."""
    _tks, _convs, sc = _universe20()
    sc["short_interest"] = np.nan
    sc.loc["U00", "short_interest"] = 40.0  # % of float, above the 20% default
    current = pd.Series({"U10": 0.3})  # a middling holding

    d = build_overlay(sc, current, pd.DataFrame(), max_new=2)["diagnostics"]
    assert "U00" not in d["bought"], "high-short-interest name must be screened from buys"
    assert set(d["bought"]) == {"U01", "U02"}
    assert "U00" in d.get("screened_out", [])


def test_fixnow_distress_filter_excludes_low_current_ratio_buy():
    """A strong non-held candidate with current_ratio below the distress floor
    (< 1.0 = cannot cover short-term liabilities) is not bought."""
    _tks, _convs, sc = _universe20()
    sc["current_ratio"] = np.nan
    sc.loc["U00", "current_ratio"] = 0.5  # distressed liquidity
    current = pd.Series({"U10": 0.3})

    d = build_overlay(sc, current, pd.DataFrame(), max_new=2)["diagnostics"]
    assert "U00" not in d["bought"]
    assert set(d["bought"]) == {"U01", "U02"}


def test_fixnow_de_distress_screen_only_when_threshold_set():
    """The debt/equity leg is opt-in (max_de): OFF by default (scale unconfirmed),
    active only when a threshold is passed."""
    _tks, _convs, sc = _universe20()
    sc["de"] = np.nan
    sc.loc["U00", "de"] = 5.0
    current = pd.Series({"U10": 0.3})

    # default: de screen OFF -> U00 (strongest) is bought.
    off = build_overlay(sc, current, pd.DataFrame(), max_new=2)["diagnostics"]
    assert "U00" in off["bought"]
    # threshold set -> U00 excluded as over-levered.
    on = build_overlay(sc, current, pd.DataFrame(), max_new=2, max_de=2.0)["diagnostics"]
    assert "U00" not in on["bought"]
    assert set(on["bought"]) == {"U01", "U02"}


def test_fixnow_screen_tolerates_missing_data_and_spares_holdings():
    """No screen columns / NaN values -> no exclusion (buys unchanged); a HELD name
    with high SI is never force-sold by the screen (buys-only)."""
    _tks, _convs, sc = _universe20()  # no short_interest/current_ratio/de columns
    current = pd.Series({"U10": 0.3})
    base = build_overlay(sc, current, pd.DataFrame(), max_new=2)["diagnostics"]
    assert set(base["bought"]) == {"U00", "U01"}  # unchanged from the no-screen contract

    # A held, squeeze-risky name is NOT sold by the screen (screen only gates buys).
    sc2 = sc.copy()
    sc2["short_interest"] = np.nan
    sc2.loc["U10", "short_interest"] = 90.0  # held + very high SI
    held = build_overlay(sc2, current, pd.DataFrame(), max_new=2)["diagnostics"]
    assert "U10" not in held["sold"] and "U10" in held["kept"]

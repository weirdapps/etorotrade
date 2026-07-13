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

"""TDD tests for trade_modules.v3.construct.build_portfolio.

Bridges the v3 combiner output (scored frame with conviction/eligible/sector/
price/beta/country) to the riskfirst construction PRIMITIVES:

    dedup dual listings -> select top_n by conviction -> ERC weights
    -> conviction-tilt blend -> name / USD-bloc / sector caps
    -> deploy to gross_target

and layers a report-only risk gate (CVaR, net beta, effective bets) on top.
Phase 2.x: conviction participates in sizing (Change 1); regime deployment
85-95% replaces fractional Kelly (Change 2); dual listings are deduped to the
mother market (Change 3).
"""

import numpy as np
import pandas as pd

from trade_modules.v3.construct import build_portfolio, dedup_dual_listings

# A rich synthetic universe: 9 USD-bloc + 3 EUR names; 6 Tech (sector-cap bait).
_USD = ["AA1", "AA2", "AA3", "AA4", "AA5", "AA6", "AA7", "AA8", "AA9"]
_EUR = ["E1.DE", "E2.DE", "E3.PA"]
_ALL = _USD + _EUR
_SECTORS = {
    "AA1": "Tech",
    "AA2": "Tech",
    "AA3": "Tech",
    "AA4": "Tech",
    "AA5": "Tech",
    "AA6": "Tech",
    "AA7": "Health",
    "AA8": "Health",
    "AA9": "Health",
    "E1.DE": "Financials",
    "E2.DE": "Energy",
    "E3.PA": "Utilities",
}


def _make_prices(tickers, days=300, seed=0, vols=None):
    """Synthetic GBM daily closes (dates x tickers)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=days, freq="B")
    data = {}
    for t in tickers:
        vol = 0.01 if vols is None else vols.get(t, 0.01)
        rets = rng.normal(0.0003, vol, days)
        data[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=idx)


def _make_scored(tickers=_ALL, *, betas=1.0, eligible=None, convictions=None):
    """A scored frame in compute_scores shape (conviction/eligible/sector/price/beta)."""
    n = len(tickers)
    df = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    # Descending conviction by list order (AA1 best), unless overridden.
    df["conviction"] = list(np.linspace(2.0, -2.0, n)) if convictions is None else convictions
    df["eligible"] = [True] * n if eligible is None else eligible
    df["sector"] = [_SECTORS[t] for t in tickers]
    df["price"] = 100.0
    df["beta"] = betas if isinstance(betas, (list, tuple)) else [betas] * n
    df["cap"] = [1e11] * n
    return df


# --------------------------------------------------------------------------- #
# Core construction contract
# --------------------------------------------------------------------------- #


def test_weights_sum_to_gross_and_no_leverage():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=1)
    res = build_portfolio(scored, prices, top_n=12, gross_target=1.0, cvar_budget=10.0)
    w = res["weights"]
    assert isinstance(w, pd.Series)
    np.testing.assert_allclose(w.sum(), res["gross"], rtol=1e-9)
    assert res["gross"] <= 1.0 + 1e-9
    np.testing.assert_allclose(res["cash"], max(0.0, 1.0 - res["gross"]), atol=1e-9)
    assert (w >= -1e-12).all()  # long only


def test_only_eligible_names_are_selected():
    elig = [True] * len(_ALL)
    elig[0] = False  # AA1 (highest conviction) ineligible
    elig[1] = False  # AA2 ineligible
    scored = _make_scored(eligible=elig)
    prices = _make_prices(_ALL, seed=2)
    res = build_portfolio(scored, prices, top_n=12, gross_target=1.0, cvar_budget=10.0)
    assert "AA1" not in res["selected"]
    assert "AA2" not in res["selected"]
    # Everything selected is eligible.
    assert all(bool(scored.loc[t, "eligible"]) for t in res["selected"])


# --------------------------------------------------------------------------- #
# Change 1 — conviction-tilt sizing (conviction participates in weights)
# --------------------------------------------------------------------------- #


def _small_scored(tickers, convictions, betas, sectors=None, country=None):
    df = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    df["conviction"] = convictions
    df["eligible"] = True
    df["sector"] = sectors if sectors is not None else ["Tech"] * len(tickers)
    df["price"] = 100.0
    df["beta"] = betas
    if country is not None:
        df["country"] = country
    return df


def test_conviction_tilt_higher_conviction_gets_more_weight_equal_vol():
    # Equal betas + single-factor fallback (empty prices) => ERC is equal-weight,
    # so any weight difference is driven purely by the conviction tilt.
    df = _small_scored(["A", "B"], convictions=[2.0, 1.0], betas=[1.0, 1.0])
    res = build_portfolio(
        df,
        pd.DataFrame(),
        top_n=2,
        conviction_weight=0.5,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    assert w["A"] > w["B"]  # higher conviction -> larger weight


def test_high_conviction_high_vol_name_is_not_the_smallest():
    # AA1 is the #1-conviction name AND the highest-vol name (beta 2.5 vs 1.0).
    # Single-factor fallback (empty prices) gives a clean ERC where AA1 is the
    # smallest weight (the MU problem); the conviction tilt must lift it.
    scored = _make_scored(betas=[2.5] + [1.0] * (len(_ALL) - 1))
    kw = {
        "top_n": 12,
        "name_cap": 1.0,
        "sector_cap": 1.0,
        "usd_bloc_cap": 1.0,
        "gross_target": 1.0,
        "cvar_budget": 10.0,
    }
    erc_only = build_portfolio(scored, pd.DataFrame(), conviction_weight=0.0, **kw)
    tilted = build_portfolio(scored, pd.DataFrame(), conviction_weight=0.5, **kw)
    w0, w1 = erc_only["weights"], tilted["weights"]
    # Pure ERC: the high-vol name is the smallest weight.
    assert w0["AA1"] == w0.min()
    # Conviction tilt strictly increases its weight and it is no longer smallest.
    assert w1["AA1"] > w0["AA1"]
    assert w1["AA1"] > w1.min() + 1e-9


def test_conviction_weight_zero_reproduces_pure_erc():
    """conviction_weight=0 must reproduce the pure ERC -> caps -> deploy path."""
    from trade_modules.riskfirst.construct import apply_name_cap, cap_groups, erc_weights
    from trade_modules.riskfirst.fx import USD_BLOC, cap_bloc, currency_of
    from trade_modules.v3.construct import _make_cov_fn

    scored = _make_scored()
    prices = _make_prices(_ALL, seed=21)
    name_cap, sector_cap, usd_cap, gt = 0.15, 0.30, 0.70, 1.0
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        conviction_weight=0.0,
        name_cap=name_cap,
        sector_cap=sector_cap,
        usd_bloc_cap=usd_cap,
        gross_target=gt,
        cvar_budget=10.0,
    )
    selected = res["selected"]

    # Rebuild the primitive ERC -> caps -> deploy path on the same selected set.
    sub = scored.loc[selected].copy()
    sub["SECTOR"] = sub["sector"].astype("string").str.upper()
    cov = _make_cov_fn(prices, sub)(selected)
    w = erc_weights(cov)
    w = w / w.sum()
    w = apply_name_cap(w, name_cap)
    is_bloc = np.array([currency_of(t) in USD_BLOC for t in selected])
    w = cap_bloc(w, is_bloc, usd_cap)
    w = apply_name_cap(w, name_cap)
    w = cap_groups(w, sub["SECTOR"].astype(str).to_numpy(), sector_cap)
    w = apply_name_cap(w, name_cap)
    w = w / w.sum() * gt

    np.testing.assert_allclose(res["weights"].loc[selected].to_numpy(), w, rtol=1e-9, atol=1e-12)


def test_all_nonpositive_conviction_falls_back_to_uniform():
    """All-<=0 conviction must not divide by zero; the tilt component is uniform.

    With equal-vol fallback ERC is already uniform, so at conviction_weight=1.0
    the guarded uniform w_conv yields ~equal weights (no NaNs)."""
    df = _small_scored(["A", "B", "C"], convictions=[-0.5, -1.0, -2.0], betas=[1.0, 1.0, 1.0])
    res = build_portfolio(
        df,
        pd.DataFrame(),  # single-factor fallback -> equal ERC
        top_n=3,
        conviction_weight=1.0,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    assert not w.isna().any()
    sel = w[w > 1e-9]
    assert sel.std() / sel.mean() < 0.05  # uniform tilt -> ~equal weights


# --------------------------------------------------------------------------- #
# Binding caps (still hold under the conviction tilt; gross_target=1.0)
# --------------------------------------------------------------------------- #


def test_name_cap_respected():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=3)
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=0.10,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    assert res["weights"].max() <= 0.10 + 1e-6


def test_sector_cap_respected():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=4)
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=0.30,
        sector_cap=0.25,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    # 6 Tech names would otherwise be ~50% of the book -> capped to 0.25.
    for sector, exp in res["sector_exposures"].items():
        assert exp <= 0.25 + 1e-6, f"{sector} over sector cap: {exp}"
    assert res["sector_exposures"]["TECH"] <= 0.25 + 1e-6


def test_usd_bloc_cap_respected():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=5)
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=0.60,
        sector_cap=1.0,
        usd_bloc_cap=0.60,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    # 9 of 12 names are USD-bloc (natural ~0.75); the 0.60 cap must bind.
    assert res["usd_bloc"] <= 0.60 + 1e-6


# --------------------------------------------------------------------------- #
# Risk gate diagnostics
# --------------------------------------------------------------------------- #


def test_risk_gate_diagnostics_present_and_in_band():
    scored = _make_scored(betas=1.0)
    prices = _make_prices(_ALL, seed=6)
    res = build_portfolio(scored, prices, top_n=12, gross_target=1.0, cvar_budget=10.0)
    d = res["diagnostics"]
    assert d["cvar_95_risk_book"] > 0.0
    assert d["effective_bets"] > 1.0
    assert 0.3 <= d["net_beta"] <= 1.1  # betas ~1.0
    assert d["binding"]["net_beta"] is False
    for k in ("cvar", "net_beta", "effective_bets"):
        assert k in d["binding"]


def test_cvar_matches_parametric_formula():
    scored = _make_scored(betas=1.0)
    prices = _make_prices(_ALL, seed=6)
    res = build_portfolio(scored, prices, top_n=12, gross_target=1.0, cvar_budget=10.0)
    d = res["diagnostics"]
    # cvar_95_risk_book == 2.063 * annualized (gross=1.0) portfolio vol.
    assert abs(d["cvar_95_risk_book"] - 2.063 * d["port_vol"]) < 1e-9


def test_net_beta_flag_trips_when_out_of_band():
    scored = _make_scored(betas=1.6)  # high-beta book
    prices = _make_prices(_ALL, seed=7)
    res = build_portfolio(scored, prices, top_n=12, gross_target=1.0, cvar_budget=10.0)
    d = res["diagnostics"]
    assert d["net_beta"] > 1.1
    assert d["binding"]["net_beta"] is True


def test_effective_bets_flag_trips_when_concentrated():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=8)
    res = build_portfolio(
        scored,
        prices,
        top_n=6,
        name_cap=0.30,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    d = res["diagnostics"]
    assert d["effective_bets"] < 12
    assert d["binding"]["effective_bets"] is True


# --------------------------------------------------------------------------- #
# Change 2 — regime deployment (replaces Kelly); CVaR is report-only
# --------------------------------------------------------------------------- #


def test_gross_target_sets_deployment_and_cash():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=40)
    for gt in (0.85, 0.90, 0.95):
        res = build_portfolio(scored, prices, top_n=12, gross_target=gt, cvar_budget=10.0)
        assert abs(res["gross"] - gt) < 1e-9
        assert abs(res["weights"].sum() - gt) < 1e-9
        assert abs(res["cash"] - (1.0 - gt)) < 1e-9


def test_cvar_deployed_scales_with_gross_target():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=41)
    full = build_portfolio(scored, prices, top_n=12, gross_target=1.0, cvar_budget=10.0)
    dep = build_portfolio(scored, prices, top_n=12, gross_target=0.90, cvar_budget=10.0)
    # Risk-book CVaR (gross=1.0 sleeve) is invariant to the deployment target.
    np.testing.assert_allclose(
        full["diagnostics"]["cvar_95_risk_book"],
        dep["diagnostics"]["cvar_95_risk_book"],
        rtol=1e-9,
    )
    # Deployed CVaR scales linearly with gross_target.
    np.testing.assert_allclose(
        dep["diagnostics"]["cvar_95_deployed"],
        0.90 * dep["diagnostics"]["cvar_95_risk_book"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        full["diagnostics"]["cvar_95_deployed"],
        full["diagnostics"]["cvar_95_risk_book"],
        rtol=1e-9,
    )


def test_cvar_budget_binding_flag_only_no_shrink():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=9)
    gt = 0.90
    hi = build_portfolio(scored, prices, top_n=12, gross_target=gt, cvar_budget=10.0)
    lo = build_portfolio(scored, prices, top_n=12, gross_target=gt, cvar_budget=0.02)
    assert hi["diagnostics"]["binding"]["cvar"] is False
    assert lo["diagnostics"]["binding"]["cvar"] is True
    # Report-only: a CVaR breach never shrinks gross below the deployment target.
    assert abs(lo["gross"] - gt) < 1e-9
    assert abs(hi["gross"] - gt) < 1e-9
    np.testing.assert_allclose(lo["gross"], hi["gross"], rtol=1e-9)


# --------------------------------------------------------------------------- #
# Covariance: empirical shrunk cov vs single-factor fallback (pure ERC)
# --------------------------------------------------------------------------- #


def test_empirical_cov_drives_weights_down_for_high_vol_name():
    scored = _make_scored(betas=1.0)  # equal betas -> single-factor would be equal-weight
    prices = _make_prices(_ALL, seed=13, vols={"AA1": 0.05})  # AA1 5x vol
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        conviction_weight=0.0,  # isolate the covariance mechanism from the tilt
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    # ERC on the empirical cov must underweight the high-vol name vs a calm peer.
    assert w["AA1"] < w["AA2"]


def test_falls_back_to_single_factor_cov_when_prices_missing():
    scored = _make_scored(betas=1.0)
    empty = pd.DataFrame()  # no price history at all -> fallback for every name
    res = build_portfolio(
        scored,
        empty,
        top_n=12,
        conviction_weight=0.0,  # isolate the fallback cov from the tilt
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    assert not w.isna().any()
    assert res["gross"] > 0.0
    # Equal betas -> single-factor cov is (near) equal-variance -> ~equal ERC weights.
    sel_w = w[w > 1e-9]
    assert sel_w.std() / sel_w.mean() < 0.05


# --------------------------------------------------------------------------- #
# Degenerate path: all-ineligible / empty scored
# --------------------------------------------------------------------------- #


def test_all_ineligible_returns_all_cash_with_full_diagnostics():
    """build_portfolio with all-ineligible scored must return a well-formed all-cash result."""
    scored = _make_scored(eligible=[False] * len(_ALL))
    prices = _make_prices(_ALL, seed=99)
    res = build_portfolio(scored, prices, cvar_budget=0.25)

    assert isinstance(res["weights"], pd.Series)
    assert len(res["weights"]) == 0
    assert res["gross"] == 0.0
    assert res["cash"] == 1.0
    assert res["selected"] == []

    d = res["diagnostics"]
    required_keys = (
        "cvar_95_risk_book",
        "cvar_95_deployed",
        "cvar_budget",
        "net_beta",
        "net_beta_band",
        "effective_bets",
        "port_vol",
        "gross_risk",
        "binding",
    )
    for key in required_keys:
        assert key in d, f"missing diagnostics key: {key!r}"
    for flag in ("cvar", "net_beta", "effective_bets"):
        assert flag in d["binding"], f"missing binding flag: {flag!r}"


# --------------------------------------------------------------------------- #
# net_beta NaN coercion
# --------------------------------------------------------------------------- #


def test_net_beta_nan_beta_coerced_to_one():
    """A NaN beta must be treated as 1.0 (consistent with cov_fn's fillna(1.0))."""
    tickers = ["A", "B", "C"]
    df = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    df["conviction"] = [2.0, 1.0, 0.0]
    df["eligible"] = True
    df["sector"] = ["Tech", "Tech", "Health"]
    df["price"] = 100.0
    df["beta"] = [1.5, np.nan, 0.5]  # B has NaN beta

    res = build_portfolio(
        df,
        pd.DataFrame(),  # force single-factor fallback
        top_n=3,
        conviction_weight=0.0,  # isolate the beta coercion from the tilt
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    d = res["diagnostics"]
    # With NaN coerced to 1.0 net_beta ~ weighted avg of [1.5, 1.0, 0.5] > 0.8.
    # Without coercion nansum would skip the NaN term, dragging it lower.
    assert d["net_beta"] > 0.8, (
        f"net_beta={d['net_beta']:.4f} too low; NaN beta may not be coerced to 1.0"
    )


def test_fallback_when_one_selected_name_lacks_history():
    scored = _make_scored(betas=1.0)
    prices = _make_prices([t for t in _ALL if t != "AA1"], seed=14)  # AA1 column absent
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    assert not w.isna().any()
    assert "AA1" in res["selected"]  # still selected (fallback cov keeps it in)
    np.testing.assert_allclose(w.sum(), res["gross"], rtol=1e-9)


# --------------------------------------------------------------------------- #
# Change 3 — dual-listing dedup (prefer mother market)
# --------------------------------------------------------------------------- #


def _dedup_frame(rows):
    """rows: list of (ticker, name, country)."""
    return pd.DataFrame(
        {"name": [r[1] for r in rows], "country": [r[2] for r in rows]},
        index=pd.Index([r[0] for r in rows], name="ticker"),
    )


def test_dedup_keeps_paris_listing_for_french_company():
    df = _dedup_frame([("ABVX", "Abivax SA", "France"), ("ABVX.PA", "Abivax SA", "France")])
    out = dedup_dual_listings(df)
    assert list(out.index) == ["ABVX.PA"]


def test_dedup_keeps_xetra_listing_for_german_company():
    df = _dedup_frame([("SAP", "SAP SE", "Germany"), ("SAP.DE", "SAP SE", "Germany")])
    out = dedup_dual_listings(df)
    assert list(out.index) == ["SAP.DE"]


def test_dedup_single_listing_kept():
    df = _dedup_frame([("AAPL", "Apple Inc", "United States")])
    out = dedup_dual_listings(df)
    assert list(out.index) == ["AAPL"]


def test_dedup_us_domicile_keeps_bare_over_foreign_twin():
    df = _dedup_frame(
        [("FOO", "Foo Corp", "United States"), ("FOO.L", "Foo Corp", "United States")]
    )
    out = dedup_dual_listings(df)
    assert list(out.index) == ["FOO"]


def test_dedup_different_companies_sharing_root_both_kept():
    # Same root BAR but different names AND different countries -> not the same company.
    df = _dedup_frame(
        [("BAR", "Bar Industries", "United States"), ("BAR.L", "Barclays PLC", "United Kingdom")]
    )
    out = dedup_dual_listings(df)
    assert set(out.index) == {"BAR", "BAR.L"}


def test_dedup_fallback_prefers_suffixed_when_country_missing():
    # Same name, country missing -> fallback prefers the suffixed home listing.
    df = _dedup_frame([("XYZ", "Xyz SA", None), ("XYZ.PA", "Xyz SA", None)])
    out = dedup_dual_listings(df)
    assert list(out.index) == ["XYZ.PA"]


def test_build_portfolio_dedups_dual_listing_before_selection():
    tickers = ["ABVX", "ABVX.PA", "AA2", "AA3"]
    df = _small_scored(
        tickers,
        convictions=[2.0, 2.0, 1.0, 0.5],
        betas=[1.0, 1.0, 1.0, 1.0],
        sectors=["Health", "Health", "Tech", "Tech"],
        country=["France", "France", "United States", "United States"],
    )
    df["name"] = ["Abivax SA", "Abivax SA", "Alpha", "Beta"]
    prices = _make_prices(tickers, seed=31)
    res = build_portfolio(
        df,
        prices,
        top_n=4,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        gross_target=1.0,
        cvar_budget=10.0,
    )
    # The mother (Paris) listing survives; the bare ADR is dropped before selection.
    assert "ABVX.PA" in res["weights"].index
    assert "ABVX" not in res["weights"].index
    assert "ABVX" not in res["selected"]

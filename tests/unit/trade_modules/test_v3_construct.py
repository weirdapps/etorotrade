"""TDD tests for trade_modules.v3.construct.build_portfolio.

Bridges the v3 combiner output (scored frame with conviction/eligible/sector/
price/beta) to the riskfirst construction stack (ERC -> vol target -> name /
sector / USD-bloc caps -> regime dial -> fractional Kelly) and layers a
report-only risk gate (CVaR, net beta, effective bets) on top.
"""

import numpy as np
import pandas as pd

from trade_modules.v3.construct import build_portfolio

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
    for i, t in enumerate(tickers):
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
    res = build_portfolio(
        scored, prices, top_n=12, target_vol=0.50, kelly_fraction=1.0, cvar_budget=10.0
    )
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
    res = build_portfolio(
        scored, prices, top_n=12, target_vol=0.50, kelly_fraction=1.0, cvar_budget=10.0
    )
    assert "AA1" not in res["selected"]
    assert "AA2" not in res["selected"]
    # Everything selected is eligible.
    assert all(bool(scored.loc[t, "eligible"]) for t in res["selected"])


# --------------------------------------------------------------------------- #
# Binding caps (kelly=1, no cvar shrink -> engine caps show through directly)
# --------------------------------------------------------------------------- #


def test_name_cap_respected():
    # name_cap must be feasible (top_n * cap >= gross) and is the sole active cap;
    # 0.10 binds here (uncapped ERC tops ~0.125) while staying feasible (12*0.10>1).
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=3)
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=0.10,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        target_vol=0.50,
        kelly_fraction=1.0,
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
        target_vol=0.50,
        kelly_fraction=1.0,
        cvar_budget=10.0,
    )
    # 6 Tech names would otherwise be ~50% of the book -> capped to 0.25.
    for sector, exp in res["sector_exposures"].items():
        assert exp <= 0.25 + 1e-6, f"{sector} over sector cap: {exp}"
    assert res["sector_exposures"]["TECH"] <= 0.25 + 1e-6


def test_usd_bloc_cap_respected():
    # Loose name_cap so the engine's post-bloc name-cap pass does not redistribute
    # back into the bloc; the USD-bloc cap is the sole active constraint.
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=5)
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=0.60,
        sector_cap=1.0,
        usd_bloc_cap=0.60,
        target_vol=0.50,
        kelly_fraction=1.0,
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
    res = build_portfolio(
        scored, prices, top_n=12, target_vol=0.12, kelly_fraction=1.0, cvar_budget=10.0
    )
    d = res["diagnostics"]
    assert d["cvar_95"] > 0.0
    assert d["effective_bets"] > 1.0
    assert 0.3 <= d["net_beta"] <= 1.1  # betas ~1.0
    assert d["binding"]["net_beta"] is False
    for k in ("cvar", "net_beta", "effective_bets"):
        assert k in d["binding"]


def test_cvar_matches_parametric_formula():
    scored = _make_scored(betas=1.0)
    prices = _make_prices(_ALL, seed=6)
    res = build_portfolio(
        scored, prices, top_n=12, target_vol=0.12, kelly_fraction=1.0, cvar_budget=10.0
    )
    d = res["diagnostics"]
    # cvar_95 == 2.063 * annualized portfolio vol (parametric-normal ES).
    assert abs(d["cvar_95"] - 2.063 * d["port_vol"]) < 1e-9


def test_net_beta_flag_trips_when_out_of_band():
    scored = _make_scored(betas=1.6)  # high-beta book
    prices = _make_prices(_ALL, seed=7)
    res = build_portfolio(
        scored, prices, top_n=12, target_vol=0.12, kelly_fraction=1.0, cvar_budget=10.0
    )
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
        target_vol=0.50,
        kelly_fraction=1.0,
        cvar_budget=10.0,
    )
    d = res["diagnostics"]
    assert d["effective_bets"] < 12
    assert d["binding"]["effective_bets"] is True


def test_cvar_budget_shrinks_gross_when_breached():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=9)
    hi = build_portfolio(
        scored, prices, top_n=12, target_vol=0.20, kelly_fraction=1.0, cvar_budget=10.0
    )
    lo = build_portfolio(
        scored, prices, top_n=12, target_vol=0.20, kelly_fraction=1.0, cvar_budget=0.02
    )
    assert hi["diagnostics"]["binding"]["cvar"] is False
    assert lo["diagnostics"]["binding"]["cvar"] is True
    assert lo["gross"] < hi["gross"]
    # after the soft-shrink the reported CVaR sits at (≈) the budget
    assert abs(lo["diagnostics"]["cvar_95"] - 0.02) < 1e-6


# --------------------------------------------------------------------------- #
# Gross dials: fractional Kelly + regime multiplier
# --------------------------------------------------------------------------- #


def test_kelly_fraction_scales_gross():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=10)
    full = build_portfolio(
        scored, prices, top_n=12, target_vol=0.50, kelly_fraction=1.0, cvar_budget=10.0
    )
    quarter = build_portfolio(
        scored, prices, top_n=12, target_vol=0.50, kelly_fraction=0.25, cvar_budget=10.0
    )
    np.testing.assert_allclose(quarter["gross"], 0.25 * full["gross"], rtol=1e-9)
    np.testing.assert_allclose(
        quarter["weights"].to_numpy(), 0.25 * full["weights"].to_numpy(), rtol=1e-9
    )


def test_kelly_capped_at_no_leverage():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=11)
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=0.30,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        target_vol=0.50,
        kelly_fraction=5.0,
        cvar_budget=10.0,
    )
    assert res["gross"] <= 1.0 + 1e-9  # kelly>1 cannot lever past gross 1


def test_regime_multiplier_scales_gross():
    scored = _make_scored()
    prices = _make_prices(_ALL, seed=12)
    base = build_portfolio(
        scored,
        prices,
        top_n=12,
        target_vol=0.50,
        kelly_fraction=1.0,
        regime_multiplier=1.0,
        cvar_budget=10.0,
    )
    half = build_portfolio(
        scored,
        prices,
        top_n=12,
        target_vol=0.50,
        kelly_fraction=1.0,
        regime_multiplier=0.5,
        cvar_budget=10.0,
    )
    np.testing.assert_allclose(half["gross"], 0.5 * base["gross"], rtol=1e-9)


# --------------------------------------------------------------------------- #
# Covariance: empirical shrunk cov vs single-factor fallback
# --------------------------------------------------------------------------- #


def test_empirical_cov_drives_weights_down_for_high_vol_name():
    scored = _make_scored(betas=1.0)  # equal betas -> single-factor would be equal-weight
    prices = _make_prices(_ALL, seed=13, vols={"AA1": 0.05})  # AA1 5x vol
    res = build_portfolio(
        scored,
        prices,
        top_n=12,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        target_vol=0.50,
        kelly_fraction=1.0,
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
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        target_vol=0.50,
        kelly_fraction=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    assert not w.isna().any()
    assert res["gross"] > 0.0
    # Equal betas -> single-factor cov is (near) equal-variance -> ~equal ERC weights.
    sel_w = w[w > 1e-9]
    assert sel_w.std() / sel_w.mean() < 0.05


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
        target_vol=0.50,
        kelly_fraction=1.0,
        cvar_budget=10.0,
    )
    w = res["weights"]
    assert not w.isna().any()
    assert "AA1" in res["selected"]  # still selected (fallback cov keeps it in)
    np.testing.assert_allclose(w.sum(), res["gross"], rtol=1e-9)

"""TDD tests for trade_modules.v3.risk_gate.apply_risk_gate (Phase 5A).

The HARD risk gate enforces (blocking) a portfolio vol ceiling and the
concentration caps on an already-constructed book, using the owner's HYBRID
rule:

  1. caps-to-convergence — iterate name / USD-bloc / sector caps until no cap is
     left breached;
  2. vol ceiling via tail de-weighting (lever 1) — de-weight the worst
     tail-risk names first, keeping deployment constant;
  3. gross cut (lever 2, fallback) — only if de-weighting is exhausted, scale the
     whole book down so vol == ceiling (deployment drops below target).

These tests pin the enforcement contract, the lever order, cap convergence, the
identity behaviour on a compliant book, and loop termination.
"""

import numpy as np
import pandas as pd

from trade_modules.riskfirst.construct import portfolio_vol
from trade_modules.v3.risk_gate import apply_risk_gate

_TOL = 1e-6


def _cov(vols, corr=0.0):
    """Annualised covariance from per-name vols and a common pairwise corr."""
    vols = np.asarray(vols, dtype=float)
    n = len(vols)
    C = np.full((n, n), float(corr))
    np.fill_diagonal(C, 1.0)
    return np.outer(vols, vols) * C


def _series(vals, prefix="T"):
    vals = np.asarray(vals, dtype=float)
    return pd.Series(vals, index=[f"{prefix}{i}" for i in range(len(vals))])


def _sector_props(sectors, p):
    tot: dict = {}
    for lab, wt in zip(sectors, p, strict=True):
        tot[lab] = tot.get(lab, 0.0) + float(wt)
    return tot


# --------------------------------------------------------------------------- #
# (a) vol ceiling enforced (tail de-weighting)
# --------------------------------------------------------------------------- #


def test_vol_ceiling_enforced_after_gate():
    """A book with a few very-high-vol names is pulled to/under the ceiling.

    Weight is concentrated in the hot names (the 0.90 gross means absolute book
    vol = 0.90 x proportional vol, so an equal-weight book would be too calm).
    """
    vols = np.array([0.70, 0.70, 0.70] + [0.12] * 9)
    cov = _cov(vols, corr=0.1)
    w = pd.Series([0.18, 0.18, 0.18] + [0.04] * 9, index=[f"T{i}" for i in range(12)])
    assert abs(w.sum() - 0.90) < 1e-12
    sectors = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    currencies = ["EUR"] * 12

    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=sectors,
        currencies=currencies,
        vol_ceiling=0.18,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
    )
    assert diag["vol_before"] > 0.18
    assert portfolio_vol(gated.to_numpy(), cov) <= 0.18 + _TOL
    assert diag["vol_after"] <= 0.18 + _TOL
    assert "tail_deweight" in diag["levers_fired"]
    # gated book keeps the same index/order.
    assert list(gated.index) == list(w.index)


# --------------------------------------------------------------------------- #
# (b) caps-to-convergence — no cap left breached
# --------------------------------------------------------------------------- #


def test_caps_to_convergence_all_within_tol():
    """A book breaching the USD-bloc + sector caps ends fully cap-compliant.

    15 names (11 USD + 4 EUR); TECH holds 5 names. At equal weight the USD bloc
    (0.73) and the TECH sector (0.33) both breach; the caps are feasible (enough
    names/sectors for redistribution) so convergence removes every breach and
    keeps deployment at 0.90.
    """
    usd = [f"U{i}" for i in range(11)]
    eur = [f"E{i}.DE" for i in range(4)]
    tickers = usd + eur
    currencies = ["USD"] * 11 + ["EUR"] * 4
    sectors = ["TECH"] * 5 + ["FIN"] * 3 + ["ENE"] * 3 + ["UTL"] * 2 + ["MAT"] * 2
    w = pd.Series(np.full(15, 0.90 / 15), index=tickers)
    cov = _cov(np.full(15, 0.15), corr=0.2)  # calm -> vol lever inert

    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=sectors,
        currencies=currencies,
        vol_ceiling=0.50,
        name_cap=0.15,
        sector_cap=0.25,
        usd_bloc_cap=0.60,
    )
    p = gated / gated.sum()
    assert p.max() <= 0.15 + _TOL
    assert max(_sector_props(sectors, p).values()) <= 0.25 + _TOL
    bloc = float(sum(wt for c, wt in zip(currencies, p, strict=True) if c in ("USD", "HKD")))
    assert bloc <= 0.60 + _TOL
    assert diag["caps_ok"] is True
    assert "caps" in diag["levers_fired"]
    assert diag["gross_cut"] is False  # calm book: no vol cut
    assert abs(gated.sum() - 0.90) < _TOL  # feasible caps -> deployment preserved


# --------------------------------------------------------------------------- #
# (c) lever order — de-weight before gross cut
# --------------------------------------------------------------------------- #


def test_lever1_deweight_alone_preserves_deployment():
    """Reducible by de-weighting -> gross_cut False, deployment (gross) unchanged."""
    vols = [0.70, 0.65, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    cov = _cov(vols, corr=0.2)
    w = pd.Series(
        [0.35, 0.30, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03], index=[f"T{i}" for i in range(8)]
    )
    assert abs(w.sum() - 0.90) < 1e-12

    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=["S"] * 8,
        currencies=["EUR"] * 8,
        vol_ceiling=0.18,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
    )
    assert diag["gross_cut"] is False
    assert "tail_deweight" in diag["levers_fired"]
    assert abs(gated.sum() - 0.90) < _TOL  # deployment preserved
    assert portfolio_vol(gated.to_numpy(), cov) <= 0.18 + _TOL


def test_lever2_gross_cut_when_deweight_exhausted():
    """All-high-vol, high-corr book can't be de-weighted below the ceiling ->
    gross cut fires, deployment drops, vol pinned to the ceiling."""
    cov = _cov(np.full(8, 0.40), corr=0.6)
    w = _series(np.full(8, 0.90 / 8), prefix="H")

    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=["S"] * 8,
        currencies=["EUR"] * 8,
        vol_ceiling=0.18,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
    )
    assert diag["gross_cut"] is True
    assert "gross_cut" in diag["levers_fired"]
    assert portfolio_vol(gated.to_numpy(), cov) <= 0.18 + _TOL
    assert gated.sum() < 0.90 - _TOL  # deployment reduced below target
    assert diag["gross_after"] < diag["gross_before"]


# --------------------------------------------------------------------------- #
# (d) already-compliant book -> gate is (near) identity
# --------------------------------------------------------------------------- #


def test_identity_when_within_all_limits():
    cov = _cov(np.full(10, 0.12), corr=0.1)  # calm -> vol < ceiling
    w = _series(np.full(10, 0.90 / 10))
    sectors = [f"S{i % 5}" for i in range(10)]  # 5 sectors x 2 -> 18% each
    currencies = ["EUR"] * 10

    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=sectors,
        currencies=currencies,
        vol_ceiling=0.18,
        name_cap=0.15,
        sector_cap=0.25,
        usd_bloc_cap=0.60,
    )
    np.testing.assert_allclose(gated.to_numpy(), w.to_numpy(), rtol=1e-9, atol=1e-12)
    assert diag["levers_fired"] == []
    assert diag["gross_cut"] is False
    assert diag["caps_ok"] is True
    assert abs(diag["gross_after"] - diag["gross_before"]) < 1e-12


# --------------------------------------------------------------------------- #
# (e) termination — loops respect max_iter (no hang) and lever 2 still binds
# --------------------------------------------------------------------------- #


def test_respects_max_iter_no_hang():
    vols = np.array([0.60, 0.55, 0.50, 0.45, 0.40] + [0.20] * 5)
    cov = _cov(vols, corr=0.3)
    w = _series(np.full(10, 0.90 / 10))

    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=["S"] * 10,
        currencies=["EUR"] * 10,
        vol_ceiling=0.15,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        max_iter=5,
    )
    # The tail-deweight loop never exceeds max_iter ...
    assert diag["iterations"] <= 5
    # ... and the lever-2 fallback still guarantees the ceiling.
    assert portfolio_vol(gated.to_numpy(), cov) <= 0.15 + _TOL


# --------------------------------------------------------------------------- #
# net-beta reporting + effective bets + degenerate input
# --------------------------------------------------------------------------- #


def test_net_beta_reported_and_flagged():
    cov = _cov(np.full(6, 0.12), corr=0.1)
    w = _series(np.full(6, 0.90 / 6))
    betas = [1.8] * 6  # high-beta book (net beta on proportions = 1.8 > band hi)
    gated, diag = apply_risk_gate(
        w,
        cov,
        sectors=["S"] * 6,
        currencies=["EUR"] * 6,
        betas=betas,
        vol_ceiling=1.0,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
    )
    assert diag["net_beta"] > 1.1
    assert diag["net_beta_out"] is True
    # NaN beta -> coerced to 1.0 (no NaN leaks into net beta).
    _, diag2 = apply_risk_gate(
        w,
        cov,
        sectors=["S"] * 6,
        currencies=["EUR"] * 6,
        betas=[1.0, np.nan, 1.0, 1.0, 1.0, 1.0],
        vol_ceiling=1.0,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
    )
    assert not np.isnan(diag2["net_beta"])


def test_effective_bets_on_gated_proportions():
    cov = _cov(np.full(10, 0.12), corr=0.1)
    w = _series(np.full(10, 0.90 / 10))
    _, diag = apply_risk_gate(
        w,
        cov,
        sectors=["S"] * 10,
        currencies=["EUR"] * 10,
        vol_ceiling=1.0,
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
    )
    # 10 equal names -> ~10 effective bets.
    assert abs(diag["effective_bets"] - 10.0) < 1e-6


def test_empty_book_is_safe():
    gated, diag = apply_risk_gate(
        pd.Series(dtype=float),
        np.zeros((0, 0)),
        sectors=[],
        currencies=[],
        vol_ceiling=0.18,
    )
    assert len(gated) == 0
    assert diag["levers_fired"] == []
    assert diag["gross_cut"] is False

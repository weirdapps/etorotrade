"""TDD tests for three cap-scaling vol modes in apply_risk_gate / build_overlay.

Coverage:
  cap_budget  — large-cap high-vol name kept; small-cap high-vol name trimmed.
  cap_exempt  — mega-cap kept more than small-cap; sigmoid direction verified.
  cap_ordered — smallest-cap name trimmed before largest at a hard vol ceiling.
  backward-compat — cap_mode=None / "uniform" reproduce the current gate exactly.

All tests use the single-factor beta covariance fallback (no price history),
so they are deterministic and network-free.

Design note on concentration caps: ``_gate()`` uses name_cap=0.99 / sector_cap=0.99 /
usd_bloc_cap=0.99 so that the mode-specific vol logic is what drives weight changes,
not the concentration caps step. The backward-compat tests call ``apply_risk_gate``
and ``build_overlay`` directly with default caps to exercise the real production defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_modules.riskfirst.construct import portfolio_vol
from trade_modules.v3.overlay import build_overlay
from trade_modules.v3.risk_gate import apply_risk_gate

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ALLCLOSE_TOL = {"rtol": 1e-6, "atol": 1e-9}


def _cov(vols, corr: float = 0.2) -> np.ndarray:
    """Annualised covariance from per-name vols and a common pairwise corr."""
    vols = np.asarray(vols, dtype=float)
    n = len(vols)
    C = np.full((n, n), float(corr))
    np.fill_diagonal(C, 1.0)
    return np.outer(vols, vols) * C


def _weights(vals: list[float], names: list[str] | None = None) -> pd.Series:
    names = names or [f"T{i}" for i in range(len(vals))]
    return pd.Series(vals, index=names, dtype=float)


def _caps(vals: list[float], names: list[str] | None = None) -> pd.Series:
    names = names or [f"T{i}" for i in range(len(vals))]
    return pd.Series(vals, index=names, dtype=float)


def _gate(
    weights: pd.Series,
    cov_mat: np.ndarray,
    caps: pd.Series | None = None,
    cap_mode: str | None = None,
    vol_ceiling: float = 0.25,
    managed_vol_ceiling: float = 0.18,
) -> tuple[pd.Series, dict]:
    """Thin wrapper around apply_risk_gate with non-binding concentration caps.

    Uses name_cap=1.0/sector_cap=1.0/usd_bloc_cap=1.0 so concentration caps do
    NOT pre-trim weights before the vol-mode logic runs (0.99 would spuriously
    bite a 100%-single-bloc test book, since bloc == gross there), letting each
    test exercise exactly the mode it claims to test.
    """
    n = len(weights)
    return apply_risk_gate(
        weights,
        cov_mat,
        sectors=["Tech"] * n,
        currencies=["USD"] * n,
        betas=np.ones(n),
        conviction=pd.Series(np.ones(n), index=weights.index),
        vol_ceiling=vol_ceiling,
        # Non-binding concentration caps so they don't mask vol-mode behaviour.
        name_cap=1.0,
        sector_cap=1.0,
        usd_bloc_cap=1.0,
        cap_mode=cap_mode,
        caps=caps,
        managed_vol_ceiling=managed_vol_ceiling,
    )


# ---------------------------------------------------------------------------
# cap_budget: large-cap high-vol KEPT, small-cap high-vol TRIMMED
# ---------------------------------------------------------------------------


def test_cap_budget_large_cap_kept_small_cap_trimmed():
    """RC-budget gate should allow the large-cap high-vol name to keep its weight
    while trimming the small-cap high-vol name whose RC share exceeds its allowance.
    """
    # 3 names: LARGE ($1T), SMALL ($1B), STABLE (low-vol, medium cap $100B).
    # LARGE and SMALL have identical high vol; STABLE is calm.
    names = ["LARGE", "SMALL", "STABLE"]
    vols = [0.6, 0.6, 0.10]  # LARGE and SMALL equally hot; STABLE calm
    cov_mat = _cov(vols, corr=0.1)
    w = _weights([0.30, 0.30, 0.40], names)
    cap_vals = _caps([1e12, 1e9, 1e11], names)  # LARGE > STABLE > SMALL

    gated, diag = _gate(w, cov_mat, caps=cap_vals, cap_mode="cap_budget")

    # LARGE should be held more than SMALL (it has a larger log-cap allowance).
    assert gated["LARGE"] >= gated["SMALL"], (
        f"Expected LARGE ({gated['LARGE']:.4f}) >= SMALL ({gated['SMALL']:.4f})"
    )
    # SMALL must be strictly trimmed below its initial weight.
    assert gated["SMALL"] < float(w["SMALL"]) - 1e-4, (
        f"SMALL not trimmed: init={w['SMALL']:.4f} final={gated['SMALL']:.4f}"
    )
    # rc_budget lever must have fired.
    assert "rc_budget" in diag["levers_fired"]


def test_cap_budget_equal_caps_equal_rc_share_no_trim():
    """With identical caps and a clean equal-weight zero-corr low-vol book, no trimming.

    Equal caps → equal allowances; equal weights + zero corr → equal RC shares.
    No name has RC > allowance, so the gate exits immediately with no lever fired.
    """
    names = ["A", "B", "C"]
    vols = [0.10, 0.10, 0.10]
    cov_mat = _cov(vols, corr=0.0)
    w = _weights([0.30, 0.30, 0.30], names)
    cap_vals = _caps([1e11, 1e11, 1e11], names)

    gated, diag = _gate(w, cov_mat, caps=cap_vals, cap_mode="cap_budget")

    np.testing.assert_allclose(gated.values, w.values, **_ALLCLOSE_TOL)
    assert "rc_budget" not in diag["levers_fired"]


# ---------------------------------------------------------------------------
# cap_exempt: mega cap kept more than small cap
# ---------------------------------------------------------------------------


def test_cap_exempt_mega_cap_held_more_than_small():
    """Sigmoid of z-scored log(cap): mega-cap yields higher exempt_frac → held more.

    With MEGA=$2T and SMALL=$500M (log-diff ≈ 8.3), z-scores are ≈ ±1.
    sigmoid(+1) ≈ 0.73 for MEGA (mostly exempt sleeve, uncapped),
    sigmoid(-1) ≈ 0.27 for SMALL (mostly managed sleeve, vol-capped).
    After recombining and vol-managing the SMALL-dominant managed sleeve,
    gated[MEGA] > gated[SMALL].
    """
    names = ["MEGA", "SMALL"]
    vols = [0.30, 0.30]
    cov_mat = _cov(vols, corr=0.0)
    w = _weights([0.50, 0.50], names)
    # log-ratio ≈ ln(4000) ≈ 8.3 → z-scores ≈ ±1
    cap_vals = _caps([2e12, 5e8], names)

    gated, diag = _gate(w, cov_mat, caps=cap_vals, cap_mode="cap_exempt", managed_vol_ceiling=0.10)

    # MEGA should keep more of its weight than SMALL.
    assert gated["MEGA"] > gated["SMALL"], (
        f"Expected MEGA ({gated['MEGA']:.4f}) > SMALL ({gated['SMALL']:.4f})"
    )
    # Both names should still have positive weight (gate doesn't zero anything out).
    assert gated["MEGA"] > 0.05, f"MEGA almost eliminated: {gated['MEGA']:.4f}"
    assert "cap_exempt" in diag["levers_fired"]


def test_cap_exempt_sigmoid_direction():
    """Directly verify sigmoid is monotone in log(cap): larger cap → higher exempt_frac.

    With 4 names spanning [1e13, 1e12, 1e8, 1e7] the z-scores are ≈ ±1.18 and
    ±0.79, giving sigmoid fracs ≈ [0.76, 0.69, 0.31, 0.24]. The key property is
    strict monotone ordering and that the top two are above 0.5, bottom two below.
    """
    from trade_modules.v3.risk_gate import _cap_exempt_gate

    n = 4
    caps_arr = np.array([1e13, 1e12, 1e8, 1e7])
    w = np.full(n, 0.25)
    vols = np.array([0.15, 0.15, 0.40, 0.40])
    cov_mat = _cov(vols, corr=0.0)

    # Replicate the sigmoid as the gate computes it (all in-book at equal weight).
    log_caps = np.log(np.maximum(caps_arr, 1.0))
    std_lc = float(log_caps.std())
    z = (log_caps - float(log_caps.mean())) / std_lc
    exempt_frac = 1.0 / (1.0 + np.exp(-z))

    # Strict monotone ordering.
    assert exempt_frac[0] > exempt_frac[1] > exempt_frac[2] > exempt_frac[3], (
        f"Sigmoid not monotone: {exempt_frac}"
    )
    # Top two (mega) above 0.5; bottom two (micro) below 0.5.
    assert exempt_frac[0] > 0.50, f"Largest cap exempt_frac below 0.5: {exempt_frac[0]:.4f}"
    assert exempt_frac[1] > 0.50, f"2nd cap exempt_frac below 0.5: {exempt_frac[1]:.4f}"
    assert exempt_frac[2] < 0.50, f"3rd cap exempt_frac above 0.5: {exempt_frac[2]:.4f}"
    assert exempt_frac[3] < 0.50, f"Smallest cap exempt_frac above 0.5: {exempt_frac[3]:.4f}"

    # Gate runs without error and preserves non-negative weights.
    sec_arr = np.array(["T", "T", "T", "T"], dtype=object)
    is_bloc = np.ones(n, dtype=bool)
    gated, _ = _cap_exempt_gate(
        w,
        cov_mat,
        sec_arr,
        is_bloc,
        caps_arr=caps_arr,
        managed_vol_ceiling=0.10,
        name_thr=0.99,
        bloc_thr=0.99,
        sector_thr=0.99,
        max_iter=100,
    )
    assert gated.sum() > 0
    assert np.all(gated >= 0)


# ---------------------------------------------------------------------------
# cap_ordered: smallest cap trimmed before largest
# ---------------------------------------------------------------------------


def test_cap_ordered_smallest_cap_trimmed_first():
    """At a tight vol ceiling, cap_ordered must trim the smallest-cap name
    before touching the largest-cap name.
    """
    names = ["BIG", "SMALL", "STABLE"]
    # All three names same vol so RC is equal; ceiling forces trimming.
    vols = [0.40, 0.40, 0.40]
    cov_mat = _cov(vols, corr=0.5)
    w = _weights([0.30, 0.30, 0.30], names)
    cap_vals = _caps([1e12, 1e9, 5e10], names)  # BIG > STABLE > SMALL

    vol_before = portfolio_vol(w.values, cov_mat)
    assert vol_before > 0.20, "book must start above the ceiling for this test to be meaningful"

    gated, diag = _gate(w, cov_mat, caps=cap_vals, cap_mode="cap_ordered", vol_ceiling=0.20)

    # SMALL (smallest cap) should lose more weight than BIG (largest cap).
    assert gated["SMALL"] < gated["BIG"], (
        f"Expected SMALL ({gated['SMALL']:.4f}) < BIG ({gated['BIG']:.4f})"
    )
    # Vol must be at or below the ceiling.
    vol_after = portfolio_vol(gated.values, cov_mat)
    assert vol_after <= 0.20 + 1e-6, f"Vol ceiling breached: {vol_after:.4f}"
    # Some lever must have fired.
    assert len(diag["levers_fired"]) > 0


def test_cap_ordered_no_trim_when_below_ceiling():
    """When the book is already within the vol ceiling, cap_ordered is a no-op."""
    names = ["BIG", "SMALL"]
    vols = [0.05, 0.05]
    cov_mat = _cov(vols, corr=0.0)
    w = _weights([0.40, 0.40], names)
    cap_vals = _caps([1e12, 1e9], names)

    # vol = sqrt(0.4^2*0.0025 + 0.4^2*0.0025) = sqrt(0.0008) ≈ 0.028 << 0.25 ceiling
    gated, diag = _gate(w, cov_mat, caps=cap_vals, cap_mode="cap_ordered", vol_ceiling=0.25)

    np.testing.assert_allclose(gated.values, w.values, **_ALLCLOSE_TOL)


# ---------------------------------------------------------------------------
# Backward-compat: cap_mode=None / "uniform" reproduces the current gate
# ---------------------------------------------------------------------------


def test_backward_compat_none_matches_default():
    """cap_mode=None must reproduce the default gate exactly (allclose).

    Uses the REAL apply_risk_gate defaults (name_cap=0.08 etc.) — not the
    high-cap test helper — to verify production code path is unchanged.
    """
    names = ["A", "B", "C", "D"]
    vols = [0.50, 0.30, 0.20, 0.15]
    cov_mat = _cov(vols, corr=0.3)
    w = _weights([0.30, 0.30, 0.25, 0.15], names)
    cap_vals = _caps([1e12, 1e11, 1e10, 1e9], names)
    n = len(names)

    def _real_gate(cap_mode, caps):
        return apply_risk_gate(
            w,
            cov_mat,
            sectors=["Tech"] * n,
            currencies=["USD"] * n,
            betas=np.ones(n),
            conviction=pd.Series(np.ones(n), index=w.index),
            cap_mode=cap_mode,
            caps=caps,
        )

    gated_default, _ = _real_gate(None, None)
    gated_explicit, _ = _real_gate(None, cap_vals)
    gated_uniform, _ = _real_gate("uniform", cap_vals)

    np.testing.assert_allclose(gated_explicit.values, gated_default.values, **_ALLCLOSE_TOL)
    np.testing.assert_allclose(gated_uniform.values, gated_default.values, **_ALLCLOSE_TOL)


def test_backward_compat_build_overlay():
    """build_overlay with cap_mode=None must match the default (no cap_mode arg)."""
    tickers = [f"U{i:02d}" for i in range(10)]
    convs = np.linspace(2.0, -1.0, 10)
    scored = pd.DataFrame(
        {
            "conviction": convs,
            "eligible": [True] * 10,
            "sector": ["Tech"] * 5 + ["Health"] * 5,
            "price": 100.0,
            "beta": 1.0,
            "name": [f"Name {t}" for t in tickers],
            "cap": [1e12, 5e11, 1e11, 5e10, 1e10, 5e9, 1e9, 5e8, 1e8, 5e7],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    current_weights = pd.Series(dict.fromkeys(tickers[:5], 0.08) | dict.fromkeys(tickers[5:], 0.04))

    res_default = build_overlay(scored, current_weights, pd.DataFrame())
    res_none = build_overlay(scored, current_weights, pd.DataFrame(), cap_mode=None)

    np.testing.assert_allclose(
        res_none["weights"].reindex(res_default["weights"].index).fillna(0.0).values,
        res_default["weights"].values,
        **_ALLCLOSE_TOL,
    )


def test_unknown_cap_mode_raises():
    """An unrecognised cap_mode raises ValueError."""
    names = ["A", "B"]
    cov_mat = _cov([0.2, 0.2])
    w = _weights([0.5, 0.5], names)
    with pytest.raises(ValueError, match="cap_mode"):
        _gate(w, cov_mat, cap_mode="nonexistent_mode")

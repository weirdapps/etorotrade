"""Tests for riskfirst.construct — ERC weights, vol targeting, name caps.

ERC (Equal Risk Contribution): each asset contributes equal risk to the
portfolio. For UNCORRELATED assets that reduces to inverse-volatility weights.
"""

import numpy as np
import pytest

from trade_modules.riskfirst.construct import apply_name_cap, erc_weights, vol_target_scale


def _risk_contributions(w, cov):
    sigma_w = cov @ w
    return w * sigma_w


# ---- ERC ----


def test_erc_uncorrelated_is_inverse_vol():
    # vols 0.1, 0.2 uncorrelated -> w proportional to 1/sigma -> 2:1
    cov = np.diag([0.01, 0.04])
    w = erc_weights(cov)
    assert w == pytest.approx([2 / 3, 1 / 3], abs=1e-4)
    assert w.sum() == pytest.approx(1.0)


def test_erc_equal_vol_is_equal_weight():
    cov = np.diag([0.04, 0.04, 0.04])
    w = erc_weights(cov)
    assert w == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-4)


def test_erc_risk_contributions_are_equal_when_correlated():
    sig = np.array([0.10, 0.20, 0.30])
    rho = 0.3
    cov = np.outer(sig, sig) * rho
    np.fill_diagonal(cov, sig**2)
    w = erc_weights(cov)
    rc = _risk_contributions(w, cov)
    assert np.std(rc) / np.mean(rc) < 1e-3  # all risk contributions equal
    assert (w > 0).all()
    assert w.sum() == pytest.approx(1.0)


# ---- vol targeting ----


def test_vol_target_scales_down_when_too_hot():
    cov = np.array([[0.04]])  # vol 0.20
    w = vol_target_scale(np.array([1.0]), cov, target_vol=0.10)
    assert w == pytest.approx([0.5])  # half invested, half cash


def test_vol_target_capped_by_max_gross_no_leverage():
    cov = np.array([[0.0025]])  # vol 0.05, would want 4x to hit 0.20
    w = vol_target_scale(np.array([1.0]), cov, target_vol=0.20, max_gross=1.0)
    assert w == pytest.approx([1.0])  # cannot lever beyond gross cap


# ---- single-name cap ----


def test_apply_name_cap_redistributes_excess():
    w = apply_name_cap(np.array([0.5, 0.3, 0.2]), cap=0.4)
    assert w == pytest.approx([0.4, 0.36, 0.24])
    assert w.sum() == pytest.approx(1.0)


def test_apply_name_cap_noop_when_within_cap():
    w = apply_name_cap(np.array([0.4, 0.35, 0.25]), cap=0.5)
    assert w == pytest.approx([0.4, 0.35, 0.25])

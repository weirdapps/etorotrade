"""Tests for riskfirst.covariance — a transparent single-factor (beta) covariance.

Without per-name price history we approximate the covariance with a one-factor
(market) model: Cov = beta beta' * market_var + diag(idio_var). Transparent,
needs only betas + two assumptions; a full sample/Ledoit-Wolf cov is the upgrade.
"""

import numpy as np
import pytest

from trade_modules.riskfirst.covariance import single_factor_cov


def test_equal_beta_unit_case():
    cov = single_factor_cov(np.array([1.0, 1.0]), market_vol=0.2, idio_vol=0.2)
    assert cov == pytest.approx(np.array([[0.08, 0.04], [0.04, 0.08]]))


def test_heterogeneous_betas_scalar_idio():
    cov = single_factor_cov(np.array([0.5, 1.5]), market_vol=0.2, idio_vol=0.1)
    assert cov == pytest.approx(np.array([[0.02, 0.03], [0.03, 0.10]]))


def test_is_symmetric_with_positive_diagonal():
    cov = single_factor_cov(np.array([0.8, 1.2, 1.0]), market_vol=0.18, idio_vol=0.25)
    assert cov == pytest.approx(cov.T)
    assert (np.diag(cov) > 0).all()


def test_per_name_idio_vector():
    cov = single_factor_cov(np.array([1.0, 1.0]), market_vol=0.2, idio_vol=np.array([0.1, 0.3]))
    # diag = market_var + idio_var per name
    assert cov[0, 0] == pytest.approx(0.04 + 0.01)
    assert cov[1, 1] == pytest.approx(0.04 + 0.09)

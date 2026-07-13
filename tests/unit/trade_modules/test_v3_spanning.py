"""Tests for trade_modules/validation/spanning.py (Phase 2B).

TDD: written FIRST — should fail on import until spanning.py exists.

Covers:
  - spanning_test : time-series OLS alpha of a candidate vs a factor set
  - grs_test      : Gibbons-Ross-Shanken joint-alpha F-test

All assertions check real numeric behaviour, not tautologies.
"""

import numpy as np

from trade_modules.validation.spanning import grs_test, spanning_test

# ---------------------------------------------------------------------------
# spanning_test
# ---------------------------------------------------------------------------


class TestSpanningTest:
    def _factor(self, T: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).normal(0.0, 0.02, T)

    def test_keys_present(self):
        f = self._factor(200, 0)
        cand = 1.0 * f + np.random.default_rng(1).normal(0.0, 0.005, 200)
        res = spanning_test(cand, f)
        assert set(res.keys()) >= {
            "alpha",
            "alpha_tstat",
            "alpha_pvalue",
            "betas",
            "r2",
            "n_obs",
            "spanned",
        }

    def test_candidate_is_scaled_factor_is_spanned(self):
        rng = np.random.default_rng(10)
        T = 250
        f = rng.normal(0.0, 0.02, T)
        cand = 1.0 * f + rng.normal(0.0, 0.005, T)  # tracks factor, no drift
        res = spanning_test(cand, f)
        assert abs(res["alpha"]) < 0.002
        assert abs(res["alpha_tstat"]) < 2.0
        assert res["spanned"] is True
        assert res["alpha_pvalue"] > 0.05
        assert res["n_obs"] == T
        # beta on the single factor recovered near 1.0
        beta_val = list(res["betas"].values())[0]
        assert abs(beta_val - 1.0) < 0.15
        assert res["r2"] > 0.8

    def test_candidate_with_drift_is_not_spanned(self):
        rng = np.random.default_rng(11)
        T = 250
        f = rng.normal(0.0, 0.02, T)
        drift = 0.01
        cand = f + drift + rng.normal(0.0, 0.005, T)
        res = spanning_test(cand, f)
        assert res["alpha"] > 0.005
        assert abs(res["alpha_tstat"]) > 2.0
        assert res["spanned"] is False
        assert res["alpha_pvalue"] < 0.05

    def test_multifactor_2d_input(self):
        rng = np.random.default_rng(12)
        T = 300
        f1 = rng.normal(0.0, 0.02, T)
        f2 = rng.normal(0.0, 0.015, T)
        F = np.column_stack([f1, f2])
        cand = 0.5 * f1 + 0.3 * f2 + rng.normal(0.0, 0.004, T)
        res = spanning_test(cand, F)
        assert len(res["betas"]) == 2
        assert res["spanned"] is True
        assert res["r2"] > 0.5

    def test_dict_factor_input_named_betas(self):
        rng = np.random.default_rng(13)
        T = 300
        mkt = rng.normal(0.0, 0.02, T)
        smb = rng.normal(0.0, 0.015, T)
        cand = 0.8 * mkt - 0.2 * smb + rng.normal(0.0, 0.004, T)
        res = spanning_test(cand, {"mkt": mkt, "smb": smb})
        assert set(res["betas"].keys()) == {"mkt", "smb"}
        assert abs(res["betas"]["mkt"] - 0.8) < 0.15
        assert abs(res["betas"]["smb"] - (-0.2)) < 0.15

    def test_threshold_controls_spanned_flag(self):
        rng = np.random.default_rng(14)
        T = 250
        f = rng.normal(0.0, 0.02, T)
        cand = f + 0.01 + rng.normal(0.0, 0.005, T)
        strict = spanning_test(cand, f, alpha_threshold_t=2.0)
        loose = spanning_test(cand, f, alpha_threshold_t=1000.0)
        assert strict["spanned"] is False
        assert loose["spanned"] is True  # same alpha_tstat, looser threshold

    def test_nan_rows_dropped(self):
        rng = np.random.default_rng(15)
        T = 250
        f = rng.normal(0.0, 0.02, T)
        cand = 1.0 * f + rng.normal(0.0, 0.005, T)
        cand[5] = np.nan
        cand[10] = np.nan
        res = spanning_test(cand, f)
        assert res["n_obs"] == T - 2
        assert res["spanned"] is True

    def test_insufficient_obs_graceful(self):
        res = spanning_test(np.array([0.01, 0.02]), np.array([0.01, 0.02]))
        # dof <= 0 → cannot form t-stat; must not crash
        assert res["alpha_tstat"] is None
        assert res["spanned"] is None


# ---------------------------------------------------------------------------
# grs_test
# ---------------------------------------------------------------------------


class TestGRSTest:
    def test_keys_present(self):
        res = grs_test(
            alphas=[0.0, 0.0],
            resid_cov=np.eye(2) * 0.01,
            factor_means=[0.005],
            factor_cov=[[0.0004]],
            n_obs=250,
        )
        assert set(res.keys()) == {"f_stat", "p_value", "n_assets", "n_obs"}

    def test_zero_alphas_high_pvalue(self):
        res = grs_test(
            alphas=[1e-4, -1e-4, 5e-5],
            resid_cov=np.eye(3) * 0.01,
            factor_means=[0.005],
            factor_cov=[[0.0004]],
            n_obs=250,
        )
        assert res["f_stat"] >= 0.0
        assert res["p_value"] > 0.1  # fail to reject H0: all alphas = 0
        assert res["n_assets"] == 3
        assert res["n_obs"] == 250

    def test_large_alphas_low_pvalue(self):
        res = grs_test(
            alphas=[0.02, 0.03, 0.025],
            resid_cov=np.eye(3) * 0.01,
            factor_means=[0.005],
            factor_cov=[[0.0004]],
            n_obs=250,
        )
        assert res["f_stat"] > 5.0
        assert res["p_value"] < 0.05  # reject H0

    def test_monotonic_in_alpha_magnitude(self):
        small = grs_test([0.005, 0.005], np.eye(2) * 0.01, [0.005], [[0.0004]], 250)
        big = grs_test([0.05, 0.05], np.eye(2) * 0.01, [0.005], [[0.0004]], 250)
        assert big["f_stat"] > small["f_stat"]
        assert big["p_value"] < small["p_value"]

    def test_scalar_single_asset_single_factor(self):
        res = grs_test(
            alphas=[0.0],
            resid_cov=0.01,
            factor_means=0.005,
            factor_cov=0.0004,
            n_obs=200,
        )
        assert res["n_assets"] == 1
        assert res["p_value"] > 0.1

    def test_degenerate_dof_graceful(self):
        # T - N - K <= 0 → F undefined; must not crash
        res = grs_test(
            alphas=[0.01, 0.02, 0.03],
            resid_cov=np.eye(3) * 0.01,
            factor_means=[0.005],
            factor_cov=[[0.0004]],
            n_obs=3,
        )
        assert res["f_stat"] is None
        assert res["p_value"] is None

"""Tests for riskfirst.edgegate.pbo_cscv — Probability of Backtest Overfitting
via Combinatorially Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado,
Zhu 2015). The review flagged the ABSENCE of PBO; this closes that gap.

PBO ~= P(the config that looked best in-sample underperforms the median
out-of-sample). Near 0 => robust selection; near 0.5+ => selection is overfit.
"""

import numpy as np

from trade_modules.riskfirst.edgegate import pbo_cscv


def test_pbo_near_zero_for_dominant_config():
    # config 0 is best in EVERY period -> IS-best is always genuinely best OOS.
    T, N = 40, 6
    M = np.tile(np.linspace(0.0, -0.1, N), (T, 1))  # col 0 highest, decreasing
    res = pbo_cscv(M, n_splits=10)
    assert res["pbo"] < 0.05


def test_pbo_moderate_for_noise_averaged_over_seeds():
    # Single-seed CSCV on a small matrix is noisy; average over seeds. Pure noise
    # should land in a moderate band (no genuine persistence, no built-in reversal).
    pbos = [
        pbo_cscv(np.random.default_rng(s).standard_normal((120, 10)), n_splits=12)["pbo"]
        for s in range(8)
    ]
    assert 0.2 <= float(np.mean(pbos)) <= 0.6


def test_pbo_high_for_anti_persistent():
    # Configs that win in the first half lose in the second half (pure overfit).
    T, N = 20, 6
    top = np.tile(np.linspace(1.0, 0.0, N), (T // 2, 1))  # col 0 best
    bot = np.tile(np.linspace(0.0, 1.0, N), (T - T // 2, 1))  # col 0 worst
    M = np.vstack([top, bot])
    res = pbo_cscv(M, n_splits=8)
    assert res["pbo"] > 0.6


def test_pbo_reports_shape():
    res = pbo_cscv(np.random.default_rng(1).standard_normal((30, 5)), n_splits=6)
    assert 0.0 <= res["pbo"] <= 1.0
    assert res["n_configs"] == 5
    assert res["n_combinations"] > 0
    assert len(res["logits"]) == res["n_combinations"]

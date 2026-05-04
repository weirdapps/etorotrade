"""
N4: Research-rigor improvements to the M11 modifier calibrator.

Adds three best-practice statistical improvements:
1. Walk-forward validation: split observations into rolling train/test
   windows so we test on data the calibration didn't see.
2. Multiple-comparison correction (Bonferroni): with 63 modifiers
   tested, ~3 will look "p<0.05" by pure chance. Adjust threshold.
3. Bootstrap 95% CI on every Spearman ρ: only mark PREDICTIVE if the
   lower CI bound is above zero (not just the point estimate).
"""

import pytest


class TestBonferroniCorrection:
    def test_bonferroni_adjusted_threshold_default(self):
        from scripts.calibrate_modifiers_t30 import bonferroni_threshold

        # 63 modifiers tested at α=0.05 → adjusted = 0.05 / 63 ≈ 0.000794
        assert bonferroni_threshold(63, alpha=0.05) == pytest.approx(
            0.05 / 63,
            abs=1e-6,
        )

    def test_bonferroni_with_one_test(self):
        from scripts.calibrate_modifiers_t30 import bonferroni_threshold

        # Single test → no correction needed
        assert bonferroni_threshold(1, alpha=0.05) == pytest.approx(0.05)


class TestBootstrapCI:
    def test_perfect_correlation_yields_tight_ci_above_zero(self):
        from scripts.calibrate_modifiers_t30 import bootstrap_spearman_ci

        # n=100 perfectly rank-correlated values
        xs = list(range(100))
        ys = list(range(100))
        lo, hi = bootstrap_spearman_ci(xs, ys, n_resamples=200, seed=42)
        # Both bounds should be near +1
        assert lo > 0.9
        assert hi > 0.9
        assert lo > 0  # the key invariant

    def test_zero_correlation_ci_straddles_zero(self):
        # Random uncorrelated data → CI should include 0
        import random

        from scripts.calibrate_modifiers_t30 import bootstrap_spearman_ci

        rng = random.Random(42)
        xs = [rng.gauss(0, 1) for _ in range(100)]
        ys = [rng.gauss(0, 1) for _ in range(100)]
        lo, hi = bootstrap_spearman_ci(xs, ys, n_resamples=200, seed=42)
        assert lo < 0 < hi or abs(lo) < 0.2  # CI brackets 0 (or very near it)

    def test_insufficient_data_returns_none(self):
        from scripts.calibrate_modifiers_t30 import bootstrap_spearman_ci

        assert bootstrap_spearman_ci([1, 2], [1, 2]) == (None, None)
        assert bootstrap_spearman_ci([], []) == (None, None)


class TestWalkForwardSplit:
    def test_simple_split_yields_n_folds(self):
        from scripts.calibrate_modifiers_t30 import walk_forward_splits

        # 100 observations, 4 folds → train/test pairs
        obs = [{"value": i, "alpha": i * 0.1} for i in range(100)]
        splits = list(walk_forward_splits(obs, n_folds=4))
        # Walk-forward yields (n_folds - 1) splits typically
        assert len(splits) == 3
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            # Test indices should be AFTER train indices (no leakage)
            train_max_value = max(o["value"] for o in train)
            test_min_value = min(o["value"] for o in test)
            assert test_min_value >= train_max_value

    def test_too_few_observations_yields_empty(self):
        from scripts.calibrate_modifiers_t30 import walk_forward_splits

        # n_folds requires at least n_folds × 2 observations
        assert list(walk_forward_splits([{"value": 1}], n_folds=4)) == []


class TestVerdictWithRigor:
    """classify_verdict should be more conservative with rigor enabled."""

    def test_significant_p_but_fails_bonferroni_demoted_to_shadow(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict_rigorous

        # ρ above threshold + nominal p<0.05 but FAILS Bonferroni 0.000794
        verdict = classify_verdict_rigorous(
            rho=0.15,
            p_value=0.04,
            n=200,
            ci_lower=0.05,
            ci_upper=0.25,
            n_modifiers_tested=63,
        )
        # Bonferroni-corrected α=0.000794, p=0.04 fails → SHADOW
        assert verdict == "SHADOW"

    def test_passes_all_rigor_gates_marked_predictive(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict_rigorous

        # Strong rho, very low p, lower CI > 0
        verdict = classify_verdict_rigorous(
            rho=0.30,
            p_value=1e-8,
            n=300,
            ci_lower=0.20,
            ci_upper=0.40,
            n_modifiers_tested=63,
        )
        assert verdict == "PREDICTIVE"

    def test_lower_ci_below_zero_demoted_to_weak(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict_rigorous

        # Point estimate looks fine but bootstrap CI includes 0
        verdict = classify_verdict_rigorous(
            rho=0.12,
            p_value=0.0001,
            n=200,
            ci_lower=-0.02,
            ci_upper=0.26,
            n_modifiers_tested=63,
        )
        # CI straddles 0 → not enough confidence for PREDICTIVE
        assert verdict in ("WEAK", "SHADOW")
        assert verdict != "PREDICTIVE"

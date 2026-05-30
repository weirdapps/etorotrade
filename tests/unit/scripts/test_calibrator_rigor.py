"""
N4: Research-rigor improvements to the M11 modifier calibrator.

Adds three best-practice statistical improvements:
1. Walk-forward validation: split observations into rolling train/test
   windows so we test on data the calibration didn't see.
2. Multiple-comparison correction (Benjamini-Hochberg FDR): with ~80
   modifiers tested, several will look "p<0.05" by pure chance. BH
   controls the false-discovery rate across the whole set.
3. Bootstrap 95% CI on every Spearman ρ: only mark PREDICTIVE if the
   lower CI bound is above zero (not just the point estimate).
"""


class TestBenjaminiHochberg:
    def test_smallest_p_survives_largest_does_not(self):
        from scripts.calibrate_modifiers_t30 import benjamini_hochberg

        adj = benjamini_hochberg({"a": 0.001, "b": 0.20, "c": 0.04, "d": 0.9}, alpha=0.05)
        assert adj["a"] is True
        assert adj["d"] is False

    def test_all_null_none_survive(self):
        from scripts.calibrate_modifiers_t30 import benjamini_hochberg

        adj = benjamini_hochberg({"a": 0.6, "b": 0.7, "c": 0.9}, alpha=0.05)
        assert adj == {"a": False, "b": False, "c": False}

    def test_empty_returns_empty(self):
        from scripts.calibrate_modifiers_t30 import benjamini_hochberg

        assert benjamini_hochberg({}, alpha=0.05) == {}


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
    """classify_verdict_rigorous consumes a precomputed BH boolean + bootstrap CI."""

    def test_fails_bh_demoted_to_shadow(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict_rigorous

        verdict = classify_verdict_rigorous(
            rho=0.15, n=200, ci_lower=0.05, ci_upper=0.25, bh_significant=False
        )
        assert verdict == "SHADOW"

    def test_passes_all_gates_marked_predictive(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict_rigorous

        verdict = classify_verdict_rigorous(
            rho=0.30, n=300, ci_lower=0.20, ci_upper=0.40, bh_significant=True
        )
        assert verdict == "PREDICTIVE"

    def test_lower_ci_below_zero_not_predictive(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict_rigorous

        verdict = classify_verdict_rigorous(
            rho=0.12, n=200, ci_lower=-0.02, ci_upper=0.26, bh_significant=True
        )
        assert verdict != "PREDICTIVE"

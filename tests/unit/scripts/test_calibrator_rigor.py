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

    def test_step_up_rescues_intermediate_failure(self):
        from scripts.calibrate_modifiers_t30 import benjamini_hochberg

        # m=3, alpha=0.05. Thresholds: r1=0.0167, r2=0.0333, r3=0.05.
        # b=0.04 FAILS its own rank-2 threshold (0.0333), but c passes at
        # rank 3 (0.045 <= 0.05), so the step-up rejects ALL ranks <= 3.
        # A naive per-element implementation would wrongly return b=False.
        adj = benjamini_hochberg({"a": 0.001, "b": 0.04, "c": 0.045}, alpha=0.05)
        assert adj == {"a": True, "b": True, "c": True}


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

        obs = [
            {"date": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", "value": i, "alpha": i * 0.1}
            for i in range(100)
        ]
        splits = list(walk_forward_splits(obs, n_folds=4))
        assert len(splits) == 3
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            train_max_date = max(o["date"] for o in train)
            test_min_date = min(o["date"] for o in test)
            assert test_min_date >= train_max_date

    def test_temporal_ordering_prevents_leakage(self):
        from scripts.calibrate_modifiers_t30 import walk_forward_splits

        obs = [
            {"date": f"2026-01-{i + 1:02d}", "value": 100 - i, "alpha": i * 0.1} for i in range(20)
        ]
        splits = list(walk_forward_splits(obs, n_folds=4))
        for train, test in splits:
            train_dates = {o["date"] for o in train}
            test_dates = {o["date"] for o in test}
            assert not train_dates & test_dates
            assert max(train_dates) <= min(test_dates)

    def test_too_few_observations_yields_empty(self):
        from scripts.calibrate_modifiers_t30 import walk_forward_splits

        assert list(walk_forward_splits([{"date": "2026-01-01"}], n_folds=4)) == []

    def test_value_sort_still_works_when_explicit(self):
        from scripts.calibrate_modifiers_t30 import walk_forward_splits

        obs = [{"value": i, "alpha": i * 0.1} for i in range(100)]
        splits = list(walk_forward_splits(obs, n_folds=4, sort_key="value"))
        assert len(splits) == 3
        for train, test in splits:
            train_max_value = max(o["value"] for o in train)
            test_min_value = min(o["value"] for o in test)
            assert test_min_value >= train_max_value


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

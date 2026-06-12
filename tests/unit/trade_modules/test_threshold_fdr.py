"""
Tests for FDR correction on threshold optimizer suggestions.

Validates:
1. fdr_correction() from backtest_stats works correctly
2. ThresholdAnalyzer._find_optimal_threshold returns p_value field
3. ThresholdAnalyzer.suggest_thresholds tags suggestions with fdr_significant
"""

import numpy as np
import pandas as pd

from trade_modules.backtest_stats import fdr_correction

# ── fdr_correction unit tests ──────────────────────────────────────


def test_fdr_correction_exists():
    """fdr_correction should accept p-values and return significant set."""
    p_values = {"a": 0.001, "b": 0.04, "c": 0.80}
    significant = fdr_correction(p_values)
    assert "a" in significant  # Very low p should survive
    assert "c" not in significant  # High p should not survive


def test_fdr_with_many_marginal():
    """With a spread of marginal p-values, BH should accept the low ones.

    Construct 20 p-values spanning 0.001 to 0.04 so rank ordering
    is well-defined (no ties). The smallest p-values should survive.
    """
    p_values = {f"test_{i}": 0.001 + i * 0.002 for i in range(20)}
    significant = fdr_correction(p_values)
    # The smallest p-values (0.001, 0.003, ...) should survive.
    # At minimum test_0 (p=0.001) survives: 0.001 <= 1/20 * 0.05 = 0.0025.
    assert len(significant) >= 1
    assert "test_0" in significant  # p=0.001, threshold=0.0025


def test_fdr_all_nonsignificant():
    """All high p-values should yield empty significant set."""
    p_values = {f"test_{i}": 0.5 + i * 0.01 for i in range(10)}
    significant = fdr_correction(p_values)
    assert len(significant) == 0


def test_fdr_empty_input():
    """Empty input should return empty list."""
    assert fdr_correction({}) == []


def test_fdr_single_significant():
    """Single test with low p should survive."""
    significant = fdr_correction({"only": 0.01})
    assert significant == ["only"]


def test_fdr_single_nonsignificant():
    """Single test with high p should not survive."""
    significant = fdr_correction({"only": 0.50})
    assert significant == []


def test_fdr_mixed_separates_correctly():
    """FDR should clearly separate strong signals from noise."""
    p_values = {
        "strong1": 0.001,
        "strong2": 0.005,
        "marginal": 0.045,
        "noise1": 0.30,
        "noise2": 0.70,
    }
    significant = fdr_correction(p_values)
    assert "strong1" in significant
    assert "strong2" in significant
    assert "noise1" not in significant
    assert "noise2" not in significant


# ── _find_optimal_threshold p-value tests ──────────────────────────


class TestFindOptimalThresholdPValue:
    """Verify _find_optimal_threshold returns p_value and hit counts."""

    def _make_data(self, n=100, hit_rate=0.7, seed=42):
        """Build synthetic data where BUY signals have a known hit-rate."""
        rng = np.random.default_rng(seed)
        metric = rng.normal(10, 3, size=n)
        # alpha > 0 = hit for BUY signal
        hits = rng.random(n) < hit_rate
        alpha = np.where(hits, rng.uniform(0.5, 5.0, n), rng.uniform(-5.0, -0.5, n))
        return pd.DataFrame({"metric": metric, "alpha": alpha})

    def test_returns_p_value_field(self):
        """Suggestion dict must include p_value, n_hits, n_obs."""
        from trade_modules.backtest_engine import ThresholdAnalyzer

        data = self._make_data(n=200, hit_rate=0.75)
        # Use a very restrictive current_val so the "optimal" is a big improvement
        result = ThresholdAnalyzer._find_optimal_threshold(
            data, "metric", current_val=20.0, direction="min", signal="B"
        )
        if result is not None:
            assert "p_value" in result
            assert "n_hits" in result
            assert "n_obs" in result
            assert isinstance(result["p_value"], float)
            assert 0.0 <= result["p_value"] <= 1.0

    def test_high_hit_rate_low_p_value(self):
        """A genuine 75% hit-rate with n=200 should yield a very low p-value."""
        from trade_modules.backtest_engine import ThresholdAnalyzer

        data = self._make_data(n=200, hit_rate=0.75)
        result = ThresholdAnalyzer._find_optimal_threshold(
            data, "metric", current_val=20.0, direction="min", signal="B"
        )
        if result is not None:
            assert result["p_value"] < 0.01

    def test_chance_hit_rate_high_p_value(self):
        """A ~50% hit-rate should NOT produce a low p-value."""
        from trade_modules.backtest_engine import ThresholdAnalyzer

        data = self._make_data(n=200, hit_rate=0.50)
        result = ThresholdAnalyzer._find_optimal_threshold(
            data, "metric", current_val=20.0, direction="min", signal="B"
        )
        # With 50% hit rate the improvement threshold (>5%) may not be met,
        # so result could be None. That's fine — it means no suggestion was made.
        if result is not None:
            assert result["p_value"] > 0.01

    def test_none_when_no_improvement(self):
        """No suggestion when improvement is <= 5%."""
        from trade_modules.backtest_engine import ThresholdAnalyzer

        # Current value is already optimal (at median) — little room for improvement
        data = self._make_data(n=200, hit_rate=0.52)
        result = ThresholdAnalyzer._find_optimal_threshold(
            data, "metric", current_val=10.0, direction="min", signal="B"
        )
        # With only 52% hit rate, hard to get 5% improvement
        assert result is None


# ── suggest_thresholds FDR tagging tests ───────────────────────────


class TestSuggestThresholdsFDR:
    """Verify suggest_thresholds tags all suggestions with fdr_significant."""

    def test_fdr_significant_field_present(self, tmp_path):
        """Every suggestion in the output must have fdr_significant: bool."""
        from trade_modules.backtest_engine import ThresholdAnalyzer

        # Create a minimal config.yaml
        config_content = """
us_mega:
  buy:
    min_upside: 20.0
    min_buy_percentage: 80.0
    min_exret: 15.0
  sell:
    max_exret: -5.0
    max_upside: -10.0
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        analyzer = ThresholdAnalyzer(config_path=config_path)

        # Build synthetic merged data for us_mega BUY signals
        rng = np.random.default_rng(99)
        n = 200
        merged = pd.DataFrame(
            {
                "tier": ["mega"] * n,
                "region": ["us"] * n,
                "signal": ["B"] * n,
                "upside": rng.normal(15, 5, n),
                "buy_percentage": rng.normal(60, 15, n),
                "exret": rng.normal(8, 4, n),
                "alpha": rng.normal(2, 3, n),  # mildly positive alpha
            }
        )

        suggestions = analyzer.suggest_thresholds(merged)

        for s in suggestions:
            assert "fdr_significant" in s, (
                f"Missing fdr_significant in suggestion: {s.get('config_key')}"
            )
            assert isinstance(s["fdr_significant"], bool)

    def test_all_nonsig_when_noise(self, tmp_path):
        """With pure noise data, no suggestion should be FDR-significant."""
        from trade_modules.backtest_engine import ThresholdAnalyzer

        config_content = """
us_mega:
  buy:
    min_upside: 20.0
  sell:
    max_exret: -5.0
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        analyzer = ThresholdAnalyzer(config_path=config_path)

        # Pure coin-flip alpha — no real signal
        rng = np.random.default_rng(42)
        n = 200
        merged = pd.DataFrame(
            {
                "tier": ["mega"] * n,
                "region": ["us"] * n,
                "signal": ["B"] * n,
                "upside": rng.normal(10, 5, n),
                "alpha": rng.choice([-1, 1], size=n),  # exactly 50/50
            }
        )

        suggestions = analyzer.suggest_thresholds(merged)
        # With coin-flip alpha, unlikely to get >5% improvement,
        # but if any slip through they should NOT be FDR-significant.
        for s in suggestions:
            assert not s["fdr_significant"], (
                f"Noise-only suggestion should not be FDR-significant: {s}"
            )

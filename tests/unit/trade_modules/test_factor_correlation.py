"""
N5: Factor correlation diagnostic for M4 EmpiricalFactorScore.

Multi-factor models are at risk of multicollinearity — if two "different"
factors actually measure the same underlying signal, weighted-summing
them double-counts and inflates conviction without adding independent
information.

This module:
1. Computes pairwise correlation matrix among factor inputs in a universe
2. Flags any pair with |ρ| > threshold (default 0.7) as redundant
3. Provides an audit-only diagnostic — does NOT change scoring behavior

Used as a pre-flight check before trusting factor scores.
"""

import pytest


class TestFactorCorrelationMatrix:
    def test_uncorrelated_factors_yield_low_off_diagonal(self):
        # Random universe — factors should be ~uncorrelated
        import random

        from trade_modules.empirical_factor import factor_correlation_matrix

        rng = random.Random(42)
        universe = [
            {
                "ticker": f"T{i}",
                "pct_52w_high": rng.uniform(40, 100),
                "short_interest": rng.uniform(0, 15),
                "roe": rng.uniform(-10, 50),
                "upside": rng.uniform(-10, 60),
            }
            for i in range(60)
        ]
        matrix = factor_correlation_matrix(universe)
        # Diagonals are 1.0
        for f in ("momentum", "short_interest", "roe", "upside"):
            assert matrix[f][f] == pytest.approx(1.0, abs=0.001)
        # Off-diagonal max abs should be modest for random data
        max_offdiag = max(abs(matrix[a][b]) for a in matrix for b in matrix if a != b)
        assert max_offdiag < 0.6  # generous bound — random data, n=60

    def test_perfectly_correlated_factors_yield_high_off_diagonal(self):
        """If two factors are identical, correlation should be 1.0."""
        from trade_modules.empirical_factor import factor_correlation_matrix

        # momentum = upside (made identical) → correlation should be 1
        universe = [
            {
                "ticker": f"T{i}",
                "pct_52w_high": float(i),
                "short_interest": float(i % 5),
                "roe": float(i * 2),
                "upside": float(i),  # same as pct_52w_high
            }
            for i in range(50)
        ]
        matrix = factor_correlation_matrix(universe)
        # momentum vs upside should be ~+1.0
        assert matrix["momentum"]["upside"] == pytest.approx(1.0, abs=0.01)


class TestFlagRedundantFactors:
    def test_no_redundancy_returns_empty_list(self):
        import random

        from trade_modules.empirical_factor import flag_redundant_factor_pairs

        rng = random.Random(42)
        universe = [
            {
                "ticker": f"T{i}",
                "pct_52w_high": rng.uniform(40, 100),
                "short_interest": rng.uniform(0, 15),
                "roe": rng.uniform(-10, 50),
                "upside": rng.uniform(-10, 60),
            }
            for i in range(60)
        ]
        flagged = flag_redundant_factor_pairs(universe, threshold=0.7)
        assert flagged == []

    def test_high_correlation_flagged(self):
        from trade_modules.empirical_factor import flag_redundant_factor_pairs

        universe = [
            {
                "ticker": f"T{i}",
                "pct_52w_high": float(i),
                "short_interest": float(i % 5),
                "roe": float(i * 2),
                "upside": float(i),  # same as pct_52w_high
            }
            for i in range(50)
        ]
        flagged = flag_redundant_factor_pairs(universe, threshold=0.7)
        # momentum and upside are identical → should be flagged
        pairs = [(p["factor_a"], p["factor_b"]) for p in flagged]
        assert ("momentum", "upside") in pairs or ("upside", "momentum") in pairs

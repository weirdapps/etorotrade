"""
M4: EmpiricalFactorScore — CIO v36 Empirical Refoundation.

The parameter study (signal_log.jsonl, n=32,589, T+30) proved the system's
primary BUY criteria are INVERTED for the user's 30-day horizon:
- upside ρ=−0.087 (high upside → LOWER returns)
- exret ρ=−0.079 (high exret → LOWER returns)
- buy_percentage ρ=−0.015 (high consensus → LOWER returns)

The strongest empirical predictors are CURRENTLY UNUSED:
- pct_52w_high ρ=+0.103 (momentum factor)
- short_interest ρ=−0.111 (short-side smart-money signal)

M4 ships a multi-factor score that uses the empirically-validated signals
with the correct signs. The score is a continuous z-weighted combination,
suitable as a BUY-gate or as a tiebreaker.
"""

import math


class TestComputeFactorScore:
    def test_high_momentum_low_si_high_roe_yields_high_score(self):
        from trade_modules.empirical_factor import compute_factor_score

        # Universe with one star + 9 average stocks
        universe = []
        for i in range(9):
            universe.append(
                {
                    "ticker": f"AVG{i}",
                    "pct_52w_high": 70,
                    "short_interest": 5,
                    "roe": 12,
                    "upside": 10,
                }
            )
        universe.append(
            {
                "ticker": "STAR",
                "pct_52w_high": 95,  # near 52w high — strong momentum
                "short_interest": 1,  # very low — bulls leaning in
                "roe": 30,  # high quality
                "upside": 6,  # moderate, not crowded
            }
        )

        scores = {s["ticker"]: compute_factor_score(s, universe) for s in universe}
        # STAR should score above the average
        avg_score = sum(v["score"] for k, v in scores.items() if k != "STAR") / 9
        assert scores["STAR"]["score"] > avg_score
        assert scores["STAR"]["score"] > 0

    def test_high_upside_high_exret_yields_low_score(self):
        """Crowded analyst-darling profile gets DEMOTED — empirical inversion."""
        from trade_modules.empirical_factor import compute_factor_score

        universe = []
        for i in range(9):
            universe.append(
                {
                    "ticker": f"AVG{i}",
                    "pct_52w_high": 70,
                    "short_interest": 5,
                    "roe": 12,
                    "upside": 10,
                }
            )
        universe.append(
            {
                "ticker": "CROWDED",
                "pct_52w_high": 50,  # mid-pack momentum
                "short_interest": 12,  # high SI — shorts disagree
                "roe": 8,  # lower quality
                "upside": 35,  # massive upside — typical analyst-darling
            }
        )

        scores = {s["ticker"]: compute_factor_score(s, universe) for s in universe}
        # CROWDED should score below average (empirical: this profile underperforms)
        avg_score = sum(v["score"] for k, v in scores.items() if k != "CROWDED") / 9
        assert scores["CROWDED"]["score"] < avg_score

    def test_score_includes_component_breakdown(self):
        from trade_modules.empirical_factor import compute_factor_score

        universe = [
            {"ticker": "X", "pct_52w_high": 90, "short_interest": 2, "roe": 25, "upside": 8},
            {"ticker": "Y", "pct_52w_high": 50, "short_interest": 10, "roe": 5, "upside": 30},
        ]
        result = compute_factor_score(universe[0], universe)
        assert "components" in result
        assert "momentum" in result["components"]
        assert "short_interest" in result["components"]
        assert "roe" in result["components"]
        assert "upside" in result["components"]

    def test_missing_data_handled_gracefully(self):
        """Missing fields → score should not raise, components default to 0."""
        from trade_modules.empirical_factor import compute_factor_score

        universe = [
            {"ticker": "X"},  # all fields missing
            {"ticker": "Y", "pct_52w_high": 80, "short_interest": 3, "roe": 20, "upside": 5},
        ]
        result = compute_factor_score(universe[0], universe)
        assert isinstance(result["score"], (int, float))
        assert not math.isnan(result["score"])


class TestFactorGate:
    def test_score_above_threshold_passes_gate(self):
        from trade_modules.empirical_factor import passes_buy_gate

        assert passes_buy_gate({"score": 0.5}, threshold=0.0) is True
        assert passes_buy_gate({"score": 0.01}, threshold=0.0) is True

    def test_score_below_threshold_fails_gate(self):
        from trade_modules.empirical_factor import passes_buy_gate

        assert passes_buy_gate({"score": -0.5}, threshold=0.0) is False
        assert passes_buy_gate({"score": 0.0}, threshold=0.0) is False  # strict >


class TestApplyEmpiricalGate:
    def test_demote_buy_below_gate(self):
        from trade_modules.empirical_factor import (
            apply_empirical_gate,
        )

        # Two BUY-signal stocks: one passes, one demoted
        concordance = [
            {
                "ticker": "GOOD",
                "signal": "B",
                "action": "BUY",
                "pct_52w_high": 95,
                "short_interest": 1,
                "roe": 30,
                "upside": 6,
            },
            {
                "ticker": "CROWDED",
                "signal": "B",
                "action": "BUY",
                "pct_52w_high": 50,
                "short_interest": 12,
                "roe": 8,
                "upside": 35,
            },
            # Average pad to give cross-sectional context
            {
                "ticker": "PAD1",
                "signal": "H",
                "action": "HOLD",
                "pct_52w_high": 70,
                "short_interest": 5,
                "roe": 12,
                "upside": 10,
            },
            {
                "ticker": "PAD2",
                "signal": "H",
                "action": "HOLD",
                "pct_52w_high": 70,
                "short_interest": 5,
                "roe": 12,
                "upside": 10,
            },
            {
                "ticker": "PAD3",
                "signal": "H",
                "action": "HOLD",
                "pct_52w_high": 70,
                "short_interest": 5,
                "roe": 12,
                "upside": 10,
            },
        ]
        apply_empirical_gate(concordance)

        # Both got an empirical score
        assert "empirical_score" in concordance[0]
        assert "empirical_score" in concordance[1]
        # GOOD passes gate (score > 0); CROWDED demoted to HOLD
        assert concordance[0]["action"] == "BUY"
        assert concordance[1]["action"] == "HOLD"
        assert concordance[1].get("empirical_gate_demoted") is True

    def test_non_buy_actions_untouched(self):
        from trade_modules.empirical_factor import apply_empirical_gate

        # SELL/HOLD/TRIM should not be affected by the BUY gate
        concordance = [
            {
                "ticker": "S",
                "signal": "S",
                "action": "SELL",
                "pct_52w_high": 50,
                "short_interest": 12,
                "roe": 8,
                "upside": 35,
            },
            {
                "ticker": "H",
                "signal": "H",
                "action": "HOLD",
                "pct_52w_high": 50,
                "short_interest": 12,
                "roe": 8,
                "upside": 35,
            },
        ]
        apply_empirical_gate(concordance)
        assert concordance[0]["action"] == "SELL"
        assert concordance[1]["action"] == "HOLD"


class TestBuildConcordanceEnvFlag:
    """build_concordance must apply the gate iff CIO_V36_ENABLE_EMPIRICAL_GATE=1."""

    def _minimal_signals(self):
        return {
            "BAD": {
                "signal": "B",
                "exret": 35,
                "buy_pct": 80,
                "beta": 1.0,
                "pet": 30,
                "pef": 28,
                "pp": -5,
                "52w": 50,
                "short_interest": 12,
                "roe": 8,
                "upside": 35,
                "pct_52w_high": 50,
            },
            "PAD1": {
                "signal": "H",
                "exret": 5,
                "buy_pct": 50,
                "beta": 1.0,
                "pet": 18,
                "pef": 17,
                "pp": 5,
                "52w": 70,
                "short_interest": 5,
                "roe": 12,
                "upside": 10,
                "pct_52w_high": 70,
            },
            "PAD2": {
                "signal": "H",
                "exret": 5,
                "buy_pct": 50,
                "beta": 1.0,
                "pet": 18,
                "pef": 17,
                "pp": 5,
                "52w": 70,
                "short_interest": 5,
                "roe": 12,
                "upside": 10,
                "pct_52w_high": 70,
            },
        }

    def test_gate_disabled_by_default(self, monkeypatch):
        from trade_modules.committee_synthesis import build_concordance

        monkeypatch.delenv("CIO_V36_ENABLE_EMPIRICAL_GATE", raising=False)
        signals = self._minimal_signals()
        concordance = build_concordance(
            signals,
            {"stocks": {}},
            {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {
                "divergences": {
                    "consensus_aligned": [],
                    "signal_divergences": [],
                    "census_divergences": [],
                }
            },
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map=dict.fromkeys(signals, "Technology"),
        )
        for entry in concordance:
            # No empirical_score field → gate not applied
            assert "empirical_score" not in entry

    def test_gate_enabled_via_env(self, monkeypatch):
        from trade_modules.committee_synthesis import build_concordance

        monkeypatch.setenv("CIO_V36_ENABLE_EMPIRICAL_GATE", "1")
        signals = self._minimal_signals()
        concordance = build_concordance(
            signals,
            {"stocks": {}},
            {"stocks": {}},
            {"portfolio_implications": {}, "sector_rankings": {}},
            {
                "divergences": {
                    "consensus_aligned": [],
                    "signal_divergences": [],
                    "census_divergences": [],
                }
            },
            {"portfolio_news": {}},
            {"consensus_warnings": [], "position_limits": {}},
            sector_map=dict.fromkeys(signals, "Technology"),
        )
        # Gate ran → all entries have empirical_score
        for entry in concordance:
            assert "empirical_score" in entry

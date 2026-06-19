"""Tests for risk_metrics.py — canonical risk score and risk level."""

from trade_modules.risk_metrics import canonical_risk_score, risk_level


class TestCanonicalRiskScore:
    """Tests for canonical_risk_score function."""

    def test_low_risk_portfolio(self):
        """Low VaR, low drawdown, beta < 1, high Sortino => low score."""
        score = canonical_risk_score(
            var_95_pct=1.0,
            max_drawdown_pct=5.0,
            portfolio_beta=0.8,
            sortino=3.0,
        )
        # 1.0*10 + 5.0*2 + 0*20 + 0*10 = 20
        assert score == 20

    def test_high_risk_portfolio(self):
        """High VaR, deep drawdown, beta > 1, low Sortino => high score."""
        score = canonical_risk_score(
            var_95_pct=3.0,
            max_drawdown_pct=20.0,
            portfolio_beta=1.5,
            sortino=0.5,
        )
        # 3.0*10 + 20.0*2 + 0.5*20 + 1.5*10 = 30 + 40 + 10 + 15 = 95
        assert score == 95

    def test_capped_at_100(self):
        """Extreme inputs should cap at 100."""
        score = canonical_risk_score(
            var_95_pct=10.0,
            max_drawdown_pct=50.0,
            portfolio_beta=3.0,
            sortino=0.0,
        )
        assert score == 100

    def test_floored_at_zero(self):
        """All-zero inputs should return 0."""
        score = canonical_risk_score(
            var_95_pct=0.0,
            max_drawdown_pct=0.0,
            portfolio_beta=0.0,
            sortino=5.0,
        )
        # 0 + 0 + 0 + 0 = 0
        assert score == 0

    def test_negative_var_uses_abs(self):
        """Negative VaR pct should be treated as absolute."""
        score_pos = canonical_risk_score(var_95_pct=2.0, max_drawdown_pct=10.0, portfolio_beta=1.0)
        score_neg = canonical_risk_score(
            var_95_pct=-2.0, max_drawdown_pct=-10.0, portfolio_beta=1.0
        )
        assert score_pos == score_neg

    def test_default_sortino_neutral(self):
        """Default sortino=2.0 contributes zero to score."""
        score_with = canonical_risk_score(
            var_95_pct=2.0, max_drawdown_pct=10.0, portfolio_beta=1.0, sortino=2.0
        )
        score_without = canonical_risk_score(
            var_95_pct=2.0, max_drawdown_pct=10.0, portfolio_beta=1.0
        )
        assert score_with == score_without

    def test_beta_below_one_no_contribution(self):
        """Beta < 1 should contribute 0 (defensive portfolio)."""
        score_low_beta = canonical_risk_score(
            var_95_pct=2.0, max_drawdown_pct=10.0, portfolio_beta=0.5
        )
        score_one_beta = canonical_risk_score(
            var_95_pct=2.0, max_drawdown_pct=10.0, portfolio_beta=1.0
        )
        assert score_low_beta == score_one_beta

    def test_returns_int(self):
        """Score should always be an integer."""
        score = canonical_risk_score(
            var_95_pct=1.5,
            max_drawdown_pct=7.3,
            portfolio_beta=1.1,
            sortino=1.8,
        )
        assert isinstance(score, int)


class TestRiskLevel:
    """Tests for risk_level function."""

    def test_low_risk(self):
        assert risk_level(0) == "LOW"
        assert risk_level(39) == "LOW"

    def test_moderate_risk(self):
        assert risk_level(40) == "MODERATE"
        assert risk_level(69) == "MODERATE"

    def test_high_risk(self):
        assert risk_level(70) == "HIGH"
        assert risk_level(100) == "HIGH"

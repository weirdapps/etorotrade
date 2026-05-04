"""
N9 / M9: Portfolio-level vol-targeting.

VaR-based scaling (existing) catches sudden moves but misses slow grinding
losses. A portfolio that loses 1% per day for a month has stable daily VaR
but realized 60d vol is way over target.

M9 adds 60d realized portfolio vol calculation + scaling factor:
- target = 12% annualized (institutional standard for retail equity)
- if realized_vol > target, scale all sizing × target/realized
"""

import pytest


class TestRealizedPortfolioVol:
    def test_constant_returns_yield_zero_vol(self):
        from trade_modules.vol_targeting import realized_portfolio_vol

        # 60 days of identical returns → vol = 0
        daily_returns = [0.001] * 60
        vol = realized_portfolio_vol(daily_returns)
        assert vol == pytest.approx(0.0, abs=0.0001)

    def test_typical_equity_returns_yield_realistic_vol(self):
        # Mix of small ups and downs ≈ 1% daily std → ~16% annualized
        import random

        from trade_modules.vol_targeting import realized_portfolio_vol

        rng = random.Random(42)
        daily_returns = [rng.gauss(0, 0.01) for _ in range(60)]
        vol = realized_portfolio_vol(daily_returns)
        # Annualized vol = daily_std × sqrt(252) ≈ 0.01 × 15.87 ≈ 0.16
        assert 0.10 < vol < 0.25

    def test_insufficient_data_returns_none(self):
        from trade_modules.vol_targeting import realized_portfolio_vol

        assert realized_portfolio_vol([]) is None
        assert realized_portfolio_vol([0.01, 0.02]) is None  # < 5 obs


class TestVolScaleFactor:
    def test_below_target_returns_unity(self):
        from trade_modules.vol_targeting import vol_scale_factor

        # Realized vol 8%, target 12% → no scaling (we have headroom)
        assert vol_scale_factor(realized=0.08, target=0.12) == pytest.approx(1.0)

    def test_above_target_scales_down(self):
        from trade_modules.vol_targeting import vol_scale_factor

        # Realized 18%, target 12% → factor = 12/18 = 0.667
        assert vol_scale_factor(realized=0.18, target=0.12) == pytest.approx(
            12 / 18,
            abs=0.001,
        )

    def test_zero_realized_returns_unity(self):
        from trade_modules.vol_targeting import vol_scale_factor

        # No realized vol (pre-launch portfolio) → no scaling
        assert vol_scale_factor(realized=0, target=0.12) == pytest.approx(1.0)

    def test_negative_realized_returns_unity(self):
        from trade_modules.vol_targeting import vol_scale_factor

        # Defensive: negative input should not break the math
        assert vol_scale_factor(realized=-0.05, target=0.12) == pytest.approx(1.0)


class TestEnrichWithVolScaling:
    """enrich_with_position_sizes should accept a vol_scale parameter."""

    def test_vol_scale_reduces_all_buy_sizes(self):
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        conc = [
            {"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"},
            {"ticker": "MSFT", "action": "ADD", "conviction": 70, "market_cap": "MEGA"},
        ]
        # Same call without vol scaling
        baseline = [dict(c) for c in conc]
        enrich_with_position_sizes(baseline, portfolio_value=400_000)
        # With vol_scale=0.5 → all sizes halved
        scaled = [dict(c) for c in conc]
        enrich_with_position_sizes(scaled, portfolio_value=400_000, vol_scale=0.5)

        for b, s in zip(baseline, scaled):
            assert s["suggested_size_usd"] == pytest.approx(
                b["suggested_size_usd"] * 0.5,
                abs=1,
            )

    def test_vol_scale_default_is_unity(self):
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        enrich_with_position_sizes(conc, portfolio_value=400_000)
        # No vol_scale arg → no change vs explicit vol_scale=1.0
        size_default = conc[0]["suggested_size_usd"]

        conc2 = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        enrich_with_position_sizes(conc2, portfolio_value=400_000, vol_scale=1.0)
        assert conc2[0]["suggested_size_usd"] == size_default

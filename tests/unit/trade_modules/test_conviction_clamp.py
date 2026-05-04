"""
M2: Conviction-multiplier clamp tests — CIO v36 Empirical Refoundation.

The v17 review documented Spearman ρ(conviction, α30) ≈ −0.002 — the
conviction score has zero rank-order predictive power. Yet the sizer
applies a conviction-based multiplier ranging from 0.35 (low conviction)
to 1.0 (high conviction). With ρ≈0, that multiplier is sizing positions
based on noise, concentrating bets on stocks with no demonstrated edge.

M2: Clamp the conviction multiplier to 1.0 (= equal-weight at tier
baseline) until per-cell ρ proves > 0.05 out-of-sample. This removes the
largest avoidable source of risk-on-noise concentration.

The clamp is controllable via `CONVICTION_CLAMP_TO_UNITY` so the behavior
can be re-enabled when evidence accumulates (handled by M11's calibrator).
"""


class TestConvictionMultiplierClamp:
    def test_clamped_returns_unity_for_low_conviction(self, monkeypatch):
        from trade_modules import conviction_sizer

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", True)
        # Pre-clamp this would be 0.35 (low conviction)
        assert conviction_sizer.get_conviction_multiplier(0) == 1.0

    def test_clamped_returns_unity_for_high_conviction(self, monkeypatch):
        from trade_modules import conviction_sizer

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", True)
        assert conviction_sizer.get_conviction_multiplier(95) == 1.0

    def test_clamped_returns_unity_for_mid_conviction(self, monkeypatch):
        from trade_modules import conviction_sizer

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", True)
        assert conviction_sizer.get_conviction_multiplier(60) == 1.0

    def test_unclamped_preserves_legacy_continuous_function(self, monkeypatch):
        """When clamp is disabled, behavior matches the original linear curve."""
        from trade_modules import conviction_sizer

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", False)
        # Original: 0.35 + (score/100) * 0.65
        assert conviction_sizer.get_conviction_multiplier(0) == 0.35
        assert conviction_sizer.get_conviction_multiplier(100) == 1.00
        # Mid-point ≈ 0.675
        assert abs(conviction_sizer.get_conviction_multiplier(50) - 0.675) < 0.001

    def test_default_is_clamped(self):
        """The flag defaults to True so the system ships safe."""
        from trade_modules import conviction_sizer

        assert conviction_sizer.CONVICTION_CLAMP_TO_UNITY is True


class TestConvictionClampInQuarterKelly:
    """quarter_kelly_size_pct must also honor the clamp via expected_return_table."""

    def test_clamped_kelly_uses_uniform_expected_return(self, monkeypatch):
        """With clamp on, expected α should not vary by conviction bucket.

        Otherwise Kelly is still concentrating bets on noise via μ.
        """
        from trade_modules import conviction_sizer

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", True)
        size_low = conviction_sizer.quarter_kelly_size_pct(
            conviction=45,
            atr_pct_daily=1.5,
        )
        size_high = conviction_sizer.quarter_kelly_size_pct(
            conviction=85,
            atr_pct_daily=1.5,
        )
        # Same ATR + clamped conviction → same size_pct
        assert size_low["size_pct"] == size_high["size_pct"], (
            f"Clamped sizing should be conviction-invariant; "
            f"got low={size_low['size_pct']} high={size_high['size_pct']}"
        )

    def test_unclamped_kelly_still_varies_by_conviction(self, monkeypatch):
        from trade_modules import conviction_sizer

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", False)
        size_low = conviction_sizer.quarter_kelly_size_pct(
            conviction=45,
            atr_pct_daily=1.5,
        )
        size_high = conviction_sizer.quarter_kelly_size_pct(
            conviction=85,
            atr_pct_daily=1.5,
        )
        # Without clamp, high conviction → larger size
        assert size_high["size_pct"] > size_low["size_pct"], (
            f"Unclamped sizing should scale with conviction; "
            f"got low={size_low['size_pct']} high={size_high['size_pct']}"
        )

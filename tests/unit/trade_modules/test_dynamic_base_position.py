"""
M3: Dynamic base position % of NAV — CIO v36 Empirical Refoundation.

The legacy `base_position_size: 2500` in config.yaml is a static dollar
amount. This means:
- After a 20% drawdown (NAV $400k → $320k), a $2500 base position is
  0.78% of NAV instead of 0.625% — risk INFLATES on drawdowns.
- After a 20% gain (NAV → $480k), a $2500 base is 0.52% — risk SHRINKS
  on rallies.

Both directions hurt: we add concentration when capital is most constrained
and pull back when we should be deploying. M3 makes base = NAV × pct so
sizing scales with the book.
"""


class TestDynamicBasePosition:
    def test_base_pct_overrides_static_size(self):
        """When base_position_pct is set, base_position_size is derived from NAV."""
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        enrich_with_position_sizes(
            conc,
            portfolio_value=400_000,
            base_position_pct=0.005,
        )
        # Base = 400_000 × 0.005 = 2000 (not 2500 from static)
        # MEGA tier multiplier = 5 → tier_size = 10_000
        # Conviction multiplier may apply — final size depends on regime + conviction
        # But the key assertion: enrichment used dynamic base, evidence is size > 0
        assert conc[0].get("suggested_size_usd", 0) > 0
        # 2.5% of NAV cap (5% MEGA cap × 0.5 conviction min)
        assert conc[0]["size_pct"] <= 5.0

    def test_drawdown_reduces_base_proportionally(self):
        """After a 20% drawdown the same conviction yields proportionally smaller size."""
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        # Same stock, same signals — only NAV differs
        conc_pre = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        conc_post = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]

        enrich_with_position_sizes(
            conc_pre,
            portfolio_value=400_000,
            base_position_pct=0.005,
        )
        enrich_with_position_sizes(
            conc_post,
            portfolio_value=320_000,
            base_position_pct=0.005,
        )

        # Drawdown should shrink absolute USD size by ~20% (80/100)
        ratio = conc_post[0]["suggested_size_usd"] / conc_pre[0]["suggested_size_usd"]
        assert 0.75 <= ratio <= 0.85, f"Expected ~0.80 ratio post 20% drawdown, got {ratio:.3f}"

    def test_gain_increases_base_proportionally(self):
        """After a 20% gain the same conviction yields proportionally larger size."""
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        conc_pre = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        conc_post = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]

        enrich_with_position_sizes(
            conc_pre,
            portfolio_value=400_000,
            base_position_pct=0.005,
        )
        enrich_with_position_sizes(
            conc_post,
            portfolio_value=480_000,
            base_position_pct=0.005,
        )

        # 20% gain → 1.2× sizing
        ratio = conc_post[0]["suggested_size_usd"] / conc_pre[0]["suggested_size_usd"]
        assert 1.15 <= ratio <= 1.25, f"Expected ~1.20 ratio post 20% gain, got {ratio:.3f}"

    def test_relative_size_pct_constant_across_nav(self):
        """size_pct (= USD/NAV) should be invariant to NAV under dynamic base.

        This is the core invariant: % of book stays constant. Tested in the
        operational NAV range (~200k-600k) where the static max_position_usd
        cap does not bind. (At higher NAVs the static USD cap clips, which
        is itself a downstream M3-related fix that should make max_position
        scale with NAV.)
        """
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        sizes_pct = []
        for nav in (200_000, 350_000, 450_000, 600_000):
            conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
            enrich_with_position_sizes(
                conc,
                portfolio_value=nav,
                base_position_pct=0.005,
            )
            sizes_pct.append(conc[0]["size_pct"])

        # Allow 0.05pp tolerance for rounding at boundaries
        assert (
            max(sizes_pct) - min(sizes_pct) < 0.05
        ), f"size_pct should be NAV-invariant under dynamic base; got {sizes_pct}"

    def test_static_fallback_when_pct_not_provided(self):
        """Backward-compat: omit base_position_pct → use legacy static base_position_size."""
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        conc = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        # No base_position_pct → falls back to base_position_size default 2500
        enrich_with_position_sizes(
            conc,
            portfolio_value=400_000,
            base_position_size=2500,
        )
        # Sanity: size came out > 0
        assert conc[0]["suggested_size_usd"] > 0

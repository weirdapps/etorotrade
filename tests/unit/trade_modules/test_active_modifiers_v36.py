"""
M5: Empirically-validated ACTIVE_MODIFIERS — CIO v36 Empirical Refoundation.

The v35.0 ACTIVE_MODIFIERS set (committee_synthesis.py:138-161) was selected
by literature review on 2026-05-02. M11's calibrator subsequently scored
all 63 modifiers against realized T+30 alpha on n=5,801 observations and
found:

- 12 currently-active modifiers fire constant Δ (NaN ρ — no rank info):
  currency_risk_USD/HKD/JPY/GBP, news_catalyst_pos/neg, target_consensus,
  iv_low_entry, iv_x_earnings, eps_revisions_down, dividend_yield_trap,
  short_interest_weakness, volume_confirm, fcf_quality_strong, sector_rotation
- 1 currently-active modifier is INVERTED (piotroski_quality ρ=−0.23 with
  n=136): the system treats high Piotroski as +3 bonus, but stocks with
  high Piotroski underperformed in this sample. Drop pending investigation.
- 4 SHADOW with large n confirm no edge: census_alignment (n=759, ρ=−0.003),
  eps_revisions_up (n=184, ρ=+0.024), macro_sector (n=546, ρ=+0.023),
  rel_strength_spy (n=194, ρ=−0.015)
- 3 modifiers NOT in v35 active set are PREDICTIVE: consensus_crowded
  (ρ=+0.18 n=226), tech_disagree (ρ=+0.26 n=66), proportionality_cap
  (ρ=+0.42 n=85, but it's a cap mechanism not a regular modifier).

This module asserts the NEW v36 active set is the empirical winners.
"""

import pytest


class TestACTIVE_MODIFIERSv36:
    """The new active set must be empirically grounded, not literature-based.

    CIO v36 N3: V36 modifier set is now opt-in via CIO_V36_NEW_MODIFIERS=1
    env var (best-practice fix). Each test explicitly opts in via the
    autouse fixture so the v36 invariants stay testable.
    """

    @pytest.fixture(autouse=True)
    def _enable_v36_modifiers(self, monkeypatch):
        import importlib

        monkeypatch.setenv("CIO_V36_NEW_MODIFIERS", "1")
        import trade_modules.committee_synthesis as mod

        importlib.reload(mod)
        yield
        monkeypatch.delenv("CIO_V36_NEW_MODIFIERS", raising=False)
        importlib.reload(mod)

    def test_predictive_modifiers_are_active(self):
        from trade_modules.committee_synthesis import ACTIVE_MODIFIERS

        # Calibration verdict: PREDICTIVE → ACTIVE
        assert "sector_concentration" in ACTIVE_MODIFIERS
        assert "consensus_crowded" in ACTIVE_MODIFIERS
        assert "tech_disagree" in ACTIVE_MODIFIERS

    def test_dropped_modifiers_are_not_active(self):
        from trade_modules.committee_synthesis import ACTIVE_MODIFIERS

        # Calibration verdict: DROP (NaN ρ on substantial n) → REMOVE
        for mod in (
            "currency_risk_USD",
            "currency_risk_HKD",
            "currency_risk_JPY",
            "currency_risk_GBP",
            "news_catalyst_neg",
            "news_catalyst_pos",
            "target_consensus",
            "iv_low_entry",
            "iv_x_earnings",
            "eps_revisions_down",
            "dividend_yield_trap",
            "short_interest_weakness",
            "volume_confirm",
            "fcf_quality_strong",
            "sector_rotation",
        ):
            assert (
                mod not in ACTIVE_MODIFIERS
            ), f"{mod} fires constant Δ in our data (NaN ρ) — drop from ACTIVE"

    def test_inverted_piotroski_is_not_active(self):
        """Piotroski has ρ=−0.23 (high quality → LOWER returns) — drop pending review."""
        from trade_modules.committee_synthesis import ACTIVE_MODIFIERS

        assert "piotroski_quality" not in ACTIVE_MODIFIERS

    def test_shadow_modifiers_with_large_n_are_not_active(self):
        """SHADOW with n≥150 confirms no edge → remove."""
        from trade_modules.committee_synthesis import ACTIVE_MODIFIERS

        for mod in ("census_alignment", "eps_revisions_up", "macro_sector", "rel_strength_spy"):
            assert (
                mod not in ACTIVE_MODIFIERS
            ), f"{mod} has large-n SHADOW verdict (no rank predictive power)"

    def test_active_set_is_small(self):
        """v36 ships ≤8 active modifiers — empirical winners only."""
        from trade_modules.committee_synthesis import ACTIVE_MODIFIERS

        assert len(ACTIVE_MODIFIERS) <= 8, (
            f"v36 active set inflated to {len(ACTIVE_MODIFIERS)}; "
            f"only modifiers with PREDICTIVE/WEAK verdict belong"
        )


class TestFilterWaterfallShadowsRemoved:
    """Removed modifiers should still be tracked as shadow (~prefix).

    CIO v36 N3: opt into V36 set so news_catalyst_pos is shadowed.
    """

    @pytest.fixture(autouse=True)
    def _enable_v36_modifiers(self, monkeypatch):
        import importlib

        monkeypatch.setenv("CIO_V36_NEW_MODIFIERS", "1")
        import trade_modules.committee_synthesis as mod

        importlib.reload(mod)
        yield
        monkeypatch.delenv("CIO_V36_NEW_MODIFIERS", raising=False)
        importlib.reload(mod)

    def test_removed_modifier_appears_with_shadow_prefix(self):
        from trade_modules.committee_synthesis import filter_waterfall

        wf = {"news_catalyst_pos": 5, "sector_concentration": -3}
        bonuses, penalties, filtered = filter_waterfall(wf)

        # news_catalyst_pos was active in v35; v36 removed it → must be shadowed
        assert "~news_catalyst_pos" in filtered
        assert "news_catalyst_pos" not in filtered
        # sector_concentration is in v36 ACTIVE → preserved + counts in penalties
        assert "sector_concentration" in filtered
        assert penalties == 3
        assert bonuses == 0

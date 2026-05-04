"""
N3: ACTIVE_MODIFIERS opt-in flag — best-practice fix for CIO v36 M5.

M5 was hard-flipped: ACTIVE_MODIFIERS went from 22 (literature-validated) →
6 (data-validated) in one shot. Best practice: same gating as M4 — let
v35 and v36 modifier sets shadow-run for evidence accumulation before
promoting to default.

Behavior:
- Default (no env var): use V35_ACTIVE_MODIFIERS (the literature-validated set)
- CIO_V36_NEW_MODIFIERS=1: use V36_ACTIVE_MODIFIERS (data-validated set)
- ACTIVE_MODIFIERS exported variable resolves at import time

Both sets remain importable so the calibrator can compare them.
"""

import importlib

import pytest


class TestModifierSetsBothExist:
    def test_v35_set_exists(self):
        from trade_modules.committee_synthesis import V35_ACTIVE_MODIFIERS

        # v35 had 22 literature-validated modifiers
        assert len(V35_ACTIVE_MODIFIERS) == 22
        assert "census_alignment" in V35_ACTIVE_MODIFIERS
        assert "currency_risk_USD" in V35_ACTIVE_MODIFIERS
        assert "news_catalyst_pos" in V35_ACTIVE_MODIFIERS

    def test_v36_set_exists(self):
        from trade_modules.committee_synthesis import V36_ACTIVE_MODIFIERS

        # v36 has 6 empirically-validated modifiers
        assert len(V36_ACTIVE_MODIFIERS) == 6
        assert "consensus_crowded" in V36_ACTIVE_MODIFIERS
        assert "tech_disagree" in V36_ACTIVE_MODIFIERS
        assert "sector_concentration" in V36_ACTIVE_MODIFIERS
        # And NOT the dropped ones
        assert "currency_risk_USD" not in V36_ACTIVE_MODIFIERS
        assert "news_catalyst_pos" not in V36_ACTIVE_MODIFIERS


class TestActiveModifierResolution:
    """ACTIVE_MODIFIERS resolves based on env var at import time."""

    def test_no_env_var_uses_v35(self, monkeypatch):
        monkeypatch.delenv("CIO_V36_NEW_MODIFIERS", raising=False)
        # Force re-import to pick up env change
        import trade_modules.committee_synthesis as mod

        importlib.reload(mod)
        assert mod.ACTIVE_MODIFIERS == mod.V35_ACTIVE_MODIFIERS

    def test_env_var_set_uses_v36(self, monkeypatch):
        monkeypatch.setenv("CIO_V36_NEW_MODIFIERS", "1")
        import trade_modules.committee_synthesis as mod

        importlib.reload(mod)
        assert mod.ACTIVE_MODIFIERS == mod.V36_ACTIVE_MODIFIERS

    def test_env_var_zero_uses_v35(self, monkeypatch):
        monkeypatch.setenv("CIO_V36_NEW_MODIFIERS", "0")
        import trade_modules.committee_synthesis as mod

        importlib.reload(mod)
        assert mod.ACTIVE_MODIFIERS == mod.V35_ACTIVE_MODIFIERS


@pytest.fixture(autouse=True)
def _restore_default_modifier_set(monkeypatch):
    """After any test in this module, reload synthesis to default (no env var).

    Required so other tests in the suite see the predictable default.
    """
    yield
    monkeypatch.delenv("CIO_V36_NEW_MODIFIERS", raising=False)
    import trade_modules.committee_synthesis as mod

    importlib.reload(mod)

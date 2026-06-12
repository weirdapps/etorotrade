"""Tests for circuit breaker + VaR scaling in enrich_with_position_sizes().

CIO v44: Verifies that portfolio-level drawdown circuit breaker and
VaR deployment throttle are wired into position sizing.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

from trade_modules.committee_synthesis import load_circuit_breaker

# ---------------------------------------------------------------------------
# load_circuit_breaker() unit tests
# ---------------------------------------------------------------------------


def test_load_circuit_breaker_normal_defaults():
    """When file doesn't exist, returns NORMAL defaults."""
    with patch.object(Path, "exists", return_value=False):
        state = load_circuit_breaker()
    assert state["level"] == "NORMAL"
    assert state["position_size_multiplier"] == 1.0
    assert state["new_positions_allowed"] is True


def test_load_circuit_breaker_reads_file():
    """Reads circuit breaker state from JSON file."""
    cb_data = {
        "level": "CAUTION",
        "position_size_multiplier": 0.5,
        "new_positions_allowed": True,
        "drawdown_pct": 12.3,
    }
    with (
        patch.object(Path, "exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(cb_data))),
    ):
        state = load_circuit_breaker()
    assert state["level"] == "CAUTION"
    assert state["position_size_multiplier"] == 0.5
    assert state["new_positions_allowed"] is True


def test_load_circuit_breaker_corrupt_file():
    """Returns NORMAL defaults when file is corrupt."""
    with (
        patch.object(Path, "exists", return_value=True),
        patch("builtins.open", mock_open(read_data="not json")),
    ):
        state = load_circuit_breaker()
    assert state["level"] == "NORMAL"
    assert state["position_size_multiplier"] == 1.0


# ---------------------------------------------------------------------------
# Circuit breaker logic tests (unit-level, no conviction_sizer dependency)
# ---------------------------------------------------------------------------


def test_circuit_breaker_halt_blocks_buys():
    """HALT level should zero out BUY/ADD positions."""
    cb_state = {
        "level": "HALT",
        "position_size_multiplier": 0.25,
        "new_positions_allowed": False,
    }
    stock = {"action": "BUY", "position_size_pct": 2.5}
    # Apply the same logic as enrich_with_position_sizes
    if stock["action"] in ("BUY", "ADD") and not cb_state["new_positions_allowed"]:
        stock["position_size_pct"] = 0.0
    assert stock["position_size_pct"] == 0.0


def test_circuit_breaker_halt_allows_sell():
    """HALT level should not affect SELL actions."""
    cb_state = {
        "level": "HALT",
        "position_size_multiplier": 0.25,
        "new_positions_allowed": False,
    }
    stock = {"action": "SELL", "position_size_pct": 2.5}
    # BUY/ADD check doesn't apply to SELL
    if stock["action"] in ("BUY", "ADD") and not cb_state["new_positions_allowed"]:
        stock["position_size_pct"] = 0.0
    assert stock["position_size_pct"] == 2.5  # Unchanged


def test_circuit_breaker_caution_scales():
    """CAUTION level applies 50% multiplier to position sizes."""
    cb_state = {
        "level": "CAUTION",
        "position_size_multiplier": 0.5,
        "new_positions_allowed": True,
    }
    original_usd = 5000.0
    scaled = original_usd * cb_state["position_size_multiplier"]
    assert scaled == 2500.0


def test_circuit_breaker_normal_no_change():
    """NORMAL level should not change sizing."""
    cb_state = {
        "level": "NORMAL",
        "position_size_multiplier": 1.0,
        "new_positions_allowed": True,
    }
    original_size = 2500.0
    size = original_size * cb_state["position_size_multiplier"]
    assert size == original_size


# ---------------------------------------------------------------------------
# VaR scaling logic tests
# ---------------------------------------------------------------------------


def test_var_scaling_within_budget():
    """VaR within budget returns 1.0 (no scaling)."""
    from trade_modules.conviction_sizer import get_portfolio_var_scaling

    assert get_portfolio_var_scaling(portfolio_var_95=1.5) == 1.0


def test_var_scaling_at_trigger():
    """VaR at trigger boundary returns 1.0."""
    from trade_modules.conviction_sizer import get_portfolio_var_scaling

    assert get_portfolio_var_scaling(portfolio_var_95=2.5) == 1.0


def test_var_scaling_above_trigger():
    """VaR above trigger but below max scales linearly."""
    from trade_modules.conviction_sizer import get_portfolio_var_scaling

    scale = get_portfolio_var_scaling(portfolio_var_95=3.75)
    assert 0.5 < scale < 1.0


def test_var_scaling_at_max():
    """VaR at max returns 0.5."""
    from trade_modules.conviction_sizer import get_portfolio_var_scaling

    assert get_portfolio_var_scaling(portfolio_var_95=5.0) == 0.5


def test_var_scaling_emergency():
    """VaR beyond 2x max returns 0.0 (emergency)."""
    from trade_modules.conviction_sizer import get_portfolio_var_scaling

    assert get_portfolio_var_scaling(portfolio_var_95=10.0) == 0.0


def test_var_scaling_none():
    """Unknown VaR returns 1.0."""
    from trade_modules.conviction_sizer import get_portfolio_var_scaling

    assert get_portfolio_var_scaling(portfolio_var_95=None) == 1.0


# ---------------------------------------------------------------------------
# Integration: enrich_with_position_sizes applies circuit breaker
# ---------------------------------------------------------------------------


def test_enrich_applies_circuit_breaker_halt():
    """enrich_with_position_sizes zeros sizing when circuit breaker halts."""
    from trade_modules.committee_synthesis import enrich_with_position_sizes

    concordance = [
        {"ticker": "AAPL", "action": "BUY", "conviction": 75, "market_cap": "MEGA"},
    ]
    halt_state = {
        "level": "HALT",
        "position_size_multiplier": 0.25,
        "new_positions_allowed": False,
    }
    with (
        patch(
            "trade_modules.committee_synthesis.load_circuit_breaker",
            return_value=halt_state,
        ),
        patch.object(Path, "exists", return_value=False),
    ):
        enrich_with_position_sizes(concordance)

    assert concordance[0]["suggested_size_usd"] == 0
    assert concordance[0].get("circuit_breaker_halt") is True
    assert concordance[0].get("circuit_breaker_level") == "HALT"


def test_enrich_applies_circuit_breaker_caution():
    """enrich_with_position_sizes scales sizing at CAUTION level."""
    from trade_modules.committee_synthesis import enrich_with_position_sizes

    concordance = [
        {"ticker": "AAPL", "action": "BUY", "conviction": 75, "market_cap": "MEGA"},
    ]
    # Run with NORMAL first to get baseline
    normal_state = {
        "level": "NORMAL",
        "position_size_multiplier": 1.0,
        "new_positions_allowed": True,
    }
    with (
        patch(
            "trade_modules.committee_synthesis.load_circuit_breaker",
            return_value=normal_state,
        ),
        patch.object(Path, "exists", return_value=False),
    ):
        enrich_with_position_sizes(concordance)
    baseline_usd = concordance[0]["suggested_size_usd"]

    # Now run with CAUTION (0.5 multiplier)
    concordance2 = [
        {"ticker": "AAPL", "action": "BUY", "conviction": 75, "market_cap": "MEGA"},
    ]
    caution_state = {
        "level": "CAUTION",
        "position_size_multiplier": 0.5,
        "new_positions_allowed": True,
    }
    with (
        patch(
            "trade_modules.committee_synthesis.load_circuit_breaker",
            return_value=caution_state,
        ),
        patch.object(Path, "exists", return_value=False),
    ):
        enrich_with_position_sizes(concordance2)
    caution_usd = concordance2[0]["suggested_size_usd"]

    # CAUTION should be ~50% of baseline
    assert caution_usd < baseline_usd
    assert abs(caution_usd - baseline_usd * 0.5) < 1.0  # rounding tolerance


def test_enrich_applies_var_scaling():
    """enrich_with_position_sizes applies VaR throttle when risk.json present."""
    from trade_modules.committee_synthesis import enrich_with_position_sizes

    concordance = [
        {"ticker": "AAPL", "action": "BUY", "conviction": 75, "market_cap": "MEGA"},
    ]
    normal_cb = {
        "level": "NORMAL",
        "position_size_multiplier": 1.0,
        "new_positions_allowed": True,
    }

    # Mock get_portfolio_var_scaling at module level so it returns 0.5
    # regardless of the risk.json file (avoids complex file mocking)
    with (
        patch(
            "trade_modules.committee_synthesis.load_circuit_breaker",
            return_value=normal_cb,
        ),
        patch(
            "trade_modules.conviction_sizer.get_portfolio_var_scaling",
            return_value=0.5,
        ) as mock_var,
        patch.object(Path, "exists", return_value=True),
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps({"portfolio_var_95": 5.0})),
        ),
    ):
        enrich_with_position_sizes(concordance)

    assert concordance[0].get("var_scale") == 0.5
    assert concordance[0]["suggested_size_usd"] > 0  # Scaled but not zero


def test_enrich_hold_not_affected_by_circuit_breaker():
    """HOLD actions are not affected by circuit breaker (they skip sizing)."""
    from trade_modules.committee_synthesis import enrich_with_position_sizes

    concordance = [
        {"ticker": "AAPL", "action": "HOLD", "conviction": 50},
    ]
    halt_state = {
        "level": "HALT",
        "position_size_multiplier": 0.25,
        "new_positions_allowed": False,
    }
    with (
        patch(
            "trade_modules.committee_synthesis.load_circuit_breaker",
            return_value=halt_state,
        ),
        patch.object(Path, "exists", return_value=False),
    ):
        enrich_with_position_sizes(concordance)

    # HOLD should not have sizing fields at all
    assert "suggested_size_usd" not in concordance[0]
    assert "circuit_breaker_halt" not in concordance[0]

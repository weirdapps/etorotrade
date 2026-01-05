"""
Analysis Module - Split into focused sub-modules

This module provides backward compatibility by re-exporting all functions
from the split modules: tiers, signals, and criteria.
"""

# Import from tiers module
from .tiers import (
    calculate_exret,
    _safe_calc_exret,
    _parse_percentage,
    _parse_market_cap,
    _determine_market_cap_tier,
)

# Import from signals module
from .signals import (
    calculate_action_vectorized,
    calculate_action,
    filter_buy_opportunities_wrapper,
    filter_sell_candidates_wrapper,
    filter_hold_candidates_wrapper,
)

# Import from criteria module
from .criteria import (
    _check_confidence_criteria,
    _check_sell_criteria,
    _check_buy_criteria,
    _process_color_based_on_criteria,
    _apply_color_coding,
    _filter_notrade_tickers,
    process_buy_opportunities,
)

__all__ = [
    # Tier utilities
    "calculate_exret",
    "_safe_calc_exret",
    "_parse_percentage",
    "_parse_market_cap",
    "_determine_market_cap_tier",
    # Signal generation
    "calculate_action_vectorized",
    "calculate_action",
    "filter_buy_opportunities_wrapper",
    "filter_sell_candidates_wrapper",
    "filter_hold_candidates_wrapper",
    # Criteria evaluation
    "_check_confidence_criteria",
    "_check_sell_criteria",
    "_check_buy_criteria",
    "_process_color_based_on_criteria",
    "_apply_color_coding",
    "_filter_notrade_tickers",
    "process_buy_opportunities",
]

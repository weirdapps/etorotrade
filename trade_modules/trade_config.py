"""
MASTER TRADE CONFIGURATION - SINGLE SOURCE OF TRUTH

This module contains ALL trading parameters and display configurations.
To modify any buy/sell/hold/inconclusive threshold or display column,
edit this file ONLY.

Author: EtoroTrade System
Version: 1.0.0 (Production)
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from .yaml_config_loader import get_yaml_config


class TradeOption(Enum):
    """Trading analysis options."""
    PORTFOLIO = "p"
    MARKET = "m"
    ETORO = "e"
    TRADE = "t"
    INPUT = "i"


class TradeAction(Enum):
    """Trading actions."""
    BUY = "B"
    SELL = "S"
    HOLD = "H"
    INCONCLUSIVE = "I"


class TradeConfig:
    """
    Centralized configuration for all trading parameters and display settings.
    
    CRITICAL: This is the SINGLE SOURCE OF TRUTH.
    Modify ANY threshold or display setting HERE and ONLY HERE.
    """

    # ============================================
    # SECTION 1: TRADING THRESHOLDS
    # ============================================

    # Universal thresholds applied to all options
    # These will be overridden by YAML config if available
    UNIVERSAL_THRESHOLDS = {
        "min_analyst_count": 5,
        "min_price_targets": 5,
        "min_market_cap": 1_000_000_000,  # $1B minimum
        "max_processing_time": 300,  # 5 minutes max
    }
    
    @classmethod
    def get_universal_thresholds(cls) -> Dict[str, Any]:
        """Get universal thresholds, preferring YAML config over hardcoded values."""
        yaml_config = get_yaml_config()
        if yaml_config.is_config_available():
            yaml_thresholds = yaml_config.get_universal_thresholds()
            if yaml_thresholds:
                return yaml_thresholds
        return cls.UNIVERSAL_THRESHOLDS

    # Market cap tier definitions
    # These will be overridden by YAML config if available
    TIER_THRESHOLDS = {
        "value_tier_min": 100_000_000_000,  # $100B
        "growth_tier_min": 5_000_000_000,   # $5B
        # Below $5B = BETS tier
    }
    
    @classmethod
    def get_tier_definitions(cls) -> Dict[str, Any]:
        """Get tier threshold definitions, preferring YAML config over hardcoded values."""
        yaml_config = get_yaml_config()
        if yaml_config.is_config_available():
            yaml_thresholds = yaml_config.get_tier_thresholds()
            if yaml_thresholds:
                return yaml_thresholds
        return cls.TIER_THRESHOLDS

    # Option-specific trading thresholds
    THRESHOLDS = {
        # Portfolio Analysis (option: p)
        "portfolio": {
            "buy": {
                "min_upside": 20.0,
                "min_buy_percentage": 75.0,
                "min_beta": 0.25,
                "max_beta": 3.0,
                "min_forward_pe": 0.5,
                "max_forward_pe": 65.0,
                "min_trailing_pe": 0.5,
                "max_trailing_pe": 80.0,
                "max_peg": 2.5,
                "max_short_interest": 2.5,
                "min_exret": 0.15,  # 15%
                "min_earnings_growth": -10.0,
                "min_price_performance": -15.0,
            },
            "sell": {
                "max_upside": 5.0,
                "min_buy_percentage": 65.0,
                "max_forward_pe": 65.0,
                "min_short_interest": 3.0,
                "min_beta": 3.0,
                "max_exret": 0.025,  # 2.5%
                "max_earnings_growth": -15.0,
                "max_price_performance": -35.0,
            },
            "hold": {
                # Anything between buy and sell criteria
            },
            "inconclusive": {
                "insufficient_analyst_coverage": True,
                "missing_key_data": True,
            }
        },

        # Market Analysis (option: m)
        "market": {
            "buy": {
                "min_upside": 25.0,
                "min_buy_percentage": 80.0,
                "min_beta": 0.25,
                "max_beta": 3.0,
                "min_forward_pe": 0.5,
                "max_forward_pe": 60.0,
                "min_trailing_pe": 0.5,
                "max_trailing_pe": 75.0,
                "max_peg": 2.0,
                "max_short_interest": 2.0,
                "min_exret": 0.20,  # 20%
                "min_earnings_growth": -15.0,
                "min_price_performance": -10.0,
            },
            "sell": {
                "max_upside": 8.0,
                "min_buy_percentage": 60.0,
                "max_forward_pe": 65.0,
                "min_short_interest": 3.0,
                "min_beta": 3.0,
                "max_exret": 0.08,  # 8%
                "max_earnings_growth": -15.0,
                "max_price_performance": -35.0,
            }
        },

        # eToro Analysis (option: e)
        "etoro": {
            "buy": {
                "min_upside": 15.0,
                "min_buy_percentage": 70.0,
                "min_beta": 0.25,
                "max_beta": 3.0,
                "min_forward_pe": 0.5,
                "max_forward_pe": 65.0,
                "min_trailing_pe": 0.5,
                "max_trailing_pe": 85.0,
                "max_peg": 2.5,
                "max_short_interest": 2.0,
                "min_exret": 0.10,  # 10%
                "min_earnings_growth": -5.0,
                "min_price_performance": -20.0,
            },
            "sell": {
                "max_upside": 5.0,
                "min_buy_percentage": 50.0,
                "max_forward_pe": 65.0,
                "min_short_interest": 3.0,
                "min_beta": 3.0,
                "max_exret": 0.05,  # 5%
                "max_earnings_growth": -15.0,
                "max_price_performance": -35.0,
            }
        },

        # Trade Opportunities (option: t)
        "trade": {
            "buy": {
                "min_upside": 25.0,
                "min_buy_percentage": 80.0,
                "min_beta": 0.25,
                "max_beta": 3.0,
                "min_forward_pe": 0.5,
                "max_forward_pe": 60.0,
                "min_trailing_pe": 0.5,
                "max_trailing_pe": 60.0,
                "max_peg": 2.0,
                "max_short_interest": 2.0,
                "min_exret": 0.20,  # 20%
                "min_earnings_growth": -15.0,
                "min_price_performance": -10.0,
            },
            "sell": {
                "max_upside": 12.0,
                "min_buy_percentage": 70.0,
                "max_forward_pe": 65.0,
                "min_short_interest": 3.0,
                "min_beta": 3.0,
                "max_exret": 0.10,  # 10%
                "max_earnings_growth": -15.0,
                "max_price_performance": -35.0,
            }
        },

        # Manual Input (option: i)
        "input": {
            "buy": {
                "min_upside": 20.0,
                "min_buy_percentage": 75.0,
                "min_beta": 0.25,
                "max_beta": 3.0,
                "min_forward_pe": 0.5,
                "max_forward_pe": 65.0,
                "min_trailing_pe": 0.5,
                "max_trailing_pe": 80.0,
                "max_peg": 2.5,
                "max_short_interest": 2.5,
                "min_exret": 0.15,  # 15%
                "min_earnings_growth": -10.0,
                "min_price_performance": -15.0,
            },
            "sell": {
                "max_upside": 5.0,
                "min_buy_percentage": 65.0,
                "max_forward_pe": 65.0,
                "min_short_interest": 3.0,
                "min_beta": 3.0,
                "max_exret": 0.025,  # 2.5%
                "max_earnings_growth": -15.0,
                "max_price_performance": -35.0,
            }
        }
    }

    # ============================================
    # SECTION 2: DISPLAY COLUMN PROFILES
    # ============================================

    DISPLAY_PROFILES = {
        # Portfolio Analysis (option: p)
        "portfolio": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "BETA", "PEF", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "sort_by": "EXRET",
            "sort_order": "desc",
            "max_rows": 50,
        },

        # Market Analysis (option: m)
        "market": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "UPSIDE", "%BUY", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "BETA", "PEF", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "UPSIDE", "%BUY", "EXRET", "BS"],
            "sort_by": "UPSIDE",
            "sort_order": "desc",
            "max_rows": 100,
        },

        # eToro Analysis (option: e)
        "etoro": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "sort_by": "EXRET",
            "sort_order": "desc",
            "max_rows": 30,
        },

        # Trade Opportunities - Buy (option: t, sub: b)
        "trade_buy": {
            "console": ["#", "TICKER", "COMPANY", "UPSIDE", "%BUY", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "UPSIDE", "%BUY", "EXRET", "BS"],
            "sort_by": "UPSIDE",
            "sort_order": "desc",
            "max_rows": 20,
        },

        # Trade Opportunities - Sell (option: t, sub: s)
        "trade_sell": {
            "console": ["#", "TICKER", "COMPANY", "PRICE", "UPSIDE", "BS", "REASON"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "UPSIDE", "%BUY", "BS", "REASON"],
            "html": ["#", "TICKER", "COMPANY", "PRICE", "UPSIDE", "BS", "REASON"],
            "sort_by": "UPSIDE",
            "sort_order": "asc",
            "max_rows": 20,
        },

        # Trade Opportunities - Hold (option: t, sub: h)
        "trade_hold": {
            "console": ["#", "TICKER", "COMPANY", "PRICE", "UPSIDE", "%BUY", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "PRICE", "UPSIDE", "%BUY", "BS"],
            "sort_by": "UPSIDE",
            "sort_order": "desc",
            "max_rows": 30,
        },

        # Manual Input (option: i)
        "input": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "sort_by": "EXRET",
            "sort_order": "desc",
            "max_rows": 10,
        }
    }

    # ============================================
    # SECTION 3: FORMATTING RULES
    # ============================================

    FORMAT_RULES = {
        "PRICE": {
            "type": "currency",
            "decimals": 2,
            "symbol": "$",
            "threshold_high_decimals": 100,  # Use 2 decimals if > $100
        },
        "TARGET": {
            "type": "currency",
            "decimals": 2,
            "symbol": "$",
            "threshold_high_decimals": 100,
        },
        "UPSIDE": {
            "type": "percentage",
            "decimals": 1,
            "suffix": "%",
            "color_positive": "green",
            "color_negative": "red",
        },
        "%BUY": {
            "type": "percentage",
            "decimals": 0,
            "suffix": "%",
        },
        "EXRET": {
            "type": "percentage",
            "decimals": 1,
            "suffix": "%",
            "color_positive": "green",
            "color_negative": "red",
        },
        "CAP": {
            "type": "market_cap",
            "units": ["M", "B", "T"],  # Million, Billion, Trillion
            "decimals": 1,
        },
        "BETA": {
            "type": "decimal",
            "decimals": 2,
        },
        "PEF": {
            "type": "decimal",
            "decimals": 1,
        },
        "PET": {
            "type": "decimal",
            "decimals": 1,
        },
        "PEG": {
            "type": "decimal",
            "decimals": 2,
        },
        "SI": {
            "type": "percentage",
            "decimals": 1,
            "suffix": "%",
        },
        "BS": {
            "type": "action",
            "colors": {
                "B": {
                    "console": "\033[92m",  # Green
                    "html": "#28a745",
                    "name": "BUY"
                },
                "S": {
                    "console": "\033[91m",  # Red
                    "html": "#dc3545",
                    "name": "SELL"
                },
                "H": {
                    "console": "",  # No color
                    "html": "#6c757d",
                    "name": "HOLD"
                },
                "I": {
                    "console": "\033[93m",  # Yellow
                    "html": "#ffc107",
                    "name": "INCONCLUSIVE"
                }
            }
        }
    }

    # ============================================
    # SECTION 4: HELPER METHODS
    # ============================================

    @classmethod
    def get_thresholds(cls, option: str, action: str, tier: str = None) -> Dict[str, Any]:
        """
        Get trading thresholds for a specific option and action, optionally tier-specific.
        
        Args:
            option: Trading option (p, m, e, t, i)
            action: Trading action (buy, sell, hold, inconclusive)
            tier: Market cap tier (V, G, B) for tier-specific thresholds
            
        Returns:
            Dictionary of thresholds
        """
        option_map = {
            "p": "portfolio",
            "m": "market", 
            "e": "etoro",
            "t": "trade",
            "i": "input"
        }
        
        option_key = option_map.get(option, option)
        base_thresholds = cls.THRESHOLDS.get(option_key, {}).get(action, {})
        
        # If tier is specified, get tier-specific thresholds
        if tier:
            tier_thresholds = cls.get_tier_thresholds(tier, action)
            # Merge tier-specific thresholds with base thresholds
            combined = base_thresholds.copy()
            combined.update(tier_thresholds)
            return combined
        
        return base_thresholds
    
    @classmethod
    def get_tier_thresholds(cls, tier: str, action: str) -> Dict[str, Any]:
        """Get tier-specific thresholds for VALUE/GROWTH/BETS tiers."""
        # Try to load from YAML first
        yaml_config = get_yaml_config()
        
        tier_map = {"V": "value", "G": "growth", "B": "bets"}
        tier_name = tier_map.get(tier, "bets")
        
        if yaml_config.is_config_available():
            # Load from YAML configuration
            tier_criteria = yaml_config.get_tier_criteria(tier_name)
            return tier_criteria.get(action, {})
        
        # Fallback to hardcoded values if YAML not available
        if action == "buy":
            if tier_name == "value":
                return {
                    "min_upside": 15.0,              # Reasonable upside for large-caps
                    "min_buy_percentage": 70.0,      # Strong analyst consensus required
                    "min_beta": 0.25,                # Minimum beta allowed
                    "max_beta": 3.0,                 # Maximum beta allowed
                    "min_forward_pe": 0.5,           # Minimum forward PE
                    "max_forward_pe": 65.0,          # Maximum forward PE
                    "min_trailing_pe": 0.5,          # Minimum trailing PE
                    "max_trailing_pe": 85.0,         # Higher trailing PE allowed for stability
                    "max_peg": 2.5,                  # PEG requirement
                    "max_short_interest": 2.0,       # Short interest tolerance
                    "min_exret": 0.10,               # Expected return threshold (10%)
                    "min_earnings_growth": -15.0,    # More tolerance for earnings variation
                    "min_price_performance": -15.0,  # More tolerance for price performance
                }
            elif tier_name == "growth":
                return {
                    "min_upside": 20.0,              # Standard upside requirement
                    "min_buy_percentage": 75.0,      # Standard analyst consensus
                    "min_beta": 0.25,                # Standard beta range
                    "max_beta": 3.0,                 # Higher beta limit
                    "min_forward_pe": 0.5,           # Standard PE requirements
                    "max_forward_pe": 60.0,          # Standard forward PE limit
                    "min_trailing_pe": 0.5,          # Standard trailing PE minimum
                    "max_trailing_pe": 75.0,         # Standard trailing PE limit
                    "max_peg": 2.0,                  # Standard PEG requirement
                    "max_short_interest": 2.0,       # Standard short interest
                    "min_exret": 0.15,               # Expected return threshold (15%)
                    "min_earnings_growth": -10.0,    # Standard earnings tolerance
                    "min_price_performance": -10.0,  # Standard price performance tolerance
                }
            else:  # bets
                return {
                    "min_upside": 25.0,              # Higher upside for small caps
                    "min_buy_percentage": 80.0,      # Strong consensus required
                    "min_beta": 0.25,                # Standard beta minimum
                    "max_beta": 3.0,                 # Allow higher volatility
                    "min_forward_pe": 0.5,           # Standard PE requirements
                    "max_forward_pe": 50.0,          # Lower PE limit for speculation
                    "min_trailing_pe": 0.5,          # Standard trailing PE minimum
                    "max_trailing_pe": 60.0,         # Lower trailing PE limit
                    "max_peg": 1.5,                  # Stricter PEG for small caps
                    "max_short_interest": 1.5,       # Lower short interest tolerance
                    "min_exret": 0.20,               # Higher expected return (20%)
                    "min_earnings_growth": -5.0,     # Less tolerance for declining earnings
                    "min_price_performance": -5.0,   # Less tolerance for poor performance
                }
        elif action == "sell":
            if tier_name == "value":
                return {
                    "max_upside": 8.0,               # Modest upside trigger for large caps
                    "min_buy_percentage": 60.0,      # Lower consensus for sell
                    "max_forward_pe": 70.0,          # Higher PE tolerance for value
                    "min_short_interest": 3.5,       # Higher short interest tolerance
                    "min_beta": 3.5,                 # Higher beta for sell
                    "max_exret": 0.06,               # Lower expected return (6%)
                    "max_earnings_growth": -20.0,    # More tolerance for earnings decline
                    "max_price_performance": -40.0,  # More tolerance for price decline
                }
            elif tier_name == "growth":
                return {
                    "max_upside": 5.0,               # Lower upside trigger for growth
                    "min_buy_percentage": 65.0,      # Standard sell consensus
                    "max_forward_pe": 65.0,          # Standard PE limit
                    "min_short_interest": 3.0,       # Standard short interest
                    "min_beta": 3.0,                 # Standard beta for sell
                    "max_exret": 0.05,               # Lower expected return (5%)
                    "max_earnings_growth": -15.0,    # Standard earnings tolerance
                    "max_price_performance": -35.0,  # Standard price performance
                }
            else:  # bets
                return {
                    "max_upside": 3.0,               # Very low upside trigger for speculation
                    "min_buy_percentage": 70.0,      # Higher consensus needed for sell
                    "max_forward_pe": 50.0,          # Lower PE tolerance
                    "min_short_interest": 2.5,       # Lower short interest tolerance
                    "min_beta": 2.5,                 # Lower beta for sell
                    "max_exret": 0.03,               # Very low expected return (3%)
                    "max_earnings_growth": -10.0,    # Less tolerance for earnings decline
                    "max_price_performance": -25.0,  # Less tolerance for price decline
                }
        
        return {}

    @classmethod
    def get_display_columns(cls, option: str, sub_option: str = None, output_type: str = "console") -> List[str]:
        """
        Get display columns for a specific option and output type.
        
        Args:
            option: Trading option (p, m, e, t, i)
            sub_option: Sub-option for trade analysis (b, s, h)
            output_type: Output type (console, csv, html)
            
        Returns:
            List of column names
        """
        profile_key = cls._get_profile_key(option, sub_option)
        profile = cls.DISPLAY_PROFILES.get(profile_key, {})
        return profile.get(output_type, [])

    @classmethod
    def get_sort_config(cls, option: str, sub_option: str = None) -> Dict[str, str]:
        """Get sorting configuration for an option."""
        profile_key = cls._get_profile_key(option, sub_option)
        profile = cls.DISPLAY_PROFILES.get(profile_key, {})
        return {
            "sort_by": profile.get("sort_by", ""),
            "sort_order": profile.get("sort_order", "desc"),
            "max_rows": profile.get("max_rows", 50)
        }

    @classmethod
    def get_format_rule(cls, column: str) -> Dict[str, Any]:
        """Get formatting rule for a column."""
        return cls.FORMAT_RULES.get(column, {"type": "text"})

    @classmethod
    def _get_profile_key(cls, option: str, sub_option: str = None) -> str:
        """Get the profile key for display configuration."""
        option_map = {
            "p": "portfolio",
            "m": "market",
            "e": "etoro", 
            "t": "trade",
            "i": "input"
        }
        
        base_key = option_map.get(option, option)
        
        if option == "t" and sub_option:
            sub_map = {"b": "buy", "s": "sell", "h": "hold"}
            sub_key = sub_map.get(sub_option, sub_option)
            return f"trade_{sub_key}"
        
        return base_key

    @classmethod
    def modify_threshold(cls, option: str, action: str, parameter: str, value: Any) -> None:
        """
        Modify a threshold value.
        
        Args:
            option: Trading option (p, m, e, t, i)
            action: Trading action (buy, sell)
            parameter: Parameter name
            value: New value
        """
        option_map = {
            "p": "portfolio",
            "m": "market",
            "e": "etoro", 
            "t": "trade",
            "i": "input"
        }
        
        option_key = option_map.get(option, option)
        if option_key in cls.THRESHOLDS and action in cls.THRESHOLDS[option_key]:
            cls.THRESHOLDS[option_key][action][parameter] = value

    @classmethod
    def modify_display_columns(cls, option: str, sub_option: str = None, 
                             output_type: str = "console", columns: List[str] = None) -> None:
        """
        Modify display columns for an option.
        
        Args:
            option: Trading option
            sub_option: Sub-option if applicable
            output_type: Output type (console, csv, html)
            columns: New column list
        """
        profile_key = cls._get_profile_key(option, sub_option)
        if profile_key in cls.DISPLAY_PROFILES and columns:
            cls.DISPLAY_PROFILES[profile_key][output_type] = columns


# Export for easy imports
__all__ = ["TradeConfig", "TradeOption", "TradeAction"]
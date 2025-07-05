"""
Trading criteria configuration.
"""

from typing import Any, Dict


class TradingCriteriaConfig:
    """Trading criteria configuration with immutable settings."""
    
    def __init__(self):
        """Initialize trading criteria configuration."""
        # Confidence thresholds
        self.min_analyst_count = 5
        self.min_price_targets = 5
        
        # SELL criteria thresholds
        self.sell_max_upside = 5.0                    # Sell if upside < 5%
        self.sell_min_buy_percentage = 65.0           # Sell if buy% < 65%
        self.sell_max_forward_pe = 50.0               # Sell if PEF > 50
        self.sell_max_peg = 3.0                       # Sell if PEG > 3.0
        self.sell_max_short_interest = 2.0            # Sell if SI > 2%
        self.sell_max_beta = 3.0                      # Sell if Beta > 3.0
        self.sell_min_exret = 5.0                     # Sell if EXRET < 5.0
        
        # BUY criteria thresholds
        self.buy_min_upside = 20.0                    # Buy if upside >= 20%
        self.buy_min_buy_percentage = 85.0            # Buy if buy% >= 85%
        self.buy_min_beta = 0.25                      # Buy if Beta > 0.25
        self.buy_max_beta = 2.5                       # Buy if Beta <= 2.5
        self.buy_max_peg = 2.5                        # Buy if PEG < 2.5
        self.buy_max_short_interest = 1.5             # Buy if SI <= 1.5%
        self.buy_min_exret = 15.0                     # Buy if EXRET >= 15.0
        self.buy_min_forward_pe = 0.5                 # Buy if PEF > 0.5
        self.buy_max_forward_pe = 45.0                # Buy if PEF <= 45.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "CONFIDENCE": {
                "MIN_ANALYST_COUNT": self.min_analyst_count,
                "MIN_PRICE_TARGETS": self.min_price_targets,
            },
            "SELL": {
                "SELL_MAX_UPSIDE": self.sell_max_upside,
                "SELL_MIN_BUY_PERCENTAGE": self.sell_min_buy_percentage,
                "SELL_MIN_FORWARD_PE": self.sell_max_forward_pe,
                "SELL_MAX_PEG": self.sell_max_peg,
                "SELL_MIN_SHORT_INTEREST": self.sell_max_short_interest,
                "SELL_MIN_BETA": self.sell_max_beta,
                "SELL_MAX_EXRET": self.sell_min_exret,
            },
            "BUY": {
                "BUY_MIN_UPSIDE": self.buy_min_upside,
                "BUY_MIN_BUY_PERCENTAGE": self.buy_min_buy_percentage,
                "BUY_MIN_BETA": self.buy_min_beta,
                "BUY_MAX_BETA": self.buy_max_beta,
                "BUY_MAX_PEG": self.buy_max_peg,
                "BUY_MAX_SHORT_INTEREST": self.buy_max_short_interest,
                "BUY_MIN_EXRET": self.buy_min_exret,
                "BUY_MIN_FORWARD_PE": self.buy_min_forward_pe,
                "BUY_MAX_FORWARD_PE": self.buy_max_forward_pe,
            },
        }
    
    def update_for_testing(self, **kwargs) -> None:
        """Update configuration for testing purposes.
        
        WARNING: Only use this in tests!
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown trading criteria config key: {key}")
    
    def create_test_copy(self, **overrides) -> 'TradingCriteriaConfig':
        """Create a copy with test overrides."""
        copy = TradingCriteriaConfig()
        copy.update_for_testing(**overrides)
        return copy
"""
MASTER TRADE CONFIGURATION - SINGLE SOURCE OF TRUTH

This module contains ALL trading parameters and display configurations.
To modify any buy/sell/hold/inconclusive threshold or display column,
edit this file ONLY.

Author: EtoroTrade System
Version: 1.0.0 (Production)
"""

from typing import ClassVar, Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path
import yaml
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
        "min_analyst_count": 4,  # Standard for $5B+ stocks
        "min_price_targets": 4,
        "min_market_cap": 2_000_000_000,  # $2B hard floor
        "small_cap_threshold": 5_000_000_000,  # $5B boundary
        "small_cap_min_analysts": 6,  # $2-5B needs more coverage
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
        "mega_tier_min": 500_000_000_000,    # $500B+
        "large_tier_min": 100_000_000_000,   # $100B-500B
        "mid_tier_min": 10_000_000_000,      # $10B-100B
        "small_tier_min": 2_000_000_000,     # $2B-10B
        # Below $2B = MICRO tier
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
    THRESHOLDS: ClassVar[Dict[str, Dict[str, Dict[str, Any]]]] = {
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
                "min_roe": 8.0,  # Return on Equity minimum (%)
                "max_debt_equity": 200.0,  # Debt-to-Equity maximum (%)
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
                "min_roe": 5.0,  # SELL if ROE drops below 5%
                "max_debt_equity": 250.0,  # SELL if DE exceeds 250%
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
                "min_roe": 8.0,  # Return on Equity minimum (%)
                "max_debt_equity": 200.0,  # Debt-to-Equity maximum (%)
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
                "min_roe": 5.0,  # SELL if ROE drops below 5%
                "max_debt_equity": 250.0,  # SELL if DE exceeds 250%
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
                "min_roe": 8.0,  # Return on Equity minimum (%)
                "max_debt_equity": 200.0,  # Debt-to-Equity maximum (%)
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
                "min_roe": 5.0,  # SELL if ROE drops below 5%
                "max_debt_equity": 250.0,  # SELL if DE exceeds 250%
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
                "min_roe": 8.0,  # Return on Equity minimum (%)
                "max_debt_equity": 200.0,  # Debt-to-Equity maximum (%)
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
                "min_roe": 5.0,  # SELL if ROE drops below 5%
                "max_debt_equity": 250.0,  # SELL if DE exceeds 250%
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
                "min_roe": 8.0,  # Return on Equity minimum (%)
                "max_debt_equity": 200.0,  # Debt-to-Equity maximum (%)
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
                "min_roe": 5.0,  # SELL if ROE drops below 5%
                "max_debt_equity": 250.0,  # SELL if DE exceeds 250%
            }
        }
    }

    # ============================================
    # SECTION 1B: SECTOR-SPECIFIC RULES
    # ============================================
    # These rules override standard ROE/DE thresholds for specific sectors
    # where different capital structures are normal/expected

    SECTOR_RULES: ClassVar[Dict[str, Dict[str, Any]]] = {
        'FINANCIAL': {
            'description': 'Banks, Asset Managers, Insurance - Leverage is their business model',
            'min_roe_buy': 6.0,  # Lower threshold for mature financials
            'min_roe_sell': 3.0,  # SELL if drops below 3%
            'max_debt_equity_buy': 500.0,  # Much higher - banks typically 500-1500%
            'max_debt_equity_sell': 800.0,  # SELL if exceeds 800%
            'tickers': ['C', 'STT', 'WBS', 'RITM', 'MET', 'GL', 'SCHW', 'BAC', 'JPM', 'WFC', 'GS', 'MS'],
        },
        'REIT': {
            'description': 'REITs - Negative ROE due to depreciation is normal',
            'skip_roe': True,  # Ignore ROE entirely for REITs
            'max_debt_equity_buy': 300.0,
            'max_debt_equity_sell': 400.0,
            'tickers': ['CCI', 'AMT', 'SBAC', 'EQIX', 'DLR', 'PLD'],
        },
        'MLP': {
            'description': 'Master Limited Partnerships - Energy infrastructure with extreme leverage',
            'min_roe_buy': 15.0,  # Should have high ROE to justify the leverage
            'min_roe_sell': 8.0,
            'max_debt_equity_buy': 800.0,  # Very high leverage is structural
            'max_debt_equity_sell': 1000.0,
            'tickers': ['TRGP', 'EPD', 'ET', 'MMP', 'MPLX'],
        },
        'PHARMA_HIGH_LEVERAGE': {
            'description': 'Pharma companies with strategic R&D/acquisition leverage',
            'min_roe_buy': 20.0,  # Must have excellent ROE to justify leverage
            'min_roe_sell': 12.0,
            'max_debt_equity_buy': 300.0,  # Allow higher for R&D financing
            'max_debt_equity_sell': 400.0,
            'tickers': ['LLY', 'BMY', 'ABBV', 'GILD'],
        },
        'EQUIPMENT_FINANCING': {
            'description': 'Equipment manufacturers with captive financing arms',
            'min_roe_buy': 12.0,
            'min_roe_sell': 8.0,
            'max_debt_equity_buy': 300.0,  # Financing operations inflate DE
            'max_debt_equity_sell': 400.0,
            'tickers': ['DE', 'CAT', 'CNH'],
        },
        'UTILITY': {
            'description': 'Utilities and infrastructure - Capital intensive with stable cash flows',
            'min_roe_buy': 8.0,  # Stable but lower ROE is normal
            'min_roe_sell': 5.0,
            'max_debt_equity_buy': 250.0,  # Higher leverage for infrastructure
            'max_debt_equity_sell': 350.0,
            'tickers': ['VIE.PA', 'NEE', 'DUK', 'SO', 'D'],
        },
        'PAYMENT_PROCESSORS': {
            'description': 'Asset-light payment networks with operating leverage from share buybacks',
            'min_roe_buy': 30.0,  # Must have excellent ROE to justify leverage
            'min_roe_sell': 20.0,
            'max_debt_equity_buy': 250.0,  # Higher threshold - leverage from buybacks, not operational debt
            'max_debt_equity_sell': 400.0,
            'max_pe_vs_sector_buy': 1.8,  # Payment networks trade at premium to fin services (tech moat)
            'tickers': ['V', 'MA', 'AXP', 'PYPL'],
        },
        'TELECOM': {
            'description': 'Telecommunications - Capital intensive with stable recurring revenues',
            'min_roe_buy': 10.0,  # Lower ROE threshold - infrastructure heavy
            'min_roe_sell': 5.0,
            'max_debt_equity_buy': 220.0,  # Higher leverage for spectrum/infrastructure
            'max_debt_equity_sell': 350.0,
            'max_short_interest_buy': 4.0,  # Higher SI tolerance - often shorted for yield
            'tickers': ['TMUS', 'VZ', 'T', 'VOD', 'ORAN'],
        },
    }

    @classmethod
    def get_sector_from_ticker(cls, ticker: str) -> Optional[str]:
        """
        Detect which sector a ticker belongs to based on SECTOR_RULES.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Sector name if found, None otherwise
        """
        if not ticker:
            return None

        ticker_upper = ticker.upper()

        for sector_name, sector_config in cls.SECTOR_RULES.items():
            if ticker_upper in sector_config.get('tickers', []):
                return sector_name

        return None

    @classmethod
    def get_sector_adjusted_thresholds(cls, ticker: str, action: str, base_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get sector-adjusted ROE and DE thresholds for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            action: 'buy' or 'sell'
            base_criteria: Base criteria dictionary from THRESHOLDS

        Returns:
            Adjusted criteria dictionary with sector-specific ROE/DE thresholds
        """
        sector = cls.get_sector_from_ticker(ticker)

        if not sector:
            # No sector override, return base criteria as-is
            return base_criteria

        sector_config = cls.SECTOR_RULES.get(sector, {})
        adjusted_criteria = base_criteria.copy()

        # Apply sector-specific ROE thresholds
        if action == 'buy':
            if 'skip_roe' in sector_config and sector_config['skip_roe']:
                # Remove ROE requirement entirely (for REITs)
                adjusted_criteria.pop('min_roe', None)
            elif 'min_roe_buy' in sector_config:
                adjusted_criteria['min_roe'] = sector_config['min_roe_buy']

            if 'max_debt_equity_buy' in sector_config:
                adjusted_criteria['max_debt_equity'] = sector_config['max_debt_equity_buy']

            if 'max_short_interest_buy' in sector_config:
                adjusted_criteria['max_short_interest'] = sector_config['max_short_interest_buy']

            if 'max_pe_vs_sector_buy' in sector_config:
                adjusted_criteria['max_pe_vs_sector'] = sector_config['max_pe_vs_sector_buy']

        elif action == 'sell':
            if 'skip_roe' in sector_config and sector_config['skip_roe']:
                # Remove ROE requirement entirely (for REITs)
                adjusted_criteria.pop('min_roe', None)
            elif 'min_roe_sell' in sector_config:
                adjusted_criteria['min_roe'] = sector_config['min_roe_sell']

            if 'max_debt_equity_sell' in sector_config:
                adjusted_criteria['max_debt_equity'] = sector_config['max_debt_equity_sell']

        return adjusted_criteria

    # ============================================
    # SECTION 1C: SECTOR BENCHMARKS (GICS-based)
    # ============================================
    # Used for sector-relative valuations (PE vs sector median)

    _sector_benchmarks_cache: ClassVar[Optional[Dict[str, Any]]] = None

    @classmethod
    def load_sector_benchmarks(cls) -> Dict[str, Any]:
        """Load sector benchmark valuations from YAML file."""
        if cls._sector_benchmarks_cache is not None:
            return cls._sector_benchmarks_cache

        benchmark_path = Path(__file__).parent / "sector_benchmarks.yaml"
        if benchmark_path.exists():
            try:
                with open(benchmark_path, 'r') as f:
                    cls._sector_benchmarks_cache = yaml.safe_load(f)
                    return cls._sector_benchmarks_cache
            except Exception:
                pass

        # Return empty dict if file not found
        cls._sector_benchmarks_cache = {}
        return cls._sector_benchmarks_cache

    @classmethod
    def get_sector_benchmarks(cls, sector: str) -> Dict[str, float]:
        """
        Get benchmark valuations for a GICS sector.

        Args:
            sector: Sector name from yfinance (e.g., "Technology", "Financial Services")

        Returns:
            Dictionary with median_pe, median_roe, median_de for the sector
        """
        benchmarks = cls.load_sector_benchmarks()
        if not benchmarks:
            return {"median_pe": 20.0, "median_roe": 15.0, "median_de": 80.0}

        # Get sector mapping
        mapping = benchmarks.get("sector_mapping", {})
        sectors = benchmarks.get("sectors", {})
        default = benchmarks.get("default", {"median_pe": 20.0, "median_roe": 15.0, "median_de": 80.0})

        # Apply mapping if sector name needs normalization
        mapped_sector = mapping.get(sector, sector) if sector else None

        if mapped_sector and mapped_sector in sectors:
            sector_data = sectors[mapped_sector]
            return {
                "median_pe": sector_data.get("median_pe", default["median_pe"]),
                "median_roe": sector_data.get("median_roe", default["median_roe"]),
                "median_de": sector_data.get("median_de", default["median_de"]),
            }

        return default

    @classmethod
    def calculate_pe_vs_sector(cls, pe_forward: Optional[float], sector: str) -> Optional[float]:
        """
        Calculate PE relative to sector median.

        Args:
            pe_forward: Forward PE ratio
            sector: Sector name from yfinance

        Returns:
            Ratio of PE to sector median (e.g., 1.5 means 50% above sector)
        """
        if pe_forward is None or pe_forward <= 0:
            return None

        benchmarks = cls.get_sector_benchmarks(sector)
        sector_median_pe = benchmarks.get("median_pe", 20.0)

        if sector_median_pe <= 0:
            return None

        return round(pe_forward / sector_median_pe, 2)

    # ============================================
    # SECTION 2: DISPLAY COLUMN PROFILES
    # ============================================

    DISPLAY_PROFILES: ClassVar[Dict[str, Dict[str, Any]]] = {
        # Portfolio Analysis (option: p)
        "portfolio": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "BETA", "PEF", "PP", "ROE", "DE", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "sort_by": "EXRET",
            "sort_order": "desc",
            "max_rows": 50,
        },

        # Market Analysis (option: m)
        "market": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "UPSIDE", "%BUY", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "BETA", "PEF", "PP", "ROE", "DE", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "UPSIDE", "%BUY", "EXRET", "BS"],
            "sort_by": "UPSIDE",
            "sort_order": "desc",
            "max_rows": 100,
        },

        # eToro Analysis (option: e)
        "etoro": {
            "console": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "PP", "ROE", "DE", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "sort_by": "EXRET",
            "sort_order": "desc",
            "max_rows": 30,
        },

        # Trade Opportunities - Buy (option: t, sub: b)
        "trade_buy": {
            "console": ["#", "TICKER", "COMPANY", "UPSIDE", "%BUY", "EXRET", "BS"],
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "PP", "ROE", "DE", "EXRET", "BS"],
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
            "csv": ["TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "%BUY", "PP", "ROE", "DE", "EXRET", "BS"],
            "html": ["#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", "EXRET", "BS"],
            "sort_by": "EXRET",
            "sort_order": "desc",
            "max_rows": 10,
        }
    }

    # ============================================
    # SECTION 3: FORMATTING RULES
    # ============================================

    FORMAT_RULES: ClassVar[Dict[str, Dict[str, Any]]] = {
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
        "ROE": {
            "type": "decimal",
            "decimals": 1,
        },
        "DE": {
            "type": "decimal",
            "decimals": 1,
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
    def get_thresholds(cls, option: str, action: str, ticker: str = None, market_cap: float = None) -> Dict[str, Any]:
        """
        Get trading thresholds based on geographic region and market cap tier.

        Args:
            option: Trading option (p, m, e, t, i)
            action: Trading action (buy, sell, hold, inconclusive)
            ticker: Stock ticker to determine region
            market_cap: Market capitalization to determine tier

        Returns:
            Dictionary of thresholds
        """
        # Determine region from ticker
        region = cls.get_region_from_ticker(ticker) if ticker else "us"

        # Determine tier from market cap
        tier = cls.get_tier_from_market_cap(market_cap) if market_cap else "mid"

        # Get region-tier specific thresholds from YAML
        yaml_config = get_yaml_config()
        if yaml_config.is_config_available():
            region_tier_criteria = yaml_config.get_region_tier_criteria(region, tier)
            if region_tier_criteria:
                return region_tier_criteria.get(action, {})

        # Fallback to old system if YAML not configured
        option_map = {
            "p": "portfolio",
            "m": "market",
            "e": "etoro",
            "t": "trade",
            "i": "input"
        }

        option_key = option_map.get(option, option)
        return cls.THRESHOLDS.get(option_key, {}).get(action, {})
    
    @classmethod
    def get_region_from_ticker(cls, ticker: str) -> str:
        """
        Determine geographic region from ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Region code: 'us', 'eu', or 'hk'
        """
        if not ticker:
            return "us"

        ticker = ticker.upper()

        # Hong Kong / China stocks
        if ticker.endswith(".HK"):
            return "hk"

        # European stocks
        eu_suffixes = [".L", ".CO", ".DE", ".PA", ".AS", ".BR", ".MI", ".MC", ".LI",
                       ".OL", ".ST", ".HE", ".WA", ".PR", ".AT", ".IR", ".IC"]
        if any(ticker.endswith(suffix) for suffix in eu_suffixes):
            return "eu"

        # Japanese stocks (could be added to Asia category if needed)
        if ticker.endswith(".T"):
            return "us"  # Treating Japan as developed similar to US for now

        # Default to US for stocks without suffix or with US suffixes
        return "us"

    @classmethod
    def get_tier_from_market_cap(cls, market_cap: float) -> str:
        """
        Determine tier from market capitalization.

        Args:
            market_cap: Market capitalization in dollars

        Returns:
            Tier name: 'mega', 'large', 'mid', 'small', or 'micro'
        """
        if not market_cap or market_cap <= 0:
            return "micro"

        # Get thresholds from YAML or use defaults
        thresholds = cls.get_tier_definitions()

        if market_cap >= thresholds.get("mega_tier_min", 500_000_000_000):
            return "mega"
        elif market_cap >= thresholds.get("large_tier_min", 100_000_000_000):
            return "large"
        elif market_cap >= thresholds.get("mid_tier_min", 10_000_000_000):
            return "mid"
        elif market_cap >= thresholds.get("small_tier_min", 2_000_000_000):
            return "small"
        else:
            return "micro"

    @classmethod
    def get_tier_thresholds(cls, tier: str, action: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        # Map old tier names to new system
        tier_map = {"V": "large", "G": "mid", "B": "small"}
        new_tier = tier_map.get(tier, "mid")

        yaml_config = get_yaml_config()
        if yaml_config.is_config_available():
            # Default to US region for legacy calls
            return yaml_config.get_region_tier_criteria("us", new_tier).get(action, {})
        
        # Fallback to hardcoded values if YAML not available
        if action == "buy":
            if new_tier == "large":
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
            elif new_tier == "mid":
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
            else:  # small
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
            if new_tier == "large":
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
            elif new_tier == "mid":
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
            else:  # small
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
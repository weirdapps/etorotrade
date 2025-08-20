"""
Unified Configuration Manager

Single source of truth for all configuration, combining config.yaml
with necessary runtime settings. Replaces the fragmented configuration
system with one clean interface.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set
from .yaml_config_loader import YamlConfigLoader


class ConfigManager:
    """Unified configuration management for the entire application."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._initialized:
            return
            
        # Load YAML configuration
        self.yaml_loader = YamlConfigLoader()
        self._config = self.yaml_loader.load_config()
        
        # Set up paths (from Python config)
        self.project_root = Path(__file__).parent.parent
        self.yahoofinance_root = self.project_root / "yahoofinance"
        
        # Directory paths
        self.paths = {
            "INPUT_DIR": str(self.yahoofinance_root / "input"),
            "OUTPUT_DIR": str(self.yahoofinance_root / "output"),
            "LOG_DIR": str(self.project_root / "logs"),
            "DEFAULT_LOG_FILE": str(self.project_root / "logs" / "yahoofinance.log"),
        }
        
        # Build file paths
        self.file_paths = self._build_file_paths()
        
        # Portfolio settings (merge YAML with defaults)
        self.portfolio = self._get_portfolio_config()
        
        # Rate limiting settings
        self.rate_limit = self._get_rate_limit_config()
        
        # Special tickers and grades
        self.positive_grades = {
            "Buy", "Outperform", "Strong Buy", "Overweight", "Accumulate",
            "Add", "Conviction Buy", "Top Pick", "Positive",
        }
        
        self.special_tickers = {
            "US_SPECIAL_CASES": {"BRK.A", "BRK.B", "BF.A", "BF.B"},
        }
        
        # Initialize ticker mappings
        self._init_ticker_mappings()
        
        self._initialized = True
    
    def _build_file_paths(self) -> Dict[str, str]:
        """Build file paths dictionary."""
        input_dir = self.paths["INPUT_DIR"]
        output_dir = self.paths["OUTPUT_DIR"]
        
        return {
            # Input files
            "MARKET_FILE": os.path.join(input_dir, "market.csv"),
            "PORTFOLIO_FILE": os.path.join(input_dir, "portfolio.csv"),
            "ETORO_FILE": os.path.join(input_dir, "etoro.csv"),
            "YFINANCE_FILE": os.path.join(input_dir, "yfinance.csv"),
            "NOTRADE_FILE": os.path.join(input_dir, "notrade.csv"),
            "CONS_FILE": os.path.join(input_dir, "cons.csv"),
            "US_TICKERS_FILE": os.path.join(input_dir, "us_tickers.csv"),
            "EUROPE_FILE": os.path.join(input_dir, "europe.csv"),
            "CHINA_FILE": os.path.join(input_dir, "china.csv"),
            "USA_FILE": os.path.join(input_dir, "usa.csv"),
            "USINDEX_FILE": os.path.join(input_dir, "usindex.csv"),
            # Output files
            "MARKET_OUTPUT": os.path.join(output_dir, "market.csv"),
            "PORTFOLIO_OUTPUT": os.path.join(output_dir, "portfolio.csv"),
            "BUY_OUTPUT": os.path.join(output_dir, "buy.csv"),
            "SELL_OUTPUT": os.path.join(output_dir, "sell.csv"),
            "HOLD_OUTPUT": os.path.join(output_dir, "hold.csv"),
            "MANUAL_OUTPUT": os.path.join(output_dir, "manual.csv"),
            "HTML_OUTPUT": os.path.join(output_dir, "index.html"),
        }
    
    def _get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio configuration, merging YAML with defaults."""
        # Defaults from Python config
        defaults = {
            "PORTFOLIO_VALUE": 450_000,
            "MIN_POSITION_USD": 1_000,
            "MAX_POSITION_USD": 40_000,
            "MAX_POSITION_PCT": 8.9,
            "BASE_POSITION_PCT": 0.5,
        }
        
        # Override with YAML if available
        position_sizing = self._config.get('position_sizing', {})
        if position_sizing:
            defaults["BASE_POSITION_PCT"] = position_sizing.get('base_position_size', 2500) / defaults["PORTFOLIO_VALUE"] * 100
        
        return defaults
    
    def _get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        performance = self._config.get('performance', {})
        
        return {
            "max_concurrent_requests": performance.get('max_concurrent_requests', 15),
            "request_timeout_seconds": performance.get('request_timeout_seconds', 30),
            "retry_attempts": performance.get('retry_attempts', 3),
            "rate_limit_per_second": 10,  # Default from Python config
            "burst_size": 20,  # Default from Python config
        }
    
    def get_tier_thresholds(self) -> Dict[str, Any]:
        """Get tier threshold definitions from YAML."""
        return self._config.get('tier_thresholds', {})
    
    def get_tier_criteria(self, tier: str) -> Dict[str, Any]:
        """Get criteria for a specific tier from YAML."""
        return self._config.get(tier, {})
    
    def get_universal_thresholds(self) -> Dict[str, Any]:
        """Get universal thresholds from YAML."""
        return self._config.get('universal_thresholds', {})
    
    def get_position_sizing_config(self) -> Dict[str, Any]:
        """Get position sizing configuration from YAML."""
        return self._config.get('position_sizing', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        Supports dot notation for nested keys (e.g., 'rate_limit.max_concurrent_requests')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    # Check instance attributes
                    if hasattr(self, keys[0]):
                        attr = getattr(self, keys[0])
                        if len(keys) > 1 and isinstance(attr, dict):
                            return attr.get(keys[1], default)
                        return attr if len(keys) == 1 else default
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_path(self, path_key: str) -> Optional[str]:
        """Get a file or directory path."""
        if path_key in self.paths:
            return self.paths[path_key]
        return self.file_paths.get(path_key)
    
    def _init_ticker_mappings(self):
        """Initialize ticker mapping data for dual-listed stocks."""
        # US Ticker -> Original Exchange Ticker mappings
        self.dual_listed_mappings = {
            # European stocks with US ADRs/cross-listings
            "NVO": "NOVO-B.CO",      # Novo Nordisk ADR → Copenhagen
            "SNY": "SAN.PA",         # Sanofi ADR → Paris  
            "ASML": "ASML.NV",       # ASML NASDAQ → Netherlands
            "SHEL": "SHEL.L",        # Shell ADR → London
            "UL": "ULVR.L",          # Unilever ADR → London
            "RDS.A": "SHEL.L",       # Shell (old ticker) → London
            "RDS.B": "SHEL.L",       # Shell (old ticker) → London
            "SAP": "SAP.DE",         # SAP ADR → Germany
            "TM": "7203.T",          # Toyota ADR → Tokyo
            "SONY": "6758.T",        # Sony ADR → Tokyo
            "NTT": "9432.T",         # NTT ADR → Tokyo
            
            # Hong Kong stocks with US ADRs
            "JD": "9618.HK",         # JD.com ADR → Hong Kong
            "JD.US": "9618.HK",      # JD.com US listing → Hong Kong
            "BABA": "9988.HK",       # Alibaba ADR → Hong Kong
            "TCEHY": "0700.HK",      # Tencent ADR → Hong Kong  
            "BYDDY": "1211.HK",      # BYD ADR → Hong Kong
            "MEITX": "3690.HK",      # Meituan ADR → Hong Kong
            "YUMC": "YUMC",          # Yum China (already US-based, no mapping needed)
            
            # Google share classes (GOOG is the main ticker)
            "GOOGL": "GOOG",         # Google Class A → Google Class C (main ticker)
            
            # Other cross-listings
            "TSM": "TSM",            # Taiwan Semiconductor (already properly listed)
            "RIO": "RIO.L",          # Rio Tinto ADR → London primary
            "BHP": "BHP.AX",         # BHP ADR → Australia primary
        }
        
        # Reverse mapping: Original Exchange Ticker -> US Ticker
        self.reverse_mappings = {v: k for k, v in self.dual_listed_mappings.items()}
        
        # Set of all tickers that have dual listings (for quick lookup)
        self.dual_listed_tickers = set(self.dual_listed_mappings.keys()) | set(self.dual_listed_mappings.values())
        
        # Geographic region mapping for dual-listed stocks
        self.ticker_geography = {
            # European tickers
            "NOVO-B.CO": "EU",
            "SAN.PA": "EU", 
            "ASML.NV": "EU",
            "SHEL.L": "UK",
            "ULVR.L": "UK",
            "SAP.DE": "EU",
            
            # Asian tickers
            "9618.HK": "HK",
            "9988.HK": "HK",
            "0700.HK": "HK",
            "1211.HK": "HK",
            "3690.HK": "HK",
            "7203.T": "JP",
            "6758.T": "JP",
            "9432.T": "JP",
            
            # Other regions
            "RIO.L": "UK",
            "BHP.AX": "AU",
        }
    
    def get_normalized_ticker(self, ticker: str) -> str:
        """
        Get the normalized (original exchange) ticker for a given ticker.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Normalized ticker (original exchange ticker if dual-listed, otherwise unchanged)
        """
        if not ticker:
            return ticker
            
        # Convert to uppercase for consistent matching
        ticker_upper = ticker.upper()
        
        # Handle special Copenhagen Stock Exchange ticker normalization
        if ticker_upper.endswith('.CO'):
            base_ticker = ticker_upper.split('.')[0]
            if base_ticker.endswith('B') and len(base_ticker) > 1:
                if base_ticker in ['MAERSKB', 'NOVOB', 'COLOB']:
                    if base_ticker == 'MAERSKB':
                        ticker_upper = 'MAERSK-B.CO'
                    elif base_ticker == 'NOVOB':
                        ticker_upper = 'NOVO-B.CO'
                    elif base_ticker == 'COLOB':
                        ticker_upper = 'COLO-B.CO'
        
        # If this is a US ticker with a mapped original, return the original
        if ticker_upper in self.dual_listed_mappings:
            return self.dual_listed_mappings[ticker_upper]
        
        # Return the ticker in standardized format (uppercase)
        return ticker_upper
    
    def get_us_ticker(self, ticker: str) -> str:
        """
        Get the US ticker equivalent for a given ticker.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            US ticker if available, otherwise the original ticker
        """
        if not ticker:
            return ticker
            
        # Convert to uppercase for consistent matching
        ticker_upper = ticker.upper()
        
        # If this is an original ticker with a US equivalent, return the US ticker
        if ticker_upper in self.reverse_mappings:
            return self.reverse_mappings[ticker_upper]
        
        # Return the ticker in standardized format (uppercase)
        return ticker_upper
    
    def is_dual_listed(self, ticker: str) -> bool:
        """
        Check if a ticker has dual listings.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            True if ticker has dual listings, False otherwise
        """
        if not ticker:
            return False
            
        return ticker.upper() in self.dual_listed_tickers
    
    def get_display_ticker(self, ticker: str) -> str:
        """
        Get the preferred display ticker (always the original exchange ticker).
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Preferred display ticker (original exchange ticker)
        """
        # Handle VIX pattern replacement: VIX, VIX.??? -> ^VIX for display consistency  
        if ticker and ticker.upper().startswith('VIX') and (ticker.upper() == 'VIX' or ticker.upper().startswith('VIX.')):
            return '^VIX'
        
        return self.get_normalized_ticker(ticker)
    
    def get_data_fetch_ticker(self, ticker: str) -> str:
        """
        Get the best ticker for data fetching (may use US ticker for better data availability).
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Best ticker for data fetching
        """
        # Handle VIX pattern replacement: VIX, VIX.??? -> ^VIX for data retrieval
        if ticker and ticker.upper().startswith('VIX') and (ticker.upper() == 'VIX' or ticker.upper().startswith('VIX.')):
            return '^VIX'  # Yahoo Finance uses ^VIX for the VIX index
        
        # For data fetching, we might want to use US tickers when available
        # as they often have better data coverage, but we'll normalize the results
        normalized = self.get_normalized_ticker(ticker)
        
        # For now, prefer the normalized ticker, but this can be customized
        # based on data quality preferences per ticker
        return normalized
    
    def get_ticker_geography(self, ticker: str) -> str:
        """
        Get the geographic region for a ticker.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Geographic region code (HK, EU, UK, US, JP, AU, etc.)
        """
        if not ticker:
            return "US"  # Default to US
            
        normalized_ticker = self.get_normalized_ticker(ticker)
        
        # Check explicit mapping first
        if normalized_ticker in self.ticker_geography:
            return self.ticker_geography[normalized_ticker]
        
        # Infer from ticker suffix
        if normalized_ticker.endswith('.HK'):
            return "HK"
        elif normalized_ticker.endswith('.L'):
            return "UK"  
        elif normalized_ticker.endswith(('.PA', '.DE', '.NV', '.MI', '.BR')):
            return "EU"
        elif normalized_ticker.endswith('.T'):
            return "JP"
        elif normalized_ticker.endswith('.AX'):
            return "AU"
        else:
            return "US"  # Default to US for unrecognized patterns
    
    def are_equivalent_tickers(self, ticker1: str, ticker2: str) -> bool:
        """
        Check if two tickers represent the same underlying asset.
        
        This is critical for portfolio filtering - if someone owns NVO, 
        they shouldn't see NOVO-B.CO as a buy opportunity (and vice versa).
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            
        Returns:
            True if tickers represent the same underlying asset, False otherwise
        """
        if not ticker1 or not ticker2:
            return False
        
        # Normalize both tickers to their canonical forms
        normalized1 = self.get_normalized_ticker(ticker1.upper())
        normalized2 = self.get_normalized_ticker(ticker2.upper())
        
        # If normalized forms are the same, they're equivalent
        if normalized1 == normalized2:
            return True
        
        # Check if one is the US version of the other
        us_ticker1 = self.get_us_ticker(normalized1)
        us_ticker2 = self.get_us_ticker(normalized2)
        
        # If either ticker maps to the other's US equivalent, they're the same
        return (normalized1 == us_ticker2 or normalized2 == us_ticker1 or 
                us_ticker1 == normalized2 or us_ticker2 == normalized1)
    
    def get_all_equivalent_tickers(self, ticker: str) -> Set[str]:
        """
        Get all known ticker variants for the same underlying asset.
        
        Args:
            ticker: Input ticker symbol
            
        Returns:
            Set of all equivalent ticker symbols (including the input)
        """
        if not ticker:
            return set()
        
        ticker_upper = ticker.upper()
        normalized = self.get_normalized_ticker(ticker_upper)
        us_ticker = self.get_us_ticker(normalized)
        
        # Start with the normalized form
        equivalents = {normalized}
        
        # Add the US ticker if different
        if us_ticker != normalized:
            equivalents.add(us_ticker)
        
        # Add the original input if different from normalized
        if ticker_upper != normalized:
            equivalents.add(ticker_upper)
        
        return equivalents
    
    def reload(self):
        """Reload configuration from YAML file."""
        self.yaml_loader._config_cache = None
        self._config = self.yaml_loader.load_config()
        self.portfolio = self._get_portfolio_config()
        self.rate_limit = self._get_rate_limit_config()
        self._init_ticker_mappings()  # Re-initialize ticker mappings


# Global instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def reload_config():
    """Force reload of configuration."""
    config = get_config()
    config.reload()

# Compatibility functions for smooth migration
def get_max_concurrent_requests() -> int:
    """Get max concurrent requests setting."""
    return get_config().rate_limit["max_concurrent_requests"]

def get_request_timeout() -> int:
    """Get request timeout in seconds."""
    return get_config().rate_limit["request_timeout_seconds"]

def get_portfolio_value() -> float:
    """Get portfolio value."""
    return get_config().portfolio["PORTFOLIO_VALUE"]

def get_input_dir() -> str:
    """Get input directory path."""
    return get_config().paths["INPUT_DIR"]

def get_output_dir() -> str:
    """Get output directory path."""
    return get_config().paths["OUTPUT_DIR"]
"""
Simple configuration loader for trading analysis system.

Loads user preferences from config.yaml and provides easy access
to configuration values throughout the application.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Default configuration - used if config.yaml doesn't exist
DEFAULT_CONFIG = {
    "data": {
        "portfolio_csv": "yahoofinance/input/portfolio.csv",
        "cache_enabled": True,
        "cache_ttl_hours": 24
    },
    "trading": {
        "value_threshold": 100,
        "growth_threshold": 5,
        "min_analysts": 5,
        "min_price_targets": 5
    },
    "output": {
        "save_to_csv": True,
        "save_to_html": True,
        "display_colors": True,
        "max_display_rows": 50
    },
    "performance": {
        "max_concurrent_requests": 10,
        "request_timeout_seconds": 30,
        "retry_attempts": 3
    },
    "logging": {
        "level": "INFO",
        "file": "logs/trading_analysis.log",
        "console": True
    }
}


class ConfigLoader:
    """Simple configuration loader for the trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in project root.
        """
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[Dict[str, Any]] = None
    
    def _find_config_file(self) -> str:
        """Find config.yaml in the project root."""
        # Start from current file and go up to find project root
        current_dir = Path(__file__).parent
        while current_dir.parent != current_dir:
            config_file = current_dir / "config.yaml"
            if config_file.exists():
                return str(config_file)
            current_dir = current_dir.parent
        
        # Return default path if not found
        return "config.yaml"
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        if self._config is not None:
            return self._config
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                
                # Merge user config with defaults
                self._config = self._merge_configs(DEFAULT_CONFIG, user_config)
            else:
                print(f"Config file not found at {self.config_path}, using defaults")
                self._config = DEFAULT_CONFIG.copy()
                
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            self._config = DEFAULT_CONFIG.copy()
        
        return self._config
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults recursively."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation.
        
        Args:
            key_path: Dot-separated path like 'trading.min_analysts'
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load()
        keys = key_path.split('.')
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current


# Global config instance for easy access
_config_loader = ConfigLoader()


def get_config() -> Dict[str, Any]:
    """Get the full configuration dictionary."""
    return _config_loader.load()


def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a specific config value using dot notation.
    
    Args:
        key_path: Dot-separated path like 'trading.min_analysts'
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return _config_loader.get(key_path, default)


def reload_config() -> Dict[str, Any]:
    """Force reload configuration from file."""
    _config_loader._config = None
    return _config_loader.load()
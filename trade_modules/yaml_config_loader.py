"""
YAML Trading Criteria Configuration Loader

This module loads trading criteria from the unified config.yaml file
and provides them to the TradeConfig system.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class YamlConfigLoader:
    """Loads trading criteria configuration from YAML file."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the YAML config loader.
        
        Args:
            config_path: Path to YAML config file. If None, searches for config.yaml
        """
        self.config_path = config_path or self._find_config_file()
        self._config_cache = None
    
    def _find_config_file(self) -> str:
        """Find the unified config file in the project root."""
        current_dir = Path(__file__).parent
        
        # Look for config.yaml in project root
        while current_dir.parent != current_dir:  # Not at filesystem root
            config_file = current_dir / "config.yaml"
            if config_file.exists():
                return str(config_file)
            current_dir = current_dir.parent
        
        # Fallback to current directory
        return "config.yaml"
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing all configuration data
        """
        if self._config_cache is not None:
            return self._config_cache
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_cache = yaml.safe_load(f) or {}
                print(f"✅ Loaded unified configuration from: {self.config_path}")
                return self._config_cache
            else:
                print(f"⚠️ Unified config file not found: {self.config_path}")
                print("   Using hardcoded defaults from trade_config.py")
                return {}
        except Exception as e:
            print(f"❌ Error loading unified config: {str(e)}")
            print("   Using hardcoded defaults from trade_config.py")
            return {}
    
    def get_tier_thresholds(self) -> Dict[str, Any]:
        """Get tier threshold definitions."""
        config = self.load_config()
        return config.get('tier_thresholds', {})
    
    def get_universal_thresholds(self) -> Dict[str, Any]:
        """Get universal thresholds applied to all tiers."""
        config = self.load_config()
        return config.get('universal_thresholds', {})
    
    def get_tier_criteria(self, tier: str) -> Dict[str, Any]:
        """
        Get criteria for a specific tier.
        
        Args:
            tier: Tier name ('value', 'growth', 'bets')
            
        Returns:
            Dictionary with buy/sell criteria for the tier
        """
        config = self.load_config()
        return config.get(tier, {})
    
    def get_position_sizing_config(self) -> Dict[str, Any]:
        """Get position sizing configuration."""
        config = self.load_config()
        return config.get('position_sizing', {})
    
    def is_config_available(self) -> bool:
        """Check if YAML config file is available and loaded successfully."""
        config = self.load_config()
        return len(config) > 0


# Global instance for easy access
_yaml_loader = YamlConfigLoader()

def get_yaml_config() -> YamlConfigLoader:
    """Get the global YAML config loader instance."""
    return _yaml_loader

def reload_config():
    """Force reload of configuration from YAML file."""
    # No need for global declaration since we're not reassigning _yaml_loader
    _yaml_loader._config_cache = None
    _yaml_loader.load_config()
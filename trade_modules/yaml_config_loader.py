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
        self._config_cache: Optional[Dict[str, Any]] = None
    
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
                    loaded = yaml.safe_load(f)
                self._config_cache = loaded if isinstance(loaded, dict) else {}
                # Configuration loaded silently
                return self._config_cache
            else:
                print(f"⚠️ Unified config file not found: {self.config_path}")
                print("   Using hardcoded defaults from trade_config.py")
                return {}
        except (yaml.YAMLError, OSError, IOError, UnicodeDecodeError) as e:
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

    def get_region_tier_criteria(self, region: str, tier: str) -> Dict[str, Any]:
        """
        Get criteria for a specific region and tier combination.

        Args:
            region: Region name ('us', 'eu', 'hk')
            tier: Tier name ('mega', 'large', 'mid', 'small', 'micro')

        Returns:
            Dictionary with buy/sell criteria for the region-tier combination
        """
        config = self.load_config()
        key = f"{region}_{tier}"
        return config.get(key, {})
    
    def get_position_sizing_config(self) -> Dict[str, Any]:
        """Get position sizing configuration."""
        config = self.load_config()
        return config.get('position_sizing', {})
    
    def is_config_available(self) -> bool:
        """Check if YAML config file is available and loaded successfully."""
        config = self.load_config()
        return len(config) > 0

    def is_signal_scoring_enabled(self) -> bool:
        """Check if multi-factor signal scoring is enabled."""
        config = self.load_config()
        return config.get('use_signal_scoring', False)

    def get_sell_scoring_config(self, region: str, tier: str) -> Dict[str, Any]:
        """
        Get sell scoring configuration for a specific region and tier.

        Args:
            region: Region name ('us', 'eu', 'hk')
            tier: Tier name ('mega', 'large', 'mid', 'small', 'micro')

        Returns:
            Dictionary with sell scoring thresholds and weights
        """
        config = self.load_config()

        # Try region-tier specific config first
        key = f"{region}_{tier}_sell_scoring"
        tier_config = config.get(key, {})

        # Get default config
        default_config = config.get('default_sell_scoring', {
            'score_threshold': 65,
            'hard_trigger_upside': -5,
            'hard_trigger_buy_pct': 35,
            'quality_override_buy_pct': 85,
            'quality_override_upside': 20,
            'quality_override_exret': 15,
            'weight_analyst': 0.35,
            'weight_momentum': 0.25,
            'weight_valuation': 0.20,
            'weight_fundamental': 0.20,
        })

        # Merge tier-specific with defaults (tier overrides defaults)
        merged = {**default_config, **tier_config}
        return merged

    def get_buy_scoring_config(self, region: str, tier: str) -> Dict[str, Any]:
        """
        Get buy scoring configuration for a specific region and tier.

        Args:
            region: Region name ('us', 'eu', 'hk')
            tier: Tier name ('mega', 'large', 'mid', 'small', 'micro')

        Returns:
            Dictionary with buy scoring weights
        """
        config = self.load_config()

        # Try region-tier specific config first
        key = f"{region}_{tier}_buy_scoring"
        tier_config = config.get(key, {})

        # Get default config
        default_config = config.get('default_buy_scoring', {
            'enabled': True,
            'weight_upside': 0.30,
            'weight_consensus': 0.25,
            'weight_momentum': 0.20,
            'weight_valuation': 0.15,
            'weight_fundamental': 0.10,
        })

        # Merge tier-specific with defaults (tier overrides defaults)
        merged = {**default_config, **tier_config}
        return merged


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
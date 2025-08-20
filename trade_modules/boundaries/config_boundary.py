"""
Configuration boundary interface for module decoupling.

This module provides a stable interface for configuration access
between trade_modules and yahoofinance packages.
"""

from typing import Dict, List, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..errors import ConfigurationError


logger = logging.getLogger(__name__)


class IConfigBoundary(ABC):
    """Interface for configuration operations across module boundaries."""
    
    @abstractmethod
    def get_trading_criteria_config(self) -> Dict[str, Any]:
        """Get trading criteria configuration."""
        pass
    
    @abstractmethod
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration."""
        pass
    
    @abstractmethod
    def get_file_paths_config(self) -> Dict[str, Any]:
        """Get file paths configuration."""
        pass
    
    @abstractmethod
    def get_provider_config(self) -> Dict[str, Any]:
        """Get data provider configuration."""
        pass
    
    @abstractmethod
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration section."""
        pass


class ConfigBoundary(IConfigBoundary):
    """
    Configuration boundary implementation.
    
    This class provides controlled access to configuration across
    module boundaries while maintaining clean separation.
    """
    
    def __init__(self):
        """Initialize the configuration boundary."""
        self._config_cache = {}
        self._config_loaded = False
    
    def get_trading_criteria_config(self) -> Dict[str, Any]:
        """
        Get trading criteria configuration.
        
        Returns:
            Dictionary with trading criteria configuration
        """
        if 'trading_criteria' not in self._config_cache:
            try:
                from trade_modules.config_manager import get_config
                config = get_config()
                # Merge tier criteria for backward compatibility
                self._config_cache['trading_criteria'] = {
                    **config.get_tier_criteria('value'),
                    **config.get_universal_thresholds()
                }
            except ImportError:
                logger.warning("Could not import config manager, using defaults")
                self._config_cache['trading_criteria'] = self._get_default_trading_criteria()
        
        return self._config_cache['trading_criteria']
    
    def get_display_config(self) -> Dict[str, Any]:
        """
        Get display configuration.
        
        Returns:
            Dictionary with display configuration
        """
        if 'display' not in self._config_cache:
            try:
                # Try to import display config from yahoofinance
                from yahoofinance.core.config import get_display_config
                self._config_cache['display'] = get_display_config()
            except ImportError:
                logger.warning("Could not import yahoofinance display config, using defaults")
                self._config_cache['display'] = self._get_default_display_config()
        
        return self._config_cache['display']
    
    def get_file_paths_config(self) -> Dict[str, Any]:
        """
        Get file paths configuration.
        
        Returns:
            Dictionary with file paths configuration
        """
        if 'file_paths' not in self._config_cache:
            try:
                from yahoofinance.core.config import get_file_paths_config
                self._config_cache['file_paths'] = get_file_paths_config()
            except ImportError:
                logger.warning("Could not import yahoofinance file paths config, using defaults")
                self._config_cache['file_paths'] = self._get_default_file_paths_config()
        
        return self._config_cache['file_paths']
    
    def get_provider_config(self) -> Dict[str, Any]:
        """
        Get data provider configuration.
        
        Returns:
            Dictionary with provider configuration
        """
        if 'provider' not in self._config_cache:
            try:
                from trade_modules.config_manager import get_config
                config = get_config()
                self._config_cache['provider'] = {
                    'default_provider': 'yahoo_finance',
                    'timeout': config.rate_limit.get('request_timeout_seconds', 30),
                    'max_retries': config.rate_limit.get('retry_attempts', 3),
                    'cache_enabled': True
                }
            except ImportError:
                logger.warning("Could not import config manager, using defaults")
                self._config_cache['provider'] = self._get_default_provider_config()
        
        return self._config_cache['provider']
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration section.
        
        Args:
            section: Configuration section name
            updates: Dictionary with configuration updates
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if section in self._config_cache:
                self._config_cache[section].update(updates)
            else:
                self._config_cache[section] = updates.copy()
            
            logger.info(f"Updated configuration section: {section}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration section {section}: {e}")
            return False
    
    def _get_default_trading_criteria(self) -> Dict[str, Any]:
        """Get default trading criteria configuration."""
        return {
            'min_market_cap': 1e9,  # $1B minimum
            'max_pe_ratio': 25,
            'min_volume': 100000,
            'max_beta': 2.0,
            'min_price': 5.0,
            'max_price': 1000.0,
            'min_expected_return': 0.05,  # 5%
            'min_confidence': 0.6,
            'sectors_exclude': [],
            'regions_include': ['US'],
            'max_portfolio_weight': 0.1,  # 10% max position
            'risk_tolerance': 'moderate'
        }
    
    def _get_default_display_config(self) -> Dict[str, Any]:
        """Get default display configuration."""
        return {
            'decimal_places': 2,
            'currency_symbol': '$',
            'percentage_format': '%.2f%%',
            'date_format': '%Y-%m-%d',
            'number_format': '{:,.2f}',
            'table_max_rows': 50,
            'html_template': 'default',
            'color_scheme': 'professional',
            'font_size': 'medium',
            'show_grid': True,
            'show_totals': True
        }
    
    def _get_default_file_paths_config(self) -> Dict[str, Any]:
        """Get default file paths configuration."""
        base_path = Path.cwd()
        output_path = base_path / 'yahoofinance' / 'output'
        input_path = base_path / 'yahoofinance' / 'input'
        
        return {
            'base_directory': str(base_path),
            'output_directory': str(output_path),
            'input_directory': str(input_path),
            'portfolio_file': str(input_path / 'portfolio.csv'),
            'buy_opportunities_file': str(output_path / 'buy.csv'),
            'sell_opportunities_file': str(output_path / 'sell.csv'),
            'hold_positions_file': str(output_path / 'hold.csv'),
            'market_data_file': str(output_path / 'market.csv'),
            'html_report_file': str(output_path / 'report.html'),
            'backup_directory': str(output_path / 'backups'),
            'log_directory': str(base_path / 'logs'),
            'cache_directory': str(base_path / '.cache')
        }
    
    def _get_default_provider_config(self) -> Dict[str, Any]:
        """Get default provider configuration."""
        return {
            'default_provider': 'yahoo_finance',
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1.0,
            'cache_enabled': True,
            'cache_ttl': 3600,  # 1 hour
            'rate_limit': {
                'requests_per_second': 10,
                'burst_limit': 50
            },
            'headers': {
                'User-Agent': 'YFinance-Client/1.0'
            },
            'verify_ssl': True,
            'proxy': None
        }
    
    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configuration sections.
        
        Returns:
            Dictionary with all configuration sections
        """
        return {
            'trading_criteria': self.get_trading_criteria_config(),
            'display': self.get_display_config(),
            'file_paths': self.get_file_paths_config(),
            'provider': self.get_provider_config()
        }
    
    def reload_config(self) -> bool:
        """
        Reload configuration from sources.
        
        Returns:
            True if reload successful, False otherwise
        """
        try:
            self._config_cache.clear()
            self._config_loaded = False
            
            # Pre-load all configurations
            self.get_trading_criteria_config()
            self.get_display_config()
            self.get_file_paths_config()
            self.get_provider_config()
            
            self._config_loaded = True
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            return False
    
    def validate_config(self) -> Dict[str, bool]:
        """
        Validate all configuration sections.
        
        Returns:
            Dictionary with validation results for each section
        """
        results = {}
        
        try:
            # Validate trading criteria
            criteria = self.get_trading_criteria_config()
            results['trading_criteria'] = self._validate_trading_criteria(criteria)
            
            # Validate display config
            display = self.get_display_config()
            results['display'] = self._validate_display_config(display)
            
            # Validate file paths
            file_paths = self.get_file_paths_config()
            results['file_paths'] = self._validate_file_paths_config(file_paths)
            
            # Validate provider config
            provider = self.get_provider_config()
            results['provider'] = self._validate_provider_config(provider)
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            results['validation_error'] = False
        
        return results
    
    def _validate_trading_criteria(self, config: Dict[str, Any]) -> bool:
        """Validate trading criteria configuration."""
        required_keys = ['min_market_cap', 'max_pe_ratio', 'min_volume']
        return all(key in config for key in required_keys)
    
    def _validate_display_config(self, config: Dict[str, Any]) -> bool:
        """Validate display configuration."""
        required_keys = ['decimal_places', 'currency_symbol', 'date_format']
        return all(key in config for key in required_keys)
    
    def _validate_file_paths_config(self, config: Dict[str, Any]) -> bool:
        """Validate file paths configuration."""
        required_keys = ['base_directory', 'output_directory', 'input_directory']
        return all(key in config for key in required_keys)
    
    def _validate_provider_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration."""
        required_keys = ['default_provider', 'timeout', 'max_retries']
        return all(key in config for key in required_keys)


# Create default boundary instance
default_config_boundary = ConfigBoundary()
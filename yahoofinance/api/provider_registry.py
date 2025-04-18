"""
Provider registry for Yahoo Finance API.

This module provides a registry for all available finance data providers,
ensuring consistent access and configuration across the application.
"""

from typing import Dict, Any, Optional, Union, Type, List, Set, cast

from ..core.logging_config import get_logger
from ..core.errors import ValidationError
from ..utils.dependency_injection import registry

# Import provider interfaces
from .providers.base_provider import FinanceDataProvider, AsyncFinanceDataProvider

# Import all provider implementations (but not directly dependent on them)
# Use lazy imports to avoid circular dependencies if needed

# Set up logging
logger = get_logger(__name__)

# Provider type mapping
PROVIDER_TYPES = {
    'yahoo': {
        'sync': 'yahoofinance.api.providers.yahoo_finance.YahooFinanceProvider',
        'async': 'yahoofinance.api.providers.async_yahoo_finance.AsyncYahooFinanceProvider',
        'async_enhanced': 'yahoofinance.api.providers.enhanced_async_yahoo_finance.EnhancedAsyncYahooFinanceProvider',
    },
    'yahooquery': {
        'sync': 'yahoofinance.api.providers.yahooquery_provider.YahooQueryProvider',
        'async': 'yahoofinance.api.providers.async_yahooquery_provider.AsyncYahooQueryProvider',
    },
    'hybrid': {
        'sync': 'yahoofinance.api.providers.hybrid_provider.HybridProvider',
        'async': 'yahoofinance.api.providers.async_hybrid_provider.AsyncHybridProvider',
    },
    'optimized': {
        'async': 'yahoofinance.api.providers.optimized_async_yfinance.OptimizedAsyncYFinanceProvider',
    },
}

# Default provider configuration
DEFAULT_PROVIDER_TYPE = 'hybrid'
DEFAULT_ASYNC_MODE = False
DEFAULT_ENHANCED = False

# Initialize the registry with provider factories
def initialize_registry():
    """
    Initialize the provider registry with all provider factories.
    
    This function registers all provider factory functions with the registry.
    It is called automatically when this module is imported.
    """
    import importlib
    
    # Register factory functions for all providers
    for provider_type, providers in PROVIDER_TYPES.items():
        for mode, provider_path in providers.items():
            # Parse module path and class name
            module_path, class_name = provider_path.rsplit('.', 1)
            
            # Create a factory function for this provider
            def create_factory(module_path, class_name):
                def factory(**kwargs):
                    try:
                        module = importlib.import_module(module_path)
                        provider_class = getattr(module, class_name)
                        return provider_class(**kwargs)
                    except ImportError as e:
                        logger.error(f"Failed to import provider {class_name}: {str(e)}")
                        raise ValidationError(f"Provider {class_name} is not available") from e
                    except AttributeError as e:
                        logger.error(f"Provider class {class_name} not found in {module_path}: {str(e)}")
                        raise ValidationError(f"Provider {class_name} is not properly implemented") from e
                    except Exception as e:
                        logger.error(f"Failed to create provider {class_name}: {str(e)}")
                        raise ValidationError(f"Failed to create provider {class_name}") from e
                return factory
            
            # Register the factory with a unique key
            registry.register(f"{provider_type}.{mode}", create_factory(module_path, class_name))
            
            # Log the registration
            logger.debug(f"Registered provider factory for {provider_type}.{mode}: {provider_path}")

# Register factory for the get_provider function
@registry.register('get_provider')
def get_provider(
    provider_type: str = None,
    async_mode: bool = None,
    enhanced: bool = None,
    **kwargs
) -> Union[FinanceDataProvider, AsyncFinanceDataProvider]:
    """
    Get a provider instance for accessing financial data.
    
    This factory function creates the appropriate provider instance based on
    the specified type and mode.
    
    Args:
        provider_type: Type of provider (yahoo, yahooquery, hybrid, optimized)
        async_mode: Whether to use asynchronous API
        enhanced: Whether to use enhanced async implementation
        **kwargs: Additional arguments to pass to provider constructor
        
    Returns:
        Provider instance
        
    Raises:
        ValidationError: When provider_type is invalid or provider is not available
    """
    # Use defaults if not specified
    provider_type = provider_type or DEFAULT_PROVIDER_TYPE
    async_mode = async_mode if async_mode is not None else DEFAULT_ASYNC_MODE
    enhanced = enhanced if enhanced is not None else DEFAULT_ENHANCED
    
    # Convert to lowercase for consistency
    provider_type = provider_type.lower()
    
    # Validate provider type
    if provider_type not in PROVIDER_TYPES:
        raise ValidationError(f"Unknown provider type: {provider_type}")
    
    # Determine which provider to use based on mode
    if async_mode:
        if enhanced and 'async_enhanced' in PROVIDER_TYPES[provider_type]:
            provider_key = f"{provider_type}.async_enhanced"
        elif 'async' in PROVIDER_TYPES[provider_type]:
            provider_key = f"{provider_type}.async"
        else:
            raise ValidationError(f"No async provider available for {provider_type}")
    else:
        if 'sync' in PROVIDER_TYPES[provider_type]:
            provider_key = f"{provider_type}.sync"
        else:
            raise ValidationError(f"No sync provider available for {provider_type}")
    
    # Create the provider instance
    try:
        return registry.resolve(provider_key, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create provider {provider_key}: {str(e)}")
        raise ValidationError(f"Failed to create provider {provider_key}") from e

# Register factory for getting all provider types
@registry.register('get_all_providers')
def get_all_providers(async_mode: bool = None, **kwargs) -> Dict[str, Union[FinanceDataProvider, AsyncFinanceDataProvider]]:
    """
    Get all available provider instances.
    
    This function creates instances of all registered provider types.
    
    Args:
        async_mode: Whether to use asynchronous API
        **kwargs: Additional arguments to pass to provider constructors
        
    Returns:
        Dictionary of provider instances keyed by provider type
    """
    # Use defaults if not specified
    async_mode = async_mode if async_mode is not None else DEFAULT_ASYNC_MODE
    
    # Create a provider instance for each type
    providers = {}
    for provider_type in PROVIDER_TYPES:
        try:
            providers[provider_type] = get_provider(provider_type, async_mode, **kwargs)
        except ValidationError:
            # Skip unavailable providers
            logger.warning(f"Provider {provider_type} is not available")
    
    return providers

# Register factory for the default provider
@registry.register('default_provider')
def get_default_provider(**kwargs) -> Union[FinanceDataProvider, AsyncFinanceDataProvider]:
    """
    Get the default provider instance.
    
    This function creates an instance of the default provider type.
    
    Args:
        **kwargs: Additional arguments to pass to provider constructor
        
    Returns:
        Default provider instance
    """
    return get_provider(DEFAULT_PROVIDER_TYPE, **kwargs)

# Initialize the registry
initialize_registry()
"""
Dependency Injection Container for Yahoo Finance.

This module sets up the dependency injection container for the application,
registering all components and their factories. It serves as the main
entry point for the dependency injection system.
"""

from typing import Dict, Any, Optional, Union, Callable

from .core.logging_config import get_logger
from .core.errors import YFinanceError, ValidationError
from .utils.dependency_injection import registry, inject, provides, lazy_import

# Set up logging
logger = get_logger(__name__)

# Import necessary factories and providers
from .api.provider_registry import get_provider, get_all_providers, get_default_provider
from .analysis.analyzer_factory import create_stock_analyzer, create_portfolio_analyzer
from .analysis.analyzer_factory import with_analyzer, with_portfolio_analyzer

# Define a function to register all application components
def setup_application():
    """
    Set up the application dependency injection container.
    
    This function registers all application components in the registry,
    ensuring that they can be resolved by other components.
    """
    logger.info("Setting up dependency injection container")
    
    # Register core services
    try:
        from .core.logging_config import get_logger as logger_factory
        registry.register('logger_factory', logger_factory)
        
        # Register a logger instance for direct injection
        app_logger = logger_factory('application')
        registry.register_instance('app_logger', app_logger)
        
        logger.debug("Registered core services")
    except Exception as e:
        logger.error(f"Failed to register core services: {str(e)}")
    
    # Register helper services
    try:
        # Register formatters and utilities
        from .utils.data.format_utils import format_number
        registry.register('format_number', format_number)
        
        from .utils.market.filter_utils import filter_tickers_by_criteria
        registry.register('filter_tickers', filter_tickers_by_criteria)
        
        # Register helpers for data processing and display
        from .utils.display_helpers import create_display
        registry.register('display_factory', create_display)
        
        logger.debug("Registered helper services")
    except Exception as e:
        logger.error(f"Failed to register helper services: {str(e)}")
    
    # Register data access services
    try:
        # Register cache services if available
        try:
            from .data.cache import get_cache, clear_cache
            registry.register('get_cache', get_cache)
            registry.register('clear_cache', clear_cache)
            logger.debug("Registered cache services")
        except ImportError:
            logger.warning("Cache services not available")
        
        # Register data loaders
        from .data.download import download_market_data
        registry.register('download_market_data', download_market_data)
        
        logger.debug("Registered data access services")
    except Exception as e:
        logger.error(f"Failed to register data access services: {str(e)}")
    
    # Register presentation services
    try:
        from .presentation.formatter import create_formatter
        registry.register('formatter_factory', create_formatter)
        
        from .presentation.console import ConsoleDisplay
        registry.register('console_display', ConsoleDisplay)
        
        logger.debug("Registered presentation services")
    except Exception as e:
        logger.error(f"Failed to register presentation services: {str(e)}")
    
    # Register factories for the main application components
    try:
        # These factories are already registered in their respective modules,
        # but we re-register them here for completeness
        registry.register('get_provider', get_provider)
        registry.register('get_all_providers', get_all_providers)
        registry.register('get_default_provider', get_default_provider)
        
        registry.register('get_analyzer', create_stock_analyzer)
        registry.register('get_portfolio_analyzer', create_portfolio_analyzer)
        
        logger.debug("Registered analysis factories")
    except Exception as e:
        logger.error(f"Failed to register analysis factories: {str(e)}")
    
    logger.info("Dependency injection container setup complete")


# Factory for creating display instances
@registry.register('create_display')
def create_display(output_format: str = 'console', **kwargs) -> Any:
    """
    Create a display instance based on the specified format.
    
    This factory function creates an appropriate display instance
    based on the output format.
    
    Args:
        output_format: Format of the display (console, html)
        **kwargs: Additional arguments to pass to the display constructor
        
    Returns:
        Display instance appropriate for the format
        
    Raises:
        ValidationError: When the format is invalid
    """
    try:
        if output_format == 'console':
            from .presentation.console import ConsoleDisplay
            return ConsoleDisplay(**kwargs)
        elif output_format == 'html':
            from .presentation.html import HTMLDisplay
            return HTMLDisplay(**kwargs)
        else:
            raise ValidationError(f"Invalid display format: {output_format}")
    except ImportError as e:
        logger.error(f"Failed to create display for format '{output_format}': {str(e)}")
        raise ValidationError(f"Display format '{output_format}' is not available") from e
    except Exception as e:
        logger.error(f"Failed to create display: {str(e)}")
        raise ValidationError(f"Failed to create display: {str(e)}") from e


# Initialize the dependency injection container
@inject('app_logger')
def initialize(app_logger=None):
    """
    Initialize the dependency injection container.
    
    This function initializes the dependency injection container
    by setting up all application components.
    
    Returns:
        True if initialization is successful, False otherwise
    """
    try:
        # Set up the application
        setup_application()
        
        # Log success
        if app_logger:
            app_logger.info("Dependency injection container initialized successfully")
            
        return True
    except Exception as e:
        # Log failure
        if app_logger:
            app_logger.error(f"Failed to initialize dependency injection container: {str(e)}")
        else:
            logger.error(f"Failed to initialize dependency injection container: {str(e)}")
        
        return False


# Auto-initialize when the module is imported
initialized = initialize()

# Export decorators for convenience
with_logger = inject('app_logger')
with_provider = inject('get_provider')
with_display = inject('create_display')
with_formatter = inject('formatter_factory')
with_cache = inject('get_cache')
"""
Dependency injection system for Yahoo Finance.

This module provides a centralized registry for providers and other dependencies,
allowing for cleaner, more testable code by decoupling component creation from usage.
"""

import importlib
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

from ..core.errors import ValidationError, YFinanceError
from ..core.logging import get_logger


# Set up logging
logger = get_logger(__name__)

# Type variables for generic types
T = TypeVar("T")  # Component type


class Registry:
    """
    Registry for dependency injection.

    This class manages the registration and resolution of dependencies,
    providing a centralized location for component creation.

    Attributes:
        _registry: Dictionary mapping component keys to factory functions
        _instances: Dictionary of singleton instances
    """

    def __init__(self):
        """Initialize the registry."""
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._instances: Dict[str, Any] = {}

    def register(
        self, key: str, factory: Optional[Callable[..., T]] = None
    ) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
        """
        Register a factory function for a component.

        This method can be used in two ways:
        1. As a decorator: @registry.register('key')
        2. Directly: registry.register('key', factory_func)

        Args:
            key: Key to register the factory under
            factory: Factory function to register (optional when used as decorator)

        Returns:
            If used directly: The registered factory function
            If used as decorator: A decorator function

        Examples:
            ```
            # As a decorator
            @registry.register('yahoo_provider')
            def create_yahoo_provider(**kwargs):
                return YahooFinanceProvider(**kwargs)

            # Direct registration
            def create_another_provider(**kwargs):
                return AnotherProvider(**kwargs)
            registry.register('another_provider', create_another_provider)
            ```
        """
        # Handle case when decorator is called without parentheses
        if callable(factory) and not isinstance(factory, type):
            func = factory
            self._registry[key] = func
            logger.debug(f"Registered factory for '{key}' via decorator (no args)")
            return func

        if factory is None:
            # Used as a decorator
            def decorator(func: Callable[..., T]) -> Callable[..., T]:
                self._registry[key] = func
                logger.debug(f"Registered factory for '{key}' via decorator")
                return func

            return decorator
        else:
            # Used directly
            self._registry[key] = factory
            logger.debug(f"Registered factory for '{key}' directly")
            return factory

    def register_instance(self, key: str, instance: T) -> T:
        """
        Register a singleton instance.

        Args:
            key: Key to register the instance under
            instance: The instance to register

        Returns:
            The registered instance

        Example:
            ```
            config = Config()
            registry.register_instance('config', config)
            ```
        """
        self._instances[key] = instance
        logger.debug(f"Registered singleton instance for '{key}'")
        return instance

    def resolve(self, key: str, **kwargs) -> Any:
        """
        Resolve a component from the registry.

        Args:
            key: Key to resolve
            **kwargs: Additional arguments to pass to the factory

        Returns:
            The resolved component

        Raises:
            ValidationError: If the key is not registered

        Example:
            ```
            provider = registry.resolve('yahoo_provider', async_mode=True)
            ```
        """
        # Check for singleton instance
        if key in self._instances:
            return self._instances[key]

        # Check for factory
        if key in self._registry:
            factory = self._registry[key]
            logger.debug(f"Resolving '{key}' with args: {kwargs}")

            # Call factory with provided kwargs
            try:
                return factory(**kwargs)
            except Exception as e:
                logger.error(f"Error resolving '{key}': {str(e)}")
                raise ValidationError(f"Failed to resolve component '{key}': {str(e)}") from e

        # Key not found
        raise ValidationError(f"No component registered for key '{key}'")

    def resolve_all(self, **kwargs) -> Dict[str, Any]:
        """
        Resolve all registered components.

        Args:
            **kwargs: Additional arguments to pass to the factories

        Returns:
            Dictionary of resolved components

        Example:
            ```
            all_providers = registry.resolve_all(async_mode=True)
            ```
        """
        result = {}
        errors = []

        # Resolve all singletons
        for key, instance in self._instances.items():
            result[key] = instance

        # Resolve all factories
        for key, factory in self._registry.items():
            if key not in result:  # Don't overwrite singletons
                try:
                    result[key] = factory(**kwargs)
                except Exception as e:
                    errors.append(f"Failed to resolve '{key}': {str(e)}")
                    logger.warning(f"Error resolving component '{key}': {str(e)}")

        # Log any errors
        if errors:
            logger.warning(f"Encountered {len(errors)} errors while resolving all components")

        return result

    def set_instance(self, key: str, instance: Any) -> None:
        """
        Set a singleton instance.

        Args:
            key: Key to set the instance for
            instance: The instance to set

        Example:
            ```
            mock_provider = MockProvider()
            registry.set_instance('yahoo_provider', mock_provider)
            ```
        """
        self._instances[key] = instance
        logger.debug(f"Set singleton instance for '{key}'")

    def clear_instances(self) -> None:
        """
        Clear all singleton instances.

        This is useful for testing or when you need to reset the registry.

        Example:
            ```
            # In test teardown
            registry.clear_instances()
            ```
        """
        self._instances.clear()
        logger.debug("Cleared all singleton instances")

    def reset(self) -> None:
        """
        Reset the entire registry.

        This clears all registrations and instances.

        Example:
            ```
            # In test setup
            registry.reset()
            ```
        """
        self._registry.clear()
        self._instances.clear()
        logger.debug("Reset registry (cleared all registrations and instances)")

    def get_keys(self) -> List[str]:
        """
        Get all registered keys.

        Returns:
            List of registered keys

        Example:
            ```
            all_keys = registry.get_keys()
            ```
        """
        # Combine keys from registry and instances, removing duplicates
        return list(set(list(self._registry.keys()) + list(self._instances.keys())))

    def has_key(self, key: str) -> bool:
        """
        Check if a key is registered.

        Args:
            key: Key to check

        Returns:
            True if the key is registered, False otherwise

        Example:
            ```
            if registry.has_key('yahoo_provider'):
                print("Yahoo provider is registered")
            ```
        """
        return key in self._registry or key in self._instances


# Create a global registry instance
registry = Registry()


def inject(component_key: str, **resolve_kwargs):
    """
    Decorator to inject dependencies into a function.

    This decorator adds a parameter to the function that is resolved
    from the registry.

    Args:
        component_key: Key to resolve from the registry
        **resolve_kwargs: Additional arguments to pass to resolve()

    Returns:
        Decorated function

    Example:
        ```
        @inject('yahoo_provider', async_mode=True)
        def analyze_stock(ticker, provider=None):
            return provider.get_ticker_info(ticker)
        ```
    """
    # Handle case when decorator is called without arguments
    if callable(component_key) and not isinstance(component_key, str):
        func = component_key
        # Use function name as component key
        key = func.__name__

        @wraps(func)
        def direct_wrapper(*args, **kwargs):
            # Check if function name is in registry
            try:
                if registry.has_key(key):
                    component = registry.resolve(key)
                    # Add component to kwargs with function name as parameter name
                    kwargs[key] = component
            except Exception as e:
                logger.warning(f"Failed to inject component '{key}': {str(e)}")

            # Call the original function
            return func(*args, **kwargs)

        return direct_wrapper

    def decorator(func):
        # Get parameter name from component key or function signature
        param_name = component_key.split(".")[-1]

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only inject if the parameter is not provided or is None
            if param_name not in kwargs or kwargs[param_name] is None:
                try:
                    # Resolve the dependency and add it to kwargs
                    kwargs[param_name] = registry.resolve(component_key, **resolve_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to inject component '{component_key}': {str(e)}")
                    # Continue without the component if it can't be resolved

            # Call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def provides(component_key: str):
    """
    Decorator to register a function's return value as a component.

    This decorator registers the return value of the function
    as a singleton instance in the registry.

    Args:
        component_key: Key to register the instance under

    Returns:
        Decorated function

    Example:
        ```
        @provides('config')
        def create_config():
            return Config()
        ```
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function to get the instance
            instance = func(*args, **kwargs)

            # Register the instance
            registry.register_instance(component_key, instance)

            # Return the instance
            return instance

        return wrapper

    return decorator


def lazy_import(module_path: str, class_name: Optional[str] = None) -> Any:
    """
    Lazily import a module or class.

    This function allows for lazy importing of modules to avoid circular imports.

    Args:
        module_path: Path to the module to import
        class_name: Optional class name to import from the module

    Returns:
        The imported module or class

    Example:
        ```
        YahooFinanceProvider = lazy_import('yahoofinance.api.providers.yahoo_finance', 'YahooFinanceProvider')
        ```
    """
    try:
        module = importlib.import_module(module_path)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError):
        # Return a function that will attempt the import when called
        def _lazy_import():
            try:
                module = importlib.import_module(module_path)
                if class_name:
                    return getattr(module, class_name)
                return module
            except Exception as ex:
                logger.error(
                    f"Lazy import failed: {module_path}.{class_name if class_name else ''} - {str(ex)}"
                )
                raise ImportError(
                    f"Failed to import {module_path}.{class_name if class_name else ''}"
                ) from ex

        return _lazy_import

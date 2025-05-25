"""
Utility functions for handling imports and resolving circular dependencies.

This module provides utilities for lazy imports, dependency injection,
and other techniques to help resolve circular dependencies.
"""

import importlib
import inspect
import sys
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from ..core.errors import ValidationError, YFinanceError
from ..core.logging import get_logger
from .error_handling import enrich_error_context, safe_operation, translate_error, with_retry


# Set up logging
logger = get_logger(__name__)

# Type variables for type annotation
T = TypeVar("T")
R = TypeVar("R")


class LazyImport:
    """
    Lazily import a module or object from a module.

    This class allows for lazy imports, which can help resolve
    circular dependencies by deferring the import until the
    object is actually needed.

    Examples:
        ```
        # Lazy import a module
        np = LazyImport('numpy')

        # Lazy import an object from a module
        DataFrame = LazyImport('pandas', 'DataFrame')

        # Later, when actually needed:
        df = DataFrame(data)  # This will trigger the import
        ```
    """

    def __init__(self, module_name: str, object_name: Optional[str] = None):
        """
        Initialize a lazy import.

        Args:
            module_name: Name of the module to import
            object_name: Optional name of an object to import from the module
        """
        self._module_name = module_name
        self._object_name = object_name
        self._module = None
        self._object = None

    def __call__(self, *args, **kwargs):
        """
        Allow the lazy import to be called as a function.

        This triggers the import and calls the imported object
        with the provided arguments.

        Args:
            *args: Positional arguments to pass to the called object
            **kwargs: Keyword arguments to pass to the called object

        Returns:
            Result of calling the imported object

        Raises:
            TypeError: If the imported object is not callable
        """
        obj = self._resolve()
        if not callable(obj):
            raise TypeError(f"'{self._module_name}.{self._object_name}' is not callable")
        return obj(*args, **kwargs)

    @with_retry
    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute from the imported object.

        This triggers the import and returns the requested attribute.

        Args:
            name: Name of the attribute to get

        Returns:
            The requested attribute

        Raises:
            AttributeError: If the attribute doesn't exist
        """
        return getattr(self._resolve(), name)

    def __getitem__(self, key: Any) -> Any:
        """
        Get an item from the imported object.

        This triggers the import and returns the requested item.

        Args:
            key: Key to use for indexing

        Returns:
            The requested item

        Raises:
            TypeError: If the imported object doesn't support indexing
        """
        return self._resolve()[key]

    def _resolve(self) -> Any:
        """
        Resolve the import, loading the module and object as needed.

        Returns:
            The imported module or object
        """
        if self._object is not None:
            return self._object

        if self._module is None:
            self._module = importlib.import_module(self._module_name)

        if self._object_name is not None:
            self._object = getattr(self._module, self._object_name)
            return self._object

        return self._module


class DependencyProvider:
    """
    Provide dependencies to functions that need them.

    This class implements a basic dependency injection pattern
    to help break circular dependencies by providing dependencies
    at runtime rather than import time.

    Examples:
        ```
        # Create a dependency provider
        dependencies = DependencyProvider()

        # Register dependencies
        dependencies.register('config', config_instance)
        dependencies.register_factory('database', create_database)

        # Use dependencies in a function
        @dependencies.inject('config', 'database')
        def process_data(data, config, database):
            # Use config and database
        ```
    """

    def __init__(self):
        """Initialize an empty dependency provider."""
        self._dependencies: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}

    def register(self, name: str, dependency: Any) -> None:
        """
        Register a dependency.

        Args:
            name: Name of the dependency
            dependency: The dependency object
        """
        self._dependencies[name] = dependency

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """
        Register a factory function for a dependency.

        The factory function will be called the first time
        the dependency is requested.

        Args:
            name: Name of the dependency
            factory: Factory function that creates the dependency
        """
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """
        Get a dependency by name.

        Args:
            name: Name of the dependency

        Returns:
            The dependency object

        Raises:
            KeyError: If the dependency is not registered
        """
        if name in self._dependencies:
            return self._dependencies[name]

        if name in self._factories:
            # Lazily create the dependency and cache it
            dependency = self._factories[name]()
            self._dependencies[name] = dependency
            return dependency

        raise KeyError(f"Dependency '{name}' not registered")

    def inject(self, *dependency_names: str) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """
        Decorator to inject dependencies into a function.

        Args:
            *dependency_names: Names of dependencies to inject

        Returns:
            Decorator function

        Example:
            ```
            @dependencies.inject('config', 'database')
            def process_data(data, config, database):
                # Use config and database
            ```
        """

        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            # Use inspect to get function signature
            sig = inspect.signature(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create bound arguments
                bound_args = sig.bind_partial(*args, **kwargs)

                # Add dependencies as keyword arguments if not already provided
                for name in dependency_names:
                    if name not in bound_args.arguments:
                        kwargs[name] = self.get(name)

                return func(*args, **kwargs)

            return wrapper

        return decorator


# Create a global dependency provider instance
dependencies = DependencyProvider()


def import_module_or_object(module_path: str, object_name: Optional[str] = None) -> Any:
    """
    Import a module or object from a module.

    This function provides a way to dynamically import
    modules or objects from modules at runtime.

    Args:
        module_path: Path to the module
        object_name: Optional name of an object to import from the module

    Returns:
        The imported module or object

    Raises:
        ImportError: If the module or object cannot be imported
    """
    try:
        module = importlib.import_module(module_path)
        if object_name is None:
            return module
        return getattr(module, object_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {object_name or module_path}: {str(e)}") from e


def local_import(name: str) -> Any:
    """
    Import an object from within the current package.

    This function makes it easier to import objects from
    within the current package, which can help with
    relative imports.

    Args:
        name: Relative import path within the yahoofinance package

    Returns:
        The imported object

    Example:
        ```
        # In yahoofinance/utils/some_module.py
        Config = local_import('core.config.Config')
        ```
    """
    # Get the caller's module
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Could not get current frame")

    try:
        caller_module = frame.f_back.f_globals["__name__"]
    finally:
        del frame  # Avoid reference cycles

    # Split the caller's module to get the package
    package_parts = caller_module.split(".")
    if package_parts[0] != "yahoofinance":
        raise ImportError(
            f"local_import can only be used within yahoofinance package, not {caller_module}"
        )

    # Determine the full import path
    full_path = f"yahoofinance.{name}"

    # Import and return the object
    module_path, _, object_name = full_path.rpartition(".")
    return import_module_or_object(module_path, object_name)


@with_retry
def delayed_import(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator to delay imports until function execution.

    This decorator allows for imports to be defined inside
    a function body, which can help resolve circular dependencies
    by delaying the import until the function is actually called.

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    Example:
        ```
        @delayed_import
        def get_provider():
            from yahoofinance.api.providers import YahooFinanceProvider
            return YahooFinanceProvider()
        ```
    """
    return func

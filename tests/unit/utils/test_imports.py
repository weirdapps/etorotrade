"""
Unit tests for the import utilities.

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation

These tests verify the behavior of the import utilities including
lazy imports, dependency injection, and other techniques to resolve
circular dependencies.
"""

from unittest.mock import Mock, call, patch

import pytest

from yahoofinance.utils.imports import (
    DependencyProvider,
    LazyImport,
    delayed_import,
    dependencies,
    import_module_or_object,
)


class TestLazyImport:
    """Tests for the LazyImport class."""

    def test_lazy_module_import(self):
        """Test lazy import of a module."""
        # Create a lazy import for a standard library module
        time = LazyImport("time")

        # Verify it's a LazyImport instance
        assert isinstance(time, LazyImport)

        # Access an attribute to trigger the import
        sleep_func = time.sleep

        # Verify the attribute is a function
        assert callable(sleep_func)

    def test_lazy_object_import(self):
        """Test lazy import of an object from a module."""
        # Create a lazy import for a function from a standard library module
        sleep = LazyImport("time", "sleep")

        # Verify it's a LazyImport instance
        assert isinstance(sleep, LazyImport)

        # Call the function to trigger the import
        try:
            sleep(0.001)
        except YFinanceError as e:
            pytest.fail(f"LazyImport object should be callable: {str(e)}")

    def test_lazy_import_caching(self):
        """Test that lazy imports are cached after first resolution."""
        # Create a lazy import
        time = LazyImport("time")

        # Access module to trigger import
        time._resolve()

        # Mock importlib.import_module to verify it's not called again
        with patch("importlib.import_module") as mock_import:
            # Access module again
            time._resolve()

            # Verify importlib.import_module was not called
            mock_import.assert_not_called()

    def test_lazy_import_attribute_access(self):
        """Test accessing attributes of a lazy import."""
        # Create a lazy import for os.path
        path = LazyImport("os.path")

        # Access join method
        join = path.join

        # Verify join is callable
        assert callable(join)

        # Call join with safe test paths (avoid publicly writable directories)
        result = join("home", "user", "file.txt")

        # Verify result
        assert result == "home/user/file.txt" or result == "home\\user\\file.txt"  # Handle both Unix and Windows

    def test_lazy_import_item_access(self):
        """Test accessing items of a lazy import."""
        # Create a lazy import for a dictionary-like object
        import sys

        modules = LazyImport("sys", "modules")

        # Access an item
        os_module = modules["os"]

        # Verify the item is the os module
        assert os_module.__name__ == "os"

    def test_lazy_import_error_handling(self):
        """Test error handling in lazy imports."""
        # Create a lazy import for a non-existent module
        non_existent = LazyImport("this_module_does_not_exist")

        # Attempt to access an attribute, should raise ImportError
        with pytest.raises(ImportError):
            non_existent.some_attribute

        # Create a lazy import for a non-existent attribute
        missing_attr = LazyImport("os", "this_attribute_does_not_exist")

        # Attempt to access the attribute, should raise AttributeError
        with pytest.raises(AttributeError):
            missing_attr()


class TestDependencyProvider:
    """Tests for the DependencyProvider class."""

    def test_register_and_get_dependency(self):
        """Test registering and getting a dependency."""
        # Create a dependency provider
        provider = DependencyProvider()

        # Create a test dependency
        test_dep = {"key": "value"}

        # Register the dependency
        provider.register("test_dep", test_dep)

        # Get the dependency
        result = provider.get("test_dep")

        # Verify the result
        assert result is test_dep

    def test_register_factory(self):
        """Test registering a factory function for a dependency."""
        # Create a dependency provider
        provider = DependencyProvider()

        # Create a factory function
        factory_mock = Mock(return_value={"key": "value"})

        # Register the factory
        provider.register_factory("factory_dep", factory_mock)

        # Get the dependency
        result = provider.get("factory_dep")

        # Verify the result
        assert result == {"key": "value"}

        # Verify the factory was called exactly once
        factory_mock.assert_called_once()

        # Get the dependency again
        result2 = provider.get("factory_dep")

        # Verify the factory was not called again
        assert factory_mock.call_count == 1

        # Verify the same instance was returned
        assert result2 is result

    def test_inject_decorator(self):
        """Test the inject decorator."""
        # Create a dependency provider
        provider = DependencyProvider()

        # Register dependencies
        provider.register("config", {"debug": True})
        provider.register("logger", Mock())

        # Define a function with the inject decorator
        @provider.inject("config", "logger")
        def process_data(data, config, logger):
            logger.info(f"Processing data with config: {config}")
            return data + config["debug"]

        # Call the function
        result = process_data(False)

        # Verify the result (False + True = 1, which is truthy but not identical to True)
        assert result == True  # Use equality instead of identity

        # Verify the logger was called with the expected arguments
        provider.get("logger").info.assert_called_once_with(
            "Processing data with config: {'debug': True}"
        )

    def test_inject_with_explicit_args(self):
        """Test the inject decorator with explicitly provided args."""
        # Create a dependency provider
        provider = DependencyProvider()

        # Register dependencies
        provider.register("config", {"debug": True})

        # Define a function with the inject decorator
        @provider.inject("config")
        def get_debug(config=None):
            return config["debug"]

        # Call the function with explicit config
        explicit_config = {"debug": False}
        result = get_debug(config=explicit_config)

        # Verify explicit config was used
        assert result is False

    def test_dependency_not_found(self):
        """Test behavior when a dependency is not found."""
        # Create a dependency provider
        provider = DependencyProvider()

        # Attempt to get a non-existent dependency
        with pytest.raises(KeyError) as excinfo:
            provider.get("non_existent")

        # Verify the error message
        assert "Dependency 'non_existent' not registered" in str(excinfo.value)


class TestImportModuleOrObject:
    """Tests for the import_module_or_object function."""

    def test_import_module(self):
        """Test importing a module."""
        # Import a standard library module
        os = import_module_or_object("os")

        # Verify it's the correct module
        assert os.__name__ == "os"

    def test_import_object(self):
        """Test importing an object from a module."""
        # Import a function from a standard library module
        sleep = import_module_or_object("time", "sleep")

        # Verify it's the correct function
        assert callable(sleep)
        # Check function directly from time module for comparison
        import time as time_module

        assert sleep == time_module.sleep

    def test_import_error_handling(self):
        """Test error handling for imports."""
        # Attempt to import a non-existent module
        with pytest.raises(ImportError):
            import_module_or_object("this_module_does_not_exist")

        # Attempt to import a non-existent attribute
        with pytest.raises(ImportError):
            import_module_or_object("os", "this_attribute_does_not_exist")


class TestDelayedImport:
    """Tests for the delayed_import decorator."""

    def test_delayed_import_decorator(self):
        """Test the delayed_import decorator."""

        # Define a function that imports a module
        @delayed_import
        def get_time_module():
            import time

            return time

        # Mock importlib.import_module to verify import timing
        with patch("importlib.import_module") as mock_import:
            # Define the function (should not trigger import)
            mock_import.assert_not_called()

            # Call the function (should trigger import)
            result = get_time_module()

            # Verify the function ran without errors
            assert result.__name__ == "time"

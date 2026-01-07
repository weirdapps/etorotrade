"""
Tests for yahoofinance/utils/error_handling.py

This module tests error handling utilities.
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from yahoofinance.utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)
from yahoofinance.core.errors import (
    YFinanceError,
    APIError,
    DataError,
    ValidationError,
    ConnectionError as YFConnectionError,
    TimeoutError as YFTimeoutError,
)


class TestEnrichErrorContext:
    """Tests for enrich_error_context function."""

    def test_enrich_adds_context(self):
        """Test that error context is enriched."""
        original = YFinanceError("Original error")
        enriched = enrich_error_context(original, {"ticker": "AAPL", "operation": "fetch"})
        assert enriched.details is not None
        assert enriched.details.get("ticker") == "AAPL"

    def test_enrich_preserves_original(self):
        """Test that original error is preserved."""
        original = YFinanceError("Original message")
        enriched = enrich_error_context(original, {"extra": "data"})
        assert "Original message" in str(enriched)

    def test_enrich_does_not_overwrite(self):
        """Test that existing keys are not overwritten."""
        original = YFinanceError("Error", {"ticker": "AAPL"})
        enriched = enrich_error_context(original, {"ticker": "MSFT", "new_key": "value"})
        assert enriched.details["ticker"] == "AAPL"  # Not overwritten
        assert enriched.details["new_key"] == "value"  # Added


class TestTranslateError:
    """Tests for translate_error function."""

    def test_translate_value_error(self):
        """Test translating ValueError to ValidationError."""
        original = ValueError("Invalid value")
        translated = translate_error(original)
        assert isinstance(translated, ValidationError)

    def test_translate_key_error(self):
        """Test translating KeyError to DataError."""
        original = KeyError("missing_key")
        translated = translate_error(original)
        assert isinstance(translated, DataError)
        assert "missing_key" in str(translated)

    def test_translate_attribute_error(self):
        """Test translating AttributeError to DataError."""
        original = AttributeError("no attribute")
        translated = translate_error(original)
        assert isinstance(translated, DataError)

    def test_translate_with_context(self):
        """Test translating with context."""
        original = ValueError("Error")
        translated = translate_error(original, context={"ticker": "AAPL"})
        assert translated.details.get("ticker") == "AAPL"

    def test_translate_unknown_error(self):
        """Test translating unknown error type."""
        original = RuntimeError("Unknown")
        translated = translate_error(original)
        assert isinstance(translated, YFinanceError)


class TestSafeOperation:
    """Tests for safe_operation decorator."""

    def test_safe_operation_success(self):
        """Test safe_operation with successful function."""
        @safe_operation(default_value="default")
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_safe_operation_with_yfinance_error(self):
        """Test safe_operation returns default on YFinanceError."""
        @safe_operation(default_value="default", log_errors=False)
        def failing_func():
            raise YFinanceError("Error")

        result = failing_func()
        assert result == "default"

    def test_safe_operation_none_default(self):
        """Test safe_operation with None default on YFinanceError."""
        @safe_operation(default_value=None, log_errors=False)
        def failing_func():
            raise DataError("Data error")

        result = failing_func()
        assert result is None

    def test_safe_operation_with_args(self):
        """Test safe_operation with function arguments."""
        @safe_operation(default_value=0)
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)
        assert result == 8

    def test_safe_operation_reraise(self):
        """Test safe_operation with reraise=True."""
        @safe_operation(default_value="default", reraise=True, log_errors=False)
        def failing_func():
            raise APIError("API Error")

        with pytest.raises(APIError):
            failing_func()

    def test_safe_operation_non_yfinance_error_propagates(self):
        """Test that non-YFinanceError exceptions propagate."""
        @safe_operation(default_value="default", log_errors=False)
        def failing_func():
            raise ValueError("Not a YFinanceError")

        # ValueError is not caught by safe_operation
        with pytest.raises(ValueError):
            failing_func()


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_with_retry_success_first_try(self):
        """Test with_retry when function succeeds on first try."""
        call_count = 0

        @with_retry(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_with_retry_success_after_connection_errors(self):
        """Test with_retry when function succeeds after ConnectionError."""
        call_count = 0

        @with_retry(max_retries=3, retry_delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise YFConnectionError("Connection failed")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_with_retry_all_failures_timeout(self):
        """Test with_retry when all attempts fail with TimeoutError."""
        call_count = 0

        @with_retry(max_retries=2, retry_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise YFTimeoutError("Always times out")

        with pytest.raises(YFTimeoutError):
            always_fails()
        # Initial try + 2 retries = 3 calls
        assert call_count == 3

    def test_with_retry_backoff_factor(self):
        """Test that backoff factor works with ConnectionError."""
        call_count = 0

        @with_retry(max_retries=2, retry_delay=0.01, backoff_factor=2)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise YFConnectionError("Connection fail")

        with pytest.raises(YFConnectionError):
            always_fails()

        # Should have made initial + retries attempts
        assert call_count == 3

    def test_with_retry_non_retryable_error_not_retried(self):
        """Test that non-retryable errors are not retried."""
        call_count = 0

        @with_retry(max_retries=3, retry_delay=0.01)
        def fails_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            fails_with_value_error()
        # Should only be called once - ValueError is not retryable
        assert call_count == 1


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_translate_then_enrich(self):
        """Test translating then enriching error."""
        original = ValueError("Base error")
        translated = translate_error(original, context={"ticker": "AAPL"})
        enriched = enrich_error_context(translated, {"operation": "fetch"})

        assert enriched.details.get("ticker") == "AAPL"
        assert enriched.details.get("operation") == "fetch"

    def test_safe_and_translate(self):
        """Test combining safe_operation with error translation."""
        @safe_operation(default_value=None, log_errors=False)
        def might_fail():
            error = ValueError("Error")
            raise translate_error(error)

        result = might_fail()
        assert result is None


class TestUserFriendlyErrors:
    """Tests for user-friendly error functions."""

    def test_handle_file_not_found_portfolio(self):
        """Test handling portfolio file not found."""
        from yahoofinance.utils.error_handling import handle_file_not_found
        result = handle_file_not_found("/path/to/portfolio.csv")
        assert "Portfolio file not found" in result
        assert "Export your eToro portfolio" in result

    def test_handle_file_not_found_generic(self):
        """Test handling generic file not found."""
        from yahoofinance.utils.error_handling import handle_file_not_found
        result = handle_file_not_found("/path/to/other.txt")
        assert "File not found" in result
        assert "Check the file path" in result

    def test_handle_csv_error(self):
        """Test handling CSV error."""
        from yahoofinance.utils.error_handling import handle_csv_error
        result = handle_csv_error("/path/to/data.csv", "Invalid delimiter")
        assert "Error reading CSV" in result
        assert "Invalid delimiter" in result
        assert "valid CSV" in result

    def test_handle_api_error(self):
        """Test handling API error."""
        from yahoofinance.utils.error_handling import handle_api_error
        result = handle_api_error("Yahoo Finance", "Rate limit exceeded")
        assert "API error" in result
        assert "Rate limit exceeded" in result
        assert "internet connection" in result

    def test_format_user_error_file_not_found(self):
        """Test formatting FileNotFoundError."""
        from yahoofinance.utils.error_handling import format_user_error
        error = FileNotFoundError("portfolio.csv")
        result = format_user_error(error)
        assert "Portfolio file not found" in result

    def test_format_user_error_connection(self):
        """Test formatting ConnectionError."""
        from yahoofinance.utils.error_handling import format_user_error
        # Create a fake ConnectionError
        class FakeConnectionError(Exception):
            pass
        FakeConnectionError.__name__ = "ConnectionError"
        error = FakeConnectionError("Connection refused")
        result = format_user_error(error)
        assert "API error" in result

    def test_format_user_error_yfinance_error(self):
        """Test formatting YFinanceError."""
        from yahoofinance.utils.error_handling import format_user_error
        error = YFinanceError("Custom error message")
        # NOSONAR: S2259 - format_user_error() never returns None (always returns str)
        maybe_result = format_user_error(error)  # NOSONAR
        # Guard clause with explicit type narrowing
        if maybe_result is None:
            raise AssertionError("format_user_error() returned None")
        # New variable binding after guard - SonarCloud should see this as safe
        result: str = maybe_result
        # Verify content
        assert "Custom error message" in result
        assert "Check logs" in result

    def test_format_user_error_unknown(self):
        """Test formatting unknown error."""
        from yahoofinance.utils.error_handling import format_user_error
        error = RuntimeError("Unknown error type")
        result = format_user_error(error)
        assert "Unexpected error" in result
        assert "Unknown error type" in result


class TestWithRetryDecorator:
    """Additional tests for with_retry decorator."""

    def test_with_retry_without_parentheses(self):
        """Test with_retry can be used without parentheses."""
        call_count = 0

        @with_retry
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_with_retry_yfinance_error_not_retried(self):
        """Test that generic YFinanceError is not retried."""
        call_count = 0

        @with_retry(max_retries=3, retry_delay=0.01)
        def fails_with_yfinance_error():
            nonlocal call_count
            call_count += 1
            raise YFinanceError("Generic error")

        with pytest.raises(YFinanceError):
            fails_with_yfinance_error()
        # Should only be called once - generic YFinanceError is not retryable
        assert call_count == 1


class TestTranslateErrorAdditional:
    """Additional tests for translate_error edge cases."""

    def test_translate_file_not_found(self):
        """Test translating FileNotFoundError."""
        from yahoofinance.core.errors import ResourceNotFoundError
        original = FileNotFoundError("config.yaml")
        translated = translate_error(original)
        assert isinstance(translated, ResourceNotFoundError)
        assert "File not found" in str(translated)

    def test_translate_permission_error(self):
        """Test translating PermissionError."""
        from yahoofinance.core.errors import ResourceNotFoundError
        original = PermissionError("Access denied")
        translated = translate_error(original)
        assert isinstance(translated, ResourceNotFoundError)
        assert "Permission denied" in str(translated)

    def test_translate_memory_error(self):
        """Test translating MemoryError."""
        from yahoofinance.core.errors import ResourceNotFoundError
        original = MemoryError("Out of memory")
        translated = translate_error(original)
        assert isinstance(translated, ResourceNotFoundError)
        assert "Out of memory" in str(translated)

    def test_translate_with_default_message(self):
        """Test translating with default message."""
        class EmptyError(Exception):
            def __str__(self):
                return ""
        original = EmptyError()
        translated = translate_error(original, default_message="Default error")
        assert "Default error" in str(translated)

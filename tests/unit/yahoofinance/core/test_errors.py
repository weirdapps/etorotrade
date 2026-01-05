#!/usr/bin/env python3
"""
ITERATION 34: Errors Tests
Target: Test error handling and exception hierarchy
File: yahoofinance/core/errors.py (62 statements, 97% coverage)
"""

import pytest


class TestYFinanceError:
    """Test YFinanceError base class."""

    def test_create_simple_error(self):
        """Create error with message only."""
        from yahoofinance.core.errors import YFinanceError

        error = YFinanceError("Test error")

        assert error.message == "Test error"
        assert error.details == {}
        assert str(error) == "Test error"

    def test_create_error_with_details(self):
        """Create error with details."""
        from yahoofinance.core.errors import YFinanceError

        details = {"ticker": "AAPL", "reason": "test"}
        error = YFinanceError("Test error", details)

        assert error.message == "Test error"
        assert error.details == details
        assert "ticker=AAPL" in str(error)
        assert "reason=test" in str(error)

    def test_error_inherits_from_exception(self):
        """YFinanceError inherits from Exception."""
        from yahoofinance.core.errors import YFinanceError

        error = YFinanceError("Test")

        assert isinstance(error, Exception)


class TestAPIError:
    """Test APIError class."""

    def test_api_error_inherits_from_yfinance_error(self):
        """APIError inherits from YFinanceError."""
        from yahoofinance.core.errors import APIError, YFinanceError

        error = APIError("API failed")

        assert isinstance(error, YFinanceError)
        assert isinstance(error, Exception)

    def test_api_error_with_details(self):
        """Create APIError with details."""
        from yahoofinance.core.errors import APIError

        details = {"status_code": 500}
        error = APIError("Server error", details)

        assert error.message == "Server error"
        assert error.details == details


class TestValidationError:
    """Test ValidationError class."""

    def test_validation_error(self):
        """Create ValidationError."""
        from yahoofinance.core.errors import ValidationError, YFinanceError

        error = ValidationError("Invalid ticker")

        assert isinstance(error, YFinanceError)
        assert error.message == "Invalid ticker"


class TestRateLimitError:
    """Test RateLimitError class."""

    def test_rate_limit_without_retry_after(self):
        """Create RateLimitError without retry_after."""
        from yahoofinance.core.errors import RateLimitError

        error = RateLimitError("Rate limited")

        assert error.message == "Rate limited"
        assert error.retry_after is None
        assert str(error) == "Rate limited"

    def test_rate_limit_with_retry_after(self):
        """Create RateLimitError with retry_after."""
        from yahoofinance.core.errors import RateLimitError

        error = RateLimitError("Rate limited", retry_after=60.0)

        assert error.message == "Rate limited"
        assert error.retry_after == pytest.approx(60.0)
        assert "retry after 60.0 seconds" in str(error)

    def test_rate_limit_with_details_and_retry(self):
        """Create RateLimitError with both details and retry_after."""
        from yahoofinance.core.errors import RateLimitError

        details = {"endpoint": "/api/data"}
        error = RateLimitError("Rate limited", retry_after=30.0, details=details)

        assert error.retry_after == pytest.approx(30.0)
        assert error.details == details
        assert "endpoint=/api/data" in str(error)
        assert "retry after 30.0 seconds" in str(error)

    def test_rate_limit_inherits_from_api_error(self):
        """RateLimitError inherits from APIError."""
        from yahoofinance.core.errors import RateLimitError, APIError

        error = RateLimitError("Rate limited")

        assert isinstance(error, APIError)


class TestNetworkErrors:
    """Test network error classes."""

    def test_network_error(self):
        """Create NetworkError."""
        from yahoofinance.core.errors import NetworkError, APIError

        error = NetworkError("Network failed")

        assert isinstance(error, APIError)
        assert error.message == "Network failed"

    def test_connection_error(self):
        """Create ConnectionError."""
        from yahoofinance.core.errors import ConnectionError, NetworkError

        error = ConnectionError("Connection lost")

        assert isinstance(error, NetworkError)

    def test_timeout_error(self):
        """Create TimeoutError."""
        from yahoofinance.core.errors import TimeoutError, NetworkError

        error = TimeoutError("Request timed out")

        assert isinstance(error, NetworkError)


class TestResourceNotFoundError:
    """Test ResourceNotFoundError class."""

    def test_resource_not_found(self):
        """Create ResourceNotFoundError."""
        from yahoofinance.core.errors import ResourceNotFoundError, APIError

        error = ResourceNotFoundError("Not found")

        assert isinstance(error, APIError)


class TestDataErrors:
    """Test data error classes."""

    def test_data_error(self):
        """Create DataError."""
        from yahoofinance.core.errors import DataError, YFinanceError

        error = DataError("Data processing failed")

        assert isinstance(error, YFinanceError)

    def test_data_quality_error(self):
        """Create DataQualityError."""
        from yahoofinance.core.errors import DataQualityError, DataError

        error = DataQualityError("Poor data quality")

        assert isinstance(error, DataError)

    def test_missing_data_error(self):
        """Create MissingDataError."""
        from yahoofinance.core.errors import MissingDataError, DataError

        error = MissingDataError("Data not available")

        assert isinstance(error, DataError)


class TestOtherErrors:
    """Test other error classes."""

    def test_cache_error(self):
        """Create CacheError."""
        from yahoofinance.core.errors import CacheError, YFinanceError

        error = CacheError("Cache operation failed")

        assert isinstance(error, YFinanceError)

    def test_config_error(self):
        """Create ConfigError."""
        from yahoofinance.core.errors import ConfigError, YFinanceError

        error = ConfigError("Invalid configuration")

        assert isinstance(error, YFinanceError)

    def test_monitoring_error(self):
        """Create MonitoringError."""
        from yahoofinance.core.errors import MonitoringError, YFinanceError

        error = MonitoringError("Monitoring failed")

        assert isinstance(error, YFinanceError)


class TestFormatErrorDetails:
    """Test format_error_details function."""

    def test_format_yfinance_error_no_details(self):
        """Format YFinanceError without details."""
        from yahoofinance.core.errors import YFinanceError, format_error_details

        error = YFinanceError("Test error")

        result = format_error_details(error)

        assert result == "YFinanceError: Test error"

    def test_format_yfinance_error_with_details(self):
        """Format YFinanceError with details."""
        from yahoofinance.core.errors import YFinanceError, format_error_details

        details = {"ticker": "AAPL"}
        error = YFinanceError("Test error", details)

        result = format_error_details(error)

        assert "YFinanceError: Test error" in result
        assert "ticker=AAPL" in result

    def test_format_standard_exception(self):
        """Format standard Python exception."""
        from yahoofinance.core.errors import format_error_details

        error = ValueError("Invalid value")

        result = format_error_details(error)

        assert result == "ValueError: Invalid value"

    def test_format_api_error(self):
        """Format APIError."""
        from yahoofinance.core.errors import APIError, format_error_details

        details = {"status_code": 500}
        error = APIError("Server error", details)

        result = format_error_details(error)

        assert "APIError: Server error" in result
        assert "status_code=500" in result


class TestClassifyApiError:
    """Test classify_api_error function."""

    def test_classify_404_error(self):
        """Classify 404 as ResourceNotFoundError."""
        from yahoofinance.core.errors import classify_api_error, ResourceNotFoundError

        result = classify_api_error(404, "Not found")

        assert isinstance(result, ResourceNotFoundError)
        assert "404" in result.message
        assert result.details["status_code"] == 404

    def test_classify_429_error(self):
        """Classify 429 as RateLimitError."""
        from yahoofinance.core.errors import classify_api_error, RateLimitError

        result = classify_api_error(429, "Too many requests")

        assert isinstance(result, RateLimitError)
        assert "429" in result.message

    def test_classify_500_error(self):
        """Classify 500 as server error."""
        from yahoofinance.core.errors import classify_api_error, APIError

        result = classify_api_error(500, "Internal server error")

        assert isinstance(result, APIError)
        assert "Server error" in result.message
        assert result.details["status_code"] == 500

    def test_classify_400_error(self):
        """Classify 400 as client error."""
        from yahoofinance.core.errors import classify_api_error, APIError

        result = classify_api_error(400, "Bad request")

        assert isinstance(result, APIError)
        assert "Client error" in result.message

    def test_classify_unexpected_error(self):
        """Classify unexpected status code."""
        from yahoofinance.core.errors import classify_api_error, APIError

        result = classify_api_error(200, "OK but error")

        assert isinstance(result, APIError)
        assert "Unexpected API error" in result.message

    def test_truncate_long_response(self):
        """Truncate long response text."""
        from yahoofinance.core.errors import classify_api_error

        long_text = "x" * 200

        result = classify_api_error(500, long_text)

        assert len(result.details["response_text"]) <= 103  # 100 + "..."
        assert result.details["response_text"].endswith("...")


class TestModuleStructure:
    """Test module structure."""

    def test_module_docstring(self):
        """Module has docstring."""
        from yahoofinance.core import errors

        assert errors.__doc__ is not None
        assert "Error handling" in errors.__doc__

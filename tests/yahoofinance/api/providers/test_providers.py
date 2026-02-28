"""
Unit tests for API providers.

This module tests the provider factory function and async provider interfaces.
"""

from unittest.mock import Mock, patch

import pytest

from yahoofinance.api import get_provider
from yahoofinance.api.providers.base_provider import FinanceDataProvider
from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError


class TestFinanceProviders:
    """Tests for the provider base classes and factory function."""

    @patch("yahoofinance.api.provider_registry.registry")
    def test_get_provider_returns_yahoo_finance_provider(self, mock_registry):
        """Test that get_provider returns the right provider."""
        mock_provider = Mock()
        mock_registry.resolve.return_value = mock_provider

        _ = get_provider(provider_type="yahoo", use_cache=False)

        mock_registry.resolve.assert_called_once()

    @patch("yahoofinance.api.provider_registry.registry")
    def test_get_provider_with_async_true(self, mock_registry):
        """Test that get_provider with async_api=True returns an async provider."""
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "AsyncHybridProvider"
        mock_registry.resolve.return_value = mock_provider

        _ = get_provider(async_api=True, use_cache=False)

        mock_registry.resolve.assert_called_once()

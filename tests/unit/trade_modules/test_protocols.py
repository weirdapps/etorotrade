"""
Tests for trade_modules/protocols.py

This module tests the Protocol definitions for trade modules.
"""

import pytest
from typing import Any, Dict, List
import pandas as pd
import logging

from trade_modules.protocols import (
    FinanceDataProviderProtocol,
    LoggerProtocol,
    ConfigProtocol,
    TradingCriteriaProtocol,
    AnalysisServiceProtocol,
    FilterServiceProtocol,
    PortfolioServiceProtocol,
    DataProcessingServiceProtocol,
)


class TestLoggerProtocol:
    """Tests for LoggerProtocol."""

    def test_logging_logger_matches_protocol(self):
        """Test that standard logging.Logger satisfies LoggerProtocol."""
        logger = logging.getLogger("test")
        assert isinstance(logger, LoggerProtocol)

    def test_protocol_requires_debug(self):
        """Test that protocol requires debug method."""
        logger = logging.getLogger("test")
        assert hasattr(logger, "debug")
        assert callable(logger.debug)

    def test_protocol_requires_info(self):
        """Test that protocol requires info method."""
        logger = logging.getLogger("test")
        assert hasattr(logger, "info")
        assert callable(logger.info)

    def test_protocol_requires_warning(self):
        """Test that protocol requires warning method."""
        logger = logging.getLogger("test")
        assert hasattr(logger, "warning")
        assert callable(logger.warning)

    def test_protocol_requires_error(self):
        """Test that protocol requires error method."""
        logger = logging.getLogger("test")
        assert hasattr(logger, "error")
        assert callable(logger.error)


class TestConfigProtocol:
    """Tests for ConfigProtocol."""

    def test_dict_matches_protocol(self):
        """Test that a dict satisfies ConfigProtocol."""
        config = {"key": "value"}
        assert isinstance(config, ConfigProtocol)

    def test_custom_config_class(self):
        """Test custom config class can implement protocol."""
        class CustomConfig:
            def get(self, key: str, default: Any = None) -> Any:
                return default

        config = CustomConfig()
        assert isinstance(config, ConfigProtocol)


class TestFinanceDataProviderProtocol:
    """Tests for FinanceDataProviderProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        # Protocol should be able to be used with isinstance
        assert hasattr(FinanceDataProviderProtocol, "__protocol_attrs__")

    def test_mock_provider_not_matching(self):
        """Test that object without methods doesn't match."""
        class NotAProvider:
            pass

        obj = NotAProvider()
        # Should not match because it lacks required methods
        assert not isinstance(obj, FinanceDataProviderProtocol)

    def test_partial_provider_not_matching(self):
        """Test that partial implementation doesn't match."""
        class PartialProvider:
            async def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
                return {}
            # Missing other required methods

        obj = PartialProvider()
        # Should not match because it lacks all required methods
        assert not isinstance(obj, FinanceDataProviderProtocol)


class TestTradingCriteriaProtocol:
    """Tests for TradingCriteriaProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        assert hasattr(TradingCriteriaProtocol, "__protocol_attrs__")

    def test_mock_criteria_not_matching(self):
        """Test that object without attributes doesn't match."""
        class NotCriteria:
            pass

        obj = NotCriteria()
        assert not isinstance(obj, TradingCriteriaProtocol)


class TestAnalysisServiceProtocol:
    """Tests for AnalysisServiceProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        assert hasattr(AnalysisServiceProtocol, "__protocol_attrs__")

    def test_implementation_check(self):
        """Test that proper implementation matches protocol."""
        class MockAnalysisService:
            def calculate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            def calculate_confidence_score(self, row: pd.Series) -> float:
                return 0.5

        service = MockAnalysisService()
        assert isinstance(service, AnalysisServiceProtocol)


class TestFilterServiceProtocol:
    """Tests for FilterServiceProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        assert hasattr(FilterServiceProtocol, "__protocol_attrs__")

    def test_implementation_check(self):
        """Test that proper implementation matches protocol."""
        class MockFilterService:
            def filter_buy_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            def filter_sell_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            def filter_hold_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            def filter_notrade_tickers(self, df: pd.DataFrame, notrade_path: str) -> pd.DataFrame:
                return df

        service = MockFilterService()
        assert isinstance(service, FilterServiceProtocol)


class TestPortfolioServiceProtocol:
    """Tests for PortfolioServiceProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        assert hasattr(PortfolioServiceProtocol, "__protocol_attrs__")

    def test_implementation_check(self):
        """Test that proper implementation matches protocol."""
        class MockPortfolioService:
            def apply_portfolio_filter(
                self, market_df: pd.DataFrame, portfolio_df: pd.DataFrame
            ) -> pd.DataFrame:
                return market_df

            def apply_portfolio_filters(
                self, results: Dict[str, pd.DataFrame], portfolio_df: pd.DataFrame
            ) -> Dict[str, pd.DataFrame]:
                return results

        service = MockPortfolioService()
        assert isinstance(service, PortfolioServiceProtocol)


class TestDataProcessingServiceProtocol:
    """Tests for DataProcessingServiceProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        assert hasattr(DataProcessingServiceProtocol, "__protocol_attrs__")


class TestProtocolIntegration:
    """Integration tests for protocols."""

    def test_all_protocols_are_runtime_checkable(self):
        """Test all protocols have runtime_checkable decorator."""
        protocols = [
            FinanceDataProviderProtocol,
            LoggerProtocol,
            ConfigProtocol,
            TradingCriteriaProtocol,
            AnalysisServiceProtocol,
            FilterServiceProtocol,
            PortfolioServiceProtocol,
            DataProcessingServiceProtocol,
        ]

        for protocol in protocols:
            # All should have _is_runtime_protocol attribute
            assert hasattr(protocol, "_is_runtime_protocol") or hasattr(protocol, "__protocol_attrs__")

    def test_protocols_have_no_concrete_implementation(self):
        """Test that protocols define abstract methods only."""
        # Protocols should not be instantiatable
        with pytest.raises(TypeError):
            FinanceDataProviderProtocol()

    def test_standard_library_types_compatibility(self):
        """Test standard library types satisfy appropriate protocols."""
        # logging.Logger satisfies LoggerProtocol
        logger = logging.getLogger()
        assert isinstance(logger, LoggerProtocol)

        # dict satisfies ConfigProtocol
        config = {}
        assert isinstance(config, ConfigProtocol)

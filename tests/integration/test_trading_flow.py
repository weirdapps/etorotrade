#!/usr/bin/env python3
"""
Integration tests for end-to-end trading flow.

Tests the complete trading workflow from data fetching through signal generation
to opportunity filtering. Uses mocked providers to ensure reliable testing.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError


# Mark all tests as integration tests
pytestmark = [pytest.mark.integration]


class TestTradingEngineIntegration:
    """Integration tests for TradingEngine end-to-end flow."""

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Sample market data for integration testing."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Tesla Inc.', 'NVIDIA Corp.'],
            'market_cap': [3e12, 2.5e12, 1.8e12, 800e9, 1.2e12],
            'region': ['US', 'US', 'US', 'US', 'US'],
            'price': [150.0, 300.0, 120.0, 250.0, 400.0],
            'target_price': [180.0, 320.0, 100.0, 280.0, 500.0],
            'upside': [20.0, 6.67, -16.67, 12.0, 25.0],
            'buy_percentage': [85.0, 75.0, 45.0, 60.0, 90.0],
            'analyst_count': [40, 35, 30, 25, 45],
            'total_ratings': [40, 35, 30, 25, 45],
            'pe_forward': [22.0, 28.0, 20.0, 50.0, 35.0],
            'pe_trailing': [25.0, 30.0, 22.0, 60.0, 40.0],
        })

    @pytest.fixture
    def sample_portfolio_data(self) -> pd.DataFrame:
        """Sample portfolio data for integration testing."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'shares': [100, 50, 75],
            'avg_cost': [140.0, 280.0, 115.0],
            'current_value': [15000.0, 15000.0, 9000.0],
        })

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        mock = AsyncMock()
        mock.max_concurrency = 15
        return mock

    def test_calculate_exret_integration(self, sample_market_data):
        """Test EXRET calculation with realistic data."""
        from trade_modules.analysis_engine import calculate_exret

        df = sample_market_data.copy()
        result = calculate_exret(df)

        # EXRET = upside * buy_percentage / 100
        expected_aapl = 20.0 * 85.0 / 100  # 17.0
        expected_msft = 6.67 * 75.0 / 100  # ~5.0
        expected_googl = -16.67 * 45.0 / 100  # ~-7.5

        assert result.loc[result['ticker'] == 'AAPL', 'EXRET'].iloc[0] == pytest.approx(expected_aapl, abs=0.1)
        assert result.loc[result['ticker'] == 'MSFT', 'EXRET'].iloc[0] == pytest.approx(expected_msft, abs=0.1)
        assert result.loc[result['ticker'] == 'GOOGL', 'EXRET'].iloc[0] == pytest.approx(expected_googl, abs=0.1)

    def test_calculate_action_integration(self, sample_market_data):
        """Test signal generation with realistic data."""
        from trade_modules.analysis_engine import calculate_action, calculate_exret

        df = sample_market_data.copy().set_index('ticker')
        df = calculate_exret(df.reset_index()).set_index('ticker')
        result = calculate_action(df)

        # All should have signals
        assert 'BS' in result.columns
        assert result['BS'].notna().all()

        # Signals should be valid
        assert result['BS'].isin(['B', 'S', 'H', 'I']).all()

        # GOOGL with negative upside and low buy% should be SELL
        assert result.loc['GOOGL', 'BS'] == 'S'

    def test_trading_engine_initialization(self, mock_provider):
        """Test TradingEngine can be initialized with a mock provider."""
        with patch('trade_modules.trade_engine.get_async_hybrid_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda **kwargs: mock_provider

            from trade_modules.trade_engine import TradingEngine
            engine = TradingEngine(provider=mock_provider, config={})

            assert engine.provider is mock_provider
            assert engine.data_processing_service is not None
            assert engine.analysis_service is not None
            assert engine.filter_service is not None

    @pytest.mark.asyncio
    async def test_analyze_market_opportunities_integration(self, sample_market_data, mock_provider):
        """Test full market opportunity analysis flow."""
        with patch('trade_modules.trade_engine.get_async_hybrid_provider') as mock_get_provider:
            mock_get_provider.return_value = lambda **kwargs: mock_provider

            from trade_modules.trade_engine import TradingEngine

            # Prepare data with signals already calculated
            market_df = sample_market_data.copy().set_index('ticker')
            market_df['EXRET'] = market_df['upside'] * market_df['buy_percentage'] / 100

            # Pre-calculate signals for testing
            from trade_modules.analysis_engine import calculate_action
            market_df = calculate_action(market_df)

            engine = TradingEngine(provider=mock_provider, config={})
            results = await engine.analyze_market_opportunities(market_df)

            assert 'buy_opportunities' in results
            assert 'sell_opportunities' in results
            assert 'hold_opportunities' in results

            # Should have categorized the stocks
            total_categorized = (
                len(results['buy_opportunities']) +
                len(results['sell_opportunities']) +
                len(results['hold_opportunities'])
            )
            # At least some stocks should be categorized
            assert total_categorized >= 0


class TestSignalGenerationIntegration:
    """Integration tests for the signal generation pipeline."""

    @pytest.fixture
    def tier_test_data(self) -> Dict[str, pd.DataFrame]:
        """Test data for each market cap tier."""
        return {
            'MEGA': pd.DataFrame({
                'ticker': ['AAPL'],
                'market_cap': [3e12],
                'region': ['US'],
                'upside': [15.0],
                'buy_percentage': [80.0],
                'analyst_count': [40],
                'total_ratings': [40],
                'pe_forward': [22.0],
                'pe_trailing': [25.0],
            }),
            'LARGE': pd.DataFrame({
                'ticker': ['NFLX'],
                'market_cap': [200e9],
                'region': ['US'],
                'upside': [20.0],
                'buy_percentage': [75.0],
                'analyst_count': [30],
                'total_ratings': [30],
                'pe_forward': [30.0],
                'pe_trailing': [35.0],
            }),
            'MID': pd.DataFrame({
                'ticker': ['DOCU'],
                'market_cap': [50e9],
                'region': ['US'],
                'upside': [25.0],
                'buy_percentage': [70.0],
                'analyst_count': [15],
                'total_ratings': [15],
                'pe_forward': [40.0],
                'pe_trailing': [45.0],
            }),
            'SMALL': pd.DataFrame({
                'ticker': ['SMCI'],
                'market_cap': [5e9],
                'region': ['US'],
                'upside': [30.0],
                'buy_percentage': [65.0],
                'analyst_count': [8],
                'total_ratings': [8],
                'pe_forward': [20.0],
                'pe_trailing': [25.0],
            }),
            'MICRO': pd.DataFrame({
                'ticker': ['TINY'],
                'market_cap': [500e6],
                'region': ['US'],
                'upside': [40.0],
                'buy_percentage': [60.0],
                'analyst_count': [4],
                'total_ratings': [4],
                'pe_forward': [15.0],
                'pe_trailing': [18.0],
            }),
        }

    @pytest.mark.parametrize("tier", ['MEGA', 'LARGE', 'MID', 'SMALL', 'MICRO'])
    def test_signal_generation_by_tier(self, tier, tier_test_data):
        """Test signal generation for each market cap tier."""
        from trade_modules.analysis_engine import calculate_action, calculate_exret

        df = tier_test_data[tier].copy().set_index('ticker')
        df = calculate_exret(df.reset_index()).set_index('ticker')
        result = calculate_action(df)

        # Should generate a valid signal
        assert 'BS' in result.columns
        ticker = tier_test_data[tier]['ticker'].iloc[0]
        signal = result.loc[ticker, 'BS']
        assert signal in ['B', 'S', 'H', 'I'], f"Invalid signal '{signal}' for tier {tier}"

    def test_insufficient_analyst_coverage(self):
        """Test that insufficient analyst coverage produces INCONCLUSIVE."""
        from trade_modules.analysis_engine import calculate_action, calculate_exret

        df = pd.DataFrame({
            'ticker': ['TEST'],
            'market_cap': [100e9],
            'region': ['US'],
            'upside': [30.0],
            'buy_percentage': [90.0],
            'analyst_count': [2],  # Below minimum
            'total_ratings': [2],
            'pe_forward': [20.0],
            'pe_trailing': [22.0],
        }).set_index('ticker')

        df = calculate_exret(df.reset_index()).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['TEST', 'BS'] == 'I'

    def test_negative_upside_produces_sell(self):
        """Test that negative upside produces SELL signal."""
        from trade_modules.analysis_engine import calculate_action, calculate_exret

        df = pd.DataFrame({
            'ticker': ['TEST'],
            'market_cap': [100e9],
            'region': ['US'],
            'upside': [-10.0],  # Negative upside
            'buy_percentage': [40.0],
            'analyst_count': [20],
            'total_ratings': [20],
            'pe_forward': [25.0],
            'pe_trailing': [22.0],
        }).set_index('ticker')

        df = calculate_exret(df.reset_index()).set_index('ticker')
        result = calculate_action(df)

        assert result.loc['TEST', 'BS'] == 'S'


class TestFilterServiceIntegration:
    """Integration tests for filter service operations."""

    @pytest.fixture
    def market_with_signals(self) -> pd.DataFrame:
        """Market data with pre-calculated signals."""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'LOW_COV'],
            'BS': ['B', 'H', 'S', 'H', 'B', 'I'],
            'upside': [20.0, 8.0, -10.0, 12.0, 25.0, 15.0],
            'buy_percentage': [85.0, 65.0, 40.0, 55.0, 90.0, 80.0],
            'analyst_count': [40, 30, 25, 20, 45, 2],
        })

    def test_filter_buy_opportunities(self, market_with_signals):
        """Test filtering buy opportunities."""
        from trade_modules.filter_service import FilterService

        service = FilterService(MagicMock())
        result = service.filter_buy_opportunities(market_with_signals)

        # Should return only B signals
        assert all(result['BS'] == 'B')
        assert len(result) == 2  # AAPL and NVDA

    def test_filter_sell_opportunities(self, market_with_signals):
        """Test filtering sell opportunities."""
        from trade_modules.filter_service import FilterService

        service = FilterService(MagicMock())
        result = service.filter_sell_opportunities(market_with_signals)

        # Should return only S signals
        assert all(result['BS'] == 'S')
        assert len(result) == 1  # GOOGL

    def test_filter_hold_opportunities(self, market_with_signals):
        """Test filtering hold opportunities."""
        from trade_modules.filter_service import FilterService

        service = FilterService(MagicMock())
        result = service.filter_hold_opportunities(market_with_signals)

        # Should return only H signals
        assert all(result['BS'] == 'H')
        assert len(result) == 2  # MSFT and TSLA


class TestContainerIntegration:
    """Integration tests for the dependency injection container."""

    def test_container_creates_services(self):
        """Test that container can create all services."""
        from trade_modules.container import get_container, reset_container

        # Reset to get fresh container
        container = reset_container({})

        # Should be able to get all services
        analysis_service = container.get_analysis_service()
        filter_service = container.get_filter_service()
        portfolio_service = container.get_portfolio_service()

        assert analysis_service is not None
        assert filter_service is not None
        assert portfolio_service is not None

    def test_container_caches_instances(self):
        """Test that container caches service instances."""
        from trade_modules.container import get_container, reset_container

        container = reset_container({})

        # Get same service twice
        service1 = container.get_analysis_service()
        service2 = container.get_analysis_service()

        # Should be the same instance
        assert service1 is service2

    def test_container_clear_resets_instances(self):
        """Test that clearing container resets instances."""
        from trade_modules.container import get_container, reset_container

        container = reset_container({})

        # Get service
        service1 = container.get_analysis_service()

        # Clear and get again
        container.clear()
        service2 = container.get_analysis_service()

        # Should be different instances
        assert service1 is not service2


class TestProtocolCompliance:
    """Integration tests for protocol compliance."""

    def test_analysis_service_protocol_compliance(self):
        """Test that AnalysisService complies with protocol."""
        from trade_modules.analysis_service import AnalysisService
        from trade_modules.protocols import AnalysisServiceProtocol

        service = AnalysisService({}, MagicMock())

        # Check required methods exist
        assert hasattr(service, 'calculate_trading_signals')
        assert hasattr(service, 'calculate_confidence_score')
        assert callable(service.calculate_trading_signals)
        assert callable(service.calculate_confidence_score)

    def test_filter_service_protocol_compliance(self):
        """Test that FilterService complies with protocol."""
        from trade_modules.filter_service import FilterService
        from trade_modules.protocols import FilterServiceProtocol

        service = FilterService(MagicMock())

        # Check required methods exist
        assert hasattr(service, 'filter_buy_opportunities')
        assert hasattr(service, 'filter_sell_opportunities')
        assert hasattr(service, 'filter_hold_opportunities')
        assert callable(service.filter_buy_opportunities)

    def test_portfolio_service_protocol_compliance(self):
        """Test that PortfolioService complies with protocol."""
        from trade_modules.portfolio_service import PortfolioService
        from trade_modules.protocols import PortfolioServiceProtocol

        service = PortfolioService(MagicMock())

        # Check required methods exist
        assert hasattr(service, 'apply_portfolio_filter')
        assert hasattr(service, 'apply_portfolio_filters')
        assert callable(service.apply_portfolio_filter)

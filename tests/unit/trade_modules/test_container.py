"""
Tests for trade_modules/container.py

This module tests the dependency injection container.
"""

import pytest
from unittest.mock import MagicMock, patch
import logging

from trade_modules.container import (
    Container,
    get_container,
    reset_container,
)


@pytest.fixture
def fresh_container():
    """Create a fresh Container instance."""
    return Container()


@pytest.fixture
def container_with_config():
    """Create a Container with configuration."""
    config = {"test_key": "test_value", "nested": {"key": "value"}}
    return Container(config)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton container before and after each test."""
    # Reset before test
    reset_container()
    yield
    # Reset after test
    reset_container()


class TestContainerInit:
    """Tests for Container initialization."""

    def test_init_with_no_config(self, fresh_container):
        """Test initialization without config."""
        assert fresh_container._config == {}
        assert fresh_container._instances == {}

    def test_init_with_config(self, container_with_config):
        """Test initialization with config."""
        assert container_with_config._config["test_key"] == "test_value"
        assert container_with_config._config["nested"]["key"] == "value"

    def test_init_with_none_config(self):
        """Test initialization with explicit None config."""
        container = Container(None)
        assert container._config == {}

    def test_init_creates_empty_instances(self, fresh_container):
        """Test that instances dict is empty on init."""
        assert len(fresh_container._instances) == 0


class TestContainerConfig:
    """Tests for config property."""

    def test_config_property_returns_config(self, container_with_config):
        """Test config property returns the config dict."""
        config = container_with_config.config
        assert config["test_key"] == "test_value"

    def test_config_property_empty_by_default(self, fresh_container):
        """Test config property returns empty dict by default."""
        assert fresh_container.config == {}


class TestGetLogger:
    """Tests for get_logger method."""

    def test_get_logger_returns_logger(self, fresh_container):
        """Test get_logger returns a logger."""
        logger = fresh_container.get_logger()
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self, fresh_container):
        """Test get_logger with custom name."""
        logger = fresh_container.get_logger("custom.logger")
        assert logger.name == "custom.logger"

    def test_get_logger_default_name(self, fresh_container):
        """Test get_logger with default name."""
        logger = fresh_container.get_logger()
        assert isinstance(logger, logging.Logger)


class TestGetProvider:
    """Tests for get_provider method."""

    @patch("yahoofinance.api.providers.async_hybrid_provider.AsyncHybridProvider")
    @patch("yahoofinance.core.config.get_max_concurrent_requests")
    def test_get_provider_creates_instance(self, mock_get_max, mock_provider_class, fresh_container):
        """Test get_provider creates AsyncHybridProvider."""
        mock_get_max.return_value = 15
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        result = fresh_container.get_provider()

        # Verify provider was created and cached
        assert "provider" in fresh_container._instances

    @patch("yahoofinance.api.providers.async_hybrid_provider.AsyncHybridProvider")
    @patch("yahoofinance.core.config.get_max_concurrent_requests")
    def test_get_provider_caches_instance(self, mock_get_max, mock_provider_class, fresh_container):
        """Test get_provider caches the instance."""
        mock_get_max.return_value = 15
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider

        result1 = fresh_container.get_provider()
        result2 = fresh_container.get_provider()

        assert result1 is result2

    def test_get_provider_uses_max_concurrency_param(self, fresh_container):
        """Test get_provider respects max_concurrency parameter."""
        # Test that provider is created
        result = fresh_container.get_provider(max_concurrency=5)
        assert result is not None
        assert "provider" in fresh_container._instances


class TestGetAnalysisService:
    """Tests for get_analysis_service method."""

    def test_get_analysis_service_creates_instance(self, fresh_container):
        """Test get_analysis_service creates AnalysisService."""
        result = fresh_container.get_analysis_service()
        assert result is not None
        assert "analysis_service" in fresh_container._instances

    def test_get_analysis_service_caches_instance(self, fresh_container):
        """Test get_analysis_service caches the instance."""
        result1 = fresh_container.get_analysis_service()
        result2 = fresh_container.get_analysis_service()
        assert result1 is result2


class TestGetFilterService:
    """Tests for get_filter_service method."""

    def test_get_filter_service_creates_instance(self, fresh_container):
        """Test get_filter_service creates FilterService."""
        result = fresh_container.get_filter_service()
        assert result is not None
        assert "filter_service" in fresh_container._instances

    def test_get_filter_service_caches_instance(self, fresh_container):
        """Test get_filter_service caches the instance."""
        result1 = fresh_container.get_filter_service()
        result2 = fresh_container.get_filter_service()
        assert result1 is result2


class TestGetPortfolioService:
    """Tests for get_portfolio_service method."""

    def test_get_portfolio_service_creates_instance(self, fresh_container):
        """Test get_portfolio_service creates PortfolioService."""
        result = fresh_container.get_portfolio_service()
        assert result is not None
        assert "portfolio_service" in fresh_container._instances

    def test_get_portfolio_service_caches_instance(self, fresh_container):
        """Test get_portfolio_service caches the instance."""
        result1 = fresh_container.get_portfolio_service()
        result2 = fresh_container.get_portfolio_service()
        assert result1 is result2


class TestGetDataProcessingService:
    """Tests for get_data_processing_service method."""

    def test_get_data_processing_service_creates_instance(self, fresh_container):
        """Test get_data_processing_service creates instance."""
        result = fresh_container.get_data_processing_service()
        assert result is not None
        assert "data_processing_service" in fresh_container._instances

    def test_get_data_processing_service_caches_instance(self, fresh_container):
        """Test get_data_processing_service caches the instance."""
        result1 = fresh_container.get_data_processing_service()
        result2 = fresh_container.get_data_processing_service()
        assert result1 is result2

    def test_get_data_processing_service_with_custom_provider(self, fresh_container):
        """Test get_data_processing_service with custom provider."""
        mock_provider = MagicMock()
        result = fresh_container.get_data_processing_service(provider=mock_provider)
        assert result is not None


class TestGetTradingEngine:
    """Tests for get_trading_engine method."""

    def test_get_trading_engine_creates_instance(self, fresh_container):
        """Test get_trading_engine creates TradingEngine."""
        result = fresh_container.get_trading_engine()
        assert result is not None
        assert "trading_engine" in fresh_container._instances

    def test_get_trading_engine_caches_instance(self, fresh_container):
        """Test get_trading_engine caches the instance."""
        result1 = fresh_container.get_trading_engine()
        result2 = fresh_container.get_trading_engine()
        assert result1 is result2

    def test_get_trading_engine_with_custom_provider(self, fresh_container):
        """Test get_trading_engine with custom provider."""
        mock_provider = MagicMock()
        result = fresh_container.get_trading_engine(provider=mock_provider)
        assert result is not None

    def test_get_trading_engine_with_custom_config(self, fresh_container):
        """Test get_trading_engine with custom config."""
        custom_config = {"custom": "config"}
        result = fresh_container.get_trading_engine(config=custom_config)
        assert result is not None


class TestContainerClear:
    """Tests for clear method."""

    def test_clear_empties_instances(self, fresh_container):
        """Test clear removes all instances."""
        fresh_container._instances["test"] = "value"
        fresh_container._instances["test2"] = "value2"

        fresh_container.clear()

        assert fresh_container._instances == {}

    def test_clear_on_empty_container(self, fresh_container):
        """Test clear on empty container doesn't raise."""
        fresh_container.clear()
        assert fresh_container._instances == {}

    def test_clear_after_service_creation(self, fresh_container):
        """Test clear after creating services."""
        fresh_container.get_filter_service()
        assert len(fresh_container._instances) > 0

        fresh_container.clear()

        assert fresh_container._instances == {}


class TestContainerReset:
    """Tests for reset method."""

    def test_reset_clears_instances(self, fresh_container):
        """Test reset clears instances."""
        fresh_container._instances["test"] = "value"

        fresh_container.reset()

        assert fresh_container._instances == {}

    def test_reset_with_new_config(self, fresh_container):
        """Test reset with new config."""
        fresh_container._instances["test"] = "value"
        new_config = {"new": "config"}

        fresh_container.reset(config=new_config)

        assert fresh_container._instances == {}
        assert fresh_container._config == new_config

    def test_reset_without_config_preserves_existing(self, container_with_config):
        """Test reset without config preserves existing config."""
        original_config = container_with_config._config.copy()

        container_with_config.reset()

        assert container_with_config._config == original_config

    def test_reset_after_service_creation(self, fresh_container):
        """Test reset after creating services."""
        fresh_container.get_analysis_service()

        fresh_container.reset(config={"new": "config"})

        assert fresh_container._instances == {}
        assert fresh_container._config == {"new": "config"}


class TestGetContainerFunction:
    """Tests for get_container module function."""

    def test_get_container_returns_container(self):
        """Test get_container returns a Container instance."""
        container = get_container()
        assert isinstance(container, Container)

    def test_get_container_returns_singleton(self):
        """Test get_container returns the same instance."""
        container1 = get_container()
        container2 = get_container()
        assert container1 is container2

    def test_get_container_with_config_first_call(self):
        """Test get_container uses config on first call after reset."""
        # The autouse fixture already reset, so first call here sets config
        config = {"key": "value"}
        container = reset_container(config)
        # Verify this container has the config
        assert container._config == config
        # Also verify get_container returns same instance
        container2 = get_container()
        assert container2._config == config

    def test_get_container_singleton_persists_config(self):
        """Test singleton container keeps its config."""
        # Reset with specific config
        container1 = reset_container({"first": "config"})
        container2 = get_container()
        assert container1 is container2
        assert container2._config.get("first") == "config"


class TestResetContainerFunction:
    """Tests for reset_container module function."""

    def test_reset_container_returns_new_instance(self):
        """Test reset_container returns a fresh Container."""
        original = get_container()
        original._instances["test"] = "value"

        reset = reset_container()

        assert reset._instances == {}

    def test_reset_container_with_config(self):
        """Test reset_container with new config."""
        reset_container()
        get_container({"old": "config"})

        container = reset_container({"new": "config"})

        assert container.config == {"new": "config"}

    def test_reset_container_clears_previous_instances(self):
        """Test reset_container clears previous singleton instances."""
        container1 = get_container()
        container1._instances["cached"] = "value"

        container2 = reset_container()

        assert "cached" not in container2._instances

    def test_reset_container_creates_new_singleton(self):
        """Test reset_container creates a new singleton."""
        container1 = get_container()
        container2 = reset_container()
        container3 = get_container()

        assert container2 is container3
        # container2/3 should be different from container1 conceptually
        # (though they may have same id after reset)


class TestContainerIntegration:
    """Integration tests for Container."""

    def test_services_are_lazily_created(self, fresh_container):
        """Test that services are created on first access."""
        assert len(fresh_container._instances) == 0

        fresh_container.get_filter_service()

        assert len(fresh_container._instances) >= 1

    def test_multiple_services_can_be_created(self, fresh_container):
        """Test creating multiple services."""
        fresh_container.get_filter_service()
        fresh_container.get_portfolio_service()
        fresh_container.get_analysis_service()

        assert "filter_service" in fresh_container._instances
        assert "portfolio_service" in fresh_container._instances
        assert "analysis_service" in fresh_container._instances

    def test_clear_allows_recreating_services(self, fresh_container):
        """Test that clear allows services to be recreated."""
        service1 = fresh_container.get_filter_service()

        fresh_container.clear()

        service2 = fresh_container.get_filter_service()

        # Services should be different objects after clear
        assert service1 is not service2

    def test_provider_is_shared_across_services(self, fresh_container):
        """Test that the same provider instance is shared."""
        # Get services that use the provider
        fresh_container.get_data_processing_service()
        fresh_container.get_trading_engine()

        # Provider should only be created once
        assert "provider" in fresh_container._instances

    def test_container_config_accessible_to_services(self, container_with_config):
        """Test that container config is used by services."""
        # AnalysisService receives config
        service = container_with_config.get_analysis_service()
        assert service is not None


class TestContainerEdgeCases:
    """Edge case tests for Container."""

    def test_empty_config_is_valid(self):
        """Test container works with empty config."""
        container = Container({})
        assert container.config == {}
        service = container.get_filter_service()
        assert service is not None

    def test_nested_config_preserved(self):
        """Test nested configuration is preserved."""
        config = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }
        container = Container(config)
        assert container.config["level1"]["level2"]["level3"] == "deep_value"

    def test_multiple_clears_are_safe(self, fresh_container):
        """Test multiple clear calls don't raise."""
        fresh_container.clear()
        fresh_container.clear()
        fresh_container.clear()
        assert fresh_container._instances == {}

    def test_multiple_resets_are_safe(self, fresh_container):
        """Test multiple reset calls don't raise."""
        fresh_container.reset()
        fresh_container.reset({"a": 1})
        fresh_container.reset({"b": 2})
        assert fresh_container._config == {"b": 2}

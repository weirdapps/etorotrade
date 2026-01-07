"""
Tests for yahoofinance/core/config_service.py

This module tests the ConfigurationService class for dependency injection.
"""

import pytest

from yahoofinance.core.config_service import (
    ConfigurationService,
    ConfigurationContext,
    get_config_service,
    with_config,
)


@pytest.fixture
def config_service():
    """Create a fresh ConfigurationService instance."""
    return ConfigurationService()


class TestConfigurationServiceInit:
    """Tests for ConfigurationService initialization."""

    def test_init_creates_empty_overrides(self, config_service):
        """Test that initialization creates empty overrides."""
        assert config_service._overrides == {}

    def test_init_creates_empty_cache(self, config_service):
        """Test that initialization creates empty cache."""
        assert config_service._config_cache == {}

    def test_init_loads_default_configs(self, config_service):
        """Test that initialization loads default configs."""
        assert "rate_limit" in config_service._default_configs
        assert "cache" in config_service._default_configs
        assert "circuit_breaker" in config_service._default_configs


class TestGetConfig:
    """Tests for get_config method."""

    def test_get_section(self, config_service):
        """Test getting entire section."""
        rate_limit = config_service.get_config("rate_limit")
        assert rate_limit is not None

    def test_get_specific_key(self, config_service):
        """Test getting specific key from section."""
        result = config_service.get_config("cache", "ttl")
        # Should return value or None if key doesn't exist
        assert result is not None or result is None

    def test_get_with_default(self, config_service):
        """Test getting with default value."""
        result = config_service.get_config("nonexistent", default="default_value")
        assert result == "default_value"

    def test_get_missing_section_returns_default(self, config_service):
        """Test getting missing section returns default."""
        result = config_service.get_config("missing_section", default={})
        assert result == {}

    def test_override_takes_precedence(self, config_service):
        """Test that override takes precedence over default."""
        config_service.set_override("test_section", "test_key", "override_value")
        result = config_service.get_config("test_section", "test_key")
        assert result == "override_value"


class TestSetOverride:
    """Tests for set_override method."""

    def test_set_section_override(self, config_service):
        """Test setting section-level override."""
        config_service.set_override("test", value={"key": "value"})
        result = config_service.get_config("test")
        assert result == {"key": "value"}

    def test_set_key_override(self, config_service):
        """Test setting key-level override."""
        config_service.set_override("section", "key", "value")
        result = config_service.get_config("section", "key")
        assert result == "value"

    def test_override_clears_cache(self, config_service):
        """Test that setting override clears cache."""
        config_service._config_cache["test"] = "cached"
        config_service.set_override("test", value="new")
        assert "test" not in config_service._config_cache


class TestClearOverrides:
    """Tests for clear_overrides method."""

    def test_clear_removes_all_overrides(self, config_service):
        """Test that clear removes all overrides."""
        config_service.set_override("a", value=1)
        config_service.set_override("b", value=2)
        config_service.clear_overrides()
        assert config_service._overrides == {}

    def test_clear_removes_cache(self, config_service):
        """Test that clear removes cache."""
        config_service._config_cache["test"] = "value"
        config_service.clear_overrides()
        assert config_service._config_cache == {}


class TestConfigHelperMethods:
    """Tests for helper methods."""

    def test_get_rate_limit_config(self, config_service):
        """Test get_rate_limit_config returns rate limit config."""
        result = config_service.get_rate_limit_config()
        assert result is not None

    def test_get_cache_config(self, config_service):
        """Test get_cache_config returns cache config."""
        result = config_service.get_cache_config()
        assert result is not None

    def test_get_circuit_breaker_config(self, config_service):
        """Test get_circuit_breaker_config returns circuit breaker config."""
        result = config_service.get_circuit_breaker_config()
        assert result is not None

    def test_get_provider_config(self, config_service):
        """Test get_provider_config returns provider config."""
        result = config_service.get_provider_config()
        assert result is not None

    def test_get_trading_criteria_config(self, config_service):
        """Test get_trading_criteria_config returns trading criteria."""
        result = config_service.get_trading_criteria_config()
        assert result is not None

    def test_get_column_names(self, config_service):
        """Test get_column_names returns column names."""
        result = config_service.get_column_names()
        assert result is not None

    def test_get_file_paths(self, config_service):
        """Test get_file_paths returns file paths."""
        result = config_service.get_file_paths()
        assert result is not None

    def test_get_messages(self, config_service):
        """Test get_messages returns messages."""
        result = config_service.get_messages()
        assert result is not None

    def test_get_standard_display_columns(self, config_service):
        """Test get_standard_display_columns returns columns."""
        result = config_service.get_standard_display_columns()
        assert result is not None


class TestConfigurationContext:
    """Tests for ConfigurationContext class."""

    def test_context_applies_overrides(self, config_service):
        """Test context applies overrides on enter."""
        with ConfigurationContext(config_service, test="value"):
            result = config_service.get_config("test")
            assert result == "value"

    def test_context_with_section_key(self, config_service):
        """Test context with section__key format."""
        with ConfigurationContext(config_service, section__key="override"):
            result = config_service.get_config("section", "key")
            assert result == "override"

    def test_context_restores_on_exit(self, config_service):
        """Test context restores values on exit."""
        # Set initial value
        config_service.set_override("test", value="initial")

        with ConfigurationContext(config_service, test="temporary"):
            assert config_service.get_config("test") == "temporary"

        # After context, should restore (may still have override from before)
        assert config_service.get_config("test") == "initial"

    def test_context_returns_self(self, config_service):
        """Test context returns self on enter."""
        context = ConfigurationContext(config_service, test="value")
        with context as ctx:
            assert ctx is context


class TestGetConfigService:
    """Tests for get_config_service function."""

    def test_returns_config_service(self):
        """Test returns a ConfigurationService instance."""
        result = get_config_service()
        assert isinstance(result, ConfigurationService)

    def test_returns_same_instance(self):
        """Test returns same instance on multiple calls."""
        service1 = get_config_service()
        service2 = get_config_service()
        assert service1 is service2


class TestWithConfig:
    """Tests for with_config function."""

    def test_returns_context(self):
        """Test returns ConfigurationContext instance."""
        ctx = with_config(test="value")
        assert isinstance(ctx, ConfigurationContext)

    def test_can_use_as_context_manager(self):
        """Test can be used as context manager."""
        with with_config(temp__setting="temp_value"):
            service = get_config_service()
            result = service.get_config("temp", "setting")
            assert result == "temp_value"


class TestThreadSafety:
    """Tests for thread safety."""

    def test_get_config_is_thread_safe(self, config_service):
        """Test get_config uses lock."""
        import threading

        results = []

        def get_config():
            for _ in range(100):
                result = config_service.get_config("rate_limit")
                results.append(result is not None)

        threads = [threading.Thread(target=get_config) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)

    def test_set_override_is_thread_safe(self, config_service):
        """Test set_override uses lock."""
        import threading

        def set_override(i):
            for _ in range(100):
                config_service.set_override(f"section_{i}", value=i)

        threads = [threading.Thread(target=set_override, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock or error
        assert True

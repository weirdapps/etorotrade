#!/usr/bin/env python3
"""
Tests for provider registry.
Target: Increase coverage for yahoofinance/api/provider_registry.py
"""

import pytest
from unittest.mock import patch, MagicMock


class TestProviderTypes:
    """Test PROVIDER_TYPES configuration."""

    def test_provider_types_has_yahoo(self):
        """PROVIDER_TYPES includes yahoo."""
        from yahoofinance.api.provider_registry import PROVIDER_TYPES

        assert "yahoo" in PROVIDER_TYPES
        assert "async" in PROVIDER_TYPES["yahoo"]

    def test_provider_types_has_yahooquery(self):
        """PROVIDER_TYPES includes yahooquery."""
        from yahoofinance.api.provider_registry import PROVIDER_TYPES

        assert "yahooquery" in PROVIDER_TYPES
        assert "async" in PROVIDER_TYPES["yahooquery"]

    def test_provider_types_has_hybrid(self):
        """PROVIDER_TYPES includes hybrid."""
        from yahoofinance.api.provider_registry import PROVIDER_TYPES

        assert "hybrid" in PROVIDER_TYPES
        assert "async" in PROVIDER_TYPES["hybrid"]


class TestGetProvider:
    """Test get_provider function."""

    def test_get_provider_default(self):
        """Get provider with defaults."""
        from yahoofinance.api.provider_registry import get_provider

        provider = get_provider()

        assert provider is not None

    def test_get_provider_async(self):
        """Get async provider."""
        from yahoofinance.api.provider_registry import get_provider

        provider = get_provider(provider_type="hybrid", async_mode=True, use_cache=False)

        assert provider is not None

    def test_get_provider_invalid_type_raises(self):
        """Invalid provider type raises ValidationError."""
        from yahoofinance.api.provider_registry import get_provider
        from yahoofinance.core.errors import ValidationError

        with pytest.raises(ValidationError, match="Unknown provider type"):
            get_provider(provider_type="invalid_type")

    def test_get_provider_case_insensitive(self):
        """Provider type is case insensitive."""
        from yahoofinance.api.provider_registry import get_provider

        provider = get_provider(provider_type="HYBRID", async_mode=True, use_cache=False)

        assert provider is not None

    def test_get_provider_sync_raises(self):
        """Requesting sync provider raises ValidationError since sync providers are removed."""
        from yahoofinance.api.provider_registry import get_provider
        from yahoofinance.core.errors import ValidationError

        with pytest.raises(ValidationError, match="No sync provider available"):
            get_provider(provider_type="hybrid", async_mode=False, use_cache=False)

    def test_get_provider_caching(self):
        """Provider caching works."""
        from yahoofinance.api.provider_registry import get_provider, clear_provider_cache

        clear_provider_cache()

        provider1 = get_provider(provider_type="hybrid", use_cache=True)
        provider2 = get_provider(provider_type="hybrid", use_cache=True)

        assert provider1 is provider2

    def test_get_provider_no_cache(self):
        """Provider no caching creates new instances."""
        from yahoofinance.api.provider_registry import get_provider

        provider1 = get_provider(provider_type="hybrid", use_cache=False)
        provider2 = get_provider(provider_type="hybrid", use_cache=False)

        # Different instances when not caching
        assert provider1 is not provider2


class TestGetAllProviders:
    """Test get_all_providers function."""

    def test_get_all_providers_async(self):
        """Get all async providers."""
        from yahoofinance.api.provider_registry import get_all_providers

        providers = get_all_providers(async_mode=True)

        assert isinstance(providers, dict)
        assert len(providers) > 0


class TestGetDefaultProvider:
    """Test get_default_provider function."""

    def test_get_default_provider(self):
        """Get default provider."""
        from yahoofinance.api.provider_registry import get_default_provider

        provider = get_default_provider()

        assert provider is not None


class TestClearProviderCache:
    """Test clear_provider_cache function."""

    def test_clear_provider_cache(self):
        """Clear provider cache works."""
        from yahoofinance.api.provider_registry import (
            get_provider, clear_provider_cache, _provider_cache
        )

        # Create a provider
        get_provider(provider_type="hybrid", use_cache=True)

        # Clear cache
        clear_provider_cache()

        assert len(_provider_cache) == 0

    def test_clear_provider_cache_empty(self):
        """Clear empty cache doesn't raise."""
        from yahoofinance.api.provider_registry import clear_provider_cache, _provider_cache

        _provider_cache.clear()

        # Should not raise
        clear_provider_cache()


class TestInitializeRegistry:
    """Test initialize_registry function."""

    def test_initialize_registry_runs(self):
        """Initialize registry runs without error."""
        from yahoofinance.api.provider_registry import initialize_registry

        # Should not raise
        initialize_registry()


class TestConstants:
    """Test module constants."""

    def test_default_provider_type(self):
        """Default provider type is set."""
        from yahoofinance.api.provider_registry import DEFAULT_PROVIDER_TYPE

        assert DEFAULT_PROVIDER_TYPE == "hybrid"

    def test_default_async_mode(self):
        """Default async mode is True (async-only)."""
        from yahoofinance.api.provider_registry import DEFAULT_ASYNC_MODE

        assert DEFAULT_ASYNC_MODE is True

    def test_default_enhanced(self):
        """Default enhanced is set."""
        from yahoofinance.api.provider_registry import DEFAULT_ENHANCED

        assert DEFAULT_ENHANCED is False


class TestModuleStructure:
    """Test module structure."""

    def test_logger_exists(self):
        """Module has logger."""
        from yahoofinance.api import provider_registry

        assert hasattr(provider_registry, 'logger')
        assert provider_registry.logger is not None

    def test_provider_cache_exists(self):
        """Module has provider cache."""
        from yahoofinance.api import provider_registry

        assert hasattr(provider_registry, '_provider_cache')
        assert isinstance(provider_registry._provider_cache, dict)


class TestEdgeCases:
    """Test edge cases."""

    def test_get_provider_with_kwargs(self):
        """Get provider with additional kwargs."""
        from yahoofinance.api.provider_registry import get_provider

        provider = get_provider(
            provider_type="hybrid",
            async_mode=True,
            use_cache=False,
            max_retries=3
        )

        assert provider is not None

    def test_cache_key_with_cacheable_kwargs(self):
        """Cache key considers cacheable kwargs."""
        from yahoofinance.api.provider_registry import get_provider, clear_provider_cache

        clear_provider_cache()

        provider1 = get_provider(
            provider_type="hybrid",
            use_cache=True,
            max_retries=3
        )
        provider2 = get_provider(
            provider_type="hybrid",
            use_cache=True,
            max_retries=5
        )

        # Different kwargs should create different cache keys
        # but both should work
        assert provider1 is not None
        assert provider2 is not None

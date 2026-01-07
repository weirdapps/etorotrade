#!/usr/bin/env python3
"""
Tests for config loader module.
Target: Increase coverage for yahoofinance/core/config_loader.py
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_config_exists(self):
        """DEFAULT_CONFIG is defined."""
        from yahoofinance.core.config_loader import DEFAULT_CONFIG

        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, dict)

    def test_default_config_sections(self):
        """DEFAULT_CONFIG has expected sections."""
        from yahoofinance.core.config_loader import DEFAULT_CONFIG

        assert "data" in DEFAULT_CONFIG
        assert "trading" in DEFAULT_CONFIG
        assert "output" in DEFAULT_CONFIG
        assert "performance" in DEFAULT_CONFIG
        assert "logging" in DEFAULT_CONFIG

    def test_default_data_section(self):
        """Data section has expected keys."""
        from yahoofinance.core.config_loader import DEFAULT_CONFIG

        data = DEFAULT_CONFIG["data"]
        assert "portfolio_csv" in data
        assert "cache_enabled" in data
        assert "cache_ttl_hours" in data

    def test_default_trading_section(self):
        """Trading section has expected keys."""
        from yahoofinance.core.config_loader import DEFAULT_CONFIG

        trading = DEFAULT_CONFIG["trading"]
        assert "value_threshold" in trading
        assert "growth_threshold" in trading
        assert "min_analysts" in trading
        assert "min_price_targets" in trading


class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_init_with_path(self):
        """Initialize with explicit path."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader(config_path="/custom/path/config.yaml")

        assert loader.config_path == "/custom/path/config.yaml"

    def test_init_without_path(self):
        """Initialize without path finds config file."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()

        assert loader.config_path is not None

    def test_load_returns_dict(self):
        """Load returns configuration dictionary."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        config = loader.load()

        assert isinstance(config, dict)

    def test_load_cached(self):
        """Load returns cached config on second call."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        config1 = loader.load()
        config2 = loader.load()

        assert config1 is config2

    def test_load_with_nonexistent_file(self):
        """Load returns defaults with nonexistent file."""
        from yahoofinance.core.config_loader import ConfigLoader, DEFAULT_CONFIG

        loader = ConfigLoader(config_path="/nonexistent/path/config.yaml")
        config = loader.load()

        assert config == DEFAULT_CONFIG

    def test_load_with_real_config_file(self):
        """Load reads from real config file."""
        from yahoofinance.core.config_loader import ConfigLoader
        import yaml

        # Create temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test_key': 'test_value'}, f)
            temp_path = f.name

        try:
            loader = ConfigLoader(config_path=temp_path)
            config = loader.load()

            assert 'test_key' in config
            assert config['test_key'] == 'test_value'
        finally:
            os.unlink(temp_path)

    def test_get_with_valid_path(self):
        """Get returns value for valid key path."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = {'section': {'key': 'value'}}

        result = loader.get('section.key')

        assert result == 'value'

    def test_get_with_invalid_path(self):
        """Get returns default for invalid key path."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = {'section': {'key': 'value'}}

        result = loader.get('invalid.path', default='default_value')

        assert result == 'default_value'

    def test_get_with_partial_path(self):
        """Get returns section for partial path."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = {'section': {'key': 'value'}}

        result = loader.get('section')

        assert result == {'key': 'value'}

    def test_merge_configs_simple(self):
        """Merge configs combines dictionaries."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()

        default = {'a': 1, 'b': 2}
        user = {'b': 3, 'c': 4}

        result = loader._merge_configs(default, user)

        assert result['a'] == 1
        assert result['b'] == 3
        assert result['c'] == 4

    def test_merge_configs_nested(self):
        """Merge configs handles nested dictionaries."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()

        default = {'section': {'a': 1, 'b': 2}}
        user = {'section': {'b': 3}}

        result = loader._merge_configs(default, user)

        assert result['section']['a'] == 1
        assert result['section']['b'] == 3

    def test_find_config_file(self):
        """Find config file searches directories."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        path = loader._find_config_file()

        assert path is not None
        assert isinstance(path, str)


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_config(self):
        """get_config returns configuration dictionary."""
        from yahoofinance.core.config_loader import get_config

        config = get_config()

        assert isinstance(config, dict)

    def test_get_config_value(self):
        """get_config_value returns specific value."""
        from yahoofinance.core.config_loader import get_config_value

        # Should return a value (either from config or defaults)
        result = get_config_value('trading.min_analysts', default=5)

        assert result is not None

    def test_get_config_value_with_default(self):
        """get_config_value returns default for missing key."""
        from yahoofinance.core.config_loader import get_config_value

        result = get_config_value('nonexistent.key', default='default_value')

        assert result == 'default_value'

    def test_reload_config(self):
        """reload_config forces reload."""
        from yahoofinance.core.config_loader import reload_config

        config = reload_config()

        assert isinstance(config, dict)


class TestEdgeCases:
    """Test edge cases."""

    def test_load_with_empty_yaml(self):
        """Load handles empty YAML file."""
        from yahoofinance.core.config_loader import ConfigLoader, DEFAULT_CONFIG
        import yaml

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            loader = ConfigLoader(config_path=temp_path)
            config = loader.load()

            # Should merge with defaults
            assert config is not None
        finally:
            os.unlink(temp_path)

    def test_load_with_invalid_yaml(self):
        """Load handles invalid YAML file."""
        from yahoofinance.core.config_loader import ConfigLoader, DEFAULT_CONFIG

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            temp_path = f.name

        try:
            loader = ConfigLoader(config_path=temp_path)
            config = loader.load()

            # Should return defaults on error
            assert config == DEFAULT_CONFIG
        finally:
            os.unlink(temp_path)

    def test_get_with_non_dict_value(self):
        """Get handles non-dict intermediate values."""
        from yahoofinance.core.config_loader import ConfigLoader

        loader = ConfigLoader()
        loader._config = {'section': 'not_a_dict'}

        result = loader.get('section.key', default='default')

        assert result == 'default'

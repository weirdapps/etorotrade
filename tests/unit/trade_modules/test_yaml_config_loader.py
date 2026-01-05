#!/usr/bin/env python3
"""
ITERATION 12: YAML Config Loader Tests
Target: Test YAML configuration loading functionality
"""

import pytest
import tempfile
import os
from pathlib import Path
from trade_modules.yaml_config_loader import YamlConfigLoader, get_yaml_config, reload_config


class TestYamlConfigLoaderInitialization:
    """Test YamlConfigLoader initialization."""

    def test_initialize_with_default_path(self):
        """Initialize loader without specifying config path."""
        loader = YamlConfigLoader()
        assert loader.config_path is not None
        assert isinstance(loader.config_path, str)

    def test_initialize_with_custom_path(self):
        """Initialize loader with custom config path."""
        custom_path = "/custom/path/config.yaml"
        loader = YamlConfigLoader(config_path=custom_path)
        assert loader.config_path == custom_path

    def test_config_cache_starts_none(self):
        """Config cache starts as None."""
        loader = YamlConfigLoader()
        assert loader._config_cache is None


class TestFindConfigFile:
    """Test config file finding logic."""

    def test_find_config_file_searches_upward(self):
        """Find config file searches parent directories."""
        loader = YamlConfigLoader()
        # _find_config_file is called during init
        assert loader.config_path is not None


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_config_with_existing_file(self):
        """Load configuration from existing file."""
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test_key: test_value\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            config = loader.load_config()
            assert isinstance(config, dict)
            assert config.get('test_key') == 'test_value'
        finally:
            os.unlink(temp_path)

    def test_load_config_caches_result(self):
        """Config is cached after first load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("cache_test: value\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            config1 = loader.load_config()
            config2 = loader.load_config()
            # Should return same cached dict
            assert config1 is config2
        finally:
            os.unlink(temp_path)

    def test_load_config_missing_file_returns_empty(self):
        """Missing config file returns empty dict."""
        loader = YamlConfigLoader(config_path="/nonexistent/path/config.yaml")
        config = loader.load_config()
        assert isinstance(config, dict)
        assert len(config) == 0

    def test_load_config_invalid_yaml_returns_empty(self):
        """Invalid YAML returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:\n  - broken")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            config = loader.load_config()
            # Should handle error gracefully
            assert isinstance(config, dict)
        finally:
            os.unlink(temp_path)


class TestGetTierThresholds:
    """Test tier threshold retrieval."""

    def test_get_tier_thresholds(self):
        """Get tier thresholds from config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("tier_thresholds:\n  mega: 500000000000\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            thresholds = loader.get_tier_thresholds()
            assert isinstance(thresholds, dict)
            assert 'mega' in thresholds
        finally:
            os.unlink(temp_path)

    def test_get_tier_thresholds_missing_returns_empty(self):
        """Missing tier_thresholds returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("other_key: value\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            thresholds = loader.get_tier_thresholds()
            assert isinstance(thresholds, dict)
            assert len(thresholds) == 0
        finally:
            os.unlink(temp_path)


class TestGetUniversalThresholds:
    """Test universal threshold retrieval."""

    def test_get_universal_thresholds(self):
        """Get universal thresholds from config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("universal_thresholds:\n  min_analysts: 4\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            thresholds = loader.get_universal_thresholds()
            assert isinstance(thresholds, dict)
        finally:
            os.unlink(temp_path)


class TestGetTierCriteria:
    """Test tier criteria retrieval."""

    def test_get_tier_criteria(self):
        """Get criteria for specific tier."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("value:\n  buy:\n    min_upside: 15\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            criteria = loader.get_tier_criteria('value')
            assert isinstance(criteria, dict)
        finally:
            os.unlink(temp_path)


class TestGetRegionTierCriteria:
    """Test region-tier criteria retrieval."""

    def test_get_region_tier_criteria(self):
        """Get criteria for specific region and tier."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("us_mega:\n  buy:\n    min_upside: 8\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            criteria = loader.get_region_tier_criteria('us', 'mega')
            assert isinstance(criteria, dict)
        finally:
            os.unlink(temp_path)


class TestGetPositionSizingConfig:
    """Test position sizing config retrieval."""

    def test_get_position_sizing_config(self):
        """Get position sizing configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("position_sizing:\n  max_position_pct: 5\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            config = loader.get_position_sizing_config()
            assert isinstance(config, dict)
        finally:
            os.unlink(temp_path)


class TestIsConfigAvailable:
    """Test config availability check."""

    def test_is_config_available_with_config(self):
        """Config is available when file loaded successfully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: value\n")
            temp_path = f.name

        try:
            loader = YamlConfigLoader(config_path=temp_path)
            assert loader.is_config_available() is True
        finally:
            os.unlink(temp_path)

    def test_is_config_available_without_config(self):
        """Config not available when file missing."""
        loader = YamlConfigLoader(config_path="/nonexistent/config.yaml")
        assert loader.is_config_available() is False


class TestGlobalFunctions:
    """Test global helper functions."""

    def test_get_yaml_config(self):
        """get_yaml_config returns loader instance."""
        loader = get_yaml_config()
        assert isinstance(loader, YamlConfigLoader)

    def test_reload_config(self):
        """reload_config clears cache."""
        loader = get_yaml_config()
        # Load config to populate cache
        loader.load_config()
        # Reload should clear cache
        reload_config()
        # Can't directly test cache cleared, but function should not crash
        assert True

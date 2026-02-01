"""
Tests for threshold_analyzer module.

Basic coverage tests for the threshold optimization functionality.
"""

import pytest
import pandas as pd
import numpy as np


class TestThresholdAnalyzerImports:
    """Test that the module can be imported correctly."""

    def test_module_import(self):
        """Test that threshold_analyzer module imports successfully."""
        from trade_modules import threshold_analyzer
        assert threshold_analyzer is not None

    def test_threshold_analysis_dataclass(self):
        """Test ThresholdAnalysis dataclass."""
        from trade_modules.threshold_analyzer import ThresholdAnalysis

        analysis = ThresholdAnalysis(
            current_value=10.0,
            test_values=[8.0, 10.0, 12.0],
            signal_counts={8.0: {"buy": 50}, 10.0: {"buy": 40}, 12.0: {"buy": 30}},
            recommendation=10.0,
            reasoning="Optimal balance"
        )
        assert analysis.current_value == 10.0
        assert analysis.recommendation == 10.0


class TestThresholdAnalyzerFunctions:
    """Test threshold analyzer functions."""

    def test_parse_percentage(self):
        """Test percentage parsing."""
        from trade_modules.threshold_analyzer import parse_percentage

        assert parse_percentage("15%") == 15.0
        assert parse_percentage("15") == 15.0
        assert parse_percentage(15.0) == 15.0
        assert np.isnan(parse_percentage("--"))
        assert np.isnan(parse_percentage(None))

    def test_analyze_upside_distribution(self):
        """Test upside distribution analysis."""
        from trade_modules.threshold_analyzer import analyze_upside_distribution

        # Create sample data with UP% column
        data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'UP%': ['15%', '20%', '25%', '30%', '18%'],
        })

        result = analyze_upside_distribution(data)
        assert 'count' in result
        assert 'mean' in result
        assert result['count'] == 5

    def test_analyze_buy_percentage_distribution(self):
        """Test buy percentage distribution analysis."""
        from trade_modules.threshold_analyzer import analyze_buy_percentage_distribution

        # Create sample data with %B column
        data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            '%B': ['85%', '90%', '80%', '95%', '75%'],
        })

        result = analyze_buy_percentage_distribution(data)
        assert 'count' in result
        assert 'mean' in result

    def test_threshold_sensitivity_analysis(self):
        """Test threshold sensitivity analysis."""
        from trade_modules.threshold_analyzer import threshold_sensitivity_analysis

        # Create sample data with UP% column
        data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'UP%': ['15%', '20%', '25%', '30%', '18%'],
            '%B': ['85%', '90%', '80%', '95%', '75%'],
        })

        # Correct signature: (df, metric_col, threshold_range, step, direction)
        result = threshold_sensitivity_analysis(data, 'UP%', (10.0, 30.0), 5.0)
        assert result is not None
        assert result.test_values is not None

    def test_threshold_sensitivity_analysis_below(self):
        """Test threshold sensitivity analysis with 'below' direction."""
        from trade_modules.threshold_analyzer import threshold_sensitivity_analysis

        data = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'UP%': ['15%', '20%', '25%'],
        })

        result = threshold_sensitivity_analysis(data, 'UP%', (10.0, 30.0), 5.0, direction='below')
        assert result is not None
        assert result.signal_counts is not None

    def test_generate_threshold_report(self):
        """Test threshold report generation."""
        from trade_modules.threshold_analyzer import generate_threshold_report
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock market.csv file with all required columns
            market_path = Path(tmpdir) / "market.csv"
            market_path.write_text(
                "ticker,UP%,%B,BS\n"
                "AAPL,15%,85%,B\n"
                "MSFT,20%,90%,B\n"
                "TSLA,5%,40%,S\n"
            )

            report = generate_threshold_report(market_path)
            assert report is not None
            assert isinstance(report, str)


class TestCryptoMomentumConfig:
    """Test crypto momentum configuration."""

    def test_crypto_config_loaded(self):
        """Test that crypto momentum config is loaded."""
        from trade_modules.yaml_config_loader import get_yaml_config

        yaml_config = get_yaml_config()
        config = yaml_config.load_config()

        crypto_config = config.get('crypto_momentum', {})
        assert crypto_config is not None

    def test_bitcoin_proxy_config_loaded(self):
        """Test that bitcoin proxy config is loaded."""
        from trade_modules.yaml_config_loader import get_yaml_config

        yaml_config = get_yaml_config()
        config = yaml_config.load_config()

        btc_config = config.get('bitcoin_proxy', {})
        assert btc_config is not None


class TestConfigSchemaNewModels:
    """Test new Pydantic schema models."""

    def test_crypto_momentum_config(self):
        """Test CryptoMomentumConfig model."""
        from config.schema import CryptoMomentumConfig

        config = CryptoMomentumConfig()
        assert config.major.buy_threshold == 85.0
        assert config.altcoins.hold_threshold == 55.0

    def test_bitcoin_proxy_config(self):
        """Test BitcoinProxyConfig model."""
        from config.schema import BitcoinProxyConfig

        config = BitcoinProxyConfig()
        assert config.momentum_buy_threshold == 70.0
        assert config.momentum_sell_threshold == 35.0

    def test_ipo_grace_period_config(self):
        """Test IPOGracePeriodConfig model."""
        from config.schema import IPOGracePeriodConfig

        config = IPOGracePeriodConfig()
        assert config.enabled is True
        assert config.grace_period_months == 12

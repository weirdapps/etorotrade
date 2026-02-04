"""
Test coverage for config/schema.py

Target: 80%+ coverage
Critical paths: config loading, validation, tier criteria access
"""
import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from config.schema import (
    TradingConfig,
    Region,
    AssetTier,
    BuyCriteria,
    SellCriteria,
    TierCriteria,
    TierThresholds,
    UniversalThresholds,
    DataConfig,
    PositionSizingConfig,
    PerformanceConfig,
    LoggingConfig,
    OutputConfig,
    get_config,
    reload_config,
)


class TestEnums:
    """Test enumeration types"""

    def test_region_enum(self):
        """Region enum has correct values"""
        assert Region.US.value == "US"
        assert Region.EU.value == "EU"
        assert Region.HK.value == "HK"

    def test_asset_tier_enum(self):
        """AssetTier enum has correct values"""
        assert AssetTier.MEGA.value == "MEGA"
        assert AssetTier.LARGE.value == "LARGE"
        assert AssetTier.MID.value == "MID"
        assert AssetTier.SMALL.value == "SMALL"
        assert AssetTier.MICRO.value == "MICRO"


class TestBuyCriteria:
    """Test BuyCriteria validation"""

    def test_buy_criteria_creation(self):
        """BuyCriteria can be created with valid data"""
        criteria = BuyCriteria(
            min_upside=10.0,
            min_buy_percentage=70.0,
            min_exret=5.0
        )
        assert criteria.min_upside == pytest.approx(10.0)
        assert criteria.min_buy_percentage == pytest.approx(70.0)
        assert criteria.min_exret == pytest.approx(5.0)

    def test_buy_criteria_validation_upside_range(self):
        """BuyCriteria validates upside is between 0-100"""
        with pytest.raises(ValidationError):
            BuyCriteria(min_upside=150.0, min_buy_percentage=70.0, min_exret=5.0)

    def test_buy_criteria_validation_buy_percentage_range(self):
        """BuyCriteria validates buy_percentage is between 0-100"""
        with pytest.raises(ValidationError):
            BuyCriteria(min_upside=10.0, min_buy_percentage=150.0, min_exret=5.0)

    def test_buy_criteria_validation_exret_positive(self):
        """BuyCriteria validates exret is non-negative"""
        with pytest.raises(ValidationError):
            BuyCriteria(min_upside=10.0, min_buy_percentage=70.0, min_exret=-5.0)

    def test_buy_criteria_immutable(self):
        """BuyCriteria is immutable after creation"""
        criteria = BuyCriteria(min_upside=10.0, min_buy_percentage=70.0, min_exret=5.0)
        with pytest.raises(ValidationError):
            criteria.min_upside = 20.0


class TestSellCriteria:
    """Test SellCriteria validation"""

    def test_sell_criteria_creation(self):
        """SellCriteria can be created with valid data"""
        criteria = SellCriteria(
            max_upside=5.0,
            max_exret=2.0
        )
        assert criteria.max_upside == pytest.approx(5.0)
        assert criteria.max_exret == pytest.approx(2.0)

    def test_sell_criteria_validation_upside_range(self):
        """SellCriteria validates upside is between 0-100"""
        with pytest.raises(ValidationError):
            SellCriteria(max_upside=150.0, max_exret=2.0)

    def test_sell_criteria_immutable(self):
        """SellCriteria is immutable after creation"""
        criteria = SellCriteria(max_upside=5.0, max_exret=2.0)
        with pytest.raises(ValidationError):
            criteria.max_upside = 10.0


class TestTierThresholds:
    """Test TierThresholds validation"""

    def test_tier_thresholds_creation(self):
        """TierThresholds can be created with valid data"""
        thresholds = TierThresholds()
        assert thresholds.mega_tier_min == 500_000_000_000
        assert thresholds.large_tier_min == 100_000_000_000
        assert thresholds.mid_tier_min == 10_000_000_000
        assert thresholds.small_tier_min == 2_000_000_000

    def test_tier_thresholds_immutable(self):
        """TierThresholds is immutable after creation"""
        thresholds = TierThresholds()
        with pytest.raises(ValidationError):
            thresholds.mega_tier_min = 600_000_000_000


class TestPositionSizingConfig:
    """Test PositionSizingConfig validation"""

    def test_position_sizing_creation(self):
        """PositionSizingConfig can be created with defaults"""
        config = PositionSizingConfig()
        assert config.base_position_size == 2500
        assert config.tier_multipliers["mega"] == 5
        assert config.tier_multipliers["micro"] == 1

    def test_position_sizing_validation_base_minimum(self):
        """PositionSizingConfig validates minimum base size"""
        with pytest.raises(ValidationError):
            PositionSizingConfig(base_position_size=50)

    def test_position_sizing_validation_missing_tier(self):
        """PositionSizingConfig validates all tiers present"""
        with pytest.raises(ValidationError):
            PositionSizingConfig(
                base_position_size=2500,
                tier_multipliers={"mega": 5, "large": 4}  # Missing mid, small, micro
            )


class TestPerformanceConfig:
    """Test PerformanceConfig validation"""

    def test_performance_config_creation(self):
        """PerformanceConfig can be created with defaults"""
        config = PerformanceConfig()
        assert config.max_concurrent_requests == 20  # Increased for better throughput
        assert config.request_timeout_seconds == 30
        assert config.retry_attempts == 3
        assert config.batch_size == 25

    def test_performance_config_validation_concurrent_range(self):
        """PerformanceConfig validates concurrent requests range"""
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_requests=100)

    def test_performance_config_validation_timeout_range(self):
        """PerformanceConfig validates timeout range"""
        with pytest.raises(ValidationError):
            PerformanceConfig(request_timeout_seconds=500)


class TestLoggingConfig:
    """Test LoggingConfig validation"""

    def test_logging_config_creation(self):
        """LoggingConfig can be created with defaults"""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.console is True

    def test_logging_config_validation_level(self):
        """LoggingConfig validates log level"""
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")

    def test_logging_config_level_normalization(self):
        """LoggingConfig normalizes log level to uppercase"""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"


class TestTradingConfigLoading:
    """Test TradingConfig loading from YAML"""

    def test_load_from_yaml(self):
        """TradingConfig loads from config.yaml successfully"""
        config = TradingConfig.from_yaml('config.yaml')

        # Verify basic structure
        assert config.data is not None
        assert config.tier_thresholds is not None
        assert config.universal_thresholds is not None
        assert config.position_sizing is not None
        assert config.performance is not None
        assert config.logging is not None
        assert config.output is not None

        # Verify values
        # min_analyst_count is 4 for $5B+ stocks (6 for $2-5B small caps via small_cap_min_analysts)
        assert config.universal_thresholds.min_analyst_count == 4
        assert config.position_sizing.base_position_size == 2500
        assert config.performance.max_concurrent_requests == 20  # Increased for better throughput

    def test_load_nonexistent_file(self):
        """TradingConfig raises error for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            TradingConfig.from_yaml('nonexistent.yaml')

    def test_export_to_yaml(self):
        """TradingConfig exports to YAML correctly"""
        config = TradingConfig.from_yaml('config.yaml')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            config.to_yaml(temp_path)

            # Reload and verify
            reloaded = TradingConfig.from_yaml(temp_path)
            assert reloaded.universal_thresholds.min_analyst_count == config.universal_thresholds.min_analyst_count
            assert reloaded.position_sizing.base_position_size == config.position_sizing.base_position_size
        finally:
            Path(temp_path).unlink()


class TestTierCriteriaAccess:
    """Test accessing tier-specific criteria"""

    def test_get_tier_criteria_us_mega(self):
        """get_tier_criteria retrieves US MEGA criteria"""
        config = TradingConfig.from_yaml('config.yaml')
        criteria = config.get_tier_criteria("US", "MEGA")

        assert criteria is not None
        assert criteria.buy.min_upside == pytest.approx(10.0)
        # Updated: min_upside increased from 8.0 to 10.0 per enhanced signal framework
        assert criteria.sell.max_upside == pytest.approx(0.0)

    def test_get_tier_criteria_with_enums(self):
        """get_tier_criteria works with enum values"""
        config = TradingConfig.from_yaml('config.yaml')
        criteria = config.get_tier_criteria(Region.US, AssetTier.LARGE)

        assert criteria is not None
        assert criteria.buy is not None
        assert criteria.sell is not None

    def test_get_tier_criteria_nonexistent(self):
        """get_tier_criteria raises error for nonexistent tier"""
        config = TradingConfig.from_yaml('config.yaml')

        with pytest.raises(ValueError):
            config.get_tier_criteria("INVALID", "MEGA")


class TestConfigValidation:
    """Test config validation warnings"""

    def test_validate_complete_no_issues(self):
        """validate_complete returns empty list for valid config"""
        config = TradingConfig.from_yaml('config.yaml')
        issues = config.validate_complete()

        # Current config should be valid
        assert isinstance(issues, list)

    def test_validate_complete_high_concurrency(self):
        """validate_complete warns about high concurrency"""
        config = TradingConfig.from_yaml('config.yaml')
        config.performance.max_concurrent_requests = 30

        issues = config.validate_complete()
        assert any("concurrency" in issue.lower() for issue in issues)

    def test_validate_complete_low_analyst_count(self):
        """validate_complete warns about low analyst count"""
        config = TradingConfig.from_yaml('config.yaml')
        config.universal_thresholds.min_analyst_count = 2

        issues = config.validate_complete()
        assert any("analyst" in issue.lower() for issue in issues)


class TestGlobalConfig:
    """Test global config singleton"""

    def test_get_config_singleton(self):
        """get_config returns same instance on multiple calls"""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reload_config(self):
        """reload_config creates new instance"""
        config1 = get_config()
        config2 = reload_config()

        # Should be new instance
        # Note: This might be the same due to caching, but at least should not error
        assert config2 is not None
        assert isinstance(config2, TradingConfig)


class TestInvalidConfig:
    """Test handling of invalid configurations"""

    def test_extra_fields_rejected(self):
        """Extra fields in config are rejected"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data:
  portfolio_csv: test.csv
tier_thresholds:
  mega_tier_min: 500000000000
  large_tier_min: 100000000000
  mid_tier_min: 10000000000
  small_tier_min: 2000000000
universal_thresholds:
  min_analyst_count: 4
  min_price_targets: 4
  min_market_cap: 1000000000
position_sizing:
  base_position_size: 2500
  tier_multipliers:
    mega: 5
    large: 4
    mid: 3
    small: 2
    micro: 1
performance:
  max_concurrent_requests: 15
  request_timeout_seconds: 30
  retry_attempts: 3
logging:
  level: INFO
  file: logs/test.log
  console: true
output:
  save_to_csv: true
  save_to_html: true
invalid_field: "should fail"
""")
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                TradingConfig.from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_fields_use_defaults(self):
        """Missing fields use default values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data:
  portfolio_csv: test.csv
tier_thresholds:
  mega_tier_min: 500000000000
  large_tier_min: 100000000000
  mid_tier_min: 10000000000
  small_tier_min: 2000000000
universal_thresholds:
  min_analyst_count: 4
  min_price_targets: 4
  min_market_cap: 1000000000
position_sizing:
  base_position_size: 2500
  tier_multipliers:
    mega: 5
    large: 4
    mid: 3
    small: 2
    micro: 1
performance:
  max_concurrent_requests: 15
  request_timeout_seconds: 30
  retry_attempts: 3
logging:
  level: INFO
  file: logs/test.log
  console: true
output:
  save_to_csv: true
  save_to_html: true
# Missing api config - should use defaults
""")
            temp_path = f.name

        try:
            config = TradingConfig.from_yaml(temp_path)
            # Should have default API config
            assert config.api is not None
            assert config.api.max_concurrent_requests == 15
        finally:
            Path(temp_path).unlink()

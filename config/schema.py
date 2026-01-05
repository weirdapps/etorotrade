"""
Centralized configuration schema with Pydantic validation.

This replaces scattered config across multiple files with a single
validated source of truth.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import warnings


class Region(str, Enum):
    """Trading regions"""
    US = "US"
    EU = "EU"
    HK = "HK"


class AssetTier(str, Enum):
    """Market cap tiers"""
    MEGA = "MEGA"    # ≥$500B
    LARGE = "LARGE"  # $100-500B
    MID = "MID"      # $10-100B
    SMALL = "SMALL"  # $2-10B
    MICRO = "MICRO"  # <$2B


class BuyCriteria(BaseModel):
    """Buy signal criteria"""
    model_config = ConfigDict(frozen=True)  # Immutable after creation

    min_upside: float = Field(ge=0, le=100, description="Minimum upside %")
    min_buy_percentage: float = Field(ge=0, le=100, description="Minimum buy %")
    min_exret: float = Field(ge=0, description="Minimum EXRET")

    # Beta constraints
    min_beta: Optional[float] = Field(ge=0, default=None, description="Minimum beta")
    max_beta: Optional[float] = Field(ge=0, default=None, description="Maximum beta")

    # PE ratio constraints
    min_forward_pe: Optional[float] = Field(ge=0, default=None, description="Minimum forward P/E")
    max_forward_pe: Optional[float] = Field(ge=0, default=None, description="Maximum forward P/E")
    min_trailing_pe: Optional[float] = Field(ge=0, default=None, description="Minimum trailing P/E")
    max_trailing_pe: Optional[float] = Field(ge=0, default=None, description="Maximum trailing P/E")

    # Other constraints
    max_peg: Optional[float] = Field(ge=0, default=None, description="Maximum PEG ratio")
    max_short_interest: Optional[float] = Field(ge=0, default=None, description="Maximum short interest %")
    min_earnings_growth: Optional[float] = Field(default=None, description="Minimum earnings growth %")
    min_price_performance: Optional[float] = Field(default=None, description="Minimum price performance %")

    # Analyst requirements
    min_analysts: Optional[int] = Field(ge=1, default=None, description="Minimum analyst count")
    min_price_targets: Optional[int] = Field(ge=1, default=None, description="Minimum price targets")

    # Financial health
    min_roe: Optional[float] = Field(default=None, description="Minimum ROE %")
    max_debt_equity: Optional[float] = Field(ge=0, default=None, description="Maximum debt/equity ratio")


class SellCriteria(BaseModel):
    """Sell signal criteria"""
    model_config = ConfigDict(frozen=True)  # Immutable after creation

    max_upside: float = Field(ge=0, le=100, description="Maximum upside %")
    min_buy_percentage: Optional[float] = Field(ge=0, le=100, default=None, description="Minimum buy % for sell")
    max_exret: float = Field(ge=0, description="Maximum EXRET")

    # Beta constraints
    min_beta: Optional[float] = Field(ge=0, default=None, description="Minimum beta for sell")

    # PE ratio constraints
    min_forward_pe: Optional[float] = Field(ge=0, default=None, description="Minimum forward P/E for sell")
    min_trailing_pe: Optional[float] = Field(ge=0, default=None, description="Minimum trailing P/E for sell")

    # Other constraints
    min_peg: Optional[float] = Field(ge=0, default=None, description="Minimum PEG ratio for sell")
    min_short_interest: Optional[float] = Field(ge=0, default=None, description="Minimum short interest % for sell")
    max_earnings_growth: Optional[float] = Field(default=None, description="Maximum earnings growth % for sell")
    max_price_performance: Optional[float] = Field(default=None, description="Maximum price performance % for sell")

    # Financial health
    min_roe: Optional[float] = Field(default=None, description="Minimum ROE % for sell")
    max_debt_equity: Optional[float] = Field(ge=0, default=None, description="Maximum debt/equity for sell")


class TierCriteria(BaseModel):
    """Trading criteria for a specific tier and region"""
    model_config = ConfigDict(frozen=True)  # Immutable after creation

    buy: BuyCriteria
    sell: SellCriteria


class TierThresholds(BaseModel):
    """Market cap thresholds for tier classification"""
    model_config = ConfigDict(frozen=True)

    mega_tier_min: float = Field(
        default=500_000_000_000,  # $500B
        ge=100_000_000_000,
        description="Minimum market cap for MEGA tier ($)"
    )
    large_tier_min: float = Field(
        default=100_000_000_000,  # $100B
        ge=10_000_000_000,
        description="Minimum market cap for LARGE tier ($)"
    )
    mid_tier_min: float = Field(
        default=10_000_000_000,  # $10B
        ge=1_000_000_000,
        description="Minimum market cap for MID tier ($)"
    )
    small_tier_min: float = Field(
        default=2_000_000_000,  # $2B
        ge=100_000_000,
        description="Minimum market cap for SMALL tier ($)"
    )

    @field_validator('mega_tier_min')
    @classmethod
    def validate_mega(cls, v, info):
        """Ensure mega tier threshold is largest"""
        if 'large_tier_min' in info.data and v <= info.data['large_tier_min']:
            raise ValueError("mega_tier_min must be > large_tier_min")
        return v


class UniversalThresholds(BaseModel):
    """Universal thresholds applied to all tiers"""
    min_analyst_count: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Minimum analysts required for signal generation"
    )
    min_price_targets: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Minimum price targets required"
    )
    min_market_cap: float = Field(
        default=1_000_000_000,  # $1B
        ge=100_000_000,  # $100M
        description="Minimum market cap for trading ($)"
    )


class DataConfig(BaseModel):
    """Data and file configuration"""
    portfolio_csv: Path = Field(
        default=Path("yahoofinance/input/portfolio.csv"),
        description="Portfolio CSV file path"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable data caching"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 7 days
        description="Cache TTL in hours"
    )

    @field_validator('portfolio_csv')
    @classmethod
    def validate_portfolio_path(cls, v):
        """Ensure portfolio path is valid"""
        v = Path(v)
        if not v.exists():
            warnings.warn(f"Portfolio file {v} doesn't exist")
        return v


class APIConfig(BaseModel):
    """API and data provider configuration"""
    max_concurrent_requests: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum concurrent API requests"
    )
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )
    cache_ttl: int = Field(
        default=172800,  # 48 hours
        ge=3600,  # 1 hour
        le=604800,  # 7 days
        description="API cache TTL in seconds"
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )


class PositionSizingConfig(BaseModel):
    """Position sizing configuration"""
    base_position_size: float = Field(
        default=2500,
        ge=100,
        description="Base position size in USD"
    )
    tier_multipliers: Dict[str, int] = Field(
        default_factory=lambda: {
            "mega": 5,
            "large": 4,
            "mid": 3,
            "small": 2,
            "micro": 1
        },
        description="Position size multipliers per tier"
    )

    @field_validator('tier_multipliers')
    @classmethod
    def validate_tier_multipliers(cls, v):
        """Ensure all tiers have multipliers"""
        required_tiers = {"mega", "large", "mid", "small", "micro"}
        if not required_tiers.issubset(v.keys()):
            raise ValueError(f"Missing tier multipliers. Required: {required_tiers}")
        return v


class PerformanceConfig(BaseModel):
    """Performance and execution configuration"""
    max_concurrent_requests: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum concurrent API requests"
    )
    request_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    file: Path = Field(
        default=Path("logs/trading_analysis.log"),
        description="Log file path"
    )
    console: bool = Field(
        default=True,
        description="Enable console logging"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        """Validate logging level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()


class OutputConfig(BaseModel):
    """Output file configuration"""
    save_to_csv: bool = Field(
        default=True,
        description="Save reports to CSV"
    )
    save_to_html: bool = Field(
        default=True,
        description="Save reports to HTML"
    )
    output_dir: Path = Field(
        default=Path("yahoofinance/output"),
        description="Output directory for reports"
    )

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v):
        """Ensure directory exists or can be created"""
        v = Path(v)
        if not v.exists():
            warnings.warn(f"Output directory {v} doesn't exist, will be created")
        return v


class TradingConfig(BaseModel):
    """
    Complete trading system configuration with validation.

    This is the single source of truth for all configuration.
    All settings are validated on load, preventing configuration bugs.

    Example:
        ```python
        # Load from YAML:
        config = TradingConfig.from_yaml('config.yaml')

        # Access with type safety:
        if config.api.max_concurrent_requests > 20:
            print("High concurrency configuration!")

        # Get tier criteria:
        us_mega = config.get_tier_criteria("US", "MEGA")
        ```
    """
    model_config = ConfigDict(
        validate_assignment=True,  # Validate on changes
        extra='forbid',  # Reject unknown fields (catches typos)
        frozen=False,  # Allow runtime changes
    )

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    tier_thresholds: TierThresholds = Field(default_factory=TierThresholds)
    universal_thresholds: UniversalThresholds = Field(default_factory=UniversalThresholds)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    api: Optional[APIConfig] = Field(default_factory=APIConfig)

    # Region-tier specific criteria
    # US criteria
    us_mega: Optional[TierCriteria] = None
    us_large: Optional[TierCriteria] = None
    us_mid: Optional[TierCriteria] = None
    us_small: Optional[TierCriteria] = None
    us_micro: Optional[TierCriteria] = None

    # EU criteria
    eu_mega: Optional[TierCriteria] = None
    eu_large: Optional[TierCriteria] = None
    eu_mid: Optional[TierCriteria] = None
    eu_small: Optional[TierCriteria] = None
    eu_micro: Optional[TierCriteria] = None

    # HK criteria
    hk_mega: Optional[TierCriteria] = None
    hk_large: Optional[TierCriteria] = None
    hk_mid: Optional[TierCriteria] = None
    hk_small: Optional[TierCriteria] = None
    hk_micro: Optional[TierCriteria] = None

    # Ticker mappings
    dual_listed_mappings: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Dual-listed ticker mappings"
    )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TradingConfig":
        """Load configuration from YAML file with validation"""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Export configuration to YAML file"""
        import yaml

        # Convert to dict, handling Path objects
        data = self.model_dump(mode='json', exclude_none=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_tier_criteria(self, region: str | Region, tier: str | AssetTier) -> TierCriteria:
        """Get trading criteria for specific region and tier"""
        region_str = region.value if isinstance(region, Region) else region
        tier_str = tier.value if isinstance(tier, AssetTier) else tier

        # Build field name (e.g., "us_mega")
        field_name = f"{region_str.lower()}_{tier_str.lower()}"

        criteria = getattr(self, field_name, None)
        if criteria is None:
            raise ValueError(f"No criteria found for {region_str}/{tier_str}")

        return criteria

    def validate_complete(self) -> List[str]:
        """
        Run comprehensive validation and return list of warnings/errors.

        Returns:
            List of validation messages (empty if all valid)
        """
        issues = []

        # Check performance config
        if self.performance.max_concurrent_requests > 25:
            issues.append(
                f"⚠️  High concurrency ({self.performance.max_concurrent_requests}) "
                f"may trigger API rate limits"
            )

        # Check API config
        if self.api:
            if self.api.cache_ttl < 3600:
                issues.append(
                    f"⚠️  Cache TTL {self.api.cache_ttl}s is very short "
                    f"(may cause excessive API calls)"
                )

        # Check universal thresholds
        if self.universal_thresholds.min_analyst_count < 4:
            issues.append(
                f"⚠️  Min analyst count {self.universal_thresholds.min_analyst_count} "
                f"may produce unreliable signals (4+ recommended)"
            )

        # Check that required criteria exist
        required_combinations = [
            ("US", "MEGA"), ("US", "LARGE"), ("US", "MID"), ("US", "SMALL"), ("US", "MICRO")
        ]

        for region, tier in required_combinations:
            try:
                self.get_tier_criteria(region, tier)
            except ValueError:
                issues.append(f"⚠️  Missing criteria for {region}/{tier}")

        return issues


# Global singleton (loaded once on startup)
_config: Optional[TradingConfig] = None


def get_config() -> TradingConfig:
    """
    Get the global configuration instance.

    Loads from config.yaml on first call, then returns cached instance.
    """
    global _config

    if _config is None:
        _config = TradingConfig.from_yaml('config.yaml')

        # Validate and warn about issues
        issues = _config.validate_complete()
        if issues:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Configuration validation found %d issues:\n%s",
                len(issues),
                "\n".join(issues)
            )

    return _config


def reload_config() -> TradingConfig:
    """Reload configuration from file (useful for tests)"""
    global _config
    _config = None
    return get_config()

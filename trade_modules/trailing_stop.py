"""Position-level trailing stop (CIO v44).

Checks whether a position's drawdown from its high-water mark exceeds
a tier-aware threshold. Does NOT execute trades — returns a result
object that the committee synthesis can act on.

Thresholds are deliberately wide (20-35%) to avoid premature exit on
volatile names while still catching catastrophic drawdowns. The system
had a live position with -74.5% drawdown (UVXY) with no exit rule.
"""

from dataclasses import dataclass


@dataclass
class TrailingStopConfig:
    """Trailing stop configuration with tier-aware thresholds."""

    default_pct: float = 20.0
    mega_pct: float = 25.0
    large_pct: float = 22.0
    crypto_pct: float = 35.0
    micro_pct: float = 15.0
    enabled: bool = True


@dataclass
class TrailingStopResult:
    """Result of a trailing stop check."""

    triggered: bool
    drawdown_from_high_pct: float
    threshold_pct: float
    ticker: str = ""


def _threshold_for_tier(tier: str, is_crypto: bool, config: TrailingStopConfig) -> float:
    """Get the trailing stop threshold for a given tier."""
    if is_crypto:
        return config.crypto_pct
    tier_map = {
        "MEGA": config.mega_pct,
        "LARGE": config.large_pct,
        "MID": config.default_pct,
        "SMALL": config.default_pct,
        "MICRO": config.micro_pct,
    }
    return tier_map.get(tier.upper(), config.default_pct)


def check_trailing_stop(
    ticker: str,
    current_price: float,
    high_since_entry: float,
    config: TrailingStopConfig | None = None,
    tier: str = "MID",
) -> TrailingStopResult:
    """Check if a position has breached its trailing stop.

    Args:
        ticker: Stock ticker symbol
        current_price: Current market price
        high_since_entry: Highest price since position was opened
        config: Stop configuration (uses defaults if None)
        tier: Market cap tier (MEGA/LARGE/MID/SMALL/MICRO)

    Returns:
        TrailingStopResult with triggered flag and drawdown details
    """
    if config is None:
        config = TrailingStopConfig()

    if not config.enabled or high_since_entry <= 0 or current_price <= 0:
        return TrailingStopResult(
            triggered=False,
            drawdown_from_high_pct=0.0,
            threshold_pct=0.0,
            ticker=ticker,
        )

    is_crypto = ticker.endswith("-USD")
    threshold = _threshold_for_tier(tier, is_crypto, config)
    drawdown = (high_since_entry - current_price) / high_since_entry * 100

    return TrailingStopResult(
        triggered=drawdown >= threshold,
        drawdown_from_high_pct=round(drawdown, 2),
        threshold_pct=threshold,
        ticker=ticker,
    )


def check_portfolio_stops(
    positions: list[dict],
    config: TrailingStopConfig | None = None,
) -> list[TrailingStopResult]:
    """Check trailing stops for all positions in a portfolio.

    Args:
        positions: List of dicts with keys: ticker, current_price,
                   high_since_entry, tier (optional)

    Returns:
        List of TrailingStopResult, one per position. Only triggered
        results need action.
    """
    if config is None:
        config = TrailingStopConfig()

    results = []
    for pos in positions:
        result = check_trailing_stop(
            ticker=pos.get("ticker", ""),
            current_price=pos.get("current_price", 0),
            high_since_entry=pos.get("high_since_entry", 0),
            config=config,
            tier=pos.get("tier", "MID"),
        )
        results.append(result)
    return results

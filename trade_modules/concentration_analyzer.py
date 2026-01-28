"""
Sector and Region Concentration Analyzer

Analyzes signal distribution for concentration risks and provides
diversification warnings when BUY signals are overly concentrated
in specific sectors or regions.

P1 improvement from HEDGE_FUND_REVIEW.md recommendations.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Default concentration thresholds
DEFAULT_MAX_SECTOR_CONCENTRATION = 0.40  # 40% in any single sector
DEFAULT_MAX_REGION_CONCENTRATION = 0.60  # 60% in any single region
DEFAULT_MIN_SIGNALS_FOR_WARNING = 3  # Need at least 3 signals to analyze


class ConcentrationWarning:
    """Represents a concentration warning."""

    def __init__(
        self,
        warning_type: str,  # "sector" or "region"
        concentrated_value: str,  # Sector or region name
        percentage: float,  # Concentration percentage
        count: int,  # Number of signals in this category
        total: int,  # Total signals
        tickers: List[str],  # Affected tickers
    ):
        self.warning_type = warning_type
        self.concentrated_value = concentrated_value
        self.percentage = percentage
        self.count = count
        self.total = total
        self.tickers = tickers

    def __str__(self) -> str:
        return (
            f"⚠️ {self.warning_type.upper()} CONCENTRATION: "
            f"{self.count}/{self.total} ({self.percentage:.0%}) of BUY signals "
            f"are in {self.concentrated_value}"
        )

    def to_dict(self) -> Dict:
        return {
            "warning_type": self.warning_type,
            "concentrated_value": self.concentrated_value,
            "percentage": self.percentage,
            "count": self.count,
            "total": self.total,
            "tickers": self.tickers,
        }


def analyze_concentration(
    df: pd.DataFrame,
    signal_column: str = "BS",
    target_signal: str = "B",
    max_sector_concentration: float = DEFAULT_MAX_SECTOR_CONCENTRATION,
    max_region_concentration: float = DEFAULT_MAX_REGION_CONCENTRATION,
    min_signals: int = DEFAULT_MIN_SIGNALS_FOR_WARNING,
) -> List[ConcentrationWarning]:
    """
    Analyze a DataFrame of signals for sector/region concentration.

    Args:
        df: DataFrame with signal data
        signal_column: Column containing signal values (B/S/H/I/V)
        target_signal: Signal type to analyze for concentration (default: "B" for BUY)
        max_sector_concentration: Maximum allowed percentage in any single sector
        max_region_concentration: Maximum allowed percentage in any single region
        min_signals: Minimum number of signals required to generate warnings

    Returns:
        List of ConcentrationWarning objects
    """
    warnings = []

    if df.empty or signal_column not in df.columns:
        return warnings

    # Filter for target signals only
    target_df = df[df[signal_column] == target_signal].copy()

    if len(target_df) < min_signals:
        logger.debug(
            f"Only {len(target_df)} {target_signal} signals, "
            f"minimum {min_signals} required for concentration analysis"
        )
        return warnings

    total_signals = len(target_df)

    # Analyze sector concentration
    sector_warnings = _analyze_sector_concentration(
        target_df, total_signals, max_sector_concentration
    )
    warnings.extend(sector_warnings)

    # Analyze region concentration
    region_warnings = _analyze_region_concentration(
        target_df, total_signals, max_region_concentration
    )
    warnings.extend(region_warnings)

    return warnings


def _analyze_sector_concentration(
    df: pd.DataFrame, total: int, max_concentration: float
) -> List[ConcentrationWarning]:
    """Analyze sector concentration."""
    warnings = []

    # Find sector column
    sector_col = None
    for col in ["sector", "SECTOR", "Sector"]:
        if col in df.columns:
            sector_col = col
            break

    if not sector_col:
        logger.debug("No sector column found for concentration analysis")
        return warnings

    # Get ticker column
    ticker_col = None
    for col in ["ticker", "TICKER", "TKR", "symbol"]:
        if col in df.columns:
            ticker_col = col
            break
    if not ticker_col and df.index.name and "ticker" in df.index.name.lower():
        df = df.reset_index()
        ticker_col = df.columns[0]

    # Count by sector
    sector_counts = df[sector_col].value_counts()

    for sector, count in sector_counts.items():
        if pd.isna(sector) or sector == "" or sector == "None":
            continue

        percentage = count / total
        if percentage > max_concentration:
            # Get tickers in this sector
            tickers = (
                df[df[sector_col] == sector][ticker_col].tolist()
                if ticker_col
                else []
            )
            warning = ConcentrationWarning(
                warning_type="sector",
                concentrated_value=str(sector),
                percentage=percentage,
                count=count,
                total=total,
                tickers=tickers,
            )
            warnings.append(warning)
            logger.warning(str(warning))

    return warnings


def _analyze_region_concentration(
    df: pd.DataFrame, total: int, max_concentration: float
) -> List[ConcentrationWarning]:
    """Analyze region concentration."""
    warnings = []

    # Find region column or infer from ticker
    region_col = None
    for col in ["region", "REGION", "Region"]:
        if col in df.columns:
            region_col = col
            break

    # If no region column, try to infer from ticker
    if not region_col:
        ticker_col = None
        for col in ["ticker", "TICKER", "TKR", "symbol"]:
            if col in df.columns:
                ticker_col = col
                break
        if not ticker_col and df.index.name and "ticker" in df.index.name.lower():
            ticker_col = df.index.name

        if ticker_col:
            # Infer region from ticker suffix
            df = df.copy()
            df["_inferred_region"] = df[ticker_col].apply(_infer_region_from_ticker)
            region_col = "_inferred_region"

    if not region_col or region_col not in df.columns:
        logger.debug("No region data available for concentration analysis")
        return warnings

    # Get ticker column for reporting
    ticker_col = None
    for col in ["ticker", "TICKER", "TKR", "symbol"]:
        if col in df.columns:
            ticker_col = col
            break

    # Count by region
    region_counts = df[region_col].value_counts()

    for region, count in region_counts.items():
        if pd.isna(region) or region == "" or region == "None":
            continue

        percentage = count / total
        if percentage > max_concentration:
            # Get tickers in this region
            tickers = (
                df[df[region_col] == region][ticker_col].tolist()
                if ticker_col
                else []
            )
            warning = ConcentrationWarning(
                warning_type="region",
                concentrated_value=str(region).upper(),
                percentage=percentage,
                count=count,
                total=total,
                tickers=tickers,
            )
            warnings.append(warning)
            logger.warning(str(warning))

    return warnings


def _infer_region_from_ticker(ticker: str) -> str:
    """Infer region from ticker suffix."""
    if not ticker:
        return "unknown"

    ticker = str(ticker).upper()

    # Hong Kong
    if ticker.endswith(".HK"):
        return "HK"

    # European markets
    eu_suffixes = [
        ".DE",
        ".L",
        ".PA",
        ".AS",
        ".MI",
        ".MC",
        ".SW",
        ".ST",
        ".OL",
        ".CO",
        ".HE",
        ".BR",
        ".VI",
    ]
    for suffix in eu_suffixes:
        if ticker.endswith(suffix):
            return "EU"

    # Default to US for unsuffixed tickers
    return "US"


def format_concentration_warnings(warnings: List[ConcentrationWarning]) -> str:
    """Format concentration warnings for display."""
    if not warnings:
        return ""

    lines = ["\n" + "=" * 60]
    lines.append("⚠️  CONCENTRATION WARNINGS")
    lines.append("=" * 60)

    for warning in warnings:
        lines.append(str(warning))
        if warning.tickers:
            lines.append(f"   Tickers: {', '.join(warning.tickers[:10])}")
            if len(warning.tickers) > 10:
                lines.append(f"   ... and {len(warning.tickers) - 10} more")

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def get_diversification_score(
    df: pd.DataFrame, signal_column: str = "BS", target_signal: str = "B"
) -> Tuple[float, str]:
    """
    Calculate a diversification score for signals.

    Returns:
        Tuple of (score 0-100, description)
    """
    if df.empty or signal_column not in df.columns:
        return 0.0, "No data"

    target_df = df[df[signal_column] == target_signal]
    if len(target_df) < 2:
        return 100.0, "Too few signals to assess"

    total = len(target_df)

    # Calculate sector diversity (using Herfindahl-Hirschman Index)
    sector_hhi = 0.0
    region_hhi = 0.0

    # Sector diversity
    for col in ["sector", "SECTOR", "Sector"]:
        if col in target_df.columns:
            sector_counts = target_df[col].value_counts()
            sector_shares = (sector_counts / total) ** 2
            sector_hhi = sector_shares.sum()
            break

    # Region diversity
    for col in ["region", "REGION", "Region"]:
        if col in target_df.columns:
            region_counts = target_df[col].value_counts()
            region_shares = (region_counts / total) ** 2
            region_hhi = region_shares.sum()
            break

    # HHI ranges from 1/n (perfectly diverse) to 1 (perfectly concentrated)
    # Convert to 0-100 score where 100 is most diverse
    avg_hhi = (sector_hhi + region_hhi) / 2 if sector_hhi and region_hhi else max(sector_hhi, region_hhi)

    # Score: 100 * (1 - HHI) gives 0 for perfect concentration, ~100 for diversity
    score = 100 * (1 - avg_hhi)

    if score >= 70:
        description = "Well diversified"
    elif score >= 50:
        description = "Moderately diversified"
    elif score >= 30:
        description = "Concentrated"
    else:
        description = "Highly concentrated"

    return score, description

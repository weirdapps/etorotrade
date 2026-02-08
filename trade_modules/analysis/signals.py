"""
Trading Signal Generation Module

This module contains the core signal generation logic for buy/sell/hold decisions.
Uses vectorized operations for performance on large datasets.

Enhanced with multi-factor weighted scoring system:
- SELL scoring with quality override protection
- BUY conviction scoring for candidate ranking
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Dict, Any
from datetime import datetime, timedelta

# Import tier utilities
from .tiers import _parse_percentage, _parse_market_cap

# Import from trade_modules
from trade_modules.trade_config import TradeConfig

# Get logger for this module
logger = logging.getLogger(__name__)


def is_recent_ipo(ticker: str, yaml_config, grace_period_months: int = 12) -> bool:
    """
    Check if a stock is a recent IPO within the grace period.

    Args:
        ticker: Stock ticker symbol
        yaml_config: YAML config loader instance
        grace_period_months: Number of months after IPO to consider "recent"

    Returns:
        True if the stock IPO'd within the grace period
    """
    config = yaml_config.load_config()
    ipo_config = config.get('ipo_grace_period', {})

    if not ipo_config.get('enabled', False):
        return False

    known_ipos = ipo_config.get('known_ipos', {})

    if ticker not in known_ipos:
        return False

    try:
        ipo_date_str = known_ipos[ticker]
        ipo_date = datetime.strptime(ipo_date_str, '%Y-%m-%d')
        grace_period = timedelta(days=grace_period_months * 30)  # Approximate months
        cutoff_date = datetime.now() - grace_period

        return ipo_date > cutoff_date
    except (ValueError, TypeError):
        return False


def calculate_sell_score(
    upside: float,
    buy_pct: float,
    exret: float,
    pct_52w: float,
    pef: float,
    pet: float,
    roe: float,
    de: float,
    sell_scoring_config: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """
    Calculate weighted SELL score (0-100) using multi-factor analysis.

    The score is calculated from four components:
    - Analyst sentiment (upside, buy%, EXRET) - 35% weight
    - Momentum (52W, PEF/PET deterioration) - 25% weight
    - Valuation (implicit in other factors) - 20% weight
    - Fundamentals (ROE, DE) - 20% weight

    Args:
        upside: Upside to price target (%)
        buy_pct: Percentage of analysts with buy rating
        exret: Expected return (upside * buy% / 100)
        pct_52w: Percentage from 52-week high
        pef: Forward P/E ratio
        pet: Trailing P/E ratio
        roe: Return on equity (%)
        de: Debt to equity ratio (%)
        sell_scoring_config: Configuration dict with weights and thresholds

    Returns:
        Tuple of (score 0-100, list of triggered factor descriptions)
    """
    factors: List[str] = []

    # Get weights from config (defaults if not specified)
    w_analyst = sell_scoring_config.get('weight_analyst', 0.35)
    w_momentum = sell_scoring_config.get('weight_momentum', 0.25)
    w_valuation = sell_scoring_config.get('weight_valuation', 0.20)
    w_fundamental = sell_scoring_config.get('weight_fundamental', 0.20)

    # === ANALYST SENTIMENT COMPONENT (0-100 raw score) ===
    analyst_score = 0.0

    # Upside penalty (0-40 points within analyst bucket)
    if upside <= -10:
        analyst_score += 40
        factors.append(f"severe_negative_upside:{upside:.1f}%")
    elif upside <= 0:
        analyst_score += 30
        factors.append(f"negative_upside:{upside:.1f}%")
    elif upside < 5:
        analyst_score += 20
        factors.append(f"low_upside:{upside:.1f}%")
    elif upside < 10:
        analyst_score += 10

    # Buy% penalty (0-40 points)
    if buy_pct < 30:
        analyst_score += 40
        factors.append(f"very_low_buy_pct:{buy_pct:.1f}%")
    elif buy_pct < 45:
        analyst_score += 30
        factors.append(f"low_buy_pct:{buy_pct:.1f}%")
    elif buy_pct < 55:
        analyst_score += 20
    elif buy_pct < 65:
        analyst_score += 10

    # EXRET penalty (0-20 points)
    if exret < 0:
        analyst_score += 20
        factors.append(f"negative_exret:{exret:.1f}")
    elif exret < 3:
        analyst_score += 15
        factors.append(f"very_low_exret:{exret:.1f}")
    elif exret < 5:
        analyst_score += 8

    # === MOMENTUM COMPONENT (0-100 raw score) ===
    momentum_score = 0.0

    # 52-week high penalty
    if not pd.isna(pct_52w):
        if pct_52w < 35:
            momentum_score += 50
            factors.append(f"severe_52w_decline:{pct_52w:.1f}%")
        elif pct_52w < 50:
            momentum_score += 35
            factors.append(f"significant_52w_decline:{pct_52w:.1f}%")
        elif pct_52w < 60:
            momentum_score += 20
            factors.append(f"52w_decline:{pct_52w:.1f}%")
        elif pct_52w < 70:
            momentum_score += 10

    # PEF >> PET (deteriorating earnings outlook)
    if not pd.isna(pef) and not pd.isna(pet) and pet > 10 and pef > 10:
        pef_pet_ratio = pef / pet
        if pef_pet_ratio > 1.4:
            momentum_score += 35
            factors.append(f"severe_earnings_deterioration:PEF/PET={pef_pet_ratio:.2f}")
        elif pef_pet_ratio > 1.3:
            momentum_score += 25
            factors.append(f"earnings_deterioration:PEF/PET={pef_pet_ratio:.2f}")
        elif pef_pet_ratio > 1.2:
            momentum_score += 15

    # === VALUATION COMPONENT (0-100 raw score) ===
    # Valuation signals are partially captured through PE checks
    valuation_score = 0.0

    # High PE penalty
    if not pd.isna(pef):
        if pef > 80:
            valuation_score += 40
            factors.append(f"extreme_pe:{pef:.1f}")
        elif pef > 60:
            valuation_score += 25
        elif pef > 45:
            valuation_score += 10

    # Negative PE (losses)
    if not pd.isna(pef) and pef < 0:
        valuation_score += 30
        factors.append("negative_earnings")
    elif not pd.isna(pet) and pet < 0:
        valuation_score += 25
        factors.append("trailing_losses")

    # === FUNDAMENTAL COMPONENT (0-100 raw score) ===
    fundamental_score = 0.0

    # Low ROE penalty
    if not pd.isna(roe):
        if roe < 0:
            fundamental_score += 40
            factors.append(f"negative_roe:{roe:.1f}%")
        elif roe < 5:
            fundamental_score += 25
            factors.append(f"very_low_roe:{roe:.1f}%")
        elif roe < 8:
            fundamental_score += 10

    # High D/E penalty
    if not pd.isna(de):
        if de > 300:
            fundamental_score += 40
            factors.append(f"extreme_leverage:{de:.1f}%")
        elif de > 200:
            fundamental_score += 25
            factors.append(f"high_leverage:{de:.1f}%")
        elif de > 150:
            fundamental_score += 15

    # === CALCULATE WEIGHTED TOTAL ===
    # Normalize each component to max 100, then apply weights
    analyst_normalized = min(analyst_score, 100)
    momentum_normalized = min(momentum_score, 85) / 85 * 100  # Max raw is ~85
    valuation_normalized = min(valuation_score, 70) / 70 * 100  # Max raw is ~70
    fundamental_normalized = min(fundamental_score, 80) / 80 * 100  # Max raw is ~80

    total_score = (
        analyst_normalized * w_analyst +
        momentum_normalized * w_momentum +
        valuation_normalized * w_valuation +
        fundamental_normalized * w_fundamental
    )

    return total_score, factors


def calculate_buy_score(
    upside: float,
    buy_pct: float,
    exret: float,
    pct_52w: float,
    above_200dma: bool,
    pef: float,
    pet: float,
    roe: float,
    de: float,
    fcf_yield: float,
    buy_scoring_config: Dict[str, Any],
) -> float:
    """
    Calculate BUY conviction score (0-100) for ranking candidates.

    Higher score = higher conviction in the BUY signal.

    Components:
    - Upside (higher upside = higher score) - 30% weight
    - Consensus (higher buy% = higher score) - 25% weight
    - Momentum (near 52W high, above 200DMA) - 20% weight
    - Valuation (reasonable PE) - 15% weight
    - Fundamentals (strong ROE, low DE, positive FCF) - 10% weight

    Args:
        upside: Upside to price target (%)
        buy_pct: Percentage of analysts with buy rating
        exret: Expected return
        pct_52w: Percentage from 52-week high
        above_200dma: Whether price is above 200-day MA
        pef: Forward P/E ratio
        pet: Trailing P/E ratio
        roe: Return on equity (%)
        de: Debt to equity ratio (%)
        fcf_yield: Free cash flow yield (%)
        buy_scoring_config: Configuration dict with weights

    Returns:
        Score 0-100 (higher = stronger BUY conviction)
    """
    # Get weights from config
    w_upside = buy_scoring_config.get('weight_upside', 0.30)
    w_consensus = buy_scoring_config.get('weight_consensus', 0.25)
    w_momentum = buy_scoring_config.get('weight_momentum', 0.20)
    w_valuation = buy_scoring_config.get('weight_valuation', 0.15)
    w_fundamental = buy_scoring_config.get('weight_fundamental', 0.10)

    # === UPSIDE COMPONENT (0-100) ===
    upside_score = 0.0
    if upside >= 50:
        upside_score = 100
    elif upside >= 40:
        upside_score = 90
    elif upside >= 30:
        upside_score = 80
    elif upside >= 25:
        upside_score = 70
    elif upside >= 20:
        upside_score = 60
    elif upside >= 15:
        upside_score = 50
    elif upside >= 12:
        upside_score = 40
    elif upside >= 10:
        upside_score = 30
    elif upside >= 8:
        upside_score = 20
    elif upside > 0:
        upside_score = 10

    # === CONSENSUS COMPONENT (0-100) ===
    consensus_score = 0.0
    if buy_pct >= 95:
        consensus_score = 100
    elif buy_pct >= 92:
        consensus_score = 95
    elif buy_pct >= 88:
        consensus_score = 85
    elif buy_pct >= 85:
        consensus_score = 75
    elif buy_pct >= 82:
        consensus_score = 65
    elif buy_pct >= 80:
        consensus_score = 55
    elif buy_pct >= 77:
        consensus_score = 45
    elif buy_pct >= 75:
        consensus_score = 35
    elif buy_pct >= 70:
        consensus_score = 25
    else:
        consensus_score = 10

    # === MOMENTUM COMPONENT (0-100) ===
    momentum_score = 0.0

    # 52-week position
    if not pd.isna(pct_52w):
        if pct_52w >= 95:
            momentum_score += 50
        elif pct_52w >= 90:
            momentum_score += 45
        elif pct_52w >= 85:
            momentum_score += 40
        elif pct_52w >= 80:
            momentum_score += 35
        elif pct_52w >= 75:
            momentum_score += 30
        elif pct_52w >= 70:
            momentum_score += 25
        elif pct_52w >= 65:
            momentum_score += 15
        else:
            momentum_score += 5

    # Above 200DMA bonus
    if above_200dma is True:
        momentum_score += 50
    elif above_200dma is False:
        momentum_score += 0  # Penalty already implied by not getting bonus
    else:
        momentum_score += 20  # Unknown - give partial score

    # === VALUATION COMPONENT (0-100) ===
    valuation_score = 50  # Start neutral

    # PEF < PET is good (improving earnings)
    if not pd.isna(pef) and not pd.isna(pet) and pet > 10 and pef > 10:
        pef_pet_ratio = pef / pet
        if pef_pet_ratio < 0.8:
            valuation_score = 100  # Strong earnings improvement
        elif pef_pet_ratio < 0.9:
            valuation_score = 85
        elif pef_pet_ratio < 1.0:
            valuation_score = 70
        elif pef_pet_ratio < 1.1:
            valuation_score = 55
        else:
            valuation_score = 30  # Deteriorating

    # Reasonable PE levels
    if not pd.isna(pef):
        if 10 <= pef <= 25:
            valuation_score = min(valuation_score + 20, 100)
        elif 25 < pef <= 40:
            valuation_score = min(valuation_score + 10, 100)

    # === FUNDAMENTAL COMPONENT (0-100) ===
    fundamental_score = 50  # Start neutral

    # ROE bonus
    if not pd.isna(roe):
        if roe >= 25:
            fundamental_score = 90
        elif roe >= 20:
            fundamental_score = 80
        elif roe >= 15:
            fundamental_score = 70
        elif roe >= 12:
            fundamental_score = 60
        elif roe >= 10:
            fundamental_score = 50
        elif roe >= 8:
            fundamental_score = 40
        else:
            fundamental_score = 20

    # D/E penalty (lower is better)
    if not pd.isna(de):
        if de < 30:
            fundamental_score = min(fundamental_score + 20, 100)
        elif de < 50:
            fundamental_score = min(fundamental_score + 10, 100)
        elif de > 150:
            fundamental_score = max(fundamental_score - 20, 0)
        elif de > 100:
            fundamental_score = max(fundamental_score - 10, 0)

    # FCF yield bonus
    if not pd.isna(fcf_yield):
        if fcf_yield > 5:
            fundamental_score = min(fundamental_score + 15, 100)
        elif fcf_yield > 2:
            fundamental_score = min(fundamental_score + 10, 100)
        elif fcf_yield > 0:
            fundamental_score = min(fundamental_score + 5, 100)

    # === CALCULATE WEIGHTED TOTAL ===
    total_score = (
        upside_score * w_upside +
        consensus_score * w_consensus +
        momentum_score * w_momentum +
        valuation_score * w_valuation +
        fundamental_score * w_fundamental
    )

    return total_score


def calculate_action_vectorized(df: pd.DataFrame, option: str = "portfolio") -> pd.Series:
    """Vectorized calculation of trading actions for improved performance.

    Uses new 5-tier geographic system:
    - MEGA (≥$500B), LARGE ($100-500B), MID ($10-100B), SMALL ($2-10B), MICRO (<$2B)
    - Regions: US, EU, HK based on ticker suffix

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial metrics

    Returns
    -------
    pd.Series
        Series with action values (B/S/H/I)
    """
    # Remove duplicate index values to prevent ambiguous .loc[] lookups
    # Keep first occurrence of each ticker
    if df.index.duplicated().any():
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate tickers, keeping first occurrence")
        df = df[~df.index.duplicated(keep='first')]

    # Initialize config and YAML loader
    config = TradeConfig()
    from trade_modules.yaml_config_loader import get_yaml_config
    yaml_config = get_yaml_config()

    # Use YAML config thresholds if available, otherwise fallback to defaults
    universal_thresholds = config.get_universal_thresholds()

    # Parse percentage columns that may contain strings like "2.6%" or "94%"
    # Handle both normalized and CSV column names (including short display names)
    upside_raw = df.get("upside", df.get("UPSIDE", df.get("UP%", pd.Series([0] * len(df), index=df.index))))
    upside = pd.Series([_parse_percentage(val) for val in upside_raw], index=df.index)

    buy_pct_raw = df.get("buy_percentage", df.get("%BUY", df.get("%B", pd.Series([0] * len(df), index=df.index))))
    buy_pct = pd.Series([_parse_percentage(val) for val in buy_pct_raw], index=df.index)

    # Handle both raw CSV column names and normalized column names
    # NOTE: #T = target count (price targets), #A = analyst count
    analyst_count_raw = df.get("analyst_count", df.get("#A", df.get("# A", pd.Series([0] * len(df), index=df.index))))
    analyst_count = pd.to_numeric(analyst_count_raw, errors="coerce").fillna(0)

    # total_ratings = number of price targets (#T)
    total_ratings_raw = df.get("total_ratings", df.get("#T", df.get("# T", df.get("target_count", pd.Series([0] * len(df), index=df.index)))))
    total_ratings = pd.to_numeric(total_ratings_raw, errors="coerce").fillna(0)

    # Get market cap and parse formatted strings (e.g., "2.47T", "628B")
    # Need this before confidence check for tiered analyst requirements
    cap_raw = df.get("market_cap", df.get("CAP", pd.Series([0] * len(df), index=df.index)))
    cap_values = pd.Series([_parse_market_cap(cap) for cap in cap_raw], index=df.index)

    # Tiered analyst requirements based on market cap
    # $2-5B (small cap): requires more analysts (institutional interest signal)
    # $5B+: standard analyst requirement
    small_cap_threshold = universal_thresholds.get("small_cap_threshold", 5_000_000_000)
    small_cap_min_analysts = universal_thresholds.get("small_cap_min_analysts", 6)
    standard_min_analysts = universal_thresholds.get("min_analyst_count", 4)
    min_price_targets = universal_thresholds.get("min_price_targets", 4)

    # Market cap gate: hard floor at $2B
    min_market_cap = universal_thresholds.get("min_market_cap", 2_000_000_000)
    above_min_cap = cap_values >= min_market_cap

    # Apply tiered analyst requirement: small caps need more coverage
    is_small_cap = cap_values < small_cap_threshold
    required_analysts = pd.Series(standard_min_analysts, index=df.index)
    required_analysts[is_small_cap] = small_cap_min_analysts

    # Confidence check - vectorized with tiered requirements AND market cap gate
    has_confidence = above_min_cap & (analyst_count >= required_analysts) & (total_ratings >= min_price_targets)

    # Additional SELL/BUY criteria for stocks with data
    # Ensure we create pandas Series with proper index alignment
    pef = pd.to_numeric(
        df.get("pe_forward", df.get("PEF", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    pet = pd.to_numeric(
        df.get("pe_trailing", df.get("PET", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    peg = pd.to_numeric(
        df.get("peg_ratio", df.get("PEG", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    si = pd.to_numeric(
        df.get("short_percent", df.get("SI", pd.Series([np.nan] * len(df), index=df.index))),
        errors="coerce",
    )
    beta = pd.to_numeric(
        df.get("beta", df.get("BETA", df.get("B", pd.Series([np.nan] * len(df), index=df.index)))), errors="coerce"
    )
    # Try EXRET, then EXR (short display name), else calculate from upside and buy%
    exret_col = df.get("EXRET", df.get("EXR"))
    if exret_col is None:
        # Calculate EXRET from upside and buy_percentage if not provided
        # EXRET = upside × (buy_percentage / 100)
        exret = upside * (buy_pct / 100.0)
        logger.debug("EXRET column not found, calculated from upside and buy_percentage")
    else:
        exret = pd.Series([_parse_percentage(val) for val in exret_col], index=df.index)

    # Parse ROE and DE columns - handle missing data gracefully
    # ROE comes from provider already in percentage format (e.g., 109.4 for 109.4%)
    roe_raw = df.get("return_on_equity", df.get("ROE", pd.Series([np.nan] * len(df), index=df.index)))
    # Replace "--" and empty strings with NaN before converting to numeric
    roe_clean = roe_raw.replace(["--", "", " ", "nan"], np.nan)
    roe = pd.to_numeric(roe_clean, errors="coerce").fillna(np.nan)

    # DE comes from provider already in percentage format
    de_raw = df.get("debt_to_equity", df.get("DE", pd.Series([np.nan] * len(df), index=df.index)))
    # Replace "--" and empty strings with NaN before converting to numeric
    de_clean = de_raw.replace(["--", "", " ", "nan"], np.nan)
    de = pd.to_numeric(de_clean, errors="coerce").fillna(np.nan)

    # NEW MOMENTUM METRICS
    # Price Momentum: Percent from 52-week high
    # Fallback to "52W" (short display name from table_renderer)
    pct_52w_raw = df.get("pct_from_52w_high", df.get("52W", pd.Series([np.nan] * len(df), index=df.index)))
    pct_52w = pd.to_numeric(pct_52w_raw, errors="coerce").fillna(np.nan)

    # Price Momentum: Above 200-day moving average (boolean)
    # Fallback to "2H" (short display name from table_renderer - "200DMA" abbreviated)
    above_200dma_raw = df.get("above_200dma", df.get("2H", pd.Series([np.nan] * len(df), index=df.index)))
    # Handle boolean-like values
    above_200dma = above_200dma_raw.map(lambda x: True if x in [True, "True", "true", 1, "1", "Y", "y"] else
                                        (False if x in [False, "False", "false", 0, "0", "N", "n"] else np.nan))

    # Analyst Momentum: 3-month change in buy percentage
    # Fallback to "AM" (short display name from table_renderer)
    amom_raw = df.get("analyst_momentum", df.get("AM", pd.Series([np.nan] * len(df), index=df.index)))
    amom = pd.to_numeric(amom_raw, errors="coerce").fillna(np.nan)

    # Sector-Relative Valuation: PE vs sector median
    # NOTE: "P/S" in output is price-to-sales ratio, NOT pe_vs_sector - do not fallback to it
    pe_vs_sector_raw = df.get("pe_vs_sector", pd.Series([np.nan] * len(df), index=df.index))
    pe_vs_sector = pd.to_numeric(pe_vs_sector_raw, errors="coerce").fillna(np.nan)

    # NEW: FCF Yield - academically proven alpha factor (Sloan 1996, Lakonishok 1994)
    fcf_yield_raw = df.get("fcf_yield", df.get("FCF", pd.Series([np.nan] * len(df), index=df.index)))
    fcf_yield = pd.to_numeric(fcf_yield_raw, errors="coerce").fillna(np.nan)

    # NEW: Revenue Growth - harder to manipulate than EPS
    rev_growth_raw = df.get("revenue_growth", df.get("RG", pd.Series([np.nan] * len(df), index=df.index)))
    rev_growth = pd.to_numeric(rev_growth_raw, errors="coerce").fillna(np.nan)

    # Initialize action series
    actions = pd.Series("H", index=df.index)  # Default to HOLD
    actions[~has_confidence] = "I"  # INCONCLUSIVE for low confidence

    # Initialize BUY conviction scores (0-100, NaN for non-BUY)
    buy_scores = pd.Series(np.nan, index=df.index)

    # Log market cap and tiered analyst gate stats
    below_min_cap_count = (~above_min_cap).sum()
    if below_min_cap_count > 0:
        logger.info(
            f"Market cap gate: {below_min_cap_count} stocks below ${min_market_cap / 1e9:.1f}B "
            f"hard floor excluded"
        )

    small_cap_count = (is_small_cap & above_min_cap).sum()
    if small_cap_count > 0:
        logger.debug(
            f"Tiered analyst gate: {small_cap_count} stocks in $2-5B range require "
            f"{small_cap_min_analysts}+ analysts (vs {standard_min_analysts} for larger caps)"
        )

    # Get ticker column for region detection
    # Check if TICKER/TKR is the index (from CSV with index_col=0) or a column
    index_name = df.index.name.upper() if hasattr(df.index, 'name') and df.index.name else ""
    if index_name in ("TICKER", "TKR") or "ticker" in index_name.lower():
        ticker_col = pd.Series(df.index, index=df.index)
    else:
        # Check for various ticker column names: ticker, TICKER, TKR, symbol
        ticker_col = df.get("ticker", df.get("TICKER", df.get("TKR", df.get("symbol", pd.Series([""] * len(df), index=df.index)))))

    # Import asset type classification
    try:
        from yahoofinance.utils.data.asset_type_utils import classify_asset_type
    except ImportError:
        classify_asset_type = None

    # Get company name column for asset classification
    company_col = df.get("company_name", df.get("NAME", df.get("name", pd.Series([None] * len(df), index=df.index))))

    # Process each row with region-tier specific thresholds
    for idx in df.index:
        # Determine ticker first (needed for asset type check)
        ticker = str(ticker_col.loc[idx]) if not pd.isna(ticker_col.loc[idx]) else ""

        # Check asset type and use momentum-based signals for non-equity assets
        if classify_asset_type:
            company_name = company_col.loc[idx] if idx in company_col.index and not pd.isna(company_col.loc[idx]) else None
            asset_type = classify_asset_type(ticker, cap_values.loc[idx], company_name)

            # For bitcoin proxy stocks, use hybrid momentum + analyst scoring
            # Traditional analyst metrics may not apply well to BTC-correlated stocks
            if asset_type == "bitcoin_proxy":
                row_pct_52w = pct_52w.loc[idx]
                row_above_200dma = above_200dma.loc[idx]
                row_upside = upside.loc[idx]
                row_buy_pct = buy_pct.loc[idx]
                row_exret = exret.loc[idx]

                # Get configurable thresholds for bitcoin proxy
                btc_proxy_config = yaml_config.load_config().get('bitcoin_proxy', {})
                buy_momentum_threshold = btc_proxy_config.get('momentum_buy_threshold', 70)
                sell_momentum_threshold = btc_proxy_config.get('momentum_sell_threshold', 35)
                min_buy_pct = btc_proxy_config.get('min_buy_pct_override', 60)
                require_200dma = btc_proxy_config.get('require_above_200dma', True)

                # Hybrid logic: combine momentum with relaxed analyst thresholds
                # Bitcoin proxies are highly volatile, so standard thresholds don't apply
                momentum_ok = not pd.isna(row_pct_52w) and row_pct_52w >= 50  # Minimum momentum
                analyst_ok = row_buy_pct >= min_buy_pct and row_upside >= 0  # Relaxed thresholds

                if pd.isna(row_pct_52w):
                    # No momentum data - fall through to standard equity evaluation
                    logger.info(f"Ticker {ticker}: BITCOIN_PROXY has no momentum data, using equity eval")
                    pass  # Will continue to standard evaluation
                elif momentum_ok and analyst_ok:
                    # Both momentum and analyst sentiment positive
                    meets_200dma = row_above_200dma is True or not require_200dma
                    if meets_200dma and row_pct_52w >= buy_momentum_threshold:
                        actions.loc[idx] = "B"
                        logger.info(f"Ticker {ticker}: BITCOIN_PROXY marked BUY - momentum {row_pct_52w:.1f}% >= {buy_momentum_threshold}%, buy% {row_buy_pct:.1f}%")
                    else:
                        actions.loc[idx] = "H"
                        logger.info(f"Ticker {ticker}: BITCOIN_PROXY marked HOLD - momentum/analyst mixed")

                    # Log signal
                    try:
                        from trade_modules.signal_tracker import log_signal
                        log_signal(
                            ticker=ticker,
                            signal=actions.loc[idx],
                            upside=row_upside,
                            buy_pct=row_buy_pct,
                            exret=row_exret,
                            market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                            tier=config.get_tier_from_market_cap(cap_values.loc[idx]),
                            region=config.get_region_from_ticker(ticker),
                            pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                            sell_triggers=[],
                        )
                    except (ImportError, Exception):
                        pass
                    continue
                elif row_pct_52w <= sell_momentum_threshold:
                    # Severe momentum decline - SELL regardless of analyst sentiment
                    actions.loc[idx] = "S"
                    logger.info(f"Ticker {ticker}: BITCOIN_PROXY marked SELL - momentum {row_pct_52w:.1f}% <= {sell_momentum_threshold}%")

                    # Log signal
                    try:
                        from trade_modules.signal_tracker import log_signal
                        log_signal(
                            ticker=ticker,
                            signal="S",
                            upside=row_upside,
                            buy_pct=row_buy_pct,
                            exret=row_exret,
                            market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                            tier=config.get_tier_from_market_cap(cap_values.loc[idx]),
                            region=config.get_region_from_ticker(ticker),
                            pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                            sell_triggers=["bitcoin_proxy_momentum_crash"],
                        )
                    except (ImportError, Exception):
                        pass
                    continue
                else:
                    # Moderate momentum - HOLD
                    actions.loc[idx] = "H"
                    logger.info(f"Ticker {ticker}: BITCOIN_PROXY marked HOLD - moderate conditions")
                    continue

            # For non-equity assets (crypto, ETF, commodity), use momentum-based signals
            # since analyst coverage is not applicable
            if asset_type in ("crypto", "etf", "commodity"):
                # Use price momentum (52-week high %) and technical indicators
                row_pct_52w = pct_52w.loc[idx]
                row_above_200dma = above_200dma.loc[idx]

                # Get configurable thresholds for crypto
                crypto_config = yaml_config.load_config().get('crypto_momentum', {})
                major_crypto_tickers = crypto_config.get('major', {}).get('tickers', ['BTC-USD', 'ETH-USD'])

                # Use different thresholds for major crypto vs altcoins
                if ticker in major_crypto_tickers:
                    buy_threshold = crypto_config.get('major', {}).get('buy_threshold', 85)
                    sell_threshold = crypto_config.get('major', {}).get('hold_threshold', 60)
                else:
                    buy_threshold = crypto_config.get('altcoins', {}).get('buy_threshold', 85)
                    sell_threshold = crypto_config.get('altcoins', {}).get('hold_threshold', 55)

                # Momentum-based signal logic for non-equity assets
                if pd.isna(row_pct_52w):
                    actions.loc[idx] = "I"  # INCONCLUSIVE if no momentum data
                    logger.info(f"Ticker {ticker}: {asset_type.upper()} marked INCONCLUSIVE - no momentum data")
                elif row_pct_52w >= buy_threshold:
                    # Strong momentum - near 52-week high
                    if row_above_200dma is True:
                        actions.loc[idx] = "B"  # BUY for strong uptrend
                        logger.info(f"Ticker {ticker}: {asset_type.upper()} marked BUY - strong momentum ({row_pct_52w:.1f}% >= {buy_threshold}%, above 200DMA)")
                    else:
                        actions.loc[idx] = "H"  # HOLD if not above 200DMA
                        logger.info(f"Ticker {ticker}: {asset_type.upper()} marked HOLD - good momentum but below 200DMA")
                elif row_pct_52w <= sell_threshold:
                    # Weak momentum - significantly below 52-week high
                    actions.loc[idx] = "S"  # SELL for weak momentum
                    logger.info(f"Ticker {ticker}: {asset_type.upper()} marked SELL - weak momentum ({row_pct_52w:.1f}% <= {sell_threshold}%)")
                else:
                    actions.loc[idx] = "H"  # HOLD for neutral momentum
                    logger.info(f"Ticker {ticker}: {asset_type.upper()} marked HOLD - neutral momentum ({row_pct_52w:.1f}%)")

                # Log signal for forward validation
                try:
                    from trade_modules.signal_tracker import log_signal
                    log_signal(
                        ticker=ticker,
                        signal=actions.loc[idx],
                        upside=upside.loc[idx] if idx in upside.index else None,
                        buy_pct=buy_pct.loc[idx] if idx in buy_pct.index else None,
                        exret=exret.loc[idx] if idx in exret.index else None,
                        market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                        tier=config.get_tier_from_market_cap(cap_values.loc[idx]),
                        region=config.get_region_from_ticker(ticker),
                        pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                        sell_triggers=[f"momentum_based_{asset_type}"] if actions.loc[idx] == "S" else [],
                    )
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"Failed to log {asset_type} signal for {ticker}: {e}")
                continue  # Skip analyst-based evaluation for non-equity assets

        if not has_confidence.loc[idx]:
            continue  # Already set to "I"

        # Determine region and tier for this stock
        region = config.get_region_from_ticker(ticker)
        tier = config.get_tier_from_market_cap(cap_values.loc[idx])

        # Get region-tier specific thresholds from YAML
        if yaml_config.is_config_available():
            criteria = yaml_config.get_region_tier_criteria(region, tier)
            buy_criteria = criteria.get("buy", {})
            sell_criteria = criteria.get("sell", {})
            logger.debug(f"Ticker {ticker}: region={region}, tier={tier}, sell_criteria keys={list(sell_criteria.keys())}")
        else:
            # Fallback to old system if YAML not available
            buy_criteria = config.get_tier_thresholds(tier, "buy")
            sell_criteria = config.get_tier_thresholds(tier, "sell")
            logger.warning(f"Ticker {ticker}: YAML config not available, using fallback")

        # Apply sector-specific ROE/DE threshold adjustments
        buy_criteria = config.get_sector_adjusted_thresholds(ticker, "buy", buy_criteria)
        sell_criteria = config.get_sector_adjusted_thresholds(ticker, "sell", sell_criteria)

        # Apply VIX regime adjustments (P1 improvement)
        # More conservative in high volatility, more aggressive in low volatility
        try:
            from trade_modules.vix_regime_provider import adjust_buy_criteria, adjust_sell_criteria
            buy_criteria = adjust_buy_criteria(buy_criteria)
            sell_criteria = adjust_sell_criteria(sell_criteria)
        except ImportError:
            pass  # VIX regime provider not available, use unadjusted criteria

        # Check for IPO grace period - recent IPOs get relaxed momentum criteria
        ipo_config = yaml_config.load_config().get('ipo_grace_period', {})
        is_ipo_stock = is_recent_ipo(ticker, yaml_config, ipo_config.get('grace_period_months', 12))

        if is_ipo_stock:
            # Apply relaxed thresholds for recent IPOs
            relaxed_momentum = ipo_config.get('relaxed_momentum_threshold', 40)
            ignore_analyst_momentum = ipo_config.get('ignore_analyst_momentum', True)

            # Relax momentum thresholds in sell criteria
            if 'max_pct_from_52w_high' in sell_criteria:
                original_threshold = sell_criteria['max_pct_from_52w_high']
                sell_criteria['max_pct_from_52w_high'] = min(relaxed_momentum, original_threshold)
                logger.info(f"Ticker {ticker}: IPO GRACE PERIOD - relaxing 52W threshold from {original_threshold}% to {sell_criteria['max_pct_from_52w_high']}%")

            # Ignore analyst momentum for recent IPOs (volatile and unreliable)
            if ignore_analyst_momentum and 'max_analyst_momentum' in sell_criteria:
                del sell_criteria['max_analyst_momentum']
                logger.info(f"Ticker {ticker}: IPO GRACE PERIOD - ignoring analyst momentum trigger")

        # Extract values for this row
        row_upside = upside.loc[idx]
        row_buy_pct = buy_pct.loc[idx]
        row_exret = exret.loc[idx]
        row_pef = pef.loc[idx]
        row_pet = pet.loc[idx]
        row_peg = peg.loc[idx]
        row_si = si.loc[idx]
        row_beta = beta.loc[idx]
        row_roe = roe.loc[idx]
        row_de = de.loc[idx]
        # New momentum metrics
        row_pct_52w = pct_52w.loc[idx]
        row_above_200dma = above_200dma.loc[idx]
        row_amom = amom.loc[idx]
        row_pe_vs_sector = pe_vs_sector.loc[idx]
        # NEW: FCF and Revenue Growth
        row_fcf_yield = fcf_yield.loc[idx]
        row_rev_growth = rev_growth.loc[idx]

        # Initialize for signal tracking
        sell_conditions: List[str] = []
        sell_score = 0.0

        logger.debug(f"Ticker {ticker}: SELL CHECK START - upside={row_upside:.1f}%, buy%={row_buy_pct:.1f}%, exret={row_exret:.1f}%, roe={row_roe}, de={row_de}")

        # FULLY VALUED check (P0 improvement - GOOG false positive fix)
        # Stocks at fair value with high analyst consensus should not be SELL
        # Conditions: upside within ±3%, buy% > 80%
        fully_valued_upside_threshold = sell_criteria.get("fully_valued_upside_threshold", 3.0)
        fully_valued_buy_pct_threshold = sell_criteria.get("fully_valued_buy_pct_threshold", 80.0)

        is_fully_valued = (
            abs(row_upside) <= fully_valued_upside_threshold
            and row_buy_pct >= fully_valued_buy_pct_threshold
        )

        if is_fully_valued:
            logger.info(f"Ticker {ticker}: FULLY VALUED (HOLD) - upside {row_upside:.1f}% (within ±{fully_valued_upside_threshold}%), buy% {row_buy_pct:.1f}% >= {fully_valued_buy_pct_threshold}%")
            actions.loc[idx] = "H"  # Fully Valued stocks are HOLD (skip SELL evaluation)
            # Log signal for forward validation
            try:
                from trade_modules.signal_tracker import log_signal
                price_raw = df.get("price", df.get("PRICE", pd.Series([None] * len(df), index=df.index)))
                row_price = price_raw.loc[idx] if idx in price_raw.index else None
                target_raw = df.get("target_price", df.get("TARGET", pd.Series([None] * len(df), index=df.index)))
                row_target = target_raw.loc[idx] if idx in target_raw.index else None
                sector_raw = df.get("sector", df.get("SECTOR", pd.Series([None] * len(df), index=df.index)))
                row_sector = sector_raw.loc[idx] if idx in sector_raw.index else None

                log_signal(
                    ticker=ticker,
                    signal="H",  # Fully valued stocks logged as HOLD
                    price=float(row_price) if row_price and not pd.isna(row_price) else None,
                    target=float(row_target) if row_target and not pd.isna(row_target) else None,
                    upside=row_upside,
                    buy_pct=row_buy_pct,
                    exret=row_exret,
                    market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                    tier=tier,
                    region=region,
                    sector=str(row_sector) if row_sector and not pd.isna(row_sector) else None,
                    pe_forward=float(row_pef) if not pd.isna(row_pef) else None,
                    pe_trailing=float(row_pet) if not pd.isna(row_pet) else None,
                    peg=float(row_peg) if not pd.isna(row_peg) else None,
                    short_interest=float(row_si) if not pd.isna(row_si) else None,
                    roe=float(row_roe) if not pd.isna(row_roe) else None,
                    debt_equity=float(row_de) if not pd.isna(row_de) else None,
                    pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                )
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Failed to log FULLY VALUED signal for {ticker}: {e}")
            continue  # Skip SELL/BUY evaluation for fully valued stocks

        # ================================================================
        # ENHANCED SELL SIGNAL FRAMEWORK
        # Uses multi-factor weighted scoring OR legacy ANY-trigger logic
        # ================================================================

        # Check if enhanced signal scoring is enabled
        use_scoring = yaml_config.is_signal_scoring_enabled()

        if use_scoring:
            # Get sell scoring config for this region-tier
            sell_scoring_config = yaml_config.get_sell_scoring_config(region, tier)

            # === QUALITY OVERRIDE CHECK ===
            # Never SELL stocks with exceptionally strong fundamentals
            quality_override_buy_pct = sell_scoring_config.get('quality_override_buy_pct', 85)
            quality_override_upside = sell_scoring_config.get('quality_override_upside', 20)
            quality_override_exret = sell_scoring_config.get('quality_override_exret', 15)

            is_quality_stock = (
                row_buy_pct >= quality_override_buy_pct and
                row_upside >= quality_override_upside and
                row_exret >= quality_override_exret
            )

            if is_quality_stock:
                logger.info(
                    f"Ticker {ticker}: QUALITY OVERRIDE - strong fundamentals protect from SELL "
                    f"(buy%={row_buy_pct:.1f}% >= {quality_override_buy_pct}%, "
                    f"upside={row_upside:.1f}% >= {quality_override_upside}%, "
                    f"exret={row_exret:.1f}% >= {quality_override_exret}%)"
                )
                # Skip SELL evaluation entirely for quality stocks
                # They will be evaluated for BUY criteria below
                pass  # Fall through to BUY evaluation
            else:
                # === HARD TRIGGERS (Immediate SELL, bypass scoring) ===
                hard_trigger_upside = sell_scoring_config.get('hard_trigger_upside', -5)
                hard_trigger_buy_pct = sell_scoring_config.get('hard_trigger_buy_pct', 35)

                # Hard sell conditions:
                # 1. Negative upside (stock at/below target) with weak-to-moderate sentiment
                # 2. Very low buy% (most analysts bearish) regardless of upside
                # 3. Severe negative upside (-10%+) with moderate sentiment
                is_hard_sell = (
                    (row_upside <= hard_trigger_upside and row_buy_pct <= 55)  # Negative upside + weak/moderate sentiment
                    or (row_buy_pct <= hard_trigger_buy_pct)  # Very low analyst buy%
                    or (row_upside <= -10 and row_buy_pct < 65)  # Severe negative upside (-10%+) with moderate sentiment
                )

                if is_hard_sell:
                    sell_conditions.append("hard_trigger")
                    if row_upside <= hard_trigger_upside:
                        sell_conditions.append(f"severe_negative_upside:{row_upside:.1f}%")
                    if row_buy_pct <= hard_trigger_buy_pct:
                        sell_conditions.append(f"very_low_buy_pct:{row_buy_pct:.1f}%")

                    logger.info(f"Ticker {ticker}: HARD SELL TRIGGER - upside={row_upside:.1f}%, buy%={row_buy_pct:.1f}%")
                    actions.loc[idx] = "S"

                # === SOFT TRIGGERS (Combined condition - SELL when BOTH conditions met) ===
                # Negative upside + weak sentiment = SELL (even if not extreme)
                soft_trigger_upside = sell_scoring_config.get('soft_trigger_upside', 0)
                soft_trigger_buy_pct = sell_scoring_config.get('soft_trigger_buy_pct', 50)

                is_soft_sell = (
                    row_upside < soft_trigger_upside and  # Negative upside (below target)
                    row_buy_pct < soft_trigger_buy_pct    # Less than 50% buy consensus
                )

                if is_soft_sell and actions.loc[idx] != "S":  # Don't override hard trigger
                    sell_conditions.append("soft_trigger")
                    sell_conditions.append(f"negative_upside_weak_sentiment:{row_upside:.1f}%/{row_buy_pct:.1f}%")

                    logger.info(f"Ticker {ticker}: SOFT SELL TRIGGER - upside={row_upside:.1f}% < 0% AND buy%={row_buy_pct:.1f}% < {soft_trigger_buy_pct}%")
                    actions.loc[idx] = "S"

                    # Log SELL signal
                    try:
                        from trade_modules.signal_tracker import log_signal
                        price_raw = df.get("price", df.get("PRICE", pd.Series([None] * len(df), index=df.index)))
                        row_price = price_raw.loc[idx] if idx in price_raw.index else None
                        target_raw = df.get("target_price", df.get("TARGET", pd.Series([None] * len(df), index=df.index)))
                        row_target = target_raw.loc[idx] if idx in target_raw.index else None
                        sector_raw = df.get("sector", df.get("SECTOR", pd.Series([None] * len(df), index=df.index)))
                        row_sector = sector_raw.loc[idx] if idx in sector_raw.index else None

                        log_signal(
                            ticker=ticker,
                            signal="S",
                            price=float(row_price) if row_price and not pd.isna(row_price) else None,
                            target=float(row_target) if row_target and not pd.isna(row_target) else None,
                            upside=row_upside,
                            buy_pct=row_buy_pct,
                            exret=row_exret,
                            market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                            tier=tier,
                            region=region,
                            sector=str(row_sector) if row_sector and not pd.isna(row_sector) else None,
                            pe_forward=float(row_pef) if not pd.isna(row_pef) else None,
                            pe_trailing=float(row_pet) if not pd.isna(row_pet) else None,
                            peg=float(row_peg) if not pd.isna(row_peg) else None,
                            short_interest=float(row_si) if not pd.isna(row_si) else None,
                            roe=float(row_roe) if not pd.isna(row_roe) else None,
                            debt_equity=float(row_de) if not pd.isna(row_de) else None,
                            pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                            sell_triggers=list(str(c) for c in sell_conditions),
                        )
                    except ImportError:
                        pass
                    except Exception as e:
                        logger.debug(f"Failed to log SOFT SELL signal for {ticker}: {e}")
                    continue

                # === MULTI-FACTOR SCORING ===
                sell_score, sell_factors = calculate_sell_score(
                    upside=row_upside,
                    buy_pct=row_buy_pct,
                    exret=row_exret,
                    pct_52w=row_pct_52w,
                    pef=row_pef,
                    pet=row_pet,
                    roe=row_roe,
                    de=row_de,
                    sell_scoring_config=sell_scoring_config,
                )

                score_threshold = sell_scoring_config.get('score_threshold', 65)

                if sell_score >= score_threshold:
                    sell_conditions = sell_factors
                    logger.info(
                        f"Ticker {ticker}: SCORED SELL (score={sell_score:.1f} >= {score_threshold}) - "
                        f"factors: {', '.join(sell_factors[:5])}"  # Log first 5 factors
                    )
                    actions.loc[idx] = "S"

                    # Log SELL signal
                    try:
                        from trade_modules.signal_tracker import log_signal
                        price_raw = df.get("price", df.get("PRICE", pd.Series([None] * len(df), index=df.index)))
                        row_price = price_raw.loc[idx] if idx in price_raw.index else None
                        target_raw = df.get("target_price", df.get("TARGET", pd.Series([None] * len(df), index=df.index)))
                        row_target = target_raw.loc[idx] if idx in target_raw.index else None
                        sector_raw = df.get("sector", df.get("SECTOR", pd.Series([None] * len(df), index=df.index)))
                        row_sector = sector_raw.loc[idx] if idx in sector_raw.index else None

                        log_signal(
                            ticker=ticker,
                            signal="S",
                            price=float(row_price) if row_price and not pd.isna(row_price) else None,
                            target=float(row_target) if row_target and not pd.isna(row_target) else None,
                            upside=row_upside,
                            buy_pct=row_buy_pct,
                            exret=row_exret,
                            market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                            tier=tier,
                            region=region,
                            sector=str(row_sector) if row_sector and not pd.isna(row_sector) else None,
                            pe_forward=float(row_pef) if not pd.isna(row_pef) else None,
                            pe_trailing=float(row_pet) if not pd.isna(row_pet) else None,
                            peg=float(row_peg) if not pd.isna(row_peg) else None,
                            short_interest=float(row_si) if not pd.isna(row_si) else None,
                            roe=float(row_roe) if not pd.isna(row_roe) else None,
                            debt_equity=float(row_de) if not pd.isna(row_de) else None,
                            pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                            sell_triggers=list(str(c) for c in sell_conditions),
                        )
                    except ImportError:
                        pass
                    except Exception as e:
                        logger.debug(f"Failed to log SCORED SELL signal for {ticker}: {e}")
                    continue

                elif sell_score >= 50:
                    # Weak sell - log for monitoring but don't mark as SELL
                    logger.debug(
                        f"Ticker {ticker}: WEAK_SELL (score={sell_score:.1f}, threshold={score_threshold}) - monitoring"
                    )
                    # Stays as HOLD unless BUY criteria met

        else:
            # ================================================================
            # LEGACY ANY-TRIGGER SELL LOGIC (when scoring disabled)
            # ================================================================

            # Basic criteria from config - only apply if explicitly defined in YAML
            # NO DEFAULTS - criteria must be explicitly configured to avoid false positives
            if "max_upside" in sell_criteria:
                if row_upside <= sell_criteria["max_upside"]:
                    sell_conditions.append("max_upside")
                    logger.info(f"Ticker {ticker}: SELL TRIGGER - upside {row_upside:.1f}% <= {sell_criteria['max_upside']:.1f}%")

            if "min_buy_percentage" in sell_criteria:
                if row_buy_pct <= sell_criteria["min_buy_percentage"]:
                    sell_conditions.append("min_buy_percentage")
                    logger.info(f"Ticker {ticker}: SELL TRIGGER - buy% {row_buy_pct:.1f}% <= {sell_criteria['min_buy_percentage']:.1f}%")

            if "max_exret" in sell_criteria:
                if row_exret <= sell_criteria["max_exret"]:
                    sell_conditions.append("max_exret")
                    logger.info(f"Ticker {ticker}: SELL TRIGGER - exret {row_exret:.1f}% <= {sell_criteria['max_exret']:.1f}%")

            # Optional criteria from YAML (only apply if defined in YAML)
            if "max_forward_pe" in sell_criteria and not pd.isna(row_pef):
                if row_pef > sell_criteria.get("max_forward_pe"):
                    sell_conditions.append("max_forward_pe")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - forward PE {row_pef:.1f} > {sell_criteria['max_forward_pe']:.1f}")

            if "max_trailing_pe" in sell_criteria and not pd.isna(row_pet):
                if row_pet > sell_criteria.get("max_trailing_pe"):
                    sell_conditions.append("max_trailing_pe")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - trailing PE {row_pet:.1f} > {sell_criteria['max_trailing_pe']:.1f}")

            # PEF > PET requirement for SELL: Forward PE significantly higher than Trailing PE (deteriorating earnings)
            # Only apply this check when both values are meaningful (> 10)
            if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
                # PEF more than 20% higher than PET suggests deteriorating earnings outlook
                if row_pef > row_pet * 1.2:
                    sell_conditions.append("pef_greater_pet_deteriorating")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - PEF:{row_pef:.1f} > PET*1.2:{row_pet * 1.2:.1f} (deteriorating earnings)")

            if "max_peg" in sell_criteria and not pd.isna(row_peg):
                if row_peg > sell_criteria.get("max_peg"):
                    sell_conditions.append("max_peg")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - PEG {row_peg:.2f} > {sell_criteria['max_peg']:.2f}")

            if "min_short_interest" in sell_criteria and not pd.isna(row_si):
                if row_si > sell_criteria.get("min_short_interest"):
                    sell_conditions.append("high_short_interest")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - short interest {row_si:.1f}% > {sell_criteria['min_short_interest']:.1f}%")

            if "min_beta" in sell_criteria and not pd.isna(row_beta):
                if row_beta > sell_criteria.get("min_beta"):
                    sell_conditions.append("high_beta")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - beta {row_beta:.2f} > {sell_criteria['min_beta']:.2f}")

            # ROE and DE SELL criteria (with sector adjustments)
            if "min_roe" in sell_criteria and not pd.isna(row_roe):
                if row_roe < sell_criteria.get("min_roe"):
                    sell_conditions.append("low_roe")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - ROE:{row_roe:.1f}% < min:{sell_criteria['min_roe']:.1f}%")

            if "max_debt_equity" in sell_criteria and not pd.isna(row_de):
                if row_de > sell_criteria.get("max_debt_equity"):
                    sell_conditions.append("high_debt_equity")
                    logger.debug(f"Ticker {ticker}: SELL TRIGGER - DE:{row_de:.1f}% > max:{sell_criteria['max_debt_equity']:.1f}%")

            # NEW MOMENTUM SELL CRITERIA
            # Price Momentum: SELL if fallen too far from 52-week high
            if "max_pct_from_52w_high" in sell_criteria and not pd.isna(row_pct_52w):
                if row_pct_52w < sell_criteria.get("max_pct_from_52w_high"):
                    sell_conditions.append("max_pct_from_52w_high")
                    logger.info(f"Ticker {ticker}: SELL TRIGGER - 52w% {row_pct_52w:.1f}% < {sell_criteria['max_pct_from_52w_high']:.1f}%")

            # Analyst Momentum: SELL if analysts are significantly downgrading
            if "max_analyst_momentum" in sell_criteria and not pd.isna(row_amom):
                if row_amom < sell_criteria.get("max_analyst_momentum"):
                    sell_conditions.append("max_analyst_momentum")
                    logger.info(f"Ticker {ticker}: SELL TRIGGER - analyst momentum {row_amom:.1f}% < {sell_criteria['max_analyst_momentum']:.1f}%")

            # Sector-Relative Valuation: SELL if PE way above sector median
            if "max_pe_vs_sector" in sell_criteria and not pd.isna(row_pe_vs_sector):
                if row_pe_vs_sector > sell_criteria.get("max_pe_vs_sector"):
                    sell_conditions.append("max_pe_vs_sector")
                    logger.info(f"Ticker {ticker}: SELL TRIGGER - PE/sector {row_pe_vs_sector:.2f}x > {sell_criteria['max_pe_vs_sector']:.2f}x")

            if any(sell_conditions):
                logger.info(f"Ticker {ticker}: MARKED AS SELL - triggered by: {', '.join(str(c) for c in sell_conditions)}")
                actions.loc[idx] = "S"
                # Log SELL signal for forward validation (must log before continue)
                try:
                    from trade_modules.signal_tracker import log_signal
                    price_raw = df.get("price", df.get("PRICE", pd.Series([None] * len(df), index=df.index)))
                    row_price = price_raw.loc[idx] if idx in price_raw.index else None
                    target_raw = df.get("target_price", df.get("TARGET", pd.Series([None] * len(df), index=df.index)))
                    row_target = target_raw.loc[idx] if idx in target_raw.index else None
                    sector_raw = df.get("sector", df.get("SECTOR", pd.Series([None] * len(df), index=df.index)))
                    row_sector = sector_raw.loc[idx] if idx in sector_raw.index else None

                    log_signal(
                        ticker=ticker,
                        signal="S",
                        price=float(row_price) if row_price and not pd.isna(row_price) else None,
                        target=float(row_target) if row_target and not pd.isna(row_target) else None,
                        upside=row_upside,
                        buy_pct=row_buy_pct,
                        exret=row_exret,
                        market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                        tier=tier,
                        region=region,
                        sector=str(row_sector) if row_sector and not pd.isna(row_sector) else None,
                        # Add additional metrics for comprehensive tracking
                        pe_forward=float(row_pef) if not pd.isna(row_pef) else None,
                        pe_trailing=float(row_pet) if not pd.isna(row_pet) else None,
                        peg=float(row_peg) if not pd.isna(row_peg) else None,
                        short_interest=float(row_si) if not pd.isna(row_si) else None,
                        roe=float(row_roe) if not pd.isna(row_roe) else None,
                        debt_equity=float(row_de) if not pd.isna(row_de) else None,
                        pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
                        sell_triggers=list(str(c) for c in sell_conditions),
                    )
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"Failed to log SELL signal for {ticker}: {e}")
                continue

        logger.debug(f"Ticker {ticker}: Passed SELL checks, evaluating BUY criteria")

        # BUY criteria - ALL conditions must be true
        # Start with assuming all conditions pass
        is_buy_candidate = True

        # CRITICAL SAFETY CHECK: Never mark negative upside stocks as BUY
        # This is a hard constraint regardless of configuration to prevent catastrophic errors
        if row_upside < 0:
            is_buy_candidate = False
            logger.debug(f"Ticker {ticker}: No buy - negative upside {row_upside:.1f}% (safety check)")

        # Required criteria - only apply if explicitly defined in YAML
        # NO DEFAULTS - criteria must be explicitly configured
        if "min_upside" in buy_criteria:
            if row_upside < buy_criteria["min_upside"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - upside {row_upside:.1f}% < {buy_criteria['min_upside']:.1f}%")

        if "min_buy_percentage" in buy_criteria:
            if row_buy_pct < buy_criteria["min_buy_percentage"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - buy% {row_buy_pct:.1f}% < {buy_criteria['min_buy_percentage']:.1f}%")

        if "min_exret" in buy_criteria:
            if row_exret < buy_criteria["min_exret"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - exret {row_exret:.1f}% < {buy_criteria['min_exret']:.1f}%")

        # Check analyst requirements with rating type differentiation
        # Type "E" = ratings since last earnings (more relevant, lower threshold)
        # Type "A" = all-time ratings (less relevant, higher threshold)
        if "min_analysts" in buy_criteria:
            # Get rating type from DataFrame - column "A" contains E/A rating type
            rating_type_col = df.get("A", df.get("rating_type", pd.Series(["A"] * len(df), index=df.index)))
            rating_type = str(rating_type_col.loc[idx]) if idx in rating_type_col.index else "A"

            # Determine base threshold based on rating type
            if rating_type == "E" and "min_analysts_earnings" in buy_criteria:
                # Post-earnings ratings: use lower threshold
                threshold = buy_criteria["min_analysts_earnings"]
            else:
                # All-time ratings: use standard threshold
                threshold = buy_criteria["min_analysts"]

            # Quality override: positive analyst momentum can lower threshold by 1
            # This rewards stocks where analyst sentiment is improving
            if buy_criteria.get("quality_override", False):
                if not pd.isna(row_amom) and row_amom >= 0:
                    original_threshold = threshold
                    threshold = max(4, threshold - 1)  # Never go below 4 analysts
                    if threshold < original_threshold:
                        logger.debug(f"Ticker {ticker}: Quality override - lowered analyst threshold from {original_threshold} to {threshold} (AM={row_amom:.1f}%)")

            if analyst_count.loc[idx] < threshold:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - analysts {analyst_count.loc[idx]} < {threshold} (type={rating_type})")

        # Optional criteria from YAML (only apply if defined in YAML)
        if "min_beta" in buy_criteria and "max_beta" in buy_criteria and not pd.isna(row_beta):
            beta_min = buy_criteria.get("min_beta")
            beta_max = buy_criteria.get("max_beta")
            if not (beta_min <= row_beta <= beta_max):
                is_buy_candidate = False

        if "min_forward_pe" in buy_criteria and "max_forward_pe" in buy_criteria and not pd.isna(row_pef):
            pef_min = buy_criteria.get("min_forward_pe")
            pef_max = buy_criteria.get("max_forward_pe")
            if not (pef_min < row_pef <= pef_max):
                is_buy_candidate = False

        if "min_trailing_pe" in buy_criteria and "max_trailing_pe" in buy_criteria and not pd.isna(row_pet):
            pet_min = buy_criteria.get("min_trailing_pe")
            pet_max = buy_criteria.get("max_trailing_pe")
            if not (pet_min < row_pet <= pet_max):
                is_buy_candidate = False

        # PEF < PET requirement: Forward PE should be lower than Trailing PE (improving earnings)
        # Only apply this check when both values are meaningful (> 10) and PEF is significantly higher
        # Academic research (Liu, Nissim & Thomas 2002) shows 10-15% differentials are estimate noise
        # Using 20% threshold (1.2x) better distinguishes genuine deterioration from analyst variance
        if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
            # PEF should not be more than 20% higher than PET (configurable, default 1.2x)
            pef_pet_threshold = buy_criteria.get('max_pef_pet_ratio', 1.20)
            if row_pef > row_pet * pef_pet_threshold:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed PEF<PET check - PEF:{row_pef:.1f} > PET*{pef_pet_threshold}:{row_pet * pef_pet_threshold:.1f}")

        if "max_peg" in buy_criteria and not pd.isna(row_peg):
            if row_peg > buy_criteria.get("max_peg"):
                is_buy_candidate = False

        if "max_short_interest" in buy_criteria and not pd.isna(row_si):
            if row_si > buy_criteria.get("max_short_interest"):
                is_buy_candidate = False

        # ROE and DE BUY criteria (with sector adjustments)
        if "min_roe" in buy_criteria and not pd.isna(row_roe):
            if row_roe < buy_criteria.get("min_roe"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed ROE check - ROE:{row_roe:.1f}% < min:{buy_criteria['min_roe']:.1f}%")

        if "max_debt_equity" in buy_criteria and not pd.isna(row_de):
            if row_de > buy_criteria.get("max_debt_equity"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed DE check - DE:{row_de:.1f}% > max:{buy_criteria['max_debt_equity']:.1f}%")

        # NEW MOMENTUM BUY CRITERIA
        # Price Momentum: Must be within range of 52-week high
        if "min_pct_from_52w_high" in buy_criteria and not pd.isna(row_pct_52w):
            if row_pct_52w < buy_criteria.get("min_pct_from_52w_high"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed 52w% check - {row_pct_52w:.1f}% < min:{buy_criteria['min_pct_from_52w_high']:.1f}%")

        # Price Momentum: Must be above 200-day moving average (if required)
        if buy_criteria.get("require_above_200dma", False):
            if row_above_200dma is False and not pd.isna(row_above_200dma):  # Explicitly False, not NaN
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed above 200DMA check - trading below 200-day MA")

        # Analyst Momentum: Don't buy if analysts are significantly downgrading
        if "min_analyst_momentum" in buy_criteria and not pd.isna(row_amom):
            if row_amom < buy_criteria.get("min_analyst_momentum"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed analyst momentum check - {row_amom:.1f}% < min:{buy_criteria['min_analyst_momentum']:.1f}%")

        # Sector-Relative Valuation: PE should not be too far above sector median
        if "max_pe_vs_sector" in buy_criteria and not pd.isna(row_pe_vs_sector):
            if row_pe_vs_sector > buy_criteria.get("max_pe_vs_sector"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed PE/sector check - {row_pe_vs_sector:.2f}x > max:{buy_criteria['max_pe_vs_sector']:.2f}x")

        # Sector-Relative Valuation: PE should not be too far below sector (value trap warning)
        if "min_pe_vs_sector" in buy_criteria and not pd.isna(row_pe_vs_sector):
            if row_pe_vs_sector < buy_criteria.get("min_pe_vs_sector"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed PE/sector check - {row_pe_vs_sector:.2f}x < min:{buy_criteria['min_pe_vs_sector']:.2f}x (value trap)")

        # NEW: FCF Yield - academically proven alpha factor (Sloan 1996, Lakonishok 1994)
        # Positive FCF indicates business generates real cash, not just accounting profits
        # Data quality check: FCF < -100% is likely bad data (e.g., ADR data issues)
        if "min_fcf_yield" in buy_criteria and not pd.isna(row_fcf_yield):
            # Skip FCF check if value is clearly bad data (< -100% is impossible in reality)
            if row_fcf_yield < -100:
                logger.debug(f"Ticker {ticker}: Skipping FCF check - value {row_fcf_yield:.1f}% appears to be bad data")
            elif row_fcf_yield < buy_criteria.get("min_fcf_yield"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed FCF yield check - {row_fcf_yield:.1f}% < min:{buy_criteria['min_fcf_yield']:.1f}%")

        # NEW: Revenue Growth - harder to manipulate than EPS
        # Avoid companies with collapsing top-line (can be signal of structural decline)
        if "min_revenue_growth" in buy_criteria and not pd.isna(row_rev_growth):
            if row_rev_growth < buy_criteria.get("min_revenue_growth"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed revenue growth check - {row_rev_growth:.1f}% < min:{buy_criteria['min_revenue_growth']:.1f}%")

        if is_buy_candidate:
            actions.loc[idx] = "B"

            # Calculate BUY conviction score for ranking candidates
            if yaml_config.is_signal_scoring_enabled():
                buy_scoring_config = yaml_config.get_buy_scoring_config(region, tier)
                if buy_scoring_config.get('enabled', True):
                    buy_conviction_score = calculate_buy_score(
                        upside=row_upside,
                        buy_pct=row_buy_pct,
                        exret=row_exret,
                        pct_52w=row_pct_52w,
                        above_200dma=row_above_200dma,
                        pef=row_pef,
                        pet=row_pet,
                        roe=row_roe,
                        de=row_de,
                        fcf_yield=row_fcf_yield,
                        buy_scoring_config=buy_scoring_config,
                    )
                    buy_scores.loc[idx] = buy_conviction_score
                    logger.debug(f"Ticker {ticker}: BUY with conviction score {buy_conviction_score:.1f}")

        # Otherwise remains "H" (HOLD)

        # Log signal for forward validation (P1 improvement)
        try:
            from trade_modules.signal_tracker import log_signal
            # Get price data if available
            price_raw = df.get("price", df.get("PRICE", pd.Series([None] * len(df), index=df.index)))
            row_price = price_raw.loc[idx] if idx in price_raw.index else None
            target_raw = df.get("target_price", df.get("TARGET", pd.Series([None] * len(df), index=df.index)))
            row_target = target_raw.loc[idx] if idx in target_raw.index else None
            sector_raw = df.get("sector", df.get("SECTOR", pd.Series([None] * len(df), index=df.index)))
            row_sector = sector_raw.loc[idx] if idx in sector_raw.index else None

            log_signal(
                ticker=ticker,
                signal=actions.loc[idx],
                price=float(row_price) if row_price and not pd.isna(row_price) else None,
                target=float(row_target) if row_target and not pd.isna(row_target) else None,
                upside=row_upside,
                buy_pct=row_buy_pct,
                exret=row_exret,
                market_cap=cap_values.loc[idx] if idx in cap_values.index else None,
                tier=tier,
                region=region,
                sector=str(row_sector) if row_sector and not pd.isna(row_sector) else None,
                # Additional metrics for comprehensive tracking
                pe_forward=float(row_pef) if not pd.isna(row_pef) else None,
                pe_trailing=float(row_pet) if not pd.isna(row_pet) else None,
                peg=float(row_peg) if not pd.isna(row_peg) else None,
                short_interest=float(row_si) if not pd.isna(row_si) else None,
                roe=float(row_roe) if not pd.isna(row_roe) else None,
                debt_equity=float(row_de) if not pd.isna(row_de) else None,
                pct_52w_high=float(row_pct_52w) if not pd.isna(row_pct_52w) else None,
            )
        except ImportError:
            pass  # Signal tracker not available
        except Exception as e:
            logger.debug(f"Failed to log signal for {ticker}: {e}")

    return actions, buy_scores


def calculate_action(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trading action (B/S/H) for each row based on trading criteria.

    Uses vectorized operations instead of row-by-row apply for better performance.
    Adds BUY_SCORE column for BUY candidates when signal scoring is enabled.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial metrics

    Returns
    -------
    pd.DataFrame
        DataFrame with BS column added (and BUY_SCORE if scoring enabled)
    """
    working_df = df.copy()

    try:
        # Use vectorized action calculation for better performance
        actions, buy_scores = calculate_action_vectorized(working_df)
        working_df["BS"] = actions

        # Add BUY_SCORE column if any BUY candidates have scores
        if buy_scores.notna().any():
            working_df["BUY_SCORE"] = buy_scores
            logger.debug(f"Added BUY_SCORE for {buy_scores.notna().sum()} BUY candidates")

        logger.debug(f"Calculated actions for {len(working_df)} rows using vectorized operations")
        return working_df
    except (KeyError, TypeError, ValueError) as e:
        # Data-related errors: missing columns, type mismatches
        logger.warning(f"Data error calculating actions: {str(e)}")
        working_df["BS"] = "H"  # Default to HOLD
        return working_df
    except (AttributeError, IndexError, pd.errors.InvalidIndexError) as e:
        # Unexpected pandas/index errors - log with full traceback for debugging
        import traceback
        logger.error(f"Unexpected error calculating actions: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        working_df["BS"] = "H"  # Default to HOLD
        return working_df


def filter_buy_opportunities_wrapper(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and identify buy opportunities from market data.

    Args:
        market_df: Market data DataFrame

    Returns:
        pd.DataFrame: Filtered buy opportunities
    """
    from yahoofinance.core.errors import DataError, ValidationError

    try:
        logger.info("Filtering buy opportunities...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_buy_opportunities

        # Use the centralized filter function
        buy_opps = filter_buy_opportunities(market_df)

        logger.info(f"Found {len(buy_opps)} buy opportunities")
        return buy_opps
    except (DataError, ValidationError) as e:
        logger.warning(f"Data validation error filtering buy opportunities: {str(e)}")
        return pd.DataFrame()
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Data error filtering buy opportunities: {str(e)}")
        return pd.DataFrame()
    except (AttributeError, IndexError, ImportError, pd.errors.InvalidIndexError) as e:
        logger.error(f"Unexpected error filtering buy opportunities: {str(e)}")
        return pd.DataFrame()


def filter_sell_candidates_wrapper(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and identify sell candidates from portfolio data.

    Args:
        portfolio_df: Portfolio data DataFrame

    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    from yahoofinance.core.errors import DataError, ValidationError

    try:
        logger.info("Filtering sell candidates...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_sell_candidates

        # Use the centralized filter function
        sell_candidates = filter_sell_candidates(portfolio_df)

        logger.info(f"Found {len(sell_candidates)} sell candidates")
        return sell_candidates
    except (DataError, ValidationError) as e:
        logger.warning(f"Data validation error filtering sell candidates: {str(e)}")
        return pd.DataFrame()
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Data error filtering sell candidates: {str(e)}")
        return pd.DataFrame()
    except (AttributeError, IndexError, ImportError, pd.errors.InvalidIndexError) as e:
        logger.error(f"Unexpected error filtering sell candidates: {str(e)}")
        return pd.DataFrame()


def filter_hold_candidates_wrapper(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and identify hold candidates from market data.

    Args:
        market_df: Market data DataFrame

    Returns:
        pd.DataFrame: Filtered hold candidates
    """
    from yahoofinance.core.errors import DataError, ValidationError

    try:
        logger.info("Filtering hold candidates...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_hold_candidates

        # Use the centralized filter function
        hold_candidates = filter_hold_candidates(market_df)

        logger.info(f"Found {len(hold_candidates)} hold candidates")
        return hold_candidates
    except (DataError, ValidationError) as e:
        logger.warning(f"Data validation error filtering hold candidates: {str(e)}")
        return pd.DataFrame()
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Data error filtering hold candidates: {str(e)}")
        return pd.DataFrame()
    except (AttributeError, IndexError, ImportError, pd.errors.InvalidIndexError) as e:
        logger.error(f"Unexpected error filtering hold candidates: {str(e)}")
        return pd.DataFrame()

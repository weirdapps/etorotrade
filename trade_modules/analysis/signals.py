"""
Trading Signal Generation Module

This module contains the core signal generation logic for buy/sell/hold decisions.
Uses vectorized operations for performance on large datasets.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Union

# Import tier utilities
from .tiers import _parse_percentage, _parse_market_cap

# Import from trade_modules
from trade_modules.trade_config import TradeConfig

# Get logger for this module
logger = logging.getLogger(__name__)


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

    # Confidence check - vectorized
    has_confidence = (analyst_count >= config.UNIVERSAL_THRESHOLDS["min_analyst_count"]) & (
        total_ratings >= config.UNIVERSAL_THRESHOLDS["min_price_targets"]
    )

    # Get market cap and parse formatted strings (e.g., "2.47T", "628B")
    cap_raw = df.get("market_cap", df.get("CAP", pd.Series([0] * len(df), index=df.index)))
    cap_values = pd.Series([_parse_market_cap(cap) for cap in cap_raw], index=df.index)

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
    # Fallback to "P/S" (short display name from table_renderer)
    pe_vs_sector_raw = df.get("pe_vs_sector", df.get("P/S", pd.Series([np.nan] * len(df), index=df.index)))
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

    # Apply minimum market cap filter (from config.yaml universal_thresholds.min_market_cap)
    # Stocks below minimum are marked as INCONCLUSIVE to prevent trading in illiquid micro-caps
    min_market_cap = config.UNIVERSAL_THRESHOLDS.get("min_market_cap", 1_000_000_000)  # Default $1B
    below_min_cap = cap_values < min_market_cap
    if below_min_cap.sum() > 0:
        actions[below_min_cap] = "I"  # INCONCLUSIVE for stocks below minimum market cap
        logger.info(
            f"Market cap filter: {below_min_cap.sum()} stocks below ${min_market_cap / 1e9:.1f}B "
            f"minimum threshold set to INCONCLUSIVE"
        )

    # Get ticker column for region detection
    # Check if TICKER is the index (from CSV with index_col=0) or a column
    if df.index.name == "TICKER" or (hasattr(df.index, 'name') and df.index.name and "ticker" in df.index.name.lower()):
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

            # For non-equity assets (crypto, ETF, commodity), use momentum-based signals
            # since analyst coverage is not applicable
            if asset_type in ("crypto", "etf", "commodity"):
                # Use price momentum (52-week high %) and technical indicators
                row_pct_52w = pct_52w.loc[idx]
                row_above_200dma = above_200dma.loc[idx]

                # Simple momentum-based signal logic for non-equity assets
                if pd.isna(row_pct_52w):
                    actions.loc[idx] = "I"  # INCONCLUSIVE if no momentum data
                    logger.info(f"Ticker {ticker}: {asset_type.upper()} marked INCONCLUSIVE - no momentum data")
                elif row_pct_52w >= 85:
                    # Strong momentum - near 52-week high
                    if row_above_200dma is True:
                        actions.loc[idx] = "B"  # BUY for strong uptrend
                        logger.info(f"Ticker {ticker}: {asset_type.upper()} marked BUY - strong momentum ({row_pct_52w:.1f}% of 52w high, above 200DMA)")
                    else:
                        actions.loc[idx] = "H"  # HOLD if not above 200DMA
                        logger.info(f"Ticker {ticker}: {asset_type.upper()} marked HOLD - good momentum but below 200DMA")
                elif row_pct_52w <= 60:
                    # Weak momentum - significantly below 52-week high
                    actions.loc[idx] = "S"  # SELL for weak momentum
                    logger.info(f"Ticker {ticker}: {asset_type.upper()} marked SELL - weak momentum ({row_pct_52w:.1f}% of 52w high)")
                else:
                    actions.loc[idx] = "H"  # HOLD for neutral momentum
                    logger.info(f"Ticker {ticker}: {asset_type.upper()} marked HOLD - neutral momentum ({row_pct_52w:.1f}% of 52w high)")

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

        # SELL criteria - ANY condition triggers SELL
        sell_conditions: List[Union[str, bool]] = []

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
                sell_conditions.append(True)

        if "max_trailing_pe" in sell_criteria and not pd.isna(row_pet):
            if row_pet > sell_criteria.get("max_trailing_pe"):
                sell_conditions.append(True)

        # PEF > PET requirement for SELL: Forward PE significantly higher than Trailing PE (deteriorating earnings)
        # Only apply this check when both values are meaningful (> 10)
        if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
            # PEF more than 20% higher than PET suggests deteriorating earnings outlook
            if row_pef > row_pet * 1.2:
                sell_conditions.append(True)
                logger.debug(f"Ticker {ticker}: Sell signal - PEF:{row_pef:.1f} > PET*1.2:{row_pet * 1.2:.1f} (deteriorating earnings)")

        if "max_peg" in sell_criteria and not pd.isna(row_peg):
            if row_peg > sell_criteria.get("max_peg"):
                sell_conditions.append(True)

        if "min_short_interest" in sell_criteria and not pd.isna(row_si):
            if row_si > sell_criteria.get("min_short_interest"):
                sell_conditions.append(True)

        if "min_beta" in sell_criteria and not pd.isna(row_beta):
            if row_beta > sell_criteria.get("min_beta"):
                sell_conditions.append(True)

        # ROE and DE SELL criteria (with sector adjustments)
        if "min_roe" in sell_criteria and not pd.isna(row_roe):
            if row_roe < sell_criteria.get("min_roe"):
                sell_conditions.append(True)
                logger.debug(f"Ticker {ticker}: Sell signal - ROE:{row_roe:.1f}% < min:{sell_criteria['min_roe']:.1f}%")

        if "max_debt_equity" in sell_criteria and not pd.isna(row_de):
            if row_de > sell_criteria.get("max_debt_equity"):
                sell_conditions.append(True)
                logger.debug(f"Ticker {ticker}: Sell signal - DE:{row_de:.1f}% > max:{sell_criteria['max_debt_equity']:.1f}%")

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

        # High Consensus Contrarian Warning (P0 improvement)
        # When >95% of analysts agree, it may indicate overcrowding
        if "max_buy_percentage" in buy_criteria:
            if row_buy_pct > buy_criteria["max_buy_percentage"]:
                is_buy_candidate = False
                logger.info(f"Ticker {ticker}: Contrarian warning - buy% {row_buy_pct:.1f}% > {buy_criteria['max_buy_percentage']:.1f}% (overcrowded)")

        if "min_exret" in buy_criteria:
            if row_exret < buy_criteria["min_exret"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - exret {row_exret:.1f}% < {buy_criteria['min_exret']:.1f}%")

        # Check analyst requirements
        if "min_analysts" in buy_criteria:
            if analyst_count.loc[idx] < buy_criteria["min_analysts"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - analysts {analyst_count.loc[idx]} < {buy_criteria['min_analysts']}")

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
        if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
            # PEF should not be more than 10% higher than PET
            if row_pef > row_pet * 1.1:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed PEF<PET check - PEF:{row_pef:.1f} > PET*1.1:{row_pet * 1.1:.1f}")

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
        if "min_fcf_yield" in buy_criteria and not pd.isna(row_fcf_yield):
            if row_fcf_yield < buy_criteria.get("min_fcf_yield"):
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

    return actions


def calculate_action(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trading action (B/S/H) for each row based on trading criteria.

    Uses vectorized operations instead of row-by-row apply for better performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial metrics

    Returns
    -------
    pd.DataFrame
        DataFrame with BS column added
    """
    working_df = df.copy()

    try:
        # Use vectorized action calculation for better performance
        working_df["BS"] = calculate_action_vectorized(working_df)

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

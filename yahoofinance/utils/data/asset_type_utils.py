"""
Asset Type Classification and Universal Sorting Utilities

This module provides utilities for classifying assets by type and implementing
universal sorting across all trade operations with asset type priority and
market cap descending.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Asset type ordering priority (lower number = higher priority)
ASSET_TYPE_PRIORITY = {
    "stock": 1,
    "etf": 2, 
    "crypto": 3,
    "commodity": 4,
    "other": 5
}


def classify_asset_type(ticker: str, market_cap: Optional[float] = None, 
                       company_name: Optional[str] = None) -> str:
    """
    Classify an asset by its type based on ticker symbol and other attributes.
    
    Args:
        ticker: Ticker symbol
        market_cap: Market capitalization in USD (optional)
        company_name: Company/asset name (optional)
        
    Returns:
        Asset type: "stock", "etf", "crypto", "commodity", or "other"
    """
    if not ticker:
        return "other"
        
    ticker_upper = ticker.upper().strip()
    
    # Crypto classification (highest priority for crypto patterns)
    if _is_crypto_asset(ticker_upper):
        return "crypto"
    
    # ETF classification
    if _is_etf_asset(ticker_upper, company_name):
        return "etf"
    
    # Commodity classification
    if _is_commodity_asset(ticker_upper, company_name):
        return "commodity"
    
    # Default to stock if not classified otherwise
    return "stock"


def _is_crypto_asset(ticker: str) -> bool:
    """Check if ticker represents a cryptocurrency."""
    # Crypto tickers typically end with -USD
    if ticker.endswith('-USD'):
        return True
    
    # Known crypto tickers without -USD suffix
    known_crypto = {
        'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 
        'XLM', 'DOGE', 'SOL', 'HBAR', 'MATIC', 'AVAX', 'ATOM',
        'ALGO', 'VET', 'FIL', 'THETA', 'TRX', 'EOS', 'XMR',
        'DASH', 'ZEC', 'NEO', 'QTUM', 'ONT', 'IOTA', 'XTZ'
    }
    
    return ticker in known_crypto


def _is_etf_asset(ticker: str, company_name: Optional[str] = None) -> bool:
    """Check if ticker represents an ETF."""
    # Common ETF patterns
    etf_patterns = ['ETF', 'FUND', 'INDEX', 'TRUST']
    
    # Check company name for ETF indicators using whole word matching
    if company_name:
        company_upper = company_name.upper()
        # Split company name into words and check each word
        company_words = company_upper.replace(',', ' ').replace('.', ' ').split()
        for pattern in etf_patterns:
            if pattern in company_words:
                return True
    
    # Known major ETF tickers
    known_etfs = {
        'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND',
        'AGG', 'LQD', 'HYG', 'EMB', 'TLT', 'IEF', 'SHY', 'TIP',
        'GLD', 'SLV', 'USO', 'UNG', 'PDBC', 'DJP', 'IAU', 'SGOL',
        'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLY',
        'XLP', 'XLU', 'VGT', 'VHT', 'VFH', 'VNQ', 'VDE', 'VAW',
        'VIS', 'VCR', 'VDC', 'VPU', 'ARKK', 'ARKQ', 'ARKW', 'ARKG',
        'ARKF', 'ICLN', 'CLEAN', 'PBW', 'QCLN', 'SMOG', 'FAN',
        'EWJ', 'EWZ', 'EWW', 'EWG', 'EWU', 'EWC', 'EWA', 'EWS',
        'EWT', 'EWY', 'EWP', 'EWI', 'EWQ', 'EWL', 'EWK', 'EWD',
        'EWN', 'EWO', 'EWH', 'EPP', 'EZA', 'ECH', 'EPHE', 'EPU',
        'ERUS', 'RSX', 'EEM', 'VWO', 'IEMG', 'SCHE', 'EDC', 'EWX',
        'FXI', 'ASHR', 'MCHI', 'KWEB', 'CXSE', 'GXC', 'TAO',  # China ETFs
        'LYXGRE.DE'  # Lyxor Green Bond (EUR) ETF
    }
    
    return ticker in known_etfs


def _is_commodity_asset(ticker: str, company_name: Optional[str] = None) -> bool:
    """Check if ticker represents a commodity."""
    # Handle VIX patterns (volatility index)
    if ticker.startswith(('VIX', '^VIX')):
        return True
    
    # Known commodity tickers
    known_commodities = {
        'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PA=F', 'PL=F',
        'GOLD', 'SILVER', 'OIL', 'GAS', 'COPPER', 'PLATINUM', 'PALLADIUM',
        'WHEAT', 'CORN', 'SOYBEAN', 'SUGAR', 'COFFEE', 'COTTON'
    }
    
    if ticker in known_commodities:
        return True
    
    # Check company name for commodity indicators
    if company_name:
        company_upper = company_name.upper()
        commodity_keywords = ['GOLD', 'SILVER', 'OIL', 'GAS', 'COPPER', 'PLATINUM', 'COMMODITY']
        for keyword in commodity_keywords:
            if keyword in company_upper:
                return True
    
    return False


def get_market_cap_usd(row: pd.Series) -> float:
    """
    Extract market cap in USD from various possible columns and formats.
    
    Args:
        row: DataFrame row containing market cap data
        
    Returns:
        Market cap in USD as float, 0 if not available
    """
    # Try different column names for market cap
    market_cap_columns = [
        'market_cap', 'market_cap_usd', 'marketCap', 'market_capitalization',
        'CAP', 'cap', 'market_cap_formatted', 'mktCap', 'market_value'
    ]
    
    for col in market_cap_columns:
        if col in row.index and pd.notna(row[col]):
            value = row[col]
            
            # If it's already a number
            if isinstance(value, (int, float)):
                return float(value)
            
            # If it's a formatted string, parse it
            if isinstance(value, str):
                parsed_value = _parse_market_cap_string(value)
                if parsed_value > 0:
                    return parsed_value
    
    return 0.0


def _parse_market_cap_string(value: str) -> float:
    """
    Parse market cap from formatted string (e.g., '100.5B', '1.2T', '500M').
    
    Args:
        value: Formatted market cap string
        
    Returns:
        Market cap in USD as float, 0 if parsing fails
    """
    try:
        if not value or value == '--' or value == 'N/A':
            return 0.0
        
        value_clean = str(value).strip().upper().replace('$', '').replace(',', '')
        
        # Handle different suffixes
        multipliers = {
            'T': 1_000_000_000_000,  # Trillion
            'B': 1_000_000_000,      # Billion  
            'M': 1_000_000,          # Million
            'K': 1_000               # Thousand
        }
        
        for suffix, multiplier in multipliers.items():
            if value_clean.endswith(suffix):
                numeric_part = value_clean[:-1]
                try:
                    return float(numeric_part) * multiplier
                except ValueError:
                    continue
        
        # If no suffix, try to parse as plain number
        try:
            return float(value_clean)
        except ValueError:
            return 0.0

    except (ValueError, TypeError, AttributeError):
        return 0.0


def add_asset_type_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add asset type classification to DataFrame.
    
    Args:
        df: Input DataFrame with ticker data
        
    Returns:
        DataFrame with added 'asset_type' and 'asset_priority' columns
    """
    if df.empty:
        return df
    
    result_df = df.copy()
    
    # Determine ticker column name
    ticker_col = None
    for col in ['TICKER', 'ticker', 'symbol', 'Symbol', 'SYMBOL']:
        if col in result_df.columns:
            ticker_col = col
            break
    
    if not ticker_col:
        logger.warning("No ticker column found for asset classification")
        result_df['asset_type'] = 'other'
        result_df['asset_priority'] = ASSET_TYPE_PRIORITY['other']
        return result_df
    
    # Determine company name column
    company_col = None
    for col in ['COMPANY', 'company_name', 'name', 'Name', 'company', 'longName']:
        if col in result_df.columns:
            company_col = col
            break
    
    # Classify each asset
    def classify_row(row):
        ticker = row[ticker_col] if pd.notna(row[ticker_col]) else ""
        company_name = row[company_col] if company_col and pd.notna(row[company_col]) else None
        market_cap = get_market_cap_usd(row)
        
        asset_type = classify_asset_type(ticker, market_cap, company_name)
        return asset_type
    
    result_df['asset_type'] = result_df.apply(classify_row, axis=1)
    result_df['asset_priority'] = result_df['asset_type'].map(ASSET_TYPE_PRIORITY)
    
    # Extract market cap in USD for sorting
    result_df['market_cap_usd_sort'] = result_df.apply(get_market_cap_usd, axis=1)
    
    logger.debug(f"Classified {len(result_df)} assets by type")
    return result_df


def universal_sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply universal sorting: asset type priority first, then market cap descending.
    
    Args:
        df: DataFrame to sort
        
    Returns:
        Sorted DataFrame with asset classification
    """
    if df.empty:
        return df
    
    try:
        # Add asset type classification if not already present
        if 'asset_type' not in df.columns:
            sorted_df = add_asset_type_classification(df)
        else:
            sorted_df = df.copy()
            # Ensure we have the USD market cap for sorting
            if 'market_cap_usd_sort' not in sorted_df.columns:
                sorted_df['market_cap_usd_sort'] = sorted_df.apply(get_market_cap_usd, axis=1)
        
        # Sort by asset priority (ascending) then market cap (descending)
        sorted_df = sorted_df.sort_values([
            'asset_priority',      # 1=stocks, 2=ETFs, 3=crypto, 4=commodities, 5=other
            'market_cap_usd_sort'  # Descending market cap within each asset type
        ], ascending=[True, False])
        
        # Reset index
        sorted_df = sorted_df.reset_index(drop=True)
        
        # Clean up temporary columns for final output
        columns_to_drop = ['asset_priority', 'market_cap_usd_sort']
        for col in columns_to_drop:
            if col in sorted_df.columns:
                sorted_df = sorted_df.drop(col, axis=1)
        
        logger.debug(f"Applied universal sorting to {len(sorted_df)} rows")
        return sorted_df

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error in universal sorting: {str(e)}")
        return df


def get_asset_type_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get summary counts by asset type.
    
    Args:
        df: DataFrame with asset_type column
        
    Returns:
        Dictionary with asset type counts
    """
    if df.empty or 'asset_type' not in df.columns:
        return {}
    
    try:
        summary = df['asset_type'].value_counts().to_dict()
        
        # Ensure all asset types are represented
        for asset_type in ASSET_TYPE_PRIORITY.keys():
            if asset_type not in summary:
                summary[asset_type] = 0
        
        return summary

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error creating asset type summary: {str(e)}")
        return {}


def format_asset_type_summary(summary: Dict[str, int]) -> str:
    """
    Format asset type summary for display.
    
    Args:
        summary: Asset type counts dictionary
        
    Returns:
        Formatted summary string
    """
    if not summary:
        return "No asset type data available"
    
    # Order by priority
    ordered_types = sorted(summary.keys(), key=lambda x: ASSET_TYPE_PRIORITY.get(x, 999))
    
    summary_lines = ["Asset Type Distribution:"]
    for asset_type in ordered_types:
        count = summary[asset_type]
        if count > 0:
            summary_lines.append(f"  {asset_type.title()}: {count}")
    
    total = sum(summary.values())
    summary_lines.append(f"  Total: {total}")
    
    return "\n".join(summary_lines)
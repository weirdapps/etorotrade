"""
Table rendering utilities for console display.

This module provides table formatting, display, and data preparation for console output.
"""

import math
from typing import Any, Dict, List, Optional

import pandas as pd
from tabulate import tabulate

from yahoofinance.core.config import COLUMN_NAMES, DISPLAY, STANDARD_DISPLAY_COLUMNS
from yahoofinance.core.logging import get_logger
from yahoofinance.utils.data.asset_type_utils import universal_sort_dataframe
from yahoofinance.utils.data.format_utils import (
    calculate_position_size,
    format_number,
    format_position_size,
)
from yahoofinance.utils.data.ticker_utils import (
    get_ticker_for_display,
    process_ticker_input,
)


logger = get_logger(__name__)


def sort_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort market data using universal sorting: asset type first, then market cap descending.

    Args:
        df: DataFrame containing market data

    Returns:
        Sorted DataFrame
    """
    if df.empty:
        return df

    # Apply universal sorting (asset type priority, then market cap descending)
    return universal_sort_dataframe(df)


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame for display.

    Args:
        df: Raw DataFrame

    Returns:
        Formatted DataFrame ready for display
    """
    if df.empty:
        return df

    # Map raw column names to display column names using config values
    column_mapping = {
        'symbol': 'TICKER',
        'ticker': 'TICKER',
        'company': 'COMPANY',
        'name': 'COMPANY',
        'current_price': 'PRICE',
        'price': 'PRICE',
        'target_price': 'TARGET',
        'upside': 'UPSIDE',
        'analyst_count': COLUMN_NAMES['ANALYST_COUNT'],
        'total_ratings': COLUMN_NAMES['TOTAL_RATINGS'],
        'buy_percentage': COLUMN_NAMES['BUY_PERCENTAGE'],
        'market_cap_fmt': 'CAP',
        'market_cap': 'CAP',
        'pe_trailing': 'PET',
        'pe_forward': 'PEF',
        'peg_ratio': 'PEG',
        'beta': 'BETA',
        'short_percent': 'SI',
        'dividend_yield': COLUMN_NAMES['DIVIDEND_YIELD_DISPLAY'],
        'earnings_date': 'EARNINGS',
        'E': 'E',  # Earnings filter type column
        'earnings_growth': 'EG',
        'twelve_month_performance': 'PP',
        'return_on_equity': 'ROE',
        'debt_to_equity': 'DE',
        'EXRET': 'EXRET',
        'A': 'A',
        # Identity mappings for columns already in display format
        'TICKER': 'TICKER',
        'COMPANY': 'COMPANY',
        'PRICE': 'PRICE',
        'TARGET': 'TARGET',
        'UPSIDE': 'UPSIDE',
        COLUMN_NAMES['ANALYST_COUNT']: COLUMN_NAMES['ANALYST_COUNT'],
        COLUMN_NAMES['TOTAL_RATINGS']: COLUMN_NAMES['TOTAL_RATINGS'],
        COLUMN_NAMES['BUY_PERCENTAGE']: COLUMN_NAMES['BUY_PERCENTAGE'],
        'CAP': 'CAP',
        'PET': 'PET',
        'PEF': 'PEF',
        'PEG': 'PEG',
        'BETA': 'BETA',
        'SI': 'SI',
        COLUMN_NAMES['DIVIDEND_YIELD_DISPLAY']: COLUMN_NAMES['DIVIDEND_YIELD_DISPLAY'],
        'EARNINGS': 'EARNINGS',
        'EG': 'EG',
        'PP': 'PP',
        'ROE': 'ROE',
        'DE': 'DE',
        'SIZE': 'SIZE',
        'M': 'M'
    }

    # Create new DataFrame with only mapped columns to avoid duplicates
    new_df = pd.DataFrame()

    # Copy index from original
    new_df.index = df.index

    # Apply column mapping, taking the first available source column for each target
    for source_col, target_col in column_mapping.items():
        if source_col in df.columns and target_col not in new_df.columns:
            if target_col == 'TICKER':
                # Apply ticker normalization for dual-listed stock display
                new_df[target_col] = df[source_col].apply(
                    lambda x: get_ticker_for_display(process_ticker_input(x)) if pd.notna(x) and x else x
                )
            else:
                new_df[target_col] = df[source_col]

    # Handle special cases for EG and PP columns that might already exist with raw names
    if 'earnings_growth' in df.columns and 'EG' not in new_df.columns:
        new_df['EG'] = df['earnings_growth']
    if 'twelve_month_performance' in df.columns and 'PP' not in new_df.columns:
        new_df['PP'] = df['twelve_month_performance']
    if 'return_on_equity' in df.columns and 'ROE' not in new_df.columns:
        new_df['ROE'] = df['return_on_equity']
    if 'debt_to_equity' in df.columns and 'DE' not in new_df.columns:
        new_df['DE'] = df['debt_to_equity']

    # Copy any EXRET column if present
    if 'EXRET' in df.columns:
        new_df['EXRET'] = df['EXRET']

    # Copy any A column if present
    if 'A' in df.columns:
        new_df['A'] = df['A']

    df = new_df

    # Handle BS/ACTION column - ALWAYS recalculate to ensure current criteria are applied
    bs_col = COLUMN_NAMES['ACTION']

    # Drop any existing action columns to force recalculation with current criteria
    # This ensures ROE/DE and other new criteria are properly applied
    existing_action_cols = [col for col in ["BS", "ACTION", "ACT", "action"] if col in df.columns]
    if existing_action_cols:
        logger.info(f"Dropping existing action columns {existing_action_cols} to force recalculation")
        df = df.drop(columns=existing_action_cols)

    # Create reverse aliases for short column names to long names
    # This allows _calculate_actions() to find ROE/DE when reading from CSV
    if 'ROE' in df.columns and 'return_on_equity' not in df.columns:
        df['return_on_equity'] = df['ROE']
    if 'DE' in df.columns and 'debt_to_equity' not in df.columns:
        df['debt_to_equity'] = df['DE']

    # Always calculate actions using current trading criteria
    df[bs_col] = calculate_actions(df)

    # Apply number formatting based on FORMATTERS configuration
    formatters = DISPLAY.get("FORMATTERS", {})

    # Column mapping from display name to formatter key
    format_mapping = {
        'PRICE': 'price',
        'TARGET': 'target_price',
        'UPSIDE': 'upside',
        '%BUY': 'buy_percentage',  # Updated to match new column header
        'BETA': 'beta',
        'PET': 'pe_trailing',
        'PEF': 'pe_forward',
        'PEG': 'peg_ratio',
        'DIV%': 'dividend_yield',  # Updated to match new column header
        'SI': 'short_float_pct',
        'EXRET': 'exret',
        'ROE': 'return_on_equity',
        'DE': 'debt_to_equity'
    }

    # Clean up NaN, None, and 0 values with "--" for better display
    df = df.copy()  # Don't modify original

    # List of columns that should show "--" for 0 values (percentages, counts, etc.)
    zero_to_dash_cols = [COLUMN_NAMES['ANALYST_COUNT'], COLUMN_NAMES['BUY_PERCENTAGE'],
                        COLUMN_NAMES['TOTAL_RATINGS'], "SI", COLUMN_NAMES['DIVIDEND_YIELD_DISPLAY']]

    # Special handling for EARNINGS date formatting
    if "EARNINGS" in df.columns:
        def format_earnings_date(value):
            try:
                if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    return "--"
                date_str = str(value).strip()
                # Remove dashes from date format (YYYY-MM-DD -> YYYYMMDD)
                if len(date_str) >= 10 and "-" in date_str:
                    return date_str.replace("-", "")[:8]  # YYYYMMDD format
                elif len(date_str) >= 8:
                    return date_str[:8]  # Already in YYYYMMDD format
                return date_str
            except Exception:
                return "--"

        df["EARNINGS"] = df["EARNINGS"].apply(format_earnings_date)

    # Special handling for ROE formatting (already in percentage from API)
    if "ROE" in df.columns:
        def format_roe(value):
            try:
                if value is None or value == '' or value == '--':
                    return "--"
                # Try to convert to float (handles both numeric and string inputs from CSV/API)
                float_value = float(value)
                if math.isnan(float_value) or math.isinf(float_value) or float_value == 0:
                    return "--"
                return f"{float_value:.1f}"
            except (ValueError, TypeError):
                return "--"

        df["ROE"] = df["ROE"].apply(format_roe)

    # Special handling for DE formatting (1 decimal place)
    if "DE" in df.columns:
        def format_de(value):
            try:
                if value is None or value == '' or value == '--':
                    return "--"
                # Try to convert to float (handles both numeric and string inputs from CSV/API)
                float_value = float(value)
                if math.isnan(float_value) or math.isinf(float_value) or float_value == 0:
                    return "--"
                return f"{float_value:.1f}"
            except (ValueError, TypeError):
                return "--"

        df["DE"] = df["DE"].apply(format_de)

    for col in df.columns:
        if col not in ["#", "TICKER", "COMPANY", "EARNINGS", "ROE", "DE"]:  # Don't format these columns
            # Apply specific formatting if configured
            if col in format_mapping:
                formatter_key = format_mapping[col]
                formatter_config = formatters.get(formatter_key, {})

                # Format numbers using the configuration
                formatted_values = []
                for value in df[col]:
                    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                        formatted_values.append("--")
                    elif value == 0 and col in zero_to_dash_cols:
                        formatted_values.append("--")
                    elif isinstance(value, (int, float)) and value != 0:
                        formatted_values.append(format_number(
                            value,
                            precision=formatter_config.get('precision', 2),
                            as_percentage=formatter_config.get('as_percentage', False)
                        ))
                    else:
                        formatted_values.append(str(value) if value not in ["nan", "NaN"] else "--")

                df[col] = formatted_values
            else:
                # Default cleanup for non-configured columns
                # Replace NaN, None, and "nan" string with "--"
                df[col] = df[col].replace([float('nan'), None, "nan", "NaN"], "--")

                # Replace 0 with "--" for specific columns
                if col in zero_to_dash_cols:
                    df[col] = df[col].replace(0, "--")
                    df[col] = df[col].replace(0.0, "--")

    # Remove helper columns used for sorting
    drop_cols = ["_sort_exret", "_sort_earnings", "_not_found"]
    existing_cols = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=existing_cols)


def calculate_actions(df: pd.DataFrame) -> pd.Series:
    """Calculate trading actions for each row in the DataFrame."""
    actions = []

    for _, row in df.iterrows():
        try:
            # Import the calculation function and trading criteria
            from yahoofinance.utils.trade_criteria import calculate_action_for_row
            from yahoofinance.core.config import TRADING_CRITERIA
            action, _ = calculate_action_for_row(row, TRADING_CRITERIA, "short_percent")
            # Preserve all valid actions including "I" (Inconclusive), only default empty/None to "H"
            actions.append(action if action is not None and action != "" else "H")
        except Exception:
            # Fallback action calculation
            actions.append("H")  # Default to Hold

    return pd.Series(actions, index=df.index)


def display_stock_table(stock_data: List[Dict[str, Any]], title: str = "Stock Analysis") -> None:
    """
    Display a table of stock data in the console.

    Args:
        stock_data: List of stock data dictionaries
        title: Title for the table
    """
    if not stock_data:
        return

    # Convert to DataFrame
    df = pd.DataFrame(stock_data)

    # Format for display FIRST (before any sorting that depends on formatted values)
    df = format_dataframe(df)

    # Add position size calculation
    try:
        df = add_position_size_column(df)
    except Exception as e:
        # If position size calculation fails, continue without it
        logger.warning(f"Failed to add position size column: {e}")
        # Ensure SIZE column exists for column filtering
        df['SIZE'] = '--'

    # Sort data AFTER all formatting and calculations are complete
    df = sort_market_data(df)

    # Add position numbers AFTER all sorting and formatting is complete
    if "#" not in df.columns:
        df.insert(0, "#", range(1, len(df) + 1))

    # Get the standard column order from config
    bs_col = COLUMN_NAMES['ACTION']

    # Only include columns that exist in both the DataFrame and standard columns
    final_col_order = [col for col in STANDARD_DISPLAY_COLUMNS if col in df.columns]

    # If we have fewer than 5 essential columns, fall back to basic set
    essential_cols = ["#", "TICKER", "COMPANY", "PRICE", bs_col]
    if len(final_col_order) < 5:
        final_col_order = [col for col in essential_cols if col in df.columns]

    # Reorder the DataFrame to only show standard display columns
    df = df[final_col_order]

    # Apply color coding based on ACTION column
    colored_data = []
    for _, row in df.iterrows():
        colored_row = row.copy()

        # Apply color based on ACTION or BS value
        action = row.get(bs_col, "") if bs_col in row else row.get("ACTION", "")
        if action == "B":  # BUY
            colored_row = {k: f"\033[92m{v}\033[0m" for k, v in colored_row.items()}  # Green
        elif action == "S":  # SELL
            colored_row = {k: f"\033[91m{v}\033[0m" for k, v in colored_row.items()}  # Red
        elif action == "I":  # INCONCLUSIVE
            colored_row = {k: f"\033[93m{v}\033[0m" for k, v in colored_row.items()}  # Yellow
        # No special coloring for HOLD ('H')

        # Keep column order
        colored_data.append([colored_row.get(col, "") for col in df.columns])

    # Define column alignment based on content type
    colalign = []
    for col in df.columns:
        if col in ["TICKER", "COMPANY"]:
            colalign.append("left")
        elif col == "#":
            colalign.append("right")
        else:
            colalign.append("right")

    # Display the table without title/generation time

    # Use tabulate for display with the defined alignment and fancy_grid format
    table = tabulate(
        colored_data if colored_data else df.values,
        headers=df.columns,
        tablefmt="fancy_grid",
        colalign=colalign,
    )
    print(table)

    # Display processing statistics if available
    from yahoofinance.utils.async_utils.enhanced import display_processing_stats
    display_processing_stats()


def add_position_size_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add position size column and new metrics (EG, PP) to DataFrame.

    Args:
        df: DataFrame with market data

    Returns:
        DataFrame with SIZE, EG, and PP columns added
    """
    if df.empty:
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Add EG and PP columns first, but only if they don't already exist with valid data
    if 'EG' not in df.columns or df['EG'].isna().all() or (df['EG'] == '--').all():
        earnings_growths = []

        for _, row in df.iterrows():
            # Get earnings growth from the row if available, or set default
            # Check multiple possible column names
            earnings_growth = row.get('earnings_growth', row.get('EG', None))
            if earnings_growth is not None and earnings_growth != '--':
                try:
                    # Convert to percentage if it's in decimal form
                    eg_value = float(earnings_growth)
                    if abs(eg_value) <= 1:  # Likely in decimal form (0.15 = 15%)
                        eg_value *= 100
                    # Display all earnings growth values
                    earnings_growths.append(f"{eg_value:.1f}%")
                except (ValueError, TypeError):
                    earnings_growths.append("--")
            else:
                earnings_growths.append("--")

        # Add the new column
        df['EG'] = earnings_growths

    if 'PP' not in df.columns or df['PP'].isna().all() or (df['PP'] == '--').all():
        three_month_perfs = []

        for _, row in df.iterrows():
            # Use pre-calculated 12-month performance from provider (no additional API calls)
            # Check multiple possible column names
            twelve_month_perf = row.get('twelve_month_performance', row.get('PP', None))
            if twelve_month_perf is not None and twelve_month_perf != '--':
                try:
                    pp_value = float(twelve_month_perf)
                    three_month_perfs.append(f"{pp_value:.1f}%")
                except (ValueError, TypeError):
                    three_month_perfs.append("--")
            else:
                three_month_perfs.append("--")

        # Add the new column
        df['PP'] = three_month_perfs

    # Calculate position sizes with new criteria
    position_sizes = []
    for i, row in df.iterrows():
        # Get market cap value
        market_cap = None
        if 'CAP' in row:
            market_cap_raw = row['CAP']
            if market_cap_raw and market_cap_raw != '--':
                # Parse market cap from formatted string (e.g., "3.14T" -> 3140000000000)
                market_cap = parse_market_cap_value(market_cap_raw)

        # Get EXRET value
        exret = None
        if 'EXRET' in row:
            exret_raw = row['EXRET']
            if exret_raw and exret_raw != '--':
                # Parse EXRET from percentage string (e.g., "6.3%" -> 6.3)
                exret = parse_percentage_value(exret_raw)

        # Get earnings growth value
        earnings_growth_value = None
        eg_str = df.loc[i, 'EG']
        if eg_str and eg_str != '--':
            try:
                # Handle both string and numeric values
                if isinstance(eg_str, str):
                    earnings_growth_value = float(eg_str.rstrip('%'))
                else:
                    earnings_growth_value = float(eg_str)
            except (ValueError, TypeError):
                pass

        # Get 3-month performance value
        three_month_perf_value = None
        mop_str = df.loc[i, 'PP']
        if mop_str and mop_str != '--':
            try:
                # Handle both string and numeric values
                if isinstance(mop_str, str):
                    three_month_perf_value = float(mop_str.rstrip('%'))
                else:
                    three_month_perf_value = float(mop_str)
            except (ValueError, TypeError):
                pass

        # Get beta value
        beta_value = None
        if 'BETA' in row:
            beta_raw = row['BETA']
            if beta_raw and beta_raw != '--':
                try:
                    beta_value = float(beta_raw)
                except (ValueError, TypeError):
                    pass

        # Get ticker for ETF/commodity detection
        ticker = row.get('TICKER', '') if 'TICKER' in row else ''

        # Calculate position size with new criteria
        position_size = calculate_position_size(
            market_cap, exret, ticker, earnings_growth_value, three_month_perf_value, beta_value
        )
        position_sizes.append(position_size)

    # Add formatted position size column
    df['SIZE'] = [format_position_size(size) for size in position_sizes]

    return df


def parse_market_cap_value(market_cap_str: str) -> Optional[float]:
    """
    Parse market cap string to numeric value.

    Args:
        market_cap_str: Market cap string (e.g., "3.14T", "297B")

    Returns:
        Market cap value in USD or None if parsing fails
    """
    if not market_cap_str or market_cap_str == '--':
        return None

    try:
        # If it's already a number, return it
        if isinstance(market_cap_str, (int, float)):
            return float(market_cap_str)

        # Remove any whitespace
        market_cap_str = str(market_cap_str).strip()

        # Handle different suffixes
        if market_cap_str.endswith('T'):
            return float(market_cap_str[:-1]) * 1_000_000_000_000
        elif market_cap_str.endswith('B'):
            return float(market_cap_str[:-1]) * 1_000_000_000
        elif market_cap_str.endswith('M'):
            return float(market_cap_str[:-1]) * 1_000_000
        elif market_cap_str.endswith('K'):
            return float(market_cap_str[:-1]) * 1_000
        else:
            # Try to parse as raw number
            return float(market_cap_str)
    except (ValueError, TypeError):
        return None


def parse_percentage_value(percentage_str) -> Optional[float]:
    """
    Parse percentage string or float to numeric value.

    Args:
        percentage_str: Percentage string (e.g., "6.3%", "-2.2%") or float

    Returns:
        Percentage value as float or None if parsing fails
    """
    if not percentage_str or percentage_str == '--':
        return None

    try:
        # If it's already a float, return it
        if isinstance(percentage_str, (int, float)):
            return float(percentage_str)

        # Remove % sign and any whitespace
        clean_str = str(percentage_str).replace('%', '').strip()
        return float(clean_str)
    except (ValueError, TypeError):
        return None

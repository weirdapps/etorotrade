"""
Helper functions for display and reporting in trade.py.

This module provides utility functions to help reduce cognitive complexity
in the main trade.py module, particularly for display_report_for_source function.
"""

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from yahoofinance.core.config import COLUMN_NAMES
from yahoofinance.core.trade_criteria_config import TradingCriteria
from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)
from ..utils.trade_criteria import calculate_action_for_row, normalize_row_columns

from ..core.logging import get_logger


# Configure logger
logger = get_logger(__name__)

# Constants for column names
BUY_PERCENTAGE = COLUMN_NAMES["BUY_PERCENTAGE"]
TICKER_COL = "TICKER"
COMPANY_COL = "COMPANY"
ACTION_COL = "ACTION"
SECTOR_COL = "SECTOR"
PRICE_COL = "PRICE"
TARGET_COL = "TARGET"
UPSIDE_COL = "UPSIDE"
EXRET_COL = "EXRET"
PET_COL = "PET"
PEF_COL = "PEF"
PEG_COL = "PEG"
BETA_COL = "BETA"
SI_COL = "SI"
CAP_COL = "CAP"
RANK_COL = "#"
ANALYST_COUNT_COL = "# A"
PRICE_TARGET_COUNT_COL = "# T"


def handle_manual_input_tickers(tickers):
    """
    Process manual input tickers string into a list of individual tickers.

    Args:
        tickers: Raw ticker input which might be comma or space separated

    Returns:
        List of cleaned ticker symbols
    """
    # Process the ticker string (it might be comma or space separated)
    tickers_str = " ".join(tickers)
    tickers_list = []

    # Split by common separators (comma, space, semicolon)
    for ticker in re.split(r"[,;\s]+", tickers_str):
        ticker = ticker.strip().upper()
        if ticker:  # Skip empty strings
            tickers_list.append(ticker)

    # Silent processing - no output for clean display
    return tickers_list


def get_output_file_and_title(report_source, output_dir):
    """
    Determine the appropriate output file and report title based on the source.

    Args:
        report_source: Source type ('P', 'M', 'I')
        output_dir: Directory to store output files

    Returns:
        Tuple containing (output_file, report_title)
    """
    if report_source == "M":
        return f"{output_dir}/market.csv", "Market Analysis"
    elif report_source == "P":
        return f"{output_dir}/portfolio.csv", "Portfolio Analysis"
    else:
        return f"{output_dir}/manual.csv", "Manual Ticker Analysis"


def format_market_cap(row):
    """
    Format market cap values for display in the CAP column.

    Args:
        row: DataFrame row

    Returns:
        Formatted market cap string
    """
    # Try different possible market cap column names
    market_cap = None
    for col in ["market_cap", "marketCap", "market_cap_value"]:
        if col in row and pd.notna(row[col]):
            market_cap = row[col]
            break

    if market_cap is not None:
        value = market_cap
        # Apply size-based formatting
        if value >= 1e12:
            if value >= 10e12:
                return f"{value / 1e12:.1f}T"
            else:
                return f"{value / 1e12:.2f}T"
        elif value >= 1e9:
            if value >= 100e9:
                return f"{int(value / 1e9)}B"
            elif value >= 10e9:
                return f"{value / 1e9:.1f}B"
            else:
                return f"{value / 1e9:.2f}B"
        elif value >= 1e6:
            if value >= 100e6:
                return f"{int(value / 1e6)}M"
            elif value >= 10e6:
                return f"{value / 1e6:.1f}M"
            else:
                return f"{value / 1e6:.2f}M"
        else:
            return f"{int(value):,}"
    return "--"


def _apply_color_to_row(colored_row, color_code):
    """
    Apply the specified color to all cells in a row.

    Args:
        colored_row: Row to color
        color_code: ANSI color code (e.g., '92' for green)

    Returns:
        Row with color applied to all cells
    """
    for col in colored_row.index:
        val = colored_row[col]
        colored_row[col] = f"\033[{color_code}m{val}\033[0m"
    return colored_row


# Import action and color constants from centralized config
from yahoofinance.core.trade_criteria_config import (
    BUY_ACTION,
    SELL_ACTION,
    HOLD_ACTION,
    INCONCLUSIVE_ACTION,
    GREEN_COLOR,
    RED_COLOR,
    YELLOW_COLOR,
    get_action_color
)


def _color_based_on_action(row, colored_row):
    """
    Apply color based on the ACTION column value.

    Args:
        row: Original row with ACTION column
        colored_row: Row to apply colors to

    Returns:
        Colored row or None if no action found
    """
    # Check for both ACT and ACTION columns
    action = None

    # Check ACT column first
    if (
        "ACT" in row
        and not pd.isna(row["ACT"])
        and row["ACT"] in [BUY_ACTION, SELL_ACTION, HOLD_ACTION, "I"]
    ):
        action = row["ACT"]
        # Make a copy so we don't modify the original row
        colored_row_copy = colored_row.copy()
        # Ensure ACT column shows the correct value in display
        if "ACT" in colored_row_copy:
            colored_row_copy["ACT"] = action

    # Check ACTION column next
    elif (
        ACTION_COL in row
        and not pd.isna(row[ACTION_COL])
        and row[ACTION_COL] in [BUY_ACTION, SELL_ACTION, HOLD_ACTION, "I"]
    ):
        action = row[ACTION_COL]
        # Make a copy so we don't modify the original row
        colored_row_copy = colored_row.copy()
        # Ensure ACTION column shows the correct value in display
        if ACTION_COL in colored_row_copy:
            colored_row_copy[ACTION_COL] = action
    else:
        return None

    # Apply coloring based on the action
    if action == BUY_ACTION:  # BUY - Green
        return _apply_color_to_row(colored_row_copy, GREEN_COLOR)
    elif action == SELL_ACTION:  # SELL - Red
        return _apply_color_to_row(colored_row_copy, RED_COLOR)
    elif action == "I":  # INCONCLUSIVE - Yellow
        return _apply_color_to_row(colored_row_copy, YELLOW_COLOR)

    # HOLD - no color changes
    return colored_row_copy


def _has_required_metrics(row):
    """
    Check if row has all required metrics for criteria evaluation.

    Args:
        row: DataFrame row to check

    Returns:
        Boolean indicating if required metrics are present
    """
    required_columns = [EXRET_COL, UPSIDE_COL, BUY_PERCENTAGE, SI_COL, PEF_COL, BETA_COL]
    return all(col in row and pd.notna(row[col]) for col in required_columns)


def _color_based_on_criteria(row, colored_row):
    """
    Apply color based on trading criteria evaluation.

    Args:
        row: Original row with metrics
        colored_row: Row to apply colors to

    Returns:
        Colored row
    """
    # Use the centralized action calculation
    action = get_action_from_row(row)

    if not action:  # Empty action means insufficient confidence/data
        # INCONCLUSIVE (yellow)
        return _apply_color_to_row(colored_row, YELLOW_COLOR)

    if action == "S":  # SELL
        return _apply_color_to_row(colored_row, RED_COLOR)

    if action == "B":  # BUY
        return _apply_color_to_row(colored_row, GREEN_COLOR)

    # Default is HOLD - no color changes
    return colored_row


def apply_color_formatting(row, trading_criteria):
    """
    Apply color formatting to a row based on trading criteria.

    Args:
        row: DataFrame row to color
        trading_criteria: Trading criteria from config

    Returns:
        Colored row with ANSI color codes
    """
    colored_row = row.copy()

    try:
        # First try to color based on ACTION column if available
        colored = _color_based_on_action(row, colored_row)
        if colored is not None:
            return colored

        # Fallback to evaluating criteria directly if required metrics are available
        if _has_required_metrics(row):
            return _color_based_on_criteria(row, colored_row)

    except YFinanceError as e:
        # If any error in color logic, use the original row
        logger.debug(f"Error applying color: {str(e)}")

    return colored_row


def _extract_numeric_value(row, column_name):
    """
    Safely extract a numeric value from a row.

    Args:
        row: DataFrame row
        column_name: Name of the column to extract

    Returns:
        Numeric value or None if invalid
    """
    if column_name not in row or pd.isna(row[column_name]) or row[column_name] == "--":
        return None

    try:
        return float(str(row[column_name]).replace(",", ""))
    except (ValueError, TypeError):
        return None


def check_confidence_threshold(row, trading_criteria):
    """
    Check if the row meets confidence thresholds for analysts and price targets.

    Args:
        row: DataFrame row
        trading_criteria: Trading criteria from config

    Returns:
        Boolean indicating if confidence threshold is met
    """
    # Extract values using helper function
    analyst_count = _extract_numeric_value(row, ANALYST_COUNT_COL)
    price_targets = _extract_numeric_value(row, PRICE_TARGET_COUNT_COL)

    # Get threshold values from centralized criteria
    min_analysts = TradingCriteria.MIN_ANALYST_COUNT
    min_targets = TradingCriteria.MIN_PRICE_TARGETS

    # Check if confidence threshold is met
    return (
        analyst_count is not None
        and price_targets is not None
        and analyst_count >= min_analysts
        and price_targets >= min_targets
    )


def _parse_percentage_value(value):
    """
    Parse a percentage value, handling string formatting.

    Args:
        value: Value to parse, can be string or numeric

    Returns:
        Parsed numeric value
    """
    if isinstance(value, str):
        return float(value.rstrip("%"))
    return value


def _parse_numeric_value(value):
    """
    Parse a numeric value, handling string formatting and special values.

    Args:
        value: Value to parse, can be string or numeric

    Returns:
        Parsed numeric value
    """
    if isinstance(value, str) and value != "--":
        return float(value)
    return value


def parse_row_values(row):
    """
    Parse numeric values from a row, handling string formatting.

    Args:
        row: DataFrame row

    Returns:
        Dictionary of parsed values
    """
    return {
        "upside": _parse_percentage_value(row["UPSIDE"]),
        "buy_pct": _parse_percentage_value(row[BUY_PERCENTAGE]),
        "si": _parse_percentage_value(row["SI"]),
        "pef": _parse_numeric_value(row["PEF"]),
        "beta": _parse_numeric_value(row["BETA"]),
    }


def get_action_from_row(row, short_field="short_percent"):
    """
    Get action (B/S/H) from a row using the official trade criteria.

    Args:
        row: DataFrame row with metrics
        short_field: Name of the field containing short interest data

    Returns:
        String action code ('B', 'S', 'H', or '')
    """
    # Use the central criteria calculation function
    action, _ = calculate_action_for_row(row, {}, short_field)
    return action


# Constants for status messages
PASS_STATUS = "PASS"
INCONCLUSIVE_STATUS = "INCONCLUSIVE"


def _format_color_text(text, color_code):
    """
    Format text with ANSI color code.

    Args:
        text: Text to format
        color_code: ANSI color code

    Returns:
        Colored text
    """
    return f"\033[{color_code}m{text}\033[0m"


@with_retry
def create_display(output_format="console", **kwargs):
    """
    Factory function to create an appropriate display instance.

    This function creates and returns a display appropriate for the
    specified output format. It acts as a bridge to the formatter system.

    Args:
        output_format: The desired output format ('console', 'html', etc.)
        **kwargs: Additional keyword arguments passed to display constructor

    Returns:
        An appropriate display instance for the specified format

    Raises:
        ValidationError: When the output format is invalid or not supported
    """
    from yahoofinance.core.errors import ValidationError
    from yahoofinance.core.logging import get_logger

    logger = get_logger(__name__)

    try:
        if output_format == "console":
            # Import console display dynamically to avoid circular imports
            from yahoofinance.presentation.console import ConsoleDisplay

            return ConsoleDisplay(**kwargs)

        elif output_format == "html":
            # Import HTML display dynamically to avoid circular imports
            from yahoofinance.presentation.html import HTMLDisplay

            return HTMLDisplay(**kwargs)

        else:
            logger.error(f"Invalid display format: {output_format}")
            raise ValidationError(f"Invalid display format: {output_format}")

    except Exception as e:
        logger.error(f"Error creating display: {str(e)}")
        # Fall back to basic display if possible
        try:
            from yahoofinance.presentation.console import ConsoleDisplay

            logger.info("Falling back to console display")
            return ConsoleDisplay(**kwargs)
        except Exception:
            raise ValidationError(f"Could not create any display: {str(e)}")


def _get_confidence_status(analyst_count, price_targets, min_analysts, min_targets):
    """
    Determine the confidence status based on analyst and price target counts.

    Args:
        analyst_count: Number of analysts covering the stock
        price_targets: Number of price targets available
        min_analysts: Minimum required analyst count
        min_targets: Minimum required price target count

    Returns:
        Tuple of (boolean indicating if confidence met, formatted status string)
    """
    confidence_met = (
        analyst_count is not None
        and price_targets is not None
        and not isinstance(analyst_count, str)
        and not isinstance(price_targets, str)
        and analyst_count >= min_analysts
        and price_targets >= min_targets
    )

    if confidence_met:
        status = _format_color_text(PASS_STATUS, GREEN_COLOR)
    else:
        status = f"{_format_color_text(INCONCLUSIVE_STATUS, YELLOW_COLOR)} (min: {min_analysts}A/{min_targets}T)"

    return confidence_met, status


def _format_metric_value(value, default="N/A"):
    """
    Format a metric value for display, handling None values.

    Args:
        value: Value to format
        default: Default value if None

    Returns:
        Formatted value
    """
    if isinstance(value, str) and "%" in value:
        return value.replace("%", "")
    return value if pd.notna(value) else default


def print_confidence_details(row, i, trading_criteria):
    """
    Create and return confidence details string for a row.

    Args:
        row: DataFrame row
        i: Row index
        trading_criteria: Trading criteria from config

    Returns:
        Formatted string with confidence details
    """
    # Get basic ticker info
    ticker = row.get("TICKER", f"row_{i}")
    action = row.get("ACTION", "N/A")

    # Extract analyst and price target values
    try:
        analyst_count = _extract_numeric_value(row, "# A")
        if analyst_count is None:
            analyst_count = "Invalid"
    except YFinanceError:
        analyst_count = "Invalid"

    try:
        price_targets = _extract_numeric_value(row, "# T")
        if price_targets is None:
            price_targets = "Invalid"
    except YFinanceError:
        price_targets = "Invalid"

    # Get confidence thresholds from centralized criteria
    min_analysts = TradingCriteria.MIN_ANALYST_COUNT
    min_targets = TradingCriteria.MIN_PRICE_TARGETS

    # Determine confidence status
    _, confidence_status = _get_confidence_status(
        analyst_count, price_targets, min_analysts, min_targets
    )

    # Get key metrics
    metrics = {
        "SI": _format_metric_value(row.get("SI", "N/A")),
        "PEF": _format_metric_value(row.get("PEF", "N/A")),
        "BETA": _format_metric_value(row.get("BETA", "N/A")),
        "UPSIDE": _format_metric_value(row.get("UPSIDE", "N/A")),
        "BUY%": _format_metric_value(row.get(BUY_PERCENTAGE, "N/A")),
    }

    # Return formatted confidence details string
    return (
        f"{ticker}: ACTION={action}, CONFIDENCE={confidence_status}, "
        f"ANALYSTS={analyst_count}/{min_analysts}, TARGETS={price_targets}/{min_targets}, "
        f"UPSIDE={metrics['UPSIDE']}, BUY%={metrics['BUY%']}, SI={metrics['SI']}, "
        f"PEF={metrics['PEF']}, BETA={metrics['BETA']}"
    )


def generate_html_dashboard(result_df, display_df, report_source, output_dir):
    """
    Generate HTML dashboard for the report.

    Args:
        result_df: Original result DataFrame
        display_df: Display DataFrame with formatted values
        report_source: Source type ('P', 'M', 'I')
        output_dir: Directory to store output files
    """
    try:
        # Import necessary modules
        from yahoofinance.presentation.html import generate_dashboard

        # Generate dashboard with all available data
        if not result_df.empty:
            # Add action column to result_df as it may not be there yet
            if "action" not in result_df.columns and "ACTION" in display_df.columns:
                # Map display ACTION values to result_df
                actions_map = {
                    row["TICKER"]: row["ACTION"]
                    for _, row in display_df.iterrows()
                    if "TICKER" in row and "ACTION" in row
                }
                result_df["action"] = result_df["ticker"].map(actions_map)

            # Generate appropriate dashboard based on source
            if report_source == "M":
                # Market dashboard
                output_path = f"{output_dir}/index.html"
                title = "Market Analysis"

                # Generate the dashboard silently
                generate_dashboard(result_df, output_path, title)

            elif report_source == "P":
                # Portfolio dashboard
                output_path = f"{output_dir}/portfolio_dashboard.html"
                title = "Portfolio Analysis"

                # Generate the dashboard silently
                generate_dashboard(result_df, output_path, title, is_portfolio=True)
    except YFinanceError as e:
        logger.error(f"Error generating HTML dashboard: {str(e)}")


def get_display_columns(colored_df):
    """
    Get the list of columns to display based on the DataFrame content.

    Args:
        colored_df: DataFrame with colored values

    Returns:
        List of column names to display
    """
    # Base columns for all views
    base_cols = [
        RANK_COL,
        TICKER_COL,
        COMPANY_COL,
        PRICE_COL,
        TARGET_COL,
        UPSIDE_COL,
        BUY_PERCENTAGE,
        EXRET_COL,
        PET_COL,
        PEF_COL,
        PEG_COL,
        BETA_COL,
        SI_COL,
        CAP_COL,
        ACTION_COL,
    ]

    # Add SECTOR column for market analysis view
    if SECTOR_COL in colored_df.columns:
        # Insert SECTOR after COMPANY
        company_index = base_cols.index(COMPANY_COL)
        display_cols = (
            base_cols[: company_index + 1] + [SECTOR_COL] + base_cols[company_index + 1 :]
        )
    else:
        display_cols = base_cols

    # Filter to only include columns that exist in the DataFrame
    return [col for col in display_cols if col in colored_df.columns]


def display_tabulate_results(colored_df, colalign):
    """
    Display results using tabulate with appropriate column selections.

    Args:
        colored_df: DataFrame with colored values
        colalign: Column alignment dictionary
    """
    from tabulate import tabulate

    try:
        # Get the display columns
        display_cols = get_display_columns(colored_df)

        # Get column alignments for the display columns
        column_alignments = [colalign.get(col, "left") for col in display_cols]

        # Display using tabulate
        print(
            tabulate(
                colored_df[display_cols],
                headers="keys",
                tablefmt="simple",
                showindex=False,
                colalign=column_alignments,
            )
        )

    except YFinanceError as e:
        # Fall back to basic display if tabulate fails
        logger.error(f"Error in tabulate display: {str(e)}")
        print(colored_df.to_string())

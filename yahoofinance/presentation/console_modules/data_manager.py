"""
Data management utilities for loading and saving market data.

This module provides data loading from files, CSV saving, and report orchestration.
"""

import asyncio
import csv
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from yahoofinance.api.providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from yahoofinance.core.config import COLUMN_NAMES, FILE_PATHS, MESSAGES, PATHS, RATE_LIMIT, get_max_concurrent_requests
from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.logging import get_logger
from yahoofinance.presentation.html import HTMLGenerator
from yahoofinance.utils.data.ticker_utils import normalize_ticker
from yahoofinance.utils.error_handling import with_retry


logger = get_logger(__name__)


def load_tickers(source_type: str) -> List[str]:
    """
    Load tickers from file based on source type.

    Args:
        source_type: Source type for tickers ('P' for portfolio, 'M' for market,
                    'E' for eToro, 'I' for manual input, 'U' for USA market,
                    'C' for China market, 'EU' for Europe market)

    Returns:
        List of tickers
    """
    # Get input directory from MARKET_FILE path by removing the filename
    input_dir = os.path.dirname(FILE_PATHS["MARKET_FILE"])

    if source_type == "P":
        return _load_tickers_from_file(
            FILE_PATHS["PORTFOLIO_FILE"],
            ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
        )
    elif source_type == "M":
        # For market, check if we need to prompt for which market file to use
        try:
            market_choice = (
                input("Select market: USA (U), Europe (E), China (C), or Manual (M)? ")
                .strip()
                .upper()
            )
        except EOFError:
            market_choice = "U"

        if market_choice == "U":
            return _load_tickers_from_file(
                os.path.join(input_dir, "usa.csv"),
                ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
            )
        elif market_choice == "E":
            return _load_tickers_from_file(
                os.path.join(input_dir, "europe.csv"),
                ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
            )
        elif market_choice == "C":
            return _load_tickers_from_file(
                os.path.join(input_dir, "china.csv"),
                ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
            )
        else:  # Default to using the main market.csv file
            return _load_tickers_from_file(
                FILE_PATHS["MARKET_FILE"],
                ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
            )
    elif source_type == "E":
        return _load_tickers_from_file(
            FILE_PATHS["ETORO_FILE"],
            ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
        )
    elif source_type == "I":
        # For Manual Input, call the method directly and don't rely on the decorator
        # This avoids the issue with the decorator returning a function
        try:
            result = _get_manual_tickers()
            if callable(result):
                return result()
            return result
        except Exception:
            return ["AAPL", "MSFT"]  # Default tickers for error cases
    else:
        raise ValueError(f"Unknown source type: {source_type}")


def _load_tickers_from_file(file_path: str, ticker_column: List[str]) -> List[str]:
    """
    Load tickers from CSV file.

    Args:
        file_path: Path to CSV file
        ticker_column: Possible column names for tickers

    Returns:
        List of tickers
    """
    if not os.path.exists(file_path):
        return []

    try:
        tickers = []
        column_found = False

        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Find the ticker column
            found_column = None
            for col in ticker_column:
                if col in headers:
                    found_column = col
                    column_found = True
                    break

            if not column_found:
                return []

            # Read tickers
            for row in reader:
                ticker = row.get(found_column, "").strip()
                if ticker:
                    # Special case: convert BSX.US to BSX
                    if ticker == "BSX.US":
                        ticker = "BSX"

                    # Apply Yahoo Finance ticker format fixing for portfolio files
                    if "portfolio" in file_path.lower():
                        try:
                            from yahoofinance.utils.data.ticker_utils import normalize_ticker
                            original_ticker = ticker
                            ticker = normalize_ticker(ticker)
                            if ticker != original_ticker:
                                logger.info(f"Fixed portfolio ticker: {original_ticker} -> {ticker}")
                        except ImportError:
                            logger.warning("Could not import _fix_yahoo_ticker_format function")

                    tickers.append(ticker)

        return tickers
    except YFinanceError:
        pass  # Silent error handling
        return []


@with_retry
def _get_manual_tickers() -> List[str]:
    """
    Get tickers from manual input.

    Returns:
        List of tickers
    """
    try:
        ticker_input = input(MESSAGES["PROMPT_ENTER_TICKERS"]).strip()
        if not ticker_input:
            return []
    except EOFError:
        # Default tickers for testing in non-interactive environments
        ticker_input = "AAPL, MSFT"

    # Split by comma and clean up
    tickers = [normalize_ticker(t.strip()) for t in ticker_input.split(",") if t.strip()]
    return tickers


def filter_by_trade_action(results: List[Dict], trade_filter: str) -> List[Dict]:
    """
    Filter results by trade action with file-based filtering logic.

    For B (BUY): Check market.csv for buy opportunities NOT in portfolio.csv OR sell.csv
    For S (SELL): Check portfolio.csv for sell opportunities
    For H (HOLD): Check market.csv for hold opportunities NOT in portfolio.csv OR sell.csv

    Args:
        results: List of ticker data dictionaries
        trade_filter: Trade filter ('B' for buy, 'S' for sell, 'H' for hold)

    Returns:
        Filtered list of results
    """
    if not trade_filter or not results:
        return results

    # Load portfolio and sell file tickers for exclusion
    portfolio_tickers = set()
    sell_tickers = set()

    try:
        from yahoofinance.core.config import FILE_PATHS
        import pandas as pd
        import os

        # Load portfolio tickers
        portfolio_path = os.path.join(FILE_PATHS["INPUT_DIR"], "portfolio.csv")
        if os.path.exists(portfolio_path):
            portfolio_df = pd.read_csv(portfolio_path)
            if 'symbol' in portfolio_df.columns:
                portfolio_tickers = set(portfolio_df['symbol'].astype(str).apply(normalize_ticker))

        # Load sell file tickers (acting as notrade)
        sell_path = os.path.join(FILE_PATHS["OUTPUT_DIR"], "sell.csv")
        if os.path.exists(sell_path):
            sell_df = pd.read_csv(sell_path)
            if 'TICKER' in sell_df.columns:
                sell_tickers = set(sell_df['TICKER'].astype(str).apply(normalize_ticker))

    except Exception as e:
        logger.warning(f"Failed to load portfolio/sell files for filtering: {e}")

    filtered_results = []
    exclusion_tickers = portfolio_tickers | sell_tickers  # Union of both sets

    for ticker_data in results:
        try:
            ticker = normalize_ticker(str(ticker_data.get('symbol', ticker_data.get('ticker', ''))))

            # Calculate action for this ticker
            from yahoofinance.utils.trade_criteria import calculate_action_for_row
            from yahoofinance.core.config import TRADING_CRITERIA
            action, _ = calculate_action_for_row(ticker_data, TRADING_CRITERIA, "short_percent")

            # Apply file-based filtering logic
            if trade_filter == "B":
                # BUY: market opportunities not in portfolio or sell files
                if action == "B" and ticker not in exclusion_tickers:
                    filtered_results.append(ticker_data)
            elif trade_filter == "S":
                # SELL: portfolio opportunities only
                if action == "S" and ticker in portfolio_tickers:
                    filtered_results.append(ticker_data)
            elif trade_filter == "H":
                # HOLD: market opportunities not in portfolio or sell files
                if action == "H" and ticker not in exclusion_tickers:
                    filtered_results.append(ticker_data)

        except Exception as e:
            logger.warning(f"Error filtering ticker {ticker_data.get('symbol', 'unknown')}: {e}")
            # If action calculation fails, include in HOLD filter only if not excluded
            if trade_filter == "H":
                ticker = normalize_ticker(str(ticker_data.get('symbol', ticker_data.get('ticker', ''))))
                if ticker not in exclusion_tickers:
                    filtered_results.append(ticker_data)

    return filtered_results


def save_to_csv(
    data: List[Dict[str, Any]],
    filename: str,
    output_dir: Optional[str] = None,
    _format_dataframe_fn = None,
    _add_position_size_fn = None,
    _sort_market_data_fn = None
) -> str:
    """
    Save data to CSV file.

    Args:
        data: List of data dictionaries to save
        filename: Name of the CSV file
        output_dir: Directory to save to (defaults to config value)
        _format_dataframe_fn: Function to format DataFrame
        _add_position_size_fn: Function to add position size column
        _sort_market_data_fn: Function to sort market data

    Returns:
        Path to saved file
    """
    try:
        # Determine output path
        output_dir = output_dir or PATHS["OUTPUT_DIR"]
        output_path = f"{output_dir}/{filename}"

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Apply same processing as display (format, then sort, add position sizes, and add tiers)
        try:
            # Format data first (same as display method)
            if _format_dataframe_fn:
                df = _format_dataframe_fn(df)
            if _add_position_size_fn:
                df = _add_position_size_fn(df)

            # Sort data AFTER all formatting and calculations are complete (same as display method)
            if _sort_market_data_fn:
                df = _sort_market_data_fn(df)

            # Apply column filtering to match display format
            from yahoofinance.core.config import STANDARD_DISPLAY_COLUMNS, COLUMN_NAMES
            bs_col = COLUMN_NAMES['ACTION']
            final_col_order = [col for col in STANDARD_DISPLAY_COLUMNS if col in df.columns]

            # If we have fewer than 5 essential columns, fall back to basic set
            essential_cols = ["#", "TICKER", "COMPANY", "PRICE", bs_col]
            if len(final_col_order) < 5:
                final_col_order = [col for col in essential_cols if col in df.columns]

            # Reorder the DataFrame to only show standard display columns
            df = df[final_col_order]

        except Exception as e:
            logger.warning(f"Failed to add position size to CSV: {e}")

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")

        # Generate corresponding HTML file using the same beautiful format as portfolio.html
        try:
            html_filename = filename.replace('.csv', '.html')
            html_path = f"{output_dir}/{html_filename}"
            html_generator = HTMLGenerator()
            stocks_data = df.to_dict('records')
            base_filename = os.path.splitext(html_filename)[0]

            # Determine title based on filename
            if 'portfolio' in filename.lower():
                title = "Portfolio Analysis"
            elif 'buy' in filename.lower():
                title = "Buy Opportunities"
            elif 'sell' in filename.lower():
                title = "Sell Candidates"
            elif 'hold' in filename.lower():
                title = "Hold Candidates"
            elif 'market' in filename.lower():
                title = "Market Analysis"
            else:
                title = "Analysis Results"

            # Rename BS column to ACTION for proper color coding
            if 'BS' in df.columns:
                df = df.rename(columns={'BS': 'ACTION'})
                stocks_data = df.to_dict('records')

            # Use generate_stock_table for beautiful color-coded HTML like portfolio.html
            html_generator.generate_stock_table(
                stocks_data=stocks_data,
                title=title,
                output_filename=base_filename,
                include_columns=list(df.columns)
            )
            logger.info(f"Generated HTML report: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML file: {str(e)}")

        return output_path
    except YFinanceError as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        raise e


def display_report(
    tickers: List[str],
    report_type: Optional[str],
    provider: Optional[AsyncFinanceDataProvider],
    display_table_fn: callable,
    process_tickers_fn: callable = None
) -> None:
    """
    Display report for tickers.

    Args:
        tickers: List of tickers to display report for
        report_type: Type of report ('M' for market, 'P' for portfolio)
        provider: Data provider instance
        display_table_fn: Function to display table
        process_tickers_fn: Function to process tickers with progress
    """
    if not tickers:
        return

    if not provider:
        return

    # Determine if the provider is async
    is_async = isinstance(provider, AsyncFinanceDataProvider)

    if is_async:
        # Handle async provider
        asyncio.run(_async_display_report(tickers, report_type, provider, display_table_fn))
    else:
        # Handle sync provider
        _sync_display_report(tickers, report_type, provider, display_table_fn, process_tickers_fn)


def _sync_display_report(
    tickers: List[str],
    report_type: Optional[str],
    provider: FinanceDataProvider,
    display_table_fn: callable,
    process_tickers_fn: callable
) -> None:
    """
    Display report for tickers using synchronous provider.

    Args:
        tickers: List of tickers to display report for
        report_type: Type of report ('M' for market, 'P' for portfolio)
        provider: Data provider instance
        display_table_fn: Function to display table
        process_tickers_fn: Function to process tickers with progress
    """
    # Process tickers with progress
    results = process_tickers_fn(
        tickers, lambda ticker: provider.get_ticker_info(ticker)
    )

    # Display results
    if results:
        # Determine report title
        title = "Portfolio Analysis" if report_type == "P" else "Market Analysis"

        # Display table
        display_table_fn(results, title)

        # Save to CSV if report type provided
        if report_type:
            if report_type == "P":
                filename = "portfolio.csv"
            elif report_type == "I":
                filename = "manual.csv"
            else:
                filename = "market.csv"

            # Import formatting functions
            from yahoofinance.presentation.console_modules.table_renderer import (
                format_dataframe,
                add_position_size_column,
                sort_market_data
            )
            save_to_csv(
                results,
                filename,
                _format_dataframe_fn=format_dataframe,
                _add_position_size_fn=add_position_size_column,
                _sort_market_data_fn=sort_market_data
            )
    else:
        pass


async def _async_display_report(
    tickers: List[str],
    report_type: Optional[str],
    provider: AsyncFinanceDataProvider,
    display_table_fn: callable,
    trade_filter: Optional[str] = None
) -> None:
    """
    Display report for tickers using asynchronous provider.

    Args:
        tickers: List of tickers to display report for
        report_type: Type of report ('M' for market, 'P' for portfolio)
        provider: Data provider instance
        display_table_fn: Function to display table
        trade_filter: Trade analysis filter (B, S, H) for filtering results
    """
    from yahoofinance.utils.async_utils.enhanced import process_batch_async

    # Silent processing - no progress messages for clean display

    # Set report type name based on context
    if trade_filter:
        if trade_filter == "S":
            report_type_name = "Portfolio"
        else:
            report_type_name = "Market"
    else:
        report_type_name = "Portfolio" if report_type == "P" else "Market"

    # Use batch processing for async provider with enhanced progress
    results_dict = await process_batch_async(
        tickers,
        provider.get_ticker_info,  # type: ignore (we know it's async)
        batch_size=1,  # Process one ticker at a time for real-time progress updates
        concurrency=get_max_concurrent_requests(),
        delay_between_batches=RATE_LIMIT.get("BATCH_DELAY", 0.0),
        description=f"Processing {report_type_name} tickers",
        show_progress=True,
    )

    # Convert dict to list, filtering out None values
    results = [result for result in results_dict.values() if result is not None]

    # Display results
    if results:
        # Apply trade filter if specified
        if trade_filter:
            results = filter_by_trade_action(results, trade_filter)
            if trade_filter == "B":
                title = "Trade Analysis - BUY Opportunities (Market data excluding portfolio/notrade)"
            elif trade_filter == "S":
                title = "Trade Analysis - SELL Opportunities (Portfolio data)"
            elif trade_filter == "H":
                title = "Trade Analysis - HOLD Opportunities (Market data excluding portfolio/notrade)"
            else:
                title = f"{report_type_name} Analysis"
        else:
            # Determine report title
            title = f"{report_type_name} Analysis"

        # Display table
        display_table_fn(results, title)

        # Display processing statistics after the table
        from yahoofinance.utils.async_utils.enhanced import display_processing_stats
        display_processing_stats()

        # Save to CSV with appropriate filename based on trade filter
        # Import formatting functions
        from yahoofinance.presentation.console_modules.table_renderer import (
            format_dataframe,
            add_position_size_column,
            sort_market_data
        )

        if trade_filter:
            if trade_filter == "B":
                filename = "buy.csv"
            elif trade_filter == "S":
                filename = "sell.csv"
            elif trade_filter == "H":
                filename = "hold.csv"
            else:
                filename = "market.csv"
            save_to_csv(
                results,
                filename,
                _format_dataframe_fn=format_dataframe,
                _add_position_size_fn=add_position_size_column,
                _sort_market_data_fn=sort_market_data
            )
        elif report_type:
            if report_type == "P":
                filename = "portfolio.csv"
            elif report_type == "I":
                filename = "manual.csv"
            else:
                filename = "market.csv"
            save_to_csv(
                results,
                filename,
                _format_dataframe_fn=format_dataframe,
                _add_position_size_fn=add_position_size_column,
                _sort_market_data_fn=sort_market_data
            )
    else:
        pass

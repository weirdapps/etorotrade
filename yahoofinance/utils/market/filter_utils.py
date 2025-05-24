"""
Market filtering utilities for Yahoo Finance data.

This module provides functions for filtering market data based on
various criteria such as market cap, sector, and performance.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..trade_criteria import evaluate_trade_criteria


def filter_by_market_cap(
    stocks: List[Dict[str, Any]], min_cap: Optional[float] = None, max_cap: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Filter stocks based on market capitalization.

    Args:
        stocks: List of stock dictionaries
        min_cap: Minimum market cap (in dollars)
        max_cap: Maximum market cap (in dollars)

    Returns:
        Filtered list of stocks
    """
    if min_cap is None and max_cap is None:
        return stocks

    filtered = []

    for stock in stocks:
        market_cap = stock.get("market_cap")

        # Skip stocks without market cap data
        if market_cap is None:
            continue

        # Apply min cap filter
        if min_cap is not None and market_cap < min_cap:
            continue

        # Apply max cap filter
        if max_cap is not None and market_cap > max_cap:
            continue

        filtered.append(stock)

    return filtered


def filter_by_sector(
    stocks: List[Dict[str, Any]],
    sectors: Optional[Union[str, List[str]]] = None,
    exclude_sectors: Optional[Union[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter stocks based on sector.

    Args:
        stocks: List of stock dictionaries
        sectors: Sector or list of sectors to include
        exclude_sectors: Sector or list of sectors to exclude

    Returns:
        Filtered list of stocks
    """
    if sectors is None and exclude_sectors is None:
        return stocks

    # Convert single values to lists
    if isinstance(sectors, str):
        sectors = [sectors]
    if isinstance(exclude_sectors, str):
        exclude_sectors = [exclude_sectors]

    # Convert to sets for faster lookups
    include_set = set(sectors) if sectors else None
    exclude_set = set(exclude_sectors) if exclude_sectors else set()

    filtered = []

    for stock in stocks:
        sector = stock.get("sector")

        # Skip stocks without sector data
        if sector is None:
            continue

        # Check if sector is excluded
        if sector in exclude_set:
            continue

        # Check if sector is included (if include list specified)
        if include_set is not None and sector not in include_set:
            continue

        filtered.append(stock)

    return filtered


def filter_by_performance(
    stocks: List[Dict[str, Any]],
    min_upside: Optional[float] = None,
    min_buy_percentage: Optional[float] = None,
    max_pe: Optional[float] = None,
    max_peg: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Filter stocks based on performance metrics.

    Args:
        stocks: List of stock dictionaries
        min_upside: Minimum upside potential percentage
        min_buy_percentage: Minimum analyst buy percentage
        max_pe: Maximum P/E ratio
        max_peg: Maximum PEG ratio

    Returns:
        Filtered list of stocks
    """
    filtered = []

    for stock in stocks:
        # Skip if any mandatory criteria is not met

        # Upside filter
        if min_upside is not None:
            upside = stock.get("upside")
            if upside is None or upside < min_upside:
                continue

        # Buy percentage filter
        if min_buy_percentage is not None:
            buy_pct = stock.get("buy_percentage")
            if buy_pct is None or buy_pct < min_buy_percentage:
                continue

        # P/E filter
        if max_pe is not None:
            pe = stock.get("pe_forward", stock.get("pe_trailing"))
            if pe is not None and pe > max_pe:
                continue

        # PEG filter
        if max_peg is not None:
            peg = stock.get("peg_ratio")
            if peg is not None and peg > max_peg:
                continue

        # Stock passed all filters
        filtered.append(stock)

    return filtered


def filter_tickers_by_criteria(
    tickers_data: List[Dict[str, Any]],
    action_filter: Optional[str] = None,
    min_upside: Optional[float] = None,
    min_buy_percentage: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_market_cap: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Filter tickers based on specified trading criteria and additional filters.

    This function first evaluates trading criteria for each ticker using the
    standard trading rules defined in the system, and then applies additional
    filters specified as parameters.

    Args:
        tickers_data: List of ticker data dictionaries
        action_filter: Filter by action type ('BUY', 'SELL', 'HOLD', 'NEUTRAL', None for all)
        min_upside: Minimum upside potential percentage
        min_buy_percentage: Minimum analyst buy percentage
        max_pe: Maximum P/E ratio
        min_market_cap: Minimum market capitalization

    Returns:
        Filtered list of tickers that match all specified criteria
    """
    # First apply the standard trading criteria
    results = []

    for ticker_data in tickers_data:
        # Skip tickers without basic required data
        if not ticker_data:
            continue

        # Evaluate trading criteria to determine action
        action = evaluate_trade_criteria(ticker_data)

        # Add action to ticker data
        ticker_data["action"] = action

        # Filter by action if specified
        if action_filter and action != action_filter:
            continue

        # Apply additional filters

        # Upside filter
        if min_upside is not None:
            upside = ticker_data.get("upside")
            if upside is None or upside < min_upside:
                continue

        # Buy percentage filter
        if min_buy_percentage is not None:
            buy_pct = ticker_data.get("buy_percentage")
            if buy_pct is None or buy_pct < min_buy_percentage:
                continue

        # P/E filter
        if max_pe is not None:
            pe = ticker_data.get("pe_forward", ticker_data.get("pe_trailing"))
            if pe is not None and pe > max_pe:
                continue

        # Market cap filter
        if min_market_cap is not None:
            market_cap = ticker_data.get("market_cap")
            if market_cap is None or market_cap < min_market_cap:
                continue

        # Ticker passed all filters
        results.append(ticker_data)

    return results

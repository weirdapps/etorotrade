"""
Market filtering utilities for Yahoo Finance data.

This module provides functions for filtering market data based on
various criteria such as market cap, sector, and performance.
"""

from typing import List, Dict, Any, Union, Optional, Tuple, Set


def filter_by_market_cap(
    stocks: List[Dict[str, Any]], 
    min_cap: Optional[float] = None, 
    max_cap: Optional[float] = None
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
        market_cap = stock.get('market_cap')
        
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
    exclude_sectors: Optional[Union[str, List[str]]] = None
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
        sector = stock.get('sector')
        
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
    max_peg: Optional[float] = None
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
            upside = stock.get('upside')
            if upside is None or upside < min_upside:
                continue
        
        # Buy percentage filter
        if min_buy_percentage is not None:
            buy_pct = stock.get('buy_percentage')
            if buy_pct is None or buy_pct < min_buy_percentage:
                continue
        
        # P/E filter
        if max_pe is not None:
            pe = stock.get('pe_forward', stock.get('pe_trailing'))
            if pe is not None and pe > max_pe:
                continue
        
        # PEG filter
        if max_peg is not None:
            peg = stock.get('peg_ratio')
            if peg is not None and peg > max_peg:
                continue
        
        # Stock passed all filters
        filtered.append(stock)
    
    return filtered
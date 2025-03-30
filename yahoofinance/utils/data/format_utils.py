"""
Formatting utilities for data display.

This module provides functions for formatting data for display in
tables, HTML, CSV, and other formats.
"""

from typing import Any, Dict, List, Union, Optional
import math


def format_number(value: Any, precision: int = 2, as_percentage: bool = False,
                 include_sign: bool = False, abbreviate: bool = False) -> str:
    """
    Format a number for display.
    
    Args:
        value: The value to format
        precision: Number of decimal places
        as_percentage: Whether to format as percentage
        include_sign: Whether to include + sign for positive values
        abbreviate: Whether to abbreviate large numbers (K, M, B, T)
        
    Returns:
        Formatted string representation
    """
    if value is None or value == '':
        return 'N/A'
    
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    # Handle special cases
    if math.isnan(num_value):
        return 'N/A'
    if math.isinf(num_value):
        return '∞' if num_value > 0 else '-∞'
    
    # Apply formatting options
    if as_percentage:
        formatted = f"{num_value:.{precision}f}%"
    elif abbreviate:
        formatted = _abbreviate_number(num_value, precision)
    else:
        formatted = f"{num_value:.{precision}f}"
    
    # Add sign if requested
    if include_sign and num_value > 0:
        formatted = f"+{formatted}"
    
    return formatted


def _abbreviate_number(value: float, precision: int = 2) -> str:
    """
    Abbreviate a large number with K, M, B, T suffix.
    
    Args:
        value: The value to abbreviate
        precision: Number of decimal places
        
    Returns:
        Abbreviated number string
    """
    abs_value = abs(value)
    
    if abs_value >= 1e12:  # Trillion
        return f"{value / 1e12:.{precision}f}T"
    elif abs_value >= 1e9:  # Billion
        return f"{value / 1e9:.{precision}f}B"
    elif abs_value >= 1e6:  # Million
        return f"{value / 1e6:.{precision}f}M"
    elif abs_value >= 1e3:  # Thousand
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def format_market_cap(value: Optional[float]) -> Optional[str]:
    """
    Format market cap value with appropriate suffix.
    
    This is the unified market cap formatter used throughout the application.
    
    Args:
        value: Market capitalization value
        
    Returns:
        Formatted market cap string
    """
    if value is None:
        return None
        
    if value >= 1e12:  # Trillion
        if value >= 10e12:
            return f"{value / 1e12:.1f}T"
        else:
            return f"{value / 1e12:.2f}T"
    elif value >= 1e9:  # Billion
        if value >= 100e9:
            return f"{int(value / 1e9)}B"
        elif value >= 10e9:
            return f"{value / 1e9:.1f}B"
        else:
            return f"{value / 1e9:.2f}B"
    elif value >= 1e6:  # Million
        if value >= 100e6:
            return f"{int(value / 1e6)}M"
        elif value >= 10e6:
            return f"{value / 1e6:.1f}M"
        else:
            return f"{value / 1e6:.2f}M"
    else:
        return f"{int(value):,}"


def format_market_metrics(metrics: Dict[str, Any], include_pct_signs: bool = True) -> Dict[str, str]:
    """
    Format market metrics for display.
    
    Args:
        metrics: Dictionary of metric values
        include_pct_signs: Whether to include % signs for percentage values
        
    Returns:
        Dictionary of formatted metric strings
    """
    formatted = {}
    
    # Define formatting rules for different metrics
    formatting_rules = {
        "price": {"precision": 2},
        "target_price": {"precision": 2},
        "upside": {"precision": 1, "as_percentage": include_pct_signs},
        "buy_percentage": {"precision": 0, "as_percentage": include_pct_signs},
        "beta": {"precision": 2},
        "pe_trailing": {"precision": 1},
        "pe_forward": {"precision": 1},
        "peg_ratio": {"precision": 2},
        "dividend_yield": {"precision": 2, "as_percentage": include_pct_signs},
        "short_percent": {"precision": 1, "as_percentage": include_pct_signs},
    }
    
    # Apply formatting to each metric
    for key, value in metrics.items():
        if key in formatting_rules:
            rules = formatting_rules[key]
            formatted[key] = format_number(
                value, 
                precision=rules.get("precision", 2),
                as_percentage=rules.get("as_percentage", False)
            )
        elif key == "market_cap" and value is not None:
            formatted[key] = format_market_cap(value)
        else:
            formatted[key] = str(value) if value is not None else "N/A"
    
    return formatted


def format_table(data: List[Dict[str, Any]], columns: List[str], formatters: Optional[Dict[str, Dict[str, Any]]] = None) -> List[List[str]]:
    """
    Format tabular data with column-specific formatting.
    
    Args:
        data: List of data dictionaries
        columns: List of column names to include
        formatters: Dictionary mapping column names to formatter dictionaries
        
    Returns:
        List of rows with formatted values
    """
    if not data:
        return []
    
    # Ensure formatters dictionary exists
    if formatters is None:
        formatters = {}
    
    # Create table header row
    table = [columns]
    
    # Format each row
    for item in data:
        row = []
        for col in columns:
            value = item.get(col, "")
            
            # Apply column-specific formatter if available
            if col in formatters:
                formatter = formatters[col]
                precision = formatter.get("precision", 2)
                as_percentage = formatter.get("as_percentage", False)
                include_sign = formatter.get("include_sign", False)
                abbreviate = formatter.get("abbreviate", False)
                
                formatted_value = format_number(
                    value,
                    precision=precision,
                    as_percentage=as_percentage,
                    include_sign=include_sign,
                    abbreviate=abbreviate
                )
            else:
                # Default formatting
                formatted_value = str(value) if value is not None else "N/A"
            
            row.append(formatted_value)
        
        table.append(row)
    
    return table


def generate_market_html(data: List[Dict[str, Any]], title: str, 
                        columns: List[str] = None, formatters: Dict[str, Dict[str, Any]] = None) -> str:
    """
    Generate HTML table for market data.
    
    Args:
        data: List of market data dictionaries
        title: Table title
        columns: List of columns to include (default: all)
        formatters: Dictionary mapping column names to formatter dictionaries
        
    Returns:
        HTML table string
    """
    if not data:
        return "<p>No data available</p>"
    
    # Use all columns if not specified
    if columns is None:
        columns = list(data[0].keys())
    
    # Start HTML
    html = f"<h2>{title}</h2>\n"
    html += "<table border='1' cellpadding='5' cellspacing='0'>\n"
    
    # Table header
    html += "  <tr>\n"
    for col in columns:
        html += f"    <th>{col}</th>\n"
    html += "  </tr>\n"
    
    # Table rows
    for item in data:
        html += "  <tr>\n"
        for col in columns:
            value = item.get(col, "")
            
            # Apply column-specific formatter if available
            if formatters and col in formatters:
                formatter = formatters[col]
                precision = formatter.get("precision", 2)
                as_percentage = formatter.get("as_percentage", False)
                include_sign = formatter.get("include_sign", False)
                abbreviate = formatter.get("abbreviate", False)
                
                formatted_value = format_number(
                    value,
                    precision=precision,
                    as_percentage=as_percentage,
                    include_sign=include_sign,
                    abbreviate=abbreviate
                )
            else:
                # Default formatting
                formatted_value = str(value) if value is not None else "N/A"
            
            html += f"    <td>{formatted_value}</td>\n"
        html += "  </tr>\n"
    
    html += "</table>\n"
    return html


def format_for_csv(data: List[Dict[str, Any]], columns: List[str] = None) -> List[List[str]]:
    """
    Format data for CSV export.
    
    Args:
        data: List of data dictionaries
        columns: List of columns to include (default: all)
        
    Returns:
        List of rows with formatted values
    """
    if not data:
        return []
    
    # Use all columns if not specified
    if columns is None:
        columns = list(data[0].keys())
    
    # Create CSV header row
    csv_data = [columns]
    
    # Format each row
    for item in data:
        row = []
        for col in columns:
            value = item.get(col, "")
            row.append(str(value) if value is not None else "")
        
        csv_data.append(row)
    
    return csv_data
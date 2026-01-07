"""
UNIFIED DISPLAY MANAGER

This module provides consistent formatting for console, CSV, and HTML outputs
across all trade options. It uses the centralized TradeConfig to ensure
perfect alignment between all output formats.

Author: EtoroTrade System
Version: 1.0.0 (Production)
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path

from .trade_config import TradeConfig


class DisplayManager:
    """
    Unified display manager for consistent formatting across all output types.
    
    This class ensures that console, CSV, and HTML outputs are perfectly aligned
    by using the same configuration source (TradeConfig) for column selection,
    formatting rules, and sorting.
    """

    def __init__(self, config: TradeConfig = None):
        """
        Initialize the display manager.
        
        Args:
            config: TradeConfig instance. If None, uses default.
        """
        self.config = config or TradeConfig()
        self.color_reset = "\033[0m"

    def prepare_dataframe(self, df: pd.DataFrame, option: str, sub_option: str = None, 
                         output_type: str = "console") -> pd.DataFrame:
        """
        Prepare DataFrame with correct columns and formatting for the specified option.
        
        Args:
            df: Input DataFrame
            option: Trading option (p, m, e, t, i)
            sub_option: Sub-option for trade analysis (b, s, h)
            output_type: Output type (console, csv, html)
            
        Returns:
            Formatted DataFrame with correct columns
        """
        if df.empty:
            return df.copy()

        # Get columns for this option and output type
        columns = self.config.get_display_columns(option, sub_option, output_type)
        
        # Filter DataFrame to only include available columns
        available_columns = [col for col in columns if col in df.columns]
        
        if not available_columns:
            # Fallback to original columns if none of the configured ones exist
            return df.copy()
        
        # Select and reorder columns
        result_df = df[available_columns].copy()
        
        # Apply formatting based on output type
        if output_type == "console":
            result_df = self._format_for_console(result_df)
        elif output_type == "csv":
            result_df = self._format_for_csv(result_df)
        elif output_type == "html":
            result_df = self._format_for_html(result_df)
        
        # Apply sorting
        sort_config = self.config.get_sort_config(option, sub_option)
        if sort_config.get("sort_by") and sort_config["sort_by"] in result_df.columns:
            ascending = sort_config.get("sort_order", "desc") == "asc"
            result_df = result_df.sort_values(
                by=sort_config["sort_by"], 
                ascending=ascending
            ).reset_index(drop=True)
        
        # Limit rows if specified
        max_rows = sort_config.get("max_rows")
        if max_rows and isinstance(max_rows, int) and len(result_df) > max_rows:
            result_df = result_df.head(max_rows)
        
        # Add row numbers for console and HTML
        if output_type in ["console", "html"] and "#" in result_df.columns:
            result_df["#"] = range(1, len(result_df) + 1)
        
        return result_df

    def format_console(self, df: pd.DataFrame, option: str, sub_option: str = None) -> pd.DataFrame:
        """Format DataFrame for console display with colors."""
        return self.prepare_dataframe(df, option, sub_option, "console")

    def format_csv(self, df: pd.DataFrame, option: str, sub_option: str = None) -> pd.DataFrame:
        """Format DataFrame for CSV export."""
        return self.prepare_dataframe(df, option, sub_option, "csv")

    def format_html(self, df: pd.DataFrame, option: str, sub_option: str = None) -> pd.DataFrame:
        """Format DataFrame for HTML export."""
        return self.prepare_dataframe(df, option, sub_option, "html")

    def _format_for_console(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply console-specific formatting including colors."""
        result_df = df.copy()
        
        for column in result_df.columns:
            format_rule = self.config.get_format_rule(column)
            result_df[column] = result_df[column].apply(
                lambda x: self._format_value(x, format_rule, "console")
            )
        
        return result_df

    def _format_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply CSV-specific formatting (clean, no colors)."""
        result_df = df.copy()
        
        for column in result_df.columns:
            format_rule = self.config.get_format_rule(column)
            result_df[column] = result_df[column].apply(
                lambda x: self._format_value(x, format_rule, "csv")
            )
        
        return result_df

    def _format_for_html(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply HTML-specific formatting."""
        result_df = df.copy()
        
        for column in result_df.columns:
            format_rule = self.config.get_format_rule(column)
            result_df[column] = result_df[column].apply(
                lambda x: self._format_value(x, format_rule, "html")
            )
        
        return result_df

    def _format_value(self, value: Any, format_rule: Dict[str, Any], output_type: str) -> str:
        """
        Format a single value according to the format rule and output type.
        
        Args:
            value: Value to format
            format_rule: Formatting configuration
            output_type: Output type (console, csv, html)
            
        Returns:
            Formatted string
        """
        if pd.isna(value) or value is None:
            return "--"
        
        format_type = format_rule.get("type", "text")
        
        try:
            if format_type == "currency":
                return self._format_currency(value, format_rule, output_type)
            elif format_type == "percentage":
                return self._format_percentage(value, format_rule, output_type)
            elif format_type == "market_cap":
                return self._format_market_cap(value, format_rule, output_type)
            elif format_type == "decimal":
                return self._format_decimal(value, format_rule, output_type)
            elif format_type == "action":
                return self._format_action(value, format_rule, output_type)
            else:
                return str(value)
        except (ValueError, TypeError):
            return str(value)

    def _format_currency(self, value: Any, format_rule: Dict[str, Any], output_type: str) -> str:
        """Format currency values."""
        try:
            num_value = float(value)
            decimals = format_rule.get("decimals", 2)
            symbol = format_rule.get("symbol", "$")
            
            # Use fewer decimals for high values
            threshold = format_rule.get("threshold_high_decimals", 100)
            if num_value > threshold:
                decimals = min(decimals, 2)
            
            formatted = f"{symbol}{num_value:,.{decimals}f}"
            return formatted
        except (ValueError, TypeError):
            return str(value)

    def _format_percentage(self, value: Any, format_rule: Dict[str, Any], output_type: str) -> str:
        """Format percentage values."""
        try:
            num_value = float(value)
            decimals = format_rule.get("decimals", 1)
            suffix = format_rule.get("suffix", "%")
            
            formatted = f"{num_value:.{decimals}f}{suffix}"
            
            # Add color for console output
            if output_type == "console":
                if num_value > 0 and format_rule.get("color_positive") == "green":
                    formatted = f"\033[92m{formatted}{self.color_reset}"
                elif num_value < 0 and format_rule.get("color_negative") == "red":
                    formatted = f"\033[91m{formatted}{self.color_reset}"
            
            return formatted
        except (ValueError, TypeError):
            return str(value)

    def _format_market_cap(self, value: Any, format_rule: Dict[str, Any], output_type: str) -> str:
        """Format market cap values."""
        try:
            num_value = float(value)
            units = format_rule.get("units", ["M", "B", "T"])
            decimals = format_rule.get("decimals", 1)
            
            if num_value >= 1_000_000_000_000:  # Trillion
                formatted = f"{num_value / 1_000_000_000_000:.{decimals}f}{units[2]}"
            elif num_value >= 1_000_000_000:  # Billion
                formatted = f"{num_value / 1_000_000_000:.{decimals}f}{units[1]}"
            elif num_value >= 1_000_000:  # Million
                formatted = f"{num_value / 1_000_000:.{decimals}f}{units[0]}"
            else:
                formatted = f"{num_value:,.0f}"
            
            return formatted
        except (ValueError, TypeError):
            return str(value)

    def _format_decimal(self, value: Any, format_rule: Dict[str, Any], output_type: str) -> str:
        """Format decimal values."""
        try:
            num_value = float(value)
            decimals = format_rule.get("decimals", 2)
            return f"{num_value:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    def _format_action(self, value: Any, format_rule: Dict[str, Any], output_type: str) -> str:
        """Format action values with colors."""
        value_str = str(value).upper()
        colors = format_rule.get("colors", {})
        
        if value_str in colors:
            color_config = colors[value_str]
            name = color_config.get("name", value_str)
            
            if output_type == "console":
                color_code = color_config.get("console", "")
                if color_code:
                    return f"{color_code}{name}{self.color_reset}"
                return name
            elif output_type == "html":
                # For HTML, we'll just return the value and let HTML generator handle colors
                return name
            else:  # CSV
                return name
        
        return value_str

    def save_csv(self, df: pd.DataFrame, file_path: str, option: str, sub_option: str = None) -> None:
        """
        Save DataFrame as CSV with proper formatting.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            option: Trading option
            sub_option: Sub-option if applicable
        """
        formatted_df = self.format_csv(df, option, sub_option)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save CSV
        formatted_df.to_csv(file_path, index=False)

    def save_html(self, df: pd.DataFrame, file_path: str, option: str, sub_option: str = None,
                  title: str = None) -> None:
        """
        Save DataFrame as HTML with proper formatting and colors.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            option: Trading option
            sub_option: Sub-option if applicable
            title: Page title
        """
        formatted_df = self.format_html(df, option, sub_option)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Generate HTML with action colors
        html_content = self._generate_html_table(formatted_df, title or "Trading Analysis")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_html_table(self, df: pd.DataFrame, title: str) -> str:
        """Generate HTML table with proper styling and colors."""
        if df.empty:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .empty {{ text-align: center; color: #666; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <div class="empty">No data available</div>
            </body>
            </html>
            """
        
        # Create HTML table with styling
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            "<style>",
            """
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; font-weight: bold; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .action-buy { color: #28a745; font-weight: bold; }
            .action-sell { color: #dc3545; font-weight: bold; }
            .action-hold { color: #6c757d; font-weight: bold; }
            .action-inconclusive { color: #ffc107; font-weight: bold; }
            .positive { color: #28a745; }
            .negative { color: #dc3545; }
            """,
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            "<table>",
        ]
        
        # Add header
        html_parts.append("<thead><tr>")
        for col in df.columns:
            html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr></thead>")
        
        # Add body
        html_parts.append("<tbody>")
        for _, row in df.iterrows():
            html_parts.append("<tr>")
            for col in df.columns:
                value = row[col]
                css_class = self._get_html_css_class(col, value)
                html_parts.append(f'<td class="{css_class}">{value}</td>')
            html_parts.append("</tr>")
        html_parts.append("</tbody>")
        
        html_parts.extend([
            "</table>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)

    def _get_html_css_class(self, column: str, value: Any) -> str:
        """Get CSS class for HTML cell based on column and value."""
        format_rule = self.config.get_format_rule(column)
        
        if format_rule.get("type") == "action":
            value_str = str(value).upper()
            if "BUY" in value_str:
                return "action-buy"
            elif "SELL" in value_str:
                return "action-sell"
            elif "HOLD" in value_str:
                return "action-hold"
            elif "INCONCLUSIVE" in value_str:
                return "action-inconclusive"
        
        # Check for positive/negative values
        try:
            num_value = float(value.replace("%", "").replace("$", "").replace(",", ""))
            if num_value > 0:
                return "positive"
            elif num_value < 0:
                return "negative"
        except (ValueError, AttributeError):
            pass
        
        return ""

    def get_option_title(self, option: str, sub_option: str = None) -> str:
        """Get human-readable title for option combination."""
        titles = {
            "p": "Portfolio Analysis",
            "m": "Market Analysis", 
            "e": "eToro Analysis",
            "t": "Trade Opportunities",
            "i": "Manual Input Analysis"
        }
        
        base_title = titles.get(option, f"Analysis ({option})")
        
        if option == "t" and sub_option:
            sub_titles = {
                "b": "Buy Opportunities",
                "s": "Sell Opportunities", 
                "h": "Hold Opportunities"
            }
            sub_title = sub_titles.get(sub_option, f"Trade {sub_option}")
            return f"{sub_title}"
        
        return base_title


# Export for easy imports
__all__ = ["DisplayManager"]
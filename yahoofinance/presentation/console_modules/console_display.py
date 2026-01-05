"""
Console display interface for finance data.

This module provides a simple console-based display interface for financial data.
"""

from typing import Any, Dict, List, Union

import pandas as pd
from tabulate import tabulate

from yahoofinance.presentation.formatter import Color, DisplayConfig, DisplayFormatter


class ConsoleDisplay:
    """
    Console-based display for finance data.

    This class provides a console interface for displaying and interacting
    with financial data, using both synchronous and asynchronous providers.
    It implements the core display features needed for dependency injection.
    """

    def __init__(self, compact_mode: bool = False, show_colors: bool = True, **kwargs):
        """
        Initialize console display with configuration options.

        Args:
            compact_mode: Use more compact display format
            show_colors: Whether to use ANSI colors in output
            **kwargs: Additional configuration options
        """
        self.compact_mode = compact_mode
        self.show_colors = show_colors
        self.config = DisplayConfig(compact_mode=compact_mode, show_colors=show_colors)

        # Apply additional config options from kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Create formatter
        self.formatter = DisplayFormatter(compact_mode=compact_mode)

    def format_table(self, data: List[Dict[str, Any]], title: str = None) -> str:
        """
        Format tabular data for console display.

        Args:
            data: List of data dictionaries
            title: Optional title for the table

        Returns:
            Formatted table as string
        """
        if not data:
            return "No data available."

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Apply column ordering if specified in config
        if self.config.reorder_columns:
            # Only include columns that exist in the DataFrame
            cols = [col for col in self.config.reorder_columns if col in df.columns]
            # Add any remaining columns not in the reorder list
            remaining = [col for col in df.columns if col not in self.config.reorder_columns]
            final_cols = cols + remaining
            df = df[final_cols]

        # Determine numeric columns for alignment
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Generate simple table using tabulate
        colalign = ["left" if col not in numeric_cols else "right" for col in df.columns]
        table = tabulate(df, headers="keys", tablefmt="simple", showindex=False, colalign=colalign)

        # Add title if provided
        if title:
            return f"{title}\n\n{table}"
        return table

    def display(self, data: Union[List[Dict[str, Any]], Dict[str, Any]], title: str = None) -> None:
        """
        Display data in the console.

        Args:
            data: Data to display (list of dictionaries or single dictionary)
            title: Optional title for the display
        """
        # Handle different data types
        if isinstance(data, dict):
            # Single data item
            data_list = [data]
        else:
            # List of data items
            data_list = data

        # Format as table and print
        table = self.format_table(data_list, title)
        print(table)

    def color_text(self, text: str, color_name: str) -> str:
        """
        Apply color to text if colors are enabled.

        Args:
            text: Text to color
            color_name: Name of color from Color enum

        Returns:
            Colored text string
        """
        if not self.show_colors:
            return text

        try:
            color = getattr(Color, color_name.upper())
            return f"{color.value}{text}{Color.RESET.value}"
        except (AttributeError, KeyError):
            # Return uncolored text if color not found
            return text

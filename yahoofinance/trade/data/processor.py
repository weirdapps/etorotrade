"""
Data processor module for handling DataFrame operations and data transformations.

This module contains pure functions for data processing extracted from trade.py.
"""

import pandas as pd
from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError
from yahoofinance.utils.error_handling import enrich_error_context
from yahoofinance.core.config import TRADING_CRITERIA

logger = get_logger(__name__)

# Column display constants
BUY_PERCENTAGE = "B %"
DIVIDEND_YIELD_DISPLAY = "%"


class DataProcessor:
    """Data processing operations for trade functionality."""
    
    @staticmethod
    def create_empty_ticker_dataframe():
        """Create an empty ticker dataframe with a placeholder row.
        
        Returns:
            pd.DataFrame: Empty dataframe with placeholder data
        """
        return pd.DataFrame(
            [
                {
                    "ticker": "NO_DATA",
                    "company": "No Data",
                    "price": None,
                    "target_price": None,
                    "market_cap": None,
                    "buy_percentage": None,
                    "total_ratings": 0,
                    "analyst_count": 0,
                    "pe_trailing": None,
                    "pe_forward": None,
                    "peg_ratio": None,
                    "beta": None,
                    "short_percent": None,
                    "dividend_yield": None,
                    "A": "",
                }
            ]
        )
    
    @staticmethod
    def calculate_exret(df):
        """Calculate EXRET (Expected Return) if not already present.
        
        Args:
            df: Dataframe with upside and buy_percentage columns
            
        Returns:
            pd.DataFrame: Dataframe with EXRET column added
        """
        # Always recalculate EXRET even if it exists to ensure consistency with upside values
        if (
            "upside" in df.columns
            and "buy_percentage" in df.columns
            and pd.api.types.is_numeric_dtype(df["upside"])
            and pd.api.types.is_numeric_dtype(df["buy_percentage"])
        ):
            # Round upside to 1 decimal place to match display formatting before calculating EXRET
            rounded_upside = df["upside"].round(1)
            # Calculate EXRET: upside% * buy% / 100 = percentage
            df["EXRET"] = (rounded_upside * df["buy_percentage"]) / 100
        else:
            df["EXRET"] = None
        return df
    
    @staticmethod
    def safe_calc_exret(row):
        """Helper function to safely calculate EXRET for a row."""
        try:
            if pd.isna(row["upside"]) or pd.isna(row["buy_percentage"]):
                return None
            
            # Convert to float if needed
            upside = float(row["upside"]) if isinstance(row["upside"], str) else row["upside"]
            buy_pct = (
                float(row["buy_percentage"])
                if isinstance(row["buy_percentage"], str)
                else row["buy_percentage"]
            )
            
            # Round upside to 1 decimal place to match display formatting before calculating EXRET
            rounded_upside = round(upside, 1)
            # Calculate EXRET: upside% * buy% / 100 = percentage
            return (rounded_upside * buy_pct) / 100
        except (TypeError, ValueError):
            return None
    
    @staticmethod
    def get_column_mapping():
        """Get the column mapping for display.
        
        Returns:
            dict: Mapping of internal column names to display names
        """
        return {
            "ticker": "TICKER",  # Keep for potential compatibility
            "symbol": "TICKER",  # Add mapping for the actual key returned by providers
            "company": "COMPANY",
            "cap": "CAP",
            "market_cap": "CAP",  # Add market_cap field mapping
            "price": "PRICE",
            "target_price": "TARGET",
            "upside": "UPSIDE",
            "analyst_count": "# T",
            "buy_percentage": BUY_PERCENTAGE,
            "total_ratings": "# A",  # Already present, just confirming
            "A": "A",  # A column shows ratings type (A/E for All-time/Earnings-based)
            "EXRET": "EXRET",
            "beta": "BETA",
            "pe_trailing": "PET",
            "pe_forward": "PEF",
            "peg_ratio": "PEG",
            "dividend_yield": DIVIDEND_YIELD_DISPLAY,
            "short_float_pct": "SI",
            "short_percent": "SI",  # V2 naming
            "last_earnings": "EARNINGS",
            "earnings_date": "EARNINGS",  # Fallback mapping for earnings_date
            "position_size": "SIZE",  # Position size mapping
            "action": "ACT",  # Update ACTION to ACT
            "ACTION": "ACT",  # Update ACTION to ACT (for backward compatibility)
        }
    
    @staticmethod
    def get_columns_to_select():
        """Get columns to select for display.
        
        Returns:
            list: List of columns to select for display
        """
        return [
            "ticker",
            "symbol",  # Add symbol for v2 compatibility
            "company",
            "market_cap",
            "cap",  # Add cap for v1 compatibility (after market_cap)
            "position_size",  # Add position size column
            "price",
            "target_price",
            "upside",
            "analyst_count",
            "buy_percentage",
            "total_ratings",
            "A",
            "EXRET",
            "beta",
            "pe_trailing",
            "pe_forward",
            "peg_ratio",
            "dividend_yield",
            "short_float_pct",
            "short_percent",
            "last_earnings",
            "earnings_date",  # Fallback column for earnings_date
            "earnings_growth",  # Add earnings_growth for EG column
            "action",  # Add action column
            "ACTION",  # Keep ACTION for compatibility
            "twelve_month_performance",  # Add twelve_month_performance for PP column
        ]
    
    @staticmethod
    def create_empty_display_dataframe():
        """Create an empty display dataframe with proper columns.
        
        Returns:
            pd.DataFrame: Empty dataframe with display columns
        """
        column_mapping = DataProcessor.get_column_mapping()
        display_columns = [
            "TICKER",
            "COMPANY",
            "CAP",
            "SIZE",
            "PRICE",
            "TARGET",
            "UPSIDE",
            "# T",
            BUY_PERCENTAGE,
            "# A",
            "A",
            "EXRET",
            "BETA",
            "PET",
            "PEF",
            "PEG",
            DIVIDEND_YIELD_DISPLAY,
            "SI",
            "EARNINGS",
            "ACT",
        ]
        
        # Create empty dataframe with all columns
        empty_df = pd.DataFrame(columns=display_columns)
        
        return empty_df
    
    @staticmethod
    def format_company_names(working_df):
        """Format company names to be cleaner and more readable.
        
        Args:
            working_df: DataFrame with company names
            
        Returns:
            pd.DataFrame: DataFrame with formatted company names
        """
        if "company" not in working_df.columns:
            return working_df
        
        # Define suffixes to remove
        suffixes_to_remove = [
            " Inc.",
            " Inc",
            " Corp.",
            " Corp",
            " Corporation",
            " Ltd.",
            " Ltd",
            " Limited",
            " LLC",
            " L.L.C.",
            " PLC",
            " plc",
            " p.l.c.",
            " N.V.",
            " NV",
            " S.A.",
            " SA",
            " AG",
            " SE",
            " Co.",
            " Co",
            " Company",
            " Holdings",
            " Holding",
            ", Inc.",
            ", Inc",
            ", Corp.",
            ", Corp",
            ", Corporation",
            ", Ltd.",
            ", Ltd",
            ", Limited",
            ", LLC",
            ", L.L.C.",
            ", PLC",
            ", plc",
            ", p.l.c.",
            ", N.V.",
            ", NV",
            ", S.A.",
            ", SA",
            ", AG",
            ", SE",
            ", Co.",
            ", Co",
            ", Company",
            ", Holdings",
            ", Holding",
        ]
        
        # Create a copy to avoid modifying the original
        formatted_names = working_df["company"].copy()
        
        # Remove suffixes
        for suffix in suffixes_to_remove:
            formatted_names = formatted_names.str.replace(suffix, "", regex=False)
        
        # Clean up any extra spaces
        formatted_names = formatted_names.str.strip()
        
        # Handle special cases
        # Example: "Alphabet Inc. (Class A)" -> "Alphabet (Class A)"
        formatted_names = formatted_names.str.replace(
            r"\s+\(Class\s+([A-Z])\)$", r" (Class \1)", regex=True
        )
        
        # Truncate very long names at word boundary
        max_length = 30
        
        def truncate_name(name):
            if pd.isna(name) or len(name) <= max_length:
                return name
            
            # Try to truncate at word boundary
            truncated = name[:max_length]
            last_space = truncated.rfind(" ")
            
            if last_space > 15:  # If we have a reasonable word boundary
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."
        
        formatted_names = formatted_names.apply(truncate_name)
        
        # Update the dataframe
        working_df["company"] = formatted_names
        
        return working_df
    
    @staticmethod
    def convert_percentage_columns(df):
        """Convert percentage string columns to numeric values.
        
        Args:
            df: DataFrame with percentage columns
            
        Returns:
            pd.DataFrame: DataFrame with numeric percentage columns
        """
        # Define percentage columns that might need conversion
        percentage_columns = [
            "upside",
            "buy_percentage",
            "dividend_yield",
            "short_percent",
            "short_float_pct",
        ]
        
        for col in percentage_columns:
            if col in df.columns:
                # Check if column contains string values with %
                if df[col].dtype == "object":
                    # Remove % sign and convert to float
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .str.strip()
                        .replace("", None)
                    )
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    @staticmethod
    def clean_si_value(value):
        """Clean short interest value for proper display.
        
        Args:
            value: The SI value to clean
            
        Returns:
            str: Cleaned SI value or "--"
        """
        if pd.isna(value) or value is None:
            return "--"
        
        # If it's already a string, check if it's a display value
        if isinstance(value, str):
            # If it already looks like a formatted value, return it
            if value == "--" or "%" in value:
                return value
            # Try to convert to float
            try:
                value = float(value)
            except ValueError:
                return "--"
        
        # Now we have a numeric value
        try:
            float_val = float(value)
            # If it's already a percentage (e.g., 1.5 for 1.5%)
            if 0 <= float_val <= 100:
                return f"{float_val:.1f}"
            # If it's a decimal (e.g., 0.015 for 1.5%)
            elif 0 <= float_val < 1:
                return f"{float_val * 100:.1f}"
            else:
                return "--"
        except (ValueError, TypeError):
            return "--"
    
    @staticmethod
    def add_ranking_column(df):
        """Add ranking column based on EXRET values.
        
        Args:
            df: DataFrame with EXRET column
            
        Returns:
            pd.DataFrame: DataFrame with ranking column added
        """
        if "EXRET" in df.columns and len(df) > 0:
            # Create a copy to avoid modifying during iteration
            df = df.copy()
            # Rank by EXRET descending, handling NaN values
            df["_rank"] = df["EXRET"].rank(method="min", ascending=False, na_option="bottom")
            # Convert to integer where possible, keep NaN as NaN
            df["_rank"] = df["_rank"].apply(lambda x: int(x) if pd.notna(x) else None)
        return df
    
    @staticmethod
    def format_numeric_columns(display_df, columns, format_str):
        """Apply consistent numeric formatting to specified columns.
        
        Args:
            display_df: DataFrame to format
            columns: List of column names to format
            format_str: Format string to apply (e.g., "{:.1f}")
            
        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        for col in columns:
            if col in display_df.columns:
                # Apply formatting only to non-null numeric values
                display_df[col] = display_df[col].apply(
                    lambda x: format_str.format(x) if pd.notna(x) and x != "" else x
                )
        return display_df
    
    @staticmethod
    def convert_to_numeric(row_dict):
        """Convert numeric string values to actual numeric types.
        
        Args:
            row_dict: Dictionary representing a row of data
            
        Returns:
            dict: Row dictionary with numeric conversions applied
        """
        # Columns that should be numeric
        numeric_columns = [
            "PRICE",
            "TARGET",
            "UPSIDE",
            "# T",
            BUY_PERCENTAGE,
            "# A",
            "EXRET",
            "BETA",
            "PET",
            "PEF",
            "PEG",
            DIVIDEND_YIELD_DISPLAY,
            "SI",
        ]
        
        for col in numeric_columns:
            if col in row_dict and row_dict[col] not in [None, "", "--", "N/A"]:
                try:
                    # Remove any percentage signs
                    value = str(row_dict[col]).replace("%", "").strip()
                    if value and value != "--":
                        row_dict[col] = float(value)
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass
        
        return row_dict
"""
Output Manager Module

This module handles all file output, report generation, data export,
and display formatting for the trade analysis application.
"""

import logging
import os
import sys
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from tabulate import tabulate

# Import presentation and formatting utilities
from yahoofinance.presentation.html import HTMLGenerator
from yahoofinance.presentation.formatter import DisplayFormatter
from yahoofinance.utils.data.format_utils import format_position_size
from yahoofinance.core.config import (
    STANDARD_DISPLAY_COLUMNS, FILE_PATHS, PATHS
)

# Color constants for terminal output
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"

# Get logger for this module
logger = logging.getLogger(__name__)


def ensure_output_directory(output_dir: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")


def _setup_output_files(report_source: str) -> Tuple[str, str, str]:
    """
    Set up output file paths based on source type.
    
    Args:
        report_source: Source type (market/portfolio/manual)
        
    Returns:
        tuple: (buy_file, sell_file, hold_file) paths
    """
    try:
        output_dir = PATHS["OUTPUT_DIR"]
        ensure_output_directory(output_dir)
        
        # Map source to appropriate files
        if report_source == "market":
            buy_file = os.path.join(output_dir, "market_buy.csv")
            sell_file = os.path.join(output_dir, "market_sell.csv")
            hold_file = os.path.join(output_dir, "market_hold.csv")
        elif report_source == "portfolio":
            buy_file = os.path.join(output_dir, "portfolio_buy.csv")
            sell_file = os.path.join(output_dir, "portfolio_sell.csv") 
            hold_file = os.path.join(output_dir, "portfolio_hold.csv")
        else:  # manual or other
            buy_file = FILE_PATHS["BUY_OUTPUT"]
            sell_file = FILE_PATHS["SELL_OUTPUT"]
            hold_file = FILE_PATHS["HOLD_OUTPUT"]
            
        return buy_file, sell_file, hold_file
        
    except Exception as e:
        logger.error(f"Error setting up output files: {str(e)}")
        # Return default paths as fallback
        return (
            FILE_PATHS["BUY_OUTPUT"],
            FILE_PATHS["SELL_OUTPUT"],
            FILE_PATHS["HOLD_OUTPUT"]
        )


def _prepare_csv_dataframe(display_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for CSV export with proper column ordering.
    
    Args:
        display_df: Display dataframe to prepare
        
    Returns:
        pd.DataFrame: Prepared CSV dataframe
    """
    try:
        csv_df = display_df.copy()
        
        # Add ranking column if not present
        if "#" not in csv_df.columns:
            csv_df = _add_ranking_column(csv_df)
        
        # Ensure standard column order
        available_columns = [col for col in STANDARD_DISPLAY_COLUMNS if col in csv_df.columns]
        
        # Add any additional columns not in standard list
        additional_columns = [col for col in csv_df.columns if col not in available_columns]
        final_columns = available_columns + additional_columns
        
        # Reorder columns
        csv_df = csv_df[final_columns]
        
        # Clean short interest values for CSV
        if "SI" in csv_df.columns:
            csv_df["SI"] = csv_df["SI"].apply(_clean_si_value)
        
        logger.debug(f"Prepared CSV dataframe with {len(csv_df)} rows and {len(csv_df.columns)} columns")
        return csv_df
        
    except Exception as e:
        logger.error(f"Error preparing CSV dataframe: {str(e)}")
        return display_df


def _clean_si_value(value: Any) -> str:
    """
    Clean short interest values for CSV export.
    
    Args:
        value: Short interest value to clean
        
    Returns:
        str: Cleaned short interest value
    """
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        
        # Convert to string and remove % if present
        str_value = str(value).replace("%", "").strip()
        
        # Try to convert to float and format
        numeric_value = float(str_value)
        return f"{numeric_value:.1f}%"
        
    except (ValueError, TypeError):
        return "--"


def _add_ranking_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranking (#) column to dataframe.
    
    Args:
        df: Dataframe to add ranking to
        
    Returns:
        pd.DataFrame: Dataframe with ranking column
    """
    try:
        result_df = df.copy()
        
        # Add ranking column at the beginning
        result_df.insert(0, "#", range(1, len(result_df) + 1))
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error adding ranking column: {str(e)}")
        return df


def get_column_alignments(display_df: pd.DataFrame) -> List[str]:
    """
    Get column alignments for tabulate display.
    
    Args:
        display_df: Display dataframe
        
    Returns:
        list: Column alignments for tabulate
    """
    try:
        alignments = []
        
        for col in display_df.columns:
            if col in ["#", "TICKER", "NAME", "ACT"]:
                alignments.append("left")
            elif col in ["PRICE", "TARGET", "UPSIDE", "BUY%", "EXRET", "SIZE"]:
                alignments.append("right")
            else:
                alignments.append("center")
                
        return alignments
        
    except Exception as e:
        logger.error(f"Error getting column alignments: {str(e)}")
        return ["left"] * len(display_df.columns)


def _get_color_by_title(title: str) -> str:
    """
    Get appropriate color code based on report title.
    
    Args:
        title: Report title
        
    Returns:
        str: ANSI color code
    """
    title_lower = title.lower()
    
    if "buy" in title_lower or "opportunity" in title_lower:
        return COLOR_GREEN
    elif "sell" in title_lower or "candidate" in title_lower:
        return COLOR_RED
    elif "hold" in title_lower:
        return COLOR_YELLOW
    else:
        return COLOR_RESET


def _apply_color_to_dataframe(display_df: pd.DataFrame, color_code: str) -> pd.DataFrame:
    """
    Apply ANSI color codes to dataframe values for terminal display.
    
    Args:
        display_df: Dataframe to colorize
        color_code: ANSI color code to apply
        
    Returns:
        pd.DataFrame: Colorized dataframe
    """
    try:
        if not color_code or color_code == COLOR_RESET:
            return display_df
            
        colored_df = display_df.copy()
        
        # Apply color to all string columns
        for col in colored_df.columns:
            if colored_df[col].dtype == 'object':
                colored_df[col] = colored_df[col].astype(str).apply(
                    lambda x: f"{color_code}{x}{COLOR_RESET}" if x != "" else x
                )
                
        return colored_df
        
    except Exception as e:
        logger.error(f"Error applying color to dataframe: {str(e)}")
        return display_df


def display_and_save_results(display_df: pd.DataFrame, title: str, output_file: str) -> None:
    """
    Display results in console and save to CSV and HTML files.
    
    Args:
        display_df: Dataframe to display and save
        title: Title for the report
        output_file: Output file path for CSV
    """
    try:
        if display_df.empty:
            create_empty_results_file(output_file)
            _display_empty_result(title)
            return
            
        # Prepare data for display
        colored_df = display_df.copy()
        
        # Get color for title-based coloring
        title_color = _get_color_by_title(title)
        
        # Display in console with tabulate
        print(f"\n{title_color}{title}{COLOR_RESET}")
        print("=" * len(title))
        
        # Get column alignments
        alignments = get_column_alignments(display_df)
        
        # Display table
        table_output = tabulate(
            colored_df.values,
            headers=colored_df.columns,
            tablefmt="grid",
            colalign=alignments
        )
        print(table_output)
        print(f"\nTotal: {len(display_df)} results")
        
        # Save to CSV
        csv_df = _prepare_csv_dataframe(display_df)
        csv_df.to_csv(output_file, index=False)
        print(f"📁 Results saved to: {output_file}")
        
        # Generate HTML file
        html_file = output_file.replace('.csv', '.html')
        html_generator = HTMLGenerator()
        html_generator.generate_results_html(csv_df, title, html_file)
        print(f"🌐 HTML report saved to: {html_file}")
        
    except Exception as e:
        logger.error(f"Error displaying and saving results: {str(e)}")
        print(f"❌ Error saving results: {str(e)}")


def create_empty_results_file(output_file: str) -> None:
    """
    Create empty CSV and HTML files when no results found.
    
    Args:
        output_file: Output file path
    """
    try:
        # Create empty DataFrame with standard columns
        empty_df = pd.DataFrame(columns=STANDARD_DISPLAY_COLUMNS)
        
        # Save empty CSV
        empty_df.to_csv(output_file, index=False)
        
        # Generate empty HTML
        html_file = output_file.replace('.csv', '.html')
        html_generator = HTMLGenerator()
        html_generator.generate_results_html(
            empty_df, 
            "No Results Found", 
            html_file
        )
        
        logger.debug(f"Created empty results files: {output_file}, {html_file}")
        
    except Exception as e:
        logger.error(f"Error creating empty results file: {str(e)}")


def _display_empty_result(report_title: str) -> None:
    """
    Display message when no data is available.
    
    Args:
        report_title: Title of the report
    """
    print(f"\n{report_title}")
    print("=" * len(report_title))
    print("📋 No results found matching the criteria.")
    print()


def _sort_display_dataframe(display_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort display dataframe by EXRET column in descending order.
    
    Args:
        display_df: Dataframe to sort
        
    Returns:
        pd.DataFrame: Sorted dataframe
    """
    try:
        if "EXRET" in display_df.columns and not display_df.empty:
            # Convert EXRET to numeric for sorting
            display_df["EXRET_numeric"] = pd.to_numeric(
                display_df["EXRET"], errors="coerce"
            ).fillna(0)
            
            # Sort by EXRET descending
            sorted_df = display_df.sort_values("EXRET_numeric", ascending=False)
            
            # Remove the temporary numeric column
            sorted_df = sorted_df.drop("EXRET_numeric", axis=1)
            
            # Reset index
            sorted_df = sorted_df.reset_index(drop=True)
            
            logger.debug(f"Sorted dataframe by EXRET: {len(sorted_df)} rows")
            return sorted_df
        else:
            return display_df
            
    except Exception as e:
        logger.error(f"Error sorting display dataframe: {str(e)}")
        return display_df


def format_display_dataframe(display_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format all columns in display dataframe for presentation.
    
    Args:
        display_df: Dataframe to format
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    try:
        formatted_df = display_df.copy()
        
        # Format price columns
        price_columns = ["PRICE", "TARGET"]
        for col in price_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(_format_price_value)
        
        # Format percentage columns
        percentage_columns = ["UPSIDE", "BUY%", "DIV", "SI"]
        for col in percentage_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(_format_percentage_value)
        
        # Format numeric columns
        numeric_columns = ["PET", "PEF", "PEG", "BETA", "EXRET"]
        for col in numeric_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(_format_numeric_value)
        
        # Format earnings date
        if "EARN" in formatted_df.columns:
            formatted_df["EARN"] = formatted_df["EARN"].apply(_format_date_value)
        
        # Format market cap and size columns
        if "CAP" in formatted_df.columns:
            formatter = DisplayFormatter()
            formatted_df["CAP"] = formatted_df["CAP"].apply(
                lambda x: formatter.format_market_cap(x)
            )
        
        if "SIZE" in formatted_df.columns:
            formatted_df["SIZE"] = formatted_df["SIZE"].apply(_format_size_value)
        
        logger.debug(f"Formatted display dataframe with {len(formatted_df.columns)} columns")
        return formatted_df
        
    except Exception as e:
        logger.error(f"Error formatting display dataframe: {str(e)}")
        return display_df


def _format_price_value(value: Any) -> str:
    """Format price values for display."""
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        numeric_value = float(value)
        return f"${numeric_value:.2f}"
    except (ValueError, TypeError):
        return "--"


def _format_percentage_value(value: Any) -> str:
    """Format percentage values for display."""
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        numeric_value = float(value)
        return f"{numeric_value:.1f}%"
    except (ValueError, TypeError):
        return "--"


def _format_numeric_value(value: Any) -> str:
    """Format numeric values for display."""
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        numeric_value = float(value)
        return f"{numeric_value:.1f}"
    except (ValueError, TypeError):
        return "--"


def _format_date_value(value: Any) -> str:
    """Format date values for display."""
    try:
        if pd.isna(value) or value is None or value == "":
            return "--"
        date_str = str(value).strip()
        if len(date_str) >= 10:
            return date_str[:10]  # YYYY-MM-DD format
        return date_str
    except Exception:
        return "--"


def _format_size_value(value: Any) -> str:
    """Format position size values for display."""
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        
        # Use the position size formatter from utils
        return format_position_size(value)
        
    except Exception:
        return "--"


def prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to prepare dataframes for display.
    
    Args:
        df: Raw dataframe to prepare
        
    Returns:
        pd.DataFrame: Prepared display dataframe
    """
    try:
        # Make a copy to avoid modifying original
        display_df = df.copy()
        
        # Format all columns
        display_df = format_display_dataframe(display_df)
        
        # Sort by EXRET
        display_df = _sort_display_dataframe(display_df)
        
        # Add ranking column
        display_df = _add_ranking_column(display_df)
        
        logger.debug(f"Prepared display dataframe: {len(display_df)} rows")
        return display_df
        
    except Exception as e:
        logger.error(f"Error preparing display dataframe: {str(e)}")
        return df


def export_results_to_files(results_dict: Dict[str, pd.DataFrame], 
                          report_source: str = "manual") -> Dict[str, str]:
    """
    Export analysis results to multiple output files.
    
    Args:
        results_dict: Dictionary with buy/sell/hold results
        report_source: Source type for file naming
        
    Returns:
        dict: Mapping of result type to output file path
    """
    try:
        # Set up output files
        buy_file, sell_file, hold_file = _setup_output_files(report_source)
        
        output_files = {}
        
        # Export buy opportunities
        if "buy_opportunities" in results_dict:
            buy_df = results_dict["buy_opportunities"]
            display_and_save_results(buy_df, "Buy Opportunities", buy_file)
            output_files["buy"] = buy_file
        
        # Export sell candidates
        if "sell_candidates" in results_dict:
            sell_df = results_dict["sell_candidates"]
            display_and_save_results(sell_df, "Sell Candidates", sell_file)
            output_files["sell"] = sell_file
        
        # Export hold candidates
        if "hold_candidates" in results_dict:
            hold_df = results_dict["hold_candidates"]
            display_and_save_results(hold_df, "Hold Candidates", hold_file)
            output_files["hold"] = hold_file
        
        logger.info(f"Exported {len(output_files)} result files")
        return output_files
        
    except Exception as e:
        logger.error(f"Error exporting results to files: {str(e)}")
        return {}


class OutputManager:
    """Main output management class for coordinating all output operations."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the output manager.
        
        Args:
            output_dir: Custom output directory (uses default if None)
        """
        self.logger = logging.getLogger(f"{__name__}.OutputManager")
        self.output_dir = output_dir or PATHS["OUTPUT_DIR"]
        self.ensure_output_directory()
    
    def ensure_output_directory(self) -> None:
        """Ensure the output directory exists."""
        ensure_output_directory(self.output_dir)
    
    def save_analysis_results(self, results: Dict[str, Any], 
                            report_source: str = "manual") -> Dict[str, str]:
        """
        Save complete analysis results to files.
        
        Args:
            results: Analysis results dictionary
            report_source: Source type for file naming
            
        Returns:
            dict: Mapping of result type to output file path
        """
        try:
            # Prepare results for display
            display_results = {}
            
            for key, df in results.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    display_results[key] = prepare_display_dataframe(df)
            
            # Export to files
            output_files = export_results_to_files(display_results, report_source)
            
            self.logger.info(f"✅ Analysis results saved successfully")
            return output_files
            
        except Exception as e:
            error_msg = f"Failed to save analysis results: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text summary of analysis results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            str: Summary report text
        """
        try:
            summary_lines = []
            summary_lines.append("📊 ANALYSIS SUMMARY")
            summary_lines.append("=" * 50)
            
            for key, df in results.items():
                if isinstance(df, pd.DataFrame):
                    count = len(df)
                    summary_lines.append(f"{key.replace('_', ' ').title()}: {count} items")
            
            total_analyzed = sum(len(df) for df in results.values() 
                               if isinstance(df, pd.DataFrame))
            summary_lines.append(f"Total Items Analyzed: {total_analyzed}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            return "Error generating summary report"
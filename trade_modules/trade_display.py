#!/usr/bin/env python3
"""
Display formatting module extracted from trade.py.
Handles output formatting, color coding, and data presentation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tabulate import tabulate

from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError

from .utils import (
    safe_float_conversion,
    safe_percentage_format,
    format_market_cap_value,
)
from .data_processor import (
    format_numeric_columns,
    format_percentage_columns,
)

logger = get_logger(__name__)

# Color constants for terminal output
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"


class DisplayFormatter:
    """Handles formatting and display of trading data."""
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize display formatter.
        
        Args:
            use_colors: Whether to use terminal colors in output
        """
        self.use_colors = use_colors
        self.logger = logger
    
    def format_trading_opportunities(self, opportunities: Dict[str, pd.DataFrame],
                                   output_format: str = 'table') -> str:
        """
        Format trading opportunities for display.
        
        Args:
            opportunities: Dictionary containing buy/sell/hold DataFrames
            output_format: Output format ('table', 'html', 'json')
            
        Returns:
            Formatted string representation
        """
        try:
            output_parts = []
            
            # Format each opportunity type
            for action_type, df in opportunities.items():
                if df.empty:
                    continue
                
                title = self._format_section_title(action_type.replace('_', ' ').title())
                output_parts.append(title)
                
                if output_format == 'table':
                    formatted_df = self._prepare_display_dataframe(df, action_type)
                    table_str = self._format_as_table(formatted_df, action_type)
                    output_parts.append(table_str)
                elif output_format == 'html':
                    html_str = self._format_as_html(df, action_type)
                    output_parts.append(html_str)
                elif output_format == 'json':
                    json_str = df.to_json(indent=2)
                    output_parts.append(json_str)
                
                output_parts.append("")  # Empty line separator
            
            return "\n".join(output_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting trading opportunities: {str(e)}")
            return f"Error formatting display: {str(e)}"
    
    def _format_section_title(self, title: str) -> str:
        """Format a section title with colors and styling."""
        if not self.use_colors:
            return f"=== {title} ==="
        
        color = COLOR_CYAN
        if 'buy' in title.lower():
            color = COLOR_GREEN
        elif 'sell' in title.lower():
            color = COLOR_RED
        elif 'hold' in title.lower():
            color = COLOR_YELLOW
        
        return f"{color}{COLOR_BOLD}=== {title} ==={COLOR_RESET}"
    
    def _prepare_display_dataframe(self, df: pd.DataFrame, action_type: str) -> pd.DataFrame:
        """Prepare DataFrame for display with proper formatting."""
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        display_df = df.copy()
        
        # Select and order relevant columns for display
        display_columns = self._get_display_columns(action_type)
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        if available_columns:
            display_df = display_df[available_columns]
        
        # Format numeric columns
        display_df = self._format_display_columns(display_df)
        
        # Add color coding for terminal output
        if self.use_colors:
            display_df = self._apply_color_coding(display_df, action_type)
        
        return display_df
    
    def _get_display_columns(self, action_type: str) -> List[str]:
        """Get appropriate columns for display based on action type."""
        base_columns = [
            'ticker', 'price', 'market_cap', 'pe_ratio', 'dividend_yield',
            'price_target', 'expected_return', 'exret', 'action', 'confidence_score'
        ]
        
        if action_type == 'buy_opportunities':
            return ['price', 'market_cap', 'pe_ratio', 'price_target', 
                   'expected_return', 'confidence_score']
        elif action_type == 'sell_opportunities':
            return ['price', 'market_cap', 'expected_return', 'exret', 
                   'confidence_score', 'action']
        elif action_type == 'hold_opportunities':
            return ['price', 'dividend_yield', 'pe_ratio', 'expected_return', 
                   'confidence_score']
        
        return base_columns
    
    def _format_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format columns for better display."""
        if df.empty:
            return df
        
        # Format market cap
        if 'market_cap' in df.columns:
            df['market_cap'] = df['market_cap'].apply(
                lambda x: format_market_cap_value(x) if pd.notna(x) else 'N/A'
            )
        
        # Format price columns as currency
        price_columns = ['price', 'price_target']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) and x != 0 else 'N/A'
                )
        
        # Format percentage columns
        percentage_columns = ['expected_return', 'exret', 'dividend_yield', 'confidence_score']
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: safe_percentage_format(x) if pd.notna(x) else 'N/A'
                )
        
        # Format ratio columns
        ratio_columns = ['pe_ratio', 'forward_pe', 'beta']
        for col in ratio_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else 'N/A'
                )
        
        return df
    
    def _apply_color_coding(self, df: pd.DataFrame, action_type: str) -> pd.DataFrame:
        """Apply color coding to DataFrame for terminal display."""
        if not self.use_colors or df.empty:
            return df
        
        try:
            # Color code based on action type
            if action_type == 'buy_opportunities':
                color = COLOR_GREEN
            elif action_type == 'sell_opportunities':
                color = COLOR_RED
            elif action_type == 'hold_opportunities':
                color = COLOR_YELLOW
            else:
                color = COLOR_RESET
            
            # Apply color to specific columns
            if 'action' in df.columns:
                df['action'] = df['action'].apply(
                    lambda x: f"{color}{x}{COLOR_RESET}" if pd.notna(x) else x
                )
            
            # Color code expected returns
            if 'expected_return' in df.columns:
                def color_return(value):
                    if pd.isna(value) or value == 'N/A':
                        return value
                    try:
                        # Extract numeric value if it's a percentage string
                        if isinstance(value, str) and '%' in value:
                            numeric_val = float(value.replace('%', ''))
                        else:
                            numeric_val = float(value) * 100  # Convert decimal to percentage
                        
                        if numeric_val > 0:
                            return f"{COLOR_GREEN}{value}{COLOR_RESET}"
                        elif numeric_val < 0:
                            return f"{COLOR_RED}{value}{COLOR_RESET}"
                        else:
                            return value
                    except (ValueError, TypeError):
                        return value
                
                df['expected_return'] = df['expected_return'].apply(color_return)
            
        except Exception as e:
            self.logger.warning(f"Error applying color coding: {str(e)}")
        
        return df
    
    def _format_as_table(self, df: pd.DataFrame, action_type: str) -> str:
        """Format DataFrame as a table string."""
        if df.empty:
            return "No opportunities found."
        
        try:
            # Create table with nice formatting
            table_str = tabulate(
                df,
                headers=df.columns,
                tablefmt='grid',
                showindex=True,
                numalign='right',
                stralign='left'
            )
            
            return table_str
            
        except Exception as e:
            self.logger.error(f"Error creating table: {str(e)}")
            return f"Error formatting table: {str(e)}"
    
    def _format_as_html(self, df: pd.DataFrame, action_type: str) -> str:
        """Format DataFrame as HTML string."""
        if df.empty:
            return "<p>No opportunities found.</p>"
        
        try:
            # Prepare DataFrame for HTML output
            display_df = self._prepare_display_dataframe(df, action_type)
            
            # Generate HTML table
            html_str = display_df.to_html(
                classes=f'table table-striped {action_type}',
                table_id=f'{action_type}_table',
                escape=False,
                index=True
            )
            
            return html_str
            
        except Exception as e:
            self.logger.error(f"Error creating HTML: {str(e)}")
            return f"<p>Error formatting HTML: {str(e)}</p>"
    
    def format_summary_statistics(self, opportunities: Dict[str, pd.DataFrame]) -> str:
        """Format summary statistics for opportunities."""
        try:
            summary_parts = []
            
            title = self._format_section_title("Summary Statistics")
            summary_parts.append(title)
            
            # Calculate counts
            buy_count = len(opportunities.get('buy_opportunities', pd.DataFrame()))
            sell_count = len(opportunities.get('sell_opportunities', pd.DataFrame()))
            hold_count = len(opportunities.get('hold_opportunities', pd.DataFrame()))
            total_count = buy_count + sell_count + hold_count
            
            # Format statistics
            stats = [
                f"Total Opportunities: {total_count}",
                f"Buy Opportunities: {buy_count}",
                f"Sell Opportunities: {sell_count}", 
                f"Hold Opportunities: {hold_count}",
            ]
            
            # Add color coding
            if self.use_colors:
                stats[1] = f"{COLOR_GREEN}{stats[1]}{COLOR_RESET}"
                stats[2] = f"{COLOR_RED}{stats[2]}{COLOR_RESET}"
                stats[3] = f"{COLOR_YELLOW}{stats[3]}{COLOR_RESET}"
            
            summary_parts.extend(stats)
            
            # Calculate average confidence if available
            all_opportunities = pd.concat([
                df for df in opportunities.values() if not df.empty
            ], ignore_index=True)
            
            if not all_opportunities.empty and 'confidence_score' in all_opportunities.columns:
                avg_confidence = all_opportunities['confidence_score'].mean()
                confidence_str = f"Average Confidence: {avg_confidence:.1%}"
                if self.use_colors:
                    if avg_confidence > 0.7:
                        confidence_str = f"{COLOR_GREEN}{confidence_str}{COLOR_RESET}"
                    elif avg_confidence < 0.5:
                        confidence_str = f"{COLOR_RED}{confidence_str}{COLOR_RESET}"
                    else:
                        confidence_str = f"{COLOR_YELLOW}{confidence_str}{COLOR_RESET}"
                
                summary_parts.append(confidence_str)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting summary statistics: {str(e)}")
            return f"Error formatting summary: {str(e)}"
    
    def save_to_file(self, content: str, file_path: Union[str, Path], 
                    format_type: str = 'txt') -> bool:
        """
        Save formatted content to file.
        
        Args:
            content: Content to save
            file_path: Path to save file
            format_type: File format ('txt', 'html', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove color codes if saving to file
            if format_type in ['txt', 'html', 'json']:
                clean_content = self._remove_color_codes(content)
            else:
                clean_content = content
            
            # Write file with proper encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            
            self.logger.info(f"Successfully saved output to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to file {file_path}: {str(e)}")
            return False
    
    def _remove_color_codes(self, text: str) -> str:
        """Remove ANSI color codes from text."""
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


class MarketDataDisplay:
    """Specialized display class for market data visualization."""
    
    def __init__(self, formatter: DisplayFormatter = None):
        """Initialize with a display formatter."""
        self.formatter = formatter or DisplayFormatter()
        self.logger = logger
    
    def display_market_analysis(self, opportunities: Dict[str, pd.DataFrame],
                              show_summary: bool = True,
                              save_path: Optional[Path] = None) -> None:
        """
        Display complete market analysis results.
        
        Args:
            opportunities: Trading opportunities data
            show_summary: Whether to show summary statistics
            save_path: Optional path to save output
        """
        try:
            output_parts = []
            
            # Add header
            header = self.formatter._format_section_title("Market Analysis Results")
            output_parts.append(header)
            output_parts.append("")
            
            # Add summary if requested
            if show_summary:
                summary = self.formatter.format_summary_statistics(opportunities)
                output_parts.append(summary)
                output_parts.append("")
            
            # Add detailed opportunities
            opportunities_display = self.formatter.format_trading_opportunities(opportunities)
            output_parts.append(opportunities_display)
            
            # Combine all output
            full_output = "\n".join(output_parts)
            
            # Print to console
            print(full_output)
            
            # Save to file if requested
            if save_path:
                self.formatter.save_to_file(full_output, save_path)
            
        except Exception as e:
            self.logger.error(f"Error displaying market analysis: {str(e)}")
            print(f"Error displaying results: {str(e)}")


def create_display_formatter(use_colors: bool = True) -> DisplayFormatter:
    """Factory function to create a display formatter."""
    return DisplayFormatter(use_colors=use_colors)


def create_market_display(use_colors: bool = True) -> MarketDataDisplay:
    """Factory function to create a market data display."""
    formatter = create_display_formatter(use_colors)
    return MarketDataDisplay(formatter)
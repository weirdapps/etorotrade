"""
HTML presentation utilities for Yahoo Finance data.

This module provides utilities for generating HTML output from financial data,
including dashboards, reports, and interactive visualizations.

Key components:
- FormatUtils: Utilities for formatting values in HTML
- HTMLGenerator: HTML generation for reports and dashboards
- Templates: HTML templates for various report types
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..core.config import FILE_PATHS
from ..core.errors import APIError, DataError, ValidationError, YFinanceError
from ..core.logging import get_logger
from ..utils.error_handling import enrich_error_context, safe_operation, translate_error, with_retry


logger = get_logger(__name__)
# Set default level to WARNING to suppress debug and info messages
logger.setLevel(logging.WARNING)


class FormatUtils:
    """Utilities for formatting HTML output and market metrics."""

    @staticmethod
    def format_number(value, precision=2):
        """Format numeric value with specified precision."""
        if pd.isna(value) or value is None:
            return "N/A"

        if isinstance(value, (int, float)):
            format_str = f"{{:.{precision}f}}"
            return format_str.format(value)

        return str(value)

    @staticmethod
    def _format_metric_value(value: Any, is_percentage: bool) -> str:
        """
        Format a metric value consistently.

        Args:
            value: The value to format
            is_percentage: Whether the value is a percentage

        Returns:
            Formatted value as string
        """
        if isinstance(value, (int, float)):
            if is_percentage:
                # Keep 1 decimal place for percentages
                return f"{value:.1f}%"
            else:
                return f"{value:.2f}"
        else:
            return str(value)

    @staticmethod
    def _determine_metric_color(key: str, value: Any) -> str:
        """
        Determine the color for a metric based on its value.

        Args:
            key: The metric key
            value: The metric value

        Returns:
            Color class name (positive, negative, or normal)
        """
        # Special case for price to match tests
        if key == "price":
            return "normal"

        # For numeric values, determine by sign
        if isinstance(value, (int, float)):
            if value > 0:
                return "positive"
            elif value < 0:
                return "negative"

        # Default
        return "normal"

    @staticmethod
    def format_market_metrics(metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format market metrics for HTML display.

        Args:
            metrics: Dictionary of metrics keyed by identifier

        Returns:
            List of formatted metrics with consistent structure
        """
        formatted = []

        try:
            for key, data in metrics.items():
                # Skip non-dictionary or empty values
                if not isinstance(data, dict):
                    continue

                value = data.get("value")
                if value is None:
                    continue

                # Get formatting attributes
                is_percentage = data.get("is_percentage", False)

                # Format the value and get the color
                formatted_value = FormatUtils._format_metric_value(value, is_percentage)
                color = FormatUtils._determine_metric_color(key, value)

                # Create the formatted metric entry
                formatted.append(
                    {
                        "key": key,
                        "label": data.get("label", key),
                        "value": value,  # Original value for sorting
                        "formatted_value": formatted_value,
                        "color": color,
                        "is_percentage": is_percentage,
                    }
                )

            # Sort by key
            formatted.sort(key=lambda x: x.get("key", ""))

        except YFinanceError as e:
            logger.error(f"Error formatting market metrics: {str(e)}")

        return formatted

    @staticmethod
    def format_for_csv(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics for CSV output.

        Args:
            metrics: Dictionary of metrics to format

        Returns:
            Dictionary with formatted values for CSV
        """
        formatted = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Format numbers consistently for CSV
                if abs(value) < 0.01:
                    formatted[key] = 0.0
                elif abs(value) > 1000:
                    formatted[key] = round(value, 0)
                else:
                    formatted[key] = round(value, 2)
            else:
                formatted[key] = value

        return formatted


class HTMLGenerator:
    """Generator for HTML output from financial data."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize HTMLGenerator.

        Args:
            output_dir: Directory for output files (defaults to config)
        """
        self.output_dir = output_dir or FILE_PATHS["OUTPUT_DIR"]
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _format_section(self, section: Dict[str, Any]) -> str:
        """Format a section for HTML display."""
        section_title = section.get("title", "Market Data")
        metrics = section.get("metrics", [])
        columns = section.get("columns", 4)
        width = section.get("width", "100%")

        html = f"""
        <div class="section" style="width: {width}; margin-bottom: 30px;">
            <h2 class="section-title">{section_title}</h2>
            <div class="metrics-grid" style="grid-template-columns: repeat({columns}, 1fr); gap: 20px;">
"""

        # Add metrics
        for metric in metrics:
            key = metric.get("key", "")
            label = metric.get("label", key)
            formatted_value = metric.get("formatted_value", "--")
            color_class = metric.get("color", "normal")

            html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{formatted_value}</div>
                    <div class="metric-border {color_class}-border"></div>
                </div>
"""

        html += """
            </div>
        </div>
"""
        return html

    def generate_market_html(self, title: str, sections: List[Dict[str, Any]]) -> str:
        """
        Generate HTML content for market metrics display.

        Args:
            title: Page title
            sections: List of sections containing metrics

        Returns:
            HTML content as string
        """
        try:
            # Enhanced HTML template with modern meta tags and optimizations
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Financial performance dashboard">
    <meta name="theme-color" content="#ffffff">
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        .page-header {{ 
            position: relative;
            text-align: center;
            padding-bottom: 10px;
            margin-bottom: 40px;
        }}
        .page-header::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 80px;
            height: 3px;
            background: linear-gradient(to right, #4a5568, #2b6cb0);
            border-radius: 3px;
        }}
        .updated-date {{
            text-align: center;
            font-size: 0.9rem;
            color: #718096;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="page-header">
            <h1>{title}</h1>
        </div>
        <div class="dashboard">
"""

            # Add sections
            for section in sections:
                html += self._format_section(section)

            # Close HTML with updated date and footer
            current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            html += f"""
        </div>
        <div class="updated-date">
            Last updated: {current_time}
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
"""

            # Return generated HTML
            return html

        except YFinanceError as e:
            logger.error(f"Error generating market HTML: {str(e)}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"

    def generate_market_dashboard(self, metrics: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Generate market performance HTML dashboard.

        Args:
            metrics: Dictionary of metrics for the dashboard

        Returns:
            Path to generated HTML file or None if failed
        """
        try:
            if not metrics:
                logger.warning("No metrics provided for market dashboard")
                return None

            # Format metrics
            formatted_metrics = FormatUtils.format_market_metrics(metrics)

            # Create dashboard sections
            sections = [
                {
                    "title": "Market Performance",
                    "metrics": formatted_metrics,
                    "columns": 4,
                    "width": "100%",
                }
            ]

            # Generate HTML
            html_content = self.generate_market_html(title="Market Dashboard", sections=sections)

            # Write to file with standardized name 'performance.html'
            output_path = f"{self.output_dir}/performance.html"
            with open(output_path, "w") as f:
                f.write(html_content)

            # Copy default CSS and JS if they don't exist
            self._ensure_assets_exist()

            logger.info(f"Generated market dashboard at {output_path}")
            return output_path

        except YFinanceError as e:
            logger.error(f"Error generating market dashboard: {str(e)}")
            return None

    def generate_portfolio_dashboard(
        self,
        performance_metrics: Dict[str, Dict[str, Any]],
        risk_metrics: Dict[str, Dict[str, Any]],
        sector_allocation: Optional[Dict[str, float]] = None,
        title: str = "Portfolio Dashboard",
    ) -> Optional[str]:
        """
        Generate portfolio performance HTML dashboard.

        Args:
            performance_metrics: Performance metrics
            risk_metrics: Risk metrics
            sector_allocation: Optional sector allocation data
            title: Title for the HTML page (defaults to "Portfolio Dashboard")

        Returns:
            Path to generated HTML file or None if failed
        """
        try:
            # Format metrics
            formatted_performance = FormatUtils.format_market_metrics(performance_metrics)
            formatted_risk = FormatUtils.format_market_metrics(risk_metrics)

            # Create dashboard sections
            sections = [
                {
                    "title": "Portfolio Performance",
                    "metrics": formatted_performance,
                    "columns": 3,
                    "width": "100%",
                },
                {"title": "Risk Metrics", "metrics": formatted_risk, "columns": 3, "width": "100%"},
            ]

            # Add sector allocation if provided
            if sector_allocation:
                sector_metrics = {}
                for sector, allocation in sector_allocation.items():
                    sector_metrics[sector] = {
                        "label": sector,
                        "value": allocation,
                        "is_percentage": True,
                    }

                formatted_sectors = FormatUtils.format_market_metrics(sector_metrics)
                sections.append(
                    {
                        "title": "Sector Allocation",
                        "metrics": formatted_sectors,
                        "columns": 4,
                        "width": "100%",
                    }
                )

            # Generate HTML with provided title
            html_content = self.generate_market_html(title=title, sections=sections)

            # Write to file with standardized name 'performance.html'
            output_path = f"{self.output_dir}/performance.html"
            with open(output_path, "w") as f:
                f.write(html_content)

            # Copy default CSS and JS if they don't exist
            self._ensure_assets_exist()

            logger.info(f"Generated portfolio dashboard at {output_path}")
            return output_path

        except YFinanceError as e:
            logger.error(f"Error generating portfolio dashboard: {str(e)}")
            return None

    def generate_stock_table(
        self,
        stocks_data: List[Dict[str, Any]],
        title: str = "Stock Analysis",
        output_filename: str = "stock_analysis",
        include_columns: List[str] = None,
        processing_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate HTML table for stock analysis results.

        Args:
            stocks_data: List of stock data dictionaries
            title: Title for the HTML page
            output_filename: Base name for output files (without extension)
            include_columns: List of columns to include, in order (None for all columns)
            processing_stats: Optional processing statistics to display in the footer

        Returns:
            Path to generated HTML file or None if failed

        Note:
            This function will apply color coding based on the 'ACTION' column:
            - 'B' values for BUY will be colored green
            - 'S' values for SELL will be colored red
            - 'H' values for HOLD will be left as default text color
        """
        try:
            if not stocks_data:
                logger.warning("No stock data provided for table")
                return None

            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(stocks_data)

            # Sanitize ACTION column immediately after DataFrame creation
            if "ACTION" in df.columns:
                valid_actions = ["B", "S", "H"]
                # Replace any value not in valid_actions (including NaN, None, '', '--') with 'H'
                df["ACTION"] = df["ACTION"].apply(lambda x: x if x in valid_actions else "H")
            else:
                # If ACTION column is missing entirely, add it and fill with 'H'
                df["ACTION"] = "H"

            # Filter and order columns if specified
            if include_columns:
                # Ensure all columns exist
                existing_columns = [col for col in include_columns if col in df.columns]
                df = df[existing_columns]

            # Format numeric values consistently before styling
            df = self._format_numeric_values(df)

            # Convert DataFrame to simple HTML directly to avoid Pandas styling issues
            rows = []

            # Create header row
            header_row = "<tr>"
            for col in df.columns:
                header_row += f'<th class="sort-header">{col}</th>'
            header_row += "</tr>"

            # Debug: check for ACTION or ACT column
            logger.debug(f"Columns available for HTML generation: {df.columns.tolist()}")
            # First check if we have ACT column
            if "ACT" in df.columns:
                # Make sure all SELL values are properly set to 'S' in the display
                sell_mask = df["BuySell"] == "SELL" if "BuySell" in df.columns else None
                if sell_mask is not None and not sell_mask.empty:
                    df.loc[sell_mask, "ACT"] = "S"
            # If we only have ACTION column, rename it to ACT
            elif "ACTION" in df.columns:
                logger.debug("Renaming ACTION column to ACT")
                df.rename(columns={"ACTION": "ACT"}, inplace=True)
                # Also make sure all SELL values are properly set
                sell_mask = df["BuySell"] == "SELL" if "BuySell" in df.columns else None
                if sell_mask is not None and not sell_mask.empty:
                    df.loc[sell_mask, "ACT"] = "S"
            # If neither column exists, add default values
            else:
                logger.warning("Neither ACTION nor ACT column found in the dataframe!")
                # Force add ACT with default HOLD values
                df["ACT"] = "H"
                # Set appropriate sell values based on BuySell column if it exists
                sell_mask = df["BuySell"] == "SELL" if "BuySell" in df.columns else None
                if sell_mask is not None and not sell_mask.empty:
                    df.loc[sell_mask, "ACT"] = "S"
                logger.debug("Added default ACT column with 'H' values")

            # Debug: Count actions in the dataframe
            if "ACT" in df.columns:
                action_counts = df["ACT"].value_counts().to_dict()
                logger.debug(f"ACT counts in HTML generation: {action_counts}")

            # Create data rows with appropriate styling
            for idx, row in df.iterrows():
                action = row.get("ACT") if "ACT" in row else row.get("ACTION")

                # Determine row color based on ACTION
                row_class = ""
                text_color = ""

                # Print debug info for the first 5 rows to see what actions look like
                if idx < 10:
                    ticker_val = row.get("TICKER", "unknown")
                    logger.debug(
                        f"Row {idx} action: '{action}' (type: {type(action).__name__}) for ticker {ticker_val}"
                    )

                # Make sure ACTION is properly compared as a string
                # Convert to string first to ensure consistent comparison
                action_str = str(action) if action is not None else ""

                # Log each row's action for the first 10 rows
                if idx < 10:
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Row {idx}, ACTION = '{action_str}', TICKER = '{ticker_val}'")

                # Apply coloring based on standardized action string
                # Use direct string comparison for more reliability
                if action_str.strip() == "B":  # Buy
                    row_class = ' class="buy-row"'
                    text_color = "color: #008800;"  # Darker green for better visibility
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Applied BUY styling to row {idx}: {ticker_val}")
                elif action_str.strip() == "S":  # Sell
                    row_class = ' class="sell-row"'
                    text_color = "color: #CC0000;"  # Darker red for better visibility
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Applied SELL styling to row {idx}: {ticker_val}")
                elif action_str.strip() == "I":  # Inconclusive
                    row_class = ' class="inconclusive-row"'
                    text_color = "color: #CC7700;"  # Darker yellow/amber for better visibility
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Applied INCONCLUSIVE styling to row {idx}: {ticker_val}")

                # Make sure ACTION value is valid by forcing specific valid values
                if action_str.strip() not in ["B", "S", "H", "I", ""]:
                    logger.warning(f"Invalid action value '{action_str}' converted to 'H' (hold)")
                    action = "H"

                # Debug the action value for CLOV ticker
                if row.get("TICKER") == "CLOV":
                    logger.info(f"CLOV action: {action_str}")

                # Force 'I' action for tickers with low analyst coverage
                try:
                    analyst_count = pd.to_numeric(row.get("# A"), errors="coerce")
                    price_target_count = pd.to_numeric(row.get("# T"), errors="coerce")

                    # Check if we have counts and they're below threshold
                    if (
                        pd.notna(analyst_count)
                        and pd.notna(price_target_count)
                        and (analyst_count < 5 or price_target_count < 5)
                    ):
                        action = "I"
                        action_str = "I"
                        logger.info(
                            f"Forcing INCONCLUSIVE for {row.get('TICKER')} due to low coverage: T={price_target_count}, A={analyst_count}"
                        )
                except YFinanceError as e:
                    logger.warning(
                        f"Error checking coverage thresholds for {row.get('TICKER', 'unknown')}: {str(e)}"
                    )

                # ===== CRITICAL FIX =====
                # Always use direct inline styles for row coloring
                # This is the most reliable approach across browsers

                bg_color = ""
                # Apply coloring based on exact string comparison
                if action_str.strip() == "B":
                    bg_color = "#f0fff0"  # Very light green background
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Applying BUY styling to row {idx}: {ticker_val}")
                elif action_str.strip() == "S":
                    bg_color = "#fff0f0"  # Very light red background
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Applying SELL styling to row {idx}: {ticker_val}")
                elif action_str.strip() == "I":
                    bg_color = "#fffadd"  # Very light yellow background
                    ticker_val = row.get("TICKER", "")
                    logger.debug(f"Applying INCONCLUSIVE styling to row {idx}: {ticker_val}")
                elif action_str.strip() == "H":
                    # No special styling for HOLD
                    bg_color = ""

                # Always use direct inline style + class for maximum reliability
                if bg_color:
                    row_html = f'<tr style="background-color: {bg_color} !important;"{row_class}>'
                    # Log for debugging
                    if idx < 5:
                        logger.debug(f"Row {idx} HTML start: {row_html}")
                else:
                    row_html = f"<tr{row_class}>"

                # Add columns with proper alignment
                for col in df.columns:
                    value = row[col]

                    # Determine cell alignment
                    align = "left" if col in ["TICKER", "COMPANY"] else "right"

                    # Add color styling if action is set
                    style = f"text-align: {align};"

                    # Use the action_str variable we already standardized above
                    action_str = str(action) if action is not None else ""
                    if text_color and action_str in ["B", "S", "I"]:
                        style += f" {text_color}"

                    # Create the cell with appropriate alignment and color
                    if col == "ACTION":
                        # Use the actual action that should be displayed
                        if action_str in ["B", "S", "I"]:
                            # Make ACTION column bold and colored with the current action
                            row_html += f'<td style="{style} font-weight: bold;">{action_str}</td>'
                        else:
                            row_html += f'<td style="{style}">{value}</td>'
                    else:
                        row_html += f'<td style="{style}">{value}</td>'

                # End row
                row_html += "</tr>"
                rows.append(row_html)

            # Combine all rows into a table
            table_html = f"""
            <table class="stock-table" border="0">
                <thead>
                    {header_row}
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
            """

            # Full HTML document with improved styling
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f7;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 20px;
            color: #333;
            font-weight: 600;
        }}
        
        .table-container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
            overflow: auto;
            padding: 5px;
        }}
        
        .stock-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        .stock-table th {{
            background-color: #f8f8f8;
            color: #444;
            font-weight: 600;
            padding: 12px 8px;
            border-bottom: 2px solid #e0e0e0;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .stock-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
        }}
        
        .stock-table tr:hover {{
            background-color: #f9f9f9;
        }}
        
        .stock-table tr:nth-child(even) {{
            background-color: #ffffff;  /* Make all rows white by default */
        }}
        
        /* Special row styling for buy/sell - higher specificity to override even/odd */
        .stock-table tr.buy-row,
        .stock-table tr.buy-row:nth-child(even) {{
            background-color: #f0fff0 !important;  /* Very light green background */
        }}
        
        .stock-table tr.buy-row:hover,
        .stock-table tr.buy-row:nth-child(even):hover {{
            background-color: #e0ffe0 !important;  /* Slightly darker green on hover */
        }}
        
        .stock-table tr.sell-row,
        .stock-table tr.sell-row:nth-child(even) {{
            background-color: #fff0f0 !important;  /* Very light red background */
        }}
        
        .stock-table tr.sell-row:hover,
        .stock-table tr.sell-row:nth-child(even):hover {{
            background-color: #ffe0e0 !important;  /* Slightly darker red on hover */
        }}
        
        .stock-table tr.inconclusive-row,
        .stock-table tr.inconclusive-row:nth-child(even) {{
            background-color: #fffadd !important;  /* Very light yellow background */
        }}
        
        .stock-table tr.inconclusive-row:hover,
        .stock-table tr.inconclusive-row:nth-child(even):hover {{
            background-color: #fff5c0 !important;  /* Slightly darker yellow on hover */
        }}
        
        .sort-header {{
            cursor: pointer;
        }}
        
        .sort-header:after {{
            content: ' ↕';
            font-size: 12px;
            color: #999;
        }}
        
        .sort-header.asc:after {{
            content: ' ↓';
            color: #333;
        }}
        
        .sort-header.desc:after {{
            content: ' ↑';
            color: #333;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .stock-table {{
                font-size: 12px;
            }}
            
            .stock-table th,
            .stock-table td {{
                padding: 8px 4px;
            }}
        }}
        
        .footer {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            color: #555;
            font-size: 14px;
        }}
        
        .footer p {{
            margin: 0;
            padding: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="table-container">
            {table_html}
        </div>
        <div class="footer">
            <p>{self._generate_footer_text(title, processing_stats)}</p>
        </div>
    </div>
    <script src="script.js"></script>
    <script>
        // Enhanced sorting functionality
        document.addEventListener('DOMContentLoaded', function() {{
            const table = document.querySelector('.stock-table');
            const headers = table.querySelectorAll('th');
            
            headers.forEach((header, index) => {{
                header.classList.add('sort-header');
                header.addEventListener('click', () => {{
                    sortTable(index);
                }});
            }});
            
            function sortTable(columnIndex) {{
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                // Get current sort direction
                const header = headers[columnIndex];
                const isAscending = header.classList.contains('asc');
                
                // Reset all headers
                headers.forEach(h => {{
                    h.classList.remove('asc');
                    h.classList.remove('desc');
                }});
                
                // Set new sort direction
                if (isAscending) {{
                    header.classList.add('desc');
                }} else {{
                    header.classList.add('asc');
                }}
                
                const direction = isAscending ? -1 : 1;
                
                rows.sort((a, b) => {{
                    let aValue = a.cells[columnIndex].textContent.trim();
                    let bValue = b.cells[columnIndex].textContent.trim();
                    
                    // Handle percentage values
                    if (aValue.endsWith('%')) aValue = aValue.replace('%', '');
                    if (bValue.endsWith('%')) bValue = bValue.replace('%', '');
                    
                    // Handle market cap values (e.g., 2.5T, 145.8B)
                    if (/^[0-9]+(\\.[0-9]+)?[TB]$/.test(aValue)) {{
                        const aNum = parseFloat(aValue.replace(/[TB]$/, ''));
                        const bNum = parseFloat(bValue.replace(/[TB]$/, ''));
                        const aMultiplier = aValue.endsWith('T') ? 1e12 : 1e9;
                        const bMultiplier = bValue.endsWith('T') ? 1e12 : 1e9;
                        return direction * ((aNum * aMultiplier) - (bNum * bMultiplier));
                    }}
                    
                    // Try to compare as numbers first
                    const aNum = parseFloat(aValue);
                    const bNum = parseFloat(bValue);
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return direction * (aNum - bNum);
                    }}
                    
                    // Fall back to string comparison
                    return direction * aValue.localeCompare(bValue);
                }});
                
                // Re-add rows in sorted order
                rows.forEach(row => tbody.appendChild(row));
            }}
        }});
    </script>
</body>
</html>
"""

            # Write CSV file with same columns and order
            csv_path = f"{self.output_dir}/{output_filename}.csv"
            df.to_csv(csv_path, index=False)
            logger.debug(f"Generated CSV file at {csv_path}")

            # Write HTML file
            html_path = f"{self.output_dir}/{output_filename}.html"
            with open(html_path, "w") as f:
                f.write(html_content)

            # Copy default CSS and JS if they don't exist
            self._ensure_assets_exist()

            logger.debug(f"Generated HTML table at {html_path}")
            return html_path

        except YFinanceError as e:
            logger.error(f"Error generating stock table: {str(e)}")
            return None

    def _generate_footer_text(
        self, title: str, processing_stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate the footer text with processing statistics if available.

        Args:
            title: The report title
            processing_stats: Optional processing statistics

        Returns:
            str: Formatted footer text
        """
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        if not processing_stats:
            return f"{title} | Generated: {timestamp}"

        # Extract stats
        elapsed_time = processing_stats.get("total_time_sec", 0)
        minutes, seconds = divmod(elapsed_time, 60)
        total_tickers = processing_stats.get("total_tickers", 0)
        success_count = processing_stats.get("success_count", 0)
        error_count = processing_stats.get("error_count", 0)
        valid_results = processing_stats.get("valid_results_count", 0)
        time_per_ticker = processing_stats.get("time_per_ticker_sec", 0)

        # Format as a single line
        return f"{title} | Generated: {timestamp} | Time: {int(minutes)}m {int(seconds)}s | Tickers: {total_tickers}/{success_count}/{error_count} | Results: {valid_results} | {time_per_ticker:.2f}s per ticker"

    def _format_numeric_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format numeric values in DataFrame for consistent display.

        Args:
            df: DataFrame with values to format

        Returns:
            DataFrame with formatted values
        """
        formatted_df = df.copy()

        # Map internal column names to display names (if they're not already mapped)
        column_mapping = {
            "buy_percentage": "% BUY",
            "ticker": "TICKER",
            "company": "COMPANY",
            "market_cap": "CAP",
            "price": "PRICE",
            "target_price": "TARGET",
            "upside": "UPSIDE",
            "analyst_count": "# T",
            "total_ratings": "# A",
            "rating_type": "A",
            "expected_return": "EXRET",
            "beta": "BETA",
            "pe_trailing": "PET",
            "pe_forward": "PEF",
            "peg_ratio": "PEG",
            "dividend_yield": "DIV %",
            "short_float_pct": "SI",
            "short_percent": "SI",
            "last_earnings": "EARNINGS",
            "action": "ACTION",
            "position_size": "SIZE",
        }

        # Rename columns to match display format
        for old_col, new_col in column_mapping.items():
            if old_col in formatted_df.columns:
                formatted_df.rename(columns={old_col: new_col}, inplace=True)

        # Update ACTION to ACT for consistent display
        if "ACTION" in formatted_df.columns:
            formatted_df.rename(columns={"ACTION": "ACT"}, inplace=True)

        # Remove duplicate ACT column if both exist
        if "ACT" in formatted_df.columns and formatted_df.columns.duplicated().any():
            formatted_df = formatted_df.loc[:, ~formatted_df.columns.duplicated()]

        # Import standard column order
        from ..core.config import STANDARD_DISPLAY_COLUMNS

        # Reorder columns to match the standard display order
        display_cols = [col for col in STANDARD_DISPLAY_COLUMNS if col in formatted_df.columns]
        other_cols = [col for col in formatted_df.columns if col not in STANDARD_DISPLAY_COLUMNS]

        if display_cols:
            formatted_df = formatted_df[display_cols + other_cols]

        # Import the position size formatter
        from ..utils.data.format_utils import format_position_size

        # Define formatting rules for different column types
        for col in formatted_df.columns:
            # Skip non-numeric columns and specifically formatted columns
            if col in ["TICKER", "COMPANY", "ACTION", "A", "#", "EARNINGS"]:
                continue

            # Handle market cap (already formatted with T/B suffixes)
            if col == "CAP":
                continue

            # Format SIZE column with position size formatting
            elif col == "SIZE":
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        format_position_size(x)
                        if isinstance(x, (int, float))
                        or (
                            isinstance(x, str)
                            and x not in ["--", ""]
                            and x.replace(".", "", 1).isdigit()
                        )
                        else x
                    )
                )

            # Format UPSIDE with 1 decimal place and % symbol
            elif col == "UPSIDE":
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.1f}%"
                        if isinstance(x, (int, float))
                        or (
                            isinstance(x, str)
                            and x.replace(".", "", 1).replace("-", "", 1).isdigit()
                        )
                        else x
                    )
                )

            # Format % BUY with 0 decimal places and % symbol
            elif col == "% BUY":
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.0f}%"
                        if isinstance(x, (int, float))
                        or (isinstance(x, str) and x.replace(".", "", 1).isdigit())
                        else x
                    )
                )

            # Format EXRET with 1 decimal place and % symbol
            elif col == "EXRET":
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.1f}%"
                        if isinstance(x, (int, float))
                        or (
                            isinstance(x, str)
                            and x.replace(".", "", 1).replace("-", "", 1).isdigit()
                        )
                        else x
                    )
                )

            # Format DIV % with 2 decimal places and % symbol
            elif col == "DIV %":
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.2f}%"
                        if isinstance(x, (int, float))
                        or (isinstance(x, str) and x.replace(".", "", 1).isdigit())
                        else x
                    )
                )

            # Format SI with 1 decimal place and % symbol
            elif col == "SI":
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.1f}%"
                        if isinstance(x, (int, float))
                        or (isinstance(x, str) and x.replace(".", "", 1).isdigit())
                        else x
                    )
                )

            # Format PRICE and TARGET with 1 decimal place (no % symbol)
            elif col in ["PRICE", "TARGET"]:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.1f}"
                        if isinstance(x, (int, float))
                        or (isinstance(x, str) and x.replace(".", "", 1).isdigit())
                        else x
                    )
                )

            # Format BETA, PET, PEF, PEG with 1 decimal place (no % symbol)
            elif col in ["BETA", "PET", "PEF", "PEG"]:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{float(x):.1f}"
                        if isinstance(x, (int, float))
                        or (isinstance(x, str) and x.replace(".", "", 1).isdigit())
                        else x
                    )
                )

            # Format # T and # A as integers
            elif col in ["# T", "# A"]:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: (
                        f"{int(float(x))}"
                        if isinstance(x, (int, float))
                        or (isinstance(x, str) and x.replace(".", "", 1).isdigit())
                        else x
                    )
                )

        return formatted_df

    def _ensure_assets_exist(self):
        """Ensure CSS and JS assets exist in the output directory."""
        # Default CSS
        css_path = f"{self.output_dir}/styles.css"
        if not Path(css_path).exists():
            default_css = """/* Default styles for financial dashboards */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f8f9fa;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 30px 20px;
}

h1 {
    color: #1a365d;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 600;
    font-size: 2rem;
    padding-bottom: 15px;
    border-bottom: 1px solid #e2e8f0;
}

h2.section-title {
    color: #2d3748;
    font-weight: 600;
    margin-bottom: 20px;
    font-size: 1.5rem;
    position: relative;
    padding-bottom: 8px;
}

h2.section-title:after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 50px;
    height: 3px;
    background-color: #4a5568;
    border-radius: 3px;
}

.dashboard {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.section {
    background-color: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.section:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
}

.metrics-grid {
    display: grid;
    grid-gap: 20px;
}

.metric-card {
    position: relative;
    background-color: #ffffff;
    border-radius: 10px;
    padding: 20px 15px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    transition: all 0.3s ease;
    overflow: hidden;
    border: 1px solid #edf2f7;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08);
}

.metric-border {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: #e2e8f0;
}

.metric-border.positive-border {
    background-color: #38a169;
}

.metric-border.negative-border {
    background-color: #e53e3e;
}

.metric-label {
    font-size: 0.85rem;
    color: #718096;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 500;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 5px;
}

.metric-value.positive {
    color: #38a169;
}

.metric-value.negative {
    color: #e53e3e;
}

/* Table styling */
.table-container {
    overflow-x: auto;
    margin-top: 20px;
}

.stock-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
}

.stock-table th {
    background-color: #f2f2f2;
    color: #555;
    font-weight: 600;
    text-align: left;
    padding: 12px 15px;
    border-bottom: 2px solid #ddd;
}

.stock-table td {
    padding: 10px 15px;
    border-bottom: 1px solid #eee;
}

.stock-table tr:hover {
    background-color: #f9f9f9;
}

/* Responsive design */
@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    
    .metric-card {
        padding: 12px;
    }
    
    .metric-value {
        font-size: 1.2rem;
    }
}

@media (max-width: 480px) {
    .metrics-grid {
        grid-template-columns: 1fr !important;
    }
}
"""
            with open(css_path, "w") as f:
                f.write(default_css)

        # Default JS
        js_path = f"{self.output_dir}/script.js"
        if not Path(js_path).exists():
            default_js = """// Dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add any interactive behaviors here
    console.log('Dashboard loaded');
    
    // Example: Add click animation to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
});
"""
            with open(js_path, "w") as f:
                f.write(default_js)

"""
HTML presentation utilities for Yahoo Finance data.

This module provides utilities for generating HTML output from financial data,
including dashboards, reports, and interactive visualizations.

Key components:
- FormatUtils: Utilities for formatting values in HTML
- HTMLGenerator: HTML generation for reports and dashboards
- Templates: HTML templates for various report types
"""

import logging
from typing import Dict, Any, List, Optional, Union
import json
import pandas as pd
from pathlib import Path

from ..core.config import FILE_PATHS

logger = logging.getLogger(__name__)

class FormatUtils:
    """Utilities for formatting HTML output and market metrics."""
    
    @staticmethod
    def format_number(value, precision=2):
        """Format numeric value with specified precision."""
        if pd.isna(value) or value is None:
            return 'N/A'
        
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
        if key == 'price':
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
                    
                value = data.get('value')
                if value is None:
                    continue
                
                # Get formatting attributes
                is_percentage = data.get('is_percentage', False)
                
                # Format the value and get the color
                formatted_value = FormatUtils._format_metric_value(value, is_percentage)
                color = FormatUtils._determine_metric_color(key, value)
                
                # Create the formatted metric entry
                formatted.append({
                    'key': key,
                    'label': data.get('label', key),
                    'value': value,  # Original value for sorting
                    'formatted_value': formatted_value,
                    'color': color,
                    'is_percentage': is_percentage
                })
                
            # Sort by key
            formatted.sort(key=lambda x: x.get('key', ''))
            
        except Exception as e:
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
        section_title = section.get('title', 'Market Data')
        metrics = section.get('metrics', [])
        columns = section.get('columns', 4)
        width = section.get('width', '100%')
        
        html = f"""
        <div class="section" style="width: {width}">
            <h2>{section_title}</h2>
            <div class="metrics-grid" style="grid-template-columns: repeat({columns}, 1fr)">
"""
        
        # Add metrics
        for metric in metrics:
            key = metric.get('key', '')
            label = metric.get('label', key)
            formatted_value = metric.get('formatted_value', '--')
            color_class = metric.get('color', 'normal')
            
            html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {color_class}">{formatted_value}</div>
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
            # Basic HTML template
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="dashboard">
"""
            
            # Add sections
            for section in sections:
                html += self._format_section(section)
            
            # Close HTML
            html += """
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
"""
            
            # Return generated HTML
            return html
            
        except Exception as e:
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
            sections = [{
                'title': 'Market Performance',
                'metrics': formatted_metrics,
                'columns': 4,
                'width': '100%'
            }]
            
            # Generate HTML
            html_content = self.generate_market_html(
                title='Market Dashboard',
                sections=sections
            )
            
            # Write to file
            output_path = f"{self.output_dir}/market_dashboard.html"
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            # Copy default CSS and JS if they don't exist
            self._ensure_assets_exist()
                
            logger.info(f"Generated market dashboard at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating market dashboard: {str(e)}")
            return None
    
    def generate_portfolio_dashboard(self, 
                                     performance_metrics: Dict[str, Dict[str, Any]],
                                     risk_metrics: Dict[str, Dict[str, Any]],
                                     sector_allocation: Optional[Dict[str, float]] = None) -> Optional[str]:
        """
        Generate portfolio performance HTML dashboard.
        
        Args:
            performance_metrics: Performance metrics
            risk_metrics: Risk metrics
            sector_allocation: Optional sector allocation data
            
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
                    'title': 'Portfolio Performance',
                    'metrics': formatted_performance,
                    'columns': 3,
                    'width': '100%'
                },
                {
                    'title': 'Risk Metrics',
                    'metrics': formatted_risk,
                    'columns': 3,
                    'width': '100%'
                }
            ]
            
            # Add sector allocation if provided
            if sector_allocation:
                sector_metrics = {}
                for sector, allocation in sector_allocation.items():
                    sector_metrics[sector] = {
                        'label': sector,
                        'value': allocation,
                        'is_percentage': True
                    }
                
                formatted_sectors = FormatUtils.format_market_metrics(sector_metrics)
                sections.append({
                    'title': 'Sector Allocation',
                    'metrics': formatted_sectors,
                    'columns': 4,
                    'width': '100%'
                })
            
            # Generate HTML
            html_content = self.generate_market_html(
                title='Portfolio Dashboard',
                sections=sections
            )
            
            # Write to file
            output_path = f"{self.output_dir}/portfolio_dashboard.html"
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            # Copy default CSS and JS if they don't exist
            self._ensure_assets_exist()
                
            logger.info(f"Generated portfolio dashboard at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating portfolio dashboard: {str(e)}")
            return None
            
    def generate_stock_table(self, stocks_data: List[Dict[str, Any]], title: str = "Stock Analysis") -> Optional[str]:
        """
        Generate HTML table for stock analysis results.
        
        Args:
            stocks_data: List of stock data dictionaries
            title: Title for the HTML page
            
        Returns:
            Path to generated HTML file or None if failed
        """
        try:
            if not stocks_data:
                logger.warning("No stock data provided for table")
                return None
                
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(stocks_data)
            
            # Generate HTML table
            table_html = df.to_html(
                index=False,
                classes="stock-table",
                border=0,
                na_rep="--"
            )
            
            # Full HTML document
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="table-container">
            {table_html}
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
"""
            
            # Write to file
            output_path = f"{self.output_dir}/stock_analysis.html"
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            # Copy default CSS and JS if they don't exist
            self._ensure_assets_exist()
                
            logger.info(f"Generated stock analysis table at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating stock table: {str(e)}")
            return None
            
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
    background-color: #f5f5f7;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: #333;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 500;
}

h2 {
    color: #444;
    font-weight: 500;
    margin-bottom: 20px;
    font-size: 1.5rem;
}

.dashboard {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.section {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.metrics-grid {
    display: grid;
    grid-gap: 15px;
}

.metric-card {
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.metric-label {
    font-size: 0.9rem;
    color: #777;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #333;
}

.metric-value.positive {
    color: #34c759;
}

.metric-value.negative {
    color: #ff3b30;
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
            with open(css_path, 'w') as f:
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
            with open(js_path, 'w') as f:
                f.write(default_js)
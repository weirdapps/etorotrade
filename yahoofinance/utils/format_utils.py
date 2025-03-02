"""
Formatting utilities for HTML output and other display formats.

This module contains utilities for generating and formatting HTML content,
including market metrics, portfolio data, and visual elements.
"""

import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FormatUtils:
    """Utilities for formatting HTML output and market metrics."""
    
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
                if not isinstance(data, dict):
                    continue
                
                value = data.get('value')
                if value is None:
                    continue
                
                # Format values consistently
                is_percentage = data.get('is_percentage', False)
                formatted_value = None
                
                if isinstance(value, (int, float)):
                    if is_percentage:
                        formatted_value = f"{value:.1f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                # Determine color
                color = "normal"
                if isinstance(value, (int, float)):
                    if value > 0:
                        color = "positive"
                    elif value < 0:
                        color = "negative"
                
                formatted.append({
                    'key': key,
                    'label': data.get('label', key),
                    'value': value,  # Original value for sorting
                    'formatted_value': formatted_value,  # Formatted for display
                    'color': color,
                    'is_percentage': is_percentage
                })
                
            # Sort by key (can be customized as needed)
            formatted.sort(key=lambda x: x.get('key', ''))
            
        except Exception as e:
            logger.error(f"Error formatting market metrics: {str(e)}")
            
        return formatted
    
    @staticmethod
    def generate_market_html(title: str, sections: List[Dict[str, Any]]) -> str:
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
                section_title = section.get('title', 'Market Data')
                metrics = section.get('metrics', [])
                columns = section.get('columns', 4)
                width = section.get('width', '100%')
                
                html += f"""
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
            
            # Close HTML
            html += """
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
"""
            
            # Add CSS file if it doesn't exist
            css = """
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    text-align: center;
    margin-bottom: 30px;
    color: #333;
}

.dashboard {
    display: flex;
    flex-direction: column;
    gap: 30px;
}

.section {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin: 0 auto;
}

h2 {
    margin-top: 0;
    margin-bottom: 20px;
    color: #444;
    font-size: 1.5em;
}

.metrics-grid {
    display: grid;
    gap: 15px;
}

.metric-card {
    background-color: #f9f9f9;
    border-radius: 6px;
    padding: 15px;
    text-align: center;
}

.metric-label {
    font-size: 0.9em;
    color: #666;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.3em;
    font-weight: bold;
    color: #333;
}

.positive {
    color: #28a745;
}

.negative {
    color: #dc3545;
}

.normal {
    color: #333;
}

@media (max-width: 768px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr) !important;
    }
    
    .section {
        width: 100% !important;
    }
}
"""
            
            # Add JavaScript file
            js = """
// Simple script to handle any dynamic behavior
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard loaded');
});
"""
            
            # Return generated HTML
            return html
            
        except Exception as e:
            logger.error(f"Error generating market HTML: {str(e)}")
            return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
    
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
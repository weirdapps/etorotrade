#!/usr/bin/env python
"""
Yahoo Finance V2 Presentation Example

This example demonstrates the presentation layer of yahoofinance_v2
including console display, HTML generation, and data formatting.

Usage:
    python v2_presentation_example.py
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import the yahoofinance_v2 package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yahoofinance_v2.api import get_provider
from yahoofinance_v2.analysis.stock import StockAnalyzer
from yahoofinance_v2.analysis.portfolio import PortfolioAnalyzer
from yahoofinance_v2.presentation.console import MarketDisplay
from yahoofinance_v2.presentation.html import HTMLGenerator
from yahoofinance_v2.presentation.templates import TemplateEngine

def ensure_output_dir():
    """Ensure output directory exists"""
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)

def format_and_display_stocks():
    """Format and display stock data in console and HTML"""
    print("\n== Stock Analysis Example ==")
    
    # Create provider and analyzers
    provider = get_provider()
    analyzer = StockAnalyzer(provider)
    
    # Define sample tickers to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Analyze each ticker
    results = []
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        analysis = analyzer.analyze_ticker(ticker)
        results.append(analysis.to_dict())
        
        # Wait between requests to avoid rate limiting
        time.sleep(1)
    
    # Display in console
    console_display = MarketDisplay()
    console_display.display_stock_table(results, "Tech Giants Analysis")
    
    # Save to CSV
    output_dir = ensure_output_dir()
    csv_path = console_display.save_to_csv(results, "tech_stocks.csv", output_dir)
    print(f"\nSaved analysis to CSV: {csv_path}")
    
    # Generate HTML
    html_generator = HTMLGenerator(output_dir)
    html_path = html_generator.generate_stock_table(results, "Tech Giants Analysis")
    print(f"Generated HTML report: {html_path}")
    
    return results

def generate_market_dashboard():
    """Generate market dashboard with HTML"""
    print("\n== Market Dashboard Example ==")
    
    # Create provider to get market data
    provider = get_provider()
    
    # Get metrics for market indices
    metrics = {
        "SPY": {
            "label": "S&P 500",
            "value": 1.2,
            "is_percentage": True
        },
        "QQQ": {
            "label": "NASDAQ",
            "value": 1.8,
            "is_percentage": True
        },
        "DIA": {
            "label": "Dow Jones",
            "value": 0.7,
            "is_percentage": True
        },
        "IWM": {
            "label": "Russell 2000",
            "value": -0.5,
            "is_percentage": True
        },
        "VIX": {
            "label": "Volatility",
            "value": -3.2,
            "is_percentage": True
        }
    }
    
    # Generate HTML
    output_dir = ensure_output_dir()
    html_generator = HTMLGenerator(output_dir)
    html_path = html_generator.generate_market_dashboard(metrics)
    print(f"Generated market dashboard: {html_path}")

def generate_portfolio_dashboard():
    """Generate portfolio dashboard with HTML"""
    print("\n== Portfolio Dashboard Example ==")
    
    # Sample portfolio metrics
    performance_metrics = {
        "TODAY": {
            "label": "Today",
            "value": 0.8,
            "is_percentage": True
        },
        "MTD": {
            "label": "Month-to-Date",
            "value": 3.2,
            "is_percentage": True
        },
        "YTD": {
            "label": "Year-to-Date",
            "value": 12.5,
            "is_percentage": True
        }
    }
    
    risk_metrics = {
        "BETA": {
            "label": "Beta",
            "value": 1.1,
            "is_percentage": False
        },
        "ALPHA": {
            "label": "Alpha",
            "value": 2.3,
            "is_percentage": True
        },
        "SHARPE": {
            "label": "Sharpe Ratio",
            "value": 1.7,
            "is_percentage": False
        }
    }
    
    # Sample sector allocation
    sector_allocation = {
        "Technology": 35.0,
        "Healthcare": 15.0,
        "Financials": 12.0,
        "Consumer Cyclical": 10.0,
        "Communication Services": 8.0,
        "Industrials": 7.0,
        "Consumer Defensive": 5.0,
        "Utilities": 3.0,
        "Energy": 3.0,
        "Real Estate": 2.0
    }
    
    # Generate HTML
    output_dir = ensure_output_dir()
    html_generator = HTMLGenerator(output_dir)
    html_path = html_generator.generate_portfolio_dashboard(
        performance_metrics, 
        risk_metrics,
        sector_allocation
    )
    print(f"Generated portfolio dashboard: {html_path}")

def advanced_template_usage():
    """Demonstrate advanced template usage"""
    print("\n== Advanced Template Usage Example ==")
    
    # Create template engine
    engine = TemplateEngine()
    
    # Render metrics for a dashboard section
    metrics_html = []
    
    # Add some metric cards
    metrics_html.append(engine.render_metric("Revenue", "$2.5B", "positive"))
    metrics_html.append(engine.render_metric("Profit", "$850M", "positive"))
    metrics_html.append(engine.render_metric("Expenses", "$1.65B", "normal"))
    metrics_html.append(engine.render_metric("Growth", "-2.1%", "negative"))
    
    # Render the section
    section_html = engine.render_section(
        title="Financial Overview",
        metrics=metrics_html,
        columns=4
    )
    
    # Create chart section
    chart_html = engine.render_chart(
        chart_id="revenue_chart",
        title="Quarterly Revenue"
    )
    
    # Create chart script
    chart_script = engine.render_chart_script(
        canvas_id="revenue_chart_canvas",
        chart_type="bar",
        labels=["Q1", "Q2", "Q3", "Q4"],
        data=[1.8, 2.2, 2.5, 2.7],
        dataset_label="Revenue ($B)"
    )
    
    # Combine sections and render full page
    dashboard_html = engine.render_dashboard([section_html, chart_html])
    page_html = engine.render_base_html(
        title="Financial Dashboard",
        content=dashboard_html,
        extra_scripts='<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n' + chart_script
    )
    
    # Write to file
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "advanced_dashboard.html")
    with open(output_path, "w") as f:
        f.write(page_html)
    
    # Also write CSS and JS files if they don't exist
    css_path = os.path.join(output_dir, "styles.css")
    if not os.path.exists(css_path):
        with open(css_path, "w") as f:
            f.write(engine.get_default_css())
    
    js_path = os.path.join(output_dir, "script.js")
    if not os.path.exists(js_path):
        with open(js_path, "w") as f:
            f.write(engine.get_default_js())
    
    print(f"Generated advanced dashboard: {output_path}")

def main():
    """Main function to run all examples"""
    print("Yahoo Finance V2 Presentation Examples")
    print("="*40)
    
    # Run examples
    stock_results = format_and_display_stocks()
    generate_market_dashboard()
    generate_portfolio_dashboard()
    advanced_template_usage()
    
    print("\nAll examples completed successfully!")
    print(f"Check the 'output' directory for generated files")

if __name__ == "__main__":
    main()
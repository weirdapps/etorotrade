"""
Example usage of the yfinance2 package.
This script demonstrates how to use the main features of the package.
"""

from . import (
    YFinanceClient,
    AnalystData,
    PricingAnalyzer,
    MarketDisplay,
    DisplayConfig
)
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def analyze_single_stock(ticker: str):
    """Demonstrate analyzing a single stock"""
    client = YFinanceClient()
    
    # Get stock price and targets
    pricing = PricingAnalyzer(client)
    price_metrics = pricing.calculate_price_metrics(ticker)
    print(f"\nPrice Metrics for {ticker}:")
    print(f"Current Price: ${price_metrics['current_price']:.2f}")
    print(f"Target Price: ${price_metrics['target_price']:.2f}")
    print(f"Upside Potential: {price_metrics['upside_potential']:.1f}%")
    
    # Get analyst ratings
    analyst = AnalystData(client)
    ratings = analyst.get_ratings_summary(ticker)
    print(f"\nAnalyst Ratings:")
    print(f"Positive Ratings: {ratings['positive_percentage']:.1f}%")
    print(f"Total Ratings: {ratings['total_ratings']}")
    
    # Show recent changes
    changes = analyst.get_recent_changes(ticker, days=30)
    if changes:
        print("\nRecent Rating Changes:")
        for change in changes:
            print(f"{change['date']} - {change['firm']}: {change['from_grade']} → {change['to_grade']}")

def display_market_analysis(tickers: list):
    """Demonstrate market analysis display"""
    # Configure display settings
    config = DisplayConfig(
        use_colors=True,
        float_precision=2,
        percentage_precision=1,
        min_analysts=4,
        high_upside=15.0,
        high_buy_percent=65.0
    )
    
    # Create display instance and show report
    display = MarketDisplay(config=config)
    display.display_report(tickers)

def main():
    """Run example demonstrations"""
    print("YFinance2 Package Demo\n")
    
    # Single stock analysis
    print("=== Single Stock Analysis ===")
    analyze_single_stock("AAPL")
    
    # Market analysis
    print("\n=== Market Analysis ===")
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    display_market_analysis(sample_tickers)

if __name__ == "__main__":
    # Import the required modules when running as script
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
#!/usr/bin/env python3
"""
Test script for HTML generation with the enhanced HTMLGenerator.
"""

import os
import pandas as pd
from yahoofinance.presentation.html import HTMLGenerator

def test_html_generation():
    """Test the HTML generation with sample data."""
    # Create sample data
    data = [
        {
            "ticker": "AAPL", 
            "company": "APPLE INC", 
            "market_cap": 2700000000000, 
            "price": 175.5, 
            "target_price": 210.25, 
            "upside": 19.8, 
            "total_ratings": 42, 
            "buy_percentage": 85, 
            "analyst_count": 45, 
            "A": "A", 
            "EXRET": 16.83, 
            "beta": 1.2, 
            "pe_trailing": 28.5, 
            "pe_forward": 25.8, 
            "peg_ratio": 2.5, 
            "dividend_yield": 0.56, 
            "short_percent": 0.8, 
            "last_earnings": "2023-07-27",
            "ACTION": "B"
        },
        {
            "ticker": "MSFT", 
            "company": "MICROSOFT CORP", 
            "market_cap": 3100000000000, 
            "price": 422.1, 
            "target_price": 465.5, 
            "upside": 10.3, 
            "total_ratings": 38, 
            "buy_percentage": 92, 
            "analyst_count": 40, 
            "A": "A", 
            "EXRET": 9.5, 
            "beta": 0.9, 
            "pe_trailing": 35.2, 
            "pe_forward": 31.5, 
            "peg_ratio": 1.9, 
            "dividend_yield": 0.72, 
            "short_percent": 0.5, 
            "last_earnings": "2023-08-15",
            "ACTION": "H"
        },
        {
            "ticker": "AMZN", 
            "company": "AMAZON.COM INC", 
            "market_cap": 1800000000000, 
            "price": 182.3, 
            "target_price": 205.8, 
            "upside": 12.9, 
            "total_ratings": 45, 
            "buy_percentage": 89, 
            "analyst_count": 48, 
            "A": "A", 
            "EXRET": 11.5, 
            "beta": 1.3, 
            "pe_trailing": 78.5, 
            "pe_forward": 45.8, 
            "peg_ratio": 2.2, 
            "dividend_yield": 0.0, 
            "short_percent": 0.7, 
            "last_earnings": "2023-08-03",
            "ACTION": "H"
        },
        {
            "ticker": "NFLX", 
            "company": "NETFLIX INC", 
            "market_cap": 250000000000, 
            "price": 622.7, 
            "target_price": 575.4, 
            "upside": -7.6, 
            "total_ratings": 35, 
            "buy_percentage": 60, 
            "analyst_count": 37, 
            "A": "A", 
            "EXRET": -4.56, 
            "beta": 1.5, 
            "pe_trailing": 52.8, 
            "pe_forward": 48.5, 
            "peg_ratio": 3.2, 
            "dividend_yield": 0.0, 
            "short_percent": 2.5, 
            "last_earnings": "2023-07-19",
            "ACTION": "S"
        }
    ]
    
    # Convert to DataFrame for display
    df = pd.DataFrame(data)
    
    # Add ranking column
    df.insert(0, "#", range(1, len(df) + 1))
    
    # Rename columns to match display format
    column_mapping = {
        "ticker": "TICKER",
        "company": "COMPANY",
        "market_cap": "CAP",
        "price": "PRICE",
        "target_price": "TARGET",
        "upside": "UPSIDE",
        "analyst_count": "# T",
        "buy_percentage": "BUY %",
        "total_ratings": "# A",
        "A": "A",
        "EXRET": "EXRET",
        "beta": "BETA",
        "pe_trailing": "PET",
        "pe_forward": "PEF",
        "peg_ratio": "PEG",
        "dividend_yield": "DIV %",
        "short_percent": "SI",
        "last_earnings": "EARNINGS",
        "ACTION": "ACTION"
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Format market cap to use T/B suffixes (simple formatting for test)
    df['CAP'] = df['CAP'].apply(lambda x: f"{x/1e12:.2f}T" if x >= 1e12 else f"{x/1e9:.2f}B")
    
    # Add SI value columns
    df['% SI'] = df['SI'].apply(lambda x: f"{x:.1f}%")
    df['SI_value'] = df['SI']
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yahoofinance/output")
    csv_path = os.path.join(output_dir, "test_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")
    
    # Generate HTML file
    html_generator = HTMLGenerator(output_dir=output_dir)
    stocks_data = df.to_dict(orient='records')
    
    html_path = html_generator.generate_stock_table(
        stocks_data=stocks_data,
        title="Test Stock Analysis",
        output_filename="test_table",
        include_columns=list(df.columns)
    )
    
    print(f"HTML file saved to {html_path}")
    
    return df

if __name__ == "__main__":
    df = test_html_generation()
    print("\nGenerated DataFrame:")
    print(df)
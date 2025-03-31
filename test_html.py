#!/usr/bin/env python3
"""
Test script to check HTML generation for market data.
"""

import os
import sys
import pandas as pd
from yahoofinance.presentation.html import HTMLGenerator
from yahoofinance.core.config import PATHS, FILE_PATHS

def test_html_generation():
    """Test function to check HTML generation for market data."""
    try:
        # Get file paths
        output_dir = PATHS["OUTPUT_DIR"]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Test with simple data
        test_data = [
            {
                '#': 1,
                'TICKER': 'AAPL',
                'COMPANY': 'APPLE INC',
                'PRICE': 175.5,
                'CAP': '2.8T',
                'TARGET': 200.0,
                'UPSIDE': '14.0%',
                '% BUY': '85%',
                '# T': 25,
                '# A': 30,
                'BETA': 1.2,
                'PET': 28.5,
                'PEF': 25.0,
                'PEG': 1.5,
                'DIV %': '0.50%',
                'SI': '0.8%',
                'ACTION': 'B'
            },
            {
                '#': 2,
                'TICKER': 'MSFT',
                'COMPANY': 'MICROSOFT CORP',
                'PRICE': 380.0,
                'CAP': '2.9T',
                'TARGET': 420.0,
                'UPSIDE': '10.5%',
                '% BUY': '90%',
                '# T': 28,
                '# A': 32,
                'BETA': 0.9,
                'PET': 32.0,
                'PEF': 28.0,
                'PEG': 1.8,
                'DIV %': '0.80%',
                'SI': '0.6%',
                'ACTION': 'H'
            }
        ]
        
        # Convert to pandas DataFrame for debugging
        df = pd.DataFrame(test_data)
        print(f"Created test DataFrame with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create HTML generator
        html_generator = HTMLGenerator(output_dir=output_dir)
        
        # Generate HTML for market data
        market_path = html_generator.generate_stock_table(
            stocks_data=test_data,
            title="Test Market Analysis",
            output_filename="market"
        )
        
        print(f"Market HTML generation complete: {market_path}")
        
    except Exception as e:
        import traceback
        print(f"Test HTML generation failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_html_generation()
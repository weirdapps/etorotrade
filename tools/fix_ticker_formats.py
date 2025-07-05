#!/usr/bin/env python
"""
Fix ticker formats in portfolio.csv to conform to Yahoo Finance standards.

Usage:
    python fix_ticker_formats.py [input_file] [output_file]
    python fix_ticker_formats.py --test

Options:
    --test    Run unit tests for ticker format conversion, especially Hong Kong tickers

If no arguments are provided, it will use default paths:
    - Input: yahoofinance/input/portfolio.csv
    - Output: yahoofinance/input/portfolio.csv

The script automatically detects ticker columns with names: 'ticker', 'Ticker', 'Symbol', or 'symbol'.

Formatting rules:
    - Hong Kong tickers (.HK): Format to have exactly 4 digits (e.g., 1.HK -> 0001.HK, 700.HK -> 0700.HK)
    - Cryptocurrencies: Add USD suffix if missing (e.g., BTC -> BTC-USD)
    - VIX futures: Convert from VIX.MAY25 to ^VIXMAY25 format
"""

import os
import sys
import csv

# String constants to avoid duplication
ERROR_PREFIX = "Error:"
NO_TICKER_COLUMN_MSG = "No ticker column found. Expected one of:"
AVAILABLE_COLUMNS_MSG = "Available columns:"
TICKER_FORMATS_FIXED_MSG = "Ticker formats fixed and saved to"
NO_CHANGES_NEEDED_MSG = "No ticker format changes were needed."
ERROR_PROCESSING_MSG = "Error processing file:"


def fix_yahoo_ticker_format(ticker):
    """
    Convert ticker to Yahoo Finance format.
    
    Rules:
    - HK tickers: Format to have exactly 4 digits (e.g., 1.HK -> 0001.HK, 700.HK -> 0700.HK)
    - Cryptocurrencies: BTC/ETH -> BTC-USD/ETH-USD
    - Options/Futures formatting with ^ and = symbols as needed
    - All other tickers remain unchanged
    """
    # Backup original ticker
    original = ticker
    
    # Handle Hong Kong tickers (ensure 4 digits with leading zeros)
    if ticker.endswith('.HK'):
        numeric_part = ticker.split('.')[0]
        try:
            # First remove any leading zeros
            cleaned_numeric = numeric_part.lstrip('0')
            # If empty, this was all zeros
            if not cleaned_numeric:
                cleaned_numeric = '0'
            # Now format to have exactly 4 digits with leading zeros
            formatted_numeric = cleaned_numeric.zfill(4)
            ticker = f"{formatted_numeric}.HK"
        except Exception:
            # If any error occurs, keep original
            ticker = original
    
    # Handle cryptocurrencies (add -USD suffix if not present)
    elif ticker in ('BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK', 'XLM', 'DOGE'):
        ticker = f"{ticker}-USD"
    
    # Handle special VIX futures like VIX.MAY25
    elif ticker.startswith('VIX.'):
        # Extract month and year
        try:
            parts = ticker.split('.')
            if len(parts) == 2 and len(parts[1]) >= 5:
                month = parts[1][:3]
                year = parts[1][3:]
                
                # Convert to Yahoo Finance format
                ticker = f"^VIX{month}{year}"
        except Exception:
            # If any error occurs, keep original
            ticker = original
    
    # Return the fixed ticker
    return ticker


def fix_portfolio_tickers(input_path, output_path):
    """
    Read portfolio CSV, fix ticker formats, and save to output path.
    """
    # Ensure input file exists
    if not os.path.exists(input_path):
        print(f"{ERROR_PREFIX} Input file {input_path} does not exist.")
        return False
    
    # Read, fix, and write in one pass if the input and output are the same file
    if input_path == output_path:
        try:
            # Read all data
            with open(input_path, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
            
            # Check if file is empty or has no header
            if not data:
                print(f"{ERROR_PREFIX} Input file is empty.")
                return False
            
            # Find the ticker column (try multiple possible column names)
            header = data[0]
            ticker_idx = None
            possible_columns = ['ticker', 'Ticker', 'Symbol', 'symbol']
            
            for col_name in possible_columns:
                try:
                    ticker_idx = header.index(col_name)
                    print(f"Found ticker column: '{col_name}'")
                    break
                except ValueError:
                    continue
            
            if ticker_idx is None:
                print(f"{ERROR_PREFIX} {NO_TICKER_COLUMN_MSG} {possible_columns}")
                print(f"{AVAILABLE_COLUMNS_MSG} {header}")
                return False
            
            # Fix tickers
            changes_made = False
            for i in range(1, len(data)):
                if i < len(data) and ticker_idx < len(data[i]):
                    original = data[i][ticker_idx]
                    fixed = fix_yahoo_ticker_format(original)
                    
                    if original != fixed:
                        data[i][ticker_idx] = fixed
                        changes_made = True
                        print(f"Changed: {original} -> {fixed}")
            
            # Write data back
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)
            
            if changes_made:
                print(f"{TICKER_FORMATS_FIXED_MSG} {output_path}")
            else:
                print(NO_CHANGES_NEEDED_MSG)
            
            return True
            
        except Exception as e:
            print(f"{ERROR_PROCESSING_MSG} {e}")
            return False
    else:
        # If input and output are different, read and write separately
        try:
            with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                # Read header
                header = next(reader)
                writer.writerow(header)
                
                # Find the ticker column (try multiple possible column names)
                ticker_idx = None
                possible_columns = ['ticker', 'Ticker', 'Symbol', 'symbol']
                
                for col_name in possible_columns:
                    try:
                        ticker_idx = header.index(col_name)
                        print(f"Found ticker column: '{col_name}'")
                        break
                    except ValueError:
                        continue
                
                if ticker_idx is None:
                    print(f"{ERROR_PREFIX} {NO_TICKER_COLUMN_MSG} {possible_columns}")
                    print(f"{AVAILABLE_COLUMNS_MSG} {header}")
                    return False
                
                # Process rows
                changes_made = False
                for row in reader:
                    if ticker_idx < len(row):
                        original = row[ticker_idx]
                        fixed = fix_yahoo_ticker_format(original)
                        
                        if original != fixed:
                            row[ticker_idx] = fixed
                            changes_made = True
                            print(f"Changed: {original} -> {fixed}")
                    
                    writer.writerow(row)
                
                if changes_made:
                    print(f"{TICKER_FORMATS_FIXED_MSG} {output_path}")
                else:
                    print(NO_CHANGES_NEEDED_MSG)
                
                return True
                
        except Exception as e:
            print(f"{ERROR_PROCESSING_MSG} {e}")
            return False


def main():
    """Main function to parse arguments and run the script."""
    # Default paths
    default_input = os.path.join('yahoofinance', 'input', 'portfolio.csv')
    default_output = default_input
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Process arguments
    if len(sys.argv) == 1:
        # Use defaults
        input_path = os.path.join(project_dir, default_input)
        output_path = os.path.join(project_dir, default_output)
    elif len(sys.argv) == 2:
        # Custom input, default output
        input_path = sys.argv[1]
        output_path = os.path.join(project_dir, default_output)
    elif len(sys.argv) >= 3:
        # Custom input and output
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    
    # Run the conversion
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    fix_portfolio_tickers(input_path, output_path)


def test_hk_ticker_format():
    """Test Hong Kong ticker formatting with various examples."""
    test_cases = [
        ("0700.HK", "0700.HK"),
        ("00700.HK", "0700.HK"),
        ("000700.HK", "0700.HK"),
        ("00001.HK", "0001.HK"),
        ("0001.HK", "0001.HK"),
        ("1.HK", "0001.HK"),   # Add leading zeros
        ("00000.HK", "0000.HK"),
        ("9988.HK", "9988.HK"),  # Already 4 digits, no change
        ("123.HK", "0123.HK"),  # Add leading zero
    ]
    
    success = True
    for input_ticker, expected_output in test_cases:
        actual_output = fix_yahoo_ticker_format(input_ticker)
        if actual_output != expected_output:
            print(f"FAILED: {input_ticker} -> {actual_output} (expected {expected_output})")
            success = False
        else:
            print(f"PASSED: {input_ticker} -> {actual_output}")
    
    if success:
        print("All Hong Kong ticker format tests passed!")
    else:
        print("Some tests failed. Please check the implementation.")
    
    return success


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_hk_ticker_format()
    else:
        main()
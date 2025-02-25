#!/usr/bin/env python3
"""
Command-line interface for the market analysis tool.
"""

import logging
import sys
import os
import pandas as pd
from tabulate import tabulate
from yahoofinance.display import MarketDisplay
from yahoofinance.formatting import DisplayFormatter, DisplayConfig, Color

# Define constants for column names
BUY_PERCENTAGE = '% BUY'
DIVIDEND_YIELD = 'DIV %'

# Set up logging with INFO level
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_trade_recommendations(action_type):
    """Generate trade recommendations based on analysis"""
    try:
        # Define file paths
        output_dir = "yahoofinance/output"
        input_dir = "yahoofinance/input"
        market_path = f"{output_dir}/market.csv"
        portfolio_path = f"{input_dir}/portfolio.csv"
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory not found: {output_dir}")
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if files exist
        if not os.path.exists(market_path):
            logger.error(f"Market file not found: {market_path}")
            print("Please run the market analysis (M) first to generate data.")
            return
            
        if not os.path.exists(portfolio_path):
            logger.error(f"Portfolio file not found: {portfolio_path}")
            print("Portfolio file not found. Please run the portfolio analysis (P) first.")
            return
        
        # Read market and portfolio data
        market_df = pd.read_csv(market_path)
        portfolio_df = pd.read_csv(portfolio_path)
        
        # Extract tickers from portfolio - check for different possible column names
        ticker_column = None
        for col in ['ticker', 'TICKER', 'symbol', 'SYMBOL']:
            if col in portfolio_df.columns:
                ticker_column = col
                break
                
        if ticker_column is None:
            logger.error("Could not find ticker column in portfolio file")
            print("Could not find ticker column in portfolio file. Expected 'ticker' or 'symbol'.")
            return
            
        portfolio_tickers = set(portfolio_df[ticker_column].str.upper())
        
        if action_type == 'N':  # New buy opportunities
            # Check for the STRONG_BUY opportunities (high upside and high buy %)
            # Buy criteria: min 4 analysts, >15% upside, >65% buy percentage
            buy_opportunities = market_df[
                (market_df['analyst_count'] >= 4) & 
                (market_df['upside'] > 15.0) & 
                (market_df['buy_percentage'] > 65.0)
            ].copy()
            
            # Filter out stocks already in portfolio
            new_opportunities = buy_opportunities[~buy_opportunities['ticker'].str.upper().isin(portfolio_tickers)]
            
            # Sort by EXRET (descending)
            if not new_opportunities.empty and 'EXRET' in new_opportunities.columns:
                new_opportunities = new_opportunities.sort_values('EXRET', ascending=False)
            
            # Select relevant columns and format for display
            if new_opportunities.empty:
                print("\nNo new buy opportunities found matching criteria.")
            else:
                # Check if EXRET column exists - calculate it if not
                if 'EXRET' not in new_opportunities.columns:
                    # Calculate EXRET if we have the necessary columns
                    if ('upside' in new_opportunities.columns and 
                        'buy_percentage' in new_opportunities.columns and
                        pd.api.types.is_numeric_dtype(new_opportunities['upside']) and
                        pd.api.types.is_numeric_dtype(new_opportunities['buy_percentage'])):
                        new_opportunities['EXRET'] = new_opportunities['upside'] * new_opportunities['buy_percentage'] / 100
                    else:
                        new_opportunities['EXRET'] = None
                
                # Convert company name to uppercase and truncate to 20 characters
                new_opportunities['company'] = new_opportunities['company'].apply(
                    lambda x: str(x).upper()[:20]
                )
                
                # Select and rename columns for display, including all requested columns
                columns_to_select = [
                    'ticker', 'company', 'price', 'target_price', 'upside', 'analyst_count',
                    'buy_percentage', 'total_ratings', 'A', 'EXRET', 'beta',
                    'pe_trailing', 'pe_forward', 'peg_ratio', 'dividend_yield',
                    'short_float_pct', 'last_earnings'
                ]
                
                # Create a dictionary mapping column names
                column_mapping = {
                    'ticker': 'TICKER',
                    'company': 'COMPANY NAME',
                    'price': 'PRICE',
                    'target_price': 'TARGET',
                    'upside': 'UPSIDE',
                    'analyst_count': '# T',
                    'buy_percentage': BUY_PERCENTAGE,
                    'total_ratings': '# A',
                    'A': 'A',
                    'EXRET': 'EXRET',
                    'beta': 'BETA',
                    'pe_trailing': 'PET',
                    'pe_forward': 'PEF',
                    'peg_ratio': 'PEG',
                    'dividend_yield': DIVIDEND_YIELD,
                    'short_float_pct': 'SI',
                    'last_earnings': 'EARNINGS'
                }
                
                # Select only columns that exist in the dataframe
                available_columns = [col for col in columns_to_select if col in new_opportunities.columns]
                display_df = new_opportunities[available_columns].copy()
                
                # Rename columns according to mapping
                display_df.rename(columns={col: column_mapping[col] for col in available_columns if col in column_mapping}, inplace=True)
                
                # Format numeric columns with proper decimal places
                price_columns = ['PRICE', 'TARGET', 'BETA', 'PET', 'PEF', 'PEG']
                for col in price_columns:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.2f}" if pd.notnull(x) else "--"
                        )
                
                # Format percentage columns
                percentage_columns = ['UPSIDE', BUY_PERCENTAGE, 'EXRET', DIVIDEND_YIELD, 'SI']
                for col in percentage_columns:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1f}%" if pd.notnull(x) else "--"
                        )
                
                # Format date columns
                if 'EARNINGS' in display_df.columns:
                    def format_date(date_str):
                        if pd.notnull(date_str) and date_str != '--':
                            try:
                                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                            except ValueError:
                                return date_str
                        return '--'
                    
                    display_df['EARNINGS'] = display_df['EARNINGS'].apply(format_date)
                
                # Define column alignment (left-align TICKER and COMPANY NAME, right-align others)
                colalign = []
                for col in display_df.columns:
                    if col in ['TICKER', 'COMPANY NAME']:
                        colalign.append('left')
                    else:
                        colalign.append('right')
                
                # Display results
                print("\nNew Buy Opportunities (not in current portfolio):")
                print(tabulate(
                    display_df,
                    headers='keys',
                    tablefmt='fancy_grid',
                    showindex=False,
                    colalign=colalign
                ))
                print(f"\nTotal opportunities: {len(display_df)}")
                
                # Save to CSV
                output_file = os.path.join(output_dir, "buy.csv")
                display_df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
                
        elif action_type == 'E':  # Sell recommendations for existing portfolio
            # If portfolio.csv doesn't have output data, use the output/portfolio.csv file
            portfolio_output_path = f"{output_dir}/portfolio.csv"
            if os.path.exists(portfolio_output_path):
                portfolio_analysis_df = pd.read_csv(portfolio_output_path)
                
                # Sell criteria: stocks with analyst coverage that have low upside or low buy percentage
                sell_candidates = portfolio_analysis_df[
                    (portfolio_analysis_df['analyst_count'] >= 4) & 
                    ((portfolio_analysis_df['upside'] < 5.0) | 
                     (portfolio_analysis_df['buy_percentage'] < 50.0))
                ].copy()
                
                if sell_candidates.empty:
                    print("\nNo sell candidates found matching criteria in your portfolio.")
                    # Create an empty CSV even when no candidates are found
                    output_file = os.path.join(output_dir, "sell.csv")
                    pd.DataFrame(columns=['TICKER', 'COMPANY NAME', 'PRICE', 'TARGET', 'UPSIDE', '# T', 
                                          BUY_PERCENTAGE, '# A', 'A', 'EXRET', 'BETA', 'PET', 'PEF', 
                                          'PEG', DIVIDEND_YIELD, 'SI', 'EARNINGS']).to_csv(output_file, index=False)
                    print(f"Empty results file created at {output_file}")
                else:
                    # Check if EXRET column exists - calculate it if not
                    if 'EXRET' not in sell_candidates.columns:
                        # Calculate EXRET if we have the necessary columns
                        if ('upside' in sell_candidates.columns and 
                            'buy_percentage' in sell_candidates.columns and
                            pd.api.types.is_numeric_dtype(sell_candidates['upside']) and
                            pd.api.types.is_numeric_dtype(sell_candidates['buy_percentage'])):
                            sell_candidates['EXRET'] = sell_candidates['upside'] * sell_candidates['buy_percentage'] / 100
                        else:
                            sell_candidates['EXRET'] = None
                    
                    # Convert company name to uppercase and truncate to 20 characters
                    sell_candidates['company'] = sell_candidates['company'].apply(
                        lambda x: str(x).upper()[:20]
                    )
                    
                    # Select and rename columns for display, including all requested columns
                    columns_to_select = [
                        'ticker', 'company', 'price', 'target_price', 'upside', 'analyst_count',
                        'buy_percentage', 'total_ratings', 'A', 'EXRET', 'beta',
                        'pe_trailing', 'pe_forward', 'peg_ratio', 'dividend_yield',
                        'short_float_pct', 'last_earnings'
                    ]
                    
                    # Create a dictionary mapping column names
                    column_mapping = {
                        'ticker': 'TICKER',
                        'company': 'COMPANY NAME',
                        'price': 'PRICE',
                        'target_price': 'TARGET',
                        'upside': 'UPSIDE',
                        'analyst_count': '# T',
                        'buy_percentage': BUY_PERCENTAGE,
                        'total_ratings': '# A',
                        'A': 'A',
                        'EXRET': 'EXRET',
                        'beta': 'BETA',
                        'pe_trailing': 'PET',
                        'pe_forward': 'PEF',
                        'peg_ratio': 'PEG',
                        'dividend_yield': DIVIDEND_YIELD,
                        'short_float_pct': 'SI',
                        'last_earnings': 'EARNINGS'
                    }
                    
                    # Select only columns that exist in the dataframe
                    available_columns = [col for col in columns_to_select if col in sell_candidates.columns]
                    display_df = sell_candidates[available_columns].copy()
                    
                    # Rename columns according to mapping
                    display_df.rename(columns={col: column_mapping[col] for col in available_columns if col in column_mapping}, inplace=True)
                    
                    # Sort by EXRET (ascending, worst first)
                    if 'EXRET' in display_df.columns:
                        display_df = display_df.sort_values('EXRET', ascending=True)
                    
                    # Format numeric columns with proper decimal places
                    price_columns = ['PRICE', 'TARGET', 'BETA', 'PET', 'PEF', 'PEG']
                    for col in price_columns:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:.2f}" if pd.notnull(x) else "--"
                            )
                    
                    # Format percentage columns
                    percentage_columns = ['UPSIDE', BUY_PERCENTAGE, 'EXRET', DIVIDEND_YIELD, 'SI']
                    for col in percentage_columns:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:.1f}%" if pd.notnull(x) else "--"
                            )
                    
                    # Format date columns
                    if 'EARNINGS' in display_df.columns:
                        def format_date(date_str):
                            if pd.notnull(date_str) and date_str != '--':
                                try:
                                    return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                                except ValueError:
                                    return date_str
                            return '--'
                        
                        display_df['EARNINGS'] = display_df['EARNINGS'].apply(format_date)
                    
                    # Define column alignment (left-align TICKER and COMPANY NAME, right-align others)
                    colalign = []
                    for col in display_df.columns:
                        if col in ['TICKER', 'COMPANY NAME']:
                            colalign.append('left')
                        else:
                            colalign.append('right')
                    
                    # Display results
                    print("\nSell Candidates in Your Portfolio:")
                    print(tabulate(
                        display_df,
                        headers='keys',
                        tablefmt='fancy_grid',
                        showindex=False,
                        colalign=colalign
                    ))
                    print(f"\nTotal sell candidates: {len(display_df)}")
                    
                    # Save to CSV
                    output_file = os.path.join(output_dir, "sell.csv")
                    display_df.to_csv(output_file, index=False)
                    print(f"Results saved to {output_file}")
            else:
                print(f"\nPortfolio analysis file not found: {portfolio_output_path}")
                print("Please run the portfolio analysis (P) first to generate sell recommendations.")
    
    except Exception as e:
        logger.error(f"Error generating trade recommendations: {str(e)}")
        print(f"Error generating recommendations: {str(e)}")

def main():
    """Command line interface for market display"""
    try:
        display = MarketDisplay()
        source = input("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ").strip().upper()
        
        if source == 'T':
            action = input("Do you want to identify BUY (B) or SELL (S) opportunities? ").strip().upper()
            if action == 'B':
                generate_trade_recommendations('N')  # 'N' for new buy opportunities
            elif action == 'S':
                generate_trade_recommendations('E')  # 'E' for existing portfolio (sell)
            else:
                print("Invalid option. Please enter 'B' or 'S'.")
            return
        
        if source == 'P':
            use_existing = input("Use existing portfolio file (E) or download new one (N)? ").strip().upper()
            if use_existing == 'N':
                from yahoofinance.download import download_portfolio
                if not download_portfolio():
                    logger.error("Failed to download portfolio")
                    return
                
        tickers = display.load_tickers(source)
        
        if not tickers:
            logger.error("No valid tickers provided")
            return
            
        try:
            # Handle special case for eToro market
            report_source = source
            if source == 'E':
                report_source = 'M'  # Still save as market.csv for eToro tickers
                
            # Only pass source for market or portfolio options
            display.display_report(tickers, report_source if report_source in ['M', 'P'] else None)
        except ValueError as e:
            logger.error(f"Error processing numeric values: {str(e)}")
        except Exception as e:
            logger.error(f"Error displaying report: {str(e)}")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
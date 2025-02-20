#!/usr/bin/env python3
"""
Command-line interface for the market analysis tool.
"""

import logging
import sys
from yahoofinance.display import MarketDisplay

# Set up logging with INFO level
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Command line interface for market display"""
    try:
        display = MarketDisplay()
        source = input("Load tickers for Portfolio (P), Market (M) or Manual Input (I)? ").strip().upper()
        
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
            # Only pass source for market or portfolio options
            display.display_report(tickers, source if source in ['M', 'P'] else None)
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
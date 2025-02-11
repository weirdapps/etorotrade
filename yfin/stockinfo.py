import yfinance as yf
import logging

def format_market_cap(market_cap):
    """Format market cap value into a readable string with B/M suffix."""
    if not market_cap:
        return "--"
    
    if market_cap >= 1e9:
        return f"${market_cap/1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.1f}M"
    else:
        return f"${market_cap:.0f}"

def get_stock_info(ticker):
    """
    Fetch market cap and short float info for a stock.
    
    Returns:
        dict: Contains 'market_cap' (formatted string) and 'short_float' (percentage)
              or None if data unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        market_cap = info.get("marketCap")
        short_float = info.get("shortPercentOfFloat")
        forward_eps = info.get("forwardEps")
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        # Try to get PEG ratio from financial data module
        try:
            data = stock.get_info(['financialData'])
            peg_ratio = data.get('pegRatio')
            if peg_ratio is None:
                data = stock.get_info(['defaultKeyStatistics'])
                peg_ratio = data.get('pegRatio')
        except:
            peg_ratio = info.get('pegRatio')
        dividend_yield = info.get("dividendYield")

        if not market_cap:
            return None

        return {
            'market_cap': format_market_cap(market_cap),
            'short_float': short_float * 100 if short_float is not None else None,
            'forward_eps': forward_eps,
            'trailingPE': trailing_pe,
            'forwardPE': forward_pe,
            'pegRatio': peg_ratio,
            'dividendYield': dividend_yield
        }

    except Exception as e:
        logging.error(f"Error getting stock info for {ticker}: {e}")
        logging.exception(e)  # Log detailed exception info
        return None

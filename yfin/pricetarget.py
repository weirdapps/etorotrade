import yfinance as yf

def get_price_target(ticker):
    """
    Fetch analyst price target and calculate change percentage from current price.
    
    Returns:
        dict: Contains 'target' (mean price target) and 'change' (percentage from current price)
              or None if data unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        
        target_price = info.get("targetMeanPrice")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        analyst_count = info.get("numberOfAnalystOpinions")
        
        if not target_price or not current_price:
            return None
            
        change_percent = ((target_price - current_price) / current_price) * 100
            
        return {
            'target': f"${target_price:.2f}", # formatted target for display
            'target_price_raw': target_price, # raw target price for calculations
            'change': change_percent,
            'number_of_analysts': analyst_count
        }
        
    except Exception as e:
        logging.error(f"Error getting price target for {ticker}: {e}")
        logging.exception(e) # Log detailed exception info
        return None
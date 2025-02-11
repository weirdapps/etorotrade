import yfinance as yf

def get_price_info(ticker):
    """
    Fetch the current price and daily change percentage for a stock.
    
    Returns:
        dict: Contains 'current_price' and 'day_change' percentage
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="2d")  # Get 2 days to calculate change
        
        if data.empty:
            return None
            
        current_price = data['Close'].iloc[-1]
        
        # Calculate daily change percentage
        if len(data) >= 2:
            prev_close = data['Close'].iloc[-2]
            day_change = ((current_price - prev_close) / prev_close) * 100
        else:
            # If we only have one day of data, use the day's own open/close
            day_change = ((current_price - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100
            
        return {
            'current_price': current_price,
            'day_change': day_change
        }
        
    except Exception as e:
        print(f"Error getting price info for {ticker}: {e}")
        return None
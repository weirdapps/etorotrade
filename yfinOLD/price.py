import yfinance as yf

def get_current_price(ticker):
    """Fetch the latest closing price of a stock."""
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")

    return data['Close'].iloc[-1] if not data.empty else None
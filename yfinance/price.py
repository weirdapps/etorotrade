## price.py
import yfinance as yf

def get_current_price(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period="1d")
    return data['Close'].iloc[-1] if not data.empty else None
import yfinback as yf

try:
    ticker = yf.Ticker("AAPL")
    print("yfinance.Ticker imported successfully")
except Exception as e:
    print(f"Error importing yfinance.Ticker: {e}")
import yfinance as yf  # Correct import statement

def get_current_price(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)  # Fixed: Use "yf" instead of "yfinance"
    data = stock.history(period="1d")
    return data['Close'].iloc[-1] if not data.empty else None

ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").strip().upper()

try:
    price = get_current_price(ticker)
    if price is not None:
        print(f"Current price of {ticker}: ${price:.2f}")
    else:
        print(f"Could not find data for {ticker}. Please check the symbol.")
except Exception as e:
    print(f"Error fetching data: {e}")
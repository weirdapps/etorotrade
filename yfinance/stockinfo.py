import yfinance as yf

def get_stock_info(ticker_symbol):
    # Create a Ticker object
    stock = yf.Ticker(ticker_symbol)
    
    # Fetch the info dictionary
    info = stock.info
    
    return info

# Example usage
if __name__ == "__main__":
    # Prompt user for ticker input
    ticker = input("Enter a stock ticker symbol (e.g., NVDA, AAPL, TSLA): ").strip().upper()
    stock_info = get_stock_info(ticker)
    
    # Print all keys in the info dictionary
    print("\nAll available keys in the info dictionary:")
    print(stock_info.keys())
    
    # Print selected metrics
    print(f"\nSelected metrics for {ticker}")
    print("----------------------------------")
    print(f"Company Name: {stock_info.get('longName', 'N/A')}")
    print(f"Sector: {stock_info.get('sector', 'N/A')}")
    print(f"Market Cap: {stock_info.get('marketCap', 'N/A'):,}")
    print(f"Current Price: {stock_info.get('currentPrice', 'N/A')}")
    print(f"Target Price: {stock_info.get('targetMeanPrice', 'N/A')}")
    print(f"Recommendation Mean: {stock_info.get('recommendationMean', 'N/A')}")
    print(f"Recommendation Key: {stock_info.get('recommendationKey', 'N/A')}")
    print(f"Number of Analysts: {stock_info.get('numberOfAnalystOpinions', 'N/A')}")
    print(f"PE Ratio (Trailing): {stock_info.get('trailingPE', 'N/A')}")
    print(f"PE Ratio (Forward): {stock_info.get('forwardPE', 'N/A')}")
    print(f"PEG Ratio (Trailing): {stock_info.get('trailingPegRatio', 'N/A')}")
    print(f"Short % of Float: {stock_info.get('shortPercentOfFloat', 'N/A')}")
    print(f"Short Ratio: {stock_info.get('shortRatio', 'N/A')}")
    print(f"Stock Beta: {stock_info.get('beta', 'N/A')}")   
    print(f"Dividend Yield: {stock_info.get('dividendYield', 'N/A')}")
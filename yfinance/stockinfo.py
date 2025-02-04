import yfinance as yf

def get_stock_info(ticker):
    """Fetch stock info from Yahoo Finance and return a structured dictionary."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    return {
        "Company Name": info.get("longName", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Market Cap": f"{info.get('marketCap', 'N/A'):,}" if info.get("marketCap") else "N/A",
        "Current Price": info.get("currentPrice", "N/A"),
        "Target Price": info.get("targetMeanPrice", "N/A"),
        "Recommendation Mean": info.get("recommendationMean", "N/A"),
        "Recommendation Key": info.get("recommendationKey", "N/A"),
        "Number of Analysts": info.get("numberOfAnalystOpinions", "N/A"),
        "PE Ratio (Trailing)": info.get("trailingPE", "N/A"),
        "PE Ratio (Forward)": info.get("forwardPE", "N/A"),
        "PEG Ratio (Trailing)": info.get("trailingPegRatio", "N/A"),
        "Quick Ratio": info.get("quickRatio", "N/A"),
        "Current Ratio": info.get("currentRatio", "N/A"),
        "Debt to Equity": info.get("debtToEquity", "N/A"),
        "Short % of Float": info.get("shortPercentOfFloat", "N/A"),
        "Short Ratio": info.get("shortRatio", "N/A"),
        "Stock Beta": info.get("beta", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A")
    }

import yfinance as yf
import pandas as pd

def get_analyst_price_targets(ticker):
    """Fetch analyst price targets from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    return pd.DataFrame({
        "Metric": ["Target Mean Price", "Target High Price", "Target Low Price", "Number of Analysts"],
        "Value": [
            info.get("targetMeanPrice"),
            info.get("targetHighPrice"),
            info.get("targetLowPrice"),
            info.get("numberOfAnalystOpinions")
        ]
    })
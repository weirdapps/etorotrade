## pricetarget.py
import yfinance as yf
import pandas as pd

def get_analyst_price_targets(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    analyst_info = stock.info

    return pd.DataFrame({
        "Metric": ["Target Mean Price", "Target High Price", "Target Low Price", "Number of Analysts"],
        "Value": [analyst_info.get("targetMeanPrice"), analyst_info.get("targetHighPrice"), analyst_info.get("targetLowPrice"), analyst_info.get("numberOfAnalystOpinions")]
    })

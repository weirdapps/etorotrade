import yfinance as yf
import pandas as pd

def get_analyst_price_targets(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    analyst_info = stock.info

    price_targets = {
        "Metric": [
            "Target Mean Price",
            "Target High Price",
            "Target Low Price",
            "Number of Analysts"
        ],
        "Value": [
            analyst_info.get("targetMeanPrice", "N/A"),
            analyst_info.get("targetHighPrice", "N/A"),
            analyst_info.get("targetLowPrice", "N/A"),
            analyst_info.get("numberOfAnalystOpinions", "N/A")
        ]
    }

    df = pd.DataFrame(price_targets)
    return df

# Example Usage
ticker = "MA"
df_price_targets = get_analyst_price_targets(ticker)
print(df_price_targets.to_markdown(index=False))
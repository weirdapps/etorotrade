import yfinance as yf
import pandas as pd
from insiders import analyze_insider_transactions

ticker = "UNH"  # Replace with your stock ticker
transactions = analyze_insider_transactions(ticker)

print(transactions)

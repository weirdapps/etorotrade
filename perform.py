import numpy as np
import pandas as pd
import yfinance as yf

# Load portfolio file
portfolio_df = pd.read_csv('portfolio.csv')  # Adjust the path to your file

# Replace LYXGRE.DE with the Athens Stock Exchange FTSE 20 index ticker
portfolio_df['ticker'] = portfolio_df['ticker'].replace('LYXGRE.DE', 'GD.AT')

# List of tickers in the portfolio
tickers = portfolio_df['ticker'].unique().tolist()

# Define the period for historical data
start_date = '2020-01-01'
end_date = '2024-06-01'

# Fetch historical price data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Fill missing values with the previous valid value
data = data.ffill()

# Filter out tickers with insufficient data
min_data_length = 252  # Minimum number of trading days required (1 year)
valid_tickers = data.columns[data.count() >= min_data_length].tolist()

# If a ticker doesn't have enough data, remove it from the portfolio
portfolio_df = portfolio_df[portfolio_df['ticker'].isin(valid_tickers)]
data = data[valid_tickers]

# Calculate daily returns
daily_returns = data.pct_change(fill_method=None).dropna()

# Calculate mean returns and covariance matrix
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# Adjust portfolio weights
weights = portfolio_df.set_index('ticker').loc[valid_tickers, 'positionValue']
weights = weights / weights.sum()  # Normalize weights to sum to 1

# Ensure the weights align with the valid tickers and mean returns
weights = weights.loc[mean_returns.index]

# Check for NaN or infinite values in weights
weights = weights.replace([np.inf, -np.inf], np.nan).dropna()

# Calculate expected portfolio return
expected_portfolio_return = np.dot(weights, mean_returns) * 252  # Annualize the returns

# Calculate portfolio variance and standard deviation (volatility)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

# Value at Risk (VaR) at 95% confidence level
confidence_level = 0.95
z_score = np.abs(np.percentile(np.random.normal(0, 1, 1000000), 100 * (1 - confidence_level)))

portfolio_VaR = z_score * portfolio_volatility

# Convert results to percentage with 2 decimals
expected_portfolio_return_percent = round(expected_portfolio_return * 100, 2)
portfolio_VaR_percent = round(portfolio_VaR * 100, 2)

# Display the length of each ticker time series and the length used in the calculation
ticker_lengths = data.count()

print(f"Expected Portfolio Return: {expected_portfolio_return_percent}%")
print(f"Portfolio VaR (95% confidence): {portfolio_VaR_percent}%")

# Print the number of tickers used in the calculation
print(f"\nNumber of valid tickers used in the calculation: {len(valid_tickers)}")

# print("\nLength of each ticker time series and the length used in the calculation:")
# print(ticker_lengths)

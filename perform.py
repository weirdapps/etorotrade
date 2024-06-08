import yfinance as yf
import numpy as np
import pandas as pd

# Load your portfolio file
portfolio_df = pd.read_csv('portfolio.csv')

# List of tickers in the portfolio
tickers = portfolio_df['ticker'].unique().tolist()  # Convert to list

# Replace LYXGRE.DE with the Athens Stock Exchange FTSE 20 index ticker
ftse20_ticker = 'GD.AT'  # Correct ticker for the Athens FTSE 20 index
tickers = ['GD.AT' if ticker == 'LYXGRE.DE' else ticker for ticker in tickers]

# Define the period for historical data
start_date = '2023-01-01'
end_date = '2024-06-01'

# Fetch historical price data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change(fill_method=None).dropna()

# Calculate mean returns and covariance matrix
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# Adjust portfolio weights if needed (in this example, assuming it's not needed)
weights = portfolio_df['positionValue'] / portfolio_df['positionValue'].sum()

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

print(f"Expected Portfolio Return: {expected_portfolio_return_percent}%")
print(f"Portfolio VaR (95% confidence): {portfolio_VaR_percent}%")
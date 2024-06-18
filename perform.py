import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate

def load_portfolio(file_path):
    portfolio_df = pd.read_csv(file_path)
    portfolio_df['ticker'] = portfolio_df['ticker'].replace('LYXGRE.DE', 'GD.AT')
    return portfolio_df

def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.ffill()
    return data

def filter_data(data, portfolio_df, min_data_length):
    valid_tickers = data.columns[data.count() >= min_data_length].tolist()
    portfolio_df = portfolio_df[portfolio_df['ticker'].isin(valid_tickers)]
    data = data[valid_tickers]
    return data, portfolio_df

def calculate_returns(data):
    daily_returns = data.pct_change(fill_method=None).dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    return mean_returns, cov_matrix

def adjust_weights(portfolio_df, valid_tickers):
    weights = portfolio_df.set_index('ticker').loc[valid_tickers, 'positionValue']
    weights = weights / weights.sum()
    return weights

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, periods):
    weights = weights.replace([np.inf, -np.inf], np.nan).dropna()
    expected_portfolio_return = np.dot(weights, mean_returns) * periods
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * periods, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    confidence_level = 0.95
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 1000000), 100 * (1 - confidence_level)))
    portfolio_VaR = z_score * portfolio_volatility
    sharpe_ratio = expected_portfolio_return / portfolio_volatility
    return round(expected_portfolio_return, 2), round(portfolio_VaR, 2), round(sharpe_ratio, 2)

if __name__ == "__main__":
    portfolio_df = load_portfolio('portfolio.csv')  # Adjust the path to your file
    tickers = portfolio_df['ticker'].unique().tolist()
    start_date = '2020-01-01'
    end_date = '2024-06-17'
    data = fetch_data(tickers, start_date, end_date)
    data, portfolio_df = filter_data(data, portfolio_df, 252)  # Minimum number of trading days required (1 year)
    mean_returns, cov_matrix = calculate_returns(data)
    weights = adjust_weights(portfolio_df, data.columns.tolist())

    periods = [("Daily", 1), ("Weekly", 5), ("Monthly", 22), ("Annual", 252)]
    results = {"ER (%)": [], "VaR (%)": [], "Sharpe": []}
    for period_name, period_value in periods:
        expected_portfolio_return_percent, portfolio_VaR_percent, sharpe_ratio = calculate_portfolio_performance(weights, mean_returns, cov_matrix, period_value)
        results["ER (%)"].append(expected_portfolio_return_percent)
        results["VaR (%)"].append(portfolio_VaR_percent)
        results["Sharpe"].append(sharpe_ratio)

    results_df = pd.DataFrame(results, index=[period_name for period_name, _ in periods])
    print(tabulate(results_df, headers='keys', tablefmt='psql', floatfmt=".2f"))
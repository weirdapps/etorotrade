import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from tabulate import tabulate
from datetime import datetime, timedelta

# Variables for start and end dates
start_date = '2020-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
risk_free_rate = 0.01

# Function to fetch data
def fetch_data(tickers, start, end):
    data = {}
    data_points = {}
    ignored_tickers = []
    
    for ticker in tickers:
        if ticker == 'LYXGRE.DE':
            ticker = 'GD.AT'
        try:
            stock_data = yf.download(ticker, start=start, end=end)
            if stock_data.empty:
                ignored_tickers.append(ticker)
            else:
                data_points[ticker] = stock_data['Adj Close'].notna().sum()
                # Handle missing values by forward-filling, then backward-filling
                stock_data['Adj Close'] = stock_data['Adj Close'].ffill().bfill()
                data[ticker] = stock_data['Adj Close']
        except Exception as e:
            ignored_tickers.append(ticker)
    
    return data, data_points, ignored_tickers

# Function to calculate portfolio metrics
def portfolio_metrics(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252  # Annualize returns
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # Annualize std dev
    return returns, std_dev

# Function to minimize (Sharpe ratio optimization)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_std_dev = portfolio_metrics(weights, mean_returns, cov_matrix)
    return - (p_returns - risk_free_rate) / p_std_dev

# Function to get optimal weights
def get_optimal_weights(data, min_weights, data_points):
    returns = pd.DataFrame(data).pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    max_data_points = max(data_points.values())
    bounds = tuple(
        (min_weights[ticker], min(0.125, 1)) if min_weights[ticker] > 0 or data_points[ticker] >= 0.8 * max_data_points else (0, 0) 
        for ticker in data.keys()
    )
    
    # Improved initial guess: start with min weights and distribute remaining weight evenly
    x0 = np.array([max(1. / num_assets, min_weights[ticker]) for ticker in data.keys()])
    x0 = x0 / np.sum(x0)  # Normalize to ensure they sum to 1
    
    print(f"Initial guess for weights: {x0}")
    print(f"Bounds: {bounds}")
    
    result = minimize(neg_sharpe_ratio, x0, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    print(f"Optimization result: {result}")
    return result.x, mean_returns, cov_matrix

# Function to calculate annual value at risk
def calculate_annual_var(weights, mean_returns, cov_matrix, alpha=0.05):
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    var = portfolio_return - portfolio_std_dev * np.percentile(np.random.randn(10000), (1 - alpha) * 100)
    return var

# Main function
def main():
    # Read portfolio tickers
    portfolio = pd.read_csv('portfolio.csv')
    tickers = portfolio['ticker'].tolist()
    
    # Set minimum weights for each ticker
    min_weights = {
        'AAPL': 0.025,
        'MSFT': 0.125,
        'GOOGL': 0.05,
        'AMZN': 0.10,
        'META': 0.025,
        'ARM': 0.025,
        'GD.AT': 0.125,
        'NVDA': 0.05,
        'QCOM': 0.025,
        'AVGO': 0.025,
        'AMD': 0.025
        # Add more tickers and their minimum weights as needed
    }
    
    # Ensure min_weights contains an entry for each ticker
    for ticker in tickers:
        if ticker not in min_weights:
            min_weights[ticker] = 0.0
    
    # Fetch data
    data, data_points, ignored_tickers = fetch_data(tickers, start_date, end_date)
    
    # Display ignored tickers
    if ignored_tickers:
        print(f"Ignored tickers due to no data: {ignored_tickers}")
    
    # Calculate optimal weights
    if data:
        optimal_weights, mean_returns, cov_matrix = get_optimal_weights(data, min_weights, data_points)
        sorted_data = sorted(data.keys())
        sorted_weights = [optimal_weights[list(data.keys()).index(ticker)] for ticker in sorted_data]
        sorted_data_points = [data_points[ticker] for ticker in sorted_data]
        
        # Filter tickers with weights greater than a small threshold (e.g., 0.0001)
        threshold = 0.0001
        non_zero_weights = [weight for weight in sorted_weights if weight > threshold]
        num_stocks_with_weight = len(non_zero_weights)
        
        # Display table with stock, weight, and number of data points
        weight_df = pd.DataFrame({
            'Stock': sorted_data,
            'Weight': [f"{weight * 100:6.2f}%" for weight in sorted_weights],
            'Data Points': sorted_data_points
        })
        
        # Add summary row
        total_weight = sum(sorted_weights)
        avg_data_points = np.mean(sorted_data_points)
        
        summary_row = pd.DataFrame({
            'Stock': [f"{num_stocks_with_weight} tickers"],
            'Weight': [f"{total_weight * 100:6.2f}%"],
            'Data Points': [f"{avg_data_points:6.2f}"]
        })
        
        weight_df = pd.concat([weight_df, summary_row], ignore_index=True)
        
        print(tabulate(weight_df, headers='keys', tablefmt='fancy_grid', colalign=("right", "left", "right", "right")))
        
        # Calculate portfolio metrics
        annual_return, std_dev = portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
        sharpe_ratio = (annual_return - risk_free_rate) / std_dev
        var = calculate_annual_var(optimal_weights, mean_returns, cov_matrix)
        
        # Display portfolio metrics
        print(f"Estimated Annual Return: {100 * annual_return:.2f}%")
        print(f"Annual Value at Risk (VaR): {100 * var:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Run the main function
if __name__ == "__main__":
    main()
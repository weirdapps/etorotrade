import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ta.trend import MACD

# Set the ticker symbol and timeframe
ticker_symbol = 'SPY'
start_date = '2010-01-01'
end_date = '2023-06-03'

# Fetch historical stock data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Calculate MACD
macd = MACD(data['Close'])
data['MACD'] = macd.macd()
data['Signal'] = macd.macd_signal()

# Create signals based on MACD crossover
data['Signal'] = np.where(data['MACD'] > data['Signal'], 1, -1)

# Calculate daily returns
data['Return'] = data['Close'].pct_change()

# Calculate strategy returns
data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']

# Calculate cumulative returns
data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

# Fetch benchmark (SPY) data
benchmark_data = yf.download('SPY', start=start_date, end=end_date)

# Calculate benchmark returns
benchmark_data['Benchmark_Return'] = benchmark_data['Close'].pct_change().fillna(0)

# Calculate cumulative benchmark returns
benchmark_data['Benchmark_Cumulative_Return'] = (
    1 + benchmark_data['Benchmark_Return']).cumprod()

# Merge strategy and benchmark data
merged_data = pd.merge(
    data, benchmark_data['Benchmark_Cumulative_Return'], left_index=True, right_index=True, how='outer')

# Print performance metrics
strategy_returns = data['Strategy_Return'].dropna()
benchmark_returns = benchmark_data['Benchmark_Return'].dropna()

strategy_cumulative_return = data['Cumulative_Return'].iloc[-1]
benchmark_cumulative_return = benchmark_data['Benchmark_Cumulative_Return'].iloc[-1]

strategy_annual_return = (strategy_cumulative_return **
                          (252 / len(data)) - 1) * 100
benchmark_annual_return = (
    benchmark_cumulative_return ** (252 / len(data)) - 1) * 100

strategy_std_dev = strategy_returns.std() * np.sqrt(252) * 100
benchmark_std_dev = benchmark_returns.std() * np.sqrt(252) * 100

strategy_sharpe_ratio = (strategy_annual_return - 2) / \
    strategy_std_dev  # Assuming risk-free rate of 2%
benchmark_sharpe_ratio = (benchmark_annual_return - 2) / benchmark_std_dev

print('Strategy Performance:')
print(f'Cumulative Return: {strategy_cumulative_return:.2f}')
print(f'Annual Return: {strategy_annual_return:.2f}%')
print(f'Standard Deviation: {strategy_std_dev:.2f}%')
print(f'Sharpe Ratio: {strategy_sharpe_ratio:.2f}')

print('Benchmark Performance (SPY):')
print(f'Cumulative Return: {benchmark_cumulative_return:.2f}')
print(f'Annual Return: {benchmark_annual_return:.2f}%')
print(f'Standard Deviation: {benchmark_std_dev:.2f}%')
print(f'Sharpe Ratio: {benchmark_sharpe_ratio:.2f}')

# Plot equity value
plt.plot(data.index, data['Cumulative_Return'], label='Strategy')
plt.plot(benchmark_data.index,
         benchmark_data['Benchmark_Cumulative_Return'], label='Benchmark (SPY)')
plt.xlabel('Date')
plt.ylabel('Equity Value')
plt.title('Equity Value Comparison')
plt.legend()
plt.show()

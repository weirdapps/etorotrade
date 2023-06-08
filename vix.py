import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download historical data for VIX and SPY
vix = yf.download("^VIX", start="1995-01-01", end="2023-06-04")
spy = yf.download("SPY", start="1995-01-01", end="2023-06-04")

# Extract the 'Close' prices
vix_close = vix['Close']
spy_close = spy['Close']

# Combine the data into a single DataFrame
data = pd.concat([vix_close, spy_close], axis=1)
data.columns = ['VIX_Close', 'SPY_Close']

# Remove any rows with missing data
data.dropna(inplace=True)

# Calculate the rolling 30-day correlation
rolling_corr = data['VIX_Close'].rolling(window=30).corr(data['SPY_Close'])

# Apply trading strategy
trades = []
holding_spy = False
trade_size = 100000
equity = trade_size
total_trades = 0
trade_return = 0.0
benchmark_returns = 0.0
days_in_market = 0

past_vix_values = []
past_days = 5

for i in range(len(data)):
    vix_close_value = data['VIX_Close'].iloc[i]
    spy_close_value = data['SPY_Close'].iloc[i]
    rolling_corr_value = rolling_corr.iloc[i]

    # Count how many days we are in the market
    if holding_spy:
        days_in_market += 1

    # Buy condition
    if not holding_spy:
        if all(value > 30 for value in past_vix_values) and vix_close_value < 30 and rolling_corr_value < -0.75:
            trades.append(('Buy', data.index[i], spy_close_value, trade_size))
            holding_spy = True
            total_trades += 1

    # Sell condition
    elif all(value < 20 for value in past_vix_values) and vix_close_value > 20:
        trades.append(('Sell', data.index[i], spy_close_value, trade_size))
        holding_spy = False
        total_trades += 1

    # Update past VIX values
    past_vix_values.append(vix_close_value)
    if len(past_vix_values) > past_days:
        past_vix_values.pop(0)


# Count total days in the backtest
total_days = len(data)

# Calculate the total return of the benchmark SPY
benchmark_returns = (data['SPY_Close'].iloc[-1] -
                     data['SPY_Close'].iloc[0]) / data['SPY_Close'].iloc[0]

# Calculate the daily returns of the benchmark
spy_daily_returns = benchmark_returns / total_days

# Calculate the Sharpe ratio for the benchmark
benchmark_sharpe = (benchmark_returns - 0.00) / \
    (data['SPY_Close'].pct_change().std() * np.sqrt(252))

# Calculate the final equity for the benchmark
final_equity_benchmark = (1 + benchmark_returns) * trade_size

# Calculate the equity value of the strategy
for trade in trades:
    trade_type, trade_date, spy_close_value, trade_size = trade
    if trade_type == 'Buy':
        spy_buy_volume = equity / spy_close_value
        spy_buy_value = spy_close_value * spy_buy_volume
    elif trade_type == 'Sell':
        spy_sell_volume = spy_buy_volume
        spy_sell_value = spy_close_value * spy_sell_volume
        trade_return = spy_sell_value - spy_buy_value
        equity += trade_return

# Calculate the total return of the strategy
total_return_pct = equity / trade_size - 1

# Calculate the daily returns of the strategy
strategy_daily_returns = total_return_pct / days_in_market

# Calculate the Sharpe ratio for the strategy (replace SPY change with strategy change)
strategy_sharpe = (total_return_pct - 0.00) / \
    (data['SPY_Close'].pct_change().std() * np.sqrt(252))

# Calculate the final equity for the strategy
final_equity_strategy = (1 + total_return_pct) * trade_size


# Print the strategy statistics
print()
print("Trading SPY using VIX signals Statistics:")
print("------------------------------------")
print("Total trades:", total_trades)
print("Days in market:", days_in_market)
print("Total return:", f"{total_return_pct:.0%}")
print("Daily return:", f"{strategy_daily_returns:.2%}")
print("Sharpe ratio (strategy):", f"{strategy_sharpe:.2f}")
print("Final equity:", f"${final_equity_strategy:.0f}")
print()

# Print the benchmark statistics
print("Buy and Hold SPY Benchmark Statistics:")
print("------------------------------------")
print("Total days:", total_days)
print("Total return:", f"{benchmark_returns:.0%}")
print("Daily return:", f"{spy_daily_returns:.2%}")
print("Sharpe ratio (benchmark):", f"{benchmark_sharpe:.2f}")
print("Final equity:", f"${final_equity_benchmark:.0f}")
print()

# Create subplots with different heights for SPY, VIX, and Correlation
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9),
                                    sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

# Plot SPY close price
ax1.set_ylabel('SPY Close Price', color='tab:blue')
ax1.plot(data.index, data['SPY_Close'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Add buy and sell signals to SPY chart
for trade in trades:
    trade_type, trade_date, spy_close_value, _ = trade
    if trade_type == 'Buy':
        ax1.scatter(trade_date, spy_close_value, marker='^', color='g', s=200)
        ax1.annotate(f'Buy: {spy_close_value:.0f}', (trade_date, spy_close_value),
                     xytext=(-20, -20), textcoords='offset points', color='g',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
    elif trade_type == 'Sell':
        ax1.scatter(trade_date, spy_close_value, marker='v', color='r', s=200)
        ax1.annotate(f'Sell: {spy_close_value:.0f}', (trade_date, spy_close_value),
                     xytext=(-20, 10), textcoords='offset points', color='r',
                     bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

# Plot VIX close price
ax2.set_ylabel('VIX Close Price', color='tab:red')
ax2.plot(data.index, data['VIX_Close'], color='tab:red')
ax2.axhline(y=15, color='r', linestyle='--', alpha=0.5)
ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
ax2.fill_between(data.index, vix_close, 15,
                 where=(vix_close < 15), color='r', alpha=0.1)
ax2.fill_between(data.index, vix_close, 30, where=(
    vix_close > 30), color='g', alpha=0.1)
ax2.tick_params(axis='y', labelcolor='tab:red')

# Plot rolling correlation
ax3.set_xlabel('Date')
ax3.set_ylabel('Rolling Correlation', color='tab:orange')
ax3.plot(data.index, rolling_corr, color='tab:orange')
ax3.axhline(y=-0.75, color='r', linestyle='--', alpha=0.5)
ax3.axhline(y=-0.25, color='g', linestyle='--', alpha=0.5)
ax3.fill_between(data.index, rolling_corr, -0.75,
                 where=(rolling_corr < -0.75), color='r', alpha=0.1)
ax3.fill_between(data.index, rolling_corr, -0.25, where=(
    rolling_corr > -0.25), color='g', alpha=0.1)
ax3.tick_params(axis='y', labelcolor='tab:orange')

# Set the plot title
fig.suptitle('Trading SPY using VIX signals', fontsize=16, fontweight='bold')

fig.tight_layout()
plt.show()

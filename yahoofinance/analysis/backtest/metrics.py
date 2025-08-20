"""
Backtest Performance Metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class BacktestMetrics:
    """Calculate backtest performance metrics"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from prices"""
        return prices.pct_change()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + equity_curve).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """Calculate win rate from trades"""
        if len(trades) == 0:
            return 0.0
        winning_trades = trades[trades['pnl'] > 0]
        return len(winning_trades) / len(trades)

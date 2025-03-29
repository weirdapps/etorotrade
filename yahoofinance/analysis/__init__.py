"""
Analysis modules for Yahoo Finance data.

This package contains modules for analyzing financial data including:
- Analyst data: Analyst ratings and price targets
- Earnings: Earnings dates and surprises
- Insiders: Insider transactions
- Market: Market-wide analysis and sector analysis
- Metrics: Financial metrics and ratios
- Portfolio: Portfolio analysis and tracking
- Stock: Comprehensive stock analysis with trading recommendations
- Backtest: Backtesting and optimization of trading strategies
"""

from .analyst import AnalystData
from .earnings import EarningsAnalyzer
from .insiders import InsiderAnalyzer
from .market import MarketAnalyzer, MarketMetrics, SectorAnalysis
from .metrics import PricingAnalyzer, PriceTarget, PriceData
from .stock import StockAnalyzer, AnalysisResults
from .portfolio import PortfolioAnalyzer, PortfolioHolding, PortfolioSummary
from .backtest import (
    BacktestSettings, BacktestResult, BacktestPosition,
    Backtester, BacktestOptimizer, run_backtest, optimize_criteria
)

__all__ = [
    # Analyst
    'AnalystData',
    
    # Earnings
    'EarningsAnalyzer',
    
    # Insiders
    'InsiderAnalyzer',
    
    # Market
    'MarketAnalyzer',
    'MarketMetrics',
    'SectorAnalysis',
    
    # Metrics
    'PricingAnalyzer',
    'PriceTarget',
    'PriceData',
    
    # Portfolio
    'PortfolioAnalyzer',
    'PortfolioHolding',
    'PortfolioSummary',
    
    # Stock Analysis
    'StockAnalyzer',
    'AnalysisResults',
    
    # Backtesting
    'BacktestSettings',
    'BacktestResult',
    'BacktestPosition',
    'Backtester',
    'BacktestOptimizer',
    'run_backtest',
    'optimize_criteria',
]
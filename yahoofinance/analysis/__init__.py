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
- Optimize: Portfolio optimization using Modern Portfolio Theory
"""

from .analyst import AnalystData, CompatAnalystData
from .earnings import EarningsAnalyzer, EarningsCalendar, format_earnings_table
from .insiders import InsiderAnalyzer
from .market import MarketAnalyzer, MarketMetrics, SectorAnalysis
from .metrics import PricingAnalyzer, PriceTarget, PriceData
from .stock import StockAnalyzer, AnalysisResults
from .portfolio import PortfolioAnalyzer, PortfolioHolding, PortfolioSummary
from .backtest import (
    BacktestSettings, BacktestResult, BacktestPosition,
    Backtester, BacktestOptimizer, run_backtest, optimize_criteria
)
from .optimize import PortfolioOptimizer, optimize_portfolio

__all__ = [
    # Analyst
    'AnalystData',
    'CompatAnalystData',  # For compatibility
    
    # Earnings
    'EarningsAnalyzer',
    'EarningsCalendar',   # For compatibility
    'format_earnings_table',  # For compatibility
    
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
    
    # Portfolio Optimization
    'PortfolioOptimizer',
    'optimize_portfolio',
]
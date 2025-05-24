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

This package also provides factory functions for creating and managing analyzers
with dependency injection, making it easier to test and maintain the codebase.
"""

# Direct imports of analysis classes
from .analyst import AnalystData, CompatAnalystData

# Import factory functions for dependency injection
from .analyzer_factory import (
    create_portfolio_analyzer,
    create_stock_analyzer,
    with_analyzer,
    with_portfolio_analyzer,
)
from .backtest import (
    Backtester,
    BacktestOptimizer,
    BacktestPosition,
    BacktestResult,
    BacktestSettings,
    optimize_criteria,
    run_backtest,
)
from .earnings import EarningsAnalyzer, EarningsCalendar, format_earnings_table
from .insiders import InsiderAnalyzer
from .market import MarketAnalyzer, MarketMetrics, SectorAnalysis
from .metrics import PriceData, PriceTarget, PricingAnalyzer
from .optimize import PortfolioOptimizer, optimize_portfolio
from .portfolio import PortfolioAnalyzer, PortfolioHolding, PortfolioSummary
from .stock import AnalysisResults, StockAnalyzer


__all__ = [
    # Analyst
    "AnalystData",
    "CompatAnalystData",  # For compatibility
    # Earnings
    "EarningsAnalyzer",
    "EarningsCalendar",  # For compatibility
    "format_earnings_table",  # For compatibility
    # Insiders
    "InsiderAnalyzer",
    # Market
    "MarketAnalyzer",
    "MarketMetrics",
    "SectorAnalysis",
    # Metrics
    "PricingAnalyzer",
    "PriceTarget",
    "PriceData",
    # Portfolio
    "PortfolioAnalyzer",
    "PortfolioHolding",
    "PortfolioSummary",
    # Stock Analysis
    "StockAnalyzer",
    "AnalysisResults",
    # Backtesting
    "BacktestSettings",
    "BacktestResult",
    "BacktestPosition",
    "Backtester",
    "BacktestOptimizer",
    "run_backtest",
    "optimize_criteria",
    # Portfolio Optimization
    "PortfolioOptimizer",
    "optimize_portfolio",
    # Factory Functions (Dependency Injection)
    "create_stock_analyzer",
    "with_analyzer",
    "create_portfolio_analyzer",
    "with_portfolio_analyzer",
]

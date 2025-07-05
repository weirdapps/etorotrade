"""
Trade module - Breaking down the monolithic trade.py into logical components.

This module contains:
- engine: Main orchestration logic
- data: DataFrame operations and data processing
- criteria: Trading criteria logic and calculations
- files: File I/O operations
- portfolio: Portfolio-specific operations
- reports: Report generation and formatting
"""

from yahoofinance.trade.engine import TradeEngine

__all__ = ['TradeEngine']
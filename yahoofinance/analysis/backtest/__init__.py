"""
Backtest module - Split into components for better maintainability
"""

from .engine import BacktestEngine, BacktestConfig
from .strategies import *
from .metrics import BacktestMetrics
from .reporting import BacktestReporter

# Compatibility aliases for existing code
BacktestSettings = BacktestConfig
Backtester = BacktestEngine
BacktestResult = dict  # Simplified for now
BacktestPosition = dict  # Simplified for now
BacktestOptimizer = BacktestEngine  # Use same engine for now

# Compatibility functions
def optimize_criteria(*args, **kwargs):
    """Legacy function for backward compatibility"""
    return {}

def run_backtest(data, strategy=None, config=None):
    """Legacy backtest function for compatibility"""
    engine = BacktestEngine(config)
    return engine.run(data, strategy)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestMetrics', 
    'BacktestReporter',
    'Backtester',
    'BacktestSettings',
    'BacktestResult',
    'BacktestPosition',
    'BacktestOptimizer',
    'optimize_criteria',
    'run_backtest',
]

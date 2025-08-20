"""
Backtest module - Compatibility wrapper
This file maintains backward compatibility while using the new modular structure
"""

# Import all components from the new structure
from .backtest.engine import BacktestEngine, BacktestConfig
from .backtest.strategies import *
from .backtest.metrics import BacktestMetrics
from .backtest.reporting import BacktestReporter

# Re-export for backward compatibility
__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestMetrics',
    'BacktestReporter',
    'MomentumStrategy',
    'MeanReversionStrategy',
]

# Compatibility function if needed
def run_backtest(data, strategy=None, config=None):
    """Legacy backtest function for compatibility"""
    engine = BacktestEngine(config)
    return engine.run(data, strategy)

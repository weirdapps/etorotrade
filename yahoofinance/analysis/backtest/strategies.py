"""
Trading Strategies for Backtesting
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        pass

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['signal'][self.lookback:] = np.where(
            data['returns'][self.lookback:] > 0, 1, -1
        )
        return signals

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, window: int = 20, num_std: float = 2):
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        # Simplified implementation
        return signals

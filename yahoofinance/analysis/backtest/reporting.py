"""
Backtest Reporting and Visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json

class BacktestReporter:
    """Generate backtest reports"""
    
    def __init__(self, results: Dict):
        self.results = results
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics"""
        return {
            'total_return': self.results.get('total_return', 0),
            'sharpe_ratio': self.results.get('sharpe_ratio', 0),
            'max_drawdown': self.results.get('max_drawdown', 0),
            'win_rate': self.results.get('win_rate', 0),
        }
    
    def export_to_json(self, filepath: str):
        """Export results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.generate_summary(), f, indent=2)
    
    def export_to_csv(self, filepath: str):
        """Export results to CSV"""
        df = pd.DataFrame([self.generate_summary()])
        df.to_csv(filepath, index=False)

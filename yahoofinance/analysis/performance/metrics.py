"""Performance metrics calculations"""
import numpy as np

class PerformanceMetrics:
    @staticmethod
    def calculate_cagr(returns):
        return ((1 + returns).prod() ** (252/len(returns))) - 1

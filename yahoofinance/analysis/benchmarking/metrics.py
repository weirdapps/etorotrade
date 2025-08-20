"""Benchmark metrics calculation"""
import numpy as np

class BenchmarkMetrics:
    @staticmethod
    def calculate_alpha(returns, benchmark_returns):
        return returns.mean() - benchmark_returns.mean()

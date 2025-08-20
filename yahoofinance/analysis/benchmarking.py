"""Benchmarking - Compatibility wrapper"""
from .benchmarking.engine import BenchmarkEngine
from .benchmarking.metrics import BenchmarkMetrics
from .benchmarking.comparisons import BenchmarkComparator

__all__ = ['BenchmarkEngine', 'BenchmarkMetrics', 'BenchmarkComparator']

"""
Performance benchmarking module for etorotrade.

This module provides tools for measuring and tracking performance metrics
across different components of the system. It enables consistent benchmarking,
performance regression testing, and optimization validation.
"""

import asyncio
import cProfile
import gc
import io
import json
import os
import pstats
import statistics
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import psutil


# Define constants for repeated strings
TOP_MEMORY_CONSUMERS_HEADER = "\nTop memory consumers:"
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path

from ..api import get_provider
from ..core.config import PERFORMANCE_CONFIG
from ..core.logging import get_logger
from ..utils.market.ticker_utils import is_us_ticker


logger = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")

# Default benchmark configuration if not defined in config
DEFAULT_BENCHMARK_CONFIG = {
    "BENCHMARK_DIR": os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks"),
    "SAMPLE_TICKERS": ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"],
    "BASELINE_FILE": "baseline_performance.json",
    "MEMORY_PROFILE_THRESHOLD": 1.2,  # 20% increase is concerning
    "RESOURCE_MONITOR_INTERVAL": 0.5,  # seconds
    "MAX_BENCHMARK_DURATION": 300,  # 5 minutes max for any benchmark
    "DEFAULT_ITERATIONS": 3,
    "DEFAULT_WARMUP_ITERATIONS": 1,
}

# Get benchmark config or use defaults
BENCHMARK_SETTINGS = (
    PERFORMANCE_CONFIG.get("BENCHMARK", DEFAULT_BENCHMARK_CONFIG)
    if hasattr(PERFORMANCE_CONFIG, "BENCHMARK")
    else DEFAULT_BENCHMARK_CONFIG
)


class PerformanceBenchmark:
    """
    Performance benchmarking utility for measuring execution time and resource usage.

    This class provides methods for measuring execution time, memory usage, and
    other performance metrics. It can be used to benchmark both synchronous and
    asynchronous code, and provides detailed reports on performance.

    Attributes:
        name: Name of the benchmark
        results: Dictionary of benchmark results
        baseline: Optional baseline to compare against
    """

    def __init__(self, name: str, baseline: Optional[Dict[str, Any]] = None):
        """
        Initialize a new performance benchmark.

        Args:
            name: Name of the benchmark
            baseline: Optional baseline metrics to compare against
        """
        self.name = name
        self.results: Dict[str, Any] = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "iterations": [],
            "memory": {},
            "system_info": self._get_system_info(),
        }
        self.baseline = baseline
        self._start_time = 0
        self._end_time = 0
        self._memory_start = 0
        self._memory_end = 0
        self._cpu_percent_start = 0
        self._cpu_percent_end = 0

        # Initialize tracemalloc for memory profiling if enabled
        if PERFORMANCE_CONFIG.get("ENABLE_MEMORY_PROFILING", False):
            tracemalloc.start()

    def start(self) -> None:
        """Start the benchmark timing and resource monitoring."""
        # Force garbage collection to ensure consistent measurements
        gc.collect()

        # Start timing
        self._start_time = time.time()

        # Record memory usage at start
        process = psutil.Process(os.getpid())
        self._memory_start = process.memory_info().rss

        # Record CPU usage at start
        self._cpu_percent_start = process.cpu_percent()

        # Record tracemalloc snapshot if enabled
        if PERFORMANCE_CONFIG.get("ENABLE_MEMORY_PROFILING", False):
            self._tracemalloc_snapshot_start = tracemalloc.take_snapshot()

    def stop(self) -> Dict[str, Any]:
        """
        Stop the benchmark and calculate metrics.

        Returns:
            Dictionary of benchmark results
        """
        # Record end time
        self._end_time = time.time()

        # Record memory usage at end
        process = psutil.Process(os.getpid())
        self._memory_end = process.memory_info().rss

        # Record CPU usage at end
        self._cpu_percent_end = process.cpu_percent()

        # Calculate metrics
        elapsed_time = self._end_time - self._start_time
        memory_used = self._memory_end - self._memory_start
        cpu_percent = self._cpu_percent_end - self._cpu_percent_start

        # Store metrics
        self.results["metrics"] = {
            "elapsed_time": elapsed_time,
            "elapsed_time_ms": elapsed_time * 1000,
            "memory_used": memory_used,
            "memory_used_mb": memory_used / (1024 * 1024),
            "cpu_percent": cpu_percent,
        }

        # Add memory profiling info if enabled
        if PERFORMANCE_CONFIG.get("ENABLE_MEMORY_PROFILING", False) and hasattr(
            self, "_tracemalloc_snapshot_start"
        ):
            snapshot_end = tracemalloc.take_snapshot()
            top_stats = snapshot_end.compare_to(self._tracemalloc_snapshot_start, "lineno")

            memory_stats = []
            for stat in top_stats[:10]:  # Top 10 allocations
                memory_stats.append(
                    {
                        "file": str(stat.traceback.frame.filename),
                        "line": stat.traceback.frame.lineno,
                        "size": stat.size,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                    }
                )

            self.results["memory"]["top_allocations"] = memory_stats

        # Compare to baseline if available
        if self.baseline and "metrics" in self.baseline:
            baseline_time = self.baseline["metrics"].get("elapsed_time", 0)
            baseline_memory = self.baseline["metrics"].get("memory_used", 0)

            time_diff = elapsed_time - baseline_time
            time_percent = (time_diff / baseline_time) * 100 if baseline_time > 0 else 0

            memory_diff = memory_used - baseline_memory
            memory_percent = (memory_diff / baseline_memory) * 100 if baseline_memory > 0 else 0

            self.results["comparison"] = {
                "time_diff": time_diff,
                "time_percent": time_percent,
                "memory_diff": memory_diff,
                "memory_percent": memory_percent,
                "is_improvement": time_percent < 0 and memory_percent < 0,
            }

        return self.results

    def record_iteration(self, iteration_name: str, metrics: Dict[str, Any]) -> None:
        """
        Record metrics for a single iteration of the benchmark.

        Args:
            iteration_name: Name of the iteration
            metrics: Dictionary of metrics for this iteration
        """
        self.results["iterations"].append(
            {
                "name": iteration_name,
                "metrics": metrics,
            }
        )

    def save_results(self, output_dir: Optional[str] = None) -> str:
        """
        Save benchmark results to a JSON file.

        Args:
            output_dir: Directory to save results to (default: PERFORMANCE_CONFIG["RESULTS_DIR"])

        Returns:
            Path to the saved results file
        """
        if output_dir is None:
            output_dir = PERFORMANCE_CONFIG.get("RESULTS_DIR", "performance_results")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Save results to file
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Saved benchmark results to {filepath}")
        return filepath

    def print_summary(self) -> None:
        """Print a summary of the benchmark results to the console."""
        if not self.results.get("metrics"):
            logger.warning("No benchmark results available. Run benchmark first.")
            return

        metrics = self.results["metrics"]

        print(f"\n=== Benchmark: {self.name} ===")
        print(f"Time: {metrics['elapsed_time_ms']:.2f} ms")
        print(f"Memory: {metrics['memory_used_mb']:.2f} MB")
        print(f"CPU: {metrics['cpu_percent']:.2f}%")

        if "comparison" in self.results:
            comp = self.results["comparison"]
            time_change = "improved" if comp["time_percent"] < 0 else "regressed"
            memory_change = "improved" if comp["memory_percent"] < 0 else "regressed"

            print("\n=== Comparison to Baseline ===")
            print(f"Time: {abs(comp['time_percent']):.2f}% {time_change}")
            print(f"Memory: {abs(comp['memory_percent']):.2f}% {memory_change}")

        if self.results["iterations"]:
            print("\n=== Iterations ===")
            for i, iteration in enumerate(self.results["iterations"]):
                print(
                    f"[{i+1}] {iteration['name']}: {iteration['metrics'].get('elapsed_time_ms', 0):.2f} ms"
                )

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for benchmark context.

        Returns:
            Dictionary of system information
        """
        return {
            "platform": os.name,
            "python_version": os.sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }

    @staticmethod
    def load_baseline(name: str, baseline_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load a baseline result for comparison.

        Args:
            name: Name of the baseline benchmark
            baseline_dir: Directory containing baseline results

        Returns:
            Baseline results as a dictionary, or None if not found
        """
        if baseline_dir is None:
            baseline_dir = PERFORMANCE_CONFIG.get("BASELINE_DIR", "performance_baselines")

        baseline_path = os.path.join(baseline_dir, f"{name}_baseline.json")

        if not os.path.exists(baseline_path):
            logger.warning(f"Baseline not found: {baseline_path}")
            return None

        try:
            with open(baseline_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
            return None

    @staticmethod
    def save_as_baseline(results: Dict[str, Any], baseline_dir: Optional[str] = None) -> str:
        """
        Save benchmark results as a baseline for future comparisons.

        Args:
            results: Benchmark results to save as baseline
            baseline_dir: Directory to save baseline to

        Returns:
            Path to the saved baseline file
        """
        if baseline_dir is None:
            baseline_dir = PERFORMANCE_CONFIG.get("BASELINE_DIR", "performance_baselines")

        # Create directory if it doesn't exist
        os.makedirs(baseline_dir, exist_ok=True)

        name = results.get("name", "unknown")
        baseline_path = os.path.join(baseline_dir, f"{name}_baseline.json")

        # Save results to file
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved benchmark as baseline: {baseline_path}")
        return baseline_path


def benchmark(name: Optional[str] = None, save_results: bool = False, iterations: int = 1):
    """
    Decorator for benchmarking function execution.

    This decorator measures the execution time and resource usage of a function
    and records the results. It can be used with both synchronous and asynchronous
    functions.

    Args:
        name: Name of the benchmark (defaults to function name)
        save_results: Whether to save results to a file
        iterations: Number of times to run the function for averaging results

    Returns:
        Decorated function that includes benchmarking
    """

    def decorator(func):
        benchmark_name = name or func.__name__

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create benchmark
            bench = PerformanceBenchmark(benchmark_name)

            # Run multiple iterations if requested
            results = []

            for i in range(iterations):
                # Start timing
                iter_start = time.time()

                # Run the function
                bench.start()
                result = func(*args, **kwargs)
                bench.stop()

                # Record iteration time
                iter_end = time.time()
                iter_time = iter_end - iter_start

                bench.record_iteration(
                    f"Iteration {i+1}",
                    {
                        "elapsed_time": iter_time,
                        "elapsed_time_ms": iter_time * 1000,
                    },
                )

                results.append(result)

            # Calculate aggregate metrics if multiple iterations
            if iterations > 1:
                iter_times = [it["metrics"]["elapsed_time"] for it in bench.results["iterations"]]
                bench.results["aggregate"] = {
                    "mean_time": statistics.mean(iter_times),
                    "median_time": statistics.median(iter_times),
                    "min_time": min(iter_times),
                    "max_time": max(iter_times),
                    "stdev_time": statistics.stdev(iter_times) if len(iter_times) > 1 else 0,
                }

            # Print summary
            bench.print_summary()

            # Save results if requested
            if save_results:
                bench.save_results()

            # Return the result of the last iteration
            return results[-1]

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create benchmark
            bench = PerformanceBenchmark(benchmark_name)

            # Run multiple iterations if requested
            results = []

            for i in range(iterations):
                # Start timing
                iter_start = time.time()

                # Run the function
                bench.start()
                result = await func(*args, **kwargs)
                bench.stop()

                # Record iteration time
                iter_end = time.time()
                iter_time = iter_end - iter_start

                bench.record_iteration(
                    f"Iteration {i+1}",
                    {
                        "elapsed_time": iter_time,
                        "elapsed_time_ms": iter_time * 1000,
                    },
                )

                results.append(result)

            # Calculate aggregate metrics if multiple iterations
            if iterations > 1:
                iter_times = [it["metrics"]["elapsed_time"] for it in bench.results["iterations"]]
                bench.results["aggregate"] = {
                    "mean_time": statistics.mean(iter_times),
                    "median_time": statistics.median(iter_times),
                    "min_time": min(iter_times),
                    "max_time": max(iter_times),
                    "stdev_time": statistics.stdev(iter_times) if len(iter_times) > 1 else 0,
                }

            # Print summary
            bench.print_summary()

            # Save results if requested
            if save_results:
                bench.save_results()

            # Return the result of the last iteration
            return results[-1]

        # Return the appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ResourceMonitor:
    """
    Monitor resource usage during code execution.

    This class provides methods for monitoring memory usage, CPU usage, and other
    resource metrics during code execution. It can be used to identify resource
    leaks and performance bottlenecks.

    Attributes:
        interval: Monitoring interval in seconds
        metrics: Dictionary of collected metrics
    """

    def __init__(self, interval: float = 1.0):
        """
        Initialize a new resource monitor.

        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "memory": [],
            "cpu": [],
            "handles": [],
            "threads": [],
        }
        self._stop_event = asyncio.Event()
        self._task = None

    async def start(self) -> None:
        """Start monitoring resources."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitor())

    async def stop(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stop monitoring and return collected metrics.

        Returns:
            Dictionary of collected metrics
        """
        if self._task is not None:
            self._stop_event.set()
            await self._task
            self._task = None

        return self.metrics

    async def _monitor(self) -> None:
        """Monitor resources at regular intervals."""
        process = psutil.Process(os.getpid())

        while not self._stop_event.is_set():
            try:
                # Record memory usage
                memory_info = process.memory_info()
                self.metrics["memory"].append(
                    {
                        "timestamp": time.time(),
                        "rss": memory_info.rss,
                        "rss_mb": memory_info.rss / (1024 * 1024),
                        "vms": memory_info.vms,
                        "vms_mb": memory_info.vms / (1024 * 1024),
                    }
                )

                # Record CPU usage
                cpu_percent = process.cpu_percent()
                self.metrics["cpu"].append(
                    {
                        "timestamp": time.time(),
                        "percent": cpu_percent,
                    }
                )

                # Record handle count (Windows only)
                try:
                    handle_count = process.num_handles() if hasattr(process, "num_handles") else 0
                    self.metrics["handles"].append(
                        {
                            "timestamp": time.time(),
                            "count": handle_count,
                        }
                    )
                except (AttributeError, psutil.AccessDenied):
                    pass

                # Record thread count
                thread_count = process.num_threads()
                self.metrics["threads"].append(
                    {
                        "timestamp": time.time(),
                        "count": thread_count,
                    }
                )

                # Wait for next interval
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    # This is expected when the timeout expires
                    pass
            except Exception as e:
                logger.error(f"Error during resource monitoring: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of resource usage.

        Returns:
            Dictionary summarizing resource usage metrics
        """
        summary = {}

        # Memory summary
        if self.metrics["memory"]:
            rss_values = [m["rss"] for m in self.metrics["memory"]]
            summary["memory"] = {
                "min_mb": min(rss_values) / (1024 * 1024),
                "max_mb": max(rss_values) / (1024 * 1024),
                "avg_mb": sum(rss_values) / len(rss_values) / (1024 * 1024),
                "growth_mb": (rss_values[-1] - rss_values[0]) / (1024 * 1024),
            }

        # CPU summary
        if self.metrics["cpu"]:
            cpu_values = [m["percent"] for m in self.metrics["cpu"]]
            summary["cpu"] = {
                "min_percent": min(cpu_values),
                "max_percent": max(cpu_values),
                "avg_percent": sum(cpu_values) / len(cpu_values),
            }

        # Thread summary
        if self.metrics["threads"]:
            thread_values = [m["count"] for m in self.metrics["threads"]]
            summary["threads"] = {
                "min_count": min(thread_values),
                "max_count": max(thread_values),
                "growth": thread_values[-1] - thread_values[0],
            }

        return summary

    def print_summary(self) -> None:
        """Print a summary of resource usage to the console."""
        summary = self.get_summary()

        print("\n=== Resource Usage Summary ===")

        if "memory" in summary:
            mem = summary["memory"]
            print(
                f"Memory: min={mem['min_mb']:.2f}MB, max={mem['max_mb']:.2f}MB, avg={mem['avg_mb']:.2f}MB"
            )
            print(f"Memory growth: {mem['growth_mb']:.2f}MB")

        if "cpu" in summary:
            cpu = summary["cpu"]
            print(
                f"CPU: min={cpu['min_percent']:.2f}%, max={cpu['max_percent']:.2f}%, avg={cpu['avg_percent']:.2f}%"
            )

        if "threads" in summary:
            threads = summary["threads"]
            print(
                f"Threads: min={threads['min_count']}, max={threads['max_count']}, growth={threads['growth']}"
            )


class MemoryProfiler:
    """
    Profile memory usage during code execution.

    This class provides methods for profiling memory usage and identifying memory
    leaks. It uses the tracemalloc module to track memory allocations and provide
    detailed information about memory usage.

    Attributes:
        top_n: Number of top memory allocations to track
        snapshots: List of memory snapshots
    """

    def __init__(self, top_n: int = 10):
        """
        Initialize a new memory profiler.

        Args:
            top_n: Number of top memory allocations to track
        """
        self.top_n = top_n
        self.snapshots = []

        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def take_snapshot(self, name: str) -> None:
        """
        Take a snapshot of current memory usage.

        Args:
            name: Name of the snapshot (for reference)
        """
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((name, snapshot))

    def compare_snapshots(self, start_index: int = 0, end_index: int = -1) -> List[Dict[str, Any]]:
        """
        Compare two memory snapshots to identify memory growth.

        Args:
            start_index: Index of the first snapshot
            end_index: Index of the second snapshot

        Returns:
            List of dictionaries describing memory allocations between snapshots
        """
        if not self.snapshots:
            return []

        start = self.snapshots[start_index][1]
        end = self.snapshots[end_index][1]

        top_stats = end.compare_to(start, "lineno")

        result = []
        for stat in top_stats[: self.top_n]:
            frames = []
            for frame in stat.traceback:
                frames.append(
                    {
                        "filename": frame.filename,
                        "lineno": frame.lineno,
                    }
                )

            result.append(
                {
                    "size": stat.size,
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count,
                    "frames": frames,
                }
            )

        return result

    def print_comparison(self, start_index: int = 0, end_index: int = -1) -> None:
        """
        Print a comparison of two memory snapshots.

        Args:
            start_index: Index of the first snapshot
            end_index: Index of the second snapshot
        """
        if not self.snapshots:
            print("No snapshots available")
            return

        start_name = self.snapshots[start_index][0]
        end_name = self.snapshots[end_index][0]

        start = self.snapshots[start_index][1]
        end = self.snapshots[end_index][1]

        print(f"\n=== Memory Growth: {start_name} -> {end_name} ===")

        top_stats = end.compare_to(start, "lineno")

        for i, stat in enumerate(top_stats[: self.top_n], 1):
            print(f"\n#{i}: {stat.size / 1024:.1f} KB, {stat.count} objects")
            for frame in stat.traceback:
                print(f"    {frame.filename}:{frame.lineno}")

    def get_current_allocated_memory(self) -> int:
        """
        Get the currently allocated memory in bytes.

        Returns:
            Currently allocated memory in bytes
        """
        return tracemalloc.get_traced_memory()[0]

    def stop(self) -> None:
        """Stop memory profiling and release resources."""
        tracemalloc.stop()
        self.snapshots = []


async def benchmark_provider(
    provider_name: str, ticker_count: int = 5, iterations: int = 3, resource_monitor: bool = True
) -> Dict[str, Any]:
    """
    Benchmark a provider implementation.

    This function measures the performance of a provider implementation by
    executing common operations and recording metrics.

    Args:
        provider_name: Name of the provider to benchmark ('yahoo', 'yahooquery', 'hybrid')
        ticker_count: Number of tickers to use in the benchmark
        iterations: Number of iterations to run
        resource_monitor: Whether to monitor resource usage during benchmarking

    Returns:
        Dictionary of benchmark results
    """
    # Test tickers (mix of US and international)
    test_tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "HD",
        "COST",
        "UNH",
    ]

    # Select subset of tickers based on ticker_count
    tickers = test_tickers[:ticker_count]

    # Create benchmark
    bench = PerformanceBenchmark(f"provider_{provider_name}")

    # Start resource monitor if enabled
    monitor = None
    if resource_monitor:
        monitor = ResourceMonitor(interval=0.1)
        await monitor.start()

    # Start benchmark
    bench.start()

    # Run benchmark
    for i in range(iterations):
        iter_start = time.time()

        # Get provider
        provider = get_provider(provider_name=provider_name, async_api=True)

        # Measure batch get_ticker_info
        batch_start = time.time()
        await provider.batch_get_ticker_info(tickers)
        batch_end = time.time()
        batch_time = batch_end - batch_start

        # Measure individual get_ticker_info
        individual_times = []
        for ticker in tickers:
            ticker_start = time.time()
            await provider.get_ticker_info(ticker)
            ticker_end = time.time()
            individual_times.append(ticker_end - ticker_start)

        # Close provider session
        await provider.close()

        # Record iteration
        iter_end = time.time()
        iter_time = iter_end - iter_start

        bench.record_iteration(
            f"Iteration {i+1}",
            {
                "elapsed_time": iter_time,
                "elapsed_time_ms": iter_time * 1000,
                "batch_time": batch_time,
                "batch_time_ms": batch_time * 1000,
                "avg_individual_time": statistics.mean(individual_times),
                "avg_individual_time_ms": statistics.mean(individual_times) * 1000,
                "min_individual_time_ms": min(individual_times) * 1000,
                "max_individual_time_ms": max(individual_times) * 1000,
            },
        )

    # Stop benchmark
    results = bench.stop()

    # Stop resource monitor if enabled
    if monitor:
        await monitor.stop()
        monitor.print_summary()

        # Add resource metrics to results
        results["resources"] = monitor.get_summary()

    # Print summary
    bench.print_summary()

    # Save results
    bench.save_results()

    return results


def find_memory_leaks(func, *args, iterations: int = 10, **kwargs) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for memory leaks in a function by running it multiple times.

    This function runs another function multiple times and monitors memory usage
    to detect potential memory leaks. It uses tracemalloc to track memory allocations
    and identify the source of potential leaks.

    Args:
        func: Function to check for memory leaks
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (leak_detected, metrics)
    """
    # Stop tracing if already running to ensure a clean state
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    # Start tracing with 10 frames to get detailed traceback
    tracemalloc.start(10)

    memory_usage = []
    gc_stats = []

    # Force garbage collection before starting
    gc.collect()

    # Take initial snapshot
    snapshot_start = tracemalloc.take_snapshot()
    start_memory = tracemalloc.get_traced_memory()[0]
    peak_memory = tracemalloc.get_traced_memory()[1]

    logger.info(f"Starting memory leak check, running {iterations} iterations")

    # Run function multiple times
    for i in range(iterations):
        logger.debug(f"Memory leak check iteration {i+1}/{iterations}")

        # Collect memory before iteration
        before_iter_memory = tracemalloc.get_traced_memory()[0]

        # Run function with appropriate handling for async
        if asyncio.iscoroutinefunction(func):
            try:
                # Handle case where we might already be in an event loop
                if asyncio.get_event_loop().is_running():
                    loop = asyncio.get_event_loop()
                    fut = asyncio.ensure_future(func(*args, **kwargs), loop=loop)
                    loop.run_until_complete(fut)
                else:
                    asyncio.run(func(*args, **kwargs))
            except Exception as e:
                logger.error(f"Error during iteration {i+1}: {str(e)}")
        else:
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error during iteration {i+1}: {str(e)}")

        # Record GC stats before collection
        gc_count_before = (gc.get_count()[0], gc.get_count()[1], gc.get_count()[2])

        # Force garbage collection
        gc_collected = gc.collect(2)  # Full collection (all generations)

        # Record memory after iteration and GC
        current_memory = tracemalloc.get_traced_memory()[0]
        current_peak = tracemalloc.get_traced_memory()[1]
        peak_memory = max(peak_memory, current_peak)

        # Record memory growth in this iteration
        memory_growth_in_iteration = current_memory - before_iter_memory

        # Record memory usage and GC stats
        memory_usage.append(
            {
                "iteration": i + 1,
                "memory": current_memory,
                "memory_mb": current_memory / (1024 * 1024),
                "growth_in_iteration": memory_growth_in_iteration,
                "growth_in_iteration_mb": memory_growth_in_iteration / (1024 * 1024),
            }
        )

        gc_stats.append(
            {
                "iteration": i + 1,
                "gc_count_before": gc_count_before,
                "objects_collected": gc_collected,
            }
        )

    # Take final snapshot
    snapshot_end = tracemalloc.take_snapshot()
    end_memory = tracemalloc.get_traced_memory()[0]

    # Compare snapshots
    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")

    # Process top memory consumers
    memory_growth = []
    for stat in top_stats[:20]:  # Capture more information for detailed analysis
        # Only include significant allocations (>100 bytes)
        if stat.size_diff > 100:
            frames = []
            for frame in stat.traceback[:3]:  # Include up to 3 frames
                frames.append(
                    {
                        "filename": os.path.basename(frame.filename),
                        "line": frame.lineno,
                    }
                )

            memory_growth.append(
                {
                    "size": stat.size_diff,
                    "size_kb": stat.size_diff / 1024,
                    "count_diff": stat.count_diff,
                    "frames": frames,
                    "file": (
                        os.path.basename(stat.traceback[0].filename)
                        if stat.traceback
                        else "unknown"
                    ),
                    "line": stat.traceback[0].lineno if stat.traceback else 0,
                }
            )

    # Calculate metrics
    memory_diff = end_memory - start_memory
    memory_growth_per_iteration = memory_diff / iterations if iterations > 0 else 0

    # Analyze memory growth pattern
    # Check if memory is consistently growing across iterations
    memory_values = [entry["memory"] for entry in memory_usage]

    # Check for steady growth (not just fluctuating)
    # We look for a clear upward trend, not just random increases
    consistent_growth = False
    if len(memory_values) >= 3:  # Need at least 3 points for a trend
        # Calculate a simple linear regression slope
        n = len(memory_values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(memory_values) / n

        numerator = sum((x[i] - x_mean) * (memory_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Consider it a leak if slope is positive and memory grew by more than 10%
        growth_percent = ((end_memory / start_memory) - 1) * 100 if start_memory > 0 else 0
        consistent_growth = slope > 0 and growth_percent > 10

    # Apply heuristics to detect potential leaks
    threshold_mb = BENCHMARK_SETTINGS.get("MEMORY_PROFILE_THRESHOLD", 1.2)  # Default 1.2MB
    absolute_growth_significant = memory_diff > (threshold_mb * 1024 * 1024)  # Convert MB to bytes

    # Consider it a leak if:
    # 1. Memory consistently grows across iterations, OR
    # 2. Total memory growth is significant by absolute standards
    is_leaking = consistent_growth or absolute_growth_significant

    detailed_metrics = {
        "start_memory": start_memory,
        "start_memory_mb": start_memory / (1024 * 1024),
        "end_memory": end_memory,
        "end_memory_mb": end_memory / (1024 * 1024),
        "peak_memory": peak_memory,
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "memory_diff": memory_diff,
        "memory_diff_mb": memory_diff / (1024 * 1024),
        "memory_growth_per_iteration": memory_growth_per_iteration,
        "memory_growth_per_iteration_mb": memory_growth_per_iteration / (1024 * 1024),
        "memory_growth_percent": ((end_memory / start_memory) - 1) * 100 if start_memory > 0 else 0,
        "iterations": iterations,
        "memory_usage": memory_usage,
        "gc_stats": gc_stats,
        "top_consumers": memory_growth,
        "is_leaking": is_leaking,
        "consistent_growth": consistent_growth,
        "absolute_growth_significant": absolute_growth_significant,
    }

    # Stop tracing
    tracemalloc.stop()

    return is_leaking, detailed_metrics


async def find_memory_leaks_async(
    func, *args, iterations: int = 10, **kwargs
) -> Tuple[bool, Dict[str, Any]]:
    """
    Async version of find_memory_leaks that properly handles async functions.

    Args:
        func: Async function to check for memory leaks
        *args: Arguments to pass to the function
        iterations: Number of iterations to run
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Tuple of (leak_detected, metrics)
    """
    # Stop tracing if already running to ensure a clean state
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    # Start tracing with 10 frames to get detailed traceback
    tracemalloc.start(10)

    memory_usage = []
    gc_stats = []

    # Force garbage collection before starting - run multiple collections to ensure we clean up all resources
    gc.collect()
    gc.collect()
    gc.collect()

    # Take initial snapshot
    snapshot_start = tracemalloc.take_snapshot()
    start_memory = tracemalloc.get_traced_memory()[0]
    peak_memory = tracemalloc.get_traced_memory()[1]

    # Print detailed memory statistics for debugging
    gc_stats_start = {
        "gc_counts": gc.get_count(),
        "gc_objects": len(gc.get_objects()),
        "gc_garbage": len(gc.garbage) if hasattr(gc, "garbage") else 0,
    }
    logger.info(f"Starting async memory leak check, running {iterations} iterations")
    logger.info(f"Initial memory: {start_memory / 1024 / 1024:.3f} MB, GC stats: {gc_stats_start}")

    # Run function multiple times
    for i in range(iterations):
        logger.debug(f"Memory leak check iteration {i+1}/{iterations}")

        # Collect memory before iteration
        before_iter_memory = tracemalloc.get_traced_memory()[0]

        # Run function with error handling
        try:
            # Directly await the function since we're already in an async context
            await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during iteration {i+1}: {str(e)}")

        # Record GC stats before collection
        gc_count_before = (gc.get_count()[0], gc.get_count()[1], gc.get_count()[2])

        # Force garbage collection
        gc_collected = gc.collect(2)  # Full collection (all generations)

        # Record memory after iteration and GC
        current_memory = tracemalloc.get_traced_memory()[0]
        current_peak = tracemalloc.get_traced_memory()[1]
        peak_memory = max(peak_memory, current_peak)

        # Record memory growth in this iteration
        memory_growth_in_iteration = current_memory - before_iter_memory

        # Record memory usage and GC stats
        memory_usage.append(
            {
                "iteration": i + 1,
                "memory": current_memory,
                "memory_mb": current_memory / (1024 * 1024),
                "growth_in_iteration": memory_growth_in_iteration,
                "growth_in_iteration_mb": memory_growth_in_iteration / (1024 * 1024),
            }
        )

        gc_stats.append(
            {
                "iteration": i + 1,
                "gc_count_before": gc_count_before,
                "objects_collected": gc_collected,
            }
        )

        # Give the event loop a chance to clean up resources
        await asyncio.sleep(0.1)

    # Explicitly run garbage collection multiple times to ensure all objects are properly cleaned up
    gc.collect()
    gc.collect()
    gc.collect()

    # Explicitly clean up pandas caches if they exist
    try:
        import pandas

        if hasattr(pandas.core.common, "_possibly_clean_cache"):
            pandas.core.common._possibly_clean_cache()
    except (ImportError, AttributeError):
        pass

    # Explicitly clean up ABC module caches
    try:
        import abc

        if hasattr(abc, "_abc_registry"):
            abc._abc_registry.clear()
        if hasattr(abc, "_abc_cache"):
            abc._abc_cache.clear()
    except (ImportError, AttributeError):
        pass

    # Take final snapshot
    snapshot_end = tracemalloc.take_snapshot()
    end_memory = tracemalloc.get_traced_memory()[0]

    # Print final memory stats
    gc_stats_end = {
        "gc_counts": gc.get_count(),
        "gc_objects": len(gc.get_objects()),
        "gc_garbage": len(gc.garbage) if hasattr(gc, "garbage") else 0,
    }
    logger.info(f"Final memory: {end_memory / 1024 / 1024:.3f} MB, GC stats: {gc_stats_end}")

    # Compare snapshots
    top_stats = snapshot_end.compare_to(snapshot_start, "lineno")

    # Process top memory consumers
    memory_growth = []
    for stat in top_stats[:20]:  # Capture more information for detailed analysis
        # Only include significant allocations (>100 bytes)
        if stat.size_diff > 100:
            frames = []
            for frame in stat.traceback[:3]:  # Include up to 3 frames
                frames.append(
                    {
                        "filename": os.path.basename(frame.filename),
                        "line": frame.lineno,
                    }
                )

            memory_growth.append(
                {
                    "size": stat.size_diff,
                    "size_kb": stat.size_diff / 1024,
                    "count_diff": stat.count_diff,
                    "frames": frames,
                    "file": (
                        os.path.basename(stat.traceback[0].filename)
                        if stat.traceback
                        else "unknown"
                    ),
                    "line": stat.traceback[0].lineno if stat.traceback else 0,
                }
            )

    # Calculate metrics
    memory_diff = end_memory - start_memory
    memory_growth_per_iteration = memory_diff / iterations if iterations > 0 else 0

    # Analyze memory growth pattern
    # Check if memory is consistently growing across iterations
    memory_values = [entry["memory"] for entry in memory_usage]

    # Check for steady growth (not just fluctuating)
    # We look for a clear upward trend, not just random increases
    consistent_growth = False
    if len(memory_values) >= 3:  # Need at least 3 points for a trend
        # Calculate a simple linear regression slope
        n = len(memory_values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(memory_values) / n

        numerator = sum((x[i] - x_mean) * (memory_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Consider it a leak if slope is positive and memory grew by more than 10%
        growth_percent = ((end_memory / start_memory) - 1) * 100 if start_memory > 0 else 0
        consistent_growth = slope > 0 and growth_percent > 10

    # Apply heuristics to detect potential leaks
    threshold_mb = BENCHMARK_SETTINGS.get("MEMORY_PROFILE_THRESHOLD", 1.2)  # Default 1.2MB
    absolute_growth_significant = memory_diff > (threshold_mb * 1024 * 1024)  # Convert MB to bytes

    # Consider it a leak if:
    # 1. Memory consistently grows across iterations, OR
    # 2. Total memory growth is significant by absolute standards
    is_leaking = consistent_growth or absolute_growth_significant

    detailed_metrics = {
        "start_memory": start_memory,
        "start_memory_mb": start_memory / (1024 * 1024),
        "end_memory": end_memory,
        "end_memory_mb": end_memory / (1024 * 1024),
        "peak_memory": peak_memory,
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "memory_diff": memory_diff,
        "memory_diff_mb": memory_diff / (1024 * 1024),
        "memory_growth_per_iteration": memory_growth_per_iteration,
        "memory_growth_per_iteration_mb": memory_growth_per_iteration / (1024 * 1024),
        "memory_growth_percent": ((end_memory / start_memory) - 1) * 100 if start_memory > 0 else 0,
        "iterations": iterations,
        "memory_usage": memory_usage,
        "gc_stats": gc_stats,
        "top_consumers": memory_growth,
        "is_leaking": is_leaking,
        "consistent_growth": consistent_growth,
        "absolute_growth_significant": absolute_growth_significant,
    }

    # Stop tracing
    tracemalloc.stop()

    return is_leaking, detailed_metrics


async def run_batch_size_benchmark(
    batch_sizes: List[int] = [1, 5, 10, 15, 20, 25],
    ticker_count: int = 20,
    provider_name: str = "hybrid",
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark the impact of different batch sizes on performance.

    This function tests the performance of the API with different batch sizes
    to determine the optimal batch size for the current environment.

    Args:
        batch_sizes: List of batch sizes to test
        ticker_count: Number of tickers to use in the benchmark
        provider_name: Name of the provider to benchmark

    Returns:
        Dictionary mapping batch sizes to benchmark results
    """
    from ..core.config import RATE_LIMIT

    results = {}

    # Test tickers (mix of US and international)
    test_tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "HD",
        "COST",
        "UNH",
        "BAC",
        "DIS",
        "NFLX",
        "INTC",
        "AMD",
        "CSCO",
        "ORCL",
        "IBM",
        "CRM",
        "ADBE",
    ]

    # Select subset of tickers based on ticker_count (max 25)
    tickers = test_tickers[: min(ticker_count, 25)]

    # Save original batch size
    original_batch_size = RATE_LIMIT["BATCH_SIZE"]

    # Test each batch size
    for batch_size in batch_sizes:
        print(f"\n=== Testing Batch Size: {batch_size} ===")

        # Set new batch size
        RATE_LIMIT["BATCH_SIZE"] = batch_size

        # Create benchmark
        bench = PerformanceBenchmark(f"batch_size_{batch_size}")

        # Start benchmark
        bench.start()

        # Get provider
        provider = get_provider(provider_name=provider_name, async_api=True)

        # Measure batch get_ticker_info
        batch_start = time.time()
        await provider.batch_get_ticker_info(tickers)
        batch_end = time.time()
        batch_time = batch_end - batch_start

        # Close provider session
        await provider.close()

        # Stop benchmark
        bench_results = bench.stop()

        # Record batch-specific metrics
        bench_results["batch_metrics"] = {
            "batch_size": batch_size,
            "ticker_count": len(tickers),
            "batch_time": batch_time,
            "batch_time_ms": batch_time * 1000,
            "time_per_ticker_ms": batch_time * 1000 / len(tickers),
        }

        # Print summary
        bench.print_summary()
        print(f"Time per ticker: {batch_time * 1000 / len(tickers):.2f} ms")

        # Save results
        bench.save_results()

        # Store in results dictionary
        results[batch_size] = bench_results

        # Small delay to ensure rate limits are reset
        await asyncio.sleep(2)

    # Restore original batch size
    RATE_LIMIT["BATCH_SIZE"] = original_batch_size

    # Print comparison table
    print("\n=== Batch Size Comparison ===")
    print(f"{'Batch Size':<10} | {'Total Time (ms)':<15} | {'Time/Ticker (ms)':<15}")
    print("-" * 45)
    for batch_size, result in sorted(results.items()):
        total_time = result["batch_metrics"]["batch_time_ms"]
        time_per_ticker = result["batch_metrics"]["time_per_ticker_ms"]
        print(f"{batch_size:<10} | {total_time:<15.2f} | {time_per_ticker:<15.2f}")

    return results


async def run_concurrency_benchmark(
    concurrency_levels: List[int] = [1, 5, 10, 15, 20],
    ticker_count: int = 20,
    provider_name: str = "hybrid",
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark the impact of different concurrency levels on performance.

    This function tests the performance of the API with different concurrency
    levels to determine the optimal concurrency for the current environment.

    Args:
        concurrency_levels: List of concurrency levels to test
        ticker_count: Number of tickers to use in the benchmark
        provider_name: Name of the provider to benchmark

    Returns:
        Dictionary mapping concurrency levels to benchmark results
    """
    from ..core.config import RATE_LIMIT

    results = {}

    # Test tickers (mix of US and international)
    test_tickers = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "HD",
        "COST",
        "UNH",
        "BAC",
        "DIS",
        "NFLX",
        "INTC",
        "AMD",
        "CSCO",
        "ORCL",
        "IBM",
        "CRM",
        "ADBE",
    ]

    # Select subset of tickers based on ticker_count (max 25)
    tickers = test_tickers[: min(ticker_count, 25)]

    # Save original concurrency
    original_concurrency = RATE_LIMIT["MAX_CONCURRENT_CALLS"]

    # Test each concurrency level
    for concurrency in concurrency_levels:
        print(f"\n=== Testing Concurrency Level: {concurrency} ===")

        # Set new concurrency
        RATE_LIMIT["MAX_CONCURRENT_CALLS"] = concurrency

        # Create benchmark
        bench = PerformanceBenchmark(f"concurrency_{concurrency}")

        # Start benchmark
        bench.start()

        # Get provider
        provider = get_provider(provider_name=provider_name, async_api=True)

        # Measure batch get_ticker_info
        batch_start = time.time()
        await provider.batch_get_ticker_info(tickers)
        batch_end = time.time()
        batch_time = batch_end - batch_start

        # Close provider session
        await provider.close()

        # Stop benchmark
        bench_results = bench.stop()

        # Record concurrency-specific metrics
        bench_results["concurrency_metrics"] = {
            "concurrency": concurrency,
            "ticker_count": len(tickers),
            "batch_time": batch_time,
            "batch_time_ms": batch_time * 1000,
            "time_per_ticker_ms": batch_time * 1000 / len(tickers),
        }

        # Print summary
        bench.print_summary()
        print(f"Time per ticker: {batch_time * 1000 / len(tickers):.2f} ms")

        # Save results
        bench.save_results()

        # Store in results dictionary
        results[concurrency] = bench_results

        # Small delay to ensure rate limits are reset
        await asyncio.sleep(2)

    # Restore original concurrency
    RATE_LIMIT["MAX_CONCURRENT_CALLS"] = original_concurrency

    # Print comparison table
    print("\n=== Concurrency Level Comparison ===")
    print(f"{'Concurrency':<10} | {'Total Time (ms)':<15} | {'Time/Ticker (ms)':<15}")
    print("-" * 45)
    for concurrency, result in sorted(results.items()):
        total_time = result["concurrency_metrics"]["batch_time_ms"]
        time_per_ticker = result["concurrency_metrics"]["time_per_ticker_ms"]
        print(f"{concurrency:<10} | {total_time:<15.2f} | {time_per_ticker:<15.2f}")

    return results


async def run_all_benchmarks(save_as_baseline: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks for all providers and configurations.

    This function runs comprehensive benchmarks across different providers,
    configurations, and workloads to compare performance.

    Args:
        save_as_baseline: Whether to save results as new baselines

    Returns:
        Dictionary mapping benchmark names to results
    """
    results = {}

    # Standard provider benchmarks
    providers = ["yahoo", "yahooquery", "hybrid"]
    for provider in providers:
        print(f"\nBenchmarking {provider} provider...")
        provider_results = await benchmark_provider(provider, ticker_count=5, iterations=3)

        results[f"provider_{provider}"] = provider_results

        if save_as_baseline:
            PerformanceBenchmark.save_as_baseline(provider_results)

    # Batch size scaling benchmark
    print("\nBenchmarking batch size scaling...")
    batch_results = await run_batch_size_benchmark(batch_sizes=[1, 5, 10, 15, 20])
    results["batch_size_scaling"] = batch_results

    # Concurrency scaling benchmark
    print("\nBenchmarking concurrency scaling...")
    concurrency_results = await run_concurrency_benchmark(concurrency_levels=[1, 5, 10, 15, 20])
    results["concurrency_scaling"] = concurrency_results

    # Print overall comparison
    print("\n=== Overall Performance Comparison ===")
    print(f"{'Provider':<15} | {'Time/Ticker (ms)':<15} | {'Memory (MB)':<15}")
    print("-" * 50)
    for provider in providers:
        provider_result = results[f"provider_{provider}"]
        time_per_ticker = 0
        memory_used = 0

        if "iterations" in provider_result and provider_result["iterations"]:
            # Average time per ticker across all iterations
            iteration_times = []
            for iteration in provider_result["iterations"]:
                if "metrics" in iteration and "elapsed_time" in iteration["metrics"]:
                    iteration_times.append(iteration["metrics"]["elapsed_time"])

            if iteration_times:
                avg_time = statistics.mean(iteration_times)
                # Assuming 5 tickers per test
                time_per_ticker = avg_time * 1000 / 5

        if "metrics" in provider_result and "memory_used_mb" in provider_result["metrics"]:
            memory_used = provider_result["metrics"]["memory_used_mb"]

        print(f"{provider:<15} | {time_per_ticker:<15.2f} | {memory_used:<15.2f}")

    return results


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results including timing and resource usage.
    """

    operation: str
    iterations: int
    start_time: float
    end_time: float
    duration: float
    min_time: float = field(default=0)
    max_time: float = field(default=0)
    avg_time: float = field(default=0)
    median_time: float = field(default=0)
    std_dev: float = field(default=0)
    success_rate: float = field(default=1.0)
    error_count: int = field(default=0)
    memory_before: int = field(default=0)
    memory_after: int = field(default=0)
    memory_peak: int = field(default=0)
    cpu_usage_avg: float = field(default=0)
    thread_count_avg: int = field(default=0)
    data_size: Optional[int] = field(default=None)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to a dictionary."""
        result = {
            "operation": self.operation,
            "iterations": self.iterations,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration_seconds": self.duration,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "median_time": self.median_time,
            "std_dev": self.std_dev,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "parameters": self.parameters,
        }

        # Add resource metrics if available
        if self.memory_before > 0:
            result["memory_before_mb"] = self.memory_before / (1024 * 1024)
            result["memory_after_mb"] = self.memory_after / (1024 * 1024)
            result["memory_peak_mb"] = self.memory_peak / (1024 * 1024)
            result["memory_change_mb"] = (self.memory_after - self.memory_before) / (1024 * 1024)
            result["memory_change_percent"] = (
                ((self.memory_after / self.memory_before) - 1) * 100 if self.memory_before else 0
            )

        if self.cpu_usage_avg > 0:
            result["cpu_usage_avg_percent"] = self.cpu_usage_avg
            result["thread_count_avg"] = self.thread_count_avg

        if self.data_size is not None:
            result["data_size_bytes"] = self.data_size

        return result

    def print_summary(self) -> None:
        """Print a formatted summary of the benchmark results."""
        print(f"\n=== Benchmark Results: {self.operation} ===")
        print(f"Iterations: {self.iterations}")
        print(f"Total duration: {self.duration:.2f} seconds")
        print(f"Average time: {self.avg_time:.4f} seconds")
        print(f"Median time: {self.median_time:.4f} seconds")
        print(f"Min/Max time: {self.min_time:.4f}s / {self.max_time:.4f}s")
        print(f"Standard deviation: {self.std_dev:.4f}")
        print(f"Success rate: {self.success_rate * 100:.1f}%")

        if self.memory_before > 0:
            memory_change = self.memory_after - self.memory_before
            memory_change_mb = memory_change / (1024 * 1024)
            memory_change_pct = (
                ((self.memory_after / self.memory_before) - 1) * 100 if self.memory_before else 0
            )

            print("\n--- Resource Usage ---")
            print(f"Memory before: {self.memory_before / (1024 * 1024):.2f} MB")
            print(f"Memory after: {self.memory_after / (1024 * 1024):.2f} MB")
            print(f"Memory change: {memory_change_mb:.2f} MB ({memory_change_pct:.1f}%)")
            print(f"Peak memory: {self.memory_peak / (1024 * 1024):.2f} MB")

        if self.cpu_usage_avg > 0:
            print(f"Average CPU usage: {self.cpu_usage_avg:.1f}%")
            print(f"Average thread count: {self.thread_count_avg}")

        if self.parameters:
            print("\n--- Parameters ---")
            for key, value in self.parameters.items():
                print(f"{key}: {value}")


class PriorityAsyncRateLimiter:
    """
    Enhanced async rate limiter with support for priority tiers and token bucket algorithm.

    This rate limiter allows different rate limits for different priority tiers,
    ensuring high-priority operations get preferential treatment. It uses a token
    bucket algorithm for smoother rate limiting.

    Attributes:
        high_priority_quota: Maximum API calls allowed for high priority operations
        medium_priority_quota: Maximum API calls allowed for medium priority operations
        low_priority_quota: Maximum API calls allowed for low priority operations
        window_size: Time window for rate limiting (seconds)
    """

    def __init__(
        self,
        window_size: int = None,
        max_calls: int = None,
        base_delay: float = None,
        min_delay: float = None,
        max_delay: float = None,
    ):
        """
        Initialize the priority-based rate limiter.

        Args:
            window_size: Time window for rate limiting in seconds
            max_calls: Maximum API calls in the window
            base_delay: Base delay between calls in seconds
            min_delay: Minimum delay after successful calls
            max_delay: Maximum delay after errors
        """
        from ..core.config import RATE_LIMIT

        # Set default values from config if not provided
        self.window_size = window_size or RATE_LIMIT.get("WINDOW_SIZE", 60)
        self.max_calls = max_calls or RATE_LIMIT.get("MAX_CALLS", 75)
        self.base_delay = base_delay or RATE_LIMIT.get("BASE_DELAY", 0.3)
        self.min_delay = min_delay or RATE_LIMIT.get("MIN_DELAY", 0.1)
        self.max_delay = max_delay or RATE_LIMIT.get("MAX_DELAY", 30.0)

        # Initialize priority quotas
        self.high_priority_quota = self.max_calls * 0.5  # 50% for high priority
        self.medium_priority_quota = self.max_calls * 0.3  # 30% for medium priority
        self.low_priority_quota = self.max_calls * 0.2  # 20% for low priority

        # Initialize token buckets for each priority tier
        self.tokens = {
            "HIGH": self.high_priority_quota,
            "MEDIUM": self.medium_priority_quota,
            "LOW": self.low_priority_quota,
        }

        # Track calls per priority tier
        self.priority_call_times = {"HIGH": [], "MEDIUM": [], "LOW": []}

        # Track current delays per priority tier
        self.current_delays = {
            "HIGH": self.base_delay * 0.7,  # 30% faster
            "MEDIUM": self.base_delay,  # Standard delay
            "LOW": self.base_delay * 1.5,  # 50% slower
        }

        # Track success and error counts for adaptive adjustments
        self.success_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        self.error_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        # Thread safety with asyncio lock
        self._lock = asyncio.Lock()

        # Last token refill time
        self.last_refill_time = time.time()

    async def _refill_tokens(self) -> None:
        """Refill token buckets based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill_time

        # Only refill if some time has passed
        if elapsed > 0:
            # Calculate token refill rate (tokens per second)
            high_refill_rate = self.high_priority_quota / self.window_size
            medium_refill_rate = self.medium_priority_quota / self.window_size
            low_refill_rate = self.low_priority_quota / self.window_size

            # Add tokens based on elapsed time
            self.tokens["HIGH"] = min(
                self.high_priority_quota,  # Cap at max quota
                self.tokens["HIGH"] + (high_refill_rate * elapsed),
            )

            self.tokens["MEDIUM"] = min(
                self.medium_priority_quota, self.tokens["MEDIUM"] + (medium_refill_rate * elapsed)
            )

            self.tokens["LOW"] = min(
                self.low_priority_quota, self.tokens["LOW"] + (low_refill_rate * elapsed)
            )

            # Update last refill time
            self.last_refill_time = current_time

    async def consume_token(self, priority: str = "MEDIUM") -> bool:
        """
        Consume a token from the appropriate bucket.

        Args:
            priority: Priority level ("HIGH", "MEDIUM", "LOW")

        Returns:
            True if token was consumed, False if no tokens available
        """
        # Default to MEDIUM if invalid priority provided
        if priority not in ["HIGH", "MEDIUM", "LOW"]:
            priority = "MEDIUM"

        async with self._lock:
            # Refill tokens based on elapsed time
            await self._refill_tokens()

            # Check if token is available
            if self.tokens[priority] >= 1:
                self.tokens[priority] -= 1
                return True

            # If no tokens for the requested priority, try lower priorities
            if priority == "HIGH" and self.tokens["MEDIUM"] >= 1:
                self.tokens["MEDIUM"] -= 1
                return True

            if priority in ["HIGH", "MEDIUM"] and self.tokens["LOW"] >= 1:
                self.tokens["LOW"] -= 1
                return True

            # No tokens available in any bucket
            return False

    async def _clean_old_calls(self, priority: str) -> None:
        """
        Remove calls outside the current window.

        Args:
            priority: Priority level to clean
        """
        current_time = time.time()
        window_start = current_time - self.window_size

        # Filter out calls older than the window
        self.priority_call_times[priority] = [
            t for t in self.priority_call_times[priority] if t > window_start
        ]

    async def _get_current_call_count(self, priority: str) -> int:
        """
        Get the number of calls in the current window for a priority level.

        Args:
            priority: Priority level to check

        Returns:
            Number of calls in the current window
        """
        await self._clean_old_calls(priority)
        return len(self.priority_call_times[priority])

    async def _calculate_delay(self, priority: str) -> float:
        """
        Calculate delay based on priority and current rate.

        Args:
            priority: Priority level ("HIGH", "MEDIUM", "LOW")

        Returns:
            Delay in seconds
        """
        # Get current delay for this priority
        delay = self.current_delays[priority]

        # Apply jitter (±20%) using cryptographically secure random
        import secrets

        jitter_factor = 0.2
        secure_random = secrets.SystemRandom()
        jitter = secure_random.uniform(-jitter_factor, jitter_factor)
        delay = delay * (1 + jitter)

        # Apply region-specific multipliers if ticker is provided
        # This would be handled in the rate_limited decorator

        return max(self.min_delay, min(delay, self.max_delay))

    async def update_delay(self, priority: str, success: bool) -> None:
        """
        Update delay based on API call success/failure.

        Args:
            priority: Priority level
            success: Whether the API call was successful
        """
        async with self._lock:
            if success:
                # Increment success counter
                self.success_counts[priority] += 1
                self.error_counts[priority] = 0  # Reset error counter

                # If several consecutive successes, reduce delay
                if self.success_counts[priority] >= 5:  # Threshold for adjustment
                    # Reduce delay by 10%
                    self.current_delays[priority] = max(
                        self.min_delay, self.current_delays[priority] * 0.9
                    )
                    self.success_counts[priority] = 0  # Reset counter
            else:
                # Increment error counter
                self.error_counts[priority] += 1
                self.success_counts[priority] = 0  # Reset success counter

                # If consecutive errors, increase delay
                if self.error_counts[priority] >= 2:  # Threshold for adjustment
                    # Increase delay by 50%
                    self.current_delays[priority] = min(
                        self.max_delay, self.current_delays[priority] * 1.5
                    )
                    self.error_counts[priority] = 0  # Reset counter

    async def wait(self, ticker: Optional[str] = None, priority: str = "MEDIUM") -> float:
        """
        Wait for the appropriate delay based on priority and ticker properties.

        Args:
            ticker: Optional ticker symbol to determine region-specific delay
            priority: Priority level ("HIGH", "MEDIUM", "LOW")

        Returns:
            Actual delay waited in seconds
        """
        async with self._lock:
            # Clean old calls
            await self._clean_old_calls(priority)

            # Check if we need to wait
            current_calls = await self._get_current_call_count(priority)
            quota = getattr(self, f"{priority.lower()}_priority_quota")

            # If under quota, may not need to wait
            if current_calls < quota:
                if await self.consume_token(priority):
                    # Record this call
                    self.priority_call_times[priority].append(time.time())
                    return 0.0

            # Calculate appropriate delay
            delay = await self._calculate_delay(priority)

            # Apply region-specific adjustments if ticker provided
            if ticker:
                if is_us_ticker(ticker):
                    # Standard delay for US tickers
                    pass
                elif ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".L"):
                    # European markets - slightly higher delay
                    delay *= 1.1
                elif ticker.endswith(".HK") or ticker.endswith(".T") or ticker.endswith(".SS"):
                    # Asian markets - higher delay
                    delay *= 1.2

            # Wait for the calculated delay
            await asyncio.sleep(delay)

            # Record this call
            self.priority_call_times[priority].append(time.time())

            return delay

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics for rate limiting.

        Returns:
            Dictionary of rate limiting statistics
        """
        async with self._lock:
            stats = {}

            for priority in ["HIGH", "MEDIUM", "LOW"]:
                await self._clean_old_calls(priority)
                calls = await self._get_current_call_count(priority)
                quota = getattr(self, f"{priority.lower()}_priority_quota")

                stats[priority] = {
                    "calls_in_window": calls,
                    "quota": quota,
                    "usage_percentage": (calls / quota) * 100 if quota > 0 else 0,
                    "available_tokens": self.tokens[priority],
                    "current_delay": self.current_delays[priority],
                }

            return stats


def profile(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to run cProfile on a function and print stats.

    Args:
        func: Function to profile

    Returns:
        Decorated function that performs profiling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            # Print profile stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(20)  # Print top 20 entries
            print(s.getvalue())

    return wrapper


def profile_memory(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to profile memory usage of a function.

    Args:
        func: Function to profile

    Returns:
        Decorated function that performs memory profiling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            stats = profiler.stop()
            print("\n=== Memory Profile ===")
            print(f"Total memory change: {stats.get('total_diff_mb', 0):.2f} MB")
            print(TOP_MEMORY_CONSUMERS_HEADER)
            for item in stats.get("top_consumers", []):
                print(
                    f"{item['file']}:{item['line']} - {item['size_diff_kb']:.2f} KB ({item['count_diff']} objects)"
                )

    return wrapper


async def adaptive_fetch(
    items: List[T],
    fetch_func: Callable[[T], "asyncio.coroutine[Any, Any, R]"],
    initial_concurrency: int = 5,
    max_concurrency: int = 15,
    performance_monitor_interval: int = 10,
    batch_size: int = None,
    priority_items: Optional[List[T]] = None,
) -> Dict[T, R]:
    """
    Fetch data with adaptive concurrency based on performance.

    This function dynamically adjusts concurrency based on success rates
    and response times to optimize throughput.

    Args:
        items: List of items to process
        fetch_func: Async function to fetch data for each item
        initial_concurrency: Starting concurrency level
        max_concurrency: Maximum concurrency level
        performance_monitor_interval: Number of items to process before adjusting concurrency
        batch_size: Size of each batch (defaults to initial_concurrency)
        priority_items: List of high-priority items to process first

    Returns:
        Dictionary mapping items to results
    """
    if not items:
        return {}

    batch_size = batch_size or initial_concurrency
    results: Dict[T, R] = {}
    errors = 0
    total_processed = 0
    current_concurrency = initial_concurrency

    # Prioritize items if specified
    if priority_items:
        # Move priority items to the front of the queue
        priority_set = set(priority_items)
        prioritized_items = [item for item in items if item in priority_set]
        remaining_items = [item for item in items if item not in priority_set]
        items_queue = prioritized_items + remaining_items
    else:
        items_queue = list(items)

    # Process in batches
    for i in range(0, len(items_queue), batch_size):
        batch = items_queue[i : i + batch_size]

        # Fetch data for batch with current concurrency
        time.time()
        coroutines = [fetch_func(item) for item in batch]

        try:
            from ..utils.async_utils.enhanced import gather_with_concurrency

            batch_results = await gather_with_concurrency(current_concurrency, *coroutines)

            # Process results
            for j, item in enumerate(batch):
                if j < len(batch_results):
                    if isinstance(batch_results[j], Exception):
                        errors += 1
                    else:
                        results[item] = batch_results[j]
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            errors += len(batch)

        time.time()
        total_processed += len(batch)

        # Adjust concurrency based on performance
        if total_processed % performance_monitor_interval == 0:
            success_rate = 1 - (errors / total_processed)

            # Increase concurrency if success rate is high and we're not at max
            if success_rate > 0.95 and current_concurrency < max_concurrency:
                current_concurrency = min(current_concurrency + 1, max_concurrency)
                logger.debug(
                    f"Increased concurrency to {current_concurrency} (success rate: {success_rate:.2f})"
                )

            # Decrease concurrency if success rate is low
            elif success_rate < 0.75 and current_concurrency > 1:
                current_concurrency = max(current_concurrency - 1, 1)
                logger.debug(
                    f"Decreased concurrency to {current_concurrency} (success rate: {success_rate:.2f})"
                )

    return results


async def prioritized_batch_process(
    items: List[T],
    processor: Callable[[T], "asyncio.coroutine[Any, Any, R]"],
    priority_func: Optional[Callable[[T], str]] = None,
    concurrency: int = 5,
    rate_limiter: Optional[PriorityAsyncRateLimiter] = None,
    timeout: Optional[float] = None,
) -> Dict[T, Union[R, Exception]]:
    """
    Process items in batches with priority-based rate limiting.

    Args:
        items: List of items to process
        processor: Async function to process each item
        priority_func: Function that returns priority tier (HIGH, MEDIUM, LOW) for each item
        concurrency: Maximum concurrent operations
        rate_limiter: Priority rate limiter instance (created if not provided)
        timeout: Maximum time in seconds for each operation

    Returns:
        Dictionary mapping items to results or exceptions
    """
    if not items:
        return {}

    # Create rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = PriorityAsyncRateLimiter()

    # Default priority function maps everything to MEDIUM
    if priority_func is None:
        priority_func = lambda _: "MEDIUM"

    # Group items by priority
    priorities: Dict[str, List[T]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for item in items:
        priority = priority_func(item)
        if priority not in priorities:
            priority = "MEDIUM"  # Default to MEDIUM for invalid priorities
        priorities[priority].append(item)

    # Process each priority group
    results: Dict[T, Union[R, Exception]] = {}

    # Process high priority first, then medium, then low
    for priority in ["HIGH", "MEDIUM", "LOW"]:
        priority_items = priorities[priority]
        if not priority_items:
            continue

        # Create processing function that respects rate limiting
        async def process_with_rate_limit(item: T) -> R:
            try:
                # Wait for rate limiting
                await rate_limiter.wait(
                    ticker=str(item) if hasattr(item, "__str__") else None, priority=priority
                )

                # Process with timeout if specified
                if timeout:
                    result = await asyncio.wait_for(processor(item), timeout=timeout)
                else:
                    result = await processor(item)

                # Update rate limiter with success
                await rate_limiter.update_delay(priority, True)
                return result
            except Exception as e:
                # Update rate limiter with failure
                await rate_limiter.update_delay(priority, False)
                return e

        # Process items with concurrency control
        coroutines = [process_with_rate_limit(item) for item in priority_items]
        from ..utils.async_utils.enhanced import gather_with_concurrency

        priority_results = await gather_with_concurrency(concurrency, *coroutines)

        # Add results to overall results dictionary
        for i, item in enumerate(priority_items):
            if i < len(priority_results):
                results[item] = priority_results[i]

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for the yahoofinance package"
    )
    parser.add_argument(
        "--provider",
        choices=["yahoo", "yahooquery", "hybrid", "all"],
        default="all",
        help="Provider to benchmark (default: all)",
    )
    parser.add_argument(
        "--ticker-count",
        type=int,
        default=5,
        help="Number of tickers to use in benchmarks (default: 5)",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations to run (default: 3)"
    )
    parser.add_argument("--batch-benchmark", action="store_true", help="Run batch size benchmarks")
    parser.add_argument(
        "--concurrency-benchmark", action="store_true", help="Run concurrency benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as baselines")
    parser.add_argument(
        "--memory-check", action="store_true", help="Run memory leak detection tests"
    )
    parser.add_argument(
        "--priority-test", action="store_true", help="Test priority-based rate limiting"
    )

    args = parser.parse_args()

    if args.memory_check:
        # Run memory leak detection
        async def test_memory():
            provider = get_provider(async_api=True)
            print("\n=== Testing Memory Usage in Async Provider ===")

            # We'll test both sync and async methods
            print("\nTesting get_ticker_info method...")
            is_leaking, stats = await find_memory_leaks_async(
                provider.get_ticker_info, "AAPL", iterations=10
            )

            # Print detailed report
            if is_leaking:
                print("\n⚠️ MEMORY LEAK DETECTED ⚠️")
                print(f"Memory increased by {stats['memory_diff_mb']:.2f} MB over 10 iterations")
                print(
                    f"Memory growth per iteration: {stats['memory_growth_per_iteration_mb']:.2f} MB"
                )
                print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")
                print(f"Growth percentage: {stats['memory_growth_percent']:.1f}%")

                # Show top memory consumers
                print("\nTop memory consumers:")
                for i, item in enumerate(stats["top_consumers"][:5], 1):
                    print(
                        f"#{i}: {item['file']}:{item['line']} - {item['size_kb']:.1f} KB ({item['count_diff']} objects)"
                    )
            else:
                print("\n✅ No significant memory leaks detected")
                print(f"Total memory change: {stats['memory_diff_mb']:.2f} MB")
                print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")

            # Test batch operation
            print("\nTesting batch_get_ticker_info method...")
            is_leaking, stats = await find_memory_leaks_async(
                provider.batch_get_ticker_info,
                ["AAPL", "MSFT", "GOOG", "AMZN", "META"],
                iterations=5,
            )

            # Print detailed report
            if is_leaking:
                print("\n⚠️ MEMORY LEAK DETECTED IN BATCH PROCESSING ⚠️")
                print(f"Memory increased by {stats['memory_diff_mb']:.2f} MB over 5 iterations")
                print(
                    f"Memory growth per iteration: {stats['memory_growth_per_iteration_mb']:.2f} MB"
                )
                print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")
                print(f"Growth percentage: {stats['memory_growth_percent']:.1f}%")

                # Show top memory consumers
                print("\nTop memory consumers:")
                for i, item in enumerate(stats["top_consumers"][:5], 1):
                    print(
                        f"#{i}: {item['file']}:{item['line']} - {item['size_kb']:.1f} KB ({item['count_diff']} objects)"
                    )
            else:
                print("\n✅ No significant memory leaks detected in batch processing")
                print(f"Total memory change: {stats['memory_diff_mb']:.2f} MB")
                print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")

        asyncio.run(test_memory())
    elif args.priority_test:
        # Test priority-based rate limiting with simplified implementation
        async def test_priority():
            limiter = PriorityAsyncRateLimiter()
            print("\n=== Testing PriorityAsyncRateLimiter ===")

            # Configure for test
            limiter.high_priority_quota = 10
            limiter.medium_priority_quota = 5
            limiter.low_priority_quota = 3

            # Initialize delays for test
            limiter.current_delays = {
                "HIGH": 0.1,  # 100ms
                "MEDIUM": 0.2,  # 200ms
                "LOW": 0.4,  # 400ms
            }

            # Simplified test
            print("Testing priority tiers with short delays...")

            # Test high priority
            print("HIGH priority:", end=" ")
            high_delay = await limiter.wait(priority="HIGH")
            print(f"delay={high_delay:.3f}s")

            # Test medium priority
            print("MEDIUM priority:", end=" ")
            medium_delay = await limiter.wait(priority="MEDIUM")
            print(f"delay={medium_delay:.3f}s")

            # Test low priority
            print("LOW priority:", end=" ")
            low_delay = await limiter.wait(priority="LOW")
            print(f"delay={low_delay:.3f}s")

            # Test region-specific delays
            print("\nTesting region-specific delays:")

            # US ticker
            print("US ticker (AAPL):", end=" ")
            us_delay = await limiter.wait(ticker="AAPL", priority="MEDIUM")
            print(f"delay={us_delay:.3f}s")

            # European ticker
            print("European ticker (BMW.DE):", end=" ")
            eu_delay = await limiter.wait(ticker="BMW.DE", priority="MEDIUM")
            print(f"delay={eu_delay:.3f}s")

            # Asian ticker
            print("Asian ticker (9988.HK):", end=" ")
            asia_delay = await limiter.wait(ticker="9988.HK", priority="MEDIUM")
            print(f"delay={asia_delay:.3f}s")

            # Print statistics
            stats = await limiter.get_statistics()
            print("\nRate limiter statistics:")
            for priority, data in stats.items():
                print(
                    f"{priority}: {data['calls_in_window']}/{data['quota']} calls "
                    f"({data['usage_percentage']:.1f}%), {data['available_tokens']:.1f} tokens"
                )

        asyncio.run(test_priority())
    elif args.all:
        asyncio.run(run_all_benchmarks(save_as_baseline=args.save_baseline))
    elif args.provider != "all":
        asyncio.run(
            benchmark_provider(
                args.provider, ticker_count=args.ticker_count, iterations=args.iterations
            )
        )
    elif args.batch_benchmark:
        asyncio.run(run_batch_size_benchmark(ticker_count=args.ticker_count))
    elif args.concurrency_benchmark:
        asyncio.run(run_concurrency_benchmark(ticker_count=args.ticker_count))
    else:
        print("No benchmark selected. Use --help for options.")

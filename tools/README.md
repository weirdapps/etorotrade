# Tools Directory

This directory contains development tools and utility scripts for the etorotrade project.

## Core Scripts

### Performance & Benchmarking
- `performance_benchmark.py` - **NEW (2025-01-06)** Comprehensive performance benchmarking with real-time metrics
- `run_enhanced_monitoring.py` - Enhanced monitoring with health endpoints and structured logging
- `run_monitoring.py` - Basic system monitoring script

### Portfolio Management
- `download_portfolio_data.py` - Download historical portfolio data for analysis
- `run_optimizer.py` - Run trading criteria optimization with backtesting
- `optimize_criteria.py` - Advanced criteria optimization with multiple strategies

### Development Tools
- `lint.sh` - Run code quality checks (black, isort, flake8, mypy)
- `cleanup.sh` - Clean up temporary files and caches
- `fix_ticker_formats.py` - Fix ticker format inconsistencies
- `split_etoro_by_region.py` - Split eToro data by geographic regions (US/Europe/China)
- `test_optimize.py` - Test optimization functionality

## Usage

Run scripts from the project root directory:

```bash
# Performance Benchmarking (NEW)
python tools/performance_benchmark.py

# Comprehensive performance benchmark
# Shows API throughput, DataFrame operations, memory usage
# Example output: 390 tickers/minute (127% improvement over baseline)

# Enhanced Monitoring  
python tools/run_enhanced_monitoring.py --timeout 300 --health-port 8081

# Portfolio Optimization
python tools/run_optimizer.py --min 1000 --max 25000 --periods 1 3 5 --use-cache
python tools/optimize_criteria.py --mode optimize --period 2y --metric sharpe_ratio

# Development Tools
./tools/lint.sh          # Check code quality
./tools/lint.sh fix      # Auto-fix formatting issues
python tools/split_etoro_by_region.py  # Split eToro data by region
```

## Performance Improvements (2025-01-06)

The tools directory includes new performance monitoring capabilities:

- **`performance_benchmark.py`**: Real-time performance metrics showing 127% API processing improvement
- **Enhanced monitoring**: Health endpoints and structured logging for production use
- **Optimization tools**: Criteria tuning with backtesting and validation

## Parameter Files

- `sample_parameters.json` - Sample parameter configuration
- `sample_parameters_simple.json` - Simplified parameter configuration  
- `working_params.json` - Current working parameters
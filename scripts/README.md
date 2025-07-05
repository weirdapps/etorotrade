# Scripts Directory

This directory contains utility scripts organized by function:

## Directory Structure

- `analysis/` - Scripts for analyzing trading recommendations and criteria changes
- `monitoring/` - Scripts for monitoring system performance and health
- `optimization/` - Scripts for optimizing trading parameters and strategies
- `utilities/` - General utility scripts for data processing and debugging

## Analysis Scripts

- `analyze_recommendation_changes.py` - Compare trading recommendations before/after criteria changes
- `analyze_stricter_criteria.py` - Analyze impact of stricter trading criteria
- `summarize_changes.py` - Summarize changes in trading recommendations

## Utilities Scripts  

- `check_buy_csv.py` - Validate assets in buy.csv against criteria
- `debug_eden.py` - Debug specific ticker issues
- `generate_buy_table.py` - Generate formatted buy recommendation tables

## Other Scripts

- `cleanup.sh` - Clean up temporary files and caches
- `download_portfolio_data.py` - Download portfolio data
- `fix_ticker_formats.py` - Fix ticker format inconsistencies
- `lint.sh` - Run linting and code quality checks
- `run_enhanced_monitoring.py` - Run enhanced monitoring with detailed metrics
- `run_monitoring.py` - Basic monitoring script
- `run_optimizer.py` - Run trading criteria optimization
- `split_etoro_by_region.py` - Split eToro data by geographic region
- `test_optimize.py` - Test optimization functionality

## Usage

Run scripts from the project root directory:

```bash
# Analysis
python scripts/analysis/analyze_recommendation_changes.py

# Utilities  
python scripts/utilities/check_buy_csv.py

# Monitoring
python scripts/run_monitoring.py
```

## Parameter Files

- `sample_parameters.json` - Sample parameter configuration
- `sample_parameters_simple.json` - Simplified parameter configuration  
- `working_params.json` - Current working parameters
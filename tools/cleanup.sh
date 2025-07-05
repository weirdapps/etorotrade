#!/bin/bash

# Project cleanup script for etorotrade
# This script cleans up temporary files, cache files, and build artifacts

echo "🧹 Starting etorotrade project cleanup..."

# Clean Python build artifacts
echo "Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Clean monitoring files (auto-generated)
echo "Cleaning monitoring files..."
rm -f yahoofinance/data/monitoring/health_*.json 2>/dev/null || true
rm -f yahoofinance/data/monitoring/metrics_*.json 2>/dev/null || true

# Clean cache and temporary data files
echo "Cleaning cache files..."
rm -f yahoofinance/data/portfolio_cache.pkl 2>/dev/null || true
rm -f yahoofinance/data/portfolio_prices.json 2>/dev/null || true
rm -f yahoofinance/data/circuit_state.json 2>/dev/null || true

# Clean logs (optional - keep recent logs)
if [ "$1" = "--clean-logs" ]; then
    echo "Cleaning log files..."
    rm -f logs/yahoofinance.log 2>/dev/null || true
fi

# Clean test artifacts
echo "Cleaning test artifacts..."
rm -rf .pytest_cache 2>/dev/null || true
rm -f .coverage 2>/dev/null || true

# Clean any temporary files created during development
echo "Cleaning temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "test_*.py" -path "./test_*" -delete 2>/dev/null || true

echo "✅ Cleanup completed!"
echo ""
echo "Kept important files:"
echo "  - Configuration files (*.toml, *.ini, requirements.txt)"
echo "  - Source code and documentation"
echo "  - Static assets and input data"
echo "  - Example output files for reference"
echo ""
echo "To also clean log files, run: ./scripts/cleanup.sh --clean-logs"
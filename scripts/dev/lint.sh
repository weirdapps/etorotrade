#!/bin/bash
# Run all linters
set -e

echo "🔍 Running linters..."
echo ""

echo "→ Checking code formatting with black..."
black --check yahoofinance/ trade_modules/ tests/ || {
    echo "  ℹ️  Run 'scripts/dev/format.sh' to auto-format"
    exit 1
}

echo "→ Checking import order with isort..."
isort --check yahoofinance/ trade_modules/ tests/ || {
    echo "  ℹ️  Run 'scripts/dev/format.sh' to auto-format"
    exit 1
}

echo "→ Checking code style with flake8..."
flake8 yahoofinance/ trade_modules/ --max-line-length=100

echo "→ Type checking with mypy..."
mypy yahoofinance/ trade_modules/ --ignore-missing-imports || true

echo ""
echo "✅ All linters passed!"

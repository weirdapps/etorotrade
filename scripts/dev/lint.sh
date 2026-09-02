#!/bin/bash
# Run all linters
set -e

echo "🔍 Running linters..."
echo ""

echo "→ Checking formatting with ruff..."
ruff format --check yahoofinance/ trade_modules/ tests/ || {
    echo "  ℹ️  Run 'scripts/dev/format.sh' to auto-format"
    exit 1
}

echo "→ Linting with ruff (includes import order)..."
ruff check yahoofinance/ trade_modules/ tests/ || {
    echo "  ℹ️  Run 'scripts/dev/format.sh' to auto-fix what is fixable"
    exit 1
}

echo "→ Type checking with mypy..."
mypy yahoofinance/ trade_modules/ --ignore-missing-imports || true

echo ""
echo "✅ All linters passed!"

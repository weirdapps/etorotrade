#!/bin/bash
# Auto-format code

echo "🎨 Formatting code..."
echo ""

echo "→ Fixing lint issues with ruff (includes import order)..."
ruff check --fix yahoofinance/ trade_modules/ tests/

echo "→ Formatting with ruff..."
ruff format yahoofinance/ trade_modules/ tests/

echo ""
echo "✅ Code formatted!"

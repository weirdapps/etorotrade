#!/bin/bash
# Auto-format code

echo "🎨 Formatting code..."
echo ""

echo "→ Formatting with black..."
black yahoofinance/ trade_modules/ tests/

echo "→ Sorting imports with isort..."
isort yahoofinance/ trade_modules/ tests/

echo ""
echo "✅ Code formatted!"

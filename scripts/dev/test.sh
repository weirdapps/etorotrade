#!/bin/bash
# Run tests with common options

echo "🧪 Running test suite..."

pytest tests/ \
  --cov=yahoofinance \
  --cov=trade_modules \
  --cov-report=html \
  --cov-report=term \
  --cov-fail-under=41 \
  -v \
  "$@"  # Pass additional args from command line

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed"
fi

exit $exit_code

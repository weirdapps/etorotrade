#!/bin/bash
# Test runner script for etorotrade

set -e  # Exit on error

# Display usage information
display_help() {
    echo "Usage: ./run_tests.sh [OPTION]"
    echo "Run various tests and benchmarks for etorotrade."
    echo ""
    echo "Options:"
    echo "  --all             Run all tests and benchmarks"
    echo "  --unit            Run only unit tests"
    echo "  --integration     Run integration tests"
    echo "  --performance     Run performance benchmarks"
    echo "  --memory          Run memory leak tests"
    echo "  --priority        Run priority limiter tests"
    echo "  --monitoring      Run monitoring system tests"
    echo "  --help            Display this help message"
    echo ""
}

# Default to running all tests if no arguments provided
if [ $# -eq 0 ]; then
    run_all=true
else
    run_all=false
fi

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            run_all=true
            ;;
        --unit)
            run_unit=true
            ;;
        --integration)
            run_integration=true
            ;;
        --performance)
            run_performance=true
            ;;
        --memory)
            run_memory=true
            ;;
        --priority)
            run_priority=true
            ;;
        --monitoring)
            run_monitoring=true
            ;;
        --help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
    shift
done

# Run all tests if --all flag is set
if [ "$run_all" = true ]; then
    run_unit=true
    run_integration=true
    run_performance=true
    run_memory=true
    run_priority=true
    run_monitoring=true
fi

# Execute tests based on flags

# Unit tests
if [ "$run_unit" = true ]; then
    echo "===== Running unit tests ====="
    python -m pytest tests/unit/ -v
    echo "===== Unit tests completed ====="
    echo ""
fi

# Integration tests
if [ "$run_integration" = true ]; then
    echo "===== Running integration tests ====="
    python -m pytest tests/integration/ -v
    echo "===== Integration tests completed ====="
    echo ""
fi

# Performance benchmarks
if [ "$run_performance" = true ]; then
    echo "===== Running performance benchmarks ====="
    python -m yahoofinance.analysis.benchmarking --provider hybrid --ticker-count 3 --iterations 2
    echo "===== Performance benchmarks completed ====="
    echo ""
fi

# Memory leak tests
if [ "$run_memory" = true ]; then
    echo "===== Running memory leak tests ====="
    python benchmarks/test_memory_leak.py
    echo "===== Memory leak tests completed ====="
    echo ""
fi

# Priority limiter tests
if [ "$run_priority" = true ]; then
    echo "===== Running priority limiter tests ====="
    python benchmarks/test_priority_limiter.py
    echo "===== Priority limiter tests completed ====="
    echo ""
fi

# Monitoring system tests
if [ "$run_monitoring" = true ]; then
    echo "===== Running monitoring system tests ====="
    python -m pytest tests/unit/core/test_monitoring.py -v
    python -m pytest tests/unit/api/middleware/test_monitoring_middleware.py -v
    # Generate example monitoring data
    python scripts/monitoring_examples.py
    echo "===== Monitoring system tests completed ====="
    echo ""
fi

echo "All tests completed successfully!"
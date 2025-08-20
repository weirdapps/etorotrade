#!/usr/bin/env python3
"""
Test runner for TradingEngine comprehensive tests.

This script runs all the comprehensive tests for the TradingEngine
and provides a summary of what behaviors are being validated.

Run with: python test_runner.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all TradingEngine tests and provide summary."""
    
    print("=" * 80)
    print("TRADING ENGINE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    print("🎯 PURPOSE: Catch ANY behavioral changes during TradingEngine refactoring")
    print("📊 COVERAGE: God object split regression prevention")
    print()
    
    test_files = [
        "test_trade_engine_comprehensive.py",
        "test_trade_engine_integration.py"
    ]
    
    test_descriptions = {
        "test_trade_engine_comprehensive.py": [
            "✅ analyze_market_opportunities workflow with BS/ACT columns",
            "✅ All filtering methods (_filter_buy/sell/hold_opportunities)",
            "✅ Portfolio integration logic (_apply_portfolio_filters)",
            "✅ Async batch processing (process_ticker_batch)",
            "✅ Confidence score calculations with various inputs",
            "✅ Notrade filtering with ticker equivalence",
            "✅ Trading signal calculation vs existing BS column",
            "✅ PositionSizer calculations and constraints",
            "✅ Factory functions and error handling"
        ],
        "test_trade_engine_integration.py": [
            "✅ Real CSV data structure integration",
            "✅ Portfolio filtering with column variations",
            "✅ International ticker format handling",
            "✅ Data validation with mixed/missing data",
            "✅ Performance with large datasets",
            "✅ Complete end-to-end workflows"
        ]
    }
    
    print("🧪 TEST CATEGORIES:")
    print()
    
    for test_file, descriptions in test_descriptions.items():
        print(f"📁 {test_file}:")
        for desc in descriptions:
            print(f"   {desc}")
        print()
    
    print("🚀 RUNNING TESTS...")
    print("=" * 80)
    
    # Run tests
    test_dir = Path(__file__).parent
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n📋 Running {test_file}...")
            print("-" * 60)
            
            # Run pytest on the specific file
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_path), 
                "-v", 
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode == 0:
                print(f"✅ {test_file} - ALL TESTS PASSED")
                # Count passed tests from output
                passed_count = result.stdout.count(" PASSED")
                total_passed += passed_count
                print(f"   Passed: {passed_count} tests")
            else:
                print(f"❌ {test_file} - SOME TESTS FAILED")
                # Count failed tests from output
                failed_count = result.stdout.count(" FAILED")
                passed_count = result.stdout.count(" PASSED")
                total_failed += failed_count
                total_passed += passed_count
                print(f"   Passed: {passed_count}, Failed: {failed_count}")
                print("\nFailure details:")
                print(result.stdout)
                print(result.stderr)
        else:
            print(f"❌ {test_file} not found!")
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ TradingEngine behavior is fully validated")
        print("✅ Safe to proceed with refactoring")
        return True
    else:
        print(f"\n⚠️  {total_failed} TESTS FAILED!")
        print("❌ DO NOT proceed with refactoring until all tests pass")
        print("❌ Fix failing tests first to establish baseline behavior")
        return False


def validate_test_coverage():
    """Validate that tests cover all critical TradingEngine methods."""
    
    print("\n🔍 VALIDATING TEST COVERAGE...")
    print("-" * 40)
    
    # Critical methods that MUST be tested
    critical_methods = [
        "analyze_market_opportunities",
        "_filter_buy_opportunities", 
        "_filter_sell_opportunities",
        "_filter_hold_opportunities",
        "_apply_portfolio_filters",
        "_calculate_confidence_score",
        "_filter_notrade_tickers",
        "_calculate_trading_signals",
        "process_ticker_batch",
        "_process_batch",
        "_process_single_ticker"
    ]
    
    test_files_content = []
    test_dir = Path(__file__).parent
    
    for test_file in ["test_trade_engine_comprehensive.py", "test_trade_engine_integration.py"]:
        test_path = test_dir / test_file
        if test_path.exists():
            with open(test_path, 'r') as f:
                test_files_content.append(f.read())
    
    all_content = '\n'.join(test_files_content)
    
    missing_coverage = []
    for method in critical_methods:
        if method not in all_content:
            missing_coverage.append(method)
    
    if missing_coverage:
        print("❌ Missing test coverage for:")
        for method in missing_coverage:
            print(f"   - {method}")
        return False
    else:
        print("✅ All critical methods have test coverage")
        return True


if __name__ == "__main__":
    print("Starting TradingEngine comprehensive test validation...")
    
    # Validate coverage
    coverage_ok = validate_test_coverage()
    
    if coverage_ok:
        # Run tests
        tests_passed = run_tests()
        
        if tests_passed:
            print("\n🏆 SUCCESS: TradingEngine is ready for refactoring!")
            print("📋 All behavioral tests pass - proceed with confidence")
            sys.exit(0)
        else:
            print("\n💥 FAILURE: Tests must pass before refactoring!")
            sys.exit(1)
    else:
        print("\n💥 FAILURE: Insufficient test coverage!")
        sys.exit(1)
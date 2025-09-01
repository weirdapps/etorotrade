#!/usr/bin/env python3
"""
eToro Trade Analysis Tool - Command Line Interface

A comprehensive investment analysis system providing data-driven trading decisions
through advanced financial modeling and analyst consensus analysis.
"""

import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
# Specifically suppress yfinance deprecation warnings
warnings.filterwarnings("ignore", message=".*Ticker.earnings.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Net Income.*", category=DeprecationWarning)

# Configure logging early
from yahoofinance.core.logging import suppress_yfinance_noise
suppress_yfinance_noise()

# Import main CLI module
from trade_modules.trade_cli import main, setup_secure_file_copy, config_validator

if __name__ == "__main__":
    # Handle special validation-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-config":
        is_valid = config_validator.print_validation_report()
        sys.exit(0 if is_valid else 1)
    
    try:
        # Run configuration validation first
        if not config_validator.print_validation_report():
            sys.exit(1)
        
        # Setup secure file operations
        setup_secure_file_copy()

        # Run the main function
        main()
        
    except Exception as e:
        print(f"Critical error: {str(e)}", file=sys.stderr)
        sys.exit(1)
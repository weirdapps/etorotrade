#!/usr/bin/env python3
"""
Convenient wrapper for manual ticker input.
Usage: python i TICKER1 TICKER2 ...
"""

import sys
import os
from pathlib import Path

# Add the directory containing trade.py to Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import and run the main function with manual input arguments
if __name__ == "__main__":
    # Reconstruct arguments as if called with "python trade.py i ..."
    sys.argv = ["trade.py", "i"] + sys.argv[1:]
    
    # Import and run trade
    import trade
    trade.main()
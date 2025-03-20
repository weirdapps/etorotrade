#!/usr/bin/env python3
"""
Test script for trade2.py forcing MSFT to use earnings-based ratings
"""

import sys
from trade2 import main

# This will run before importing the module
print("Patching MSFT to use earnings-based ratings...")
sys.modules['trade2']._ratings_cache = {
    'MSFT': {
        "buy_percentage": 90.0,
        "total_ratings": 15,
        "ratings_type": "E"
    }
}

# Run the main function
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example of using the performance tracking functionality in yahoofinance_v2.

This example demonstrates the new performance.py module for tracking market
index performance and portfolio performance from external sources.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the performance functions
from yahoofinance_v2.analysis.performance import (
    track_index_performance,
    track_portfolio_performance,
    track_performance_async
)

def main():
    """Main entry point for the performance tracking example."""
    print("\n=== PERFORMANCE TRACKING EXAMPLE ===")
    
    # Ask user what to track
    while True:
        choice = input("\nWhat would you like to track?\n"
                      "1. Weekly Market Index Performance\n"
                      "2. Monthly Market Index Performance\n"
                      "3. Portfolio Performance\n"
                      "4. Both (Asynchronously)\n"
                      "5. Quit\n"
                      "Enter choice (1-5): ")
        
        try:
            choice = int(choice.strip())
        except ValueError:
            print("Please enter a number between 1 and 5.")
            continue
        
        if choice == 1:
            print("\nTracking weekly market performance...")
            track_index_performance(period_type="weekly")
            
        elif choice == 2:
            print("\nTracking monthly market performance...")
            track_index_performance(period_type="monthly")
            
        elif choice == 3:
            print("\nTracking portfolio performance...")
            # Get portfolio URL
            portfolio_url = input("Enter portfolio URL (or press Enter for default): ").strip()
            if not portfolio_url:
                portfolio_url = "https://bullaware.com/etoro/plessas"
                
            track_portfolio_performance(url=portfolio_url)
            
        elif choice == 4:
            print("\nTracking both market and portfolio performance asynchronously...")
            # Get period type
            period_choice = input("Select period type (W for weekly, M for monthly): ").strip().upper()
            period_type = "weekly" if period_choice == "W" else "monthly"
            
            # Get portfolio URL
            portfolio_url = input("Enter portfolio URL (or press Enter for default): ").strip()
            if not portfolio_url:
                portfolio_url = "https://bullaware.com/etoro/plessas"
                
            # Run async tracking
            asyncio.run(track_performance_async(period_type=period_type, portfolio_url=portfolio_url))
            
        elif choice == 5:
            print("\nExiting performance tracking example.")
            break
            
        else:
            print("Please enter a number between 1 and 5.")
            
        # Ask if user wants to continue
        cont = input("\nPress Enter to continue or 'q' to quit: ")
        if cont.lower() == 'q':
            break

if __name__ == "__main__":
    main()
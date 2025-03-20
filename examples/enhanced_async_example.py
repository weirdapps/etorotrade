#!/usr/bin/env python3
"""
Example of using the enhanced async Yahoo Finance API.

This example demonstrates how to use the enhanced async API with true async I/O,
circuit breaking, and improved resilience patterns.
"""

import asyncio
import logging
import sys
import os
import time
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yahoofinance_v2.api.providers.enhanced_async_yahoo_finance import EnhancedAsyncYahooFinanceProvider
from yahoofinance_v2.utils.network.circuit_breaker import get_all_circuits, reset_all_circuits
from yahoofinance_v2.utils.async_utils.enhanced import process_batch_async

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def compare_tickers(tickers: List[str]):
    """
    Compare multiple tickers using the enhanced async provider.
    
    Args:
        tickers: List of ticker symbols to compare
    """
    # Create enhanced async provider
    provider = EnhancedAsyncYahooFinanceProvider(max_concurrency=5)
    
    # Fetch data for all tickers in parallel
    logger.info(f"Fetching data for {len(tickers)} tickers...")
    start_time = time.time()
    
    # Use the batch method to fetch all ticker data at once
    results = await provider.batch_get_ticker_info(tickers)
    
    end_time = time.time()
    logger.info(f"Fetched data in {end_time - start_time:.2f} seconds")
    
    # Display comparison
    print("\n=== TICKER COMPARISON ===")
    print(f"{'Symbol':<6}  {'Name':<20}  {'Price':>8}  {'Target':>8}  {'Upside':>8}  {'P/E':>6}  {'Beta':>6}")
    print("-" * 70)
    
    for ticker in tickers:
        data = results.get(ticker, {})
        if data and "error" not in data:
            name = data.get("name", "")[:18] + ".." if len(data.get("name", "")) > 20 else data.get("name", "")
            price = data.get("current_price")
            target = data.get("target_price")
            upside = data.get("upside")
            pe = data.get("pe_forward")
            beta = data.get("beta")
            
            print(f"{ticker:<6}  {name:<20}  {price:>8.2f}  {target:>8.2f}  {upside:>7.1f}%  {pe:>6.1f}  {beta:>6.2f}")
        else:
            error = data.get("error", "Unknown error") if data else "No data"
            print(f"{ticker:<6}  ERROR: {error}")
    
    # Display circuit breaker status
    circuits = get_all_circuits()
    print("\n=== CIRCUIT BREAKER STATUS ===")
    for name, metrics in circuits.items():
        state = metrics.get("state", "UNKNOWN")
        success_rate = 100 - metrics.get("failure_rate", 0)
        total_requests = metrics.get("total_requests", 0)
        print(f"Circuit '{name}': {state} (Success rate: {success_rate:.1f}%, Requests: {total_requests})")
    
    # Clean up
    await provider.close()

async def fetch_sector_metrics(sector_tickers: Dict[str, List[str]]):
    """
    Fetch and calculate metrics for different market sectors.
    
    Args:
        sector_tickers: Dictionary mapping sector names to lists of ticker symbols
    """
    # Create enhanced async provider
    provider = EnhancedAsyncYahooFinanceProvider(max_concurrency=5)
    
    # Process sectors in parallel
    async def analyze_sector(sector_name: str, tickers: List[str]) -> Dict[str, Any]:
        logger.info(f"Analyzing {sector_name} sector ({len(tickers)} tickers)...")
        
        # Fetch data for all tickers in this sector
        data = await provider.batch_get_ticker_info(tickers)
        
        # Calculate sector metrics
        valid_data = {k: v for k, v in data.items() if v and "error" not in v}
        
        pe_values = [d.get("pe_forward") for d in valid_data.values() 
                     if d.get("pe_forward") is not None and d.get("pe_forward") > 0]
        
        beta_values = [d.get("beta") for d in valid_data.values() 
                       if d.get("beta") is not None]
        
        upside_values = [d.get("upside") for d in valid_data.values() 
                         if d.get("upside") is not None]
        
        market_caps = [d.get("market_cap") for d in valid_data.values() 
                       if d.get("market_cap") is not None]
        
        # Return sector analysis
        return {
            "sector": sector_name,
            "tickers_analyzed": len(valid_data),
            "avg_pe": sum(pe_values) / len(pe_values) if pe_values else None,
            "avg_beta": sum(beta_values) / len(beta_values) if beta_values else None,
            "avg_upside": sum(upside_values) / len(upside_values) if upside_values else None,
            "total_market_cap": sum(market_caps) if market_caps else None,
            "largest_company": max(
                [(ticker, d.get("market_cap", 0)) for ticker, d in valid_data.items()],
                key=lambda x: x[1],
                default=(None, 0)
            )[0]
        }
    
    # Process all sectors in parallel
    logger.info(f"Analyzing {len(sector_tickers)} sectors...")
    start_time = time.time()
    
    # Create tasks for each sector
    tasks = []
    for sector_name, tickers in sector_tickers.items():
        tasks.append(analyze_sector(sector_name, tickers))
    
    # Run tasks concurrently with our enhanced gather function
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    
    # Display sector comparison
    print("\n=== SECTOR COMPARISON ===")
    print(f"{'Sector':<15}  {'Tickers':>8}  {'Avg P/E':>8}  {'Avg Beta':>8}  {'Avg Upside':>11}  {'Largest':<6}")
    print("-" * 70)
    
    for sector_data in results:
        sector = sector_data['sector']
        count = sector_data['tickers_analyzed']
        pe = sector_data['avg_pe']
        beta = sector_data['avg_beta']
        upside = sector_data['avg_upside']
        largest = sector_data['largest_company']
        
        print(f"{sector:<15}  {count:>8}  {pe:>8.1f}  {beta:>8.2f}  {upside:>10.1f}%  {largest:<6}")
    
    # Clean up
    await provider.close()

async def demonstrate_circuit_breaker():
    """Demonstrate the circuit breaker pattern with simulated failures."""
    # Create enhanced async provider
    provider = EnhancedAsyncYahooFinanceProvider(max_concurrency=5)
    
    # Reset all circuits
    reset_all_circuits()
    
    # Simulate requests with some failures
    async def make_request(ticker: str, should_fail: bool) -> Dict[str, Any]:
        if should_fail:
            # Simulate a failure
            raise Exception(f"Simulated failure for {ticker}")
        else:
            # Make a real request
            return await provider.get_ticker_info(ticker)
    
    # Try a mix of successful and failed requests
    requests = [
        ("AAPL", False),  # Should succeed
        ("MSFT", False),  # Should succeed
        ("GOOG", True),   # Simulate failure
        ("AMZN", True),   # Simulate failure
        ("META", True),   # Simulate failure
        ("TSLA", True),   # Simulate failure
        ("NFLX", True),   # Simulate failure - this should trip the circuit
        ("NVDA", False),  # Should be blocked by circuit breaker
    ]
    
    print("\n=== CIRCUIT BREAKER DEMONSTRATION ===")
    for i, (ticker, should_fail) in enumerate(requests):
        try:
            print(f"Request {i+1}: {ticker} (Simulating {'failure' if should_fail else 'success'})")
            await make_request(ticker, should_fail)
            print(f"  Result: SUCCESS")
        except Exception as e:
            if "circuit" in str(e).lower():
                print(f"  Result: CIRCUIT OPEN - Request rejected by circuit breaker")
            else:
                print(f"  Result: FAILED - {str(e)}")
        
        # Display circuit state after each request
        circuits = get_all_circuits()
        for name, metrics in circuits.items():
            state = metrics.get("state", "UNKNOWN")
            failures = metrics.get("current_failure_count", 0)
            print(f"  Circuit '{name}': {state} (Failures: {failures})")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    # Clean up
    await provider.close()

async def main():
    """Main entry point for the enhanced async example."""
    # Example 1: Compare multiple tickers
    tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    await compare_tickers(tech_tickers)
    
    # Example 2: Analyze different sectors
    sector_tickers = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "INTC"],
        "Finance": ["JPM", "BAC", "GS", "WFC", "C"],
        "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
        "Consumer": ["AMZN", "WMT", "PG", "KO", "PEP"]
    }
    await fetch_sector_metrics(sector_tickers)
    
    # Example 3: Demonstrate circuit breaker pattern
    await demonstrate_circuit_breaker()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
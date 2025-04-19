"""
Simple standalone test script for PriorityAsyncRateLimiter.

This script tests the PriorityAsyncRateLimiter implementation
without relying on the full codebase. It helps verify the
functionality of the rate limiter in isolation.
"""

import asyncio
import time
from collections import deque


class PriorityAsyncRateLimiter:
    """
    Async rate limiter with priority tiers.
    
    This limiter provides different quotas for different priority
    levels (HIGH, MEDIUM, LOW) and region-specific adjustments.
    """
    
    def __init__(self):
        # Initialize priority quotas
        self.high_priority_quota = 10
        self.medium_priority_quota = 5
        self.low_priority_quota = 3
        
        # Track calls per priority tier
        self.priority_call_times = {
            "HIGH": deque(maxlen=20),
            "MEDIUM": deque(maxlen=20),
            "LOW": deque(maxlen=20)
        }
        
        # Current delays per priority tier
        self.current_delays = {
            "HIGH": 0.1,    # 100ms
            "MEDIUM": 0.2,  # 200ms
            "LOW": 0.4      # 400ms
        }
        
        # Thread safety with asyncio lock
        self._lock = asyncio.Lock()
    
    async def wait(self, ticker=None, priority="MEDIUM"):
        """Wait for the appropriate delay based on priority and ticker."""
        async with self._lock:
            # Calculate appropriate delay
            delay = self.current_delays[priority]
            
            # Apply region-specific adjustments if ticker provided
            if ticker:
                if ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".L"):
                    # European markets - slightly higher delay
                    delay *= 1.1
                elif ticker.endswith(".HK") or ticker.endswith(".T") or ticker.endswith(".SS"):
                    # Asian markets - higher delay
                    delay *= 1.2
            
            # Randomize delay slightly (±20%)
            from random import uniform
            jitter = uniform(-0.2, 0.2)
            delay = delay * (1 + jitter)
            
            # Wait for the calculated delay
            await asyncio.sleep(delay)
            
            # Record this call
            self.priority_call_times[priority].append(time.time())
            
            return delay
    
    async def get_statistics(self):
        """Get current statistics for rate limiting."""
        async with self._lock:
            stats = {}
            
            for priority in ["HIGH", "MEDIUM", "LOW"]:
                quota = getattr(self, f"{priority.lower()}_priority_quota")
                calls = len(self.priority_call_times[priority])
                
                stats[priority] = {
                    "calls_in_window": calls,
                    "quota": quota,
                    "usage_percentage": (calls / quota) * 100 if quota > 0 else 0,
                    "available_tokens": quota - calls,
                    "current_delay": self.current_delays[priority],
                }
            
            return stats


async def main():
    """Main test function."""
    limiter = PriorityAsyncRateLimiter()
    print("\n=== Testing PriorityAsyncRateLimiter ===")
    
    # Test different priority tiers
    print("Testing priority tiers with short delays...")
    
    # Test high priority
    print("HIGH priority:", end=" ")
    high_delay = await limiter.wait(priority="HIGH")
    print(f"delay={high_delay:.3f}s")
    
    # Test medium priority
    print("MEDIUM priority:", end=" ")
    medium_delay = await limiter.wait(priority="MEDIUM")
    print(f"delay={medium_delay:.3f}s")
    
    # Test low priority
    print("LOW priority:", end=" ")
    low_delay = await limiter.wait(priority="LOW")
    print(f"delay={low_delay:.3f}s")
    
    # Test region-specific delays
    print("\nTesting region-specific delays:")
    
    # US ticker
    print("US ticker (AAPL):", end=" ")
    us_delay = await limiter.wait(ticker="AAPL", priority="MEDIUM")
    print(f"delay={us_delay:.3f}s")
    
    # European ticker
    print("European ticker (BMW.DE):", end=" ")
    eu_delay = await limiter.wait(ticker="BMW.DE", priority="MEDIUM")
    print(f"delay={eu_delay:.3f}s")
    
    # Asian ticker
    print("Asian ticker (9988.HK):", end=" ")
    asia_delay = await limiter.wait(ticker="9988.HK", priority="MEDIUM")
    print(f"delay={asia_delay:.3f}s")
    
    # Print statistics
    stats = await limiter.get_statistics()
    print("\nRate limiter statistics:")
    for priority, data in stats.items():
        print(f"{priority}: {data['calls_in_window']}/{data['quota']} calls "
              f"({data['usage_percentage']:.1f}%), available: {data['available_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
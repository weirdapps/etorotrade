#!/usr/bin/env python3
"""
Test PP column in actual trade system output
"""

import subprocess
import sys
import os

def test_pp_in_trade_output():
    """Test if PP column appears in trade system output"""
    
    # Create a small test file with just a few stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create temporary market file
    with open('temp_market.csv', 'w') as f:
        f.write('Ticker\n')
        for ticker in test_tickers:
            f.write(f'{ticker}\n')
    
    try:
        # Run the trade system with the test file
        cmd = [sys.executable, 'trade.py', '-o', 'm', '-m', 'temp_market.csv']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Check if PP column appears in the output
            if 'PP' in output:
                print("✓ PP column found in output!")
                
                # Extract lines containing PP column header or data
                lines = output.split('\n')
                for i, line in enumerate(lines):
                    if 'PP' in line or any(ticker in line for ticker in test_tickers):
                        print(f"Line {i}: {line}")
                        # Show a few lines around it for context
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            if j != i:
                                print(f"     {j}: {lines[j]}")
                        break
            else:
                print("✗ PP column not found in output")
                print("First 1000 chars of output:")
                print(output[:1000])
                
        else:
            print(f"Command failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error running command: {e}")
    finally:
        # Clean up
        if os.path.exists('temp_market.csv'):
            os.remove('temp_market.csv')

if __name__ == "__main__":
    test_pp_in_trade_output()
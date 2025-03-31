#!/usr/bin/env python3
"""
Direct HTML generation test with predictable action values
"""

import os
import logging
from yahoofinance.presentation.html import HTMLGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
html_logger = logging.getLogger('yahoofinance.presentation.html')
html_logger.setLevel(logging.DEBUG)

# Set output path
output_dir = '/Users/plessas/SourceCode/etorotrade/yahoofinance/output'

# Create test data with explicit actions
stocks_data = [
    {
        "#": 1,
        "TICKER": "TEST1",
        "COMPANY": "TEST COMPANY 1",
        "PRICE": 100.5,
        "UPSIDE": 25.0,
        "% BUY": 90.0,
        "BETA": 1.2,
        "PET": 15.5,
        "PEF": 14.2,
        "ACTION": "B"  # Buy
    },
    {
        "#": 2,
        "TICKER": "TEST2",
        "COMPANY": "TEST COMPANY 2",
        "PRICE": 50.25,
        "UPSIDE": 5.0,
        "% BUY": 60.0,
        "BETA": 0.8,
        "PET": 12.0,
        "PEF": 10.5,
        "ACTION": "S"  # Sell
    },
    {
        "#": 3,
        "TICKER": "TEST3",
        "COMPANY": "TEST COMPANY 3",
        "PRICE": 75.0,
        "UPSIDE": 15.0,
        "% BUY": 75.0,
        "BETA": 1.0,
        "PET": 18.0,
        "PEF": 16.5,
        "ACTION": "H"  # Hold
    }
]

# Format values as strings to match real data
for record in stocks_data:
    record["UPSIDE"] = f"{record['UPSIDE']:.1f}%"
    record["% BUY"] = f"{record['% BUY']:.0f}%"
    record["BETA"] = f"{record['BETA']:.1f}"
    record["PET"] = f"{record['PET']:.1f}"
    record["PEF"] = f"{record['PEF']:.1f}"
    record["PRICE"] = f"{record['PRICE']:.1f}"

# Generate HTML
print("Generating HTML...")
html_generator = HTMLGenerator(output_dir=output_dir)

# Debug the records
print("\nTest records:")
for record in stocks_data:
    print(f"TICKER: {record['TICKER']}, ACTION: {record['ACTION']}")

# Generate HTML
html_path = html_generator.generate_stock_table(
    stocks_data=stocks_data,
    title="Test Actions HTML",
    output_filename="direct_test"
)

if html_path:
    print(f"HTML generated successfully at {html_path}")
    
    # Also generate a stand-alone HTML file with inline styles
    standalone_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Direct Coloring Test</title>
    <style>
        .buy-row {{ background-color: #e0ffe0; }}
        .sell-row {{ background-color: #ffe0e0; }}
    </style>
</head>
<body>
    <h1>Direct Coloring Test</h1>
    <table border="1">
        <tr><th>Ticker</th><th>Action</th></tr>
        <tr class="buy-row"><td>TEST1</td><td>B</td></tr>
        <tr class="sell-row"><td>TEST2</td><td>S</td></tr>
        <tr><td>TEST3</td><td>H</td></tr>
    </table>
    
    <h1>Inline Style Test</h1>
    <table border="1">
        <tr><th>Ticker</th><th>Action</th></tr>
        <tr style="background-color: #e0ffe0;"><td>TEST1</td><td>B</td></tr>
        <tr style="background-color: #ffe0e0;"><td>TEST2</td><td>S</td></tr>
        <tr><td>TEST3</td><td>H</td></tr>
    </table>
</body>
</html>
"""
    
    # Write standalone HTML
    standalone_path = os.path.join(output_dir, "direct_test_standalone.html")
    with open(standalone_path, "w") as f:
        f.write(standalone_html)
    print(f"Standalone HTML generated at {standalone_path}")
    
else:
    print("Failed to generate HTML")

print("\nHTML generation test completed")
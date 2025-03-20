# Testing Scripts

This directory contains manual testing scripts that are used for debugging and exploration. These are not formal tests and are not run as part of the test suite.

## Available Scripts

- `test_peg.py`: Tests PEG ratio values from Yahoo Finance API
- `test_ticker.py`: Tests a specific ticker with the trade.py application
- `test_display.py`: Tests display fixes for PEG ratio and earnings date
- `test_fields.py`: Directly examines Yahoo Finance API fields and values

## Usage

These scripts can be run directly from the command line:

```
python scripts/test_peg.py
python scripts/test_ticker.py AAPL
python scripts/test_display.py
python scripts/test_fields.py MSFT
```

## Purpose

These scripts are primarily used for:

1. Debugging API issues
2. Examining raw data from the Yahoo Finance API
3. Testing specific fixes or features
4. Manual exploration of API fields

Unlike the formal tests in the `tests/` directory, these scripts:
- Don't use pytest or other test frameworks
- May have interactive components
- Often print detailed output
- Are meant to be run manually rather than as part of automated testing
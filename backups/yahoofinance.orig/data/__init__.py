"""
Data input/output for Yahoo Finance analysis.

This package organizes the storage and retrieval of input data 
(like ticker lists and portfolio data) and output data (like
analysis results and reports).
"""

import os
from typing import Dict

# Configure paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT_DIR, 'input')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True) 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input file paths
INPUT_FILES: Dict[str, str] = {
    'MARKET': os.path.join(INPUT_DIR, 'market.csv'),
    'ETORO': os.path.join(INPUT_DIR, 'etoro.csv'),
    'PORTFOLIO': os.path.join(INPUT_DIR, 'portfolio.csv'),
    'YFINANCE': os.path.join(INPUT_DIR, 'yfinance.csv'),
    'NOTRADE': os.path.join(INPUT_DIR, 'notrade.csv'),
    'CONSOLIDATED': os.path.join(INPUT_DIR, 'cons.csv'),
    'US_TICKERS': os.path.join(INPUT_DIR, 'us_tickers.csv')
}

# Output file paths
OUTPUT_FILES: Dict[str, str] = {
    'BUY': os.path.join(OUTPUT_DIR, 'buy.csv'),
    'SELL': os.path.join(OUTPUT_DIR, 'sell.csv'),
    'HOLD': os.path.join(OUTPUT_DIR, 'hold.csv'),
    'MARKET': os.path.join(OUTPUT_DIR, 'market.csv'),
    'PORTFOLIO': os.path.join(OUTPUT_DIR, 'portfolio.csv'),
    'PORTFOLIO_HTML': os.path.join(OUTPUT_DIR, 'portfolio.html'),
    'INDEX_HTML': os.path.join(OUTPUT_DIR, 'index.html'),
    'SCRIPT_JS': os.path.join(OUTPUT_DIR, 'script.js'),
    'STYLES_CSS': os.path.join(OUTPUT_DIR, 'styles.css')
}

__all__ = [
    'INPUT_DIR',
    'OUTPUT_DIR',
    'INPUT_FILES',
    'OUTPUT_FILES'
]
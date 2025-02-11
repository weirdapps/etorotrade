"""Debug utilities for market analysis tool."""
import logging

def enable_debug():
    """Enable debug logging for detailed troubleshooting."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('yfinance').setLevel(logging.DEBUG)

def disable_debug():
    """Disable debug logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('yfinance').setLevel(logging.INFO)

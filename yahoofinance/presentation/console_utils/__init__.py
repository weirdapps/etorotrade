"""Console presentation module - compatibility layer"""
# Export utility modules but don't import MarketDisplay here
# MarketDisplay should be imported from the parent console.py file

from .formatters import ConsoleFormatter
from .colors import ConsoleColors
from .tables import TableRenderer

__all__ = ['ConsoleFormatter', 'ConsoleColors', 'TableRenderer']

"""Console output formatters"""
from typing import Any

class ConsoleFormatter:
    @staticmethod
    def format_number(value: Any) -> str:
        if value is None or value == "--":
            return "--"
        try:
            if isinstance(value, (int, float)):
                return f"{value:,.2f}"
            return str(value)
        except:
            return "--"
    
    @staticmethod
    def format_percentage(value: Any) -> str:
        if value is None or value == "--":
            return "--"
        try:
            return f"{float(value):.1f}%"
        except:
            return "--"

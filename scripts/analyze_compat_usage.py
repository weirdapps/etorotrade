#!/usr/bin/env python
"""
Script to analyze usage of the yahoofinance.compat package.

This script scans Python files in a directory for imports from the
yahoofinance.compat package and suggests replacements.

Usage:
    python analyze_compat_usage.py <directory_path>

Example:
    python analyze_compat_usage.py ~/my_project/

This will scan all Python files in ~/my_project/ and report any
usage of yahoofinance.compat with suggested replacements.
"""

import os
import re
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# Mapping of deprecated imports to their canonical equivalents
IMPORT_REPLACEMENTS = {
    "yahoofinance.compat.analyst.AnalystData": "yahoofinance.analysis.analyst.CompatAnalystData",
    "yahoofinance.compat.earnings.EarningsCalendar": "yahoofinance.analysis.earnings.EarningsCalendar",
    "yahoofinance.compat.earnings.format_earnings_table": "yahoofinance.analysis.earnings.format_earnings_table",
    "yahoofinance.compat.client.YFinanceClient": "yahoofinance.api.get_provider",
    "yahoofinance.compat.client.StockData": "provider.get_ticker_info() dictionary",
    "yahoofinance.compat.formatting.DisplayFormatter": "yahoofinance.presentation.formatter.DisplayFormatter",
    "yahoofinance.compat.formatting.DisplayConfig": "yahoofinance.presentation.formatter.DisplayConfig",
    "yahoofinance.compat.display.MarketDisplay": "yahoofinance.presentation.console.MarketDisplay",
    "yahoofinance.compat.pricing.PricingAnalyzer": "yahoofinance.analysis.market.MarketAnalyzer",
}

# Regular expressions for finding different import patterns
IMPORT_PATTERNS = [
    # from yahoofinance.compat.module import Class
    r'from\s+(yahoofinance\.compat\.\w+)\s+import\s+([\w,\s]+)',
    # from yahoofinance.compat import module
    r'from\s+yahoofinance\.compat\s+import\s+([\w,\s]+)',
    # import yahoofinance.compat.module
    r'import\s+(yahoofinance\.compat\.\w+)',
]


@dataclass
class ImportUsage:
    """Represents usage of a deprecated import in a file."""
    file_path: str
    line_number: int
    line_text: str
    module_path: str
    imported_name: str
    
    def get_canonical_import(self) -> str:
        """Get the canonical import path for this usage."""
        full_path = f"{self.module_path}.{self.imported_name}"
        # Handle special cases like 'from yahoofinance.compat import analyst'
        if self.module_path == "yahoofinance.compat":
            # This is a module import, not a direct class import
            return f"See migration guide for {self.imported_name} module replacements"
        
        # Check if we have a direct replacement
        if full_path in IMPORT_REPLACEMENTS:
            return IMPORT_REPLACEMENTS[full_path]
        
        # Try partial matching
        for old_path, new_path in IMPORT_REPLACEMENTS.items():
            if old_path.endswith(f".{self.imported_name}"):
                return new_path
                
        return "No direct replacement found, see migration guide"


def find_imports(file_path: str) -> List[ImportUsage]:
    """Find all yahoofinance.compat imports in a file."""
    imports = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line_num = i + 1
            line_text = line.strip()
            
            # Skip commented lines
            if line_text.startswith('#'):
                continue
                
            # Check all import patterns
            for pattern in IMPORT_PATTERNS:
                matches = re.findall(pattern, line_text)
                if matches:
                    if isinstance(matches[0], tuple):
                        # Multiple groups captured
                        module_path, imported_names = matches[0]
                        for name in re.split(r',\s*', imported_names):
                            imports.append(ImportUsage(
                                file_path=file_path,
                                line_number=line_num,
                                line_text=line_text,
                                module_path=module_path,
                                imported_name=name.strip()
                            ))
                    else:
                        # Single group captured
                        module_path = matches[0]
                        imports.append(ImportUsage(
                            file_path=file_path,
                            line_number=line_num,
                            line_text=line_text,
                            module_path=module_path,
                            imported_name=""
                        ))
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        
    return imports


def scan_directory(directory: str) -> List[ImportUsage]:
    """Scan all Python files in a directory for yahoofinance.compat imports."""
    all_imports = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_imports = find_imports(file_path)
                all_imports.extend(file_imports)
                
    return all_imports


def generate_report(imports: List[ImportUsage]) -> str:
    """Generate a report of all found imports with suggested replacements."""
    if not imports:
        return "No yahoofinance.compat imports found."
        
    report = "# YahooFinance Compatibility Layer Migration Report\n\n"
    report += f"Found {len(imports)} deprecated imports that need to be updated.\n\n"
    
    # Group by file
    files = {}
    for imp in imports:
        if imp.file_path not in files:
            files[imp.file_path] = []
        files[imp.file_path].append(imp)
    
    for file_path, file_imports in files.items():
        report += f"## {file_path}\n\n"
        
        for imp in file_imports:
            report += f"Line {imp.line_number}: `{imp.line_text}`\n"
            canonical = imp.get_canonical_import()
            report += f"Suggested replacement: `{canonical}`\n\n"
            
    report += "\n## Migration Guide\n\n"
    report += "Please refer to the yahoofinance migration guide for detailed instructions:\n"
    report += "https://github.com/yourusername/etorotrade/blob/master/tests/MIGRATION_STATUS.md\n\n"
    
    report += "### Common Replacements\n\n"
    
    report += "| Old Import | New Import |\n"
    report += "|------------|------------|\n"
    for old, new in IMPORT_REPLACEMENTS.items():
        report += f"| `{old}` | `{new}` |\n"
        
    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze usage of yahoofinance.compat package')
    parser.add_argument('directory', help='Directory to scan for Python files')
    parser.add_argument('--output', '-o', help='Output file for report (default: stdout)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
        
    print(f"Scanning {args.directory} for yahoofinance.compat imports...")
    imports = scan_directory(args.directory)
    
    report = generate_report(imports)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)
        

if __name__ == "__main__":
    main()
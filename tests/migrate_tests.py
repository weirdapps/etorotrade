#!/usr/bin/env python3
"""
Test migration script to reorganize tests into a structure that mirrors the package.

This script scans existing test files and moves them to the appropriate directory
in the new structure, updating imports as needed.

Usage:
    python tests/migrate_tests.py [--dry-run] [--test=<test_file>]

Options:
    --dry-run        Show migration plan without executing it
    --test=<file>    Migrate a specific test file (relative path from tests/)
"""

import os
import sys
import shutil
import re
from pathlib import Path

# Mapping of old test files to their new locations
# Format: 'old_path_relative_to_tests': 'new_path_relative_to_tests'
TEST_MAPPING = {
    # Core module tests
    'test_client.py': 'yahoofinance/core/test_client.py',
    'test_cache.py': 'yahoofinance/core/test_cache.py',
    'test_errors.py': 'yahoofinance/core/test_errors.py',
    'test_types.py': 'yahoofinance/core/test_types.py',
    'test_error_handling.py': 'yahoofinance/core/test_error_handling.py',
    
    # API tests
    'test_async_providers.py': 'yahoofinance/api/providers/test_async_providers.py',
    'unit/api/test_providers.py': 'yahoofinance/api/providers/test_providers.py',
    
    # Utilities tests
    'test_async.py': 'yahoofinance/utils/async/test_async.py',
    'test_format_utils.py': 'yahoofinance/utils/data/test_format_utils.py',
    'test_formatting.py': 'yahoofinance/utils/data/test_formatting.py',
    'test_market_utils.py': 'yahoofinance/utils/market/test_market_utils.py',
    'utils/market/test_filter_utils.py': 'yahoofinance/utils/market/test_filter_utils.py',
    'test_pagination_utils.py': 'yahoofinance/utils/network/test_pagination.py',
    'test_rate.py': 'yahoofinance/utils/network/test_rate.py',
    'test_rate_limiter.py': 'yahoofinance/utils/network/test_rate_limiter.py',
    'unit/utils/async/test_async_helpers.py': 'yahoofinance/utils/async/test_async_helpers.py',
    'unit/core/test_rate_limiter.py': 'yahoofinance/utils/network/test_rate_limiter.py',
    
    # Analysis module tests
    'test_analyst.py': 'yahoofinance/analysis/test_analyst.py',
    'test_pricing.py': 'yahoofinance/analysis/test_pricing.py',
    'test_earnings.py': 'yahoofinance/analysis/test_earnings.py',
    'test_news.py': 'yahoofinance/analysis/test_news.py',
    'test_holders.py': 'yahoofinance/analysis/test_holders.py',
    'test_insiders.py': 'yahoofinance/analysis/test_insiders.py',
    'test_metrics.py': 'yahoofinance/analysis/test_metrics.py',
    'test_monthly.py': 'yahoofinance/analysis/test_monthly.py',
    'test_weekly.py': 'yahoofinance/analysis/test_weekly.py',
    'test_index.py': 'yahoofinance/analysis/test_index.py',
    'test_econ.py': 'yahoofinance/analysis/test_econ.py',
    
    # Display and presentation tests
    'test_display.py': 'yahoofinance/presentation/test_display.py',
    'test_market_display.py': 'yahoofinance/presentation/test_market_display.py',
    'test_templates.py': 'yahoofinance/presentation/test_templates.py',
    
    # Main module tests
    'test_trade.py': 'trade/test_trade.py',
    
    # Validators
    'test_validate.py': 'yahoofinance/validators/test_validate.py',
    
    # Other tests (add custom mappings as needed)
    'test_advanced_utils.py': 'yahoofinance/utils/test_advanced_utils.py',
    'test_utils.py': 'yahoofinance/utils/test_utils.py',
    'test_utils_refactor.py': 'yahoofinance/utils/test_utils_refactor.py',
    'test_download.py': 'yahoofinance/data/test_download.py',
    'test_compatibility.py': 'yahoofinance/test_compatibility.py',
    'test_improvements.py': 'yahoofinance/test_improvements.py',
    
    # Integration tests
    'integration/test_api_integration.py': 'integration/api/test_api_integration.py',
    'integration/test_async_api.py': 'integration/api/test_async_api.py',
}

def update_imports(file_path, old_path, new_path):
    """Update imports in a file to reflect the new module structure."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Determine import path adjustments
    old_parts = old_path.split('/')
    new_parts = new_path.split('/')
    
    # Calculate relative path adjustments
    rel_levels_up = len(old_parts) - 1  # How many levels up from old location
    rel_levels_down = len(new_parts) - 1  # How many levels down to new location
    
    # Common import patterns that might need updating
    patterns = [
        # from .module import X
        (r'from \.([\w_]+) import', r'from ...\1 import'),
        # from ..module import X
        (r'from \.\.([\w_]+) import', r'from ....\1 import'),
        # import module
        (r'import ([\w_]+)', r'import \1'),
        # from module import X
        (r'from ([\w_]+) import', r'from \1 import'),
    ]
    
    # Apply replacements
    updated_content = content
    for pattern, replacement in patterns:
        # Customize replacement based on directory depth changes
        if '..' in replacement:
            # Adjust relative imports
            dots = '.' * (rel_levels_up + rel_levels_down)
            actual_replacement = replacement.replace('...', dots)
            updated_content = re.sub(pattern, actual_replacement, updated_content)
    
    # Special case for 'from tests.' imports
    updated_content = updated_content.replace('from tests.', 'from ...')
    
    return updated_content

def migrate_test(test_file, dry_run=False):
    """Migrate a single test file to its new location."""
    tests_dir = Path(__file__).parent
    
    old_path = test_file
    new_path = TEST_MAPPING.get(old_path)
    
    if not new_path:
        print(f"No mapping defined for {old_path}")
        return False
    
    old_file = tests_dir / old_path
    new_file = tests_dir / new_path
    
    if not old_file.exists():
        print(f"Source file does not exist: {old_file}")
        return False
    
    print(f"Migrating: {old_path} -> {new_path}")
    
    if not dry_run:
        # Create destination directory if it doesn't exist
        new_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files in the directory path
        path_parts = new_path.split('/')
        current_path = tests_dir
        for part in path_parts[:-1]:  # Skip the file itself
            current_path = current_path / part
            init_file = current_path / "__init__.py"
            if not init_file.exists():
                print(f"Creating: {init_file.relative_to(tests_dir)}")
                init_file.touch()
        
        # Update imports in the file
        updated_content = update_imports(old_file, old_path, new_path)
        
        # Write updated file to new location
        with open(new_file, 'w') as f:
            f.write(updated_content)
        
        print(f"Created: {new_file.relative_to(tests_dir)}")
        
        # Remove old file if it was successfully written
        if new_file.exists():
            print(f"Removing: {old_file.relative_to(tests_dir)}")
            old_file.unlink()
    
    return True

def migrate_all_tests(dry_run=False):
    """Migrate all tests according to the mapping."""
    success_count = 0
    error_count = 0
    
    for old_path in TEST_MAPPING:
        try:
            if migrate_test(old_path, dry_run):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"Error migrating {old_path}: {str(e)}")
            error_count += 1
    
    print(f"\nMigration summary:")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Total: {len(TEST_MAPPING)}")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    # Check for specific test file
    test_arg = next((arg for arg in sys.argv if arg.startswith("--test=")), None)
    if test_arg:
        test_file = test_arg.split("=", 1)[1]
        migrate_test(test_file, dry_run)
    else:
        migrate_all_tests(dry_run)
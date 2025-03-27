#!/usr/bin/env python3
"""
Test import path migration script.

This script updates import paths in test files to match the new module structure.
It handles two types of migrations:

1. Replacing 'yahoofinance_v2' with 'yahoofinance' to match the unified package
2. Updating old module paths to new module paths after reorganization

Usage:
    python tests/migrate_tests.py [--dry-run] [--path=<path>] [--mode=<mode>]

Options:
    --dry-run        Show files that would be updated without making changes
    --path=<path>    Process a specific file or directory (relative to project root)
    --mode=<mode>    Migration mode: 'v2' for yahoofinance_v2 -> yahoofinance, 
                     'modules' for updating module paths (default: 'modules')
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

# Mapping of old module paths to new module paths
MODULE_PATH_MAPPING = {
    "yahoofinance.display": "yahoofinance.presentation.console",
    "yahoofinance.formatting": "yahoofinance.presentation.formatter",
    "yahoofinance.analyst": "yahoofinance.analysis.analyst",
    "yahoofinance.earnings": "yahoofinance.analysis.earnings",
    "yahoofinance.econ": "yahoofinance.analysis.market",
    "yahoofinance.holders": "yahoofinance.analysis.portfolio",
    "yahoofinance.index": "yahoofinance.analysis.market",
    "yahoofinance.insiders": "yahoofinance.analysis.insiders",
    "yahoofinance.metrics": "yahoofinance.analysis.metrics",
    "yahoofinance.monthly": "yahoofinance.analysis.performance",
    "yahoofinance.news": "yahoofinance.analysis.news",
    "yahoofinance.portfolio": "yahoofinance.analysis.portfolio",
    "yahoofinance.pricing": "yahoofinance.analysis.stock",
    "yahoofinance.weekly": "yahoofinance.analysis.performance",
    "yahoofinance.utils.async.enhanced": "yahoofinance.utils.async_utils.enhanced",
    "yahoofinance.utils.async.helpers": "yahoofinance.utils.async_utils.helpers",
    "yahoofinance.utils.async": "yahoofinance.utils.async_utils",
}

# Class relocations
CLASS_RELOCATIONS = {
    "MarketDisplay": "yahoofinance.presentation.console",
    "PricingAnalyzer": "yahoofinance.analysis.metrics",
    "AnalystData": "yahoofinance.analysis.analyst",
}


def find_test_files(base_path: Path, pattern: str = "*.py") -> List[Path]:
    """Find all Python test files under the base path."""
    return list(base_path.glob(f"**/{pattern}"))


def update_v2_imports(file_path: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Update imports in a file, replacing 'yahoofinance_v2' with 'yahoofinance'.
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't actually write changes
        
    Returns:
        Tuple of (number of lines changed, list of samples of changed lines)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match 'yahoofinance_v2' in various import statements
    patterns = [
        (r'from yahoofinance_v2(\.\w+)', r'from yahoofinance\1'),
        (r'import yahoofinance_v2(\.\w+)', r'import yahoofinance\1'),
        (r'import yahoofinance_v2$', r'import yahoofinance'),
    ]
    
    original_content = content
    changed_samples = []
    
    for pattern, replacement in patterns:
        # Find all matches for this pattern
        matches = re.findall(pattern, content)
        if matches:
            # Create sample of the change
            for match in matches[:3]:  # Get up to 3 examples
                old_line = f"from yahoofinance_v2{match}" if "from" in pattern else f"import yahoofinance_v2{match}"
                new_line = f"from yahoofinance{match}" if "from" in pattern else f"import yahoofinance{match}"
                changed_samples.append(f"{old_line} -> {new_line}")
        
        # Replace all occurrences
        content = re.sub(pattern, replacement, content)
    
    # Also fix patch statements that reference yahoofinance_v2
    content = re.sub(r'"yahoofinance_v2\.', r'"yahoofinance.', content)
    content = re.sub(r"'yahoofinance_v2\.", r"'yahoofinance.", content)
    
    # Count the number of lines changed
    changes = sum(1 for a, b in zip(original_content.splitlines(), content.splitlines()) if a != b)
    
    if not dry_run and changes > 0:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return changes, changed_samples


def update_module_paths(file_path: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Update module paths in a file to match the new module structure.
    
    Args:
        file_path: Path to the file to update
        dry_run: If True, don't actually write changes
        
    Returns:
        Tuple of (number of lines changed, list of samples of changed lines)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    changed_samples = []
    
    # Update imports for modules
    for old_path, new_path in MODULE_PATH_MAPPING.items():
        # Pattern for from xxx import yyy
        from_pattern = fr'from\s+{re.escape(old_path)}([\s\.]+)'
        from_replacement = f'from {new_path}\\1'
        
        # Find all matches
        from_matches = re.findall(from_pattern, content)
        if from_matches:
            for match in from_matches[:3]:
                old_line = f"from {old_path}{match}"
                new_line = f"from {new_path}{match}"
                changed_samples.append(f"{old_line} -> {new_line}")
        
        # Replace all occurrences
        content = re.sub(from_pattern, from_replacement, content)
        
        # Pattern for import xxx
        import_pattern = fr'import\s+{re.escape(old_path)}([\s]+|$)'
        import_replacement = f'import {new_path}\\1'
        
        # Find all matches
        import_matches = re.findall(import_pattern, content)
        if import_matches:
            for match in import_matches[:3]:
                old_line = f"import {old_path}{match}"
                new_line = f"import {new_path}{match}"
                changed_samples.append(f"{old_line} -> {new_line}")
        
        # Replace all occurrences
        content = re.sub(import_pattern, import_replacement, content)
        
        # Fix patch statements
        content = re.sub(f'"{re.escape(old_path)}', f'"{new_path}', content)
        content = re.sub(f"'{re.escape(old_path)}", f"'{new_path}", content)
    
    # Update class imports for specific classes
    for class_name, module_path in CLASS_RELOCATIONS.items():
        # Pattern for from xxx import ClassA
        class_pattern = fr'from\s+[\w\.]+\s+import\s+[\w\s,]+({re.escape(class_name)})'
        
        # Find all matches
        if re.search(class_pattern, content):
            # This is a more complex update that requires us to potentially modify
            # multiple import statements. Let's simplify by adding a comment
            for line in content.splitlines():
                if class_name in line and ("import" in line or "from" in line):
                    changed_samples.append(f"{line} -> # Update: Import {class_name} from {module_path}")
    
    # Count the number of lines changed
    changes = sum(1 for a, b in zip(original_content.splitlines(), content.splitlines()) if a != b)
    
    if not dry_run and changes > 0:
        with open(file_path, 'w') as f:
            f.write(content)
    
    return changes, changed_samples


def process_directory(directory: Path, dry_run: bool = False, mode: str = "modules") -> Tuple[int, int]:
    """
    Process all Python files in a directory recursively.
    
    Args:
        directory: Directory to process
        dry_run: If True, don't actually write changes
        mode: Migration mode, either 'v2' or 'modules'
        
    Returns:
        Tuple of (number of files changed, total number of lines changed)
    """
    test_files = find_test_files(directory)
    files_changed = 0
    total_lines_changed = 0
    
    for file_path in test_files:
        lines_changed, changed_samples = process_file(file_path, dry_run, mode)
        if lines_changed > 0:
            files_changed += 1
            total_lines_changed += lines_changed
    
    return files_changed, total_lines_changed


def process_file(file_path: Path, dry_run: bool = False, mode: str = "modules") -> Tuple[int, int]:
    """
    Process a single file.
    
    Args:
        file_path: File to process
        dry_run: If True, don't actually write changes
        mode: Migration mode, either 'v2' or 'modules'
        
    Returns:
        Tuple of (1 if file was changed else 0, number of lines changed)
    """
    # Choose the appropriate update function based on mode
    if mode == "v2":
        update_func = update_v2_imports
    else:  # mode == "modules"
        update_func = update_module_paths
    
    # Update the file
    lines_changed, changed_samples = update_func(file_path, dry_run)
    if lines_changed > 0:
        print(f"{'Would update' if dry_run else 'Updated'}: {file_path}")
        for sample in changed_samples[:3]:  # Show up to 3 examples
            print(f"  {sample}")
        return 1, lines_changed
    return 0, 0


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    
    path_arg = next((arg for arg in sys.argv if arg.startswith("--path=")), None)
    specific_path = path_arg.split("=", 1)[1] if path_arg else None
    
    mode_arg = next((arg for arg in sys.argv if arg.startswith("--mode=")), None)
    mode = mode_arg.split("=", 1)[1] if mode_arg else "modules"
    
    if mode not in ["v2", "modules"]:
        print(f"Error: Invalid mode: {mode}. Must be 'v2' or 'modules'.")
        return 1
    
    # Get project root directory (one level up from the script)
    project_root = Path(__file__).parent.parent
    
    print(f"{'Dry run mode - no files will be changed' if dry_run else 'Processing files...'}")
    print(f"Mode: {mode} {'(yahoofinance_v2 -> yahoofinance)' if mode == 'v2' else '(update module paths)'}")
    
    if specific_path:
        path = project_root / specific_path
        if path.is_file():
            files_changed, lines_changed = process_file(path, dry_run, mode)
        elif path.is_dir():
            files_changed, lines_changed = process_directory(path, dry_run, mode)
        else:
            print(f"Error: Path not found: {path}")
            return 1
    else:
        # Process the tests directory by default
        files_changed, lines_changed = process_directory(project_root / 'tests', dry_run, mode)
    
    print(f"\nSummary:")
    print(f"  {'Would update' if dry_run else 'Updated'} {files_changed} files")
    print(f"  {'Would change' if dry_run else 'Changed'} {lines_changed} import lines")
    
    if dry_run and files_changed > 0:
        print("\nRun without --dry-run to make these changes.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
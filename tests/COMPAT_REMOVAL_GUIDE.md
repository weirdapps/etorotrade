# Guide to Removing the Compatibility Layer

This document provides a step-by-step guide for testing and removing the `yahoofinance/compat/` directory once all code has been migrated to use the canonical sources.

## Step 1: Run Migration Tests

First, run the migration test suite to verify that canonical sources provide equivalent functionality:

```bash
# Run the migration test suite
pytest tests/test_migration.py -v

# Also run integration tests that verify compatibility
pytest tests/integration/api/test_api_integration.py::TestProviderCompatibility -v
```

These tests should pass, ensuring that canonical sources can replace the compatibility layer.

## Step 2: Verify No Internal Dependencies

Run these commands to verify that no internal code depends on the compatibility layer:

```bash
# Find any internal imports from compat
grep -r "from yahoofinance\.compat" --include="*.py" yahoofinance/ | grep -v "compat/"

# Find any internal imports using 'import yahoofinance.compat'
grep -r "import yahoofinance\.compat" --include="*.py" yahoofinance/ | grep -v "compat/"
```

If these commands return any results, those imports need to be updated to use canonical sources before proceeding.

## Step 3: Update All Test Code

Update all test files to use canonical sources instead of the compatibility layer:

```bash
# Find any test imports from compat
grep -r "from yahoofinance\.compat" --include="*.py" tests/
```

For each file that imports from the compat layer, update it to use the canonical imports as documented in `MIGRATION_STATUS.md`.

## Step 4: Run All Tests Without Compat

Temporarily rename the compat directory to verify that all tests pass without it:

```bash
# Rename the directory
mv yahoofinance/compat yahoofinance/compat_backup

# Run all tests
pytest

# Restore the directory if needed for further testing
mv yahoofinance/compat_backup yahoofinance/compat
```

If all tests pass, the codebase is ready for removal of the compatibility layer.

## Step 5: Remove the Compatibility Layer

Once you've verified everything works without the compatibility layer:

```bash
# Remove the compatibility layer
rm -rf yahoofinance/compat/

# Update version number in core/__init__.py to indicate breaking change
# This should be a major version bump (e.g., 1.0.0 -> 2.0.0)

# Commit the changes
git add yahoofinance/
git commit -m "Remove compatibility layer, breaking change for 2.0.0 release"
```

## Step 6: Update Documentation

Update the main README.md and other documentation to:

1. Remove references to the compatibility layer
2. Clearly document that this is a breaking change
3. Update all examples to use the canonical imports

## Step 7: Create a Migration Guide

Create a migration guide for users of the library, explaining:

1. Why the compatibility layer was removed
2. How to update imports to use canonical sources
3. The benefits of using the new patterns (provider pattern, etc.)
4. Include before/after examples for common use cases

## Step 8: Release

Create a new release with a major version bump to signify the breaking change:

```bash
# Tag the release
git tag -a v2.0.0 -m "Release 2.0.0 - Remove compatibility layer"

# Push the tag
git push origin v2.0.0
```

## Keeping Backward Compatibility (Alternative)

If removing the compatibility layer is too disruptive, consider these alternatives:

1. Mark it as deprecated but keep it for one more major release
2. Move it to a separate package (e.g., `yahoofinance-compat`)
3. Create a feature flag to enable/disable compatibility layer

The recommended approach is complete removal with a major version bump, as it:
- Reduces maintenance burden
- Encourages users to adopt the new patterns
- Simplifies the codebase
- Prevents reliance on deprecated interfaces
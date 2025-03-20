# Test Migration Plan

This document outlines the plan for migrating tests to the new structure that mirrors the package organization.

## Goals

1. Improve test organization and discoverability
2. Make the relationship between code and tests more intuitive
3. Group related tests together for easier maintenance
4. Establish a consistent pattern for new tests
5. Simplify navigation between implementation and tests

## Migration Process

### Step 1: Create Directory Structure

The new directory structure has been created to mirror the main package:

```
tests/
├── yahoofinance/
│   ├── analysis/
│   ├── api/
│   │   └── providers/
│   ├── core/
│   ├── data/
│   ├── presentation/
│   ├── utils/
│   │   ├── async/
│   │   ├── data/
│   │   ├── date/
│   │   ├── market/
│   │   └── network/
│   │       └── async_utils/
│   └── validators/
├── trade/
├── e2e/
├── integration/
└── fixtures/
```

### Step 2: Move Tests to New Locations

Tests will be moved to the appropriate locations in the new structure. The mapping of old test files to new locations is defined in `migrate_tests.py`.

To perform a dry run of the migration:

```bash
python tests/migrate_tests.py --dry-run
```

To migrate a specific test file:

```bash
python tests/migrate_tests.py --test=test_cache.py
```

To migrate all tests:

```bash
python tests/migrate_tests.py
```

### Step 3: Update Imports

The migration script will update imports in the test files to reflect their new locations. For example:

- `from tests.fixtures.async_fixtures import ...` → `from ...fixtures.async_fixtures import ...`
- `from .utils import ...` → `from ....utils import ...`

### Step 4: Create __init__.py Files

The migration script will create `__init__.py` files in the directory path to ensure proper package structure. These files will be empty for now but can be updated later with exports if needed.

### Step 5: Update CI/CD Configuration

If there are CI/CD configurations that reference specific test files or directories, they will need to be updated to reflect the new structure.

### Step 6: Run Tests to Verify

After migration, run the tests to verify they still work with the new structure:

```bash
pytest tests/
```

### Step 7: Update Documentation

Update any documentation references to test files or directories.

## Test Standards

All tests should follow these standards:

1. **File Location**: Tests should be located in the directory that mirrors the module they test.
2. **File Naming**: Test files should be named `test_<module>.py` or `test_<module>_<component>.py`.
3. **Test Class**: Test classes should be named `Test<Component>` or `Test<Component><Functionality>`.
4. **Test Methods**: Test methods should be named `test_<functionality>_<scenario>`.
5. **Docstrings**: Every test class and method should have a docstring describing what it tests.
6. **Markers**: Tests should be marked with appropriate pytest markers:
   - `@pytest.mark.unit`: Unit tests for isolated components
   - `@pytest.mark.integration`: Tests that verify component interactions
   - `@pytest.mark.e2e`: End-to-end workflow tests
   - `@pytest.mark.slow`: Tests that take longer to run
   - `@pytest.mark.api`: Tests that require API access
   - `@pytest.mark.network`: Tests that require network connectivity
   - `@pytest.mark.asyncio`: Tests for async functionality
7. **Fixtures**: Tests should use fixtures from `conftest.py` or the `fixtures/` directory when possible.
8. **Setup/Teardown**: Tests should follow the Arrange-Act-Assert pattern and properly clean up after themselves.

## Example

See `tests/yahoofinance/core/test_cache_example.py` for an example of a well-organized test file in the new structure.

## Timeline

1. **Week 1**: Create directory structure and migration script
2. **Week 2**: Migrate core module tests
3. **Week 3**: Migrate API and analysis module tests
4. **Week 4**: Migrate utility module tests
5. **Week 5**: Migrate remaining tests and update documentation

## Status Tracking

A tracking issue will be created in the repository to track the progress of the migration. Each test file will be marked as:

- ❌ Not Migrated
- 🔄 In Progress
- ✅ Migrated

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test Directory Organization](https://docs.pytest.org/en/stable/explanation/goodpractices.html#test-directory-structure)
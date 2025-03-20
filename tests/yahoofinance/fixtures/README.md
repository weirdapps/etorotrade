# Test Fixtures

This directory contains reusable test fixtures and helpers for testing the etorotrade application.

## Available Fixtures

### Pagination Fixtures (`pagination.py`)

- `create_paginated_data(num_pages, items_per_page)` - Creates mock paginated data responses
- `create_mock_fetcher(pages)` - Creates a mock page fetcher function 
- `create_bulk_fetch_mocks()` - Creates mock objects for bulk fetching tests

### Async Fixtures (`async_fixtures.py`)

- `create_flaky_function(fail_count)` - Creates a mock async function that initially fails
- `create_async_processor_mock(error_item)` - Creates a mock async processor function

## Usage

Import these fixtures in your tests like this:

```python
# Option 1: Import from fixtures package (recommended)
from tests.fixtures import create_paginated_data, create_mock_fetcher

# Option 2: Import directly from specific module
from tests.fixtures.pagination import create_paginated_data
from tests.fixtures.async_fixtures import create_flaky_function
```

## Writing New Fixtures

When adding new fixtures:

1. Create them in the appropriate module based on functionality
2. Add proper type hints and comprehensive docstrings
3. Add them to `__all__` in `__init__.py`
4. Document them in this README
# Dependency Injection System

The dependency injection (DI) system provides a clean, maintainable, and testable way to manage component dependencies in the Yahoo Finance API.

## Key Components

### Registry

The `Registry` class is the central component of the DI system. It maintains a registry of factory functions and singleton instances.

```python
from yahoofinance.utils.dependency_injection import registry

# Get a registered component
provider = registry.resolve('yahoo_provider', async_mode=True)

# Register a factory function
@registry.register('my_component')
def create_my_component(**kwargs):
    return MyComponent(**kwargs)

# Register a singleton instance
my_singleton = MySingleton()
registry.register_instance('my_singleton', my_singleton)
```

### Injection Decorators

The system provides decorators for injecting dependencies into functions:

```python
from yahoofinance.utils.dependency_injection import inject
from yahoofinance.core.di_container import with_analyzer, with_provider

# Using the inject decorator directly
@inject('yahoo_provider', async_mode=True)
def my_function(provider=None):
    # provider is automatically injected
    return provider.get_data()

# Using a specialized decorator
@with_analyzer(async_mode=True)
def analyze_stock(ticker, analyzer=None):
    # analyzer is automatically injected
    return analyzer.analyze(ticker)
```

### Factory Functions

Factory functions are responsible for creating instances of components:

```python
@registry.register('yahoo_provider')
def create_yahoo_provider(async_mode=False, **kwargs):
    if async_mode:
        return AsyncYahooFinanceProvider(**kwargs)
    else:
        return YahooFinanceProvider(**kwargs)
```

### Provides Decorator

The `provides` decorator registers a function's return value as a singleton:

```python
from yahoofinance.utils.dependency_injection import provides

@provides('config')
def create_config():
    config = Config()
    config.load_from_file('config.ini')
    return config
```

### Lazy Imports

Lazy imports help avoid circular dependencies:

```python
from yahoofinance.utils.dependency_injection import lazy_import

# Import lazily
AsyncYahooFinanceProvider = lazy_import(
    'yahoofinance.api.providers.async_yahoo_finance',
    'AsyncYahooFinanceProvider'
)
```

## Usage Examples

### Basic Component Resolution

```python
from yahoofinance.utils.dependency_injection import registry

# Resolve a provider
provider = registry.resolve('yahoo_provider', async_mode=True)

# Use the provider
data = provider.get_ticker_info('AAPL')
```

### Using Injection Decorators

```python
from yahoofinance.core.di_container import with_analyzer

# Inject a StockAnalyzer
@with_analyzer(async_mode=True)
def analyze_stock(ticker, analyzer=None):
    return analyzer.analyze(ticker)

# Now you can call the function without passing an analyzer
result = analyze_stock('AAPL')  # analyzer is injected automatically
```

### Asynchronous Component Usage

```python
from yahoofinance.core.di_container import with_analyzer
import asyncio

@with_analyzer(async_mode=True)
async def analyze_stocks(tickers, analyzer=None):
    results = {}
    for ticker in tickers:
        results[ticker] = await analyzer.analyze_async(ticker)
    return results

# Run the async function
results = asyncio.run(analyze_stocks(['AAPL', 'MSFT', 'GOOG']))
```

### Batch Processing

```python
from yahoofinance.core.di_container import with_analyzer
import asyncio

@with_analyzer(async_mode=True)
async def analyze_batch(tickers, analyzer=None):
    return await analyzer.analyze_batch_async(tickers)

# Run the batch analysis
results = asyncio.run(analyze_batch(['AAPL', 'MSFT', 'GOOG', 'AMZN']))
```

### Portfolio Analysis

```python
from yahoofinance.core.di_container import with_portfolio_analyzer

@with_portfolio_analyzer()
def analyze_portfolio(file_path, portfolio_analyzer=None):
    portfolio_analyzer.load_portfolio_from_csv(file_path)
    return portfolio_analyzer.analyze_portfolio()

# Analyze a portfolio
portfolio_summary = analyze_portfolio('portfolio.csv')
```

## Testing with DI

The DI system makes testing much easier by allowing components to be mocked:

```python
from yahoofinance.utils.dependency_injection import registry
import pytest

@pytest.fixture
def setup_test():
    # Register a mock provider
    mock_provider = MockProvider()
    registry.register_instance('yahoo_provider', mock_provider)
    yield
    # Clear all singleton instances
    registry.clear_instances()

def test_analysis(setup_test):
    # The function will use the mock provider automatically
    result = analyze_stock('AAPL')
    assert result.category == 'BUY'
```

## Application Container

The `di_container` module serves as the main entry point for the DI system. It sets up all application components and provides convenient decorators:

```python
from yahoofinance.core.di_container import (
    initialize, 
    with_provider, 
    with_analyzer, 
    with_portfolio_analyzer,
    with_display,
    with_logger
)

# Initialize the DI container
initialize()

# Use the convenience decorators
@with_analyzer()
@with_logger
def my_function(ticker, analyzer=None, app_logger=None):
    app_logger.info(f"Analyzing {ticker}")
    return analyzer.analyze(ticker)
```

## Best Practices

1. **Component Registration**
   - Register all major components as factory functions
   - Use descriptive keys for component registration
   - Register components in a central location (`core/di_container.py`)

2. **Dependency Injection**
   - Use the `inject` decorator for injecting dependencies
   - Use specialized decorators for clarity
   - Document the dependencies each function requires

3. **Testing**
   - Use `register_instance` to mock components during testing
   - Use `clear_instances` between tests
   - Create test fixtures for common test setup

4. **Error Handling**
   - Handle missing dependencies gracefully
   - Provide clear error messages
   - Log dependency resolution failures

5. **Circular Dependencies**
   - Use lazy imports to avoid circular imports
   - Use factory functions to break dependency cycles
   - Ensure components are loosely coupled

6. **Documentation**
   - Document component dependencies in docstrings
   - Use type hints for clarity
   - Provide examples of component usage
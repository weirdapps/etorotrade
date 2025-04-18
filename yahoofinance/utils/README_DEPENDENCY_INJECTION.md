# Dependency Injection System

This document explains the dependency injection (DI) system implemented in the YahooFinance library.

## Overview

The dependency injection system provides a way to:

1. Centralize component creation
2. Decouple component creation from component usage
3. Make testing easier by allowing components to be swapped out
4. Avoid circular dependencies through lazy imports
5. Improve code clarity and maintainability

## Key Components

### Registry

The `Registry` class in `dependency_injection.py` is the central component of the DI system. It maintains a registry of factory functions and singleton instances.

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

### Factory Functions

Factory functions are responsible for creating instances of components. They encapsulate the creation logic and any dependencies required.

```python
# Provider factory
@registry.register('yahoo_provider')
def create_yahoo_provider(async_mode=False, **kwargs):
    if async_mode:
        return AsyncYahooFinanceProvider(**kwargs)
    else:
        return YahooFinanceProvider(**kwargs)

# Analyzer factory
@registry.register('stock_analyzer')
def create_stock_analyzer(provider=None, async_mode=False, **kwargs):
    if provider is None:
        provider = registry.resolve('yahoo_provider', async_mode=async_mode)
    return StockAnalyzer(provider=provider, **kwargs)
```

### Injection Decorators

The system provides decorators for injecting dependencies into functions:

```python
# Using the inject decorator directly
@inject('yahoo_provider', async_mode=True)
def my_function(param1, provider=None):
    # provider is automatically injected
    return provider.get_data(param1)

# Using a specialized decorator
@with_analyzer(async_mode=True)
def analyze_stock(ticker, analyzer=None):
    # analyzer is automatically injected
    return analyzer.analyze(ticker)
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
from yahoofinance.analysis import with_analyzer

# Inject a StockAnalyzer
@with_analyzer(async_mode=True, enhanced=True)
def analyze_stock(ticker, analyzer=None):
    return analyzer.analyze_async(ticker)

# Now you can call the function without passing an analyzer
result = analyze_stock('AAPL')  # analyzer is injected automatically
```

### Asynchronous Component Usage

```python
from yahoofinance.analysis import with_analyzer
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
from yahoofinance.analysis import with_analyzer
import asyncio

@with_analyzer(async_mode=True, enhanced=True)
async def analyze_batch(tickers, analyzer=None):
    return await analyzer.analyze_batch_async(tickers)

# Run the batch analysis
results = asyncio.run(analyze_batch(['AAPL', 'MSFT', 'GOOG', 'AMZN']))
```

### Portfolio Analysis

```python
from yahoofinance.analysis import with_portfolio_analyzer

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

# In your test setup
def setup_test():
    # Register a mock provider
    mock_provider = MockProvider()
    registry.set_instance('yahoo_provider', mock_provider)

# In your test teardown
def teardown_test():
    # Clear all singleton instances
    registry.clear_instances()

# Test a function that uses DI
def test_analysis():
    setup_test()
    result = analyze_stock('AAPL')  # Uses the mock provider automatically
    assert result.category == 'BUY'
    teardown_test()
```

## Best Practices

1. Register factory functions for all major components
2. Use the `inject` decorator for injecting dependencies
3. Don't mix DI with direct instantiation in the same function
4. Use lazy imports to avoid circular dependencies
5. Clear the registry between tests
6. Use specialized decorators (`with_analyzer`, `with_portfolio_analyzer`) for clarity
7. Document the dependencies each function requires
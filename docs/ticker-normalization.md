# Ticker Normalization System

## Overview

The ticker normalization system handles dual-listed stocks that trade on multiple exchanges, ensuring consistent handling throughout the entire application. When a US ticker (ADR or cross-listing) is encountered, it is normalized to the original exchange ticker for display, position sizing, and all other operations.

## Key Features

- **Centralized Mapping**: All dual-listed stock mappings are defined in one location
- **Automatic Normalization**: US tickers are automatically converted to their canonical forms
- **Portfolio Filtering**: Prevents duplicate holdings of the same stock under different tickers
- **Geographic Risk Management**: Correct geographic multipliers are applied based on the original exchange
- **Consistent Display**: All output uses the original exchange ticker for consistency

## Core Components

### 1. Ticker Mappings Configuration (`yahoofinance/core/config/ticker_mappings.py`)

Central configuration file containing:
- `DUAL_LISTED_MAPPINGS`: Maps US tickers to original exchange tickers
- `REVERSE_MAPPINGS`: Maps original exchange tickers back to US tickers
- `TICKER_GEOGRAPHY`: Geographic region mapping for risk calculations
- Core functions for ticker normalization and equivalence checking

### 2. Ticker Utilities (`yahoofinance/utils/data/ticker_utils.py`)

Convenient wrapper functions for application-wide use:
- `normalize_ticker()`: Main normalization function
- `check_equivalent_tickers()`: Portfolio filtering support
- `get_geographic_region()`: Risk multiplier calculations
- `standardize_ticker_format()`: Format standardization (HK padding, crypto suffixes)

## Supported Dual-Listed Stocks

The system currently handles these major dual-listings:

### European Stocks with US ADRs
- **NVO** (Novo Nordisk ADR) → **NOVO-B.CO** (Copenhagen)
- **SNY** (Sanofi ADR) → **SAN.PA** (Paris)
- **ASML** (NASDAQ) → **ASML.NV** (Netherlands)
- **SHEL** (Shell ADR) → **SHEL.L** (London)
- **UL** (Unilever ADR) → **ULVR.L** (London)
- **SAP** (SAP ADR) → **SAP.DE** (Germany)

### Asian Stocks with US ADRs
- **JD** (JD.com ADR) → **9618.HK** (Hong Kong)
- **BABA** (Alibaba ADR) → **9988.HK** (Hong Kong)
- **TCEHY** (Tencent ADR) → **0700.HK** (Hong Kong)
- **TM** (Toyota ADR) → **7203.T** (Tokyo)
- **SONY** (Sony ADR) → **6758.T** (Tokyo)

### Share Class Normalization
- **GOOGL** (Google Class A) → **GOOG** (Google Class C, main ticker)

## Usage Examples

### Basic Ticker Normalization

```python
from yahoofinance.utils.data.ticker_utils import normalize_ticker

# US ticker gets normalized to original exchange
normalized = normalize_ticker("NVO")  # Returns "NOVO-B.CO"
normalized = normalize_ticker("GOOGL")  # Returns "GOOG"

# Original exchange tickers remain unchanged
normalized = normalize_ticker("NOVO-B.CO")  # Returns "NOVO-B.CO"
normalized = normalize_ticker("AAPL")  # Returns "AAPL"
```

### Portfolio Filtering

```python
from yahoofinance.utils.data.ticker_utils import check_equivalent_tickers

# Check if two tickers represent the same stock
are_same = check_equivalent_tickers("NVO", "NOVO-B.CO")  # Returns True
are_same = check_equivalent_tickers("GOOGL", "GOOG")  # Returns True
are_same = check_equivalent_tickers("AAPL", "MSFT")  # Returns False

# Filter buy opportunities to avoid duplicates
portfolio_holdings = ["NVO", "GOOGL", "AAPL"]
buy_opportunities = ["NOVO-B.CO", "GOOG", "MSFT", "TSLA"]

filtered_opportunities = []
for opportunity in buy_opportunities:
    is_already_held = any(
        check_equivalent_tickers(opportunity, holding)
        for holding in portfolio_holdings
    )
    if not is_already_held:
        filtered_opportunities.append(opportunity)

# Result: ["MSFT", "TSLA"] (NOVO-B.CO and GOOG filtered out)
```

### Geographic Risk Calculation

```python
from yahoofinance.utils.data.ticker_utils import get_geographic_region

# Get correct geographic region for risk multipliers
region = get_geographic_region("NVO")  # Returns "EU" (from NOVO-B.CO)
region = get_geographic_region("9988.HK")  # Returns "HK"
region = get_geographic_region("AAPL")  # Returns "US"
```

### HK Ticker Format Standardization

```python
from yahoofinance.utils.data.ticker_utils import standardize_ticker_format

# HK tickers are padded to 4 digits
standardized = standardize_ticker_format("700.HK")  # Returns "0700.HK"
standardized = standardize_ticker_format("1.HK")  # Returns "0001.HK"

# Leading zeros removed from 5+ digit HK tickers
standardized = standardize_ticker_format("03690.HK")  # Returns "3690.HK"
```

### Crypto Ticker Normalization

```python
# Crypto tickers get -USD suffix automatically
standardized = standardize_ticker_format("BTC")  # Returns "BTC-USD"
standardized = standardize_ticker_format("ETH")  # Returns "ETH-USD"
```

## Integration Points

### 1. Data Providers (`yahoofinance/api/providers/hybrid_provider.py`)

All data fetching methods use ticker normalization:
- `get_ticker_info()`: Normalizes tickers before fetching data
- `batch_get_ticker_info()`: Handles batch normalization
- `get_price_data()`: Ensures consistent ticker handling
- Returns normalized tickers in response data

### 2. Position Sizing (`yahoofinance/utils/data/format_utils.py`)

Position sizing uses normalized tickers for geographic risk assessment:
- `calculate_position_size()`: Gets geography from normalized ticker
- Applies correct multipliers (0.5x for HK, 0.75x for EU, etc.)

### 3. Portfolio Filtering (`trade_modules/trade_filters.py`)

Trading filters use equivalence checking:
- `filter_new_opportunities()`: Prevents suggesting equivalent stocks
- Compares all opportunities against normalized portfolio holdings

### 4. Display Systems

All CSV outputs and console displays use normalized tickers:
- Market data files show original exchange tickers
- Portfolio files maintain consistency
- Trade recommendations use normalized forms

## Testing

Comprehensive test suites ensure system reliability:

### Core Mapping Tests (`tests/core/config/test_ticker_mappings.py`)
- Validates mapping configuration structure
- Tests ticker normalization functions
- Verifies equivalence checking logic
- Tests geographic region detection
- Integration scenarios for portfolio filtering

### Utility Function Tests (`tests/utils/data/test_ticker_utils.py`)
- Tests all utility wrapper functions
- Validates format standardization
- Error handling and edge cases
- Backward compatibility functions
- Complete integration workflows

### Legacy Test Updates
- Updated existing tests to expect normalized ticker behavior
- Fixed GOOGL→GOOG normalization expectations
- Corrected HK ticker format handling

## Migration Guide

### For Developers

1. **Use centralized functions**: Always use `normalize_ticker()` instead of manual ticker handling
2. **Portfolio comparisons**: Use `check_equivalent_tickers()` for any stock equivalence checks
3. **Geographic calculations**: Use `get_geographic_region()` for risk multiplier determination
4. **Format standardization**: Use `standardize_ticker_format()` for input processing

### For Users

No user-facing changes required. The system automatically:
- Normalizes all ticker inputs
- Maintains consistent display formatting
- Prevents duplicate stock holdings
- Applies correct geographic risk multipliers

## Error Handling

The system gracefully handles edge cases:
- **None inputs**: Return None or appropriate defaults
- **Empty strings**: Return empty strings
- **Invalid formats**: Validate and reject malformed tickers
- **Unknown tickers**: Pass through unchanged with uppercase normalization
- **Case insensitivity**: All matching is case-insensitive

## Future Enhancements

Potential areas for expansion:
1. **Additional Dual-Listings**: Easy to add new mappings to configuration
2. **Exchange-Specific Data Preferences**: Could fetch from different exchanges based on data quality
3. **Dynamic Mapping Updates**: Could load mappings from external data sources
4. **Currency Considerations**: Could incorporate currency conversion for truly equivalent valuations

## Performance Considerations

- **Caching**: Mapping lookups use dictionary operations (O(1))
- **Batch Processing**: Efficient handling of ticker lists
- **Memory Usage**: Minimal memory overhead from mapping dictionaries
- **Thread Safety**: All functions are stateless and thread-safe

## Configuration Management

All mappings are centralized in `ticker_mappings.py`:
- Easy to add new dual-listings
- Single source of truth for all mappings
- Version-controlled configuration
- Clear documentation of mapping rationale

## Monitoring and Validation

The system includes validation functions:
- `validate_ticker_format()`: Ensures ticker format correctness
- Comprehensive test coverage for all edge cases
- Integration tests for end-to-end workflows
- Error logging for investigation of issues

This ticker normalization system ensures consistent, reliable handling of dual-listed stocks throughout the entire trading application, preventing errors and improving data quality.
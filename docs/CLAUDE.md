# CLAUDE.md - Technical Reference for etorotrade

This document serves as the technical reference for the etorotrade project, covering architecture, design patterns, best practices, and key components.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Modular Trade Components](#modular-trade-components)
3. [Performance Optimizations](#performance-optimizations)
4. [Provider Pattern](#provider-pattern)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Async Operations](#async-operations)
8. [Trading Criteria](#trading-criteria)
9. [Development Commands](#development-commands)
10. [Testing Guidelines](#testing-guidelines)
11. [Best Practices](#best-practices)

## System Architecture

The codebase follows a modern provider-based architecture with these key components:

1. **Provider Layer**: Abstract interfaces and implementations for data access
   - Base provider interfaces: `FinanceDataProvider`, `AsyncFinanceDataProvider`
   - Available provider implementations:
     - `YahooFinanceProvider`: Synchronous provider using yfinance
     - `AsyncYahooFinanceProvider`: Asynchronous provider using yfinance
     - `YahooQueryProvider`: Synchronous provider using yahooquery library
     - `AsyncYahooQueryProvider`: Asynchronous provider using yahooquery library
     - `HybridProvider`: Combined provider that uses YFinance with YahooQuery supplements
     - `AsyncHybridProvider`: Asynchronous combined provider that uses AsyncYahooFinance with AsyncYahooQuery supplements
   - Factory function: `get_provider()` for obtaining appropriate provider instances (defaults to hybrid)

2. **Core Layer**: Fundamental services and definitions
   - Error handling (`core.errors`), Configuration (`core.config`), Logging (`core.logging`), Type definitions (`core.types`)

3. **Utilities Layer**: Reusable components with specialized submodules
   - Network utilities: Rate limiting, pagination, circuit breakers
   - Data, Market, Date, and Async utilities

4. **Analysis Layer**: Domain-specific processing modules

5. **Presentation Layer**: Output formatting and display

6. **Modular Trade Components**: Specialized modules for trading operations (2025-01-06)
   - Analysis Engine (`trade_modules/analysis_engine.py`): Core trading logic and criteria evaluation
   - Output Manager (`trade_modules/output_manager.py`): Display formatting and file generation
   - Data Processor (`trade_modules/data_processor.py`): Data transformation and validation
   - CLI Handler (`trade_modules/cli_handler.py`): User interface and command processing

### Critical Components

Take extra care when modifying these critical components:

1. **Provider Interfaces**: Ensure all implementations maintain the contract
2. **Enhanced Rate Limiting System**: Prevents API throttling with smart adaptation
3. **Async Operation Support**: Maintains performance and concurrency control
4. **Error Handling Hierarchy**: Provides consistent error management
5. **Trading Criteria**: Core business logic
6. **Modular Trade Components**: Clean separation of trading functionality

## Modular Trade Components

The trading functionality has been modularized into specialized components for better maintainability and performance (implemented 2025-01-06):

### Analysis Engine (`trade_modules/analysis_engine.py`)

**Core Trading Logic**: Centralized criteria evaluation and recommendation engine
- **EXRET Calculation**: Vectorized expected return computation (`calculate_exret()`)
- **Action Calculation**: High-performance trading decision logic (`calculate_action_vectorized()`)
- **Opportunity Filtering**: Buy/sell/hold candidate identification
- **Confidence Assessment**: Analyst coverage validation
- **Performance**: >1.7M rows/second processing with vectorized pandas operations

```python
from trade_modules.analysis_engine import calculate_exret, calculate_action, AnalysisEngine

# High-performance analysis
df_with_exret = calculate_exret(market_df)
df_with_actions = calculate_action(df_with_exret)

# Full analysis engine
engine = AnalysisEngine()
results = engine.analyze_market(market_df)
```

### Output Manager (`trade_modules/output_manager.py`)

**Display and File Management**: Professional formatting and multi-format output
- **Console Display**: Color-coded tabular output with `tabulate`
- **CSV Export**: Clean data export for analysis
- **HTML Generation**: Interactive dashboards with responsive design
- **Result Formatting**: Consistent null handling and professional appearance

```python
from trade_modules.output_manager import display_and_save_results, OutputManager

# Display and save results
display_and_save_results(buy_df, "Buy Opportunities", "yahoofinance/output/buy.csv")

# Advanced output management
manager = OutputManager()
manager.export_results_to_files(results_dict, output_dir)
```

### Key Modularization Benefits

1. **Performance**: 127% faster API processing through optimized components
2. **Maintainability**: Clean separation of concerns and single responsibility
3. **Testability**: Each module has comprehensive test coverage (19+ test cases)
4. **Reusability**: Components can be used independently or together
5. **Scalability**: Easy to extend with new analysis or output formats

## Clean Display Output System

**Ultra-Clean User Experience** (2025-01-06): The application now provides enterprise-level clean output with zero noise.

### Display Suppression Features

- **Silent Processing**: All INFO/DEBUG/WARNING messages completely suppressed
- **No Progress Messages**: Processing happens silently without status updates
- **Error Filtering**: Comprehensive filtering of irrelevant API errors
- **Clean Tables**: Data tables display without surrounding noise
- **Professional Output**: Suitable for automated workflows and production environments

### Implementation Details

**Key Files Modified**:
- `yahoofinance/utils/async_utils/enhanced.py`: Silent processing statistics
- `yahoofinance/presentation/console.py`: Removed progress messages
- `yahoofinance/utils/display_helpers.py`: Suppressed dashboard generation messages
- `yahoofinance/core/di_container.py`: Logger level set to WARNING for clean output

**Configuration**:
```python
# App logger configured for clean output
app_logger.setLevel(logging.WARNING)  # Suppresses INFO messages
```

**Result**: Pure data display with zero operational noise - ideal for professional and automated use cases.

### Backward Compatibility

All existing imports continue to work through compatibility layer in `trade_modules/__init__.py`:

```python
# Legacy imports still work
from trade_modules import calculate_exret, display_and_save_results

# New modular imports
from trade_modules.analysis_engine import AnalysisEngine
from trade_modules.output_manager import OutputManager
```

## Performance Optimizations

Major performance improvements implemented in 2025-01-06 release:

### API Processing Performance

**127% Throughput Improvement**: Optimized rate limiting and batch processing
- **Before**: 171 tickers/minute average processing
- **After**: 390 tickers/minute average processing  
- **Batch Optimization**: Increased batch sizes from 10 to 25 requests
- **Delay Reduction**: Reduced base delay from 0.3s to 0.15s between calls

### DataFrame Operations Performance

**Vectorized Computing**: Replaced inefficient row-by-row operations
- **EXRET Calculation**: >1.7M rows/second with pandas vectorization
- **Action Calculation**: Vectorized criteria evaluation vs row-by-row apply()
- **Memory Optimization**: Reduced DataFrame copying and memory allocation
- **Large Dataset Performance**: 56% faster processing for 100+ ticker portfolios

### Rate Limiting Optimizations

**Enhanced Rate Limiting Configuration** (`yahoofinance/core/config/rate_limiting.py`):

```python
# Performance-optimized settings
RATE_LIMIT_CONFIG = {
    "base_delay": 0.15,        # Reduced from 0.3s (50% faster)
    "batch_size": 25,          # Increased from 10 (150% larger batches)
    "batch_delay": 0.1,        # Optimized batch processing
    "adaptive_strategy": True,  # Smart delay adjustment
    "success_threshold": 5,     # Quick delay reduction on success
}
```

### Performance Monitoring

**Built-in Benchmarking Tools** (`tools/performance_benchmark.py`):

```bash
# Run comprehensive performance benchmark
python tools/performance_benchmark.py

# Example output:
# 🚀 etorotrade Performance Benchmark
# ⚡ API Processing: 390 tickers/minute (127% improvement)  
# 📊 DataFrame Operations: 1.7M+ rows/second
# 💾 Memory Usage: Optimized with reduced copying
# ⏱️ Realistic Portfolio (50 tickers): <0.002s processing
```

### Configuration Tuning

**Optimized Settings**: Fine-tuned for maximum throughput while respecting API limits
- **Window Management**: 60-second sliding windows with 75 calls max
- **Success/Error Handling**: Quick adaptation based on response patterns
- **Ticker Prioritization**: VIP ticker support for critical holdings
- **Market Hours Awareness**: Adjusted delays based on trading hours

### Measurement and Validation

**Performance Testing**: Comprehensive benchmarks validate improvements
- **Unit Tests**: Performance assertions for critical functions
- **Integration Tests**: End-to-end timing validation  
- **Realistic Scenarios**: Tests with actual portfolio sizes (10-100 tickers)
- **Memory Profiling**: Leak detection and optimization validation

## Provider Pattern

The provider pattern abstracts data access behind consistent interfaces:

```python
# Using the provider pattern (recommended)
from yahoofinance import get_provider

# Get the default hybrid provider (combines YahooFinance + YahooQuery for best data)
provider = get_provider()
ticker_info = provider.get_ticker_info("AAPL")

# Specify a particular provider implementation if needed
yf_provider = get_provider(provider_name='yahoo')
yq_provider = get_provider(provider_name='yahooquery')

# Get asynchronous provider
async_provider = get_provider(async_api=True)
ticker_info = await async_provider.get_ticker_info("MSFT")

# Batch processing with async provider
batch_results = await async_provider.batch_get_ticker_info(["AAPL", "MSFT", "GOOG"])
```

### YahooQuery Integration Toggle

The hybrid provider can be configured to disable yahooquery integration when needed:

```python
# In yahoofinance/core/config.py
PROVIDER_CONFIG = {
    # Toggle yahooquery integration in hybrid provider
    "ENABLE_YAHOOQUERY": False,  # Set to False to disable yahooquery and prevent crumb errors
}
```

When the `ENABLE_YAHOOQUERY` flag is set to `False`:
- The HybridProvider and AsyncHybridProvider will skip all yahooquery API calls
- Both providers will still create yahooquery provider instances for interface compatibility
- Methods like `_supplement_with_yahooquery` will return early without making API calls
- Batch processing will skip the yahooquery supplementation step
- Data source will be marked as "yfinance" only, not "hybrid (yf+yq)"
- PEG ratio and PE Forward values might be less available for some tickers

This toggle is useful for:
1. Preventing "Failed to obtain crumb" errors when Yahoo Finance makes API changes
2. Reducing API call volume to avoid rate limiting
3. Speeding up operations when yahooquery supplementation isn't critical
4. Temporarily working around yahooquery package issues

The implementation preserves the provider interface so your code doesn't need to change whether yahooquery is enabled or not.

## Error Handling

A comprehensive error handling system is implemented throughout the codebase:

### Error Hierarchy
- **Centralized Error System**: Comprehensive exception hierarchy in `core.errors`
- **Error Hierarchy**:
  - `YFinanceError` - Base class for all errors
  - `ValidationError`, `APIError`, `NetworkError`, `DataError`, `ConfigError`
  - Always import errors from core.errors module
  - Use specific error types for better error handling
  - Include context information in error messages

### Error Handling Utilities

The codebase includes comprehensive error handling utilities in `yahoofinance/utils/error_handling.py`:

- **Error Context Enrichment**: Adds debugging context to errors
- **Error Translation**: Converts standard exceptions to custom hierarchy
- **Retry Decorator**: Automatic retry with exponential backoff
- **Safe Operation Decorator**: Graceful error handling with fallbacks

### Import Practices

Always follow these import guidelines:

1. Import directly from canonical source modules for new code:
   ```python
   from yahoofinance.utils.network.rate_limiter import AdaptiveRateLimiter
   from yahoofinance.utils.data.format_utils import format_number
   from yahoofinance.core.errors import YFinanceError, ValidationError
   ```

2. Use provider pattern for financial data access:
   ```python
   from yahoofinance import get_provider
   
   provider = get_provider()  # For synchronous operations
   async_provider = get_provider(async_api=True)  # For async operations
   ```

3. For error handling, always import from core.errors:
   ```python
   from yahoofinance.core.errors import YFinanceError, APIError, ValidationError
   ```

## Rate Limiting

The codebase includes a sophisticated adaptive rate limiting system optimized for direct API access:

### Key Features

- **Ultra-adaptive Rate Limiter**: Dynamically adjusts delays based on real-time API response patterns
- **Ticker Prioritization**: Different delay tiers for HIGH, MEDIUM, and LOW priority tickers
- **Smart Jitter**: Randomized delays to avoid predictable patterns and rate limiting detection
- **Region-aware Delays**: Optimized settings for US, European, and Asian markets
- **Market Hours Detection**: Automatically adjusts rate limiting based on market open/close status
- **Performance Metrics Tracking**: Comprehensive monitoring of API response times and success rates
- **Self-tuning Parameters**: Automatically adapts based on error rates and API responses
- **VIP Ticker Support**: Special handling for critical tickers with gentler backoff

### Rate Limiting Configuration

The rate limiting can be configured in `yahoofinance/core/config/rate_limiting.py` with optimized settings (updated 2025-01-06):

```python
# Rate limiting configuration - PERFORMANCE OPTIMIZED
RATE_LIMIT_CONFIG = {
    # Window and call limits
    "window_size": 60,         # 1 minute sliding window
    "max_calls": 75,           # Max 75 calls per minute window
    
    # Base delay parameters - OPTIMIZED FOR PERFORMANCE
    "base_delay": 0.15,        # 150ms between API calls (was 0.3s)
    "min_delay": 0.1,          # Minimum 100ms delay
    "max_delay": 2.0,          # Maximum 2s delay after errors (was 30s)
    
    # Batch processing - OPTIMIZED FOR THROUGHPUT
    "batch_size": 25,          # Larger batches for better throughput (was 10)
    "batch_delay": 0.1,        # Reduced delay between batches
    
    # Success/failure adjustments
    "success_threshold": 5,    # Reduce delay after 5 consecutive successes
    "success_delay_reduction": 0.8,  # 20% reduction on success streak
    "error_threshold": 2,      # Increase delay after 2 consecutive errors
    "error_delay_increase": 1.5,     # 50% increase on error streak
    
    # Ticker priority tiers
    "ticker_priority": {
        "HIGH": 0.7,    # 30% faster for important tickers
        "MEDIUM": 1.0,  # Standard delay
        "LOW": 1.5,     # 50% slower for problematic tickers
    },
    
    # Special ticker sets
    "vip_tickers": set(),      # Always process with highest priority
    "slow_tickers": set(),     # Always process with lowest priority
    
    # Adaptive strategy settings
    "adaptive_strategy": True,  # Self-tuning system
    "monitor_interval": 60,     # Check every minute
}
```

### Using Rate Limiting in Code

The rate limiter can be used directly or through decorators:

```python
# Using the decorator (recommended)
from yahoofinance.utils.network.rate_limiter import rate_limited

@rate_limited(ticker_arg="ticker")
def get_ticker_data(ticker):
    # This function will be automatically rate-limited
    # The rate limiter will extract the ticker from the first argument
    # and apply appropriate delays based on ticker priority
    pass

# For async functions
from yahoofinance.utils.async.rate_limiter import async_rate_limited

@async_rate_limited()
async def get_ticker_data_async(ticker):
    # This async function will be rate-limited
    pass
```

### Rate Limiting Metrics and Monitoring

You can retrieve real-time rate limiting statistics:

```python
from yahoofinance.utils.network.rate_limiter import global_rate_limiter

# Get current stats
stats = global_rate_limiter.get_stats()
print(f"Current delay: {stats['current_delay']:.3f}s")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Request count: {stats['total_calls']}")
```

## Async Operations

The codebase includes comprehensive async support with:

- **AsyncRateLimiter**: Thread-safe rate limiting with adaptive behavior
- **PriorityAsyncRateLimiter**: Priority-based quotas with token bucket algorithm
- **Batch Processing**: Efficient parallel execution with proper error handling
- **Adaptive Concurrency**: Dynamic adjustment based on performance metrics
- **Region-aware Rate Limiting**: Optimized for different markets

### Example Usage

```python
import asyncio
from yahoofinance import get_provider
from yahoofinance.utils.async_utils.helpers import prioritized_batch_process, adaptive_fetch

# Basic provider usage
async def fetch_data():
    # Async hybrid provider
    provider = get_provider(async_api=True)
    info = await provider.get_ticker_info("MSFT")
    
    # Batch processing (much faster with hybrid provider)
    batch_results = await provider.batch_get_ticker_info(["AAPL", "MSFT", "GOOG"])

# Advanced usage with prioritization
async def prioritized_fetch():
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA", "META", "JPM", "V", "JNJ"]
    high_priority = ["AAPL", "MSFT"]  # Process these first
    
    async def fetch_ticker(ticker):
        provider = get_provider(async_api=True)
        return await provider.get_ticker_info(ticker)
    
    # Prioritized processing
    results = await prioritized_batch_process(
        items=tickers,
        processor=fetch_ticker,
        high_priority_items=high_priority,
        batch_size=3,
        concurrency=5
    )

# Adaptive concurrency based on performance
async def adaptive_processing():
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "NVDA", "META", "JPM", "V", "JNJ"]
    
    async def fetch_ticker(ticker):
        provider = get_provider(async_api=True)
        return await provider.get_ticker_info(ticker)
    
    # Adaptive concurrency (starts low and increases based on success)
    results = await adaptive_fetch(
        items=tickers,
        fetch_func=fetch_ticker,
        initial_concurrency=3,
        max_concurrency=10
    )

asyncio.run(fetch_data())
```


## Trading Criteria

The codebase implements a sophisticated three-tier trading system that classifies stocks by market cap and applies appropriate risk-adjusted criteria. **All trading criteria are centralized in `yahoofinance/core/trade_criteria_config.py`** to ensure consistency across all components.

### Three-Tier Market Cap System (Updated 2025-01-27)

The system categorizes stocks into three tiers based on market capitalization for risk-appropriate trading decisions:

**VALUE Tier (≥$100B)**: Large-cap quality companies
- **Conservative Criteria**: 15% upside minimum, 70% buy ratings
- **Philosophy**: Reasonable returns for established blue chips
- **Risk Management**: Higher position size tolerance due to stability

**GROWTH Tier ($5B-$100B)**: Mid-cap growth companies  
- **Standard Criteria**: 20% upside minimum, 75% buy ratings
- **Philosophy**: Balanced risk/reward for established growers
- **Risk Management**: Standard position sizing and thresholds

**BETS Tier (<$5B)**: Small-cap speculative positions
- **Aggressive Criteria**: 25% upside minimum, 80% buy ratings
- **Philosophy**: High returns required for speculative investments
- **Risk Management**: Reduced position sizes, stricter fundamentals

### Market Cap Tier Classification

**M Column**: Shows tier classification (V/G/B) in all trading outputs
- **Implementation**: `TradingCriteria.get_market_cap_tier()` method
- **Data Source**: Parses market cap from CAP column (e.g., "3.67B" → $3.67B)
- **Fallback Logic**: Defaults to BETS tier if market cap unavailable

### Tier-Specific Trading Logic

**SELL Criteria** (tier-specific thresholds):
- **VALUE**: <5% upside OR <50% buy ratings OR <5% expected return
- **GROWTH**: <8% upside OR <60% buy ratings OR <8% expected return  
- **BETS**: <12% upside OR <70% buy ratings OR <10% expected return
- **Common**: Forward P/E >65, Poor fundamentals (EG <-15%, PP <-35%), High risk factors

**BUY Criteria** (tier-specific thresholds):
- **VALUE**: ≥15% upside, ≥70% buy ratings, ≥10% expected return
- **GROWTH**: ≥20% upside, ≥75% buy ratings, ≥15% expected return
- **BETS**: ≥25% upside, ≥80% buy ratings, ≥20% expected return
- **Common Requirements**: Valid beta (0.25-3.0), PE ratios in range, sufficient analyst coverage

### Centralized Configuration

**File**: `yahoofinance/core/trade_criteria_config.py`
- **Single Source of Truth**: All trading logic, thresholds, and constants
- **TradingCriteria Class**: Contains tier-specific and legacy criteria
- **Tier Methods**: `get_tier_criteria()`, `get_market_cap_tier()` for dynamic classification
- **Consistent Logic**: Same criteria used for ACT column, filtering, and color coding

### Key Implementation Details

1. **Centralized Evaluation**: `TradingCriteria.calculate_action(row)` provides consistent logic
2. **Column Normalization**: `normalize_row_for_criteria()` maps display columns to internal names
3. **Backward Compatibility**: Legacy TRADING_CRITERIA dict maintained in config.py
4. **Filter Delegation**: All filtering functions delegate to centralized logic

### Customizing Criteria

To modify trading criteria, edit **only** `yahoofinance/core/trade_criteria_config.py`:

```python
class TradingCriteria:
    # Confidence thresholds
    MIN_ANALYST_COUNT = 5
    MIN_PRICE_TARGETS = 5
    
    # SELL criteria thresholds
    SELL_MAX_UPSIDE = 5.0              # Sell if upside < 5%
    SELL_MIN_BUY_PERCENTAGE = 65.0     # Sell if buy% < 65%
    # ... other SELL criteria
    
    # BUY criteria thresholds  
    BUY_MIN_UPSIDE = 20.0              # Buy if upside >= 20%
    BUY_MIN_BUY_PERCENTAGE = 85.0      # Buy if buy% >= 85%
    # ... other BUY criteria
```

Changes automatically apply to:
- ACT column calculation
- Color coding (green/red/yellow)
- Buy/Sell/Hold opportunity filtering
- All output files and dashboards

## Robust Target Price Mechanism

The codebase implements a **sophisticated quality validation system** for analyst price targets to ensure reliable upside calculations and trading decisions.

### Core Implementation

**Primary Module**: `yahoofinance/utils/data/price_target_utils.py`
- **`calculate_price_target_robustness()`**: Comprehensive quality scoring (0-100)
- **`validate_price_target_data()`**: Determines confidence levels and recommended actions
- **`get_preferred_price_target()`**: Returns quality-validated target price

**Integration Points**:
- **`trade.py:986-988, 4020-4021`**: Main upside calculation pipeline
- **`yahoofinance/utils/data/format_utils.py:60-82`**: `calculate_validated_upside()` function

### Quality Assessment Algorithm

**Robustness Scoring Factors** (stricter penalties as of latest update):
```python
# Spread penalties (percentage of median price)
if spread_percent > 100:    score -= 50  # Extreme spread
elif spread_percent > 75:   score -= 35  # Very high spread  
elif spread_percent > 50:   score -= 25  # High spread
elif spread_percent > 30:   score -= 15  # Moderate spread

# Quality grading thresholds (stricter as of latest update)
Grade A: score >= 85  # Excellent consensus (was 80)
Grade B: score >= 70  # Good quality (was 65)  
Grade C: score >= 55  # Moderate quality (was 50)
Grade D: score >= 40  # Poor quality (was 35)
Grade F: score < 40   # Unreliable data
```

**Additional Penalties**:
- Mean-median difference >10%: Indicates outlier skewness
- Outlier ratio >100%: Extreme targets relative to consensus
- Low analyst coverage <5: Insufficient data confidence
- Extreme targets >300% vs current price: Unrealistic projections

### Trading Decision Integration

**Data Flow**:
```
Provider Data (median/mean/high/low targets)
    ↓
calculate_price_target_robustness() → Quality grade (A-F)
    ↓  
validate_price_target_data() → Confidence level + recommended action
    ↓
get_preferred_price_target() → Quality-validated target price
    ↓
calculate_validated_upside() → Robust upside calculation
    ↓
TradingCriteria.calculate_action() → BUY/SELL/HOLD recommendation
```

**Quality-Based Actions**:
- **Grades A-C**: Uses median target with appropriate confidence marking
- **Grade D**: Still participates but flagged for manual review
- **Grade F**: Excluded from trading recommendations (falls back to simple calculation)

### Implementation Examples

**High-Quality Case (Grade A)**:
```python
# GNFT.PA: €9.0-11.5 range, 26.5% spread
robustness_score = 85  # Grade A
confidence_level = "high" 
action = "use_median"  # €9.45 median target
upside_source = "median_robust_high_confidence"
```

**Poor-Quality Case (Grade D)**:
```python
# Hypothetical: $50-150 range, 100% spread, few analysts  
robustness_score = 45  # Grade D
confidence_level = "low"
action = "manual_review"  # Still uses median but flagged
upside_source = "median_robust_low_confidence"
```

**Excluded Case (Grade F)**:
```python
# Extreme outliers or <20 score
robustness_score = 15  # Grade F
action = "exclude"
fallback = calculate_upside(price, simple_median)  # Graceful degradation
```

### Key Benefits

1. **Outlier Resistance**: Filters extreme analyst predictions that skew averages
2. **Quality Transparency**: Provides confidence indicators for all calculations  
3. **Fallback Protection**: Gracefully degrades to simple median if validation fails
4. **Consistent Integration**: Same quality assessment used across all trading decisions

### Monitoring and Debugging

**Quality Metrics Available**:
- Robustness score (0-100)
- Quality grade (A-F)
- Spread percentage
- Mean-median difference
- Warning flags array
- Confidence level

**Debug Functions**:
```python
from yahoofinance.utils.data.price_target_utils import calculate_price_target_robustness

robustness = calculate_price_target_robustness(mean, median, high, low, price, analyst_count)
print(f"Score: {robustness['robustness_score']}, Grade: {robustness['quality_grade']}")
print(f"Warnings: {robustness['warning_flags']}")
```

## Position Sizing System

The codebase implements a sophisticated position sizing system that calculates recommended trade sizes based on portfolio configuration, risk management, and market analysis.

### Position Size Configuration (Updated 2025-01-17)

**File**: `yahoofinance/core/config.py` - PORTFOLIO_CONFIG section
- **Portfolio Value**: $450,000 total portfolio value
- **Position Limits**: $1,000 minimum, $45,000 maximum ($45K = 10% max position)
- **Base Position**: 0.5% of portfolio = $2,250 for standard positions
- **High Conviction**: Up to 10% of portfolio = $45,000 for exceptional opportunities
- **High Conviction Criteria**: EG >15% AND PP >0% AND EXRET >20%

### Position Sizing Logic

**Key Implementation**: `yahoofinance/utils/data/format_utils.py:calculate_position_size()`

1. **Base Calculation**: Starts with base position (0.5% of portfolio = $2,250)
2. **EXRET Adjustment**: Higher expected returns get larger positions
   - EXRET >= 15%: High conviction multiplier (2-4x base)
   - EXRET 10-15%: Moderate increase (1.5-2x base)
   - EXRET < 10%: Standard or reduced position
3. **Market Cap Scaling**: 
   - Large cap (>$50B): Can support larger positions
   - Mid cap ($10-50B): Standard scaling
   - Small cap (<$10B): Reduced positions for higher risk
4. **Safety Limits**: All positions capped at $1K-$40K range

### SIZE Column Display

**Integration Points**:
- **Data Processing**: `trade.py:_process_data_for_display()` converts CAP strings to numeric market_cap
- **Position Calculation**: Converts market cap + EXRET + portfolio rules → position size
- **Display Formatting**: $2,000 → "2k", $7,500 → "7.5k", $15,000 → "15k"

### Position Sizing Examples

```python
# Example calculations based on portfolio config:
# Portfolio: $450K, Base: 0.5% = $2,250

# Low EXRET stock (EXRET=4.9%)
# → Base position with small cap reduction → ~$2,000 → "2k"

# Medium EXRET stock (EXRET=12.5%)  
# → Base * 1.8 * market_cap_multiplier → ~$7,500 → "7.5k"

# High EXRET stock (EXRET=18.2%)
# → High conviction * market_cap_multiplier → ~$15,000 → "15k"
```

### Exclusions and Special Cases

- **ETFs and Commodities**: No position sizing (SIZE shows "--")
- **Low Market Cap**: Stocks under $500M market cap excluded
- **Insufficient Data**: Missing EXRET or market cap → no position size
- **Risk Management**: Automatic scaling down for high beta or volatile stocks

### Recent Fixes (2025-01-05)

**Issue Resolved**: SIZE column was showing "--" instead of calculated values
- **Root Cause**: Configuration import failures and missing data conversion
- **Solution**: Added fallback configuration mechanism and CAP→market_cap conversion
- **Result**: SIZE column now displays proper position sizes like "2k", "7.5k", "15k"

The position sizing system provides intelligent, risk-adjusted trade recommendations that align with your portfolio management strategy and risk tolerance.

## Enhanced Fundamental Analysis Columns (Added 2025-01-17)

The system now includes additional fundamental analysis columns for more comprehensive stock evaluation:

### New Data Columns

**EG (Earnings Growth)**: Year-over-year earnings growth percentage
- **Implementation**: `yahoofinance/api/providers/yahoo_finance_base.py:_calculate_earnings_growth()`
- **Data Source**: quarterly_income_stmt (quarterly_earnings is deprecated)
- **Calculation**: Current quarter vs same quarter last year, with fallback to quarter-over-quarter
- **Display**: All values shown (no filtering) - negative growth displayed as negative percentages

**PP (Price Performance)**: 3-month price performance showing recent momentum  
- **Implementation**: `yahoofinance/analysis/performance.py:calculate_3month_price_performance()`
- **Data Source**: yfinance historical data with 90-day lookback
- **Calculation**: ((current_price - price_90_days_ago) / price_90_days_ago) * 100
- **Display**: All values shown (no filtering) - negative performance displayed as negative percentages

### Enhanced Position Sizing Integration

**High Conviction Criteria**: The system now identifies exceptional opportunities using:
- **EG >15%**: Strong earnings growth momentum
- **PP >0%**: Positive recent price performance  
- **EXRET >20%**: High expected return potential

**Position Sizing Impact**:
- **High Conviction Stocks**: Get significantly larger positions (up to 10% of portfolio)
- **Standard Stocks**: Use base position sizing (0.5% of portfolio)
- **Range**: $2,250 (0.5%) to $45,000 (10%) based on conviction level

### Trading Criteria Integration

**BUY Criteria Enhancement**:
- EG >= -10% (conditional - only checked if data available)
- PP >= -10% (conditional - only checked if data available)

**SELL Criteria Enhancement**:
- EG < -15% (conditional - only checked if data available)  
- PP < -20% (conditional - only checked if data available)

### Display Fix (2025-01-17)

**Issue Resolved**: EG and PP columns were showing "--" instead of actual values
- **Root Cause**: -10% filters in `yahoofinance/presentation/console.py` were hiding negative values
- **Solution**: Removed display filters to show all actual values including negative percentages
- **Result**: EG and PP columns now display actual values like "-54.3%" and "15.5%"

### Technical Implementation

**Data Flow**:
```
Provider Data (quarterly_income_stmt + historical prices)
    ↓
_calculate_earnings_growth() → EG value (e.g., -54.3)
calculate_3month_price_performance() → PP value (e.g., 15.5)
    ↓
_add_position_size_column() → Format as percentages: "-54.3%", "15.5%"
    ↓
Trading Criteria → Use EG/PP in buy/sell logic
    ↓
Position Sizing → Use EG/PP/EXRET for high conviction detection
```

**Key Files Modified**:
- `yahoofinance/api/providers/yahoo_finance_base.py`: Added earnings growth calculation
- `yahoofinance/analysis/performance.py`: Added 3-month price performance  
- `yahoofinance/presentation/console.py`: Fixed display filtering (removed -10% filters)
- `yahoofinance/core/trade_criteria_config.py`: Integrated EG/PP into trading logic
- `yahoofinance/utils/data/format_utils.py`: Enhanced position sizing with high conviction

The enhanced fundamental analysis provides deeper insights into stock quality and momentum, enabling more informed investment decisions with appropriate position sizing.

## Development Commands

**Application:**
- `python trade.py` - Run main app (P=Portfolio, M=Market, E=eToro, T=Trade, I=Manual)

**Testing:**
- `pytest tests/` - Run all tests
- `pytest tests/ --cov=yahoofinance` - Run tests with coverage

**Development:**
- `make lint` - Run code quality checks (black, isort, flake8, mypy)
- `make lint-fix` - Auto-fix formatting issues
- `make test` - Run tests

## Testing Guidelines

Tests are organized by component and use pytest markers for categorization:

**Test Structure:**
- `tests/unit/` - Unit tests for isolated components
- `tests/integration/` - Component interaction tests
- `tests/e2e/` - End-to-end workflow tests
- `tests/fixtures/` - Shared test fixtures

**Key Markers:**
- `@pytest.mark.asyncio` - Async functionality tests
- `@pytest.mark.api` - Tests requiring API access
- `@pytest.mark.slow` - Long-running tests

**Key Testing Practices:**

1. **Test Isolation**: Use unique identifiers and save/restore global state
2. **Thread Safety**: Use proper locks when testing shared resources
3. **Async Testing**: Clean up resources and cancel remaining tasks
4. **Circuit Breakers**: Save/restore global state with fixtures
5. **Rate Limiters**: Create isolated instances for testing



## Best Practices

1. Use the provider pattern for all new code (the hybrid provider is now the default)
2. Import utilities directly from their canonical sources
3. Prefer async mode for processing multiple tickers
4. Use batch methods when available
5. Handle potential None values in batch results
6. Use try/except blocks to handle potential errors
7. **Always use rate limiting decorators for API calls**:
   ```python
   from yahoofinance.utils.network.rate_limiter import rate_limited
   
   @rate_limited(ticker_arg="ticker")
   def my_api_function(ticker, other_args):
       # Function implementation
   ```
8. **Prioritize important tickers** for faster processing by adding them to VIP_TICKERS in config:
   ```python
   # In core/config.py
   RATE_LIMIT = {
       # Other settings...
       "VIP_TICKERS": {"AAPL", "MSFT", "GOOGL"}  # These will get faster processing
   }
   ```
9. Check `data_source` field in provider responses to see if data came from YFinance, YahooQuery, or both
10. Consider enabling or disabling yahooquery integration in the PROVIDER_CONFIG based on your needs
11. Remember that hybrid provider prioritizes PE Forward data from YahooQuery for more accurate trade criteria evaluation (when yahooquery is enabled)
12. Special handling exists for certain tickers like NVDA in the AsyncHybridProvider
13. Be aware that PEG ratio values might be missing or invalid, especially for high growth stocks; the hybrid provider includes logic to supplement these values (when yahooquery is enabled)
14. If you encounter "Failed to obtain crumb" errors, consider setting PROVIDER_CONFIG["ENABLE_YAHOOQUERY"] to False

### Key Features

- **Commodity Support**: Automatic mapping (GOLD→GC=F, OIL→CL=F, SILVER→SI=F)
- **Asset Types**: Stocks, ETFs, commodities, cryptocurrencies, international
- **Consistent Formatting**: Missing data displays as "--" across all outputs
- **Data Quality**: Professional appearance with proper null handling


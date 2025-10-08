# eToro Trade Analysis - Developer & AI Assistant Guide

## 🏗️ System Architecture

### Core Components

```
etorotrade/
├── trade.py                     # Main CLI entry point (41 lines)
├── trade_modules/               # Business Logic Layer
│   ├── analysis_engine.py       # Trading algorithms (1053 lines)
│   ├── trade_engine.py          # Orchestration & flow control
│   ├── trade_config.py          # 5-tier configuration system
│   ├── yaml_config_loader.py    # YAML config management
│   ├── data_processing_service.py # Ticker batch processing
│   └── boundaries/              # Clean architecture interfaces
│
├── yahoofinance/               # Data & Analysis Layer
│   ├── api/providers/          # Data provider implementations
│   │   ├── async_hybrid_provider.py # Primary data provider
│   │   ├── async_yahoo_finance.py   # yfinance async wrapper
│   │   └── async_yahooquery_provider.py # YahooQuery wrapper
│   ├── analysis/               # Financial calculations
│   ├── presentation/           # Output formatting (console/HTML)
│   │   └── console.py          # Main display logic (1463 lines)
│   └── utils/                  # Shared utilities
│
├── scripts/                    # Analysis Tools
│   ├── analyze_geography.py    # Geographic exposure (ETF-aware)
│   └── analyze_industry.py     # Sector analysis (ETF-aware)
│
└── config.yaml                 # Trading thresholds configuration
```

### Data Flow

```
User Input (trade.py)
    ↓
TradingEngine.run()
    ↓
Load Tickers (CSV/Manual)
    ↓
DataProcessingService.process_ticker_batch()
    ↓
AsyncHybridProvider.batch_get_ticker_info()
    ├→ YFinance API (primary)
    └→ YahooQuery API (supplement for PEG, etc.)
    ↓
Cache Layer (48hr TTL)
    ↓
analysis_engine.calculate_action_vectorized()
    ↓
Position Size Calculation
    ↓
ConsoleDisplay + HTML Reports
```

## 🔑 Key Concepts

### 5-Tier Market Cap System

The system classifies stocks into 5 tiers with region-specific criteria:

```python
# Market Cap Tiers (config.yaml)
MEGA:  ≥$500B  (e.g., AAPL, MSFT)
LARGE: $100-500B (e.g., NFLX, DIS)
MID:   $10-100B  (e.g., ROKU, SNAP)
SMALL: $2-10B   (e.g., small caps)
MICRO: <$2B     (e.g., penny stocks)

# Regions
US: United States (default)
EU: Europe (.L, .PA, .AS suffixes)
HK: Hong Kong/Asia (.HK suffix)
```

Each tier×region combination has specific thresholds in `config.yaml`:
- `min_upside`, `min_buy_percentage`, `min_exret`
- `min/max_beta`, `min/max_forward_pe`, `max_peg`
- `min_analysts`, `min_price_targets`

### Trading Logic (analysis_engine.py)

```python
def calculate_action_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Determines BUY/SELL/HOLD/INCONCLUSIVE for each stock.

    INCONCLUSIVE: < 4 analysts or < 4 price targets
    SELL: Any sell condition met (low upside, high PE, etc.)
    BUY: All buy conditions met
    HOLD: Default (between buy and sell)
    """
```

Key decision factors:
1. **Confidence Check**: Min 4 analysts + 4 price targets
2. **SELL Triggers** (ANY condition):
   - Upside below tier threshold
   - Buy% below tier threshold
   - EXRET below tier threshold
   - PEF > PET × 1.2 (deteriorating earnings)
   - Optional: High PEG, high beta, high short interest
3. **BUY Requirements** (ALL conditions):
   - Upside above tier threshold
   - Buy% above tier threshold
   - EXRET above tier threshold
   - PEF < PET × 1.1 (improving earnings)
   - Beta within range
   - Optional: Forward PE, PEG within limits

### Position Sizing Algorithm

```python
def calculate_position_size(market_cap, exret, ticker, earnings_growth, perf, beta):
    """Dynamic position sizing based on multiple factors"""

    base_position = 2500  # Base $2,500

    # Tier multipliers
    tier_mult = {
        'MEGA': 5,   # $12,500 base
        'LARGE': 4,  # $10,000 base
        'MID': 3,    # $7,500 base
        'SMALL': 2,  # $5,000 base
        'MICRO': 1   # $2,500 base
    }

    # EXRET multiplier (0.5x to 2.0x)
    # Additional adjustments for earnings, performance, beta

    return min(50000, max(1000, final_position))
```

## 🚀 Performance Optimizations

### Vectorized Operations
```python
# GOOD: Vectorized (7x faster)
df['action'] = np.where(condition, 'BUY', 'SELL')

# BAD: Row-by-row apply
df['action'] = df.apply(lambda row: calc_action(row), axis=1)
```

### Efficient Filtering
```python
# GOOD: Set operations (909x faster for 5000+ tickers)
portfolio_set = set(portfolio_tickers)
filtered = market_df[~market_df['ticker'].isin(portfolio_set)]

# BAD: Nested loops
for market_ticker in market_df['ticker']:
    for portfolio_ticker in portfolio_tickers:
        if market_ticker == portfolio_ticker: ...
```

### Batch Processing
- Process tickers in batches of 25
- Max 15 concurrent API requests
- Progress bars with ETA for long operations
- Cache with 48-hour TTL

## 🐛 Common Issues & Solutions

### Issue 1: Slow eToro Market Analysis (5544 tickers)
**Problem**: Takes 15+ minutes to analyze all eToro tickers
**Current Solution**: Batch processing with progress bars
**Note**: Market cap pre-filtering was attempted but removed (no API for lightweight market cap data)

### Issue 2: Missing PEG Ratios
**Problem**: yfinance often returns None for PEG
**Solution**: AsyncHybridProvider supplements with YahooQuery data

### Issue 3: Ticker Format Issues
**Problem**: eToro uses different formats (e.g., ASML.AS vs ASML.NV)
**Solution**: `ticker_utils.py` handles normalization and equivalence checking

### Issue 4: Rate Limiting
**Problem**: Yahoo Finance rate limits at ~2000 req/hour
**Solution**: Adaptive rate limiting with exponential backoff

### Issue 5: Circular Imports
**Problem**: Complex interdependencies between modules
**Solution**: Lazy imports, dependency injection, boundaries pattern

## 📝 Recent Changes Log

### September 2024 - Market Cap Pre-filtering (REVERTED)
- **Attempted**: Pre-filter stocks below $1B market cap to speed up eToro analysis
- **Issue**: No lightweight API endpoint for market cap only
- **Result**: Fetching market cap took as long as full data
- **Action**: Feature completely reverted, code cleaned

### September 2024 - 5-Tier System Implementation
- Upgraded from 3-tier (VALUE/GROWTH/BETS) to 5-tier system
- Added MEGA and MICRO tiers for better granularity
- Region-specific thresholds (US/EU/HK)
- YAML-based configuration

### August 2024 - Performance Optimizations
- Vectorized operations in analysis_engine
- Set-based filtering for portfolio exclusion
- Async batch processing with progress bars
- Cache system implementation

## 🧪 Testing Strategy

### Running Tests
```bash
# All tests
pytest tests/

# Specific module
pytest tests/unit/trade_modules/test_analysis_engine_coverage.py

# With coverage
pytest --cov=trade_modules --cov-report=html

# Linting
flake8 trade_modules/ --max-line-length=120
```

### Key Test Files
- `tests/unit/trade_modules/test_analysis_engine_coverage.py` - Core trading logic
- `tests/unit/trade_modules/test_trade_engine_coverage.py` - Orchestration flow
- `tests/unit/utils/async/test_enhanced.py` - Async utilities
- `tests/e2e/test_trade_workflows.py` - End-to-end integration tests

### Test Data Patterns
```python
# Create test DataFrame
test_df = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT'],
    'upside': [20.0, 15.0],
    'buy_percentage': [85.0, 70.0],
    'analyst_count': [25, 18],
    'market_cap': ['3.14T', '2.85T']
})
```

## 🔧 Development Guidelines

### Adding New Features

1. **New Data Field**:
   - Add to `AsyncHybridProvider.get_ticker_info()`
   - Update `DataProcessingService._process_single_ticker()`
   - Add to display columns in `console.py`

2. **New Trading Criteria**:
   - Update `config.yaml` with thresholds
   - Modify `calculate_action_vectorized()` in analysis_engine
   - Add tests in `test_analysis_engine_coverage.py`

3. **New Analysis Script**:
   - Create in `scripts/` directory
   - Use existing patterns from `analyze_geography.py`
   - Import from `trade_modules` for consistency

### Code Patterns

```python
# Async data fetching
async def fetch_data(tickers):
    from yahoofinance.utils.async_utils.enhanced import process_batch_async

    results = await process_batch_async(
        items=tickers,
        processor=provider.get_ticker_info,
        batch_size=25,
        concurrency=15,
        show_progress=True
    )
    return results

# DataFrame processing
def process_market_data(df):
    # Always copy to avoid warnings
    df = df.copy()

    # Vectorized operations
    df['new_col'] = df['col1'] * df['col2']

    # Efficient filtering
    mask = (df['upside'] > 20) & (df['beta'] < 2)
    filtered_df = df[mask]

    return filtered_df
```

## 🔌 API Integration

### Provider Hierarchy
```
AsyncHybridProvider (primary)
├── AsyncYahooFinanceProvider (yfinance wrapper)
└── AsyncYahooQueryProvider (yahooquery supplement)
```

### Key API Limitations
- **No lightweight endpoints**: Must fetch full ticker info
- **Rate limits**: ~2000/hour for Yahoo Finance
- **Missing data**: PEG often None in yfinance
- **Delisted stocks**: Return empty/error responses

### Cache Strategy
- 48-hour TTL for market data
- Unified cache service via `trade_modules/cache_service.py`
- Backward compatibility via `yahoofinance/data/cache_compatibility.py`
- In-memory caching with TTL support

## 🎯 Common Tasks

### Analyze Specific Tickers
```python
from trade_modules.analysis_engine import calculate_action_vectorized

# Create DataFrame with ticker data
df = pd.DataFrame(ticker_data)

# Calculate actions
df['BS'] = calculate_action_vectorized(df)

# Filter results
buy_opportunities = df[df['BS'] == 'B']
```

### Process Large Ticker Lists
```python
# Use DataProcessingService for efficiency
from trade_modules.data_processing_service import DataProcessingService

service = DataProcessingService(provider, logger)
results_df = await service.process_ticker_batch(tickers, batch_size=25)
```

### Generate Reports
```python
from yahoofinance.presentation.console import MarketDisplay

display = MarketDisplay(provider)
display.display_stock_table(results, "Analysis Results")
display.save_to_csv(results, "output.csv")
```

## 📊 Data Structures

### Ticker Info Dictionary
```python
{
    'ticker': 'AAPL',
    'company': 'Apple Inc',
    'price': 185.50,
    'target_price': 210.00,
    'upside': 13.2,
    'buy_percentage': 76.0,
    'analyst_count': 25,
    'market_cap': '3.14T',
    'pe_forward': 28.5,
    'pe_trailing': 31.2,
    'peg_ratio': 2.1,
    'beta': 1.25,
    'short_percent': 0.8,
    'dividend_yield': 0.5,
    'earnings_date': '2024-10-25',
    'EXRET': 10.0,  # upside * buy% / 100
    'twelve_month_performance': 15.2,
    'earnings_growth': 8.5
}
```

## 🔒 Security & Best Practices

1. **Never commit sensitive data**:
   - Keep portfolio.csv in .gitignore
   - Don't log API keys or personal data

2. **Error handling**:
   - Always use try/except in data fetching
   - Provide fallback values for missing data
   - Log errors for debugging

3. **Performance**:
   - Use vectorized operations
   - Implement caching for expensive operations
   - Show progress bars for long operations

4. **Code quality**:
   - Run linting before commits
   - Keep functions under 50 lines
   - Write tests for new features

## 📚 Additional Resources

- [Position Sizing Details](POSITION_SIZING.md)
- [CI/CD Pipeline](CI_CD.md)
- [User Guide](USER_GUIDE.md)
- [GitHub Repository](https://github.com/weirdapps/etorotrade)

---
*Last updated: January 2025*
*Codebase cleaned and optimized - removed debug tests and unused monitoring modules*
*For user documentation, see [README.md](../README.md)*
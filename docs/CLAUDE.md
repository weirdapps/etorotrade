# Technical Reference - etorotrade

## 🏗️ Architecture Overview

### Clean Modular Design
```
etorotrade/
├── trade_modules/          # Business Logic Layer
│   ├── analysis_engine.py  # Trading algorithms
│   ├── trade_engine.py     # Orchestration
│   ├── cache_service.py    # Caching layer
│   └── boundaries/         # Clean interfaces
│
├── yahoofinance/          # Data Layer
│   ├── api/providers/     # Yahoo Finance/Query APIs
│   ├── analysis/          # Financial calculations
│   ├── presentation/      # Output formatting
│   └── utils/             # Shared utilities
│
└── scripts/               # Analysis Tools
    ├── analyze_geography.py  # Geographic exposure (ETF-aware)
    └── analyze_industry.py   # Sector analysis (ETF-aware)
```

### Key Design Patterns
- **Provider Pattern**: Swappable data sources (Yahoo Finance/Query)
- **Repository Pattern**: Abstract data persistence layer
- **Dependency Injection**: No circular dependencies
- **Service Layer**: Business logic separated from data access

## ⚡ Performance Characteristics

### Speed Metrics
- **Processing**: >1M rows/second (vectorized operations)
- **API Throughput**: 390 tickers/min (optimized rate limiting)
- **Portfolio Filter**: O(n+m) complexity (was O(n*m))
- **Cache Hit Rate**: ~80% for repeated operations

### Optimization Techniques
```python
# Vectorized operations example
df['ACT'] = np.where(
    df['condition'].values,
    'BUY',
    'SELL'
)  # 7x faster than apply()

# Efficient filtering
portfolio_set = set(portfolio_df['ticker'])
market_in_portfolio = market_df[
    market_df['ticker'].isin(portfolio_set)
]  # 909x faster for large datasets
```

## 🔐 Trading Logic

### Three-Tier Risk System
```python
TIER_THRESHOLDS = {
    'VALUE': {
        'market_cap_min': 100_000_000_000,  # $100B+
        'upside_threshold': 15,
        'consensus_threshold': 70
    },
    'GROWTH': {
        'market_cap_min': 5_000_000_000,    # $5B-$100B
        'upside_threshold': 20,
        'consensus_threshold': 75
    },
    'BETS': {
        'market_cap_min': 0,                # <$5B
        'upside_threshold': 25,
        'consensus_threshold': 80
    }
}
```

### Position Sizing Algorithm
```python
def calculate_position_size(exret, tier, portfolio_value):
    base_position = portfolio_value * 0.005  # 0.5% base
    
    # EXRET multiplier (0.5x to 5.0x)
    exret_mult = min(5.0, max(0.5, exret / 10))
    
    # Tier multiplier
    tier_mult = {'VALUE': 2.5, 'GROWTH': 1.5, 'BETS': 0.5}[tier]
    
    position = base_position * exret_mult * tier_mult
    return max(1000, min(40000, position))  # $1K-$40K bounds
```

## 🛠️ Development Workflow

### Quick Commands
```bash
# Run analysis
python trade.py -o t -t b              # Find BUY opportunities
python scripts/analyze_geography.py    # Geographic analysis
python scripts/analyze_industry.py     # Sector analysis

# Testing
pytest tests/unit/                     # Unit tests
pytest tests/integration/              # Integration tests
pytest --cov=trade_modules             # Coverage report

# Code quality
./tools/lint.sh                        # Run all linters
flake8 trade_modules/                  # Style check
mypy trade_modules/                    # Type checking
```

### Adding New Features
1. **Data Provider**: Implement `FinanceDataProvider` interface
2. **Analysis Module**: Add to `yahoofinance/analysis/`
3. **Trade Logic**: Extend `trade_modules/analysis_engine.py`
4. **Output Format**: Modify `yahoofinance/presentation/`

## 🔄 API Integration

### Provider Interface
```python
class FinanceDataProvider(ABC):
    @abstractmethod
    async def get_ticker_info(self, ticker: str) -> Dict:
        """Get comprehensive ticker data"""
        pass
    
    @abstractmethod
    async def get_analyst_info(self, ticker: str) -> AnalystData:
        """Get analyst recommendations"""
        pass
```

### Rate Limiting
- **Yahoo Finance**: 2000 requests/hour
- **YahooQuery**: Adaptive based on response times
- **Circuit Breaker**: Auto-disable after 5 consecutive failures

## 🐛 Error Handling

### Exception Hierarchy
```python
YFinanceError (base)
├── APIError          # External API failures
├── RateLimitError    # Rate limit exceeded
├── ValidationError   # Data validation issues
├── ConfigError       # Configuration problems
└── DataError         # Data processing errors
```

### Resilience Patterns
- **Exponential Backoff**: For transient failures
- **Circuit Breaker**: Prevent cascade failures
- **Fallback Provider**: YahooQuery → Yahoo Finance
- **Cache First**: Use cached data when API unavailable

## 📊 Data Flow

```
User Input → Trade CLI → Trading Engine
                              ↓
                    Data Provider (API/Cache)
                              ↓
                      Analysis Engine
                              ↓
                    Position Calculator
                              ↓
                     Output Manager
                              ↓
                    CSV/HTML Reports
```

## 🧪 Testing Strategy

### Test Categories
- **Unit Tests**: Individual functions/methods
- **Integration Tests**: Component interactions
- **Performance Tests**: Speed benchmarks
- **Mock Tests**: API response simulation

### Key Test Files
```
tests/
├── unit/
│   ├── test_analysis_engine.py
│   └── test_position_sizing.py
├── integration/
│   ├── test_trade_flow.py
│   └── test_portfolio_analysis.py
└── fixtures/
    └── mock_api_responses.py
```

## 🚀 Deployment

### Environment Variables
```bash
# Required for full functionality
ETORO_API_KEY=xxx
ETORO_USER_KEY=xxx

# Optional
ETOROTRADE_LOG_LEVEL=INFO
ETOROTRADE_CACHE_TTL=3600
```

### Docker Support
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "trade.py"]
```

## 📈 Recent Improvements (Jan 2025)

1. **Module Decomposition**: Split 3000+ line files into <200 line modules
2. **Circular Import Fix**: Lazy loading and dependency injection
3. **ETF Transparency**: Geographic and sector exposure analysis
4. **Asset Classification**: Proper handling of crypto, commodities, derivatives
5. **Performance**: 7x speed improvement through vectorization
6. **Test Coverage**: Increased to 90%+ for critical paths

## 🔗 Key Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `trade.py` | Main entry point | 41 |
| `trade_modules/analysis_engine.py` | Trading logic | 850 |
| `trade_modules/trade_engine.py` | Orchestration | 245 |
| `scripts/analyze_geography.py` | Geographic analysis | 265 |
| `scripts/analyze_industry.py` | Sector analysis | 295 |

---
*For user documentation, see [README.md](../README.md)*
*For CI/CD details, see [CI_CD.md](CI_CD.md)*
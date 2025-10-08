# eToro Trade Analysis Tool

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=coverage)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)

A quantitative analysis framework for systematic evaluation of equity securities using analyst consensus data and fundamental metrics.

![eToro Trade Analysis Tool](docs/assets/etorotrade.png)

## Overview

This tool implements a rules-based approach to security analysis by aggregating multiple data sources and applying consistent evaluation criteria across market capitalizations and geographic regions. It processes financial metrics through a multi-tier classification system to generate actionable trading signals.

## Technical Specifications

### Data Processing Capabilities
- **Throughput**: ~100 securities per second (vectorized operations)
- **Batch Processing**: 5,544 securities in approximately 15 minutes
- **API Optimization**: 48-hour cache layer reducing redundant API calls by ~80%
- **Concurrency**: 15 parallel request threads with adaptive rate limiting

### Data Sources
- **Primary**: Yahoo Finance API (yfinance)
- **Supplementary**: YahooQuery API (PEG ratios, missing metrics)
- **Coverage**: 20+ investment bank analyst recommendations
- **Update Frequency**: Real-time market data with cached analyst consensus

## Methodology

### 5-Tier Market Capitalization System

The framework employs a sophisticated five-tier classification with region-specific adjustments:

| Tier | Market Cap Range | Example Companies | Strategy Focus |
|------|------------------|-------------------|----------------|
| **MEGA** | ≥ $500B | AAPL, MSFT, GOOGL | Blue-chip stability |
| **LARGE** | $100B - $500B | NFLX, DIS, UBER | Established growth |
| **MID** | $10B - $100B | ROKU, SNAP, DOCN | Growth opportunities |
| **SMALL** | $2B - $10B | Emerging leaders | Higher growth potential |
| **MICRO** | < $2B | Micro-cap stocks | Speculative positions |

**Regional Adjustments:**
- **US**: Baseline criteria for NYSE/NASDAQ securities
- **EU**: Modified thresholds for European exchanges (.L, .PA, .AS)
- **HK**: Adjusted parameters for Hong Kong/Asian markets (.HK)

### Signal Generation

Trading signals are generated through a systematic evaluation process:

1. **Data Collection**: Aggregate analyst recommendations, price targets, and fundamental metrics
2. **Confidence Validation**: Require minimum 4 analysts and 4 price targets
3. **Criteria Application**: Apply tier and region-specific thresholds
4. **Signal Classification**: Categorize as BUY, SELL, HOLD, or INCONCLUSIVE

### Position Sizing Algorithm

Position sizes are calculated using a risk-adjusted framework:
- Base position scaled by market capitalization tier
- Adjustments for expected return (EXRET)
- Beta-weighted volatility adjustments
- Maximum position constraints

## Installation

```bash
# Clone repository
git clone https://github.com/weirdapps/etorotrade
cd etorotrade

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Analysis

```bash
# Interactive mode
python trade.py

# Analyze portfolio
python trade.py -o p

# Market screening
python trade.py -o m

# Trade signal generation
python trade.py -o t -t b  # Buy signals
python trade.py -o t -t s  # Sell signals
```

### Advanced Analysis

```bash
# Portfolio analysis with position sizing
python trade.py -o p -pv 50000  # $50,000 portfolio value

# Geographic exposure analysis
python scripts/analyze_geography.py

# Sector allocation analysis
python scripts/analyze_industry.py

# Specific ticker analysis
python trade.py -o i -t AAPL,MSFT,GOOGL
```

## Output Format

### Signal Definitions

| Signal | Criteria | Interpretation |
|--------|----------|----------------|
| **BUY** | Meets all tier-specific buy thresholds | Positive analyst consensus with favorable risk/reward |
| **SELL** | Triggers any sell condition | Deteriorating fundamentals or overvaluation |
| **HOLD** | Between buy and sell thresholds | Fairly valued at current levels |
| **INCONCLUSIVE** | Insufficient analyst coverage | Requires additional research |

### Key Metrics

- **UPSIDE**: Percentage difference between current price and analyst target
- **%BUY**: Percentage of analysts with buy recommendations
- **EXRET**: Expected return (upside × buy percentage / 100)
- **SIZE**: Calculated position size based on risk parameters
- **PP**: Twelve-month price performance
- **EG**: Year-over-year earnings growth
- **PEF/PET**: Forward P/E to Trailing P/E ratio comparison

### Output Files

The system generates both CSV and HTML reports in `yahoofinance/output/`:
- `portfolio.csv/html` - Current holdings analysis
- `market.csv/html` - Market screening results
- `buy.csv/html` - Securities meeting buy criteria
- `sell.csv/html` - Securities meeting sell criteria

## Configuration

### Portfolio Input Format

Create `yahoofinance/input/portfolio.csv`:
```csv
symbol,totalInvestmentPct,totalNetProfitPct,instrumentDisplayName
AAPL,5.2,12.5,Apple Inc
MSFT,4.8,8.3,Microsoft Corporation
```

### Threshold Customization

Trading thresholds can be modified in `config.yaml`:
- Tier-specific buy/sell criteria
- Regional adjustments
- Position size parameters
- Risk management constraints

## Architecture

### Performance Optimizations
- Vectorized pandas operations for efficient data processing
- Set-based filtering algorithms (O(n) vs O(n²))
- Asynchronous API requests with connection pooling
- Memory-efficient streaming for large datasets

### Error Handling
- Automatic retry with exponential backoff
- Graceful degradation for missing data
- Comprehensive logging for debugging
- Circuit breaker pattern for API failures

### Code Quality
- Type hints throughout codebase
- Comprehensive test coverage
- Continuous integration pipeline
- SonarCloud quality gates

## ETF Analysis

The tool provides transparency into ETF holdings:
- Geographic exposure decomposition
- Sector allocation analysis
- Underlying asset classification
- Concentration risk assessment

## Risk Considerations

### Limitations
- Analysis based on publicly available data
- No intraday trading signals
- No derivative strategies
- Historical performance not indicative of future results

### Important Disclaimers
- This tool provides analysis only, not investment advice
- All investment decisions should incorporate multiple sources
- Past signals do not guarantee future performance
- Users assume all investment risk

## Development

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/e2e/           # End-to-end tests

# Code quality checks
flake8 yahoofinance/ trade_modules/ --max-line-length=100
mypy yahoofinance/ trade_modules/ --ignore-missing-imports

# Coverage report
pytest --cov=yahoofinance --cov=trade_modules --cov-report=html
```

### Contributing
Contributions are welcome. Please ensure:
- All tests pass (`pytest tests/`)
- Code follows PEP 8 style guidelines
- Type hints are included for new functions
- Documentation is updated accordingly
- Security best practices are followed

## Documentation

- [User Guide](docs/USER_GUIDE.md) - Getting started and common workflows
- [Technical Architecture](docs/CLAUDE.md) - System design and implementation details
- [Position Sizing](docs/POSITION_SIZING.md) - Risk management algorithms and methodology
- [CI/CD Pipeline](docs/CI_CD.md) - Testing, quality gates, and deployment procedures

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please use the [GitHub issue tracker](https://github.com/weirdapps/etorotrade/issues).

---

*Last Updated: January 2025*

**Disclaimer**: This tool is designed for quantitative analysis and research purposes only. It does not constitute investment advice. Users should conduct their own due diligence and consider consulting with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.
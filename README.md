=== Follow or Copy PLESSAS on etoro ===

# Market Analysis and Portfolio Management Tool

A Python-based tool for analyzing stocks using data from Yahoo Finance. The tool provides a comprehensive market analysis report with various metrics to help in stock selection and portfolio management.

## Tools Overview

### 1. Market Analysis Tool (trade.py)

- Real-time stock data analysis
- Multiple data source options (Portfolio, Market, Manual)
- Comprehensive metrics and analysis
- Color-coded output for quick insights

### 2. News Aggregator (yahoofinance/news.py)

- Latest news from Google News API or Yahoo Finance
- Up to 5 most recent news items per ticker
- Support for both portfolio and manual ticker input
- Clean, formatted output with color coding

### 3. Earnings Calendar (yahoofinance/earnings.py)

- Track earnings announcements for major stocks
- Customizable date range
- Market cap and EPS estimates
- Automatic handling of pre/post-market announcements
- Coverage of major S&P 500 components across all sectors

### 4. Economic Calendar (yahoofinance/economics.py)

- Track major economic indicators using FRED API
- Coverage of key metrics:
  * Employment (Nonfarm Payrolls, Unemployment Rate, Initial Claims)
  * Inflation (CPI, Core CPI, PPI)
  * Growth (GDP, Retail Sales, Industrial Production)
- Customizable date range
- Previous and actual values for each indicator
- Impact level indicators (High/Medium)

### 5. Market Performance (yahoofinance/weekly.py, yahoofinance/monthly.py)

- Track weekly and monthly performance of major indices (DJI30, SP500, NQ100, VIX)
- Automatic calculation of last trading day prices
- Formatted output with change percentages
- Updates index.html with latest performance data

### 6. Portfolio Tracker (yahoofinance/scrape.py)

- Scrapes portfolio performance metrics from etoro
- Tracks daily, MTD, YTD, and 2YR returns
- Monitors risk metrics (Beta, Alpha, Sharpe, Sortino)
- Updates portfolio.html with latest data
- Color-coded console output for quick insights

## Usage

### Market Analysis

```bash
python trade.py
```

You'll be prompted to choose a data source:

- P: Load tickers from portfolio file (yahoofinance/input/portfolio.csv)
- M: Load tickers from market file (yahoofinance/input/market.csv)
- I: Manually input tickers (comma-separated)

### Stock News

```bash
python yahoofinance/news.py
```

You'll be prompted to:

1. Choose a news source:
   - G: Google News API
   - Y: Yahoo Finance
2. Select ticker input method:
   - P: Load tickers from portfolio file
   - I: Manually input tickers

### Earnings Calendar

```bash
python yahoofinance/earnings.py
```

You'll be prompted to:

1. Enter start date (YYYY-MM-DD format)
   - Press Enter to use today's date
2. Enter end date (YYYY-MM-DD format)
   - Press Enter to use start date + 7 days

### Market Performance

```bash
python yahoofinance/weekly.py  # For weekly performance
python yahoofinance/monthly.py # For monthly performance
```

Automatically calculates and displays:

- Weekly: Last Friday vs Previous Friday
- Monthly: Last month-end vs Previous month-end

### Economic Calendar

```bash
python yahoofinance/economics.py
```

You'll be prompted to:

1. Enter start date (YYYY-MM-DD format)
   - Press Enter to use today's date
2. Enter end date (YYYY-MM-DD format)
   - Press Enter to use start date + 7 days

The calendar will display:
- Major economic events in the specified date range
- Impact level (High/Medium)
- Previous and actual values (when available)
- Formatted table with event details

### Portfolio Tracking

```bash
python yahoofinance/scrape.py
```

Automatically fetches and displays:

- Portfolio returns (Today, MTD, YTD, 2YR)
- Risk metrics (Beta, Alpha, Sharpe, Sortino)
- Cash position

## Metrics Explanation

### Price and Target

- **PRICE**: Current stock price
- **TARGET**: Average analyst price target
- **UPSIDE**: Percentage difference between target and current price
- **EXRET**: Expected return (Upside × Buy %) - Used for ranking table

### Analyst Coverage

- **# T**: Number of analysts providing price targets
- **% BUY**: Percentage of analysts recommending Buy
- **# A**: Number of analysts providing ratings

### Valuation Metrics

- **PET**: Trailing P/E ratio (Price / Last 12 months earnings)
  - High: Stock might be expensive
  - Low: Might indicate value or problems
  - Industry comparison is important

- **PEF**: Forward P/E ratio (Price / Next 12 months expected earnings)
  - Lower than PET suggests expected earnings growth
  - Higher than PET suggests expected earnings decline

- **PEG**: Price/Earnings to Growth ratio
  - &lt; 1: Potentially undervalued
  - &gt; 1: Potentially overvalued
  - ~1: Fairly valued

### Risk Metrics

- **BETA**: Stock's volatility compared to the market
  - &gt; 1: More volatile than market (e.g., 1.5 = 50% more volatile)
  - &lt; 1: Less volatile than market (e.g., 0.5 = 50% less volatile)
  - = 1: Same volatility as market

### Income & Ownership

- **DIV %**: Dividend yield percentage
- **SI**: Short Interest - % of float shares sold short
  - High: Indicates bearish sentiment
  - Low: Less bearish sentiment
- **INS %**: Percentage of insider buy transactions
- **# INS**: Number of insider transactions

### Timing

- **EARNINGS**: Date of last earnings report
  - Used to track analyst reports and insider trading since last earnings

## Color Coding

The tool uses color coding for quick visual analysis:

- 🟢 **Green**: Strong Buy Signal (ALL conditions must be met)
  - More than 4 analysts providing price targets
  - More than 4 analysts providing ratings
  - Upside potential greater than 15%
  - Buy rating percentage above 65%

- 🔴 **Red**: Sell Signal (EITHER condition with sufficient coverage)
  - More than 4 analysts providing price targets AND upside potential less than 5%
  OR
  - More than 4 analysts providing ratings AND buy rating percentage below 50%

- 🟡 **Yellow**: Low Confidence Rating
  - 4 or fewer analysts providing price targets
  OR
  - 4 or fewer analysts providing ratings
  - Additional research strongly recommended

- ⚪ **White**: Hold/Neutral
  - Sufficient analyst coverage but metrics fall between buy/sell thresholds
  - Does not meet all conditions for buy signal
  - Does not trigger either sell condition

## Data Sources

Data is sourced from multiple APIs:

1. Yahoo Finance API (through yfinance package):
   - Real-time and historical price data
   - Analyst recommendations and price targets
   - Financial metrics and ratios
   - Insider trading information
   - Company fundamentals
   - Earnings announcements and estimates

2. Google News API:
   - Latest news articles and summaries

3. FRED API (Federal Reserve Economic Data):
   - Economic indicators and metrics
   - Employment statistics
   - Inflation data
   - Growth indicators

- Real-time and historical price data
- Analyst recommendations and price targets
- Financial metrics and ratios
- Insider trading information
- Company fundamentals
- Latest news articles and summaries
- Earnings announcements and estimates

## Testing

The project includes a comprehensive test suite that covers the main functionality of the application. The tests are organized into several categories:

### Unit Tests

- **MarketDisplay Tests**: Tests the core functionality of market data display and processing
  - Ticker loading from different sources
  - Stock report generation
  - Display formatting
  - Error handling

- **YFinanceClient Tests**: Tests the Yahoo Finance API client
  - Data retrieval and parsing
  - Error handling
  - Response formatting

- **DisplayFormatter Tests**: Tests the data formatting functionality
  - Number formatting
  - Percentage handling
  - Edge cases (missing data, zero values, negative values)

### Running Tests

To run the test suite:

1. Install test dependencies:

```bash
pip install -r requirements.txt
```

2. Run all tests with coverage report:

```bash
pytest --cov=yahoofinance tests/
```

3. Run specific test files:

```bash
pytest tests/test_market_display.py  # Run MarketDisplay tests
pytest tests/test_client.py          # Run YFinanceClient tests
pytest tests/test_formatting.py      # Run DisplayFormatter tests
```

### Test Coverage

The test suite covers:

- Data loading and validation
- API interaction and error handling
- Data formatting and display
- Edge cases and error conditions

## Setup Requirements

### Dependencies

- Python 3.x
- yfinance
- pandas
- tabulate
- tqdm
- python-dotenv (for Google News API)
- requests (for Google News API)
- beautifulsoup4 (for portfolio tracking)
- pytz (for timezone handling)

### API Keys Setup

To use all features, you'll need the following API keys:

1. Google News API (for news functionality):
   - Get an API key from <https://newsapi.org/>
   - Add to .env: GOOGLE_NEWS_API_KEY=your_api_key

2. FRED API (for economic calendar):
   - Get an API key from <https://fred.stlouisfed.org/docs/api/api_key.html>
   - Add to .env: FRED_API_KEY=your_api_key

### Input Files

Create CSV files in yahoofinance/input/:

- portfolio.csv: Your portfolio tickers (column name: "ticker")
- market.csv: Market watch list tickers (column name: "symbol")

### Output Files

The following files are automatically updated:

- yahoofinance/output/index.html: Market performance dashboard
- yahoofinance/output/portfolio.html: Portfolio performance dashboard

=== Follow or Copy PLESSAS on etoro ===

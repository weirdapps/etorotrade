=== Follow or Copy PLESSAS on etoro ===

# Market Analysis and Portfolio Management Tool

A Python-based tool for analyzing stocks using data from Yahoo Finance. The tool provides a comprehensive market analysis report with various metrics to help in stock selection and portfolio management.

## Tools Overview

### 1. Market Analysis Tool (trade.py)

- Real-time stock data analysis with advanced rate limiting:
  * Adaptive batch processing (15 tickers per batch)
  * Smart delay system based on success rates
  * Error pattern detection and handling
  * Progress tracking with estimated completion times
- Multiple data source options:
  * Portfolio file (CSV)
  * Market watchlist (CSV)
  * Manual ticker input
  * Automatic portfolio download
- Comprehensive metrics and analysis:
  * Price and target analysis
  * Analyst coverage tracking
  * Risk metrics calculation
  * Insider trading monitoring
- Intelligent output management:
  * Color-coded console display
  * CSV data export
  * HTML dashboard generation
  * Automatic file organization

### 2. News Aggregator (yahoofinance/news.py)

- Latest news from Yahoo Finance
- Up to 5 most recent news items per ticker
- Support for both portfolio and manual ticker input
- Clean, formatted output with color coding
- Intelligent caching system:
  * Caches news responses for 15 minutes
  * Reduces API calls and improves response times
  * Automatic cache cleanup for expired entries
- Sentiment analysis for each news article (-1 to +1 scale)
  * Color-coded sentiment scores (red for negative, yellow for neutral, green for positive)
  * Combined analysis of title and summary content using VADER sentiment analysis
  * Weighted scoring system (60% title, 40% summary)
  * Accurate sentiment detection for financial news context

### 3. Earnings Calendar (yahoofinance/earnings.py)

- Track earnings announcements for major stocks
- Customizable date range
- Market cap and EPS estimates
- Automatic handling of pre/post-market announcements
- Coverage of major S&P 500 components across all sectors
### 4. Economic Calendar (yahoofinance/econ.py)

- Real-time economic data from Federal Reserve Economic Data (FRED)
- Coverage of key economic indicators:
  * GDP Growth Rate
  * Unemployment Rate
  * CPI (Month-over-Month)
  * Federal Funds Rate
  * Industrial Production
  * Retail Sales
  * Housing Starts
  * Nonfarm Payrolls
  * Trade Balance
  * Initial Jobless Claims
- Features:
  * Automatic data scaling (K for thousands, M for millions, B for billions)
  * Percentage change calculations
  * Customizable date ranges
  * Previous vs current value comparisons
  * Properly formatted table output
- Frequency handling:
  * Weekly data (e.g., Initial Claims)
  * Monthly data (e.g., CPI, Unemployment)
  * Quarterly data (e.g., GDP)


### 5. Market Performance (yahoofinance/index.py)

- Track weekly and monthly performance of major indices (DJI30, SP500, NQ100, VIX)
- Interactive prompt to choose between weekly and monthly performance
- Automatic calculation of last trading day prices
- Formatted output with change percentages
- Formatted output with change percentages

### 6. Portfolio Tracker (yahoofinance/portfolio.py)

- Scrapes portfolio performance metrics from etoro
- Tracks daily, MTD, YTD, and 2YR returns
- Monitors risk metrics (Beta, Alpha, Sharpe, Sortino)
- Updates portfolio.html with latest data
- Color-coded console output for quick insights

### 7. Institutional Holders Analysis (yahoofinance/holders.py)

- Analyze institutional ownership for any stock
- View major holders breakdown including:
  * Insider ownership percentage
  * Institutional ownership percentage
  * Float ownership by institutions
  * Total number of institutional holders
- Detailed top 10 institutional holders information:
  * Shares held and percentage ownership
  * Position value in billions
  * Last reported date
- Support for analyzing multiple tickers in one session

## Usage

### Market Analysis

```bash
python trade.py
```

You'll be prompted to choose a data source:

- P: Load tickers from portfolio file
  * Option to use existing portfolio file (yahoofinance/input/portfolio.csv)
  * Option to download a new portfolio from pi-screener.com
- M: Load tickers from market file (yahoofinance/input/market.csv)
- I: Manually input tickers (comma-separated)

### Stock News

```bash
python -m yahoofinance.news
```

You'll be prompted to select ticker input method:
- P: Load tickers from portfolio file
- I: Manually input tickers

Each news article will display:
- Title and source
- Sentiment score (-1 to +1) with color coding
- Publication date
- Summary
- Link to full article

### Earnings Calendar

```bash
python -m yahoofinance.earnings
```

You'll be prompted to:

1. Enter start date (YYYY-MM-DD format)
   - Press Enter to use today's date
2. Enter end date (YYYY-MM-DD format)
   - Press Enter to use start date + 7 days

### Market Performance

```bash
python -m yahoofinance.index
```

You'll be prompted to:

1. Choose performance period:
   - W: Weekly performance (Last Friday vs Previous Friday)
   - M: Monthly performance (Last month-end vs Previous month-end)
### Economic Calendar

```bash
python -m yahoofinance.econ
```

You'll be prompted to:

1. Enter start date (YYYY-MM-DD format)
   - Press Enter to use default (last 30 days)
2. Enter end date (YYYY-MM-DD format)
   - Press Enter to use default (current date)

The calendar will display:
- Latest values for major economic indicators
- Percentage changes from previous readings
- Properly scaled values (K/M/B)
- Formatted table with indicator details


### Portfolio Tracking

```bash
python -m yahoofinance.portfolio
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
- **A**: Source of analyst ratings ('E' for post-earnings data, 'A' for all-time data)

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

Data is sourced from Yahoo Finance API (through yfinance package):

1. Market Data:
   - Real-time and historical price data
   - Analyst recommendations and price targets
   - Financial metrics and ratios
   - Insider trading information
   - Company fundamentals
   - Earnings announcements and estimates

2. Economic Data:
   - Market indices performance
   - Sector performance
   - Economic event tracking
   - Market sentiment indicators

3. News and Analysis:
   - Latest news articles and summaries
   - Sentiment analysis
   - Company announcements
   - Market commentary

4. Company Information:
   - Corporate fundamentals
   - Institutional holdings
   - Insider transactions
   - Financial statements
   - Key statistics


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

- **News Tests**: Tests the news functionality
  - Sentiment analysis accuracy using VADER
  - Color coding logic
  - News formatting and display

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
pytest tests/test_news.py           # Run News tests
```

### Test Coverage

The test suite covers:

- Data loading and validation
- API interaction and error handling
- Data formatting and display
- Edge cases and error conditions
- Sentiment analysis accuracy

## Setup Requirements

### Dependencies

- Python 3.x
- yfinance
- pandas
- tabulate
- tqdm
- python-dotenv (for NewsAPI)
- requests (for NewsAPI)
- beautifulsoup4 (for portfolio tracking)
- pytz (for timezone handling)
- vaderSentiment (for sentiment analysis)

### API Keys Setup

To use all features, you'll need the following API key:

1. FRED API (for economic calendar):
   - Get an API key from <https://fred.stlouisfed.org/docs/api/api_key.html>
   - Add to .env: FRED_API_KEY=your_api_key

### Input Files

Create CSV files in yahoofinance/input/:

- portfolio.csv: Your portfolio tickers (column name: "ticker")
- market.csv: Market watch list tickers (column name: "symbol")

### Output Files

The following file is automatically updated:

- yahoofinance/output/performance.html: Portfolio performance dashboard

=== Follow or Copy PLESSAS on etoro ===

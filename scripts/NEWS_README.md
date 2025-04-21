# News Analysis Module

The News Analysis module in etorotrade provides capabilities for fetching and analyzing news sentiment for stocks using VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis.

## Features

- Fetch latest news for any ticker symbol
- Calculate sentiment scores for news headlines and summaries
- Clean and format news text for better analysis
- Special handling for financial terms
- Color-coded output for sentiment visualization
- Both synchronous and asynchronous interfaces

## Installation

The news module requires the `vaderSentiment` package:

```bash
pip install vaderSentiment>=3.3.2
```

## Usage

### Basic Example

```python
from yahoofinance.analysis.news import fetch_yahoo_news

# Fetch and display news for Apple
fetch_yahoo_news("AAPL")
```

### Sentiment Analysis

```python
from yahoofinance.analysis.news import calculate_sentiment

# Calculate sentiment for a news headline and summary
sentiment = calculate_sentiment(
    "Company reports record profits and beats expectations", 
    "Strong growth seen across all business segments"
)

print(f"Sentiment score: {sentiment:.2f}")  # Scale: -1.0 (negative) to 1.0 (positive)
```

### Async Processing

```python
import asyncio
from yahoofinance.analysis.news import compare_ticker_news_async

async def get_news_for_portfolio():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    await compare_ticker_news_async(tickers)

# Run the async function
asyncio.run(get_news_for_portfolio())
```

## Command Line Usage

You can run the news module directly:

```bash
python -m yahoofinance.analysis.news AAPL MSFT
```

Or without arguments for interactive mode:

```bash
python -m yahoofinance.analysis.news
```

## Understanding Sentiment Scores

The sentiment analysis uses VADER to produce a compound score between -1.0 (extremely negative) and 1.0 (extremely positive):

- **Positive sentiment**: score > 0.05
- **Neutral sentiment**: -0.05 ≤ score ≤ 0.05
- **Negative sentiment**: score < -0.05

## Financial Term Handling

The news module includes special handling for:

- Dollar amounts ($123, $4.5 billion)
- Percentages (15%, 3.5%)
- Stock symbols (AAPL, GOOGL)
- Financial terms (Q1, EPS, revenue)

These terms are properly filtered during sentiment analysis to avoid skewing the results.

## Error Handling

The module includes circuit breaker protection for API calls and comprehensive error handling to gracefully handle network issues or API limitations.

## Additional Resources

For more information, see:
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
"""
News analysis module for Yahoo Finance data.

This module provides functionality to fetch and analyze news for tickers,
including sentiment analysis using VADER.
"""

import asyncio
import html
import os
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf
from tabulate import tabulate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..core.config import FILE_PATHS, MESSAGES, PATHS
from ..core.errors import YFinanceError
from ..core.logging import get_logger
from ..utils.network.circuit_breaker import async_circuit_protected, circuit_protected


# Configure logger
logger = get_logger(__name__)


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "=" * 50)
    print(f"{title}")
    print("=" * 50 + f"{Colors.ENDC}")


def format_timestamp(timestamp):
    """Format a timestamp to a human readable format."""
    try:
        if not timestamp:
            return "N/A"
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "N/A"


def wrap_text(text, width=80, indent="   "):
    """Wrap text to a specific width with indentation."""
    if not text:
        return text
    # Remove HTML tags and decode HTML entities
    text = html.unescape(text)
    # Simple HTML tag removal (for basic tags)
    while "<" in text and ">" in text:
        start = text.find("<")
        end = text.find(">", start)
        if end == -1:
            break
        text = text[:start] + text[end + 1 :]
    wrapped_lines = textwrap.wrap(text, width=width)
    return f"\n{indent}".join(wrapped_lines)


def clean_html(text):
    """Remove HTML tags and normalize whitespace."""
    if not text:
        return ""

    # Remove HTML tags
    text = html.unescape(text)
    while "<" in text and ">" in text:
        start = text.find("<")
        end = text.find(">", start)
        if end == -1:
            break
        text = text[:start] + text[end + 1 :]

    # Normalize whitespace and remove any remaining source references
    text = " ".join(text.split())
    if "  " in text:  # Double space often indicates source reference
        text = text.split("  ")[0]
    return text


def clean_text_for_display(text):
    """Clean text for display by removing source names and normalizing."""
    if not text:
        return ""

    # Remove source name if present after dash
    if " - " in text:
        text = text.split(" - ")[0]

    # Clean HTML and normalize
    text = clean_html(text)

    # Remove source name if it appears at the end
    words = text.split()
    for i in range(len(words) - 1, -1, -1):
        if words[i] in ["TipRanks", "CNBC", "Reuters", "Bloomberg", "Yahoo", "Finance", "Nasdaq"]:
            text = " ".join(words[:i])
            break

    return text.strip()


def is_stock_symbol(word):
    """Check if word appears to be a stock symbol."""
    return word.isupper() and 1 <= len(word) <= 5 and word.isalpha()


def is_financial_term(word):
    """Check if word is a common financial term."""
    return word.lower() in ["q1", "q2", "q3", "q4", "fy", "eps", "revenue", "earnings"]


def should_skip_word(word, next_word=None):
    """Determine if a word should be skipped in sentiment analysis."""
    # Check for dollar amounts
    is_dollar = word.startswith("$") or (word.startswith("(") and word[1:].startswith("$"))
    if is_dollar:
        if next_word and next_word.lower() in ["billion", "million", "trillion"]:
            return True, True  # Skip this word and next
        return True, False

    # Check for percentages
    if word.endswith("%") or (word.endswith(")") and word[:-1].endswith("%")):
        return True, False

    return False, False


def clean_text_for_sentiment(text):
    """Clean text for sentiment analysis by removing financial terms and normalizing."""
    if not text:
        return ""

    # First clean for display
    text = clean_text_for_display(text)
    words = text.split()
    cleaned_words = []

    i = 0
    while i < len(words):
        word = words[i]

        # Skip financial indicators
        if is_stock_symbol(word) or is_financial_term(word):
            i += 1
            continue

        # Check if word should be skipped
        next_word = words[i + 1] if i + 1 < len(words) else None
        skip_current, skip_next = should_skip_word(word, next_word)

        if not skip_current:
            cleaned_words.append(word)

        i += 2 if skip_next else 1

    return " ".join(cleaned_words).strip()


def calculate_sentiment(title, summary):
    """
    Calculate sentiment score from -1 (most negative) to +1 (most positive)
    using both title and summary with title having more weight.
    """
    analyzer = SentimentIntensityAnalyzer()

    # Clean both title and summary
    clean_title = clean_text_for_sentiment(title)
    clean_summary = clean_text_for_sentiment(summary)

    # Title has 60% weight, summary has 40% weight
    title_weight = 0.6
    summary_weight = 0.4

    title_scores = analyzer.polarity_scores(clean_title)
    summary_scores = analyzer.polarity_scores(clean_summary)

    # Use compound scores which are already normalized between -1 and 1
    title_sentiment = title_scores["compound"]
    summary_sentiment = summary_scores["compound"]

    # Combine weighted sentiments
    combined_sentiment = title_weight * title_sentiment + summary_weight * summary_sentiment

    return combined_sentiment


def get_sentiment_color(sentiment):
    """Get color code based on sentiment value."""
    if sentiment < -0.05:  # VADER's threshold for negative
        return Colors.RED
    elif sentiment > 0.05:  # VADER's threshold for positive
        return Colors.GREEN
    return Colors.YELLOW


def get_url(content):
    """Safely extract URL from content."""
    if not content:
        return "N/A"

    url_locations = [("clickThroughUrl", "url"), ("canonicalUrl", "url"), ("link", None)]

    for main_key, sub_key in url_locations:
        if main_key in content:
            if sub_key:
                if isinstance(content[main_key], dict):
                    return content[main_key].get(sub_key, "N/A")
            else:
                return content[main_key]

    return "N/A"


def format_yahoo_news(news, ticker, limit=5):
    """Format and display news from Yahoo Finance."""
    print_section(f"LATEST NEWS FOR {ticker}")

    if not news:
        print(MESSAGES["NO_NEWS_FOUND_TICKER"].format(ticker=ticker))
        return

    for i, item in enumerate(news[:limit], 1):
        try:
            content = item.get("content", {}) or item  # Handle both formats

            # Extract news details
            title = content.get("title", "N/A")
            summary = content.get("summary", content.get("description", ""))
            sentiment = calculate_sentiment(title, summary)
            sentiment_color = get_sentiment_color(sentiment)

            # Display the news item
            print(f"\n{Colors.BOLD}• {title}{Colors.ENDC}")
            print(
                f"   {Colors.BLUE}Sentiment:{Colors.ENDC} "
                f"{sentiment_color}{sentiment:.2f}{Colors.ENDC}"
            )

            # Format and display metadata
            timestamp = format_timestamp(content.get("pubDate", ""))
            print(f"   {Colors.BLUE}Published:{Colors.ENDC} {timestamp}")

            provider = content.get("provider", {})
            provider_name = (
                provider.get("displayName", "N/A") if isinstance(provider, dict) else "N/A"
            )
            print(f"   {Colors.BLUE}Publisher:{Colors.ENDC} {provider_name}")

            # Display summary if available
            if summary:
                print(f"   {Colors.BLUE}Summary:{Colors.ENDC}")
                wrapped_summary = wrap_text(summary)
                print(f"   {wrapped_summary}")

            # Display URL
            url = get_url(content)
            print(f"   {Colors.BLUE}Link:{Colors.ENDC} {Colors.YELLOW}{url}{Colors.ENDC}")
            print("-" * 50)

        except YFinanceError as e:
            logger.error(MESSAGES["ERROR_PROCESSING_NEWS"].format(error=str(e)))
            print(MESSAGES["ERROR_PROCESSING_NEWS"].format(error=str(e)))
            continue


def get_portfolio_tickers():
    """Read tickers from portfolio.csv."""
    try:
        portfolio_path = FILE_PATHS["PORTFOLIO_FILE"]
        df = pd.read_csv(portfolio_path)

        # Check if 'ticker' or 'symbol' column exists
        if "ticker" in df.columns:
            tickers = df["ticker"].tolist()
        elif "symbol" in df.columns:
            tickers = df["symbol"].tolist()
        else:
            logger.error("Portfolio file doesn't have ticker or symbol column")
            return []

        return [ticker for ticker in tickers if not str(ticker).endswith("USD")]
    except YFinanceError as e:
        logger.error(MESSAGES["ERROR_READING_PORTFOLIO"].format(error=str(e)))
        print(MESSAGES["ERROR_READING_PORTFOLIO"].format(error=str(e)))
        return []


def get_user_tickers():
    """Get tickers from user input."""
    tickers_input = input(MESSAGES["PROMPT_ENTER_TICKERS"]).strip()
    return [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]


def get_ticker_source():
    """Get user's choice of ticker input method."""
    print(MESSAGES["PROMPT_TICKER_SOURCE"])
    print(MESSAGES["PROMPT_TICKER_SOURCE_OPTIONS"])

    while True:
        choice = input(MESSAGES["PROMPT_TICKER_SOURCE_CHOICE"]).strip().upper()
        if choice in ["P", "I"]:
            return choice
        print(MESSAGES["PROMPT_INVALID_CHOICE"])


@circuit_protected("yahoo_finance")
def fetch_yahoo_news(ticker):
    """
    Fetch and display news from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol
    """
    try:
        # Use yfinance directly for news
        stock = yf.Ticker(ticker)
        news = stock.news

        # Format and display the news
        if news:
            format_yahoo_news(news, ticker, limit=5)
        else:
            print(f"\n{MESSAGES['NO_NEWS_FOUND_TICKER'].format(ticker=ticker)}")
    except YFinanceError as e:
        logger.error(MESSAGES["ERROR_FETCHING_NEWS"].format(ticker=ticker, error=str(e)))
        print(f"\n{MESSAGES['ERROR_FETCHING_NEWS'].format(ticker=ticker, error=str(e))}")


async def fetch_yahoo_news_async(ticker):
    """
    Fetch and display news from Yahoo Finance asynchronously.

    Args:
        ticker: Stock ticker symbol
    """
    try:
        # Use asyncio.to_thread to call yfinance in a separate thread
        news = await asyncio.to_thread(lambda: yf.Ticker(ticker).news)

        # Format and display the news
        if news:
            format_yahoo_news(news, ticker, limit=5)
        else:
            print(f"\n{MESSAGES['NO_NEWS_FOUND_TICKER'].format(ticker=ticker)}")
    except YFinanceError as e:
        logger.error(MESSAGES["ERROR_FETCHING_NEWS_ASYNC"].format(ticker=ticker, error=str(e)))
        print(f"\n{MESSAGES['ERROR_FETCHING_NEWS_ASYNC'].format(ticker=ticker, error=str(e))}")


def display_news(ticker):
    """
    Display news for a ticker.

    Args:
        ticker: Stock ticker symbol
    """
    fetch_yahoo_news(ticker)


async def display_news_async(ticker):
    """
    Display news for a ticker asynchronously.

    Args:
        ticker: Stock ticker symbol
    """
    await fetch_yahoo_news_async(ticker)


async def compare_ticker_news_async(tickers):
    """
    Compare news for multiple tickers asynchronously.

    Args:
        tickers: List of stock ticker symbols
    """
    # Create tasks for fetching news for each ticker
    tasks = [fetch_yahoo_news_async(ticker) for ticker in tickers]

    # Run tasks concurrently
    await asyncio.gather(*tasks)


def main():
    """Main entry point for the news module."""
    print(f"{Colors.BOLD}Stock Market News{Colors.ENDC}")

    if len(os.sys.argv) > 1:
        # Get tickers from command line
        tickers = [ticker.strip().upper() for ticker in os.sys.argv[1:] if ticker.strip()]
    else:
        # Interactive mode
        choice = get_ticker_source()
        tickers = get_portfolio_tickers() if choice == "P" else get_user_tickers()

    if not tickers:
        print(MESSAGES["NO_TICKERS_FOUND"])
        return

    print(MESSAGES["INFO_FETCHING_NEWS"].format(tickers=", ".join(tickers)))

    # Fetch news for each ticker
    for ticker in tickers:
        fetch_yahoo_news(ticker)


if __name__ == "__main__":
    main()

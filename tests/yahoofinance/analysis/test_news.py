import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import yfinance as yf

from yahoofinance.analysis.news import (
    Colors,
    calculate_sentiment,
    format_timestamp,
    format_yahoo_news,
    get_portfolio_tickers,
    get_sentiment_color,
    get_ticker_source,
    get_url,
    get_user_tickers,
    wrap_text,
)


class TestNews(unittest.TestCase):
    def test_format_timestamp(self):
        """Test timestamp formatting for different inputs"""
        # Test Yahoo format
        self.assertEqual(format_timestamp("2024-02-11T14:30:00Z"), "2024-02-11 14:30:00")

        # Test invalid format
        self.assertEqual(format_timestamp("invalid"), "N/A")
        self.assertEqual(format_timestamp(None), "N/A")

    def test_wrap_text(self):
        """Test text wrapping functionality"""
        # Test basic wrapping
        text = "This is a test string that should be wrapped"
        wrapped = wrap_text(text, width=20)
        self.assertTrue(all(len(line.strip()) <= 20 for line in wrapped.split("\n")))

        # Test HTML removal
        html_text = "This is <b>bold</b> and <i>italic</i> text"
        wrapped = wrap_text(html_text)
        self.assertNotIn("<b>", wrapped)
        self.assertNotIn("</b>", wrapped)
        self.assertNotIn("<i>", wrapped)
        self.assertNotIn("</i>", wrapped)

        # Test empty input
        self.assertEqual(wrap_text(""), "")
        self.assertIsNone(wrap_text(None))

    @patch("builtins.print")
    def test_format_yahoo_news(self, mock_print):
        """Test Yahoo news formatting"""
        news = [
            {
                "content": {
                    "title": "Test Yahoo Title",
                    "summary": "Test Yahoo Summary",
                    "pubDate": "2024-02-11T14:30:00Z",
                    "provider": {"displayName": "Test Provider"},
                    "clickThroughUrl": {"url": "https://test.com"},
                }
            }
        ]

        format_yahoo_news(news, "AAPL")

        # Verify print calls contain expected content
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Test Yahoo Title" in str(call) for call in print_calls))
        self.assertTrue(any("Test Yahoo Summary" in str(call) for call in print_calls))
        self.assertTrue(any("Test Provider" in str(call) for call in print_calls))

    @patch("pandas.read_csv")
    def test_get_portfolio_tickers(self, mock_read_csv):
        """Test portfolio tickers retrieval"""
        # Mock DataFrame
        mock_df = pd.DataFrame({"ticker": ["AAPL", "MSFT", "BTC-USD"]})
        mock_read_csv.return_value = mock_df

        tickers = get_portfolio_tickers()
        self.assertEqual(tickers, ["AAPL", "MSFT"])  # BTC-USD should be filtered out

    @patch("builtins.input")
    def test_get_user_tickers(self, mock_input):
        """Test user ticker input"""
        mock_input.return_value = "AAPL, MSFT, GOOGL"
        tickers = get_user_tickers()
        self.assertEqual(tickers, ["AAPL", "MSFT", "GOOGL"])

        # Test empty input
        mock_input.return_value = ""
        tickers = get_user_tickers()
        self.assertEqual(tickers, [])

    @patch("builtins.input")
    def test_get_ticker_source(self, mock_input):
        """Test ticker source selection"""
        # Test valid inputs
        mock_input.side_effect = ["P"]
        self.assertEqual(get_ticker_source(), "P")

        mock_input.side_effect = ["I"]
        self.assertEqual(get_ticker_source(), "I")

        # Test invalid then valid input
        mock_input.side_effect = ["X", "P"]
        self.assertEqual(get_ticker_source(), "P")

    def test_get_url(self):
        """Test URL extraction from content"""
        test_cases = [
            ({"clickThroughUrl": {"url": "https://test.com"}}, "https://test.com"),
            ({"canonicalUrl": {"url": "https://test.com"}}, "https://test.com"),
            ({"link": "https://test.com"}, "https://test.com"),
            ({}, "N/A"),
        ]

        for content, expected_url in test_cases:
            with self.subTest(content=content):
                self.assertEqual(get_url(content), expected_url)

    def test_calculate_sentiment(self):
        """Test sentiment calculation with VADER"""
        test_cases = (
            # Positive case
            (
                "Company reports record profits and strong growth",
                "Excellent performance across all sectors",
                "positive",
            ),
            # Negative case
            (
                "Company faces significant losses and market decline",
                "Poor performance leads to layoffs",
                "negative",
            ),
            # Neutral case
            ("Company releases quarterly report", "Results to be discussed in meeting", "neutral"),
            # Empty summary case
            ("Company announces new product launch success", "", "positive"),
        )

        for title, summary, expected_type in test_cases:
            with self.subTest(title=title):
                sentiment = calculate_sentiment(title, summary)
                if expected_type == "positive":
                    self.assertGreater(sentiment, 0.05)  # VADER's positive threshold
                elif expected_type == "negative":
                    self.assertLess(sentiment, -0.05)  # VADER's negative threshold
                else:  # neutral
                    self.assertTrue(-0.05 <= sentiment <= 0.05)  # VADER's neutral range

    def test_get_sentiment_color(self):
        """Test sentiment color coding with VADER thresholds"""
        test_cases = [
            (-0.5, Colors.RED),  # Strong negative
            (-0.05, Colors.YELLOW),  # Borderline negative
            (0.0, Colors.YELLOW),  # Neutral
            (0.05, Colors.YELLOW),  # Borderline positive
            (0.5, Colors.GREEN),  # Strong positive
        ]

        for sentiment, expected_color in test_cases:
            with self.subTest(sentiment=sentiment):
                color = get_sentiment_color(sentiment)
                self.assertEqual(color, expected_color)


if __name__ == "__main__":
    unittest.main()

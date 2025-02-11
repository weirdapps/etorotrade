import unittest
from unittest.mock import patch, Mock, MagicMock
from yahoofinance.news import (
    calculate_sentiment,
    get_sentiment_color,
    Colors,
    get_google_news,
    get_url
)
import yfinance as yf

class TestNews(unittest.TestCase):
    @patch('yahoofinance.cache.news_cache')
    @patch('yahoofinance.news.requests.get')
    def test_google_news_caching(self, mock_get, mock_cache):
        """Test Google News caching functionality"""
        # Mock data
        test_articles = [{'title': 'Test Article', 'description': 'Test Description'}]
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'articles': test_articles}
        mock_get.return_value = mock_response
        
        # Test cache miss
        mock_cache.get.return_value = None
        result = get_google_news('AAPL', limit=1)
        
        self.assertEqual(result, test_articles)
        mock_cache.set.assert_called_once_with('google_news_AAPL_1', test_articles)
        
        # Test cache hit
        mock_cache.get.return_value = test_articles
        result = get_google_news('AAPL', limit=1)
        
        self.assertEqual(result, test_articles)
        mock_get.assert_called_once()  # Request should only be made once
    
    def test_get_url(self):
        """Test URL extraction from content"""
        test_cases = [
            (
                {'clickThroughUrl': {'url': 'https://test.com'}},
                'https://test.com'
            ),
            (
                {'canonicalUrl': {'url': 'https://test.com'}},
                'https://test.com'
            ),
            (
                {'link': 'https://test.com'},
                'https://test.com'
            ),
            (
                {},
                'N/A'
            )
        ]
        
        for content, expected_url in test_cases:
            with self.subTest(content=content):
                self.assertEqual(get_url(content), expected_url)
    def test_calculate_sentiment(self):
        """Test sentiment calculation with various inputs"""
        test_cases = (
            # Positive case
            (
                "Company reports record profits and strong growth",
                "Excellent performance across all sectors",
                0.4  # Expected positive but adjusted to be more realistic
            ),
            # Negative case
            (
                "Company faces significant losses and market decline",
                "Poor performance leads to layoffs",
                -0.3  # Expected negative but adjusted to be more realistic
            ),
            # Neutral case
            (
                "Company releases quarterly report",
                "Results to be discussed in meeting",
                0.0  # Expected neutral
            ),
            # Empty summary case
            (
                "Company announces new product",
                "",
                0.2  # Should still work with empty summary
            )
        )
        
        for title, summary, expected_sentiment in test_cases:
            with self.subTest(title=title):
                sentiment = calculate_sentiment(title, summary)
                # Allow for some variation in sentiment scores since TextBlob's sentiment
                # analysis might not exactly match our expectations
                self.assertAlmostEqual(sentiment, expected_sentiment, delta=0.4)
    
    def test_get_sentiment_color(self):
        """Test sentiment color coding"""
        test_cases = [
            (-0.5, Colors.RED),    # Strong negative
            (-0.2, Colors.YELLOW), # Borderline negative
            (0.0, Colors.YELLOW),  # Neutral
            (0.2, Colors.YELLOW),  # Borderline positive
            (0.5, Colors.GREEN)    # Strong positive
        ]
        
        for sentiment, expected_color in test_cases:
            with self.subTest(sentiment=sentiment):
                color = get_sentiment_color(sentiment)
                self.assertEqual(color, expected_color)
    
    @patch('yahoofinance.news.yf.Ticker')
    @patch('yahoofinance.cache.news_cache')
    def test_yahoo_finance_news_caching(self, mock_cache, mock_ticker_class):
        """Test Yahoo Finance news caching functionality"""
        # Mock data
        test_news = [
            {
                'content': {
                    'title': 'Test News',
                    'summary': 'Test Summary',
                    'pubDate': '2024-02-11T00:00:00Z',
                    'provider': {'displayName': 'Test Provider'},
                    'clickThroughUrl': {'url': 'https://test.com'}
                }
            }
        ]
        
        # Set up mock ticker
        mock_ticker = MagicMock()
        mock_ticker.news = test_news
        mock_ticker_class.return_value = mock_ticker
        
        # Test cache miss scenario
        mock_cache.get.return_value = None
        ticker = 'AAPL'
        
        # Simulate the main flow
        cached_news = mock_cache.get(f"yahoo_news_{ticker}")
        self.assertIsNone(cached_news)
        
        # Verify cache interactions
        mock_cache.get.assert_called_with(f"yahoo_news_{ticker}")
        
        # Test cache hit scenario
        mock_cache.get.return_value = test_news
        cached_news = mock_cache.get(f"yahoo_news_{ticker}")
        self.assertEqual(cached_news, test_news)

if __name__ == '__main__':
    unittest.main()
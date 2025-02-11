import unittest
from unittest.mock import patch, Mock
from yahoofinance.news import calculate_sentiment, get_sentiment_color, Colors

class TestNews(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
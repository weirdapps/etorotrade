"""
End-to-end tests for trade workflows.

These tests verify complete user workflows from data retrieval to 
decision-making, ensuring the system works correctly as a whole.
"""

import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from yahoofinance.display import MarketDisplay
from yahoofinance.api import get_provider


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return pd.DataFrame([
        {"symbol": "AAPL", "shares": 10, "cost": 150.0, "date": "2023-01-15"},
        {"symbol": "MSFT", "shares": 5, "cost": 280.0, "date": "2023-02-20"},
        {"symbol": "AMZN", "shares": 8, "cost": 125.0, "date": "2023-03-10"}
    ])


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return pd.DataFrame([
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
        {"symbol": "AMZN", "sector": "Consumer Cyclical"},
        {"symbol": "GOOGL", "sector": "Communication Services"},
        {"symbol": "META", "sector": "Communication Services"}
    ])


@pytest.mark.e2e
class TestPortfolioWorkflow:
    """Test portfolio analysis workflow from data loading to recommendation."""
    
    @patch("yahoofinance.display.MarketDisplay._generate_market_metrics")
    @patch("yahoofinance.display.pd.read_csv")
    @patch("yahoofinance.api.get_provider")
    def test_portfolio_analysis_workflow(self, mock_get_provider, mock_read_csv, 
                                        mock_generate_metrics, sample_portfolio_data):
        """Test complete workflow for portfolio analysis."""
        # Mock the provider
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Mock reading portfolio CSV
        mock_read_csv.return_value = sample_portfolio_data
        
        # Mock ticker data responses
        mock_provider.get_ticker_info.side_effect = lambda ticker: {
            "AAPL": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "current_price": 170.0,
                "market_cap": 2750000000000,
                "beta": 1.2,
                "pe_trailing": 25.0,
                "pe_forward": 22.0,
                "dividend_yield": 0.6,
                "peg_ratio": 1.5,
                "short_float_pct": 0.5
            },
            "MSFT": {
                "ticker": "MSFT",
                "name": "Microsoft Corporation",
                "sector": "Technology",
                "current_price": 330.0,
                "market_cap": 2450000000000,
                "beta": 1.0,
                "pe_trailing": 30.0,
                "pe_forward": 26.0,
                "dividend_yield": 0.8,
                "peg_ratio": 1.8,
                "short_float_pct": 0.7
            },
            "AMZN": {
                "ticker": "AMZN",
                "name": "Amazon.com Inc.",
                "sector": "Consumer Cyclical",
                "current_price": 140.0,
                "market_cap": 1450000000000,
                "beta": 1.3,
                "pe_trailing": 60.0,
                "pe_forward": 40.0,
                "dividend_yield": 0.0,
                "peg_ratio": 2.5,
                "short_float_pct": 0.9
            }
        }[ticker]
        
        # Mock price data responses
        mock_provider.get_price_data.side_effect = lambda ticker: {
            "AAPL": {
                "current_price": 170.0,
                "target_price": 190.0,
                "upside_potential": 11.8,
                "price_change": 2.5,
                "price_change_percentage": 1.5
            },
            "MSFT": {
                "current_price": 330.0,
                "target_price": 360.0,
                "upside_potential": 9.1,
                "price_change": 2.6,
                "price_change_percentage": 0.8
            },
            "AMZN": {
                "current_price": 140.0,
                "target_price": 170.0,
                "upside_potential": 21.4,
                "price_change": 2.9,
                "price_change_percentage": 2.1
            }
        }[ticker]
        
        # Mock analyst ratings responses
        mock_provider.get_analyst_ratings.side_effect = lambda ticker: {
            "AAPL": {
                "positive_percentage": 71.4,
                "total_ratings": 35,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 25,
                    "hold": 7,
                    "sell": 3
                }
            },
            "MSFT": {
                "positive_percentage": 85.7,
                "total_ratings": 35,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 30,
                    "hold": 4,
                    "sell": 1
                }
            },
            "AMZN": {
                "positive_percentage": 93.0,
                "total_ratings": 43,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 40,
                    "hold": 3,
                    "sell": 0
                }
            }
        }[ticker]
        
        # Skip actual metrics generation
        mock_generate_metrics.return_value = pd.DataFrame([
            {
                "ticker": "AAPL",
                "price": 170.0,
                "target_price": 190.0,
                "upside": 11.8,
                "buy_percentage": 71.4,
                "analyst_count": 35,
                "pe_trailing": 25.0,
                "pe_forward": 22.0,
                "peg_ratio": 1.5,
                "beta": 1.2,
                "market_cap": 2750000000000,
                "short_interest": 0.5,
                "recommendation": "HOLD",
                "company_name": "APPLE INC."
            },
            {
                "ticker": "MSFT",
                "price": 330.0,
                "target_price": 360.0,
                "upside": 9.1,
                "buy_percentage": 85.7,
                "analyst_count": 35,
                "pe_trailing": 30.0,
                "pe_forward": 26.0,
                "peg_ratio": 1.8,
                "beta": 1.0,
                "market_cap": 2450000000000,
                "short_interest": 0.7,
                "recommendation": "HOLD",
                "company_name": "MICROSOFT CORP"
            },
            {
                "ticker": "AMZN",
                "price": 140.0,
                "target_price": 170.0,
                "upside": 21.4,
                "buy_percentage": 93.0,
                "analyst_count": 43,
                "pe_trailing": 60.0,
                "pe_forward": 40.0,
                "peg_ratio": 2.5,
                "beta": 1.3,
                "market_cap": 1450000000000,
                "short_interest": 0.9,
                "recommendation": "BUY",
                "company_name": "AMAZON.COM INC"
            }
        ])
        
        # Create display instance and run portfolio analysis
        display = MarketDisplay()
        result = display.analyze_portfolio()
        
        # Verify the workflow executed correctly
        assert mock_read_csv.called
        assert mock_provider.get_ticker_info.call_count == 3
        assert mock_provider.get_analyst_ratings.call_count == 3
        assert mock_generate_metrics.called
        
        # Verify result contains expected data structure
        assert isinstance(result, pd.DataFrame)
        assert "ticker" in result.columns
        assert "recommendation" in result.columns
        assert len(result) == 3
        
        # Verify at least one ticker has a BUY recommendation
        assert "BUY" in result["recommendation"].values


@pytest.mark.e2e
class TestTradeWorkflow:
    """Test trade recommendation workflow."""
    
    @patch("yahoofinance.display.MarketDisplay._generate_market_metrics")
    @patch("yahoofinance.display.pd.read_csv")
    @patch("yahoofinance.api.get_provider")
    def test_buy_recommendations_workflow(self, mock_get_provider, mock_read_csv, 
                                         mock_generate_metrics, sample_market_data):
        """Test complete workflow for generating buy recommendations."""
        # Mock the provider
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Mock reading market CSV
        mock_read_csv.return_value = sample_market_data
        
        # Mock ticker data responses
        mock_provider.get_ticker_info.side_effect = lambda ticker: {
            "AAPL": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "current_price": 170.0,
                "market_cap": 2750000000000,
                "beta": 1.2,
                "pe_trailing": 25.0,
                "pe_forward": 22.0,
                "dividend_yield": 0.6,
                "peg_ratio": 1.5,
                "short_float_pct": 0.5
            },
            "MSFT": {
                "ticker": "MSFT",
                "name": "Microsoft Corporation",
                "sector": "Technology",
                "current_price": 330.0,
                "market_cap": 2450000000000,
                "beta": 1.0,
                "pe_trailing": 30.0,
                "pe_forward": 26.0,
                "dividend_yield": 0.8,
                "peg_ratio": 1.8,
                "short_float_pct": 0.7
            },
            "AMZN": {
                "ticker": "AMZN",
                "name": "Amazon.com Inc.",
                "sector": "Consumer Cyclical",
                "current_price": 140.0,
                "market_cap": 1450000000000,
                "beta": 1.3,
                "pe_trailing": 60.0,
                "pe_forward": 40.0,
                "dividend_yield": 0.0,
                "peg_ratio": 2.5,
                "short_float_pct": 0.9
            },
            "GOOGL": {
                "ticker": "GOOGL",
                "name": "Alphabet Inc.",
                "sector": "Communication Services",
                "current_price": 135.0,
                "market_cap": 1700000000000,
                "beta": 1.1,
                "pe_trailing": 27.0,
                "pe_forward": 24.0,
                "dividend_yield": 0.0,
                "peg_ratio": 1.3,
                "short_float_pct": 0.6
            },
            "META": {
                "ticker": "META",
                "name": "Meta Platforms",
                "sector": "Communication Services",
                "current_price": 310.0,
                "market_cap": 790000000000,
                "beta": 1.4,
                "pe_trailing": 29.0,
                "pe_forward": 22.0,
                "dividend_yield": 0.0,
                "peg_ratio": 1.4,
                "short_float_pct": 0.8
            }
        }[ticker]
        
        # Mock price data responses
        mock_provider.get_price_data.side_effect = lambda ticker: {
            "AAPL": {
                "current_price": 170.0,
                "target_price": 190.0,
                "upside_potential": 11.8,
                "price_change": 2.5,
                "price_change_percentage": 1.5
            },
            "MSFT": {
                "current_price": 330.0,
                "target_price": 360.0,
                "upside_potential": 9.1,
                "price_change": 2.6,
                "price_change_percentage": 0.8
            },
            "AMZN": {
                "current_price": 140.0,
                "target_price": 170.0,
                "upside_potential": 21.4,
                "price_change": 2.9,
                "price_change_percentage": 2.1
            },
            "GOOGL": {
                "current_price": 135.0,
                "target_price": 165.0,
                "upside_potential": 22.2,
                "price_change": 3.0,
                "price_change_percentage": 2.3
            },
            "META": {
                "current_price": 310.0,
                "target_price": 380.0,
                "upside_potential": 22.6,
                "price_change": 5.1,
                "price_change_percentage": 1.7
            }
        }[ticker]
        
        # Mock analyst ratings responses
        mock_provider.get_analyst_ratings.side_effect = lambda ticker: {
            "AAPL": {
                "positive_percentage": 71.4,
                "total_ratings": 35,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 25,
                    "hold": 7,
                    "sell": 3
                }
            },
            "MSFT": {
                "positive_percentage": 85.7,
                "total_ratings": 35,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 30,
                    "hold": 4,
                    "sell": 1
                }
            },
            "AMZN": {
                "positive_percentage": 93.0,
                "total_ratings": 43,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 40,
                    "hold": 3,
                    "sell": 0
                }
            },
            "GOOGL": {
                "positive_percentage": 94.0,
                "total_ratings": 50,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 47,
                    "hold": 3,
                    "sell": 0
                }
            },
            "META": {
                "positive_percentage": 88.0,
                "total_ratings": 42,
                "ratings_type": "analyst",
                "recommendations": {
                    "buy": 37,
                    "hold": 5,
                    "sell": 0
                }
            }
        }[ticker]
        
        # Skip actual metrics generation and return mock data with recommendations
        mock_metrics_df = pd.DataFrame([
            {
                "ticker": "AAPL",
                "price": 170.0,
                "target_price": 190.0,
                "upside": 11.8,
                "buy_percentage": 71.4,
                "analyst_count": 35,
                "pe_trailing": 25.0,
                "pe_forward": 22.0,
                "peg_ratio": 1.5,
                "beta": 1.2,
                "recommendation": "HOLD",
                "company_name": "APPLE INC."
            },
            {
                "ticker": "MSFT",
                "price": 330.0,
                "target_price": 360.0,
                "upside": 9.1,
                "buy_percentage": 85.7,
                "analyst_count": 35,
                "pe_trailing": 30.0,
                "pe_forward": 26.0,
                "peg_ratio": 1.8,
                "beta": 1.0,
                "recommendation": "HOLD",
                "company_name": "MICROSOFT CORP"
            },
            {
                "ticker": "AMZN",
                "price": 140.0,
                "target_price": 170.0,
                "upside": 21.4,
                "buy_percentage": 93.0,
                "analyst_count": 43,
                "pe_trailing": 60.0,
                "pe_forward": 40.0,
                "peg_ratio": 2.5,
                "beta": 1.3,
                "recommendation": "BUY",
                "company_name": "AMAZON.COM INC"
            },
            {
                "ticker": "GOOGL",
                "price": 135.0,
                "target_price": 165.0,
                "upside": 22.2,
                "buy_percentage": 94.0,
                "analyst_count": 50,
                "pe_trailing": 27.0,
                "pe_forward": 24.0,
                "peg_ratio": 1.3,
                "beta": 1.1,
                "recommendation": "BUY",
                "company_name": "ALPHABET INC"
            },
            {
                "ticker": "META",
                "price": 310.0,
                "target_price": 380.0,
                "upside": 22.6,
                "buy_percentage": 88.0,
                "analyst_count": 42,
                "pe_trailing": 29.0,
                "pe_forward": 22.0,
                "peg_ratio": 1.4,
                "beta": 1.4,
                "recommendation": "BUY",
                "company_name": "META PLATFORMS"
            }
        ])
        mock_generate_metrics.return_value = mock_metrics_df
        
        # Create display instance and run buy recommendations
        display = MarketDisplay()
        result = display.get_buy_recommendations()
        
        # Verify the workflow executed correctly
        assert mock_read_csv.called
        assert mock_generate_metrics.called
        
        # Verify result contains only BUY recommendations
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # AMZN, GOOGL, META should be BUY
        assert set(result["recommendation"]) == {"BUY"}
        assert set(result["ticker"]) == {"AMZN", "GOOGL", "META"}
        
        # Verify result is saved to output file
        mock_metrics_df[mock_metrics_df["recommendation"] == "BUY"].to_csv.assert_called_once()
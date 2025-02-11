import pandas as pd
from .lastearnings import get_last_earnings_date
from datetime import datetime, timedelta
import yfinance as yf
import logging

def is_positive_rating(grade):
    """Determine if a rating grade is positive."""
    if not isinstance(grade, str):
        return False
    positive_terms = ['buy', 'outperform', 'overweight', 'positive', 'strong buy', 'long-term buy']
    return any(term in grade.lower() for term in positive_terms)

def get_positive_rating_percentage(ticker):
    """
    Calculates the percentage of positive analyst ratings for a given stock ticker
    based on reports published on or after the last earnings date.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        float or None: The percentage of positive ratings, or None if data is unavailable or an error occurs.
    """
    try:
        company = yf.Ticker(ticker)
        upgrades = company.upgrades_downgrades

        if upgrades is None or upgrades.empty:
            logging.debug(f"{ticker} - No upgrades/downgrades data available")
            return None

        # Get last earnings date
        last_earnings_date_str = get_last_earnings_date(ticker)
        if last_earnings_date_str == "--" or not last_earnings_date_str:
            logging.debug(f"{ticker} - No earnings date available")
            return None
        
        # Convert to timezone-naive datetime for consistent comparison
        try:
            last_earnings_date = pd.to_datetime(last_earnings_date_str).tz_localize(None)
            if last_earnings_date >= pd.Timestamp.now().tz_localize(None):
                # If earnings date is today or in the future, use previous earnings date
                prev_earnings_date = get_second_last_earnings_date(ticker)
                if not prev_earnings_date:
                    logging.debug(f"{ticker} - No previous earnings date available")
                    return None
                last_earnings_date = pd.to_datetime(prev_earnings_date).tz_localize(None)
        except Exception as e:
            logging.error(f"{ticker} - Error processing earnings date: {e}")
            return None

        # Ensure upgrades index is datetime
        upgrades.index = pd.to_datetime(upgrades.index)
        # Make timezone-naive for consistent comparison
        upgrades.index = upgrades.index.tz_localize(None)

        logging.debug(f"{ticker} - Last earnings date: {last_earnings_date}")
        logging.debug(f"{ticker} - Found {len(upgrades)} total ratings")

        # Filter to get only ratings on or after the last earnings date
        recent_upgrades = upgrades[upgrades.index >= last_earnings_date]
        
        if recent_upgrades.empty:
            logging.debug(f"{ticker} - No ratings found after {last_earnings_date}")
            return None

        logging.debug(f"{ticker} - Found {len(recent_upgrades)} ratings after earnings date")

        # Count positive ratings
        if 'ToGrade' not in recent_upgrades.columns:
            logging.debug(f"{ticker} - No 'ToGrade' column found in ratings data")
            return None

        positive_count = sum(recent_upgrades['ToGrade'].apply(is_positive_rating))
        total_count = len(recent_upgrades)

        if total_count == 0:
            return None

        percentage = (positive_count / total_count) * 100
        logging.debug(f"{ticker} - {positive_count} positive out of {total_count} total = {percentage}%")
        
        return percentage

    except Exception as e:
        logging.error(f"Error fetching analyst ratings for {ticker}: {e}")
        return None

def get_total_ratings(ticker):
    """
    Calculates the total number of analyst ratings for a given stock ticker
    based on reports published on or after the last earnings date.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        int or None: The total number of ratings, or None if data is unavailable or an error occurs.
    """
    try:
        company = yf.Ticker(ticker)
        upgrades = company.upgrades_downgrades

        if upgrades is None or upgrades.empty:
            logging.debug(f"{ticker} - No upgrades/downgrades data available")
            return None

        # Get last earnings date
        last_earnings_date_str = get_last_earnings_date(ticker)
        if last_earnings_date_str == "--" or not last_earnings_date_str:
            logging.debug(f"{ticker} - No earnings date available")
            return None

        # Convert to timezone-naive datetime for consistent comparison
        try:
            last_earnings_date = pd.to_datetime(last_earnings_date_str).tz_localize(None)
            if last_earnings_date >= pd.Timestamp.now().tz_localize(None):
                # If earnings date is today or in the future, use previous earnings date
                prev_earnings_date = get_second_last_earnings_date(ticker)
                if not prev_earnings_date:
                    logging.debug(f"{ticker} - No previous earnings date available")
                    return None
                last_earnings_date = pd.to_datetime(prev_earnings_date).tz_localize(None)
        except Exception as e:
            logging.error(f"{ticker} - Error processing earnings date: {e}")
            return None

        # Ensure upgrades index is datetime
        upgrades.index = pd.to_datetime(upgrades.index)
        # Make timezone-naive for consistent comparison
        upgrades.index = upgrades.index.tz_localize(None)

        # Filter to get only ratings on or after the last earnings date
        recent_upgrades = upgrades[upgrades.index >= last_earnings_date]
        total_count = len(recent_upgrades)

        if total_count == 0:
            logging.debug(f"{ticker} - No ratings found after {last_earnings_date}")
            return None

        logging.debug(f"{ticker} - Found {total_count} ratings after {last_earnings_date}")
        return total_count

    except Exception as e:
        logging.error(f"Error fetching total analyst ratings for {ticker}: {e}")
        return None

if __name__ == '__main__':
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)
    
    ticker_symbol = "AAPL"  # Example ticker
    positive_percentage = get_positive_rating_percentage(ticker_symbol)
    total_ratings = get_total_ratings(ticker_symbol)

    if positive_percentage is not None:
        print(f"Positive analyst rating percentage for {ticker_symbol}: {positive_percentage:.2f}%")
    else:
        print(f"Could not retrieve positive analyst rating percentage for {ticker_symbol}")

    if total_ratings is not None:
        print(f"Total analyst ratings for {ticker_symbol}: {total_ratings}")
    else:
        print(f"Could not retrieve total analyst ratings for {ticker_symbol}")
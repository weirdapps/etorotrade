"""
Yahoo Finance API provider implementation.

This module implements the FinanceDataProvider interface for Yahoo Finance data.
It wraps the existing functionality to provide a consistent interface.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from .base import FinanceDataProvider
# Use direct import to avoid circular dependency
import yfinance as yf
from ...core.errors import YFinanceError

logger = logging.getLogger(__name__)


class YahooFinanceProvider(FinanceDataProvider):
    """
    Yahoo Finance data provider implementation.
    
    This provider uses the core YFinanceClientCore to fetch data
    and adapts it to the provider interface.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance provider."""
        # Use direct yfinance access to avoid circular dependency
        self._ticker_cache = {}
    
    def _get_or_create_ticker(self, ticker: str) -> yf.Ticker:
        """
        Get existing Ticker object or create a new one if needed.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            yf.Ticker: Ticker object for the given symbol
        """
        if ticker not in self._ticker_cache:
            logger.debug(f"Creating new Ticker object for {ticker}")
            self._ticker_cache[ticker] = yf.Ticker(ticker)
        return self._ticker_cache[ticker]
        
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock information
        """
        try:
            # Get ticker object directly (no circular dependency)
            stock = self._get_or_create_ticker(ticker)
            info = stock.info or {}
            
            # Build response dictionary
            return {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap'),
                'beta': info.get('beta'),
                'pe_trailing': info.get('trailingPE'),
                'pe_forward': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'price': info.get('currentPrice'),
                'analyst_count': info.get('numberOfAnalystOpinions'),
                'peg_ratio': info.get('pegRatio'),
                'short_float_pct': info.get('shortPercentOfFloat', 0) * 100 if info.get('shortPercentOfFloat') is not None else None,
                'last_earnings': None,  # Will be filled by get_earnings_data
                'previous_earnings': None,  # Will be filled by get_earnings_data
                'stock_object': stock  # Store the stock object for other methods to use
            }
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get ticker info: {str(e)}")
    
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing price information
        """
        try:
            # Get ticker object
            stock = self._get_or_create_ticker(ticker)
            info = stock.info or {}
            
            # Get price data
            current_price = info.get('currentPrice')
            target_price = info.get('targetMeanPrice')
            
            # Calculate upside potential
            upside_potential = None
            if current_price is not None and target_price is not None and current_price > 0:
                upside_potential = ((target_price / current_price) - 1) * 100
                
            # Get price change info
            try:
                hist = stock.history(period="5d")
                price_change = None
                price_change_percentage = None
                
                if not hist.empty and len(hist) > 1:
                    current = hist['Close'].iloc[-1]
                    prev_day = hist['Close'].iloc[-2]
                    price_change = current - prev_day
                    price_change_percentage = ((current - prev_day) / prev_day) * 100
            except Exception as e:
                logger.warning(f"Error getting historical data for {ticker}: {str(e)}")
                price_change = None
                price_change_percentage = None
                
            return {
                'current_price': current_price,
                'target_price': target_price,
                'upside_potential': upside_potential,
                'price_change': price_change,
                'price_change_percentage': price_change_percentage,
            }
        except Exception as e:
            logger.error(f"Error getting price data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get price data: {str(e)}")
    
    def get_historical_data(self, 
                          ticker: str, 
                          period: Optional[str] = "1y", 
                          interval: Optional[str] = "1d") -> pd.DataFrame:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame containing historical data
        """
        try:
            # Get ticker object
            stock = self._get_or_create_ticker(ticker)
            
            # Fetch historical data directly
            data = stock.history(period=period, interval=interval)
            
            # Add moving averages if using daily data and period is long enough
            if interval in ["1d", "1wk", "1mo"] and not data.empty and len(data) > 50:
                # Calculate moving averages appropriate for the data frequency
                data['ma50'] = data['Close'].rolling(window=50).mean()
                if len(data) > 200:
                    data['ma200'] = data['Close'].rolling(window=200).mean()
            
            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get historical data: {str(e)}")
    
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing analyst ratings
        """
        try:
            # Get ticker object
            stock = self._get_or_create_ticker(ticker)
            info = stock.info or {}
            
            # Get analyst ratings
            recommendation_mean = info.get('recommendationMean')
            total_ratings = info.get('numberOfAnalystOpinions')
            
            # Calculate positive percentage
            positive_percentage = None
            if recommendation_mean is not None:
                # Convert the recommendation mean (1-5 scale) to a percentage
                # 1 = Strong Buy, 2 = Buy, 3 = Hold, 4 = Sell, 5 = Strong Sell
                # Lower is better, so we invert and scale to 0-100%
                positive_percentage = ((5 - min(max(recommendation_mean, 1), 5)) / 4) * 100
            
            # Get recommendation details if available
            recommendations = {}
            try:
                rec_df = stock.recommendations
                if rec_df is not None and not rec_df.empty:
                    counts = rec_df['To Grade'].value_counts()
                    recommendations = {
                        'buy': int(counts.get('Buy', 0) + counts.get('Strong Buy', 0)),
                        'hold': int(counts.get('Hold', 0) + counts.get('Neutral', 0)),
                        'sell': int(counts.get('Sell', 0) + counts.get('Strong Sell', 0) + counts.get('Underperform', 0))
                    }
            except Exception as rec_err:
                logger.warning(f"Could not get detailed recommendations for {ticker}: {str(rec_err)}")
                
            return {
                'positive_percentage': positive_percentage,
                'total_ratings': total_ratings,
                'ratings_type': 'analyst',
                'recommendations': recommendations,
            }
        except Exception as e:
            logger.error(f"Error getting analyst ratings for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get analyst ratings: {str(e)}")
    
    def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing earnings data
        """
        try:
            # Get ticker object
            stock = self._get_or_create_ticker(ticker)
            
            # Get earnings dates
            earnings_dates = None
            try:
                earnings_dates = stock.earnings_dates
            except Exception as e:
                logger.warning(f"Error getting earnings dates for {ticker}: {str(e)}")
            
            # Process earnings dates
            last_earnings = None
            previous_earnings = None
            
            if earnings_dates is not None and not earnings_dates.empty:
                # Find past earnings dates
                current_time = pd.Timestamp.now()
                past_earnings = earnings_dates.loc[earnings_dates.index < current_time]
                
                if not past_earnings.empty:
                    # Get sorted dates (most recent first)
                    dates = sorted(past_earnings.index, reverse=True)
                    
                    if len(dates) > 0:
                        last_earnings = dates[0].strftime('%Y-%m-%d')
                    
                    if len(dates) > 1:
                        previous_earnings = dates[1].strftime('%Y-%m-%d')
            
            # Create earnings data dictionary
            earnings_data = {
                'last_earnings': last_earnings,
                'previous_earnings': previous_earnings,
                'earnings_dates': {}
            }
            
            # Add detailed earnings information if available
            if earnings_dates is not None and not earnings_dates.empty:
                earnings_dict = {}
                
                for date, row in earnings_dates.iterrows():
                    date_str = date.strftime('%Y-%m-%d')
                    earnings_dict[date_str] = {
                        'eps_estimate': row.get('EPS Estimate'),
                        'reported_eps': row.get('Reported EPS'),
                        'surprise': row.get('Surprise(%)'),
                    }
                
                earnings_data['earnings_dates'] = earnings_dict
                
            # Get upcoming earnings date if available
            try:
                calendar = stock.calendar
                if calendar is not None and not calendar.empty and 'Earnings Date' in calendar:
                    next_date = calendar['Earnings Date']
                    if isinstance(next_date, pd.Timestamp):
                        earnings_data['upcoming_earnings'] = next_date.strftime('%Y-%m-%d')
            except Exception as cal_err:
                logger.warning(f"Error getting earnings calendar for {ticker}: {str(cal_err)}")
                
            # Add historical earnings data
            try:
                hist_earnings = stock.earnings
                if hist_earnings is not None and not hist_earnings.empty:
                    earnings_data['earnings_history'] = hist_earnings.to_dict()
            except Exception as hist_err:
                logger.warning(f"Error getting earnings history for {ticker}: {str(hist_err)}")
                
            return earnings_data
        except Exception as e:
            logger.error(f"Error getting earnings data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get earnings data: {str(e)}")
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching tickers with metadata
        """
        try:
            # Create ticker object for the query
            import yfinance as yf
            
            # Try using the yfinance search function with recommendations
            try:
                search_results = yf.Ticker(query).recommendations
                
                if search_results is not None and not search_results.empty:
                    # Format results
                    formatted_results = []
                    for _, row in search_results.iterrows():
                        if len(formatted_results) >= limit:
                            break
                            
                        ticker_symbol = row.get('toSymbol')
                        if ticker_symbol:
                            formatted_results.append({
                                'symbol': ticker_symbol,
                                'name': row.get('toName', ticker_symbol),
                                'exchange': row.get('exchange', 'UNKNOWN'),
                                'type': 'EQUITY',  # Default to EQUITY
                                'score': 1.0  # Default score
                            })
                    
                    if formatted_results:
                        return formatted_results[:limit]
            except Exception as rec_e:
                logger.warning(f"Could not search using recommendations: {str(rec_e)}")
            
            # Fallback: just return the query as a ticker symbol
            return [{'symbol': query, 'name': query, 'exchange': 'UNKNOWN', 'type': 'EQUITY', 'score': 1.0}]
        except Exception as e:
            logger.error(f"Error searching tickers for '{query}': {str(e)}")
            raise YFinanceError(f"Failed to search tickers: {str(e)}")
    
    def batch_get_ticker_info(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple symbols in a batch.
        
        This provides a more efficient implementation than the default
        by using parallel processing when possible.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to their information
        """
        results = {}
        
        # Process in batches of 10 for better performance
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all ticker requests
            future_to_ticker = {
                executor.submit(self.get_ticker_info, ticker): ticker for ticker in tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    logger.warning(f"Error getting data for {ticker}: {str(e)}")
                    results[ticker] = None
                    
        return results
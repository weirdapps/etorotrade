"""
Market analysis module for financial data.

This module provides functions for analyzing market data,
identifying trading opportunities, and applying trading criteria.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field

from ..api import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
from ..core.config import TRADING_CRITERIA
from ..core.errors import YFinanceError

logger = logging.getLogger(__name__)

@dataclass
class MarketMetrics:
    """
    Container for market-wide metrics.
    
    Attributes:
        avg_upside: Average upside percentage
        median_upside: Median upside percentage
        avg_buy_percentage: Average analyst buy rating percentage
        median_buy_percentage: Median analyst buy rating percentage
        avg_pe_ratio: Average P/E ratio
        median_pe_ratio: Median P/E ratio
        avg_forward_pe: Average forward P/E ratio
        median_forward_pe: Median forward P/E ratio
        avg_peg_ratio: Average PEG ratio
        median_peg_ratio: Median PEG ratio
        avg_beta: Average beta
        median_beta: Median beta
        buy_count: Number of buy-rated stocks
        sell_count: Number of sell-rated stocks
        hold_count: Number of hold-rated stocks
        total_count: Total number of stocks analyzed
        buy_percentage: Percentage of buy-rated stocks
        sell_percentage: Percentage of sell-rated stocks
        hold_percentage: Percentage of hold-rated stocks
        net_breadth: Net market breadth (buy_percentage - sell_percentage)
    """
    
    # Average metrics
    avg_upside: Optional[float] = None
    median_upside: Optional[float] = None
    avg_buy_percentage: Optional[float] = None
    median_buy_percentage: Optional[float] = None
    avg_pe_ratio: Optional[float] = None
    median_pe_ratio: Optional[float] = None
    avg_forward_pe: Optional[float] = None
    median_forward_pe: Optional[float] = None
    avg_peg_ratio: Optional[float] = None
    median_peg_ratio: Optional[float] = None
    avg_beta: Optional[float] = None
    median_beta: Optional[float] = None
    
    # Count metrics
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    total_count: int = 0
    
    # Percentage metrics
    buy_percentage: Optional[float] = None
    sell_percentage: Optional[float] = None
    hold_percentage: Optional[float] = None
    net_breadth: Optional[float] = None
    
    # Sector metrics
    sector_counts: Dict[str, int] = field(default_factory=dict)
    sector_breadth: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SectorAnalysis:
    """
    Container for sector-specific analysis.
    
    Attributes:
        sector: Sector name
        stock_count: Number of stocks in the sector
        buy_count: Number of buy-rated stocks in the sector
        sell_count: Number of sell-rated stocks in the sector
        hold_count: Number of hold-rated stocks in the sector
        buy_percentage: Percentage of buy-rated stocks in the sector
        sell_percentage: Percentage of sell-rated stocks in the sector
        hold_percentage: Percentage of hold-rated stocks in the sector
        net_breadth: Net sector breadth (buy_percentage - sell_percentage)
        avg_upside: Average upside percentage for the sector
        avg_buy_rating: Average analyst buy rating percentage for the sector
        avg_pe_ratio: Average P/E ratio for the sector
        avg_peg_ratio: Average PEG ratio for the sector
    """
    
    sector: str
    stock_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    buy_percentage: Optional[float] = None
    sell_percentage: Optional[float] = None
    hold_percentage: Optional[float] = None
    net_breadth: Optional[float] = None
    avg_upside: Optional[float] = None
    avg_buy_rating: Optional[float] = None
    avg_pe_ratio: Optional[float] = None
    avg_peg_ratio: Optional[float] = None


class MarketAnalyzer:
    """
    Analyzer for market-wide data and sector-specific analysis.
    
    This analyzer processes market data to identify trends, categorize stocks,
    and calculate market-wide metrics.
    
    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """
    
    def __init__(self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None):
        """
        Initialize the MarketAnalyzer.
        
        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()
        
        # Check if the provider is async
        self.is_async = hasattr(self.provider, 'batch_get_ticker_info') and \
                        callable(self.provider.batch_get_ticker_info) and \
                        hasattr(self.provider.batch_get_ticker_info, '__await__')
    
    def analyze_market(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze market data for a list of tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            DataFrame with market analysis results
            
        Raises:
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use analyze_market_async instead.")
        
        try:
            # Fetch data in batch
            ticker_info_batch = self.provider.batch_get_ticker_info(tickers)
            
            # Convert to DataFrame
            market_data = []
            for ticker in tickers:
                if ticker in ticker_info_batch and ticker_info_batch[ticker]:
                    ticker_info = ticker_info_batch[ticker]
                    market_data.append({
                        'ticker': ticker,
                        'name': ticker_info.get('name', ticker),
                        'price': ticker_info.get('price'),
                        'market_cap': ticker_info.get('market_cap'),
                        'market_cap_fmt': ticker_info.get('market_cap_fmt'),
                        'upside': ticker_info.get('upside'),
                        'pe_trailing': ticker_info.get('pe_ratio'),
                        'pe_forward': ticker_info.get('forward_pe'),
                        'peg_ratio': ticker_info.get('peg_ratio'),
                        'beta': ticker_info.get('beta'),
                        'dividend_yield': ticker_info.get('dividend_yield'),
                        'buy_percentage': ticker_info.get('buy_percentage'),
                        'total_ratings': ticker_info.get('total_ratings'),
                        'analyst_count': ticker_info.get('analyst_count'),
                        'short_float_pct': ticker_info.get('short_percent'),
                        'sector': ticker_info.get('sector')
                    })
            
            # Create DataFrame
            market_df = pd.DataFrame(market_data)
            
            # Classify stocks
            market_df = classify_stocks(market_df)
            
            return market_df
        
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            raise YFinanceError(f"Failed to analyze market: {str(e)}")
    
    async def analyze_market_async(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze market data for a list of tickers asynchronously.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            DataFrame with market analysis results
            
        Raises:
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use analyze_market instead.")
        
        try:
            # Fetch data in batch asynchronously
            ticker_info_batch = await self.provider.batch_get_ticker_info(tickers)
            
            # Convert to DataFrame
            market_data = []
            for ticker in tickers:
                if ticker in ticker_info_batch and ticker_info_batch[ticker]:
                    ticker_info = ticker_info_batch[ticker]
                    market_data.append({
                        'ticker': ticker,
                        'name': ticker_info.get('name', ticker),
                        'price': ticker_info.get('price'),
                        'market_cap': ticker_info.get('market_cap'),
                        'market_cap_fmt': ticker_info.get('market_cap_fmt'),
                        'upside': ticker_info.get('upside'),
                        'pe_trailing': ticker_info.get('pe_ratio'),
                        'pe_forward': ticker_info.get('forward_pe'),
                        'peg_ratio': ticker_info.get('peg_ratio'),
                        'beta': ticker_info.get('beta'),
                        'dividend_yield': ticker_info.get('dividend_yield'),
                        'buy_percentage': ticker_info.get('buy_percentage'),
                        'total_ratings': ticker_info.get('total_ratings'),
                        'analyst_count': ticker_info.get('analyst_count'),
                        'short_float_pct': ticker_info.get('short_percent'),
                        'sector': ticker_info.get('sector')
                    })
            
            # Create DataFrame
            market_df = pd.DataFrame(market_data)
            
            # Classify stocks
            market_df = classify_stocks(market_df)
            
            return market_df
        
        except Exception as e:
            logger.error(f"Error analyzing market asynchronously: {str(e)}")
            raise YFinanceError(f"Failed to analyze market asynchronously: {str(e)}")
    
    def calculate_market_metrics(self, market_df: pd.DataFrame) -> MarketMetrics:
        """
        Calculate market-wide metrics from market data.
        
        Args:
            market_df: DataFrame with market data
            
        Returns:
            MarketMetrics object with calculated metrics
        """
        metrics = MarketMetrics()
        
        # Check if the DataFrame is empty
        if market_df.empty:
            return metrics
        
        # Calculate average and median values for key metrics
        numeric_columns = {
            'upside': ['avg_upside', 'median_upside'],
            'buy_percentage': ['avg_buy_percentage', 'median_buy_percentage'],
            'pe_trailing': ['avg_pe_ratio', 'median_pe_ratio'],
            'pe_forward': ['avg_forward_pe', 'median_forward_pe'],
            'peg_ratio': ['avg_peg_ratio', 'median_peg_ratio'],
            'beta': ['avg_beta', 'median_beta']
        }
        
        for col, metric_names in numeric_columns.items():
            if col in market_df.columns:
                valid_values = market_df[col].dropna()
                if len(valid_values) > 0:
                    setattr(metrics, metric_names[0], valid_values.mean())
                    setattr(metrics, metric_names[1], valid_values.median())
        
        # Count stocks by category
        buy_opportunities = filter_buy_opportunities(market_df)
        sell_candidates = filter_sell_candidates(market_df)
        hold_candidates = filter_hold_candidates(market_df)
        
        metrics.buy_count = len(buy_opportunities)
        metrics.sell_count = len(sell_candidates)
        metrics.hold_count = len(hold_candidates)
        metrics.total_count = len(market_df)
        
        # Calculate market breadth
        if metrics.total_count > 0:
            metrics.buy_percentage = (metrics.buy_count / metrics.total_count) * 100
            metrics.sell_percentage = (metrics.sell_count / metrics.total_count) * 100
            metrics.hold_percentage = (metrics.hold_count / metrics.total_count) * 100
            metrics.net_breadth = metrics.buy_percentage - metrics.sell_percentage
        
        # Add sector breakdown if sector column exists
        if 'sector' in market_df.columns:
            sector_counts = market_df['sector'].value_counts()
            metrics.sector_counts = sector_counts.to_dict()
            
            # Sector breadth - buy/sell ratio by sector
            sector_breadth = {}
            for sector in sector_counts.index:
                sector_stocks = market_df[market_df['sector'] == sector]
                sector_buys = len(filter_buy_opportunities(sector_stocks))
                sector_sells = len(filter_sell_candidates(sector_stocks))
                sector_total = len(sector_stocks)
                
                if sector_total > 0:
                    sector_breadth[sector] = {
                        'buy_pct': (sector_buys / sector_total) * 100,
                        'sell_pct': (sector_sells / sector_total) * 100,
                        'net_breadth': ((sector_buys - sector_sells) / sector_total) * 100
                    }
            
            metrics.sector_breadth = sector_breadth
        
        return metrics
    
    def analyze_sectors(self, market_df: pd.DataFrame) -> List[SectorAnalysis]:
        """
        Analyze sectors from market data.
        
        Args:
            market_df: DataFrame with market data
            
        Returns:
            List of SectorAnalysis objects with sector-specific metrics
        """
        sector_analysis = []
        
        # Check if the DataFrame is empty or doesn't have a sector column
        if market_df.empty or 'sector' not in market_df.columns:
            return sector_analysis
        
        # Group data by sector
        sectors = market_df['sector'].dropna().unique()
        
        for sector in sectors:
            sector_stocks = market_df[market_df['sector'] == sector]
            sector_buy = filter_buy_opportunities(sector_stocks)
            sector_sell = filter_sell_candidates(sector_stocks)
            sector_hold = filter_hold_candidates(sector_stocks)
            
            stock_count = len(sector_stocks)
            buy_count = len(sector_buy)
            sell_count = len(sector_sell)
            hold_count = len(sector_hold)
            
            # Calculate percentages
            buy_percentage = (buy_count / stock_count) * 100 if stock_count > 0 else None
            sell_percentage = (sell_count / stock_count) * 100 if stock_count > 0 else None
            hold_percentage = (hold_count / stock_count) * 100 if stock_count > 0 else None
            net_breadth = buy_percentage - sell_percentage if buy_percentage is not None and sell_percentage is not None else None
            
            # Calculate averages
            avg_upside = sector_stocks['upside'].dropna().mean() if 'upside' in sector_stocks.columns else None
            avg_buy_rating = sector_stocks['buy_percentage'].dropna().mean() if 'buy_percentage' in sector_stocks.columns else None
            avg_pe_ratio = sector_stocks['pe_trailing'].dropna().mean() if 'pe_trailing' in sector_stocks.columns else None
            avg_peg_ratio = sector_stocks['peg_ratio'].dropna().mean() if 'peg_ratio' in sector_stocks.columns else None
            
            # Create SectorAnalysis object
            analysis = SectorAnalysis(
                sector=sector,
                stock_count=stock_count,
                buy_count=buy_count,
                sell_count=sell_count,
                hold_count=hold_count,
                buy_percentage=buy_percentage,
                sell_percentage=sell_percentage,
                hold_percentage=hold_percentage,
                net_breadth=net_breadth,
                avg_upside=avg_upside,
                avg_buy_rating=avg_buy_rating,
                avg_pe_ratio=avg_pe_ratio,
                avg_peg_ratio=avg_peg_ratio
            )
            
            sector_analysis.append(analysis)
        
        # Sort sectors by net breadth (most bullish to most bearish)
        sector_analysis.sort(key=lambda x: x.net_breadth if x.net_breadth is not None else -999, reverse=True)
        
        return sector_analysis

def filter_buy_opportunities(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out buy opportunities from market data based on trading criteria.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        DataFrame with buy opportunities
    """
    # Apply BUY criteria
    buy_criteria = TRADING_CRITERIA["BUY"]
    
    # Filter condition with confidence requirements
    confidence_condition = (
        market_df['analyst_count'].notna() & 
        market_df['total_ratings'].notna() & 
        (market_df['analyst_count'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]) &
        (market_df['total_ratings'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"])
    )
    
    # Filter stocks based on upside and analyst consensus
    upside_condition = market_df['upside'] >= buy_criteria["MIN_UPSIDE"]
    analyst_condition = market_df['buy_percentage'] >= buy_criteria["MIN_BUY_PERCENTAGE"]
    
    # Additional criteria for PE ratio, PEG ratio, beta, and short interest
    beta_low_condition = market_df['beta'] > buy_criteria["MIN_BETA"]
    beta_high_condition = market_df['beta'] <= buy_criteria["MAX_BETA"]
    
    # PE condition - complex criteria with nullability handling
    # If both PE values are positive, require PEF < PET
    # If PET is <= 0 (negative), allow any PEF value
    pe_condition = (
        # PE Forward must be positive and not too high
        (market_df['pe_forward'] > buy_criteria["MIN_FORWARD_PE"]) &
        (market_df['pe_forward'] <= buy_criteria["MAX_FORWARD_PE"]) &
        (
            # PE Forward must be less than PE Trailing (improving)
            ((market_df['pe_forward'] < market_df['pe_trailing']) & 
             (market_df['pe_trailing'] > 0)) |
            # Or PE Trailing must be negative or zero (growth case)
            (market_df['pe_trailing'] <= 0)
        )
    )
    
    # PEG Ratio criteria with nullability handling
    peg_condition = (
        market_df['peg_ratio'].isna() |  # Ignore missing PEG values
        (market_df['peg_ratio'] < buy_criteria["MAX_PEG"])
    )
    
    # Short interest criteria with nullability handling
    short_condition = (
        market_df['short_float_pct'].isna() |  # Ignore missing short interest
        market_df['short_float_pct'].isnull() |
        (market_df['short_float_pct'] <= buy_criteria["MAX_SHORT_INTEREST"])
    )
    
    # Combine all criteria
    buy_filter = (
        confidence_condition &
        upside_condition &
        analyst_condition &
        beta_low_condition &
        beta_high_condition &
        pe_condition &
        peg_condition &
        short_condition
    )
    
    # Filter the dataframe
    buy_opportunities = market_df[buy_filter].copy()
    
    return buy_opportunities

def filter_sell_candidates(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out sell candidates from portfolio data based on trading criteria.
    
    Args:
        portfolio_df: DataFrame with portfolio data
        
    Returns:
        DataFrame with sell candidates
    """
    # Apply SELL criteria
    sell_criteria = TRADING_CRITERIA["SELL"]
    
    # Filter condition with confidence requirements
    confidence_condition = (
        portfolio_df['analyst_count'].notna() &
        portfolio_df['total_ratings'].notna() &
        (portfolio_df['analyst_count'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]) &
        (portfolio_df['total_ratings'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"])
    )
    
    # Initialize filters list for each SELL criterion
    filters = []
    
    # Add individual sell criteria with explicit reasons
    
    # Upside too low
    if 'upside' in portfolio_df.columns and 'MAX_UPSIDE' in sell_criteria:
        upside_condition = (
            portfolio_df['upside'].notna() &
            (portfolio_df['upside'] < sell_criteria["MAX_UPSIDE"])
        )
        filters.append(upside_condition)
    
    # Analyst buy percentage too low
    if 'buy_percentage' in portfolio_df.columns and 'MAX_BUY_PERCENTAGE' in sell_criteria:
        analyst_condition = (
            portfolio_df['buy_percentage'].notna() &
            (portfolio_df['buy_percentage'] < sell_criteria["MAX_BUY_PERCENTAGE"])
        )
        filters.append(analyst_condition)
    
    # PE Forward higher than PE Trailing (worsening outlook)
    if 'pe_forward' in portfolio_df.columns and 'pe_trailing' in portfolio_df.columns:
        pe_condition = (
            portfolio_df['pe_forward'].notna() &
            portfolio_df['pe_trailing'].notna() &
            (portfolio_df['pe_forward'] > 0) &
            (portfolio_df['pe_trailing'] > 0) &
            (portfolio_df['pe_forward'] > portfolio_df['pe_trailing'])
        )
        filters.append(pe_condition)
    
    # Forward PE too high
    if 'pe_forward' in portfolio_df.columns and 'MIN_PE_FORWARD' in sell_criteria:
        pe_high_condition = (
            portfolio_df['pe_forward'].notna() &
            (portfolio_df['pe_forward'] > sell_criteria["MIN_PE_FORWARD"])
        )
        filters.append(pe_high_condition)
    
    # PEG ratio too high
    if 'peg_ratio' in portfolio_df.columns and 'MAX_PEG_RATIO' in sell_criteria:
        peg_condition = (
            portfolio_df['peg_ratio'].notna() &
            (portfolio_df['peg_ratio'] > sell_criteria["MAX_PEG_RATIO"])
        )
        filters.append(peg_condition)
    
    # Short interest too high
    if 'short_float_pct' in portfolio_df.columns and 'MIN_SHORT_INTEREST' in sell_criteria:
        short_condition = (
            portfolio_df['short_float_pct'].notna() &
            (portfolio_df['short_float_pct'] > sell_criteria["MIN_SHORT_INTEREST"])
        )
        filters.append(short_condition)
    
    # Beta too high
    if 'beta' in portfolio_df.columns and 'MAX_BETA' in sell_criteria:
        beta_condition = (
            portfolio_df['beta'].notna() &
            (portfolio_df['beta'] > sell_criteria["MAX_BETA"])
        )
        filters.append(beta_condition)
    
    # Expected return too low
    if 'EXRET' in portfolio_df.columns and 'MAX_EXRET' in sell_criteria:
        exret_condition = (
            portfolio_df['EXRET'].notna() &
            (portfolio_df['EXRET'] < sell_criteria["MAX_EXRET"])
        )
        filters.append(exret_condition)
    elif 'upside' in portfolio_df.columns and 'buy_percentage' in portfolio_df.columns and 'MAX_EXRET' in sell_criteria:
        # Calculate EXRET on the fly if not present
        portfolio_df['EXRET'] = portfolio_df['upside'] * portfolio_df['buy_percentage'] / 100
        exret_condition = (
            portfolio_df['EXRET'].notna() &
            (portfolio_df['EXRET'] < sell_criteria["MAX_EXRET"])
        )
        filters.append(exret_condition)
    
    # If no filters were created, return empty dataframe
    if not filters:
        return portfolio_df.head(0)
    
    # Combine filters with OR (any one criterion can trigger a sell)
    sell_criteria_filter = pd.Series(False, index=portfolio_df.index)
    for condition in filters:
        sell_criteria_filter = sell_criteria_filter | condition
    
    # Final filter includes confidence threshold and at least one sell criterion
    sell_filter = confidence_condition & sell_criteria_filter
    
    # Filter the dataframe
    sell_candidates = portfolio_df[sell_filter].copy()
    
    return sell_candidates

def filter_hold_candidates(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out hold candidates from market data.
    
    These are stocks that meet confidence requirements but
    don't qualify as either buy or sell opportunities.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        DataFrame with hold candidates
    """
    # Get buy opportunities
    buy_opportunities = filter_buy_opportunities(market_df)
    
    # Get sell candidates
    sell_candidates = filter_sell_candidates(market_df)
    
    # Filter for confidence condition
    confidence_condition = (
        market_df['analyst_count'].notna() &
        market_df['total_ratings'].notna() &
        (market_df['analyst_count'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]) &
        (market_df['total_ratings'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"])
    )
    
    # Get stocks with sufficient confidence
    confident_stocks = market_df[confidence_condition].copy()
    
    # Get set of buy and sell tickers
    buy_tickers = set(buy_opportunities['ticker'].str.upper())
    sell_tickers = set(sell_candidates['ticker'].str.upper())
    
    # Get hold candidates - those with sufficient confidence but not buy or sell
    hold_filter = ~confident_stocks['ticker'].str.upper().isin(buy_tickers | sell_tickers)
    
    hold_candidates = confident_stocks[hold_filter].copy()
    
    return hold_candidates

def filter_risk_first_buy_opportunities(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for buy opportunities with a risk-first approach.
    
    This function prioritizes filtering out high-risk stocks first
    before applying standard buy criteria. This helps ensure that
    only the highest quality opportunities are included.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        DataFrame with risk-adjusted buy opportunities
    """
    # Apply BUY criteria
    buy_criteria = TRADING_CRITERIA["BUY"]
    
    # Start with confidence check
    confidence_condition = (
        market_df['analyst_count'].notna() & 
        market_df['total_ratings'].notna() & 
        (market_df['analyst_count'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]) &
        (market_df['total_ratings'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"])
    )
    
    # Apply risk filters first - eliminate stocks with excessive risk
    # Beta check (volatility not too high or too low)
    beta_condition = (
        market_df['beta'].notna() &
        (market_df['beta'] > buy_criteria["MIN_BETA"]) &
        (market_df['beta'] <= buy_criteria["MAX_BETA"])
    )
    
    # PEG Ratio check (not too expensive relative to growth)
    peg_condition = (
        market_df['peg_ratio'].isna() |  # Ignore missing PEG values
        (market_df['peg_ratio'] < buy_criteria["MAX_PEG"])
    )
    
    # Short interest check (not too heavily shorted)
    short_condition = (
        market_df['short_float_pct'].isna() |  # Ignore missing short interest
        market_df['short_float_pct'].isnull() |
        (market_df['short_float_pct'] <= buy_criteria["MAX_SHORT_INTEREST"])
    )
    
    # Apply risk filter first
    risk_filter = beta_condition & peg_condition & short_condition
    
    # Then apply return filters
    upside_condition = market_df['upside'] >= buy_criteria["MIN_UPSIDE"]
    analyst_condition = market_df['buy_percentage'] >= buy_criteria["MIN_BUY_PERCENTAGE"]
    
    # PE condition - complex criteria with nullability handling
    pe_condition = (
        # PE Forward must be positive and not too high
        (market_df['pe_forward'] > buy_criteria["MIN_FORWARD_PE"]) &
        (market_df['pe_forward'] <= buy_criteria["MAX_FORWARD_PE"]) &
        (
            # PE Forward must be less than PE Trailing (improving)
            ((market_df['pe_forward'] < market_df['pe_trailing']) & 
             (market_df['pe_trailing'] > 0)) |
            # Or PE Trailing must be negative or zero (growth case)
            (market_df['pe_trailing'] <= 0)
        )
    )
    
    # Combine all criteria
    buy_filter = (
        confidence_condition &
        risk_filter &
        upside_condition &
        analyst_condition &
        pe_condition
    )
    
    # Filter the dataframe
    buy_opportunities = market_df[buy_filter].copy()
    
    return buy_opportunities

def calculate_market_metrics(market_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall market metrics from market data.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        Dictionary with market metrics
    """
    metrics = {}
    
    # Calculate average values for key metrics
    numeric_columns = [
        'upside', 'buy_percentage', 'beta', 
        'pe_trailing', 'pe_forward', 'peg_ratio', 
        'dividend_yield', 'short_float_pct'
    ]
    
    for col in numeric_columns:
        if col in market_df.columns:
            valid_values = market_df[col].dropna()
            if len(valid_values) > 0:
                metrics[f'avg_{col}'] = valid_values.mean()
                metrics[f'median_{col}'] = valid_values.median()
                
    # Count stocks by category
    buy_opportunities = filter_buy_opportunities(market_df)
    sell_candidates = filter_sell_candidates(market_df)
    hold_candidates = filter_hold_candidates(market_df)
    
    metrics['buy_count'] = len(buy_opportunities)
    metrics['sell_count'] = len(sell_candidates)
    metrics['hold_count'] = len(hold_candidates)
    metrics['total_count'] = len(market_df)
    
    # Calculate market breadth
    if 'total_count' in metrics and metrics['total_count'] > 0:
        metrics['buy_percentage'] = (metrics['buy_count'] / metrics['total_count']) * 100
        metrics['sell_percentage'] = (metrics['sell_count'] / metrics['total_count']) * 100
        metrics['hold_percentage'] = (metrics['hold_count'] / metrics['total_count']) * 100
        
        # Net breadth (buy - sell) as percentage of analyzed stocks
        metrics['net_breadth'] = metrics['buy_percentage'] - metrics['sell_percentage']
    
    # Add sector breakdown if sector column exists
    if 'sector' in market_df.columns:
        sector_counts = market_df['sector'].value_counts()
        metrics['sector_counts'] = sector_counts.to_dict()
        
        # Sector breadth - buy/sell ratio by sector
        sector_breadth = {}
        for sector in sector_counts.index:
            sector_stocks = market_df[market_df['sector'] == sector]
            sector_buys = len(filter_buy_opportunities(sector_stocks))
            sector_sells = len(filter_sell_candidates(sector_stocks))
            sector_total = len(sector_stocks)
            
            if sector_total > 0:
                sector_breadth[sector] = {
                    'buy_pct': (sector_buys / sector_total) * 100,
                    'sell_pct': (sector_sells / sector_total) * 100,
                    'net_breadth': ((sector_buys - sector_sells) / sector_total) * 100
                }
        
        metrics['sector_breadth'] = sector_breadth
    
    return metrics

def classify_stocks(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify stocks as BUY, SELL, HOLD, or INCONCLUSIVE.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        DataFrame with classification column added
    """
    result_df = market_df.copy()
    
    # Define confidence condition
    confidence_condition = (
        market_df['analyst_count'].notna() & 
        market_df['total_ratings'].notna() & 
        (market_df['analyst_count'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]) &
        (market_df['total_ratings'] >= TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"])
    )
    
    # Initialize classification column as INCONCLUSIVE
    result_df['classification'] = 'INCONCLUSIVE'
    
    # Filter for BUY stocks
    buy_opportunities = filter_buy_opportunities(market_df)
    buy_tickers = set(buy_opportunities['ticker'].str.upper())
    result_df.loc[result_df['ticker'].str.upper().isin(buy_tickers), 'classification'] = 'BUY'
    
    # Filter for SELL stocks
    sell_candidates = filter_sell_candidates(market_df)
    sell_tickers = set(sell_candidates['ticker'].str.upper())
    result_df.loc[result_df['ticker'].str.upper().isin(sell_tickers), 'classification'] = 'SELL'
    
    # Filter for HOLD stocks (confident but neither BUY nor SELL)
    confident_stocks = market_df[confidence_condition]
    hold_tickers = set(confident_stocks['ticker'].str.upper()) - buy_tickers - sell_tickers
    result_df.loc[result_df['ticker'].str.upper().isin(hold_tickers), 'classification'] = 'HOLD'
    
    return result_df
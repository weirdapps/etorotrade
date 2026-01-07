#!/usr/bin/env python3
"""
Portfolio service module for handling portfolio-specific filtering and operations.
"""

from typing import Dict
import pandas as pd
from yahoofinance.utils.data.ticker_utils import are_equivalent_tickers


class PortfolioService:
    """Service for portfolio-specific operations and filtering."""

    def __init__(self, logger):
        """Initialize portfolio service with logger."""
        self.logger = logger

    def apply_portfolio_filters(
        self, opportunities: Dict[str, pd.DataFrame], portfolio_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Apply portfolio-specific filters to opportunities and include portfolio stocks with SELL/HOLD classifications."""
        try:
            portfolio_tickers = set()

            # Extract portfolio tickers (no normalization needed for equivalence checking)
            # Check for various column name variations
            ticker_col = None
            if "TICKER" in portfolio_df.columns:
                ticker_col = "TICKER"
            elif "Ticker" in portfolio_df.columns:
                ticker_col = "Ticker"
            elif "ticker" in portfolio_df.columns:
                ticker_col = "ticker"
            elif "symbol" in portfolio_df.columns:
                ticker_col = "symbol"
            
            if ticker_col:
                for ticker in portfolio_df[ticker_col]:
                    if pd.notna(ticker) and ticker:
                        portfolio_tickers.add(ticker)
            
            if portfolio_tickers:
                # For sell and hold, include both market opportunities AND portfolio stocks with corresponding classifications
                
                # SELL: Include market sell opportunities that match portfolio + portfolio stocks marked as SELL
                sell_opportunities = opportunities["sell_opportunities"].copy()
                
                # Add portfolio stocks that are classified as SELL (BS column = 'S')
                if "BS" in portfolio_df.columns:
                    portfolio_sells = portfolio_df[portfolio_df["BS"] == "S"].copy()
                    if not portfolio_sells.empty:
                        # Set index to ticker for consistency with market opportunities
                        if ticker_col in portfolio_sells.columns:
                            portfolio_sells = portfolio_sells.set_index(ticker_col)
                        # Combine with market sell opportunities (avoid duplicates)
                        sell_opportunities = pd.concat([sell_opportunities, portfolio_sells], ignore_index=False)
                        sell_opportunities = sell_opportunities[~sell_opportunities.index.duplicated(keep='first')]
                        self.logger.info(f"Added {len(portfolio_sells)} portfolio SELL stocks to sell opportunities")
                
                # Filter market sell opportunities to only include portfolio holdings
                market_sell_mask = pd.Series(False, index=opportunities["sell_opportunities"].index)
                for market_ticker in opportunities["sell_opportunities"].index:
                    if pd.notna(market_ticker):
                        is_in_portfolio = any(
                            are_equivalent_tickers(market_ticker, portfolio_ticker)
                            for portfolio_ticker in portfolio_tickers
                        )
                        if is_in_portfolio:
                            market_sell_mask.loc[market_ticker] = True
                
                # Combine filtered market opportunities with portfolio SELL stocks
                filtered_market_sells = opportunities["sell_opportunities"][market_sell_mask]
                if "BS" in portfolio_df.columns:
                    portfolio_sells = portfolio_df[portfolio_df["BS"] == "S"].copy()
                    if not portfolio_sells.empty and ticker_col:
                        portfolio_sells = portfolio_sells.set_index(ticker_col)
                        sell_opportunities = pd.concat([filtered_market_sells, portfolio_sells], ignore_index=False)
                        sell_opportunities = sell_opportunities[~sell_opportunities.index.duplicated(keep='first')]
                    else:
                        sell_opportunities = filtered_market_sells
                else:
                    sell_opportunities = filtered_market_sells
                    
                opportunities["sell_opportunities"] = sell_opportunities

                # HOLD: Include market hold opportunities that match portfolio + portfolio stocks marked as HOLD
                hold_opportunities = opportunities["hold_opportunities"].copy()
                
                # Filter market hold opportunities to only include portfolio holdings
                market_hold_mask = pd.Series(False, index=opportunities["hold_opportunities"].index)
                for market_ticker in opportunities["hold_opportunities"].index:
                    if pd.notna(market_ticker):
                        is_in_portfolio = any(
                            are_equivalent_tickers(market_ticker, portfolio_ticker)
                            for portfolio_ticker in portfolio_tickers
                        )
                        if is_in_portfolio:
                            market_hold_mask.loc[market_ticker] = True
                
                # Combine filtered market opportunities with portfolio HOLD stocks
                filtered_market_holds = opportunities["hold_opportunities"][market_hold_mask]
                if "BS" in portfolio_df.columns:
                    portfolio_holds = portfolio_df[portfolio_df["BS"] == "H"].copy()
                    if not portfolio_holds.empty and ticker_col:
                        portfolio_holds = portfolio_holds.set_index(ticker_col)
                        hold_opportunities = pd.concat([filtered_market_holds, portfolio_holds], ignore_index=False)
                        hold_opportunities = hold_opportunities[~hold_opportunities.index.duplicated(keep='first')]
                        self.logger.info(f"Added {len(portfolio_holds)} portfolio HOLD stocks to hold opportunities")
                    else:
                        hold_opportunities = filtered_market_holds
                else:
                    hold_opportunities = filtered_market_holds
                    
                opportunities["hold_opportunities"] = hold_opportunities

                # For buy, exclude tickers equivalent to portfolio holdings (unchanged)
                buy_mask = pd.Series(True, index=opportunities["buy_opportunities"].index)
                
                for market_ticker in opportunities["buy_opportunities"].index:
                    if pd.notna(market_ticker):
                        is_in_portfolio = any(
                            are_equivalent_tickers(market_ticker, portfolio_ticker)
                            for portfolio_ticker in portfolio_tickers
                        )
                        if is_in_portfolio:
                            buy_mask.loc[market_ticker] = False
                
                opportunities["buy_opportunities"] = opportunities["buy_opportunities"][buy_mask]

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Error applying portfolio filters: {str(e)}")

        return opportunities
    
    # Backward compatibility alias - singular version
    def apply_portfolio_filter(
        self, opportunities: Dict[str, pd.DataFrame], portfolio_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Backward compatibility alias for apply_portfolio_filters."""
        return self.apply_portfolio_filters(opportunities, portfolio_df)
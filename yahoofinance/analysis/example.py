"""
Example of using dependency injection with stock analyzer.

This module demonstrates how to use the dependency injection system
for creating and using stock analyzers.
"""

import asyncio
from typing import Any, Dict, List

from ..core.errors import YFinanceError
from ..core.logging_config import get_logger
from .analyzer_factory import with_analyzer, with_portfolio_analyzer
from .portfolio import PortfolioSummary
from .stock import AnalysisResults


# Set up logging
logger = get_logger(__name__)


# Example 1: Using the decorator to inject a stock analyzer
@with_analyzer(async_mode=False)
def analyze_stock(ticker: str, analyzer=None) -> AnalysisResults:
    """
    Analyze a stock using an injected analyzer.

    Args:
        ticker: Stock ticker symbol
        analyzer: Injected StockAnalyzer instance

    Returns:
        Analysis results
    """
    logger.info(f"Analyzing {ticker} using injected analyzer")
    return analyzer.analyze(ticker)


# Example 2: Using the decorator to inject an async stock analyzer
@with_analyzer(async_mode=True)
async def analyze_stock_async(ticker: str, analyzer=None) -> AnalysisResults:
    """
    Analyze a stock asynchronously using an injected analyzer.

    Args:
        ticker: Stock ticker symbol
        analyzer: Injected StockAnalyzer instance

    Returns:
        Analysis results
    """
    logger.info(f"Analyzing {ticker} asynchronously using injected analyzer")
    return await analyzer.analyze_async(ticker)


# Example 3: Using the decorator to inject an analyzer for batch processing
@with_analyzer(async_mode=True, enhanced=True)
async def analyze_batch_async(tickers: List[str], analyzer=None) -> Dict[str, AnalysisResults]:
    """
    Analyze multiple stocks asynchronously using an injected analyzer.

    Args:
        tickers: List of stock ticker symbols
        analyzer: Injected StockAnalyzer instance

    Returns:
        Dictionary of analysis results keyed by ticker
    """
    logger.info(f"Analyzing {len(tickers)} stocks asynchronously using injected analyzer")
    return await analyzer.analyze_batch_async(tickers)


# Example 4: Using the decorator to inject a portfolio analyzer
@with_portfolio_analyzer(async_mode=True)
def analyze_portfolio_file(file_path: str, portfolio_analyzer=None):
    """
    Analyze a portfolio from a CSV file using an injected portfolio analyzer.

    Args:
        file_path: Path to the portfolio CSV file
        portfolio_analyzer: Injected PortfolioAnalyzer instance

    Returns:
        Portfolio summary
    """
    logger.info(f"Analyzing portfolio from {file_path} using injected portfolio analyzer")
    portfolio_analyzer.load_portfolio_from_csv(file_path)
    return portfolio_analyzer.analyze_portfolio()


# Example function to demonstrate usage
async def run_example():
    """Run the examples to demonstrate dependency injection."""
    try:
        # Example 1: Sync analysis
        result = analyze_stock("AAPL")
        logger.info(
            f"AAPL analysis: {result.category} with expected return {result.expected_return}%"
        )

        # Example 2: Async analysis
        result = await analyze_stock_async("MSFT")
        logger.info(
            f"MSFT analysis: {result.category} with expected return {result.expected_return}%"
        )

        # Example 3: Batch async analysis
        batch_results = await analyze_batch_async(["AAPL", "MSFT", "GOOG", "AMZN"])
        for ticker, result in batch_results.items():
            logger.info(
                f"{ticker} batch analysis: {result.category} with expected return {result.expected_return}%"
            )

        # Example 4: Portfolio analysis
        portfolio_file = "../input/test_portfolio.csv"
        try:
            portfolio_summary = analyze_portfolio_file(portfolio_file)
            logger.info(
                f"Portfolio analysis: {portfolio_summary.total_gain_loss_pct:.2f}% gain/loss"
            )
            logger.info(f"Buy recommendations: {portfolio_summary.buy_count}")
            logger.info(f"Sell recommendations: {portfolio_summary.sell_count}")
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {str(e)}")

        return "Examples completed successfully"
    except YFinanceError as e:
        logger.error(f"Error in example: {str(e)}")
        return f"Error: {str(e)}"


# Entry point for running the example directly
if __name__ == "__main__":
    logger.info("Running dependency injection examples")
    result = asyncio.run(run_example())
    logger.info(result)

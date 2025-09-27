#!/usr/bin/env python3
"""
Enhanced Portfolio Performance Analysis Script with eToro Integration

This version attempts to fetch historical performance data from eToro API
and compares it against major market indices from Yahoo Finance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
import argparse  # noqa: E402
import yfinance as yf  # noqa: E402
import asyncio  # noqa: E402
import aiohttp  # noqa: E402
import uuid  # noqa: E402
from typing import Dict, List  # noqa: E402
from tabulate import tabulate  # noqa: E402
import warnings  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

warnings.filterwarnings('ignore')
load_dotenv()


class EtoroPortfolioAnalyzer:
    """Analyzes portfolio performance using eToro API data and compares against benchmarks."""

    # Major market indices for comparison
    BENCHMARK_INDICES = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ 100',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000'
    }

    def __init__(self, username: str = None, debug: bool = False):
        """
        Initialize the analyzer.

        Args:
            username: eToro username
            debug: Enable debug output
        """
        self.username = username or os.getenv("ETORO_USERNAME", "plessas")
        self.api_key = os.getenv("ETORO_API_KEY")
        self.user_key = os.getenv("ETORO_USER_KEY")
        self.portfolio_data = None
        self.stats_data = None
        self.benchmark_data = {}
        self.debug = debug

    async def fetch_etoro_portfolio(self) -> dict:
        """Fetch current portfolio data from eToro API."""
        if not self.api_key or not self.user_key:
            return None

        url = f"https://www.etoro.com/api/public/v1/user-info/people/{self.username}/portfolio/live"

        headers = {
            "X-REQUEST-ID": str(uuid.uuid4()),
            "X-API-KEY": self.api_key,
            "X-USER-KEY": self.user_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return None
        except Exception:
            return None

    async def fetch_etoro_stats(self) -> dict:
        """Fetch portfolio statistics and performance data from eToro API."""
        if not self.api_key or not self.user_key:
            return None

        # Try multiple possible endpoints for stats
        endpoints = [
            f"https://www.etoro.com/api/public/v1/user-info/people/{self.username}/gain",
            f"https://www.etoro.com/api/public/v1/user-info/people/{self.username}/daily-gain",
            f"https://www.etoro.com/api/public/v1/user-info/people/{self.username}/stats",
            f"https://www.etoro.com/api/public/v1/user-info/people/{self.username}/performance"
        ]

        headers = {
            "X-REQUEST-ID": str(uuid.uuid4()),
            "X-API-KEY": self.api_key,
            "X-USER-KEY": self.user_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        for endpoint in endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data
                        elif response.status == 404:
                            continue  # Try next endpoint
                        elif response.status == 401 or response.status == 403:
                            continue
            except Exception:
                continue  # Try next endpoint

        return None

    def calculate_portfolio_metrics_from_csv(self) -> dict:
        """Calculate portfolio metrics from the CSV file as fallback."""
        try:
            df = pd.read_csv("yahoofinance/input/portfolio.csv")

            # Calculate overall portfolio performance
            total_profit = 0

            for _, row in df.iterrows():
                # Estimate investment amount (this is approximate)
                investment_pct = row['totalInvestmentPct']
                profit_pct = row['totalNetProfitPct']

                # Weight the profits by investment percentage
                total_profit += profit_pct * (investment_pct / 100)

            metrics = {
                'total_return_pct': total_profit,
                'holdings_count': len(df),
                'top_gainer': df.nlargest(1, 'totalNetProfitPct').iloc[0].to_dict() if len(df) > 0 else {},
                'top_loser': df.nsmallest(1, 'totalNetProfitPct').iloc[0].to_dict() if len(df) > 0 else {},
                'average_profit': df['totalNetProfitPct'].mean()
            }

            return metrics

        except Exception:
            return {}

    def fetch_benchmark_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for benchmark indices from Yahoo Finance."""
        for symbol in self.BENCHMARK_INDICES:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2y")
                if not hist.empty:
                    self.benchmark_data[symbol] = hist
            except Exception:
                pass

        return self.benchmark_data

    def calculate_returns(self, prices: pd.Series, periods: List[str] = None) -> Dict[str, float]:
        """Calculate returns for different time periods."""
        if periods is None:
            periods = ['1D', '1W', '1M', '3M', '6M', 'YTD', '1Y', '2Y']

        returns = {}
        current_price = prices.iloc[-1] if len(prices) > 0 else 0

        for period in periods:
            try:
                if period == '1D' and len(prices) > 1:
                    prev_price = prices.iloc[-2]
                    returns[period] = ((current_price / prev_price) - 1) * 100
                elif period == '1W' and len(prices) > 5:
                    week_ago = prices.index[-1] - timedelta(days=7)
                    prev_price = prices.loc[prices.index <= week_ago].iloc[-1] if any(prices.index <= week_ago) else prices.iloc[0]
                    returns[period] = ((current_price / prev_price) - 1) * 100
                elif period == '1M' and len(prices) > 20:
                    month_ago = prices.index[-1] - timedelta(days=30)
                    prev_price = prices.loc[prices.index <= month_ago].iloc[-1] if any(prices.index <= month_ago) else prices.iloc[0]
                    returns[period] = ((current_price / prev_price) - 1) * 100
                elif period == '3M' and len(prices) > 60:
                    three_months_ago = prices.index[-1] - timedelta(days=90)
                    prev_price = prices.loc[prices.index <= three_months_ago].iloc[-1] if any(prices.index <= three_months_ago) else prices.iloc[0]
                    returns[period] = ((current_price / prev_price) - 1) * 100
                elif period == '6M' and len(prices) > 120:
                    six_months_ago = prices.index[-1] - timedelta(days=180)
                    prev_price = prices.loc[prices.index <= six_months_ago].iloc[-1] if any(prices.index <= six_months_ago) else prices.iloc[0]
                    returns[period] = ((current_price / prev_price) - 1) * 100
                elif period == 'YTD':
                    try:
                        # Get current date and year
                        current_date = prices.index[-1]
                        current_year = current_date.year

                        if self.debug:
                            print("\nDEBUG YTD Calculation:")
                            print(f"  Current date: {current_date}")
                            print(f"  Current year: {current_year}")
                            print(f"  Total prices: {len(prices)}")
                            print(f"  Date range: {prices.index[0]} to {prices.index[-1]}")

                        # Method 1: Find first price of current year
                        year_prices = prices[prices.index.year == current_year]
                        if self.debug:
                            print(f"  Prices in {current_year}: {len(year_prices)}")

                        if len(year_prices) > 1:  # Need at least 2 prices to calculate return
                            first_price_of_year = year_prices.iloc[0]
                            ytd_return = ((current_price / first_price_of_year) - 1) * 100
                            if self.debug:
                                print(f"  First price of {current_year}: ${first_price_of_year:.2f} on {year_prices.index[0]}")
                                print(f"  Current price: ${current_price:.2f}")
                                print(f"  YTD return: {ytd_return:.2f}%")
                            returns[period] = ytd_return
                        else:
                            # Method 2: Use last trading day of previous year
                            prev_year = current_year - 1
                            prev_year_prices = prices[prices.index.year == prev_year]

                            if self.debug:
                                print(f"  No/insufficient {current_year} data, checking {prev_year}")
                                print(f"  Prices in {prev_year}: {len(prev_year_prices)}")

                            if len(prev_year_prices) > 0:
                                # Get last price of previous year
                                last_price_prev_year = prev_year_prices.iloc[-1]
                                ytd_return = ((current_price / last_price_prev_year) - 1) * 100
                                if self.debug:
                                    print(f"  Last price of {prev_year}: ${last_price_prev_year:.2f} on {prev_year_prices.index[-1]}")
                                    print(f"  Current price: ${current_price:.2f}")
                                    print(f"  YTD return: {ytd_return:.2f}%")
                                returns[period] = ytd_return
                            else:
                                # Method 3: Calculate from start of data if less than a year
                                days_of_data = (prices.index[-1] - prices.index[0]).days
                                if days_of_data < 365 and len(prices) > 1:
                                    # Annualize the return
                                    total_return = ((current_price / prices.iloc[0]) - 1)
                                    ytd_return = (total_return * 365 / days_of_data) * 100
                                    if self.debug:
                                        print(f"  Limited data ({days_of_data} days), annualizing")
                                        print(f"  YTD return (annualized): {ytd_return:.2f}%")
                                    returns[period] = ytd_return
                                else:
                                    if self.debug:
                                        print("  Unable to calculate YTD return")
                                    returns[period] = 0
                    except Exception:
                        if self.debug:
                            print("  ERROR calculating YTD")
                        returns[period] = 0
                elif period == '1Y' and len(prices) > 252:
                    year_ago = prices.index[-1] - timedelta(days=365)
                    prev_price = prices.loc[prices.index <= year_ago].iloc[-1] if any(prices.index <= year_ago) else prices.iloc[0]
                    returns[period] = ((current_price / prev_price) - 1) * 100
                elif period == '2Y':
                    two_years_ago = prices.index[-1] - timedelta(days=730)
                    if any(prices.index <= two_years_ago):
                        prev_price = prices.loc[prices.index <= two_years_ago].iloc[-1]
                    elif len(prices) > 0:
                        prev_price = prices.iloc[0]  # Use earliest available data
                    else:
                        prev_price = current_price
                    if prev_price > 0:
                        returns[period] = ((current_price / prev_price) - 1) * 100
                    else:
                        returns[period] = 0
                else:
                    returns[period] = 0
            except Exception:
                returns[period] = 0

        return returns

    async def generate_performance_report(self):
        """Generate comprehensive performance report using eToro and Yahoo Finance data."""
        # Fetch eToro portfolio data
        self.portfolio_data = await self.fetch_etoro_portfolio()

        # Try to fetch stats/performance data
        self.stats_data = await self.fetch_etoro_stats()

        # Initialize portfolio performance dict to store calculated values
        portfolio_performance = {}

        # Display stats data if available
        if self.stats_data:
            # Handle monthly and yearly gain data
            if isinstance(self.stats_data, dict) and 'monthly' in self.stats_data and 'yearly' in self.stats_data:
                # Process monthly data
                monthly_data = self.stats_data.get('monthly', [])

                # Extract gain values from monthly data objects
                def get_gain(item):
                    if isinstance(item, dict) and 'gain' in item:
                        return item['gain']
                    elif isinstance(item, (int, float)):
                        return item
                    return 0

                # Calculate recent monthly stats for portfolio
                if monthly_data and isinstance(monthly_data, list):
                    if len(monthly_data) >= 1:
                        portfolio_performance['1M'] = get_gain(monthly_data[-1])
                    if len(monthly_data) >= 3:
                        portfolio_performance['3M'] = sum(get_gain(x) for x in monthly_data[-3:])
                    if len(monthly_data) >= 6:
                        portfolio_performance['6M'] = sum(get_gain(x) for x in monthly_data[-6:])

                    # Calculate YTD from monthly data
                    current_year = datetime.now().year
                    ytd_sum = 0
                    for item in monthly_data:
                        if isinstance(item, dict) and 'timestamp' in item:
                            if str(current_year) in item['timestamp']:
                                ytd_sum += get_gain(item)
                    portfolio_performance['YTD'] = ytd_sum

                    if len(monthly_data) >= 12:
                        portfolio_performance['1Y'] = sum(get_gain(x) for x in monthly_data[-12:])

                    # Calculate 2Y from yearly data if available
                    yearly_data = self.stats_data.get('yearly', [])
                    if yearly_data and len(yearly_data) >= 2:
                        two_year_sum = 0
                        for year_item in yearly_data[-2:]:
                            two_year_sum += get_gain(year_item)
                        portfolio_performance['2Y'] = two_year_sum

        # Fetch and display benchmark comparisons
        self.fetch_benchmark_data()

        if self.benchmark_data or portfolio_performance:
            # Calculate benchmark returns
            benchmark_returns = {}
            for symbol in self.benchmark_data:
                benchmark_returns[symbol] = self.calculate_returns(
                    self.benchmark_data[symbol]['Close']
                )

            # Prepare data for performance comparison table
            periods = ['1D', '1W', '1M', '3M', '6M', 'YTD', '1Y', '2Y']
            table_data = []

            for period in periods:
                row = [period]

                # Portfolio return
                if period in portfolio_performance:
                    ret = portfolio_performance[period]
                    color = '\033[92m' if ret > 0 else '\033[91m' if ret < 0 else '\033[0m'
                    row.append(f"{color}{ret:+.2f}%\033[0m")
                else:
                    row.append("--")

                # Benchmark returns
                for symbol in self.BENCHMARK_INDICES:
                    if symbol in benchmark_returns and period in benchmark_returns[symbol]:
                        ret = benchmark_returns[symbol][period]
                        if ret != 0:
                            color = '\033[92m' if ret > 0 else '\033[91m'
                            row.append(f"{color}{ret:+.2f}%\033[0m")
                        else:
                            row.append("0.00%")
                    else:
                        row.append("--")

                table_data.append(row)

            # Create headers
            headers = ["Period", "Portfolio"] + [symbol for symbol in self.BENCHMARK_INDICES if symbol in self.benchmark_data]

            # Print performance comparison table
            print("\n")
            print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))

            # Calculate and display outperformance
            outperformance_data = []
            key_periods = ['1M', '3M', '6M', 'YTD', '1Y', '2Y']

            for period in key_periods:
                if period in portfolio_performance:
                    row = [period]
                    portfolio_ret = portfolio_performance[period]

                    for symbol in self.BENCHMARK_INDICES:
                        if symbol in benchmark_returns and period in benchmark_returns[symbol]:
                            bench_ret = benchmark_returns[symbol][period]
                            if bench_ret != 0 or portfolio_ret != 0:
                                outperformance = portfolio_ret - bench_ret
                                color = '\033[92m' if outperformance > 0 else '\033[91m'
                                sign = '+' if outperformance > 0 else ''
                                row.append(f"{color}{sign}{outperformance:.2f}%\033[0m")
                            else:
                                row.append("--")
                        else:
                            row.append("--")

                    outperformance_data.append(row)

            if outperformance_data:
                # Create headers for outperformance
                out_headers = ["Period"] + [f"vs {symbol}" for symbol in self.BENCHMARK_INDICES if symbol in self.benchmark_data]

                # Print outperformance table
                print("\n")
                print(tabulate(outperformance_data, headers=out_headers, tablefmt="fancy_grid"))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze eToro portfolio performance vs market indices')
    parser.add_argument('-u', '--username', type=str, default=None,
                        help='eToro username (default: from .env file)')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh data from APIs')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for YTD calculations')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = EtoroPortfolioAnalyzer(username=args.username, debug=args.debug)

    # Run async analysis
    asyncio.run(analyzer.generate_performance_report())


if __name__ == "__main__":
    main()

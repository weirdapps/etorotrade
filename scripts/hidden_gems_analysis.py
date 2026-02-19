#!/usr/bin/env python3
"""
Hidden Gems Analysis - Find contrarian picks from top performers

This script identifies stocks that:
1. Top performing popular investors (Top 100 by YTD gain) hold
2. But are NOT widely held by the broader group (low overall popularity)
3. Represent differentiated, alpha-generating opportunities

Cross-references with etorotrade signals for fundamental validation.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd


def load_latest_census(census_dir: str) -> dict:
    """Load the most recent census data file."""
    data_path = Path(census_dir) / "public" / "data"
    files = sorted([f for f in data_path.glob("etoro-data-*.json")], reverse=True)

    if not files:
        raise FileNotFoundError(f"No census data files found in {data_path}")

    latest_file = files[0]
    print(f"Loading census data from: {latest_file.name}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def load_portfolio(portfolio_path: str) -> set:
    """Load user's current portfolio tickers."""
    df = pd.read_csv(portfolio_path)
    # Handle various column names
    ticker_col = 'symbol' if 'symbol' in df.columns else 'TKR'
    return set(df[ticker_col].str.upper().tolist())


def load_market_signals(market_path: str) -> Dict[str, dict]:
    """Load market signals from etorotrade."""
    df = pd.read_csv(market_path)
    signals = {}

    for _, row in df.iterrows():
        ticker = str(row.get('TKR', '')).upper()
        if ticker:
            signals[ticker] = {
                'name': row.get('NAME', ''),
                'cap': row.get('CAP', ''),
                'price': row.get('PRC', ''),
                'target': row.get('TGT', ''),
                'upside': row.get('UP%', ''),
                'buy_pct': row.get('%B', ''),
                'analysts': row.get('#A', ''),
                'signal': row.get('BS', 'I'),  # Buy/Sell/Hold signal
                'exret': row.get('EXR', ''),
                'pef': row.get('PEF', ''),
                'roe': row.get('ROE', ''),
            }

    return signals


def extract_top_performers(investors: list, top_n: int = 100) -> list:
    """Extract top N performers by YTD gain."""
    # Sort by gain (YTD performance) descending
    sorted_investors = sorted(investors, key=lambda x: x.get('gain', 0), reverse=True)
    return sorted_investors[:top_n]


def extract_holdings_from_investor(investor: dict) -> List[Tuple[int, str, float]]:
    """Extract holdings from an investor's portfolio.

    Returns list of (instrumentId, symbol, allocation_pct)
    """
    holdings = []
    portfolio = investor.get('portfolio', {})
    positions = portfolio.get('positions', [])

    for pos in positions:
        instrument_id = pos.get('instrumentId')
        # Try to get symbol from instrument name
        symbol = pos.get('instrumentName', '')
        allocation = pos.get('investmentPct', 0)

        if instrument_id and allocation > 0:
            holdings.append((instrument_id, symbol, allocation))

    return holdings


def build_instrument_lookup(instruments_data: dict) -> Dict[int, dict]:
    """Build instrument ID to symbol/name lookup."""
    lookup = {}
    details = instruments_data.get('details', {})

    if isinstance(details, dict):
        for key, inst in details.items():
            inst_id = inst.get('instrumentId', int(key) if key.isdigit() else None)
            if inst_id:
                lookup[inst_id] = {
                    'symbol': inst.get('symbolFull', inst.get('symbol', f'ID{inst_id}')),
                    'name': inst.get('instrumentDisplayName', inst.get('name', 'Unknown')),
                    'type': inst.get('instrumentTypeID', inst.get('instrumentTypeId', 0))
                }
    elif isinstance(details, list):
        for inst in details:
            inst_id = inst.get('instrumentId')
            if inst_id:
                lookup[inst_id] = {
                    'symbol': inst.get('symbolFull', inst.get('symbol', f'ID{inst_id}')),
                    'name': inst.get('instrumentDisplayName', inst.get('name', 'Unknown')),
                    'type': inst.get('instrumentTypeID', inst.get('instrumentTypeId', 0))
                }

    return lookup


def analyze_hidden_gems(census_data: dict, my_portfolio: set, market_signals: dict) -> pd.DataFrame:
    """
    Find hidden gems: stocks held by top performers but not widely popular.

    Logic:
    1. Get Top 100 investors by YTD gain
    2. Build their aggregated holdings
    3. Compare against broad group (Top 1500) holdings
    4. Find stocks with HIGH top-performer popularity but LOW broad popularity
    """

    investors = census_data.get('investors', [])
    instruments = build_instrument_lookup(census_data.get('instruments', {}))
    analyses = census_data.get('analyses', [])

    # Get broad group holdings (typically analyses[3] = 1500 investors)
    broad_holdings = {}
    if len(analyses) > 3:
        for holding in analyses[3].get('topHoldings', []):
            symbol = holding.get('symbol', '').upper()
            broad_holdings[symbol] = {
                'holders_count': holding.get('holdersCount', 0),
                'avg_allocation': holding.get('avgAllocation', holding.get('averageAllocation', 0))
            }

    # Get Top 100 holdings for comparison
    top100_holdings = {}
    if len(analyses) > 0:
        for holding in analyses[0].get('topHoldings', []):
            symbol = holding.get('symbol', '').upper()
            top100_holdings[symbol] = {
                'holders_count': holding.get('holdersCount', 0),
                'avg_allocation': holding.get('avgAllocation', holding.get('averageAllocation', 0))
            }

    # Extract Top 100 performers by gain
    top_performers = extract_top_performers(investors, 100)
    print(f"\nAnalyzing {len(top_performers)} top performers by YTD gain")
    top3_names = [f"{inv.get('userName')} ({inv.get('gain', 0):.1f}%)" for inv in top_performers[:3]]
    print(f"Top 3 performers: {top3_names}")

    # Build aggregated holdings from top performers' portfolios
    performer_holdings = defaultdict(lambda: {'count': 0, 'total_alloc': 0, 'holders': set()})

    for inv in top_performers:
        holdings = extract_holdings_from_investor(inv)
        username = inv.get('userName', '')
        investor_gain = inv.get('gain', 0)

        for inst_id, _, alloc in holdings:
            if inst_id in instruments:
                symbol = instruments[inst_id]['symbol'].upper()
                # Only count each holder once per stock
                if username not in performer_holdings[symbol]['holders']:
                    performer_holdings[symbol]['count'] += 1
                    performer_holdings[symbol]['holders'].add(username)
                    performer_holdings[symbol]['total_alloc'] += alloc
                    performer_holdings[symbol]['name'] = instruments[inst_id]['name']
                    performer_holdings[symbol]['type'] = instruments[inst_id]['type']
                    performer_holdings[symbol]['avg_holder_gain'] = performer_holdings[symbol].get('avg_holder_gain', 0) + investor_gain

    # Find hidden gems:
    # - Held by multiple top performers (count >= 3)
    # - Low popularity in broad group (< 20% of broad holders)
    # - Not in my portfolio

    hidden_gems = []

    for symbol, data in performer_holdings.items():
        # Skip if in my portfolio
        clean_symbol = symbol.split('.')[0] if '.' in symbol else symbol
        if clean_symbol in my_portfolio or symbol in my_portfolio:
            continue

        # Skip crypto and other non-stock instruments (type 10 = crypto)
        if data.get('type') == 10:
            continue

        # Get broad group popularity
        broad_info = broad_holdings.get(symbol, {})
        broad_holders = broad_info.get('holders_count', 0)

        # Get Top 100 popularity
        top100_info = top100_holdings.get(symbol, {})
        top100_holders = top100_info.get('holders_count', 0)

        # Calculate metrics
        performer_count = data['count']
        avg_alloc_performers = data['total_alloc'] / performer_count if performer_count > 0 else 0

        # Hidden gem criteria:
        # 1. At least 3 top performers hold it
        # 2. Broad popularity is low relative to performer popularity
        # 3. OR it's not even in the broad top holdings (truly hidden)

        if performer_count >= 3:
            # Calculate "hidden score" - higher means more hidden gem potential
            # Low broad popularity + high performer conviction = high score
            if broad_holders > 0:
                popularity_ratio = performer_count / max(broad_holders, 1) * 100
            else:
                popularity_ratio = 100  # Not even tracked = very hidden

            # Calculate average gain of holders
            avg_holder_gain = data.get('avg_holder_gain', 0) / performer_count if performer_count > 0 else 0

            # Get signal data
            signal_data = market_signals.get(clean_symbol, {})

            holders_list = list(data['holders'])
            hidden_gems.append({
                'Symbol': symbol,
                'Name': data['name'][:30] if data['name'] else '',
                'Top Performers': performer_count,
                'Avg Alloc %': round(avg_alloc_performers, 2),
                'Holder Avg Gain': round(avg_holder_gain, 1),
                'Broad Holders': broad_holders if broad_holders > 0 else 'N/A',
                'Top100 Holders': top100_holders if top100_holders > 0 else 'N/A',
                'Hidden Score': round(popularity_ratio, 1),
                'Signal': signal_data.get('signal', 'N/A'),
                'Upside': signal_data.get('upside', 'N/A'),
                'Buy %': signal_data.get('buy_pct', 'N/A'),
                'Holders': ', '.join(holders_list[:5]) + ('...' if len(holders_list) > 5 else '')
            })

    # Sort by hidden score (high performer count + low broad popularity)
    df = pd.DataFrame(hidden_gems)
    if len(df) > 0:
        df = df.sort_values(['Top Performers', 'Hidden Score'], ascending=[False, False])

    return df


def main():
    """Main analysis function."""

    # Paths
    census_dir = os.path.expanduser("~/SourceCode/etoro_census")
    portfolio_path = os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/input/portfolio.csv")
    market_path = os.path.expanduser("~/SourceCode/etorotrade/yahoofinance/output/market.csv")
    output_dir = os.path.expanduser("~/.weirdapps-trading/census")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HIDDEN GEMS ANALYSIS")
    print("Finding contrarian picks from top performers")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    census_data = load_latest_census(census_dir)
    my_portfolio = load_portfolio(portfolio_path)
    market_signals = load_market_signals(market_path)

    print(f"    - Census: {len(census_data.get('investors', []))} investors")
    print(f"    - My portfolio: {len(my_portfolio)} positions")
    print(f"    - Market signals: {len(market_signals)} tickers")

    # Analyze
    print("\n[2] Analyzing hidden gems...")
    hidden_gems_df = analyze_hidden_gems(census_data, my_portfolio, market_signals)

    if len(hidden_gems_df) == 0:
        print("\n    No hidden gems found matching criteria.")
        return

    # Display results
    print("\n" + "=" * 70)
    print("HIDDEN GEMS - Contrarian Picks from Top Performers")
    print("(Stocks held by top performers but NOT widely popular)")
    print("=" * 70)

    # Split by signal
    buy_gems = hidden_gems_df[hidden_gems_df['Signal'] == 'B']
    hold_gems = hidden_gems_df[hidden_gems_df['Signal'] == 'H']
    other_gems = hidden_gems_df[~hidden_gems_df['Signal'].isin(['B', 'H', 'S'])]

    print(f"\n### STRONG CONVICTION (BUY Signal) - {len(buy_gems)} stocks")
    if len(buy_gems) > 0:
        print(buy_gems[['Symbol', 'Name', 'Top Performers', 'Avg Alloc %', 'Broad Holders', 'Upside', 'Buy %']].to_string(index=False))

    print(f"\n### MODERATE CONVICTION (HOLD Signal) - {len(hold_gems)} stocks")
    if len(hold_gems) > 0:
        print(hold_gems.head(15)[['Symbol', 'Name', 'Top Performers', 'Avg Alloc %', 'Broad Holders', 'Upside', 'Buy %']].to_string(index=False))

    print(f"\n### NEEDS RESEARCH (No Signal Data) - {len(other_gems)} stocks")
    if len(other_gems) > 0:
        print(other_gems.head(10)[['Symbol', 'Name', 'Top Performers', 'Avg Alloc %', 'Broad Holders', 'Holders']].to_string(index=False))

    # Save results
    output_file = os.path.join(output_dir, "hidden-gems-latest.csv")
    hidden_gems_df.to_csv(output_file, index=False)
    print(f"\n[3] Results saved to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total hidden gems found: {len(hidden_gems_df)}")
    print(f"  - With BUY signal: {len(buy_gems)}")
    print(f"  - With HOLD signal: {len(hold_gems)}")
    print(f"  - Needs research: {len(other_gems)}")

    # Top recommendations
    if len(buy_gems) > 0:
        print("\nTOP RECOMMENDATIONS (BUY signals held by most top performers):")
        for _, row in buy_gems.head(5).iterrows():
            print(f"  - {row['Symbol']}: Held by {row['Top Performers']} top performers, {row['Upside']} upside")


if __name__ == "__main__":
    main()

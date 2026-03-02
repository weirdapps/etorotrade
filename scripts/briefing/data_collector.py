"""Collects data from signal CSVs, census JSON, and live market feeds."""

import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def safe_float(value):
    """Parse a value to float, returning None for '--' or invalid values."""
    if value is None or value == '--' or value == '':
        return None
    try:
        val = str(value).replace('%', '').replace(',', '')
        return float(val)
    except (ValueError, TypeError):
        return None


def read_signal_csv(filepath):
    """Read a signal CSV file with abbreviated headers."""
    if not os.path.exists(filepath):
        return []

    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_portfolio(base_dir):
    """Read portfolio.csv and return parsed rows."""
    filepath = os.path.join(base_dir, 'yahoofinance/output/portfolio.csv')
    return read_signal_csv(filepath)


def read_buy_signals(base_dir):
    """Read buy.csv and return parsed rows."""
    filepath = os.path.join(base_dir, 'yahoofinance/output/buy.csv')
    return read_signal_csv(filepath)


def read_sell_signals(base_dir):
    """Read sell.csv and return parsed rows."""
    filepath = os.path.join(base_dir, 'yahoofinance/output/sell.csv')
    return read_signal_csv(filepath)


def read_hold_signals(base_dir):
    """Read hold.csv and return parsed rows."""
    filepath = os.path.join(base_dir, 'yahoofinance/output/hold.csv')
    return read_signal_csv(filepath)


def summarize_portfolio(rows):
    """Create a summary of portfolio positions."""
    if not rows:
        return {'count': 0, 'positions': [], 'buys': [], 'sells': [], 'holds': []}

    positions = []
    buys = []
    sells = []
    holds = []

    for row in rows:
        pos = {
            'ticker': row.get('TKR', ''),
            'name': row.get('NAME', ''),
            'price': safe_float(row.get('PRC')),
            'target': safe_float(row.get('TGT')),
            'upside': row.get('UP%', ''),
            'buy_pct': row.get('%B', ''),
            'exret': row.get('EXR', ''),
            'signal': row.get('BS', ''),
            'cap': row.get('CAP', ''),
            'beta': row.get('B', ''),
            'w52': row.get('52W', ''),
            'pet': row.get('PET', ''),
            'pef': row.get('PEF', ''),
            'roe': row.get('ROE', ''),
            'de': row.get('DE', ''),
            'pp': row.get('PP', ''),
        }
        positions.append(pos)

        signal = row.get('BS', '')
        if signal == 'B':
            buys.append(pos)
        elif signal == 'S':
            sells.append(pos)
        elif signal == 'H':
            holds.append(pos)

    return {
        'count': len(positions),
        'positions': positions,
        'buys': buys,
        'sells': sells,
        'holds': holds,
    }


def summarize_buy_opportunities(rows, limit=15):
    """Summarize top buy opportunities from buy.csv."""
    opportunities = []
    for row in rows[:limit]:
        opportunities.append({
            'ticker': row.get('TKR', ''),
            'name': row.get('NAME', ''),
            'cap': row.get('CAP', ''),
            'price': safe_float(row.get('PRC')),
            'upside': row.get('UP%', ''),
            'buy_pct': row.get('%B', ''),
            'exret': row.get('EXR', ''),
        })
    return {'count': len(rows), 'top': opportunities}


def read_census(census_dir):
    """Read the latest census JSON file.

    Supports two layouts:
    - data-archive branch: census_dir/data/etoro-data-*.json
    - local worktree:      census_dir/archive/data/etoro-data-*.json
    """
    # Try data-archive branch layout first, then local worktree layout
    archive_dir = os.path.join(census_dir, 'data')
    if not os.path.isdir(archive_dir):
        archive_dir = os.path.join(census_dir, 'archive/data')
    if not os.path.isdir(archive_dir):
        return None

    files = sorted(glob.glob(os.path.join(archive_dir, 'etoro-data-*.json')))
    if not files:
        return None

    latest = files[-1]
    with open(latest, 'r') as f:
        data = json.load(f)

    result = {
        'file': os.path.basename(latest),
        'metadata': data.get('metadata', {}),
    }

    # Extract analyses (first entry = top 100 PIs)
    analyses = data.get('analyses', [])
    if analyses:
        top_analysis = analyses[0]
        result['fear_greed'] = top_analysis.get('fearGreedIndex')
        result['averages'] = top_analysis.get('averages', {})
        result['distributions'] = top_analysis.get('distributions', {})

        top_holdings = top_analysis.get('topHoldings', [])
        result['top_holdings'] = [
            {
                'symbol': h.get('symbol', ''),
                'name': h.get('instrumentName', ''),
                'holders_pct': h.get('holdersPercentage', 0),
                'avg_alloc': h.get('averageAllocation', 0),
                'mtd_return': round(h.get('monthTDReturn', 0), 2),
            }
            for h in top_holdings[:15]
        ]

    return result


def read_pi_feeds(census_dir, date_str=None):
    """Read PI feeds JSON for today or a given date."""
    feeds_dir = os.path.join(census_dir, 'analysis/output')
    if not os.path.isdir(feeds_dir):
        return None

    if date_str:
        filepath = os.path.join(feeds_dir, f'pi-feeds-{date_str}.json')
    else:
        files = sorted(glob.glob(os.path.join(feeds_dir, 'pi-feeds-*.json')))
        if not files:
            return None
        filepath = files[-1]

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    result = {
        'total_posts': data.get('totalPosts', 0),
        'total_pis': data.get('totalPIs', 0),
        'top_tickers': data.get('topTickers', [])[:10],
        'stats': data.get('stats', {}),
    }

    # Summarize by category
    by_category = data.get('byCategory', {})
    result['categories'] = {}
    for cat, posts in by_category.items():
        result['categories'][cat] = {
            'count': len(posts),
            'tickers_mentioned': [],
        }
        ticker_set = set()
        for post in posts:
            for t in post.get('tickers', []):
                ticker_set.add(t)
        result['categories'][cat]['tickers_mentioned'] = sorted(ticker_set)[:10]

    return result


MARKET_INDICES = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'NASDAQ',
    '^DJI': 'Dow Jones',
    '^VIX': 'VIX',
    '^N225': 'Nikkei 225',
    '^HSI': 'Hang Seng',
    '^FTSE': 'FTSE 100',
    '^GDAXI': 'DAX',
    '^FCHI': 'CAC 40',
}

COMMODITIES = {
    'GC=F': 'Gold',
    'CL=F': 'WTI Crude',
    'BZ=F': 'Brent Crude',
    'DX-Y.NYB': 'US Dollar Index',
}


def fetch_market_data():
    """Fetch live market data via yfinance."""
    all_symbols = list(MARKET_INDICES.keys()) + list(COMMODITIES.keys())
    tickers = yf.Tickers(' '.join(all_symbols))

    indices = {}
    commodities = {}

    for symbol in all_symbols:
        try:
            ticker = tickers.tickers.get(symbol)
            if ticker is None:
                continue
            info = ticker.fast_info
            price = getattr(info, 'last_price', None)
            prev_close = getattr(info, 'previous_close', None)

            if price and prev_close and prev_close != 0:
                change_pct = ((price - prev_close) / prev_close) * 100
            else:
                change_pct = None

            entry = {
                'price': round(price, 2) if price else None,
                'change_pct': round(change_pct, 2) if change_pct else None,
            }

            if symbol in MARKET_INDICES:
                entry['name'] = MARKET_INDICES[symbol]
                indices[symbol] = entry
            else:
                entry['name'] = COMMODITIES[symbol]
                commodities[symbol] = entry
        except Exception:
            continue

    return {'indices': indices, 'commodities': commodities}


def collect_data(base_dir, census_dir, date_str=None):
    """Collect all data needed for the morning briefing.

    Args:
        base_dir: Path to etorotrade repo root
        census_dir: Path to etoro_census repo root
        date_str: Optional date string (YYYY-MM-DD) for historical data

    Returns:
        dict with all collected data
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y-%m-%d')

    # Read signals
    portfolio_rows = read_portfolio(base_dir)
    buy_rows = read_buy_signals(base_dir)
    sell_rows = read_sell_signals(base_dir)

    portfolio_summary = summarize_portfolio(portfolio_rows)
    buy_summary = summarize_buy_opportunities(buy_rows)

    # Read census
    census = read_census(census_dir)
    pi_feeds = read_pi_feeds(census_dir, date_str)

    # Fetch live market data
    market_data = fetch_market_data()

    # Signal file metadata
    portfolio_path = os.path.join(base_dir, 'yahoofinance/output/portfolio.csv')
    signals_mtime = None
    if os.path.exists(portfolio_path):
        signals_mtime = datetime.fromtimestamp(
            os.path.getmtime(portfolio_path)
        ).strftime('%Y-%m-%d %H:%M UTC')

    return {
        'date': date_str,
        'signals_updated': signals_mtime,
        'portfolio': portfolio_summary,
        'buy_opportunities': buy_summary,
        'sell_signals_count': len(sell_rows),
        'sell_top': [
            {
                'ticker': r.get('TKR', ''),
                'name': r.get('NAME', ''),
                'upside': r.get('UP%', ''),
                'buy_pct': r.get('%B', ''),
            }
            for r in sell_rows[:10]
        ],
        'census': census,
        'pi_feeds': pi_feeds,
        'market': market_data,
    }

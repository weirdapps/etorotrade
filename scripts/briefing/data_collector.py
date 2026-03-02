"""Collects data from signal CSVs, census JSON, and live market feeds."""

import csv
import glob
import json
import os
from datetime import datetime

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


def summarize_portfolio(rows):
    """Create a comprehensive summary of portfolio positions."""
    if not rows:
        return {'count': 0, 'positions': [], 'buys': [], 'sells': [],
                'holds': [], 'inconclusive': [], 'avg_beta': '--',
                'avg_w52': '--', 'avg_pp': '--'}

    positions = []
    buys = []
    sells = []
    holds = []
    inconclusive = []

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
        elif signal == 'I':
            inconclusive.append(pos)

    # Compute averages
    betas = [safe_float(p['beta']) for p in positions if safe_float(p['beta']) is not None]
    avg_beta = f"{sum(betas)/len(betas):.2f}" if betas else '--'

    w52s = [safe_float(p['w52']) for p in positions if safe_float(p['w52']) is not None]
    avg_w52 = f"{sum(w52s)/len(w52s):.1f}" if w52s else '--'

    pps = [safe_float(p['pp']) for p in positions if safe_float(p['pp']) is not None]
    avg_pp = f"{sum(pps)/len(pps):+.2f}" if pps else '--'

    return {
        'count': len(positions),
        'positions': positions,
        'buys': buys,
        'sells': sells,
        'holds': holds,
        'inconclusive': inconclusive,
        'avg_beta': avg_beta,
        'avg_w52': avg_w52,
        'avg_pp': avg_pp,
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
    """Read the latest census JSON file with full data extraction."""
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

    analyses = data.get('analyses', [])
    if not analyses:
        return result

    # Top 100 analysis (analyses[0])
    top = analyses[0]
    result['top100'] = {
        'investor_count': top.get('investorCount', 100),
        'fear_greed': top.get('fearGreedIndex'),
        'averages': top.get('averages', {}),
        'top_holdings': [
            {
                'symbol': h.get('symbol', ''),
                'name': h.get('instrumentName', ''),
                'holders_pct': h.get('holdersPercentage', 0),
                'avg_alloc': h.get('averageAllocation', 0),
            }
            for h in top.get('topHoldings', [])[:15]
        ],
    }

    # Top investors by category from topPerformers
    performers = top.get('topPerformers', [])
    if performers:
        by_copiers = sorted(performers, key=lambda x: -x.get('copiers', 0))[:3]
        by_gain = sorted(performers, key=lambda x: -x.get('gain', 0))[:3]
        by_trades = sorted(performers, key=lambda x: -x.get('trades', 0))[:3]
        by_risk = sorted(performers, key=lambda x: x.get('riskScore', 10))[:3]

        result['top_investors'] = {
            'by_copiers': [
                {'username': p['username'], 'copiers': p.get('copiers', 0),
                 'gain': p.get('gain', 0), 'risk': p.get('riskScore', 0)}
                for p in by_copiers
            ],
            'by_gain': [
                {'username': p['username'], 'copiers': p.get('copiers', 0),
                 'gain': p.get('gain', 0), 'risk': p.get('riskScore', 0)}
                for p in by_gain
            ],
            'by_trades': [
                {'username': p['username'], 'trades': p.get('trades', 0),
                 'gain': p.get('gain', 0), 'risk': p.get('riskScore', 0)}
                for p in by_trades
            ],
            'by_risk': [
                {'username': p['username'], 'copiers': p.get('copiers', 0),
                 'gain': p.get('gain', 0), 'risk': p.get('riskScore', 0)}
                for p in by_risk
            ],
        }

    # Broad analysis (last analysis = all investors)
    broad = analyses[-1]
    result['broad'] = {
        'investor_count': broad.get('investorCount', 0),
        'fear_greed': broad.get('fearGreedIndex'),
        'averages': broad.get('averages', {}),
        'top_holdings': [
            {
                'symbol': h.get('symbol', ''),
                'name': h.get('instrumentName', ''),
                'holders_pct': h.get('holdersPercentage', 0),
                'avg_alloc': h.get('averageAllocation', 0),
            }
            for h in broad.get('topHoldings', [])[:10]
        ],
    }

    # Back-compat aliases
    result['fear_greed'] = result['top100']['fear_greed']
    result['averages'] = result['top100']['averages']
    result['top_holdings'] = result['top100']['top_holdings']

    return result


def read_pi_feeds(census_dir, pi_feeds_dir=None, date_str=None):
    """Read PI feeds JSON from standalone file or census JSON.

    Tries multiple paths in order:
    1. Explicit pi_feeds_dir (for GH Actions separate checkout)
    2. census_dir/analysis/output/ (local worktree)
    """
    # Build list of directories to try
    dirs_to_try = []
    if pi_feeds_dir:
        dirs_to_try.append(pi_feeds_dir)
    dirs_to_try.append(os.path.join(census_dir, 'analysis/output'))

    for feeds_dir in dirs_to_try:
        if not os.path.isdir(feeds_dir):
            continue

        if date_str:
            # Try exact date and day before
            for d in [date_str]:
                filepath = os.path.join(feeds_dir, f'pi-feeds-{d}.json')
                if os.path.exists(filepath):
                    return _parse_pi_feeds(filepath)
            # Try day before
            from datetime import timedelta
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                prev = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
                filepath = os.path.join(feeds_dir, f'pi-feeds-{prev}.json')
                if os.path.exists(filepath):
                    return _parse_pi_feeds(filepath)
            except ValueError:
                pass

        # Fall back to latest file
        files = sorted(glob.glob(os.path.join(feeds_dir, 'pi-feeds-*.json')))
        if files:
            return _parse_pi_feeds(files[-1])

    return None


def _parse_pi_feeds(filepath):
    """Parse a PI feeds JSON file into structured data."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    result = {
        'file': os.path.basename(filepath),
        'total_posts': data.get('totalPosts', 0),
        'total_pis': data.get('totalPIs', 0),
        'top_tickers': data.get('topTickers', [])[:10],
    }

    # Extract posts by category with full text
    by_category = data.get('byCategory', {})
    result['categories'] = {}
    cat_labels = {
        'elite': 'Elite Investors (Top by Copiers)',
        'performers': 'Top Performers',
        'conservative': 'Conservative Investors',
        'active': 'Most Active Traders',
        'engaging': 'Most Engaging',
    }

    for cat_key, cat_label in cat_labels.items():
        cat_posts = by_category.get(cat_key, [])
        posts = []
        for p in cat_posts[:2]:  # Top 2 per category
            text = p.get('text', '')
            # Truncate long posts
            if len(text) > 300:
                text = text[:297] + '...'
            posts.append({
                'author': p.get('author', ''),
                'copiers': p.get('copiers', 0),
                'gain': p.get('gain', 0),
                'risk': p.get('riskScore', 0),
                'tickers': p.get('tickers', []),
                'text': text,
                'created': p.get('created', ''),
                'label': cat_label,
            })
        if posts:
            result['categories'][cat_key] = {
                'label': cat_label,
                'posts': posts,
            }

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


def collect_data(base_dir, census_dir, pi_feeds_dir=None, date_str=None):
    """Collect all data needed for the morning briefing."""
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
    pi_feeds = read_pi_feeds(census_dir, pi_feeds_dir, date_str)

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
        'census': census,
        'pi_feeds': pi_feeds,
        'market': market_data,
    }

"""Generates HTML morning briefing using Python templates + LLM for narrative."""

import json
import os
from datetime import datetime

from anthropic import AnthropicVertex

CSS = """
body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.5; background: #fff; }
h1 { color: #003841; border-bottom: 3px solid #007B85; padding-bottom: 10px; font-size: 22px; }
h2 { color: #007B85; border-top: 1px solid #eee; padding-top: 15px; font-size: 18px; margin-top: 25px; }
h3 { color: #003841; font-size: 15px; margin-top: 15px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0; }
th { background-color: #003841; color: white; padding: 8px; text-align: left; }
td { border: 1px solid #ddd; padding: 6px; }
tr:nth-child(even) { background-color: #f9f9f9; }
.buy { color: #28a745; font-weight: bold; }
.add { color: #17a2b8; font-weight: bold; }
.sell { color: #dc3545; font-weight: bold; }
.reduce { color: #fd7e14; font-weight: bold; }
.hold { color: #856404; font-weight: bold; }
.positive { color: #28a745; }
.negative { color: #dc3545; }
.highlight { background: #e8f4f8; border-left: 4px solid #007B85; padding: 12px; margin: 10px 0; border-radius: 0 4px 4px 0; }
.warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 10px 0; border-radius: 0 4px 4px 0; }
.alert { background: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 10px 0; border-radius: 0 4px 4px 0; }
.success { background: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 10px 0; border-radius: 0 4px 4px 0; }
.section-num { background: #007B85; color: white; border-radius: 50%; width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; margin-right: 8px; font-size: 12px; }
.metric-box { display: inline-block; padding: 8px 14px; margin: 4px 6px 4px 0; background: #f0f7f8; border-radius: 20px; border: 1px solid #007B85; font-size: 13px; }
.ticker { font-family: 'Courier New', monospace; font-weight: bold; }
.small { font-size: 11px; color: #666; }
.news-item { padding: 8px 0; border-bottom: 1px solid #eee; }
.quote { font-style: italic; color: #555; border-left: 2px solid #ccc; padding-left: 10px; margin: 6px 0; font-size: 13px; }
.pi-card { margin-bottom: 12px; padding: 10px; border: 1px solid #e0e0e0; border-radius: 6px; }
.header-meta { font-size: 12px; color: #666; margin-bottom: 15px; }
"""


def _signal_class(signal):
    """Return CSS class for a signal letter."""
    return {'B': 'buy', 'S': 'sell', 'H': 'hold', 'I': ''}.get(signal, '')


def _signal_label(signal):
    return {'B': 'BUY', 'S': 'SELL', 'H': 'HOLD', 'I': 'INCONCL'}.get(signal, signal)


def _change_class(value):
    """Return 'positive' or 'negative' CSS class based on value."""
    if value is None:
        return ''
    return 'positive' if value >= 0 else 'negative'


def _fmt_change(value):
    """Format a percentage change with sign."""
    if value is None:
        return '--'
    return f"{value:+.2f}%"


def _section_1_markets(data):
    """Section 1: Global Markets."""
    indices = data.get('market', {}).get('indices', {})
    commodities = data.get('market', {}).get('commodities', {})

    rows = []
    for sym, info in indices.items():
        name = info.get('name', sym)
        price = info.get('price')
        change = info.get('change_pct')
        cls = _change_class(change)
        price_str = f"{price:,.2f}" if price else '--'
        change_str = _fmt_change(change)
        rows.append(f'<tr><td>{name}</td><td>{price_str}</td>'
                     f'<td class="{cls}">{change_str}</td></tr>')

    comm_rows = []
    for sym, info in commodities.items():
        name = info.get('name', sym)
        price = info.get('price')
        change = info.get('change_pct')
        cls = _change_class(change)
        price_str = f"{price:,.2f}" if price else '--'
        change_str = _fmt_change(change)
        comm_rows.append(f'<tr><td>{name}</td><td>{price_str}</td>'
                          f'<td class="{cls}">{change_str}</td></tr>')

    return f"""
<h2><span class="section-num">1</span> Global Markets</h2>
<h3>Indices</h3>
<table>
<tr><th>Index</th><th>Level</th><th>Change</th></tr>
{"".join(rows)}
</table>
<h3>Commodities &amp; FX</h3>
<table>
<tr><th>Asset</th><th>Price</th><th>Change</th></tr>
{"".join(comm_rows)}
</table>
"""


def _section_2_portfolio(data):
    """Section 2: Portfolio Summary."""
    p = data.get('portfolio', {})
    buys = p.get('buys', [])
    sells = p.get('sells', [])
    holds = p.get('holds', [])
    total = p.get('count', 0)
    inconcl = total - len(buys) - len(sells) - len(holds)

    buy_tickers = ', '.join(pos['ticker'] for pos in buys[:8])
    hold_tickers = ', '.join(pos['ticker'] for pos in holds[:8])
    sell_tickers = ', '.join(pos['ticker'] for pos in sells[:8])

    # Compute averages
    positions = p.get('positions', [])
    betas = [float(pos['beta']) for pos in positions
             if pos.get('beta') and pos['beta'] != '--']
    avg_beta = f"{sum(betas)/len(betas):.2f}" if betas else '--'

    return f"""
<h2><span class="section-num">2</span> Portfolio Summary</h2>
<span class="metric-box"><strong>{total}</strong> Positions</span>
<span class="metric-box">Avg Beta: <strong>{avg_beta}</strong></span>
<table>
<tr><th>Signal</th><th>Count</th><th>%</th><th>Key Tickers</th></tr>
<tr><td class="buy">BUY</td><td>{len(buys)}</td><td>{len(buys)*100//total if total else 0}%</td><td>{buy_tickers}</td></tr>
<tr><td class="hold">HOLD</td><td>{len(holds)}</td><td>{len(holds)*100//total if total else 0}%</td><td>{hold_tickers}</td></tr>
<tr><td class="sell">SELL</td><td>{len(sells)}</td><td>{len(sells)*100//total if total else 0}%</td><td>{sell_tickers}</td></tr>
<tr><td style="color:#999;">INCONCLUSIVE</td><td>{inconcl}</td><td>{inconcl*100//total if total else 0}%</td><td></td></tr>
</table>
"""


def _section_3_signals(data):
    """Section 3: Selected Signals."""
    p = data.get('portfolio', {})
    buys = p.get('buys', [])
    sells = p.get('sells', [])
    holds = p.get('holds', [])

    # Buy opportunities not in portfolio
    buy_opps = data.get('buy_opportunities', {}).get('top', [])

    opp_rows = []
    for opp in buy_opps[:8]:
        opp_rows.append(
            f'<tr><td class="ticker">{opp["ticker"]}</td><td>{opp["name"]}</td>'
            f'<td>{opp.get("price", "--")}</td>'
            f'<td class="positive">{opp["upside"]}</td>'
            f'<td>{opp["buy_pct"]}</td>'
            f'<td class="positive">{opp["exret"]}</td></tr>'
        )

    # Portfolio BUYs (ADD)
    add_rows = []
    for pos in sorted(buys, key=lambda x: -(float(x['exret'].replace('%', ''))
                       if x.get('exret') and x['exret'] != '--' else 0))[:8]:
        add_rows.append(
            f'<tr><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td class="positive">{pos["upside"]}</td>'
            f'<td>{pos["buy_pct"]}</td>'
            f'<td class="positive">{pos["exret"]}</td></tr>'
        )

    # HOLDs
    hold_rows = []
    for pos in holds[:8]:
        hold_rows.append(
            f'<tr><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>{pos["upside"]}</td>'
            f'<td>{pos["buy_pct"]}</td></tr>'
        )

    # SELLs
    sell_rows = []
    for pos in sells:
        sell_rows.append(
            f'<tr><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td class="negative">{pos["upside"]}</td>'
            f'<td class="negative">{pos["buy_pct"]}</td>'
            f'<td class="sell">SELL</td></tr>'
        )

    sell_section = ""
    if sell_rows:
        sell_section = f"""
<h3 class="sell">SELL/REDUCE - Action Required</h3>
<div class="alert">
<table>
<tr><th>Ticker</th><th>Price</th><th>Upside</th><th>Bullish</th><th>Action</th></tr>
{"".join(sell_rows)}
</table>
</div>
"""

    return f"""
<h2><span class="section-num">3</span> Selected Signals</h2>
<h3 class="buy">BUY - New Opportunities (Not in Portfolio)</h3>
<table>
<tr><th>Ticker</th><th>Name</th><th>Price</th><th>Upside</th><th>Bullish</th><th>EXRET</th></tr>
{"".join(opp_rows)}
</table>
<h3 class="add">ADD - Increase Position (In Portfolio with BUY Signal)</h3>
<table>
<tr><th>Ticker</th><th>Price</th><th>Upside</th><th>Bullish</th><th>EXRET</th></tr>
{"".join(add_rows)}
</table>
<h3 class="hold">HOLD - Maintain Position</h3>
<table>
<tr><th>Ticker</th><th>Price</th><th>Upside</th><th>Bullish</th></tr>
{"".join(hold_rows)}
</table>
{sell_section}
"""


def _section_4_census(data):
    """Section 4: Census Intelligence."""
    census = data.get('census')
    if not census:
        return '<h2><span class="section-num">4</span> Census Intelligence</h2>\n<p>Census data not available.</p>'

    fg = census.get('fear_greed', '--')
    avgs = census.get('averages', {})
    cash = avgs.get('cashPercentage', '--')

    fg_label = 'Neutral'
    if isinstance(fg, (int, float)):
        if fg >= 75:
            fg_label = 'Extreme Greed'
        elif fg >= 55:
            fg_label = 'Greed'
        elif fg >= 45:
            fg_label = 'Neutral'
        elif fg >= 25:
            fg_label = 'Fear'
        else:
            fg_label = 'Extreme Fear'

    # Top holdings with signal alignment
    portfolio_tickers = {pos['ticker'] for pos in data.get('portfolio', {}).get('positions', [])}
    portfolio_signals = {pos['ticker']: pos['signal'] for pos in data.get('portfolio', {}).get('positions', [])}

    holding_rows = []
    for i, h in enumerate(census.get('top_holdings', [])[:10], 1):
        sym = h['symbol']
        signal = portfolio_signals.get(sym, '')
        sig_cls = _signal_class(signal)
        sig_label = _signal_label(signal) if signal else '--'
        if signal and signal in ('B',):
            alignment = '<td class="positive">ALIGNED</td>'
        elif signal and signal in ('S',):
            alignment = '<td class="negative">DIVERGENT</td>'
        elif signal:
            alignment = '<td style="color:#856404;">NEUTRAL</td>'
        else:
            alignment = '<td>--</td>'

        holding_rows.append(
            f'<tr><td>{i}</td><td class="ticker">{sym}</td>'
            f'<td>{h["holders_pct"]}%</td>'
            f'<td class="{sig_cls}">{sig_label}</td>'
            f'{alignment}</tr>'
        )

    return f"""
<h2><span class="section-num">4</span> Census Intelligence</h2>
<span class="metric-box">F&amp;G Index: <strong>{fg}</strong> ({fg_label})</span>
<span class="metric-box">Avg Cash: <strong>{cash}%</strong></span>
<span class="metric-box">Avg Gain: <strong>{avgs.get('gain', '--')}%</strong></span>
<span class="metric-box">Avg Risk: <strong>{avgs.get('riskScore', '--')}</strong></span>

<h3>Top PI Holdings</h3>
<table>
<tr><th>#</th><th>Stock</th><th>Holders %</th><th>Signal</th><th>Alignment</th></tr>
{"".join(holding_rows)}
</table>
"""


def _section_5_feeds(data):
    """Section 5: PI Feeds / Social Pulse."""
    pi_feeds = data.get('pi_feeds')
    if not pi_feeds:
        return '<h2><span class="section-num">5</span> Top Posts &amp; Commentary</h2>\n<p>PI feeds data not available for today.</p>'

    parts = []
    parts.append(f'<h2><span class="section-num">5</span> Top Posts &amp; Commentary</h2>')
    parts.append(f'<span class="metric-box">Total Posts: <strong>{pi_feeds.get("total_posts", 0)}</strong></span>')
    parts.append(f'<span class="metric-box">Active PIs: <strong>{pi_feeds.get("total_pis", 0)}</strong></span>')

    # Top tickers
    top_tickers = pi_feeds.get('top_tickers', [])
    if top_tickers:
        ticker_str = ' | '.join(f'<span class="ticker">${t}</span>' for t in top_tickers[:8])
        parts.append(f'<div class="highlight"><strong>Top Mentioned Tickers:</strong><br>{ticker_str}</div>')

    # Categories summary
    categories = pi_feeds.get('categories', {})
    cat_labels = {
        'elite': 'Elite Investors (Top by Copiers)',
        'performers': 'Top Performers',
        'conservative': 'Conservative Investors',
        'active': 'Most Active Traders',
        'engaging': 'Most Engaging',
    }
    for cat_key, cat_label in cat_labels.items():
        cat_data = categories.get(cat_key)
        if cat_data:
            tickers = ', '.join(cat_data.get('tickers_mentioned', [])[:5])
            parts.append(f'<h3>{cat_label}</h3>')
            parts.append(f'<p>{cat_data["count"]} posts'
                         f'{" — Tickers: " + tickers if tickers else ""}</p>')

    return '\n'.join(parts)


def _section_6_actions(data):
    """Section 6: Action Items."""
    p = data.get('portfolio', {})
    sells = p.get('sells', [])
    buys = p.get('buys', [])
    buy_opps = data.get('buy_opportunities', {}).get('top', [])

    rows = []

    # New BUY opportunities (top 3)
    for opp in buy_opps[:3]:
        rows.append(
            f'<tr><td class="buy">BUY</td><td class="ticker">{opp["ticker"]}</td>'
            f'<td>{opp.get("price", "--")}</td>'
            f'<td>UP {opp["upside"]}, {opp["buy_pct"]} bullish</td>'
            f'<td>Top market opportunity by EXRET</td></tr>'
        )

    # Portfolio ADDs (top 3)
    sorted_buys = sorted(buys, key=lambda x: -(float(x['exret'].replace('%', ''))
                          if x.get('exret') and x['exret'] != '--' else 0))
    for pos in sorted_buys[:3]:
        rows.append(
            f'<tr><td class="add">ADD</td><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>UP {pos["upside"]}, {pos["buy_pct"]} bullish</td>'
            f'<td>Portfolio position with strong BUY signal</td></tr>'
        )

    # SELL actions
    for pos in sells:
        rows.append(
            f'<tr><td class="sell">SELL</td><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>UP {pos["upside"]}, {pos["buy_pct"]} bullish</td>'
            f'<td>SELL signal triggered — review for exit</td></tr>'
        )

    return f"""
<h2><span class="section-num">6</span> Action Items</h2>
<table>
<tr><th>Action</th><th>Ticker</th><th>Price</th><th>Metrics</th><th>Rationale</th></tr>
{"".join(rows)}
</table>
"""


def _get_llm_narrative(data):
    """Call LLM for a short market narrative (2-3 paragraphs)."""
    project_id = os.environ.get('VERTEX_PROJECT_ID')
    region = os.environ.get('VERTEX_REGION', 'europe-west1')

    if not project_id:
        return None

    creds_json = os.environ.get('GCP_CREDENTIALS', '')
    if creds_json.strip().startswith('{'):
        import tempfile
        key_path = os.path.join(tempfile.gettempdir(), 'gcp_credentials.json')
        with open(key_path, 'w') as f:
            f.write(creds_json)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

    indices = data.get('market', {}).get('indices', {})
    commodities = data.get('market', {}).get('commodities', {})

    market_summary = []
    for sym, info in indices.items():
        name = info.get('name', sym)
        change = info.get('change_pct')
        if change is not None:
            market_summary.append(f"{name}: {change:+.2f}%")

    for sym, info in commodities.items():
        name = info.get('name', sym)
        change = info.get('change_pct')
        if change is not None:
            market_summary.append(f"{name}: {change:+.2f}%")

    census = data.get('census')
    fg = census.get('fear_greed', 'N/A') if census else 'N/A'
    sells = data.get('portfolio', {}).get('sells', [])
    sell_tickers = [s['ticker'] for s in sells]

    prompt = f"""Write a 2-paragraph market summary for a morning briefing on {data['date']}.

Market data: {', '.join(market_summary)}
Census Fear & Greed: {fg}
Portfolio SELL signals: {', '.join(sell_tickers) if sell_tickers else 'None'}
Portfolio positions: {data['portfolio']['count']}

Paragraph 1: What happened overnight — which markets moved and why. Be specific with numbers.
Paragraph 2: What to watch today — risks, opportunities, key levels.

Write in plain text (no HTML, no markdown). Be concise and professional. 150 words max total."""

    try:
        client = AnthropicVertex(project_id=project_id, region=region)
        message = client.messages.create(
            model="claude-sonnet-4-6@default",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        print(f"  Warning: LLM narrative failed: {e}")
        return None


def generate_briefing(data):
    """Generate the complete HTML briefing."""
    date_str = data['date']
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = dt.strftime('%A, %B %-d, %Y')

    census_file = data.get('census', {}).get('file', 'N/A') if data.get('census') else 'N/A'

    # Get LLM narrative
    narrative = _get_llm_narrative(data)
    narrative_html = ""
    if narrative:
        paragraphs = narrative.strip().split('\n\n')
        narrative_html = '<div class="highlight">' + ''.join(
            f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()
        ) + '</div>'

    # Build sections
    sections = [
        _section_1_markets(data),
        narrative_html,
        _section_2_portfolio(data),
        _section_3_signals(data),
        _section_4_census(data),
        _section_5_feeds(data),
        _section_6_actions(data),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Morning Briefing - {formatted_date}</title>
<style>
{CSS}
</style>
</head>
<body>

<h1>Morning Briefing - {formatted_date}</h1>
<div class="header-meta">
Signals: {data.get('signals_updated', 'N/A')} | Census: {census_file}<br>
Generated for Dimitrios Plessas | <em>Analysis only, not investment advice</em>
</div>

{"".join(sections)}

<hr>
<p class="small" style="margin-top: 20px; text-align: center;">
This briefing provides analysis only, not investment advice. All investment decisions are your own responsibility.<br>
Data sources: Yahoo Finance (signals), eToro Census (sentiment), PI Feeds (commentary)<br>
Generated: {formatted_date}
</p>

</body>
</html>"""

    return html


def extract_subject_hint(data):
    """Generate a subject line hint based on data highlights."""
    hints = []

    indices = data.get('market', {}).get('indices', {})
    sp500 = indices.get('^GSPC', {})
    change = sp500.get('change_pct')
    if change is not None:
        direction = "Up" if change >= 0 else "Down"
        hints.append(f"S&P {direction} {abs(change):.1f}%")

    vix = indices.get('^VIX', {})
    vix_price = vix.get('price')
    if vix_price:
        if vix_price > 25:
            hints.append(f"VIX Elevated {vix_price:.0f}")
        elif vix_price < 15:
            hints.append(f"VIX Low {vix_price:.0f}")

    portfolio = data.get('portfolio', {})
    sells = portfolio.get('sells', [])
    if sells:
        hints.append(f"{len(sells)} SELL Alert{'s' if len(sells) > 1 else ''}")

    census = data.get('census')
    if census and census.get('fear_greed'):
        fg = census['fear_greed']
        if fg >= 75:
            hints.append("Extreme Greed")
        elif fg <= 25:
            hints.append("Extreme Fear")

    return " | ".join(hints) if hints else "Daily Analysis"

"""Generates HTML morning briefing with 8 sections and LLM-powered analysis."""

import json
import os
import tempfile
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
.crisis-banner { background: linear-gradient(135deg, #8b0000, #dc3545); color: white; padding: 15px 20px; border-radius: 6px; margin-bottom: 20px; font-size: 14px; }
.crisis-banner strong { font-size: 16px; }
"""


def _signal_class(signal):
    return {'B': 'buy', 'S': 'sell', 'H': 'hold', 'I': ''}.get(signal, '')


def _signal_label(signal):
    return {'B': 'BUY', 'S': 'SELL', 'H': 'HOLD', 'I': 'INCONCL'}.get(signal, signal)


def _change_class(value):
    if value is None:
        return ''
    return 'positive' if value >= 0 else 'negative'


def _fmt_change(value):
    if value is None:
        return '--'
    return f"{value:+.2f}%"


def _esc(text):
    """HTML-escape text."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# ---------------------------------------------------------------------------
# LLM Analysis
# ---------------------------------------------------------------------------

def _build_llm_prompt(data):
    """Build a comprehensive prompt for the LLM to analyze all data."""
    date_str = data['date']
    p = data.get('portfolio', {})
    market = data.get('market', {})
    census = data.get('census')
    pi_feeds = data.get('pi_feeds')

    # Market summary
    market_lines = []
    for sym, info in market.get('indices', {}).items():
        c = info.get('change_pct')
        if c is not None:
            market_lines.append(f"{info.get('name', sym)}: {info['price']:,.0f} ({c:+.2f}%)")
    for sym, info in market.get('commodities', {}).items():
        c = info.get('change_pct')
        if c is not None:
            market_lines.append(f"{info.get('name', sym)}: {info['price']:,.2f} ({c:+.2f}%)")

    # Portfolio summary
    buys = p.get('buys', [])
    sells = p.get('sells', [])
    holds = p.get('holds', [])

    sell_info = []
    for s in sells:
        sell_info.append(f"{s['ticker']} (price {s.get('price','--')}, upside {s['upside']}, bullish {s['buy_pct']}, 52W {s['w52']})")

    buy_top = sorted(buys, key=lambda x: -(float(x['exret'].replace('%', ''))
                     if x.get('exret') and x['exret'] != '--' else 0))[:8]
    buy_info = []
    for b in buy_top:
        buy_info.append(f"{b['ticker']} (upside {b['upside']}, bullish {b['buy_pct']}, EXRET {b['exret']}, 52W {b['w52']}, PEF {b['pef']})")

    hold_info = []
    for h in holds[:8]:
        hold_info.append(f"{h['ticker']} (upside {h['upside']}, bullish {h['buy_pct']}, 52W {h['w52']}, PEF {h['pef']})")

    # Buy opportunities (not in portfolio)
    opps = data.get('buy_opportunities', {}).get('top', [])
    opp_info = []
    for o in opps[:8]:
        opp_info.append(f"{o['ticker']} ({o['name']}, upside {o['upside']}, bullish {o['buy_pct']}, EXRET {o['exret']})")

    # Census
    census_text = "Census data not available."
    if census:
        top100 = census.get('top100', {})
        broad = census.get('broad', {})
        fg_top = top100.get('fear_greed', '--')
        fg_broad = broad.get('fear_greed', '--')
        top_h = top100.get('top_holdings', [])[:8]
        broad_h = broad.get('top_holdings', [])[:6]
        holding_lines = [f"{h['symbol']} ({h['holders_pct']}% holders)" for h in top_h]
        broad_lines = [f"{h['symbol']} ({h['holders_pct']}% holders)" for h in broad_h]
        census_text = (
            f"Top 100 PIs: F&G={fg_top}, Cash={top100.get('averages', {}).get('cashPercentage', '--')}%\n"
            f"Top holdings: {', '.join(holding_lines)}\n"
            f"Broad {broad.get('investor_count', '?')} PIs: F&G={fg_broad}, Cash={broad.get('averages', {}).get('cashPercentage', '--')}%\n"
            f"Broad holdings: {', '.join(broad_lines)}"
        )

    # PI feeds summary
    feeds_text = "No PI feeds available."
    if pi_feeds and pi_feeds.get('total_posts', 0) > 0:
        top_t = pi_feeds.get('top_tickers', [])
        ticker_str = ', '.join(f"${t['ticker']}({t['count']})" for t in top_t[:8])
        feeds_text = f"PI posts: {pi_feeds['total_posts']} from {pi_feeds['total_pis']} PIs. Top tickers: {ticker_str}"

    prompt = f"""You are a financial analyst writing a morning briefing for {date_str}.

MARKET DATA:
{chr(10).join(market_lines)}

PORTFOLIO ({p.get('count', 0)} positions, avg beta {p.get('avg_beta', '--')}, avg 52W% {p.get('avg_w52', '--')}, avg PP {p.get('avg_pp', '--')}):
BUY signals ({len(buys)}): {chr(10).join(buy_info)}
SELL signals ({len(sells)}): {chr(10).join(sell_info) if sell_info else 'None'}
HOLD signals ({len(holds)}): {chr(10).join(hold_info)}

NEW BUY OPPORTUNITIES (not in portfolio):
{chr(10).join(opp_info)}

CENSUS:
{census_text}

PI FEEDS:
{feeds_text}

Return a JSON object with these fields (no markdown, just raw JSON):

{{
  "crisis_banner": null or "SHORT crisis headline if VIX>25 or major market event evident from data",
  "market_sentiment": "One sentence: BULLISH/BEARISH/MIXED + why, based on the data",
  "sector_movers": "One sentence on which sectors/themes are moving based on the data patterns",
  "key_events_recent": [
    {{"date": "Mon DD", "event": "Brief event description", "impact": "Brief impact", "css": "positive/negative/hold"}}
  ],
  "key_events_upcoming": [
    {{"date": "Mon DD", "event": "Brief event", "importance": "CRITICAL/HIGH/MEDIUM"}}
  ],
  "portfolio_risk_analysis": "2-3 sentences analyzing portfolio risk profile based on beta, 52W%, signal distribution",
  "signal_notes": {{
    "TICKER": "Brief contextual note for this signal (max 10 words)"
  }},
  "census_divergence": "1-2 sentences on divergence between PI holdings and our signals",
  "market_news": [
    {{"headline": "News headline", "detail": "1-2 sentence detail", "css": "positive/negative/hold", "tickers": ["TICK"]}}
  ],
  "portfolio_news": [
    {{"ticker": "TICK", "note": "Brief portfolio-relevant insight"}}
  ],
  "action_rationale": {{
    "TICKER": "Contextual rationale for the action (max 15 words)"
  }},
  "priority_actions": ["Action 1", "Action 2", "Action 3", "Action 4", "Action 5"]
}}

IMPORTANT:
- key_events_recent: 3-5 major events from the past week that explain today's market moves
- key_events_upcoming: 3-5 upcoming events this week (economic data, earnings, etc.)
- signal_notes: provide notes for ALL sell tickers, top 5 buy tickers, and top 5 hold tickers
- market_news: 4-5 major market news items relevant to the portfolio
- portfolio_news: 5-8 ticker-specific insights for portfolio positions
- action_rationale: for each ticker in sells, top 3 buy opps, top 3 portfolio buys
- Be specific with numbers from the data. Reference actual prices, percentages.
- Return ONLY valid JSON, no explanation text."""

    return prompt


def _get_llm_analysis(data):
    """Call LLM for comprehensive contextual analysis."""
    project_id = os.environ.get('VERTEX_PROJECT_ID')
    region = os.environ.get('VERTEX_REGION', 'europe-west1')

    if not project_id:
        return None

    creds_json = os.environ.get('GCP_CREDENTIALS', '')
    if creds_json.strip().startswith('{'):
        key_path = os.path.join(tempfile.gettempdir(), 'gcp_credentials.json')
        with open(key_path, 'w') as f:
            f.write(creds_json)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

    prompt = _build_llm_prompt(data)

    try:
        client = AnthropicVertex(project_id=project_id, region=region)
        message = client.messages.create(
            model="claude-sonnet-4-6@default",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()
        # Strip markdown code fences if present
        if response_text.startswith('```'):
            response_text = response_text.split('\n', 1)[1]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
        return json.loads(response_text)
    except Exception as e:
        print(f"  Warning: LLM analysis failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_1_markets(data, llm):
    """Section 1: Global Markets with contextual analysis."""
    indices = data.get('market', {}).get('indices', {})
    commodities = data.get('market', {}).get('commodities', {})

    # Crisis banner
    crisis = ''
    if llm and llm.get('crisis_banner'):
        crisis = f'<div class="crisis-banner"><strong>{_esc(llm["crisis_banner"])}</strong></div>\n'

    # US indices
    us_syms = ['^GSPC', '^IXIC', '^DJI', '^VIX']
    us_rows = []
    for sym in us_syms:
        info = indices.get(sym)
        if not info:
            continue
        cls = _change_class(info.get('change_pct'))
        price = f"{info['price']:,.2f}" if info.get('price') else '--'
        change = _fmt_change(info.get('change_pct'))
        us_rows.append(f'<tr><td>{info["name"]}</td><td>{price}</td>'
                       f'<td class="{cls}">{change}</td></tr>')

    # Asia
    asia_syms = ['^N225', '^HSI']
    asia_rows = []
    for sym in asia_syms:
        info = indices.get(sym)
        if not info:
            continue
        cls = _change_class(info.get('change_pct'))
        price = f"{info['price']:,.2f}" if info.get('price') else '--'
        change = _fmt_change(info.get('change_pct'))
        asia_rows.append(f'<tr><td>{info["name"]}</td><td>{price}</td>'
                         f'<td class="{cls}">{change}</td></tr>')

    # Europe
    eu_syms = ['^FTSE', '^GDAXI', '^FCHI']
    eu_rows = []
    for sym in eu_syms:
        info = indices.get(sym)
        if not info:
            continue
        cls = _change_class(info.get('change_pct'))
        price = f"{info['price']:,.2f}" if info.get('price') else '--'
        change = _fmt_change(info.get('change_pct'))
        eu_rows.append(f'<tr><td>{info["name"]}</td><td>{price}</td>'
                       f'<td class="{cls}">{change}</td></tr>')

    # Commodities
    comm_rows = []
    for sym, info in commodities.items():
        cls = _change_class(info.get('change_pct'))
        price = f"{info['price']:,.2f}" if info.get('price') else '--'
        change = _fmt_change(info.get('change_pct'))
        comm_rows.append(f'<tr><td>{info["name"]}</td><td>{price}</td>'
                         f'<td class="{cls}">{change}</td></tr>')

    # Sector movers
    sector_html = ''
    if llm and llm.get('sector_movers'):
        sector_html = f'<div class="highlight"><strong>Sector Movers:</strong> {_esc(llm["sector_movers"])}</div>\n'

    # Sentiment
    sentiment_html = ''
    if llm and llm.get('market_sentiment'):
        sentiment_html = f'<div class="highlight"><strong>Overnight Sentiment:</strong> {_esc(llm["market_sentiment"])}</div>\n'

    return f"""{crisis}
<h2><span class="section-num">1</span> Global Markets</h2>
<h3>US Markets</h3>
<table>
<tr><th>Index</th><th>Level</th><th>Change</th></tr>
{"".join(us_rows)}
</table>
<h3>Commodities &amp; Safe Havens</h3>
<table>
<tr><th>Asset</th><th>Level</th><th>Change</th></tr>
{"".join(comm_rows)}
</table>
<h3>Asia</h3>
<table>
<tr><th>Index</th><th>Level</th><th>Change</th></tr>
{"".join(asia_rows)}
</table>
<h3>Europe</h3>
<table>
<tr><th>Index</th><th>Level</th><th>Change</th></tr>
{"".join(eu_rows)}
</table>
{sector_html}{sentiment_html}"""


def _section_2_events(llm):
    """Section 2: Key Events."""
    parts = ['<h2><span class="section-num">2</span> Key Events</h2>']

    recent = llm.get('key_events_recent', []) if llm else []
    upcoming = llm.get('key_events_upcoming', []) if llm else []

    if recent:
        parts.append('<h3>Recent Major Events</h3>')
        parts.append('<table>')
        parts.append('<tr><th>Date</th><th>Event</th><th>Impact</th></tr>')
        for evt in recent:
            css = evt.get('css', '')
            parts.append(
                f'<tr><td>{_esc(evt.get("date", ""))}</td>'
                f'<td class="{css}">{_esc(evt.get("event", ""))}</td>'
                f'<td class="{css}">{_esc(evt.get("impact", ""))}</td></tr>'
            )
        parts.append('</table>')

    if upcoming:
        parts.append('<h3>Today &amp; Upcoming</h3>')
        parts.append('<table>')
        parts.append('<tr><th>Date</th><th>Event</th><th>Importance</th></tr>')
        for evt in upcoming:
            imp = evt.get('importance', 'MEDIUM')
            css = 'sell' if imp == 'CRITICAL' else ('hold' if imp == 'HIGH' else '')
            parts.append(
                f'<tr><td>{_esc(evt.get("date", ""))}</td>'
                f'<td>{_esc(evt.get("event", ""))}</td>'
                f'<td class="{css}">{imp}</td></tr>'
            )
        parts.append('</table>')

    if not recent and not upcoming:
        parts.append('<p>Event data not available.</p>')

    return '\n'.join(parts)


def _section_3_portfolio(data, llm):
    """Section 3: Portfolio Summary with enhanced metrics."""
    p = data.get('portfolio', {})
    buys = p.get('buys', [])
    sells = p.get('sells', [])
    holds = p.get('holds', [])
    inconcl = p.get('inconclusive', [])
    total = p.get('count', 0)

    buy_tickers = ', '.join(pos['ticker'] for pos in buys[:12])
    hold_tickers = ', '.join(pos['ticker'] for pos in holds[:8])
    sell_tickers = ', '.join(pos['ticker'] for pos in sells[:8])
    inconcl_tickers = ', '.join(pos['ticker'] for pos in inconcl[:4])

    buy_pct = f"{len(buys)*100/total:.1f}" if total else '0'
    hold_pct = f"{len(holds)*100/total:.1f}" if total else '0'
    sell_pct = f"{len(sells)*100/total:.1f}" if total else '0'
    inconcl_pct = f"{len(inconcl)*100/total:.1f}" if total else '0'

    risk_html = ''
    if llm and llm.get('portfolio_risk_analysis'):
        risk_html = f'<div class="highlight"><strong>Risk Profile:</strong> {_esc(llm["portfolio_risk_analysis"])}</div>'

    return f"""
<h2><span class="section-num">3</span> Portfolio Summary</h2>
<span class="metric-box"><strong>{total}</strong> Positions</span>
<span class="metric-box">Avg Beta: <strong>{p.get('avg_beta', '--')}</strong></span>
<span class="metric-box">Avg 52W%: <strong>{p.get('avg_w52', '--')}%</strong></span>
<span class="metric-box">Avg PP: <strong>{p.get('avg_pp', '--')}%</strong></span>
<table>
<tr><th>Signal</th><th>Count</th><th>%</th><th>Key Tickers</th></tr>
<tr><td class="buy">BUY (ADD)</td><td>{len(buys)}</td><td>{buy_pct}%</td><td>{buy_tickers}</td></tr>
<tr><td class="hold">HOLD</td><td>{len(holds)}</td><td>{hold_pct}%</td><td>{hold_tickers}</td></tr>
<tr><td class="sell">SELL</td><td>{len(sells)}</td><td>{sell_pct}%</td><td>{sell_tickers}</td></tr>
<tr><td style="color:#999;">INCONCLUSIVE</td><td>{len(inconcl)}</td><td>{inconcl_pct}%</td><td>{inconcl_tickers}</td></tr>
</table>
{risk_html}"""


def _section_4_signals(data, llm):
    """Section 4: Selected Signals with contextual notes."""
    p = data.get('portfolio', {})
    buys = p.get('buys', [])
    sells = p.get('sells', [])
    holds = p.get('holds', [])
    buy_opps = data.get('buy_opportunities', {}).get('top', [])
    signal_notes = llm.get('signal_notes', {}) if llm else {}

    # BUY opportunities not in portfolio
    opp_rows = []
    for opp in buy_opps[:5]:
        note = signal_notes.get(opp['ticker'], '')
        note_td = f'<td>{_esc(note)}</td>' if note else '<td></td>'
        opp_rows.append(
            f'<tr><td class="ticker">{opp["ticker"]}</td><td>{_esc(opp["name"])}</td>'
            f'<td>{opp.get("price", "--")}</td>'
            f'<td class="positive">{opp["upside"]}</td>'
            f'<td>{opp["buy_pct"]}</td>'
            f'<td class="positive">{opp["exret"]}</td></tr>'
        )

    # Portfolio BUYs (ADD) — sorted by EXRET
    add_rows = []
    sorted_buys = sorted(buys, key=lambda x: -(float(x['exret'].replace('%', ''))
                         if x.get('exret') and x['exret'] != '--' else 0))[:5]
    for pos in sorted_buys:
        note = signal_notes.get(pos['ticker'], '')
        add_rows.append(
            f'<tr><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td class="positive">{pos["upside"]}</td>'
            f'<td>{pos["buy_pct"]}</td>'
            f'<td class="positive">{pos["exret"]}</td>'
            f'<td>{_esc(note)}</td></tr>'
        )

    # HOLDs
    hold_rows = []
    for pos in holds[:7]:
        note = signal_notes.get(pos['ticker'], '')
        hold_rows.append(
            f'<tr><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>{pos["upside"]}</td>'
            f'<td>{pos["buy_pct"]}</td>'
            f'<td>{_esc(note)}</td></tr>'
        )

    # SELLs
    sell_rows = []
    for pos in sells:
        note = signal_notes.get(pos['ticker'], '')
        sell_rows.append(
            f'<tr><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td class="negative">{pos["upside"]}</td>'
            f'<td class="negative">{pos["buy_pct"]}</td>'
            f'<td>{pos["w52"]}</td>'
            f'<td class="sell">{_esc(note) if note else "SELL"}</td></tr>'
        )

    sell_section = ""
    if sell_rows:
        sell_section = f"""
<h3 class="sell">SELL/REDUCE - Action Required</h3>
<div class="alert">
<table>
<tr><th>Ticker</th><th>Price</th><th>Upside</th><th>Bullish</th><th>52W%</th><th>Action</th></tr>
{"".join(sell_rows)}
</table>
</div>"""

    return f"""
<h2><span class="section-num">4</span> Selected Signals</h2>
<h3 class="buy">BUY - New Opportunities (Not in Portfolio)</h3>
<table>
<tr><th>Ticker</th><th>Name</th><th>Price</th><th>Upside</th><th>Bullish</th><th>EXRET</th></tr>
{"".join(opp_rows)}
</table>
<h3 class="add">ADD - Increase Position (In Portfolio with BUY Signal)</h3>
<table>
<tr><th>Ticker</th><th>Price</th><th>Upside</th><th>Bullish</th><th>EXRET</th><th>Note</th></tr>
{"".join(add_rows)}
</table>
<h3 class="hold">HOLD - Maintain Position</h3>
<table>
<tr><th>Ticker</th><th>Price</th><th>Upside</th><th>Bullish</th><th>Note</th></tr>
{"".join(hold_rows)}
</table>
{sell_section}"""


def _section_5_census(data, llm):
    """Section 5: Census Intelligence with Top 100 + Broad analysis."""
    census = data.get('census')
    if not census:
        return '<h2><span class="section-num">5</span> Census Intelligence</h2>\n<p>Census data not available.</p>'

    top100 = census.get('top100', {})
    broad = census.get('broad', {})
    fg_top = top100.get('fear_greed', '--')
    fg_broad = broad.get('fear_greed', '--')
    avgs_top = top100.get('averages', {})
    avgs_broad = broad.get('averages', {})

    def fg_label(fg):
        if not isinstance(fg, (int, float)):
            return 'N/A'
        if fg >= 75: return 'Extreme Greed'
        if fg >= 55: return 'Greed'
        if fg >= 45: return 'Neutral'
        if fg >= 25: return 'Fear'
        return 'Extreme Fear'

    # Top 100 holdings with signal alignment
    # Build signal lookup that handles crypto naming (BTC-USD -> BTC, ETH-USD -> ETH)
    portfolio_signals = {}
    for pos in data.get('portfolio', {}).get('positions', []):
        t = pos['ticker']
        portfolio_signals[t] = pos['signal']
        # Also map the short form for crypto/forex
        if '-' in t:
            portfolio_signals[t.split('-')[0]] = pos['signal']

    def _holding_rows(holdings, limit=8):
        rows = []
        for i, h in enumerate(holdings[:limit], 1):
            sym = h['symbol']
            signal = portfolio_signals.get(sym, '')
            sig_cls = _signal_class(signal)
            sig_label = _signal_label(signal) if signal else '--'
            if signal == 'B':
                alignment = '<td class="positive">ALIGNED</td>'
            elif signal == 'S':
                alignment = '<td class="negative">DIVERGENT</td>'
            elif signal:
                alignment = '<td style="color:#856404;">NEUTRAL</td>'
            else:
                alignment = '<td>--</td>'
            rows.append(
                f'<tr><td>{i}</td><td class="ticker">{sym}</td>'
                f'<td>{h["holders_pct"]}%</td>'
                f'<td class="{sig_cls}">{sig_label}</td>'
                f'{alignment}</tr>'
            )
        return ''.join(rows)

    # Top investors by category
    investors_html = ''
    top_inv = census.get('top_investors')
    if top_inv:
        inv_rows = []
        cats = [
            ('By Copiers', 'by_copiers', lambda p: f'@{p["username"]} ({p["copiers"]:,})'),
            ('By Performance', 'by_gain', lambda p: f'@{p["username"]} ({p["gain"]:+.1f}%)'),
            ('By Activity', 'by_trades', lambda p: f'@{p["username"]} ({p["trades"]} trades)'),
            ('Lowest Risk', 'by_risk', lambda p: f'@{p["username"]} (Risk {p["risk"]})'),
        ]
        for label, key, fmt in cats:
            investors = top_inv.get(key, [])
            cells = '</td><td>'.join(fmt(p) for p in investors[:3])
            inv_rows.append(f'<tr><td><strong>{label}</strong></td><td>{cells}</td></tr>')

        investors_html = f"""
<h3>Top Investors by Category</h3>
<table>
<tr><th>Category</th><th>#1</th><th>#2</th><th>#3</th></tr>
{"".join(inv_rows)}
</table>"""

    # Divergence alert
    divergence_html = ''
    if llm and llm.get('census_divergence'):
        divergence_html = f'<div class="highlight"><strong>Divergence Alert:</strong> {_esc(llm["census_divergence"])}</div>'

    return f"""
<h2><span class="section-num">5</span> Census Intelligence</h2>
<span class="metric-box">F&amp;G Top 100: <strong>{fg_top}</strong> ({fg_label(fg_top)})</span>
<span class="metric-box">F&amp;G Broad: <strong>{fg_broad}</strong> ({fg_label(fg_broad)})</span>
<span class="metric-box">Top 100 Cash: <strong>{avgs_top.get('cashPercentage', '--')}%</strong></span>
<span class="metric-box">Broad Cash: <strong>{avgs_broad.get('cashPercentage', '--')}%</strong></span>
<h3>Top {top100.get('investor_count', 100)} Popular Investors - Holdings</h3>
<table>
<tr><th>#</th><th>Stock</th><th>Holders %</th><th>Signal</th><th>Alignment</th></tr>
{_holding_rows(top100.get('top_holdings', []))}
</table>
<h3>Broad {broad.get('investor_count', '?')} Investors - Holdings</h3>
<table>
<tr><th>#</th><th>Stock</th><th>Holders %</th><th>Signal</th><th>Alignment</th></tr>
{_holding_rows(broad.get('top_holdings', []), 6)}
</table>
{investors_html}
{divergence_html}"""


def _section_6_feeds(data):
    """Section 6: Top Posts & Commentary from PI feeds."""
    pi_feeds = data.get('pi_feeds')
    if not pi_feeds or pi_feeds.get('total_posts', 0) == 0:
        return '<h2><span class="section-num">6</span> Top Posts &amp; Commentary</h2>\n<p>PI feeds data not available for today.</p>'

    parts = ['<h2><span class="section-num">6</span> Top Posts &amp; Commentary</h2>']
    parts.append(f'<span class="metric-box">Total Posts: <strong>{pi_feeds["total_posts"]}</strong></span>')
    parts.append(f'<span class="metric-box">Active PIs: <strong>{pi_feeds["total_pis"]}</strong></span>')

    # Posts by category
    cat_order = ['elite', 'performers', 'conservative', 'active']
    categories = pi_feeds.get('categories', {})
    for cat_key in cat_order:
        cat_data = categories.get(cat_key)
        if not cat_data:
            continue
        parts.append(f'<h3>{_esc(cat_data["label"])}</h3>')
        for post in cat_data.get('posts', []):
            author = post.get('author', 'Unknown')
            copiers = post.get('copiers', 0)
            gain = post.get('gain', 0)
            risk = post.get('risk', 0)
            text = post.get('text', '')
            tickers = post.get('tickers', [])
            created = post.get('created', '')[:10]  # Just date part
            ticker_str = ', '.join(tickers[:6]) if tickers else ''

            parts.append(f'<div class="pi-card">')
            parts.append(f'<strong>@{_esc(author)}</strong> ({copiers:,} copiers, {gain:+.1f}% YTD, Risk {risk})')
            if text:
                # Truncate for display
                display_text = text[:250] + '...' if len(text) > 250 else text
                parts.append(f'<p class="quote">"{_esc(display_text)}"</p>')
            meta_parts = []
            if created:
                meta_parts.append(created)
            if ticker_str:
                meta_parts.append(f'Tickers: {ticker_str}')
            if meta_parts:
                parts.append(f'<span class="small">{" | ".join(meta_parts)}</span>')
            parts.append('</div>')

    # Top mentioned tickers
    top_tickers = pi_feeds.get('top_tickers', [])
    if top_tickers:
        ticker_spans = ' |\n'.join(
            f'<span class="ticker">${t["ticker"]}</span> ({t["count"]} mentions)'
            for t in top_tickers[:8]
        )
        parts.append(f'<div class="highlight"><strong>Top Mentioned Tickers:</strong><br>{ticker_spans}</div>')

    return '\n'.join(parts)


def _section_7_news(llm):
    """Section 7: Market News from LLM analysis."""
    parts = ['<h2><span class="section-num">7</span> Market News</h2>']

    news = llm.get('market_news', []) if llm else []
    portfolio_news = llm.get('portfolio_news', []) if llm else []

    if news:
        parts.append('<h3>General Market News</h3>')
        for item in news:
            css = item.get('css', '')
            headline = item.get('headline', '')
            detail = item.get('detail', '')
            parts.append(f'<div class="news-item">')
            parts.append(f'<strong class="{css}">{_esc(headline)}</strong> {_esc(detail)}')
            parts.append('</div>')

    if portfolio_news:
        parts.append('<h3>Portfolio-Relevant News</h3>')
        parts.append('<ul>')
        for item in portfolio_news:
            ticker = item.get('ticker', '')
            note = item.get('note', '')
            parts.append(f'<li><strong>{ticker}</strong>: {_esc(note)}</li>')
        parts.append('</ul>')

    if not news and not portfolio_news:
        parts.append('<p>Market news analysis not available.</p>')

    return '\n'.join(parts)


def _section_8_actions(data, llm):
    """Section 8: Action Items with contextual rationale."""
    p = data.get('portfolio', {})
    sells = p.get('sells', [])
    buys = p.get('buys', [])
    holds = p.get('holds', [])
    buy_opps = data.get('buy_opportunities', {}).get('top', [])
    rationale = llm.get('action_rationale', {}) if llm else {}
    priority = llm.get('priority_actions', []) if llm else []

    rows = []

    # New BUY opportunities (top 3)
    for opp in buy_opps[:3]:
        reason = rationale.get(opp['ticker'], 'Top opportunity by EXRET')
        rows.append(
            f'<tr><td class="buy">BUY</td><td class="ticker">{opp["ticker"]}</td>'
            f'<td>{opp.get("price", "--")}</td>'
            f'<td>UP {opp["upside"]}, {opp["buy_pct"]} bullish</td>'
            f'<td>{_esc(reason)}</td></tr>'
        )

    # Portfolio ADDs (top 3)
    sorted_buys = sorted(buys, key=lambda x: -(float(x['exret'].replace('%', ''))
                         if x.get('exret') and x['exret'] != '--' else 0))
    for pos in sorted_buys[:3]:
        reason = rationale.get(pos['ticker'], 'Strong BUY signal')
        rows.append(
            f'<tr><td class="add">ADD</td><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>UP {pos["upside"]}, {pos["buy_pct"]} bullish</td>'
            f'<td>{_esc(reason)}</td></tr>'
        )

    # Notable HOLDs (high-upside holds worth mentioning)
    notable_holds = [h for h in holds if safe_float(h['upside']) and safe_float(h['upside']) > 20][:2]
    for pos in notable_holds:
        reason = rationale.get(pos['ticker'], 'Maintain position')
        rows.append(
            f'<tr><td class="hold">HOLD</td><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>UP {pos["upside"]}, {pos["buy_pct"]} bullish</td>'
            f'<td>{_esc(reason)}</td></tr>'
        )

    # SELL actions
    for pos in sells:
        reason = rationale.get(pos['ticker'], 'SELL signal triggered')
        rows.append(
            f'<tr><td class="sell">SELL</td><td class="ticker">{pos["ticker"]}</td>'
            f'<td>{pos.get("price", "--")}</td>'
            f'<td>UP {pos["upside"]}, {pos["buy_pct"]} bullish</td>'
            f'<td>{_esc(reason)}</td></tr>'
        )

    # Priority actions
    priority_html = ''
    if priority:
        items = ''.join(f'<li><strong>{_esc(a)}</strong></li>' for a in priority[:5])
        priority_html = f"""
<div class="warning">
<strong>Priority Actions for Today:</strong>
<ol>{items}</ol>
</div>"""

    return f"""
<h2><span class="section-num">8</span> Action Items</h2>
<table>
<tr><th>Action</th><th>Ticker</th><th>Price</th><th>Metrics</th><th>Rationale</th></tr>
{"".join(rows)}
</table>
{priority_html}"""


def safe_float(value):
    """Parse a value to float, returning None for '--' or invalid values."""
    if value is None or value == '--' or value == '':
        return None
    try:
        return float(str(value).replace('%', '').replace(',', ''))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_briefing(data):
    """Generate the complete HTML briefing with 8 sections."""
    date_str = data['date']
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = dt.strftime('%A, %B %-d, %Y')

    census = data.get('census')
    census_file = census.get('file', 'N/A') if census else 'N/A'
    pi_feeds = data.get('pi_feeds')
    pi_file = pi_feeds.get('file', 'N/A') if pi_feeds else 'N/A'

    # Get LLM analysis
    print("  Calling LLM for contextual analysis...")
    llm = _get_llm_analysis(data)
    if llm:
        print(f"  LLM analysis received ({len(json.dumps(llm))} chars)")
    else:
        print("  LLM analysis not available, using data-only mode")

    # Build all 8 sections
    sections = [
        _section_1_markets(data, llm),
        _section_2_events(llm),
        _section_3_portfolio(data, llm),
        _section_4_signals(data, llm),
        _section_5_census(data, llm),
        _section_6_feeds(data),
        _section_7_news(llm),
        _section_8_actions(data, llm),
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
Signals: {data.get('signals_updated', 'N/A')} | Census: {census_file} | PI Feeds: {pi_file}<br>
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
    if census:
        fg = census.get('fear_greed') or (census.get('top100', {}).get('fear_greed'))
        if fg:
            if fg >= 75:
                hints.append("Extreme Greed")
            elif fg <= 25:
                hints.append("Extreme Fear")

    return " | ".join(hints) if hints else "Daily Analysis"

"""Generates HTML morning briefing via Claude Sonnet on Vertex AI."""

import json
import os

from anthropic import AnthropicVertex


def build_prompt(data):
    """Build the prompt for Claude to generate the HTML briefing."""
    date_str = data['date']

    # Format market data
    market_lines = []
    indices = data.get('market', {}).get('indices', {})
    for symbol, info in indices.items():
        name = info.get('name', symbol)
        price = info.get('price')
        change = info.get('change_pct')
        if price:
            change_str = f"{change:+.2f}%" if change is not None else "N/A"
            market_lines.append(f"  {name}: {price:,.2f} ({change_str})")

    commodities = data.get('market', {}).get('commodities', {})
    commodity_lines = []
    for symbol, info in commodities.items():
        name = info.get('name', symbol)
        price = info.get('price')
        change = info.get('change_pct')
        if price:
            change_str = f"{change:+.2f}%" if change is not None else "N/A"
            commodity_lines.append(f"  {name}: {price:,.2f} ({change_str})")

    # Format portfolio summary
    portfolio = data.get('portfolio', {})
    portfolio_lines = []
    portfolio_lines.append(f"  Total positions: {portfolio.get('count', 0)}")
    portfolio_lines.append(f"  BUY signals: {len(portfolio.get('buys', []))}")
    portfolio_lines.append(f"  SELL signals: {len(portfolio.get('sells', []))}")
    portfolio_lines.append(f"  HOLD signals: {len(portfolio.get('holds', []))}")

    # Portfolio positions detail
    position_lines = []
    for pos in portfolio.get('positions', []):
        signal_map = {'B': 'BUY', 'S': 'SELL', 'H': 'HOLD', 'I': 'INCONCL'}
        sig = signal_map.get(pos['signal'], pos['signal'])
        position_lines.append(
            f"  {pos['ticker']:10s} {pos['name'][:20]:20s} "
            f"Signal={sig:8s} Price={pos['price'] or 'N/A':>10} "
            f"Upside={pos['upside']:>7s} %Buy={pos['buy_pct']:>5s} "
            f"ExRet={pos['exret']:>7s} Cap={pos['cap']}"
        )

    # Portfolio sells (action needed)
    sell_positions = portfolio.get('sells', [])
    sell_lines = []
    for pos in sell_positions:
        sell_lines.append(
            f"  {pos['ticker']:10s} {pos['name'][:20]:20s} "
            f"Upside={pos['upside']:>7s} %Buy={pos['buy_pct']:>5s}"
        )

    # Buy opportunities
    buy_opps = data.get('buy_opportunities', {})
    buy_lines = []
    buy_lines.append(f"  Total BUY signals in market: {buy_opps.get('count', 0)}")
    for opp in buy_opps.get('top', []):
        buy_lines.append(
            f"  {opp['ticker']:10s} {opp['name'][:20]:20s} "
            f"Cap={opp['cap']:>8s} Upside={opp['upside']:>7s} "
            f"%Buy={opp['buy_pct']:>5s} ExRet={opp['exret']:>7s}"
        )

    # Census data
    census = data.get('census')
    census_lines = []
    if census:
        census_lines.append(f"  Fear & Greed Index: {census.get('fear_greed', 'N/A')}")
        avgs = census.get('averages', {})
        census_lines.append(f"  Avg Cash %: {avgs.get('cashPercentage', 'N/A')}%")
        census_lines.append(f"  Avg Gain: {avgs.get('gain', 'N/A')}%")
        census_lines.append(f"  Avg Risk Score: {avgs.get('riskScore', 'N/A')}")
        census_lines.append(f"  Census file: {census.get('file', 'N/A')}")

        census_lines.append("\n  Top PI Holdings:")
        for h in census.get('top_holdings', []):
            census_lines.append(
                f"    {h['symbol']:10s} {h['name'][:20]:20s} "
                f"Holders={h['holders_pct']}% "
                f"Avg Alloc={h['avg_alloc']:.1f}% "
                f"MTD={h['mtd_return']:+.1f}%"
            )

    # PI feeds
    pi_feeds = data.get('pi_feeds')
    feeds_lines = []
    if pi_feeds:
        feeds_lines.append(f"  Total PI posts: {pi_feeds.get('total_posts', 0)}")
        feeds_lines.append(f"  Active PIs posting: {pi_feeds.get('total_pis', 0)}")
        top_tickers = pi_feeds.get('top_tickers', [])
        if top_tickers:
            feeds_lines.append(f"  Most mentioned tickers: {', '.join(str(t) for t in top_tickers[:10])}")
        for cat, info in pi_feeds.get('categories', {}).items():
            feeds_lines.append(f"  {cat}: {info['count']} posts, tickers: {', '.join(info['tickers_mentioned'][:5])}")

    prompt = f"""Generate a complete HTML morning briefing email for {date_str}.

You are a financial analyst assistant creating a daily pre-market briefing for Dimitrios Plessas, an eToro Popular Investor.

IMPORTANT: This is analysis only, NOT investment advice.

== MARKET DATA ==
Indices:
{chr(10).join(market_lines) if market_lines else '  No data available'}

Commodities:
{chr(10).join(commodity_lines) if commodity_lines else '  No data available'}

== PORTFOLIO SUMMARY ==
{chr(10).join(portfolio_lines)}

== PORTFOLIO POSITIONS ==
{chr(10).join(position_lines) if position_lines else '  No positions'}

== PORTFOLIO SELL SIGNALS (Action Needed) ==
{chr(10).join(sell_lines) if sell_lines else '  No sell signals in portfolio'}

== BUY OPPORTUNITIES (Market-wide) ==
{chr(10).join(buy_lines)}

== SELL SIGNALS (Market-wide) ==
  Total market SELL signals: {data.get('sell_signals_count', 0)}

== ETORO CENSUS (Popular Investors) ==
{chr(10).join(census_lines) if census_lines else '  No census data available'}

== PI FEEDS (Social Intelligence) ==
{chr(10).join(feeds_lines) if feeds_lines else '  No feeds data available'}

Signals last updated: {data.get('signals_updated', 'Unknown')}

---

Generate a COMPLETE HTML email with these 8 sections. Use inline CSS for email compatibility.
The HTML must be self-contained (no external stylesheets).

SECTIONS:
1. **Market Overview** — Global indices status, commodities, key moves. Use green/red colors for up/down.
2. **Portfolio Status** — Summary of my positions: how many BUY/SELL/HOLD, any positions needing attention.
3. **Action Items** — SELL signals in my portfolio (immediate attention). List each with reason.
4. **Top Buy Opportunities** — Top 10 from the buy signals list, with key metrics.
5. **Census Intelligence** — Fear & Greed index, cash levels, top PI holdings, what smart money is doing.
6. **Social Pulse** — PI feeds summary: what popular investors are talking about, trending tickers.
7. **Key Metrics Watch** — Notable ROE, P/E, beta, 52-week position across portfolio.
8. **Disclaimer** — Standard disclaimer that this is analysis, not investment advice.

STYLE REQUIREMENTS:
- Professional dark theme: background #1a1a2e, cards #16213e, text #e0e0e0
- Accent colors: green #00d4aa, red #ff6b6b, blue #4ecdc4, gold #ffd93d
- Use tables with borders for data, rounded corners on cards
- Mobile-responsive (max-width: 640px media query)
- Font: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif
- Section headers with colored left border
- Compact but readable — this is a daily email, not a report

Return ONLY the HTML. No markdown fences, no explanations. Start with <!DOCTYPE html>."""

    return prompt


def generate_briefing(data):
    """Call Claude Sonnet on Vertex AI to generate the HTML briefing.

    Requires env vars:
        VERTEX_PROJECT_ID: GCP project ID
        VERTEX_REGION: GCP region (e.g. europe-west1)
        GCP_CREDENTIALS: JSON content of GCP credentials (service account or authorized_user)
    """
    project_id = os.environ.get('VERTEX_PROJECT_ID')
    region = os.environ.get('VERTEX_REGION', 'europe-west1')

    if not project_id:
        raise ValueError("VERTEX_PROJECT_ID environment variable is required")

    # Handle GCP credentials - write JSON to temp file for GOOGLE_APPLICATION_CREDENTIALS
    # Supports both service_account and authorized_user credential types
    creds_json = os.environ.get('GCP_CREDENTIALS', '')
    if creds_json.strip().startswith('{'):
        import tempfile
        key_path = os.path.join(tempfile.gettempdir(), 'gcp_credentials.json')
        with open(key_path, 'w') as f:
            f.write(creds_json)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

    client = AnthropicVertex(project_id=project_id, region=region)

    prompt = build_prompt(data)

    message = client.messages.create(
        model="claude-sonnet-4-6@default",
        max_tokens=8192,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    html = message.content[0].text

    # Clean up if wrapped in markdown fences
    if html.startswith('```'):
        lines = html.split('\n')
        html = '\n'.join(lines[1:-1])

    return html


def extract_subject_hint(data):
    """Generate a subject line hint based on data highlights."""
    hints = []

    # Market move
    indices = data.get('market', {}).get('indices', {})
    sp500 = indices.get('^GSPC', {})
    change = sp500.get('change_pct')
    if change is not None:
        direction = "Up" if change >= 0 else "Down"
        hints.append(f"S&P {direction} {abs(change):.1f}%")

    # VIX
    vix = indices.get('^VIX', {})
    vix_price = vix.get('price')
    if vix_price:
        if vix_price > 25:
            hints.append(f"VIX Elevated {vix_price:.0f}")
        elif vix_price < 15:
            hints.append(f"VIX Low {vix_price:.0f}")

    # Portfolio actions
    portfolio = data.get('portfolio', {})
    sells = portfolio.get('sells', [])
    if sells:
        hints.append(f"{len(sells)} SELL Alert{'s' if len(sells) > 1 else ''}")

    # Census F&G
    census = data.get('census')
    if census and census.get('fear_greed'):
        fg = census['fear_greed']
        if fg >= 75:
            hints.append("Extreme Greed")
        elif fg <= 25:
            hints.append("Extreme Fear")

    return " | ".join(hints) if hints else "Daily Analysis"

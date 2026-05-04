#!/usr/bin/env python3
"""News & Events Analyst - Fetch market news, earnings calendar, and risk events."""

import json
import sys
from datetime import datetime

import yfinance as yf

# Portfolio tickers (top 15 + special holdings)
TICKERS = [
    "NVDA",
    "MSFT",
    "GOOG",
    "AAPL",
    "AMZN",
    "TSM",
    "META",
    "0700.HK",
    "LLY",
    "JPM",
    "V",
    "AMD",
    "MU",
    "PLTR",
    "BAC",
    "NKE",
    "MSTR",
    "BTC-USD",
    "ETH-USD",
]


def fetch_earnings_calendar(tickers):
    """Fetch earnings dates for the next 2 weeks."""
    earnings = []
    today = datetime.now()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar

            if cal is not None and not cal.empty:
                # yfinance returns earnings date in different formats
                if "Earnings Date" in cal.index:
                    earnings_date_raw = cal.loc["Earnings Date"]
                    if hasattr(earnings_date_raw, "iloc"):
                        earnings_date_raw = earnings_date_raw.iloc[0]

                    # Parse the date
                    if isinstance(earnings_date_raw, str):
                        try:
                            earnings_date = datetime.strptime(earnings_date_raw, "%Y-%m-%d")
                        except:
                            continue
                    elif hasattr(earnings_date_raw, "timestamp"):
                        earnings_date = earnings_date_raw
                    else:
                        continue

                    days_away = (earnings_date - today).days

                    if 0 <= days_away <= 14:
                        # Get consensus EPS if available
                        consensus_eps = None
                        info = stock.info
                        if info and "forwardEps" in info:
                            consensus_eps = info.get("forwardEps")

                        risk_level = (
                            "HIGH" if days_away <= 3 else "MEDIUM" if days_away <= 7 else "LOW"
                        )

                        earnings.append(
                            {
                                "ticker": ticker,
                                "expected_date": earnings_date.strftime("%Y-%m-%d"),
                                "consensus_eps": consensus_eps,
                                "days_away": days_away,
                                "risk_level": risk_level,
                            }
                        )
        except Exception as e:
            print(f"Error fetching earnings for {ticker}: {e}", file=sys.stderr)
            continue

    return sorted(earnings, key=lambda x: x["days_away"])


def fetch_stock_news(tickers):
    """Fetch recent news for portfolio stocks."""
    portfolio_news = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if news and len(news) > 0:
                # Get top 3 most recent news items
                ticker_news = []
                for item in news[:3]:
                    # Determine impact based on keywords
                    title = item.get("title", "").lower()
                    impact = "NEUTRAL"

                    if any(
                        word in title for word in ["upgrade", "beat", "surge", "rally", "strong"]
                    ):
                        impact = "HIGH_POSITIVE"
                    elif any(
                        word in title for word in ["downgrade", "miss", "fall", "drop", "weak"]
                    ):
                        impact = "HIGH_NEGATIVE"

                    ticker_news.append(
                        {
                            "headline": item.get("title", ""),
                            "impact": impact,
                            "source": item.get("publisher", "Unknown"),
                        }
                    )

                if ticker_news:
                    portfolio_news[ticker] = ticker_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}", file=sys.stderr)
            continue

    return portfolio_news


def generate_report():
    """Generate the full news & events report."""

    print("Fetching earnings calendar...", file=sys.stderr)
    earnings_calendar = fetch_earnings_calendar(TICKERS)

    print("Fetching portfolio news...", file=sys.stderr)
    portfolio_news = fetch_stock_news(TICKERS)

    # CIO v36 / M1: ALL hardcoded sections removed. economic_events,
    # breaking_news, regulatory_risks, analyst_waves are now empty until
    # this script is wired to a real source (news-reader MCP or WebSearch).
    # The synthesis pipeline (committee_synthesis._normalize_breaking_news)
    # detects placeholder data via the news_data_status field and skips
    # news-derived modifiers when the data is fabricated.
    economic_events: list = []
    breaking_news: list = []
    regulatory_risks: list = []
    analyst_waves: list = []

    # Event risk summary
    len(earnings_calendar)
    highest_risk = earnings_calendar[0]["ticker"] if earnings_calendar else None

    # Determine overall sentiment
    positive_count = sum(
        1
        for ticker_news in portfolio_news.values()
        for item in ticker_news
        if "POSITIVE" in item["impact"]
    )
    negative_count = sum(
        1
        for ticker_news in portfolio_news.values()
        for item in ticker_news
        if "NEGATIVE" in item["impact"]
    )

    if positive_count > negative_count * 1.5:
        sentiment = "POSITIVE"
    elif negative_count > positive_count * 1.5:
        sentiment = "NEGATIVE"
    else:
        sentiment = "MIXED"

    # Generate 6 key catalysts section for HTML renderer
    key_catalysts = []

    # Add earnings events
    for earning in earnings_calendar[:3]:
        key_catalysts.append(
            {
                "title": f"{earning['ticker']} Earnings",
                "date": earning["expected_date"],
                "importance": earning["risk_level"],
                "detail": f"{earning['days_away']} days away",
            }
        )

    # Add economic events
    for event in economic_events[:2]:
        if event["importance"] == "HIGH":
            key_catalysts.append(
                {
                    "title": event["event"],
                    "date": event["date"],
                    "importance": event["importance"],
                    "detail": event["sector_impact"],
                }
            )

    # Add regulatory risks
    if regulatory_risks:
        key_catalysts.append(
            {
                "title": f"{regulatory_risks[0]['ticker']} Regulatory Risk",
                "date": "Ongoing",
                "importance": regulatory_risks[0]["severity"],
                "detail": regulatory_risks[0]["risk"],
            }
        )

    # CIO v36 / M1: hardcoded breaking_news/regulatory_risks/economic_events
    # were removed; remaining content (portfolio_news from yfinance,
    # earnings_calendar) is real. Set OK so synthesis processes the real
    # parts; empty arrays mean those modifiers fire 0 instead of fake values.
    data_status = "OK"

    report = {
        "analyst": "news",
        "timestamp": datetime.now().isoformat(),
        "data_status": data_status,
        "breaking_news": breaking_news,
        "portfolio_news": portfolio_news,
        "earnings_calendar": {"next_2_weeks": earnings_calendar},
        "economic_events": economic_events,
        "analyst_waves": analyst_waves,
        "regulatory_risks": regulatory_risks,
        "event_risk_summary": {
            "highest_risk_ticker": highest_risk,
            "reason": f"Earnings in {earnings_calendar[0]['days_away']} days"
            if highest_risk
            else "No imminent earnings",
            "overall_news_sentiment": sentiment,
        },
        "sections": {"6_key_catalysts_this_week": {"items": key_catalysts[:6]}},
    }

    return report


if __name__ == "__main__":
    report = generate_report()

    # Write to file
    output_path = "/Users/plessas/.weirdapps-trading/committee/reports/news.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {output_path}", file=sys.stderr)
    print(json.dumps(report, indent=2))

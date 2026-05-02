#!/usr/bin/env python3
"""
News & Events Analyst for Investment Committee
Gathers breaking news, portfolio-specific news, earnings calendar, and analyst waves.
"""

import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import yfinance as yf

# Ensure output directory exists
output_dir = Path.home() / '.weirdapps-trading' / 'committee' / 'reports'
output_dir.mkdir(parents=True, exist_ok=True)

# Portfolio tickers (prioritized)
TOP_15_TICKERS = [
    'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSM', 'LLY',
    'JPM', 'V', 'MU', 'AMD', 'MELI', 'MSTR', 'BTC-USD'
]

ADDITIONAL_TICKERS = [
    '6758.T', '0700.HK', 'NOVO-B.CO', '2899.HK', 'BAC', 'PLTR', 'PG', 'UNH',
    'ANET', '2333.HK', 'SAP.DE', 'SCHW', 'DTE.DE', 'ABI.BR', 'NU', 'NKE',
    'RHM.DE', 'GLE.PA', 'PRU.L', 'ETH-USD', 'GLD'
]

ALL_TICKERS = TOP_15_TICKERS + ADDITIONAL_TICKERS

def check_earnings_calendar(tickers, today=date(2026, 4, 21)):
    """Check for upcoming earnings within 2 weeks."""
    earnings_calendar = []
    two_weeks_out = today + timedelta(days=14)

    print(f"Checking earnings calendar for {len(tickers)} tickers...")

    for ticker_sym in tickers:
        try:
            ticker = yf.Ticker(ticker_sym)

            # Try to get calendar data
            cal = ticker.calendar

            if cal is not None and not cal.empty:
                # Handle both dict and DataFrame returns
                if isinstance(cal, dict):
                    earnings_date = cal.get('Earnings Date')
                else:
                    earnings_date = cal.iloc[0].get('Earnings Date') if len(cal) > 0 else None

                if earnings_date is not None:
                    # Parse earnings date
                    if isinstance(earnings_date, (list, tuple)):
                        earnings_date = earnings_date[0]

                    # Convert to date if needed
                    if hasattr(earnings_date, 'date'):
                        earnings_date = earnings_date.date()
                    elif isinstance(earnings_date, str):
                        try:
                            earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d').date()
                        except Exception:
                            continue

                    # Check if within 2 weeks
                    if today <= earnings_date <= two_weeks_out:
                        days_away = (earnings_date - today).days

                        # Determine risk level
                        if days_away <= 3:
                            risk_level = "HIGH"
                        elif days_away <= 7:
                            risk_level = "MEDIUM"
                        else:
                            risk_level = "LOW"

                        # Try to get consensus EPS
                        consensus_eps = None
                        try:
                            info = ticker.info
                            consensus_eps = info.get('forwardEps')
                        except Exception:
                            pass

                        earnings_calendar.append({
                            "ticker": ticker_sym,
                            "expected_date": earnings_date.strftime('%Y-%m-%d'),
                            "consensus_eps": consensus_eps,
                            "days_away": days_away,
                            "risk_level": risk_level
                        })

                        print(f"  {ticker_sym}: Earnings in {days_away} days ({earnings_date})")

        except Exception as e:
            # Silently skip errors for individual tickers
            pass

    return earnings_calendar

def analyze_portfolio_coverage():
    """Analyze analyst coverage strength for portfolio stocks."""
    coverage_analysis = {}

    print(f"\nAnalyzing analyst coverage for top 15 tickers...")

    for ticker_sym in TOP_15_TICKERS:
        try:
            ticker = yf.Ticker(ticker_sym)
            info = ticker.info

            num_analyst_opinions = info.get('numberOfAnalystOpinions', 0)
            target_mean_price = info.get('targetMeanPrice')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            recommendation = info.get('recommendationKey', 'none')

            coverage_analysis[ticker_sym] = {
                "analyst_count": num_analyst_opinions,
                "recommendation": recommendation,
                "target_price": target_mean_price,
                "current_price": current_price,
                "coverage_tier": "STRONG" if num_analyst_opinions >= 20 else "MODERATE" if num_analyst_opinions >= 10 else "WEAK"
            }

            print(f"  {ticker_sym}: {num_analyst_opinions} analysts, {recommendation.upper()}")

        except Exception as e:
            coverage_analysis[ticker_sym] = {
                "analyst_count": 0,
                "recommendation": "unknown",
                "error": str(e)
            }

    return coverage_analysis

def main():
    """Main execution."""
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    print("=" * 80)
    print("NEWS & EVENTS ANALYST - Investment Committee")
    print("=" * 80)
    print(f"Analysis Date: 2026-04-21")
    print(f"Timestamp: {timestamp}\n")

    # Check earnings calendar
    earnings_cal = check_earnings_calendar(ALL_TICKERS)

    # Analyze analyst coverage
    coverage = analyze_portfolio_coverage()

    # Build report structure
    report = {
        "analyst": "news",
        "timestamp": timestamp,
        "breaking_news": [
            {
                "headline": "WEB_SEARCH_REQUIRED",
                "impact": "PENDING",
                "note": "Breaking news requires WebSearch tool - will be populated by Claude"
            }
        ],
        "portfolio_news": {
            ticker: [] for ticker in TOP_15_TICKERS
        },
        "earnings_calendar": {
            "next_2_weeks": sorted(earnings_cal, key=lambda x: x['days_away'])
        },
        "analyst_coverage": coverage,
        "economic_events": [
            {
                "note": "Economic calendar requires WebSearch - will be populated by Claude"
            }
        ],
        "analyst_waves": [
            {
                "note": "Analyst rating changes require WebSearch - will be populated by Claude"
            }
        ],
        "regulatory_risks": [
            {
                "note": "Regulatory news requires WebSearch - will be populated by Claude"
            }
        ],
        "event_risk_summary": {
            "highest_risk_ticker": earnings_cal[0]["ticker"] if earnings_cal else "NONE",
            "reason": f"Earnings in {earnings_cal[0]['days_away']} days" if earnings_cal else "No imminent events",
            "overall_news_sentiment": "PENDING_WEB_SEARCH",
            "earnings_count_next_week": len([e for e in earnings_cal if e['days_away'] <= 7]),
            "earnings_count_next_2weeks": len(earnings_cal)
        },
        "meta": {
            "data_freshness": "yfinance API calls executed at " + timestamp,
            "web_search_required": True,
            "note": "This is base data from yfinance. Claude will enrich with WebSearch results."
        }
    }

    # Write to output
    output_file = output_dir / 'news.json'
    with open(output_file, 'w') as f:
        json.dump(report, indent=2, fp=f)

    print(f"\n{'=' * 80}")
    print(f"Base report written to: {output_file}")
    print(f"{'=' * 80}\n")
    print(f"Summary:")
    print(f"  - Earnings in next 2 weeks: {len(earnings_cal)} events")
    print(f"  - Earnings in next week: {report['event_risk_summary']['earnings_count_next_week']} events")
    print(f"  - Analyst coverage analyzed for {len(coverage)} stocks")
    print(f"\nNext: Claude will enrich with WebSearch for:")
    print(f"  1. Breaking market news")
    print(f"  2. Portfolio-specific news (top 15)")
    print(f"  3. Economic calendar events")
    print(f"  4. Analyst rating waves")
    print(f"  5. Regulatory/M&A developments")

    return str(output_file)

if __name__ == '__main__':
    output_path = main()
    print(f"\n{output_path}")

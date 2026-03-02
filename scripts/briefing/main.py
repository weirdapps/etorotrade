#!/usr/bin/env python3
"""Morning briefing orchestrator.

Usage:
    python scripts/briefing/main.py                    # Run with today's date
    python scripts/briefing/main.py --date 2026-03-01  # Run with specific date
    python scripts/briefing/main.py --no-email          # Generate HTML only, skip email
"""

import argparse
import os
import sys
from datetime import datetime

from data_collector import collect_data
from briefing_generator import generate_briefing, extract_subject_hint
from email_sender import send_email


def main():
    parser = argparse.ArgumentParser(description='Generate and send morning briefing')
    parser.add_argument(
        '--date', type=str, default=None,
        help='Date for briefing (YYYY-MM-DD). Defaults to today UTC.'
    )
    parser.add_argument(
        '--no-email', action='store_true',
        help='Skip sending email, just generate HTML file.'
    )
    parser.add_argument(
        '--base-dir', type=str, default=None,
        help='Path to etorotrade repo root. Defaults to auto-detect.'
    )
    parser.add_argument(
        '--census-dir', type=str, default=None,
        help='Path to etoro_census repo root. Defaults to ./etoro_census'
    )
    parser.add_argument(
        '--recipient', type=str, default=None,
        help='Override email recipient.'
    )
    args = parser.parse_args()

    # Determine base directory
    base_dir = args.base_dir
    if base_dir is None:
        # Auto-detect: this script is at scripts/briefing/main.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Census directory
    census_dir = args.census_dir
    if census_dir is None:
        # In GitHub Actions, census is checked out into ./etoro_census
        census_dir = os.path.join(base_dir, 'etoro_census')
        if not os.path.isdir(census_dir):
            # Local dev: try the sibling directory
            census_dir = os.path.abspath(
                os.path.join(base_dir, '..', 'etoro_census')
            )

    date_str = args.date or datetime.utcnow().strftime('%Y-%m-%d')

    print(f"=== Morning Briefing Generator ===")
    print(f"Date: {date_str}")
    print(f"Base dir: {base_dir}")
    print(f"Census dir: {census_dir}")
    print()

    # Step 1: Collect data
    print("Step 1/3: Collecting data...")
    try:
        data = collect_data(base_dir, census_dir, date_str)
    except Exception as e:
        print(f"ERROR: Data collection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Portfolio: {data['portfolio']['count']} positions")
    print(f"  Buy opportunities: {data['buy_opportunities']['count']}")
    print(f"  Sell signals: {data['sell_signals_count']}")
    print(f"  Census: {'loaded' if data['census'] else 'not available'}")
    print(f"  Market data: {len(data['market'].get('indices', {}))} indices")
    print()

    # Step 2: Generate briefing
    print("Step 2/3: Generating briefing via Claude on Vertex AI...")
    try:
        html = generate_briefing(data)
    except Exception as e:
        print(f"ERROR: Briefing generation failed: {e}", file=sys.stderr)
        sys.exit(2)

    if not html or len(html) < 100:
        print("ERROR: Generated HTML is too short, likely failed", file=sys.stderr)
        sys.exit(2)

    print(f"  Generated {len(html)} chars of HTML")
    print()

    # Save HTML to output directory
    output_dir = os.path.join(base_dir, 'yahoofinance', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'briefing-{date_str}.html')
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Saved to {output_path}")

    # Step 3: Send email
    if args.no_email:
        print("Step 3/3: Skipping email (--no-email flag)")
    else:
        print("Step 3/3: Sending email...")
        subject_hint = extract_subject_hint(data)
        # Format date nicely
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        formatted_date = dt.strftime('%b %-d, %Y')
        subject = f"Morning Briefing - {formatted_date} | {subject_hint}"

        try:
            send_email(html, subject, recipient=args.recipient)
        except Exception as e:
            print(f"ERROR: Email sending failed: {e}", file=sys.stderr)
            sys.exit(3)

    print()
    print("=== Briefing complete ===")


if __name__ == '__main__':
    main()

"""
Data Freshness / Staleness Scoring

Tracks when key metrics last changed for each ticker by analyzing
signal_log.jsonl history. Classifies data as fresh, aging, or stale,
and applies confidence penalties to stale signals.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from trade_modules.backtest_engine import SIGNAL_LOG_PATH, TEST_TICKER_RE, VALID_SIGNALS

logger = logging.getLogger(__name__)

# Staleness thresholds (days since last metric change)
# CIO Review Finding M5: Original penalties were too gentle.
# 90-day stale data at 25% penalty still produced actionable signals — unacceptable.
FRESH_THRESHOLD = 30
AGING_THRESHOLD = 60
STALE_THRESHOLD = 90  # Beyond this, signal becomes INCONCLUSIVE

# Confidence penalties — stiffened per CIO review
PENALTIES = {
    'fresh': 0.0,       # < 30 days: no penalty
    'aging': 0.25,      # 30-60 days: 25% penalty (was 10%)
    'stale': 0.50,      # 60-90 days: 50% penalty (was 25%)
    'dead': 1.0,        # 90+ days: signal is INCONCLUSIVE (new tier)
}

# Metrics to track for changes
TRACKED_METRICS = ['buy_percentage', 'upside', 'exret']

# Minimum change to count as a real change (not noise)
CHANGE_THRESHOLDS = {
    'buy_percentage': 2.0,   # 2 percentage points
    'upside': 1.0,           # 1 percentage point
    'exret': 0.5,            # 0.5 points
}


class DataFreshnessTracker:
    """
    Tracks data freshness by detecting when key metrics last changed
    for each ticker in the signal log.
    """

    def __init__(self, signal_log_path: Optional[Path] = None):
        self.signal_log_path = signal_log_path or SIGNAL_LOG_PATH

    def check_freshness(
        self, tickers: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check data freshness for specified tickers (or all if None).

        Returns:
            Dict mapping ticker to freshness info:
            {
                ticker: {
                    'days_since_change': int,
                    'staleness': 'fresh' | 'aging' | 'stale',
                    'confidence_penalty': float,
                    'last_change_date': str,
                    'metrics_changed': list of metric names that changed last,
                    'total_observations': int,
                }
            }
        """
        history = self._load_metric_history()
        if not history:
            return {}

        today = datetime.now().date()
        results = {}

        ticker_set = set(tickers) if tickers else None

        for ticker, records in history.items():
            if ticker_set and ticker not in ticker_set:
                continue

            freshness = self._analyze_ticker_freshness(ticker, records, today)
            if freshness:
                results[ticker] = freshness

        return results

    def get_stale_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Convenience method returning stale and dead tickers."""
        all_freshness = self.check_freshness()
        return {
            ticker: info
            for ticker, info in all_freshness.items()
            if info['staleness'] in ('stale', 'dead')
        }

    def _load_metric_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load signal log and group records by ticker with tracked metrics.

        Returns:
            Dict mapping ticker to list of {date, metrics...} records,
            sorted by date ascending.
        """
        if not self.signal_log_path.exists():
            logger.warning(f"Signal log not found: {self.signal_log_path}")
            return {}

        ticker_records: Dict[str, List[Dict[str, Any]]] = {}

        with open(self.signal_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ticker = data.get('ticker', '')
                signal = data.get('signal', '')

                if TEST_TICKER_RE.match(ticker):
                    continue
                if signal not in VALID_SIGNALS:
                    continue

                record = {
                    'date': data.get('timestamp', '')[:10],  # YYYY-MM-DD
                }
                for metric in TRACKED_METRICS:
                    val = data.get(metric)
                    if val is not None:
                        try:
                            record[metric] = float(val)
                        except (ValueError, TypeError):
                            pass

                if ticker not in ticker_records:
                    ticker_records[ticker] = []
                ticker_records[ticker].append(record)

        # Sort each ticker's records by date
        for ticker in ticker_records:
            ticker_records[ticker].sort(key=lambda r: r['date'])
            # Deduplicate by date (keep last per day)
            seen_dates: Dict[str, Dict[str, Any]] = {}
            for rec in ticker_records[ticker]:
                seen_dates[rec['date']] = rec
            ticker_records[ticker] = list(seen_dates.values())

        return ticker_records

    def _analyze_ticker_freshness(
        self,
        ticker: str,
        records: List[Dict[str, Any]],
        today,
    ) -> Optional[Dict[str, Any]]:
        """Analyze when metrics last changed for a single ticker."""
        if len(records) < 2:
            # Can't determine change with fewer than 2 observations
            if records:
                return {
                    'days_since_change': (today - datetime.strptime(records[0]['date'], '%Y-%m-%d').date()).days,
                    'staleness': 'stale',
                    'confidence_penalty': PENALTIES['stale'],
                    'last_change_date': records[0]['date'],
                    'metrics_changed': [],
                    'total_observations': len(records),
                }
            return None

        # Walk backwards to find the most recent date where any metric changed
        last_change_date = records[0]['date']  # Default to first observation
        metrics_changed = []

        for i in range(len(records) - 1, 0, -1):
            current = records[i]
            previous = records[i - 1]
            changed = []

            for metric in TRACKED_METRICS:
                cur_val = current.get(metric)
                prev_val = previous.get(metric)
                if cur_val is None or prev_val is None:
                    continue
                threshold = CHANGE_THRESHOLDS.get(metric, 1.0)
                if abs(cur_val - prev_val) >= threshold:
                    changed.append(metric)

            if changed:
                last_change_date = current['date']
                metrics_changed = changed
                break

        # Calculate days since change
        change_date = datetime.strptime(last_change_date, '%Y-%m-%d').date()
        days_since = (today - change_date).days

        # Classify staleness — CIO Review M5: stiffened thresholds
        if days_since < FRESH_THRESHOLD:
            staleness = 'fresh'
        elif days_since < AGING_THRESHOLD:
            staleness = 'aging'
        elif days_since < STALE_THRESHOLD:
            staleness = 'stale'
        else:
            staleness = 'dead'  # 90+ days: effectively INCONCLUSIVE

        return {
            'days_since_change': days_since,
            'staleness': staleness,
            'confidence_penalty': PENALTIES[staleness],
            'last_change_date': last_change_date,
            'metrics_changed': metrics_changed,
            'total_observations': len(records),
            'is_inconclusive': staleness == 'dead',  # Signal should be marked I
        }

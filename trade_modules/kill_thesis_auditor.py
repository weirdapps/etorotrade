"""
Kill Thesis Auditor (CIO v17 op #2)

When `check_kill_theses()` reports many triggered theses, we need to
distinguish:
  * TRUE positive — the thesis genuinely broke; trigger was justified.
  * FALSE positive — the trigger fired prematurely or on noise.
  * UNVERIFIED  — too soon to tell (need T+30 evidence).

The audit walks every triggered thesis, fetches realized return since
the trigger date, and classifies. Output is consumed by the next
/committee run's adversarial-debate prompt to either:
  * Reinforce the trigger pattern (TRUE positive → keep penalising)
  * Soften the trigger condition (FALSE positive → loosen the rule)
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_KILL_THESES_PATH = Path.home() / ".weirdapps-trading" / "committee" / "kill_thesis_log.json"
DEFAULT_AUDIT_PATH = Path.home() / ".weirdapps-trading" / "committee" / "kill_thesis_audit.json"

# A trigger is TRUE positive if the position dropped >= TRUE_DROP_PCT in
# the next 30 calendar days. FALSE positive if it ROSE >= FALSE_RISE_PCT.
TRUE_DROP_PCT = -8.0
FALSE_RISE_PCT = 8.0


@dataclass
class AuditedThesis:
    ticker: str
    thesis: str
    trigger_date: str
    days_since_trigger: int
    price_at_trigger: float | None = None
    price_now: float | None = None
    return_since_trigger: float | None = None
    classification: str = "UNVERIFIED"
    reason: str = ""


def _try_get_price(ticker: str, date_str: str, prices: dict[str, Any]) -> float | None:
    """Look up a price from a {ticker:{date:price}} dict, with day fallback."""
    series = prices.get(ticker, {})
    if not series:
        return None
    if date_str in series:
        return series[date_str]
    # Try same week
    for offset in range(1, 5):
        try:
            d = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=offset)).strftime(
                "%Y-%m-%d"
            )
            if d in series:
                return series[d]
            d = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=offset)).strftime(
                "%Y-%m-%d"
            )
            if d in series:
                return series[d]
        except ValueError:
            return None
    return None


def audit_triggered_theses(
    kill_theses_path: Path | None = None,
    output_path: Path | None = None,
    use_price_cache: bool = True,
) -> dict[str, Any]:
    """
    Audit every triggered kill thesis.

    Returns:
      {
        "generated_at": "...",
        "total_audited": int,
        "true_positives": [...],
        "false_positives": [...],
        "unverified": [...],
        "summary_by_pattern": {...}
      }
    """
    kt_path = kill_theses_path or DEFAULT_KILL_THESES_PATH
    out_path = output_path or DEFAULT_AUDIT_PATH

    if not kt_path.exists():
        return {"status": "no_kill_theses_file"}

    try:
        data = json.load(open(kt_path))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "file_unreadable", "error": str(exc)}

    # The kill_thesis_log.json is a list of {ticker, kill_thesis,
    # committee_date, status, ...} where status ∈ {active, triggered,
    # expired}. Older audit-test files use {triggered_theses: [...]}
    # — accept both shapes for forward + backward compatibility.
    if isinstance(data, dict) and "triggered_theses" in data:
        triggered = data.get("triggered_theses", [])
    elif isinstance(data, list):
        triggered = [t for t in data if isinstance(t, dict) and t.get("status") == "triggered"]
    else:
        triggered = []
    if not triggered:
        return {"status": "no_triggered_theses"}

    # Build price lookup from price_cache for the unique trigger tickers.
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")

    prices: dict[str, dict[str, float]] = {}
    if use_price_cache:
        try:
            from trade_modules.price_cache import load_prices

            unique = list({t.get("ticker", "") for t in triggered if t.get("ticker")})
            cache = load_prices(unique)
            for tkr, df in cache.items():
                if df is not None and not df.empty:
                    prices[tkr] = {d.strftime("%Y-%m-%d"): float(p) for d, p in df["Close"].items()}
        except Exception as exc:
            logger.debug("price cache unavailable, continuing without prices: %s", exc)

    audits: list[AuditedThesis] = []
    for t in triggered:
        if not isinstance(t, dict):
            continue
        ticker = t.get("ticker", "")
        if not ticker:
            continue
        thesis_text = t.get("thesis") or t.get("kill_thesis") or ""
        # Trigger date may be the committee_date when the thesis was logged
        # or a separate trigger_detected_date. Fall back to today on missing.
        trigger_date_str = (
            t.get("trigger_date") or t.get("triggered_at") or t.get("committee_date") or today_str
        )
        try:
            trigger_dt = datetime.strptime(trigger_date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            trigger_dt = today

        days_since = (today - trigger_dt).days
        price_trigger = _try_get_price(ticker, trigger_date_str, prices)
        price_now = None
        if prices.get(ticker):
            # Latest available bar
            series = prices[ticker]
            sorted_dates = sorted(series.keys())
            if sorted_dates:
                price_now = series[sorted_dates[-1]]

        ret_pct = None
        if price_trigger and price_now and price_trigger > 0:
            ret_pct = round((price_now - price_trigger) / price_trigger * 100, 2)

        # Classify
        if ret_pct is None or days_since < 7:
            classification = "UNVERIFIED"
            reason = f"insufficient evidence (days_since={days_since}, return={ret_pct})"
        elif ret_pct <= TRUE_DROP_PCT:
            classification = "TRUE_POSITIVE"
            reason = f"position dropped {ret_pct}% since trigger — thesis validated"
        elif ret_pct >= FALSE_RISE_PCT:
            classification = "FALSE_POSITIVE"
            reason = f"position ROSE {ret_pct}% since trigger — thesis was wrong"
        else:
            classification = "INCONCLUSIVE"
            reason = f"position moved {ret_pct}% — within noise band"

        audits.append(
            AuditedThesis(
                ticker=ticker,
                thesis=thesis_text[:200],
                trigger_date=trigger_date_str,
                days_since_trigger=days_since,
                price_at_trigger=price_trigger,
                price_now=price_now,
                return_since_trigger=ret_pct,
                classification=classification,
                reason=reason,
            )
        )

    # Group by classification.
    true_pos = [a for a in audits if a.classification == "TRUE_POSITIVE"]
    false_pos = [a for a in audits if a.classification == "FALSE_POSITIVE"]
    inconclusive = [a for a in audits if a.classification == "INCONCLUSIVE"]
    unverified = [a for a in audits if a.classification == "UNVERIFIED"]

    # Pattern signature — extract the leading words of the thesis (first 5
    # tokens) as a rough fingerprint to find recurring patterns.
    def _signature(thesis: str) -> str:
        return " ".join(thesis.lower().split()[:5]) or "(empty)"

    by_pattern: dict[str, dict[str, int]] = {}
    for a in audits:
        sig = _signature(a.thesis)
        b = by_pattern.setdefault(
            sig, {"count": 0, "true": 0, "false": 0, "inconclusive": 0, "unverified": 0}
        )
        b["count"] += 1
        if a.classification == "TRUE_POSITIVE":
            b["true"] += 1
        elif a.classification == "FALSE_POSITIVE":
            b["false"] += 1
        elif a.classification == "INCONCLUSIVE":
            b["inconclusive"] += 1
        else:
            b["unverified"] += 1

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_audited": len(audits),
        "true_positives": [asdict(a) for a in true_pos],
        "false_positives": [asdict(a) for a in false_pos],
        "inconclusive": [asdict(a) for a in inconclusive],
        "unverified": [asdict(a) for a in unverified],
        "summary": {
            "true_positive_count": len(true_pos),
            "false_positive_count": len(false_pos),
            "inconclusive_count": len(inconclusive),
            "unverified_count": len(unverified),
            "true_positive_rate": (
                round(len(true_pos) / max(len(true_pos) + len(false_pos), 1), 3)
            ),
        },
        "by_pattern": dict(sorted(by_pattern.items(), key=lambda x: -x[1]["count"])[:20]),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    return output

"""
Post-Mortem Workflow (CIO v17 op #6)

When a recommended ADD/BUY drops ≥10% within 30 days of the
recommendation, automatically generate a structured post-mortem:

  * Which agents endorsed the position?
  * Which agents dissented (and what did they say)?
  * Was a kill thesis triggered? If so, when?
  * Which waterfall modifiers were the loudest?
  * Pattern match: any similar past failures?

The output appends to a "lessons learned" log that the next /committee
run reads as part of agent_memory. This is the institutional memory
the system was missing.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LESSONS_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "lessons_learned.jsonl"
)
DEFAULT_HISTORY_DIR = (
    Path.home() / ".weirdapps-trading" / "committee" / "history"
)

# A recommended ADD/BUY that drops by this much within 30d triggers an audit.
DRAWDOWN_TRIGGER_PCT = -10.0


@dataclass
class PostMortem:
    ticker: str
    recommendation_date: str
    action: str
    conviction: int
    entry_price: float
    drawdown_date: str
    drawdown_pct: float
    days_to_drawdown: int
    endorsing_agents: List[str] = field(default_factory=list)
    dissenting_agents: List[str] = field(default_factory=list)
    waterfall_top: List[Dict[str, Any]] = field(default_factory=list)
    kill_thesis_text: str = ""
    kill_thesis_triggered: bool = False
    days_to_kill_trigger: Optional[int] = None
    similar_failures: List[str] = field(default_factory=list)
    lesson: str = ""


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _agent_views_to_lists(stock: Dict[str, Any]) -> tuple:
    """Split agent views into endorsing/dissenting for an ADD/BUY action."""
    endorsing = []
    dissenting = []
    fund_view = stock.get("fund_view", "")
    tech = stock.get("tech_signal", "")
    macro = stock.get("macro_fit", "")
    census = stock.get("census", "")
    news = stock.get("news_impact", "")
    risk_warn = stock.get("risk_warning", False)

    if fund_view == "BUY":
        endorsing.append(f"Fundamental ({stock.get('fund_score', '?')})")
    elif fund_view == "SELL":
        dissenting.append("Fundamental (SELL)")
    if tech in ("ENTER_NOW", "WAIT_FOR_PULLBACK", "ACCUMULATE"):
        endorsing.append(f"Technical ({tech})")
    elif tech in ("AVOID", "EXIT_SOON"):
        dissenting.append(f"Technical ({tech})")
    if macro == "FAVORABLE":
        endorsing.append("Macro (FAVORABLE)")
    elif macro == "UNFAVORABLE":
        dissenting.append("Macro (UNFAVORABLE)")
    if census == "ALIGNED":
        endorsing.append("Census (ALIGNED)")
    elif census in ("DIVERGENT", "CENSUS_DIV"):
        dissenting.append(f"Census ({census})")
    if "POSITIVE" in str(news):
        endorsing.append("News (POSITIVE)")
    elif "NEGATIVE" in str(news):
        dissenting.append("News (NEGATIVE)")
    if risk_warn:
        dissenting.append("Risk (WARN)")
    return endorsing, dissenting


def _waterfall_top_n(stock: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
    """Top-N waterfall keys by abs(value)."""
    wf = stock.get("conviction_waterfall") or {}
    pairs = []
    for k, v in wf.items():
        if k.startswith("_"):
            continue
        try:
            pairs.append({"key": k, "value": int(v)})
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda r: abs(r["value"]), reverse=True)
    return pairs[:n]


def detect_post_mortems(
    history_dir: Optional[Path] = None,
    price_data: Optional[Dict[str, Dict[str, float]]] = None,
    drawdown_threshold: float = DRAWDOWN_TRIGGER_PCT,
    horizon_days: int = 30,
    use_price_cache: bool = True,
) -> List[PostMortem]:
    """
    Scan history for ADD/BUY recommendations whose forward return at
    any point in the next `horizon_days` calendar days fell to
    `drawdown_threshold` or worse.

    Args:
        history_dir: where to read concordance archives.
        price_data: optional pre-loaded {ticker:{date:price}} map. When
            omitted, loads from the price_cache module.
        drawdown_threshold: % loss that triggers a post-mortem.
        horizon_days: window after recommendation to inspect.
        use_price_cache: pull from price_cache when price_data not given.

    Returns the list of PostMortem objects.
    """
    history_dir = history_dir or DEFAULT_HISTORY_DIR
    if not history_dir.is_dir():
        return []

    # Collect candidate (ticker, date, conviction, entry_price, stock_row)
    candidates = []
    for fpath in sorted(history_dir.glob("concordance-*.json")):
        try:
            data = json.load(open(fpath))
        except (OSError, json.JSONDecodeError):
            continue
        date_str = None
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})", fpath.stem)
        if m:
            date_str = m.group(1)
        if not date_str:
            continue
        items = data.get("concordance", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        for stock in items:
            if not isinstance(stock, dict):
                continue
            if stock.get("action") not in ("ADD", "BUY"):
                continue
            entry_px = _safe_float(stock.get("price"))
            if entry_px is None or entry_px <= 0:
                continue
            candidates.append({
                "ticker": stock.get("ticker", ""),
                "date": date_str,
                "stock": stock,
                "entry_price": entry_px,
            })

    if not candidates:
        return []

    # Resolve price data from cache (best effort)
    if price_data is None and use_price_cache:
        try:
            from trade_modules.price_cache import load_prices
            unique = list({c["ticker"] for c in candidates})
            cache = load_prices(unique)
            price_data = {}
            for tkr, df in cache.items():
                if df is not None and not df.empty:
                    price_data[tkr] = {
                        d.strftime("%Y-%m-%d"): float(p)
                        for d, p in df["Close"].items()
                    }
        except Exception as exc:
            logger.debug("price cache unavailable: %s", exc)
            price_data = {}

    if not price_data:
        return []

    post_mortems: List[PostMortem] = []
    for c in candidates:
        tkr = c["ticker"]
        if tkr not in price_data:
            continue
        rec_date = datetime.strptime(c["date"], "%Y-%m-%d").date()
        entry_px = c["entry_price"]
        series = price_data[tkr]

        # Walk forward up to horizon_days, looking for the worst drawdown
        worst_pct = 0.0
        worst_date = None
        worst_day = None
        for offset in range(1, horizon_days + 1):
            d = (rec_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            if d not in series:
                continue
            px = series[d]
            ret_pct = (px - entry_px) / entry_px * 100
            if ret_pct < worst_pct:
                worst_pct = ret_pct
                worst_date = d
                worst_day = offset

        if worst_pct > drawdown_threshold:
            continue  # Did not breach threshold

        endorsing, dissenting = _agent_views_to_lists(c["stock"])
        wf_top = _waterfall_top_n(c["stock"])

        # Lesson summary
        lesson = _build_lesson(c["stock"], dissenting, wf_top, worst_pct)

        post_mortems.append(PostMortem(
            ticker=tkr,
            recommendation_date=c["date"],
            action=c["stock"].get("action", ""),
            conviction=int(c["stock"].get("conviction", 0)),
            entry_price=entry_px,
            drawdown_date=worst_date or "",
            drawdown_pct=round(worst_pct, 2),
            days_to_drawdown=worst_day or 0,
            endorsing_agents=endorsing,
            dissenting_agents=dissenting,
            waterfall_top=wf_top,
            kill_thesis_text=str(c["stock"].get("kill_thesis", ""))[:200],
            lesson=lesson,
        ))

    return post_mortems


def _build_lesson(
    stock: Dict[str, Any],
    dissenters: List[str],
    wf_top: List[Dict[str, Any]],
    worst_pct: float,
) -> str:
    """Compose a 1-2 sentence lesson summary."""
    parts = [
        f"Recommended at conviction {stock.get('conviction', '?')} but dropped {worst_pct:.1f}%."
    ]
    if dissenters:
        parts.append(f"Pre-fire dissent from: {', '.join(dissenters[:3])}.")
    else:
        parts.append("No agent dissented — full consensus failure.")
    bear_keys = [w for w in wf_top if w["value"] < 0]
    if bear_keys:
        parts.append(
            f"Loudest bearish modifiers: "
            + ", ".join(f"{w['key']}({w['value']})" for w in bear_keys[:3])
            + "."
        )
    return " ".join(parts)


def append_lessons(
    post_mortems: List[PostMortem],
    path: Optional[Path] = None,
) -> int:
    """
    Append each post-mortem as one JSONL line. Dedupes by
    (ticker, recommendation_date) so re-runs don't duplicate.
    """
    if not post_mortems:
        return 0
    out_path = path or DEFAULT_LESSONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                e = json.loads(line)
                existing.add((e.get("ticker"), e.get("recommendation_date")))
            except json.JSONDecodeError:
                continue

    written = 0
    with open(out_path, "a") as f:
        for pm in post_mortems:
            key = (pm.ticker, pm.recommendation_date)
            if key in existing:
                continue
            f.write(json.dumps(asdict(pm), default=str) + "\n")
            written += 1
    return written


def load_recent_lessons(
    n: int = 10,
    path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load the N most recent lessons for inclusion in agent_memory."""
    p = path or DEFAULT_LESSONS_PATH
    if not p.exists():
        return []
    rows = []
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("recommendation_date", ""), reverse=True)
    return rows[:n]


def summarise_for_committee(lessons: List[Dict[str, Any]]) -> str:
    """One-page summary for inclusion in next committee's prompt."""
    if not lessons:
        return ""
    by_pattern = Counter()
    for l in lessons:
        for d in l.get("dissenting_agents", []):
            by_pattern[d] += 1
    common_dissenters = by_pattern.most_common(3)
    lines = [
        f"## Post-Mortem Library ({len(lessons)} prior failures, last 90d)",
        "",
    ]
    for l in lessons[:5]:
        lines.append(
            f"* {l['ticker']} ({l.get('recommendation_date','')}, conv "
            f"{l.get('conviction','?')}): "
            f"dropped {l.get('drawdown_pct','?')}% in "
            f"{l.get('days_to_drawdown','?')}d. {l.get('lesson','')}"
        )
    if common_dissenters:
        lines.append("")
        lines.append("Recurring dissenters across failures:")
        for d, c in common_dissenters:
            lines.append(f"  * {d}: {c} cases")
    lines.append("")
    lines.append("Use these patterns when scoring conviction this run.")
    return "\n".join(lines)

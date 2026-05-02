"""
Bayesian Conviction Update Prototype (CIO v17 op #7)

Replaces the additive bonus/penalty waterfall with a Bayesian update:

  prior(α30 > 0)  = sigmoid_from_conviction(conviction)
  for each agent view v:
      posterior ∝ prior × P(α30 > 0 | view = v)
  recalibrated_conviction = sigmoid⁻¹(posterior) × 100

Per-agent likelihoods come from the rolling track record (the same
calibration evidence used by agent_sign_calibrator) with **shrinkage
towards 0.5** so we don't overfit the small early-data sample.

Runs in **shadow mode** alongside the existing waterfall scorer:
both convictions are computed, both persisted; the user's actions
still come from the waterfall conviction. After 8 weeks of comparable
output, switch the action map to Bayesian.

Math notes:
* Independence assumption between agents — false in practice but a
  reasonable prior; the per-agent calibration matrix can later be
  factor-adjusted via Cholesky once we have enough cells.
* Sigmoid maps conviction 0→p=0.05, 50→p=0.5, 100→p=0.95 (continuous).
* Beta(α=2, β=2) shrinkage prior on per-view P(α30>0) — adds 4 pseudo
  observations split between hits and misses, so a 0/0 view starts
  at 0.5 not undefined.
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LIKELIHOODS_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "bayesian_likelihoods.json"
)
DEFAULT_SHADOW_PATH = (
    Path.home() / ".weirdapps-trading" / "committee" / "bayesian_shadow.json"
)

# Beta shrinkage prior: 4 pseudo-observations evenly split.
PRIOR_HITS = 2.0
PRIOR_MISSES = 2.0


def _sigmoid(x: float) -> float:
    if x > 50:
        return 1.0 / (1.0 + math.exp(-min(x, 500)))
    if x < -50:
        return 1.0 / (1.0 + math.exp(-max(x, -500)))
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(min(p, 0.999), 0.001)
    return math.log(p / (1.0 - p))


def _conviction_to_prior(conviction: float) -> float:
    """Map conviction 0-100 to prior probability of α30>0.

    Linear in logit space: conviction=0 → p=0.05, 50 → 0.5, 100 → 0.95.
    """
    z = (conviction - 50) / 50.0 * _logit(0.95)
    return _sigmoid(z)


def _prior_to_conviction(p: float) -> int:
    """Inverse: posterior probability → conviction 0-100."""
    if p <= 0:
        return 0
    elif p >= 1:
        return 100
    else:
        z = _logit(p)
        return int(round(50 + z / _logit(0.95) * 50))


# ── Likelihood table builder ────────────────────────────────────────────

def _agent_view(stock: Dict[str, Any], agent: str) -> str:
    field_map = {
        "fundamental": "fund_view",
        "technical": "tech_signal",
        "macro": "macro_fit",
        "census": "census",
        "news": "news_impact",
        "risk": None,
    }
    field = field_map.get(agent)
    if not field:
        return "?"
    v = stock.get(field, "?")
    if agent == "news":
        if "POSITIVE" in str(v):
            return "POSITIVE"
        if "NEGATIVE" in str(v):
            return "NEGATIVE"
        return "NEUTRAL"
    return str(v) if v is not None else "?"


def compute_likelihoods(
    history: Iterable[Dict[str, Any]],
    forward_returns: Dict[str, Dict[str, float]],
    horizon: str = "T+30",
) -> Dict[str, Any]:
    """
    Build per-agent, per-view P(α30 > 0 | view) with Beta(2,2) shrinkage.

    Returns:
      {
        "horizon": "T+30",
        "evidence_total": int,
        "agents": {
          "fundamental": {
            "BUY":  {"hits": int, "misses": int, "p_alpha_pos": float, "n": int},
            "HOLD": {...},
            "SELL": {...},
          },
          ...
        }
      }
    """
    alpha_key = f"{horizon}_alpha"
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"hits": 0, "misses": 0})
    )
    total = 0
    agents = ("fundamental", "technical", "macro", "census", "news")

    for entry in history:
        date_str = entry.get("date", "")
        for stock in entry.get("concordance", []):
            if not isinstance(stock, dict):
                continue
            ticker = stock.get("ticker", "")
            if not ticker:
                continue
            alpha = (forward_returns.get(f"{ticker}:{date_str}", {}) or {}).get(alpha_key)
            if alpha is None:
                continue
            total += 1
            for agent in agents:
                view = _agent_view(stock, agent)
                if view == "?":
                    continue
                if alpha > 0:
                    counts[agent][view]["hits"] += 1
                else:
                    counts[agent][view]["misses"] += 1

    out = {
        "generated_at": datetime.now().isoformat(),
        "horizon": horizon,
        "evidence_total": total,
        "agents": {},
    }
    for agent, views in counts.items():
        out["agents"][agent] = {}
        for view, c in views.items():
            n = c["hits"] + c["misses"]
            # Beta(2,2) shrinkage
            p = (c["hits"] + PRIOR_HITS) / (n + PRIOR_HITS + PRIOR_MISSES)
            out["agents"][agent][view] = {
                "hits": c["hits"],
                "misses": c["misses"],
                "n": n,
                "p_alpha_pos": round(p, 3),
            }
    return out


# ── Posterior computation ───────────────────────────────────────────────

def bayesian_posterior(
    stock: Dict[str, Any],
    likelihoods: Dict[str, Any],
    use_independence_assumption: bool = True,
) -> Dict[str, Any]:
    """
    Compute Bayesian posterior P(α30>0 | conviction, all agent views).

    By default treats agent likelihoods as conditionally independent
    given α (false in practice but a reasonable prior). When we have
    enough cells, replace with a Cholesky-corrected joint likelihood.

    Returns:
      {
        "conviction_prior": int,
        "p_alpha_pos_prior": float,
        "p_alpha_pos_posterior": float,
        "conviction_posterior": int,
        "delta": int,                   # posterior - prior in conviction pts
        "agent_contributions": {...}    # log-likelihood per agent
      }
    """
    conv_prior = _safe_float(stock.get("conviction"), 50.0)
    p_prior = _conviction_to_prior(conv_prior)

    # In log-odds space: posterior_logit = prior_logit + Σ log(L_i / (1-L_i))
    log_post = _logit(p_prior)
    contributions: Dict[str, float] = {}
    agents_block = (likelihoods or {}).get("agents") or {}
    for agent, views in agents_block.items():
        view = _agent_view(stock, agent)
        if view == "?" or view not in views:
            continue
        p_view = views[view].get("p_alpha_pos")
        if p_view is None or p_view <= 0 or p_view >= 1:
            continue
        # Log-likelihood ratio for this agent's view
        llr = _logit(p_view) - _logit(p_prior)
        log_post += llr
        contributions[agent] = round(llr, 3)

    p_post = _sigmoid(log_post)
    conv_post = _prior_to_conviction(p_post)

    return {
        "conviction_prior": int(round(conv_prior)),
        "p_alpha_pos_prior": round(p_prior, 3),
        "p_alpha_pos_posterior": round(p_post, 3),
        "conviction_posterior": conv_post,
        "delta": conv_post - int(round(conv_prior)),
        "agent_contributions": contributions,
    }


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def shadow_score_concordance(
    concordance: Iterable[Dict[str, Any]],
    likelihoods: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the Bayesian update over an entire concordance and persist a
    shadow file. The synthesis still uses the waterfall conviction;
    this just records what the Bayesian engine *would* have said.
    """
    out_rows = []
    for stock in concordance:
        if not isinstance(stock, dict):
            continue
        ticker = stock.get("ticker", "")
        if not ticker:
            continue
        post = bayesian_posterior(stock, likelihoods)
        post["ticker"] = ticker
        post["signal"] = stock.get("signal")
        post["action_waterfall"] = stock.get("action")
        out_rows.append(post)

    return {
        "generated_at": datetime.now().isoformat(),
        "horizon": (likelihoods or {}).get("horizon", "T+30"),
        "n_stocks": len(out_rows),
        "rows": out_rows,
        "summary": _summarise_shadow(out_rows),
    }


def _summarise_shadow(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    deltas = [r["delta"] for r in rows]
    big_up = [r for r in rows if r["delta"] >= 10]
    big_down = [r for r in rows if r["delta"] <= -10]
    return {
        "mean_delta": round(sum(deltas) / len(deltas), 2),
        "max_delta": max(deltas) if deltas else 0,
        "min_delta": min(deltas) if deltas else 0,
        "n_upgrade_10pt": len(big_up),
        "n_downgrade_10pt": len(big_down),
        "biggest_upgrades": [
            {"ticker": r["ticker"], "delta": r["delta"]}
            for r in sorted(big_up, key=lambda r: -r["delta"])[:5]
        ],
        "biggest_downgrades": [
            {"ticker": r["ticker"], "delta": r["delta"]}
            for r in sorted(big_down, key=lambda r: r["delta"])[:5]
        ],
    }


def persist_likelihoods(
    likelihoods: Dict[str, Any],
    path: Optional[Path] = None,
) -> Path:
    p = path or DEFAULT_LIKELIHOODS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(likelihoods, indent=2, default=str))
    return p


def load_likelihoods(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    p = path or DEFAULT_LIKELIHOODS_PATH
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def persist_shadow(shadow: Dict[str, Any], path: Optional[Path] = None) -> Path:
    p = path or DEFAULT_SHADOW_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(shadow, indent=2, default=str))
    return p

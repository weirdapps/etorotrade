"""cio.py — CIO synthesize: combine conviction + action + persona-advisory.

synthesize(candidates, held_tickers, cfg=None) -> list[dict]

Args:
    candidates:    list of row dicts (each must have 'ticker' key plus signal fields).
    held_tickers:  set/iterable of ticker strings currently held in the portfolio.
    cfg:           optional threshold overrides for assign_action (buy/add/hold/trim).

Returns:
    list of dicts, one per actionable ticker, sorted by conviction descending.
    Each dict: {ticker, action, conviction, persona_consensus, persona_dissent, rationale}
    'NONE' actions are EXCLUDED.
    Held tickers that drop from the candidate set are included with action='SELL'.

Long-only. Never emits SHORT. Pure function (no I/O, no LLM calls).
"""

from __future__ import annotations

from .actions import assign_action
from .conviction import score_conviction
from .personas import persona_debate


def _build_rationale(
    ticker: str,
    action: str,
    conviction: float,
    persona_consensus: str,
    persona_dissent: list[str],
) -> str:
    """Build a compact human-readable rationale string."""
    dissent_str = f"; dissent: {', '.join(persona_dissent)}" if persona_dissent else ""
    return (
        f"{ticker}: {action} | conviction={conviction:.1f} "
        f"| personas={persona_consensus}{dissent_str}"
    )


def synthesize(
    candidates: list[dict],
    held_tickers: set | frozenset | list,
    cfg: dict | None = None,
) -> list[dict]:
    """Produce the final selected universe from S2 candidates + portfolio.

    Algorithm:
    1. For each candidate: compute conviction, in_universe=True, is_held,
       assign action, run persona_debate (advisory annotation).
    2. Exclude NONE actions.
    3. For each held ticker NOT present in candidates: emit SELL (dropped).
    4. Sort by conviction descending.

    ADVISORY: persona_debate output annotates but does NOT affect conviction
    or action. The function never calls persona_debate before or during
    conviction/action computation.
    """
    held_set = set(held_tickers) if not isinstance(held_tickers, set) else held_tickers
    candidate_tickers = {row["ticker"] for row in candidates}

    results: list[dict] = []

    # Step 1: process candidates
    for row in candidates:
        ticker = row["ticker"]
        is_held = ticker in held_set

        conviction = score_conviction(row)
        action = assign_action(conviction, is_held=is_held, in_universe=True, cfg=cfg)

        if action == "NONE":
            continue  # not selected

        # Advisory annotation — computed AFTER action, no coupling
        debate = persona_debate(row)
        persona_cons = debate["consensus"]
        persona_dis = debate["dissent"]

        results.append(
            {
                "ticker": ticker,
                "action": action,
                "conviction": conviction,
                "persona_consensus": persona_cons,
                "persona_dissent": persona_dis,
                "rationale": _build_rationale(
                    ticker, action, conviction, persona_cons, persona_dis
                ),
            }
        )

    # Step 2: held tickers dropped from universe → SELL
    dropped = held_set - candidate_tickers
    for ticker in sorted(dropped):  # sorted for determinism
        results.append(
            {
                "ticker": ticker,
                "action": "SELL",
                "conviction": 0.0,
                "persona_consensus": "neutral",
                "persona_dissent": [],
                "rationale": f"{ticker}: SELL | dropped from eligible universe | conviction=0.0",
            }
        )

    # Step 3: sort by conviction descending
    results.sort(key=lambda r: r["conviction"], reverse=True)

    return results

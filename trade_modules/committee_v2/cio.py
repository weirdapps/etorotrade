"""cio.py — CIO synthesize: combine conviction + action + persona-advisory.

synthesize(candidates, held_tickers, held_failed_eligibility=None, cfg=None) -> list[dict]

Args:
    candidates:    list of row dicts (each must have 'ticker' key plus signal fields).
    held_tickers:  set/iterable of ticker strings currently held in the portfolio.
    held_failed_eligibility:
                   optional set/iterable of held tickers that FAILED S1
                   eligibility (truly dropped from the investable universe —
                   lost coverage, cap floor, falling knife). These — and ONLY
                   these — are force-SELL'd. Held names that are simply absent
                   from ``candidates`` for any OTHER reason (e.g. a low relative
                   signal, or a skipped price sleeve) are NOT force-SELL'd here;
                   the caller is responsible for emitting them (e.g. as HOLD).
    cfg:           optional threshold overrides for assign_action (buy/add/hold/trim).

Returns:
    list of dicts, one per actionable ticker, sorted by conviction descending.
    Each dict: {ticker, action, conviction, persona_consensus, persona_dissent, rationale}
    'NONE' actions are EXCLUDED.
    Held tickers in ``held_failed_eligibility`` are included with action='SELL'.

ASYMMETRIC held vs non-held semantics (see final-review F1/F2):
    - Held + present in candidates → graded by assign_action with
      in_universe=True (ADD/HOLD/TRIM, or SELL only at the genuine very-low
      conviction floor — NOT merely for a low relative signal).
    - Held + in held_failed_eligibility → SELL (the ONLY "dropped → SELL" case).
    - Non-held → BUY only when conviction clears the BUY threshold, else NONE.

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
    held_failed_eligibility: set | frozenset | list | None = None,
    cfg: dict | None = None,
) -> list[dict]:
    """Produce the final selected universe from S2 candidates + portfolio.

    Algorithm:
    1. For each candidate: compute conviction, in_universe=True, is_held,
       assign action, run persona_debate (advisory annotation).
    2. Exclude NONE actions.
    3. For each held ticker in ``held_failed_eligibility`` (and NOT already a
       candidate): emit SELL (truly dropped from the investable universe).
    4. Sort by conviction descending.

    ASYMMETRIC: a held name absent from ``candidates`` is force-SELL'd ONLY if
    it is in ``held_failed_eligibility``. Held names absent for any other reason
    (low relative signal, skipped price sleeve) are the caller's responsibility.

    ADVISORY: persona_debate output annotates but does NOT affect conviction
    or action. The function never calls persona_debate before or during
    conviction/action computation.
    """
    candidate_tickers = {row["ticker"] for row in candidates}
    held_set = set(held_tickers)

    # Only held names that genuinely failed S1 eligibility are force-SELL'd.
    if held_failed_eligibility is None:
        failed_set: set = set()
    else:
        failed_set = set(held_failed_eligibility)
    # A name explicitly present as a candidate is being graded — never override
    # it with a blanket SELL (candidate grading takes precedence).
    failed_set = failed_set - candidate_tickers

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

    # Step 2: held tickers that FAILED S1 eligibility → SELL
    for ticker in sorted(failed_set):  # sorted for determinism
        results.append(
            {
                "ticker": ticker,
                "action": "SELL",
                "conviction": 0.0,
                "persona_consensus": "neutral",
                "persona_dissent": [],
                "rationale": (
                    f"{ticker}: SELL | dropped from eligible universe (failed S1) | conviction=0.0"
                ),
            }
        )

    # Step 3: sort by conviction descending
    results.sort(key=lambda r: r["conviction"], reverse=True)

    return results

"""committee_v2 — S3 final-universe selection (pure core).

Modules:
  conviction  — score_conviction(row, weights=None) -> float [0-100]
  actions     — assign_action(conviction, is_held, in_universe, cfg=None) -> str
  personas    — persona_debate(row) -> dict  [ADVISORY only]
  cio         — synthesize(candidates, held_tickers, cfg=None) -> list[dict]

Long-only. No shorts. No LLM calls. Deterministic.
"""

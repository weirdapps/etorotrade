"""personas.py — deterministic investment-philosophy lenses. ADVISORY ONLY.

persona_debate(row) -> {
    "personas": {name: "approve"|"neutral"|"reject"},
    "consensus": "approve"|"reject"|"split",
    "dissent": [names disagreeing with consensus],
}

Five lenses — each returns approve/neutral/reject based solely on documented
factor rules. OUTPUT IS ADVISORY: must NOT be an input to conviction or action.

Field parsing notes:
  Numeric fields (PET, PEF, PEG, ROE, DE, EG, FCF, 52W, B/beta, UP%) may arrive as
  numeric strings (with or without % suffix) or floats/ints.
  Missing or unparseable → treated as None (lens goes neutral on that criterion).

Lenses:

  buffett — Quality + Value + Low-Volatility (hates leverage):
    approve: ROE>15 AND DE<100 AND PET in (0, 25] AND B<1.2
    reject:  DE>200 OR PET<=0 OR (ROE is present AND ROE<=0 after applying the quality rule)
    neutral: otherwise

    Reject conditions checked first (order: leverage, then earnings):
      1. DE>200 → reject (excessive leverage)
      2. PET<=0 OR PET missing → reject (negative/missing earnings)
      3. ROE present AND ROE<=0 → reject (low quality)

  wood — Momentum + Growth (Ark Invest style):
    approve: EG>20 AND 52W>=70
    reject:  EG<0 OR 52W<40
    neutral: otherwise

  klarman — Margin of Safety (Value + FCF + Low PE):
    approve: UP%>25 AND FCF>0 AND PET>0 AND PET<=15
    reject:  FCF<0 OR PET>40
    neutral: otherwise

  dalio — Diversification / Low-Correlation / Macro-Neutral (beta proxy):
    approve: B<0.8
    reject:  B>1.8
    neutral: 0.8<=B<=1.8 or B missing

  lynch — Growth at Reasonable Price (PEG):
    approve: 0<PEG<1.5 AND EG>0
    reject:  PEG>3
    neutral: otherwise (including PEG<=0 or missing)

consensus derivation:
  Count approve vs reject (ignore neutral).
  majority approve  → consensus='approve'
  majority reject   → consensus='reject'
  tie or all neutral → consensus='split'

dissent: personas whose verdict disagrees with consensus (not neutral ones).
"""

from __future__ import annotations


def _parse_num(value: object) -> float | None:
    """Parse a numeric or percent-string value. Returns None if missing/invalid."""
    if value is None:
        return None
    s = str(value).strip().rstrip("%").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Individual lenses
# ---------------------------------------------------------------------------


def _buffett(row: dict) -> str:
    de = _parse_num(row.get("DE"))
    pet = _parse_num(row.get("PET"))
    roe = _parse_num(row.get("ROE"))
    beta = _parse_num(row.get("B"))

    # Reject checks (in priority order)
    if de is not None and de > 200:
        return "reject"
    if pet is None or pet <= 0:
        return "reject"
    if roe is not None and roe <= 0:
        return "reject"

    # Approve: all positive criteria met
    roe_ok = roe is not None and roe > 15
    de_ok = de is not None and de < 100
    pet_ok = 0 < pet <= 25
    beta_ok = beta is not None and beta < 1.2

    if roe_ok and de_ok and pet_ok and beta_ok:
        return "approve"
    return "neutral"


def _wood(row: dict) -> str:
    eg = _parse_num(row.get("EG"))
    w52 = _parse_num(row.get("52W"))

    # Reject checks
    if eg is not None and eg < 0:
        return "reject"
    if w52 is not None and w52 < 40:
        return "reject"

    # Approve
    eg_ok = eg is not None and eg > 20
    w52_ok = w52 is not None and w52 >= 70
    if eg_ok and w52_ok:
        return "approve"
    return "neutral"


def _klarman(row: dict) -> str:
    up_pct = _parse_num(row.get("UP%"))
    fcf = _parse_num(row.get("FCF"))
    pet = _parse_num(row.get("PET"))

    # Reject checks
    if fcf is not None and fcf < 0:
        return "reject"
    if pet is not None and pet > 40:
        return "reject"

    # Approve
    up_ok = up_pct is not None and up_pct > 25
    fcf_ok = fcf is not None and fcf > 0
    pet_ok = pet is not None and 0 < pet <= 15
    if up_ok and fcf_ok and pet_ok:
        return "approve"
    return "neutral"


def _dalio(row: dict) -> str:
    beta = _parse_num(row.get("B"))

    if beta is None:
        return "neutral"
    if beta < 0.8:
        return "approve"
    if beta > 1.8:
        return "reject"
    return "neutral"


def _lynch(row: dict) -> str:
    peg = _parse_num(row.get("PEG"))
    eg = _parse_num(row.get("EG"))

    # Reject: high PEG
    if peg is not None and peg > 3:
        return "reject"

    # Reject: negative growth (per spec: EG<=0 → not approve)
    if eg is not None and eg <= 0:
        return "reject"

    # Approve: GARP criteria
    peg_ok = peg is not None and 0 < peg < 1.5
    eg_ok = eg is not None and eg > 0
    if peg_ok and eg_ok:
        return "approve"
    return "neutral"


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

_LENSES = {
    "buffett": _buffett,
    "wood": _wood,
    "klarman": _klarman,
    "dalio": _dalio,
    "lynch": _lynch,
}


def persona_debate(row: dict) -> dict:
    """Run all 5 deterministic investment-philosophy lenses on a candidate row.

    ADVISORY ONLY — this function's output must NOT be an input to conviction
    scoring or action assignment.

    Args:
        row: Candidate dict with signal fields. Not mutated.

    Returns:
        {
            "personas": {name: "approve"|"neutral"|"reject"},
            "consensus": "approve"|"reject"|"split",
            "dissent": [list of persona names disagreeing with consensus],
        }
    """
    verdicts: dict[str, str] = {name: fn(row) for name, fn in _LENSES.items()}

    # Consensus: majority of non-neutral
    approve_count = sum(1 for v in verdicts.values() if v == "approve")
    reject_count = sum(1 for v in verdicts.values() if v == "reject")

    if approve_count > reject_count:
        consensus = "approve"
    elif reject_count > approve_count:
        consensus = "reject"
    else:
        consensus = "split"

    # Dissent: personas that voted the opposite of consensus (not neutral)
    if consensus in ("approve", "reject"):
        opposite = "reject" if consensus == "approve" else "approve"
        dissent = [name for name, v in verdicts.items() if v == opposite]
    else:
        # split: list both sides for visibility
        dissent = [name for name, v in verdicts.items() if v != "neutral"]

    return {
        "personas": verdicts,
        "consensus": consensus,
        "dissent": dissent,
    }

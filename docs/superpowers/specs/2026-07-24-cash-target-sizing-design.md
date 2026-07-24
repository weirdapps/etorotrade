# Cash as a first-class sizing factor — design

- **Date:** 2026-07-24
- **Status:** Approved (design), pending implementation plan
- **Owner:** Dimitrios Plessas
- **Scope:** `etorotrade` v3 overlay (portfolio construction)

## Problem

The v3 overlay's total cash level is neither deliberate nor honestly reported.

1. **Cash is a residual, not a target.** The RISK_ON deployment policy targets **98% invested / 2% cash** (`conditioning.DEPLOYMENT_BY_REGIME["risk_on"] = 0.98`). The book only lands near ~10% cash because the risk gate clamps small, tier-capped buys DOWN and drops the excess to cash with **no redistribution** (`risk_gate._clamp_caps`: "excess → cash"). So the ~10% cash is an *under-absorption shortfall*, not a chosen buffer — if absorption improved, the book would deploy toward 98% (2% cash), which is more aggressive than the owner wants for a copied book near highs.

2. **The deployment headline is misleading.** The report headline shows the **model-part only** (e.g. "deployment 76.6%"), which is `gross_target` AFTER the ~14% managed-sleeve carve-out (`LYXGRE.DE + GLD + UVXY`). Total invested including the held managed sleeves is ~90% (cash ~10%), but the headline reads as if cash were ~23%.

The owner wants cash to be an **explicit, deliberate input to sizing** — the portfolio manager should size positions to a target cash band (~10-12%), accounting for the managed sleeves, rather than letting cash fall out as a residual.

## Goals

- A deliberate, regime-conditional **total-book** cash band (~10-12% in RISK_ON).
- The overlay **sizes to** that target: redeploy un-absorbed budget into eligible names (up to caps + the vol ceiling) instead of dropping it to cash. Cash rises above the operating floor only via a genuine regime / vol / capacity constraint.
- Honest reporting: the deployment headline reflects **true total invested** (model + managed sleeves) plus a cash line and the regime target.

## Non-goals (explicitly unchanged)

- Mega-cap core floor & protection; value-trap / forward-loss gates; factor scoring; managed-sleeve carve-out (GLD/UVXY/LYXGRE stay held-out).
- No shorts (separate decision, rejected).
- The 20% annualized book-vol ceiling remains a HARD constraint and always wins over the deployment target.

## Design

### A. Regime deployment targets (`trade_modules/v3/conditioning.py`)

`DEPLOYMENT_BY_REGIME` is already the **total-book** invested target (the managed carve-out is subtracted afterward: `model gross = resolve_deployment(regime) − managed_weight`). Change the values so total-book cash lands in the intended band:

| regime | current (invested / cash) | new (invested / cash) |
|--------|---------------------------|-----------------------|
| `risk_on`  | 0.98 / 2%  | **0.89 / 11%** |
| `neutral`  | 0.88 / 12% | **0.87 / 13%** |
| `risk_off` | 0.78 / 22% | **0.80 / 20%** |

- Update the `resolve_deployment` band clamp from `(0.78, 0.98)` to `(0.75, 0.92)` so future tilts (e.g. Polymarket) stay within the intended range.
- **Resolved judgment call:** `risk_off = 0.80` (20% cash) — within the debate's 15-25% risk-off band.

### B. Redeploy-to-target — make cash deliberate (`trade_modules/v3/overlay.py`)

A new **post-gate redeploy step** in `build_overlay`, after `apply_risk_gate` and the core floor:

```text
IF sum(final) < gross_target − epsilon AND vol headroom exists:
    gap = gross_target − sum(final)
    redeploy `gap` across the held + bought ELIGIBLE names, conviction-weighted,
      each capped by its tier / sector / USD-bloc name cap,
      re-checking book vol after each allocation so the 20% ceiling is never breached.
    iterate until: gap consumed, OR every eligible name is at a cap, OR vol headroom exhausted.
residual (gap not placed) stays as cash and is REPORTED (never silently hidden).
```

- **Scope (resolved judgment call):** conviction-weighted across ALL held + bought eligible names (more diversified than concentrating in the top-conviction few). Managed sleeves, ineligible names, and core-beyond-its-cap are excluded.
- **Ordering:** allocate to the highest-conviction names first, but cap each at its tier limit so the sweep spreads rather than piling onto one name.
- **Vol ceiling wins:** if redeploying to `gross_target` would push book vol above the ceiling, stop at the ceiling — the extra stays cash. (Today's book vol ~11.2% vs 20% ceiling, so ample headroom.)
- This is additive and isolated: `apply_risk_gate`'s existing "clamp-down / excess→cash" de-gross semantics are untouched; the new step only *up-grosses* toward the target within the same caps.

### C. Reporting fix (`trade_modules/v3/report.py`, `report_email.py`)

- The deployment headline shows **true total invested = model deployment + managed-sleeve weight**, with a cash line and the regime target. Example: `deployment 89% · cash 11% (target 11%)` instead of `76.6%`.
- Surface the managed-sleeve weight explicitly so the split (model vs managed vs cash) is legible.
- Applies to both the browser report and the Outlook-safe email summary.

### D. Tests + rollout

- **TDD** (per changed area):
  - `conditioning`: new regime values + band clamp.
  - `overlay` redeploy: (1) hits `gross_target` when capacity + vol allow; (2) leaves an honest, reported residual when capacity is exhausted; (3) never breaches per-name / sector / USD / vol caps; (4) no-op when already at/above target.
  - reporting: headline reflects total invested incl. managed; cash line correct.
- **Rollout:** merge to master via PR, deploy Mac + VPS, regenerate the snapshot, verify total cash lands ~11% and the headline is honest.

## Expected impact

Net book change is modest. Relative to the **live** book (~94% invested / ~6% cash), the deliberate RISK_ON target of **~89% invested / ~11% cash** is a ~5pp de-risk — ~$55K raised to cash on a $1.097M book. Relative to where the model *accidentally* lands today (~90% invested / ~10% cash via the under-absorption shortfall), the practical book barely moves. The important change is *mechanism*: cash becomes a controlled, regime-conditional target the PM sizes to — not an accidental residual — and the report tells the truth about it.

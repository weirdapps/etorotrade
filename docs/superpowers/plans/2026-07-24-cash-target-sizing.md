# Cash-Target Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make total-book cash a deliberate, regime-conditional target the v3 overlay sizes to (~11% RISK_ON), instead of an under-absorption residual, and report true total invested.

**Architecture:** Three isolated changes: (A) lower the regime deployment targets in `conditioning.py`; (B) add a post-gate "redeploy-to-target" step in `overlay.build_overlay` (a pure `_distribute_to_headroom` helper + a vol-safe wiring block) that up-grosses toward `gross_target` within per-name caps and the vol ceiling; (C) make the report headline include the held-out managed sleeves so deployment/cash are honest.

**Tech Stack:** Python 3.12 (local `.venv`) / 3.14 (VPS), pandas, numpy, pytest. Existing helpers: `risk_gate.portfolio_vol`, `overlay._tiered_name_caps`, `combine.compute_scores`.

## Global Constraints

- TDD: write the failing test first, watch it fail, implement, watch it pass, commit. One behavior per test.
- Run tests with `.venv/bin/python -m pytest <path> -q -p no:cacheprovider`.
- Pre-commit hooks run ruff + ruff-format + mypy + markdownlint on commit; if ruff-format rewrites a file, `git add -A` and re-commit.
- Do NOT change: mega-cap core floor/protection, value-trap/forward-loss gates, factor scoring, managed-sleeve carve-out. The 20% vol ceiling is a HARD constraint the redeploy must never breach.
- Branch: `feat/cash-target-sizing` (already created). Never push to master directly; land via PR.
- Spec: `docs/superpowers/specs/2026-07-24-cash-target-sizing-design.md`.

---

## File Structure

- `trade_modules/v3/conditioning.py` — regime deployment values (Task 1).
- `trade_modules/v3/overlay.py` — `_distribute_to_headroom` helper (Task 2) + redeploy wiring in `build_overlay` (Task 3).
- `scripts/v3_overlay_report.py` + `trade_modules/v3/report.py` + `report_email.py` — reporting of total invested incl. managed (Task 4).
- Tests: `tests/unit/trade_modules/test_v3_conditioning.py`, `test_v3_overlay.py`.

---

## Task 1: Regime deployment targets

**Files:**

- Modify: `trade_modules/v3/conditioning.py:16-20` (`DEPLOYMENT_BY_REGIME`) and `:78` (band default).
- Test: `tests/unit/trade_modules/test_v3_conditioning.py`

**Interfaces:**

- Produces: `DEPLOYMENT_BY_REGIME` = `{"risk_off": 0.80, "neutral": 0.87, "risk_on": 0.89}`; `resolve_deployment(regime)` band default `(0.75, 0.92)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/trade_modules/test_v3_conditioning.py
from trade_modules.v3.conditioning import DEPLOYMENT_BY_REGIME, resolve_deployment


def test_regime_deployment_targets_are_the_cash_band():
    # Total-book invested targets -> cash 20% / 13% / 11%.
    assert DEPLOYMENT_BY_REGIME["risk_off"] == 0.80
    assert DEPLOYMENT_BY_REGIME["neutral"] == 0.87
    assert DEPLOYMENT_BY_REGIME["risk_on"] == 0.89


def test_resolve_deployment_clamps_to_new_band():
    dep, diag = resolve_deployment("risk_on")
    assert dep == 0.89
    assert diag["base_deployment"] == 0.89
    # A large positive tilt cannot exceed the new upper band.
    dep_hi, _ = resolve_deployment("risk_on", polymarket_signal=1.0, max_pm_tilt=0.20)
    assert dep_hi == 0.92
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_conditioning.py -q -p no:cacheprovider`
Expected: FAIL (values are still 0.78/0.88/0.98, band 0.78-0.98).

- [ ] **Step 3: Implement the minimal change**

In `conditioning.py`, set:

```python
DEPLOYMENT_BY_REGIME: dict[str, float] = {
    "risk_off": 0.80,   # 20% cash — the debate's confirmed-risk-off band (15-25%)
    "neutral": 0.87,    # 13% cash
    "risk_on": 0.89,    # 11% cash — deliberate buffer for a copied book near highs
}
```

And update the comment on the line above from `band 78-98%` to `band 75-92%`, and change the `resolve_deployment` signature default `band: tuple[float, float] = (0.78, 0.98)` to `band: tuple[float, float] = (0.75, 0.92)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_conditioning.py -q -p no:cacheprovider`
Expected: PASS. Then run the existing suite that imports these:
Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_overlay.py tests/unit/scripts/test_v3_overlay_report.py -q -p no:cacheprovider`
Expected: PASS (fix any test that hard-coded 0.98/0.78).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/conditioning.py tests/unit/trade_modules/test_v3_conditioning.py
git commit -m "feat(v3): regime deployment targets -> 10-12% cash band"
```

---

## Task 2: `_distribute_to_headroom` pure helper

**Files:**

- Modify: `trade_modules/v3/overlay.py` (add module-level helper near the other `_` helpers).
- Test: `tests/unit/trade_modules/test_v3_overlay.py`

**Interfaces:**

- Produces: `_distribute_to_headroom(weights: pd.Series, conviction: pd.Series, caps: pd.Series, gap: float) -> pd.Series` — returns boosted weights (same index); distributes `gap` proportional to positive conviction, each name bounded by `caps - weights` headroom; unplaceable remainder is simply not added (caller treats as residual cash).

- [ ] **Step 1: Write the failing test**

```python
def test_distribute_to_headroom_fills_by_conviction_up_to_caps():
    from trade_modules.v3.overlay import _distribute_to_headroom

    w = pd.Series({"A": 0.05, "B": 0.05, "C": 0.05})
    conv = pd.Series({"A": 2.0, "B": 1.0, "C": -1.0})  # C disliked -> gets nothing
    caps = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10})
    out = _distribute_to_headroom(w, conv, caps, gap=0.06)
    # 0.06 split 2:1 by conviction between A and B; C untouched.
    assert out["A"] == pytest.approx(0.09)
    assert out["B"] == pytest.approx(0.07)
    assert out["C"] == pytest.approx(0.05)
    assert out.sum() == pytest.approx(w.sum() + 0.06)


def test_distribute_to_headroom_spills_capped_weight_to_others():
    from trade_modules.v3.overlay import _distribute_to_headroom

    w = pd.Series({"A": 0.08, "B": 0.02})
    conv = pd.Series({"A": 3.0, "B": 1.0})  # A favored but near its cap
    caps = pd.Series({"A": 0.10, "B": 0.10})
    out = _distribute_to_headroom(w, conv, caps, gap=0.06)
    # A can only take 0.02 (to its 0.10 cap); the remaining 0.04 spills to B.
    assert out["A"] == pytest.approx(0.10)
    assert out["B"] == pytest.approx(0.06)


def test_distribute_to_headroom_leaves_unplaceable_gap():
    from trade_modules.v3.overlay import _distribute_to_headroom

    w = pd.Series({"A": 0.09, "B": 0.09})
    conv = pd.Series({"A": 1.0, "B": 1.0})
    caps = pd.Series({"A": 0.10, "B": 0.10})
    out = _distribute_to_headroom(w, conv, caps, gap=0.50)  # only 0.02 placeable
    assert out.sum() == pytest.approx(0.20)  # 0.18 + 0.02; rest unplaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_overlay.py -k distribute_to_headroom -q -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name '_distribute_to_headroom'`.

- [ ] **Step 3: Implement the helper**

```python
def _distribute_to_headroom(
    weights: pd.Series, conviction: pd.Series, caps: pd.Series, gap: float
) -> pd.Series:
    """Distribute ``gap`` across ``weights``' names proportional to positive conviction,
    each bounded by its per-name cap headroom (``caps - weights``). Weight that spills
    off a name hitting its cap is re-distributed to those with remaining headroom, so
    the whole gap is placed unless every name is capped. Any unplaceable remainder is
    left unplaced (the caller treats it as residual cash). Pure; no vol/sector logic."""
    w = weights.astype(float).copy()
    if gap is None or gap <= 1e-12:
        return w
    names = list(w.index)
    conv = conviction.reindex(names).astype(float).clip(lower=0.0).fillna(0.0)
    cap = caps.reindex(names).astype(float)
    remaining = float(gap)
    for _ in range(64):  # each pass fills >=1 name to cap or exhausts the gap -> converges
        head = (cap - w).clip(lower=0.0)
        active = (head > 1e-12) & (conv > 0.0)
        wsum = float(conv[active].sum())
        if remaining <= 1e-12 or not bool(active.any()) or wsum <= 0.0:
            break
        alloc = pd.Series(0.0, index=names)
        alloc[active] = remaining * (conv[active] / wsum)
        alloc = pd.concat([alloc, head], axis=1).min(axis=1)  # clip each to its headroom
        placed = float(alloc.sum())
        if placed <= 1e-12:
            break
        w = w + alloc
        remaining -= placed
    return w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_overlay.py -k distribute_to_headroom -q -p no:cacheprovider`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/overlay.py tests/unit/trade_modules/test_v3_overlay.py
git commit -m "feat(v3): _distribute_to_headroom — conviction-weighted, cap-bounded gap fill"
```

---

## Task 3: Wire redeploy-to-target into `build_overlay`

**Files:**

- Modify: `trade_modules/v3/overlay.py` — insert a redeploy block after the core-floor block (after the `vol_after` update near line 639) and before the turnover computation (line 644); add `redeploy` to the `diagnostics` dict (near line 663-683).
- Test: `tests/unit/trade_modules/test_v3_overlay.py`

**Interfaces:**

- Consumes: `_distribute_to_headroom` (Task 2), `portfolio_vol` (already imported), `_tiered_name_caps`, in-scope locals `final`, `gross_target`, `cov`, `pos`, `conv_all`, `elig_mask`, `vol_ceiling`, `name_cap`, `tier_name_caps`, `scored`, `target_names`.
- Produces: `diagnostics["redeploy"]` = `{"gap": float, "deployed_before": float, "deployed_after": float, "residual_cash": float}`; `final` now sums to ~`gross_target` when capacity + vol allow.

- [ ] **Step 1: Write the failing test**

```python
def test_build_overlay_redeploys_to_target_instead_of_cash():
    """Under-absorption fix: when small tier-capped buys can't fill the budget, the
    freed weight is redeployed into held eligible names (up to caps) so the book hits
    gross_target, rather than pooling as cash."""
    _tks, _convs, sc = _universe20()
    sc["cap"] = 5e11  # all large-cap so tiered caps give ample per-name headroom
    current = pd.Series({"U00": 0.30, "U01": 0.30})  # 60% held, strong names
    res = build_overlay(
        sc, current, pd.DataFrame(), max_new=2, gross_target=0.90, tier_name_caps=False,
        name_cap=0.60, vol_ceiling=None,
    )
    d, w = res["diagnostics"], res["weights"]
    assert float(w.sum()) == pytest.approx(0.90, abs=0.02)  # hit the target
    assert d["redeploy"]["deployed_after"] >= d["redeploy"]["deployed_before"]


def test_build_overlay_redeploy_respects_vol_ceiling():
    """Redeploy never breaches the vol ceiling — residual stays cash and is reported."""
    _tks, _convs, sc = _universe20()
    current = pd.Series({"U00": 0.20})
    res = build_overlay(
        sc, current, pd.DataFrame(), max_new=0, gross_target=0.95, vol_ceiling=0.01,
    )  # a 1% ceiling is unreachable at 95% gross -> cannot fully redeploy
    w = res["weights"]
    assert float(w.sum()) <= 0.95 + 1e-6
    assert res["diagnostics"]["redeploy"]["residual_cash"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_overlay.py -k "redeploy" -q -p no:cacheprovider`
Expected: FAIL — `KeyError: 'redeploy'` and/or `w.sum()` well below 0.90.

- [ ] **Step 3: Implement the redeploy block**

Insert after the core-floor block closes (immediately before the `# Turnover` comment at line ~644):

```python
    # Redeploy-to-target (owner 2026-07-24): make cash DELIBERATE, not an under-absorption
    # residual. If the gated + floored book sits BELOW gross_target and vol headroom exists,
    # top up held + bought ELIGIBLE names (conviction-weighted, per-name-cap bounded via
    # _distribute_to_headroom), then bisection-scale the increment so the 20% vol ceiling is
    # never breached. Any gap that still can't be placed stays cash and is REPORTED. The
    # gate's DE-gross (excess -> cash) path is untouched; this only up-grosses within caps.
    redeploy_diag = {"gap": 0.0, "deployed_before": float(final.sum()) if len(final) else 0.0,
                     "deployed_after": float(final.sum()) if len(final) else 0.0, "residual_cash": 0.0}
    if len(final) and gross_target is not None:
        deployed = float(final.sum())
        gap = float(gross_target) - deployed
        if gap > 1e-4:
            caps_ser = (
                pd.Series(_tiered_name_caps(scored, target_names), index=target_names)
                if tier_name_caps and scored is not None
                else pd.Series(float(name_cap), index=target_names)
            )
            elig_names = [
                t for t in final.index
                if t in caps_ser.index and bool(elig_mask.get(t, False))
            ]
            if elig_names:
                boosted = _distribute_to_headroom(
                    final.reindex(elig_names), conv_all.reindex(elig_names),
                    caps_ser.reindex(elig_names), gap,
                )
                trial = final.copy()
                trial.loc[elig_names] = boosted
                names_v = [t for t in trial.index if t in pos]
                ix = [pos[t] for t in names_v]
                covm = cov[np.ix_(ix, ix)]
                base = final.reindex(names_v).to_numpy()
                delta = (trial.reindex(names_v).to_numpy() - base)
                if vol_ceiling is not None and portfolio_vol(base + delta, covm) > float(vol_ceiling):
                    lo, hi = 0.0, 1.0
                    for _ in range(40):  # bisection: largest step with vol <= ceiling
                        mid = 0.5 * (lo + hi)
                        if portfolio_vol(base + mid * delta, covm) <= float(vol_ceiling):
                            lo = mid
                        else:
                            hi = mid
                    delta = lo * delta
                final = pd.Series(base + delta, index=names_v)
                final = final[final > 1e-12]
            deployed_after = float(final.sum())
            redeploy_diag = {
                "gap": gap, "deployed_before": deployed, "deployed_after": deployed_after,
                "residual_cash": max(0.0, float(gross_target) - deployed_after),
            }
```

Then add to the `diagnostics` dict (after `"core_floor_applied": core_floor_applied,`):

```python
        "redeploy": redeploy_diag,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/test_v3_overlay.py -q -p no:cacheprovider`
Expected: PASS (new redeploy tests + all existing overlay tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/overlay.py tests/unit/trade_modules/test_v3_overlay.py
git commit -m "feat(v3): redeploy-to-target — deliberate cash, not under-absorption residual"
```

---

## Task 4: Report true total invested (incl. managed sleeves)

**Files:**

- Modify: `scripts/v3_overlay_report.py:237` — `overlay_portfolio_view(overlay, scored)` gains a `managed_weight` param so `gross` includes the held-out managed sleeves; update its call sites (`main()` passes the real `managed_weight`; `build_overlay_preview_html` at ~line 363 passes the default `0.0`).
- Modify: `trade_modules/v3/report.py:623` and `report_email.py` render_summary — label the headline as total invested; no formula change since `portfolio["gross"]`/`["cash"]` now carry the totals.
- Test: `tests/unit/scripts/test_v3_overlay_report.py`

**Interfaces:**

- Consumes: `managed_weight` (float, computed in `main()` as `managed_weight`), the overlay result.
- Produces: `overlay_portfolio_view(overlay, scored, managed_weight=0.0)` → dict with `gross` = model deployed + managed_weight, `cash` = `1 - gross`, `model_gross` = model deployed, `managed_weight` = managed_weight.

- [ ] **Step 1: Write the failing test**

```python
def test_overlay_portfolio_view_gross_includes_managed_sleeves():
    import pandas as pd
    from scripts.v3_overlay_report import overlay_portfolio_view

    overlay = {"weights": pd.Series({"AAA": 0.40, "BBB": 0.35}), "diagnostics": {}}
    view = overlay_portfolio_view(overlay, None, managed_weight=0.14)
    assert view["model_gross"] == pytest.approx(0.75)
    assert view["gross"] == pytest.approx(0.89)      # 0.75 model + 0.14 managed
    assert view["cash"] == pytest.approx(0.11)
    assert view["managed_weight"] == pytest.approx(0.14)


def test_overlay_portfolio_view_defaults_managed_zero():
    import pandas as pd
    from scripts.v3_overlay_report import overlay_portfolio_view

    overlay = {"weights": pd.Series({"AAA": 0.40}), "diagnostics": {}}
    view = overlay_portfolio_view(overlay, None)  # default managed_weight=0.0
    assert view["gross"] == pytest.approx(0.40)
    assert view["managed_weight"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_v3_overlay_report.py -k managed -q -p no:cacheprovider`
Expected: FAIL — `overlay_portfolio_view()` takes 2 positional args / no `managed_weight`; `gross` is model-only 0.75.

- [ ] **Step 3: Implement**

Change the signature at `scripts/v3_overlay_report.py:237` to
`def overlay_portfolio_view(overlay: dict, scored: pd.DataFrame, managed_weight: float = 0.0) -> dict:`
and replace the `gross` computation (line 247) with:

```python
    model_gross = float(weights.sum()) if len(weights) else 0.0
    gross = model_gross + float(managed_weight)
```

Then in the returned dict add `"model_gross": model_gross,` and `"managed_weight": float(managed_weight),` alongside the existing `"gross": gross,` / `"cash": max(0.0, 1.0 - gross),` (these now reflect the total). Update the `main()` call site to pass `managed_weight=managed_weight` (it is already a local there); leave `build_overlay_preview_html`'s call on the default.

In `report.py:623` change the label text to `"{gross} deployed (incl. managed) / {cash} cash"` (keep reading `portfolio.get('gross')`/`('cash')`). In `report_email.py` render_summary, confirm `dep` reads `portfolio.get("gross")` (the total); if it reads `meta["gross_target"]`, switch it to `portfolio.get("gross")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/scripts/test_v3_overlay_report.py tests/unit/trade_modules/test_v3_report.py tests/unit/trade_modules/test_v3_report_email.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/v3_overlay_report.py trade_modules/v3/report.py trade_modules/v3/report_email.py tests/unit/scripts/test_v3_overlay_report.py
git commit -m "feat(v3): report true total invested incl. managed sleeves"
```

---

## Task 5: Full-suite verify, deploy, regenerate

**Files:** none (integration + ops).

- [ ] **Step 1: Run the full v3 sweep**

Run: `.venv/bin/python -m pytest tests/unit/trade_modules/ tests/unit/scripts/ -q -p no:cacheprovider -k "v3 or riskfirst or combine or overlay or conditioning or report or fundamental"`
Expected: all PASS (target ~915+ tests, 0 fail).

- [ ] **Step 2: Land via PR + deploy**

```bash
git push -u origin feat/cash-target-sizing
gh pr create --base master --title "feat(v3): cash as a deliberate regime-conditional sizing target" --body "See docs/superpowers/specs/2026-07-24-cash-target-sizing-design.md"
gh pr merge --merge --admin --delete-branch
```

Then on Mac and VPS: `git checkout master && git pull origin master`.

- [ ] **Step 3: Regenerate the snapshot on the VPS and verify**

Run the account snapshot + overlay report on the VPS with the locked env (as in `scripts/vps/run-v3-report.sh`, no email), then confirm: console prints `deployment: 89%` for RISK_ON, and the report headline shows total invested ~89% / cash ~11% (incl. managed), redeploy diagnostics present.
Expected: total cash lands ~11%; headline honest.

- [ ] **Step 4: Send the fresh snapshot** (only if the owner asks) via `run-v3-report.sh`.

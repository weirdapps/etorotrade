# Trading Model v3 — Phase 1: Price-Spine Validation Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, prices-only pipeline that scores the USD sub-universe with a price spine (12‑1 momentum + low-volatility), computes EUR forward-return labels, and runs it through the existing validation harness to answer the decisive question: **does the price spine show positive out-of-sample cross-sectional IC and pass the DSR gate?**

**Architecture:** A new `trade_modules/v3/` package. Prices (2y, USD) are loaded via the existing `riskfirst.prices.fetch_prices`, converted to EUR via `EURUSD=X`. Two cross-sectional sleeves (momentum, low-vol) are z-scored per rebalance date and combined. Forward returns are computed per (date, horizon), market-demeaned to net alpha. The decision uses (a) cross-sectional rank IC and (b) the existing `validation.harness.evaluate` (DSR/PBO). No fundamentals, no git panel, no LLM — those are later plans.

**Tech Stack:** Python 3.12 (`.venv`), pandas 2.3.3, numpy 2.4.6, scipy 1.17.1, pytest. Reuses `trade_modules.riskfirst.prices`, `trade_modules.validation.harness`, `trade_modules.validation.ic_decay`.

## Global Constraints

- **Working dir / repo:** `~/SourceCode/etorotrade`. All paths below are repo-relative. Run commands from repo root with the venv active (`.venv/bin/python`, `.venv/bin/pytest`).
- **Package location:** all new code under `trade_modules/v3/`.
- **Import-cycle rule:** if config is needed, import `from trade_modules.config_manager import get_config` — NEVER `from yahoofinance.core.config import ...` at module level (that triggers a `yahoofinance ↔ trade_modules` cycle). Phase-1 code needs no config; keep it config-free.
- **Point-in-time (hard):** at rebalance date `as_of`, no computation may read a price with index position > the position of `as_of`. Momentum/vol use `.iloc[... : i-skip]` / `.iloc[i-lookback:i+1]` only.
- **EUR denomination:** `EURUSD=X` is USD-per-EUR. Convert a USD price to EUR by **division**: `eur = usd / eurusd`.
- **Phase-1 scope:** **USD sub-universe only** (US listings = ticker matches `^[A-Z]+$`, no exchange suffix). Multi-currency deferred to Plan #2.
- **Prices:** 2y history via `fetch_prices(tickers, period="2y")` (12‑1 momentum needs a 252-bar warmup; 1y cache is insufficient).
- **Universe CSV:** read `yahoofinance/output/etoro.csv` with `na_values=["--"]`.
- **Validation harness:** `from trade_modules.validation.harness import evaluate`; it takes `results_rows: list[dict]` (NOT a DataFrame) with keys `ticker, signal_date (ISO str), horizon (int), net_alpha (float), signal (str)`, optional `regime`. Returns `report["overall"]["passed"]` / `["dsr"]`.
- **Acceptance gate (spec §7):** primary cross-sectional IC `mean ≥ +0.02`, `t ≥ 3.0`, `hit_rate ≥ 0.55`; harness `overall.passed` (DSR ≥ 0.95, PBO ≤ 0.05). Short windows may legitimately fail `MIN_REGIMES=2` — report it honestly, do not suppress.
- **Discipline:** TDD (failing test first), frequent commits, no placeholders.

---

## File Structure

- `trade_modules/v3/__init__.py` — package marker.
- `trade_modules/v3/universe.py` — load + filter the USD liquid universe from `etoro.csv`.
- `trade_modules/v3/prices.py` — load 2y USD prices, convert to EUR wide frame.
- `trade_modules/v3/labels.py` — forward returns, market-demeaning, cross-sectional IC + summary.
- `trade_modules/v3/spine.py` — 12‑1 momentum + low-vol sleeves, combined per-date scores.
- `trade_modules/v3/validate_spine.py` — build harness rows, run the combined IC + DSR gate.
- `scripts/v3_spine_gate.py` — end-to-end runner; saves the verdict JSON to `~/Downloads`.
- Tests: `tests/unit/trade_modules/test_v3_universe.py`, `test_v3_prices.py`, `test_v3_labels.py`, `test_v3_spine.py`, `test_v3_validate_spine.py`.

---

### Task 1: Package scaffold + USD universe loader

**Files:**
- Create: `trade_modules/v3/__init__.py`
- Create: `trade_modules/v3/universe.py`
- Test: `tests/unit/trade_modules/test_v3_universe.py`

**Interfaces:**
- Produces: `load_universe(etoro_csv_path: str, min_price: float = 1.0, min_cap_usd: float = 5e8) -> list[str]` — sorted USD tickers passing price + market-cap floors. `parse_cap(v) -> float` — parse `"3.5T"/"800B"/"1.2M"` to USD float.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/trade_modules/test_v3_universe.py
import pandas as pd
from trade_modules.v3.universe import load_universe, parse_cap

def test_parse_cap_suffixes():
    assert parse_cap("3.5T") == 3.5e12
    assert parse_cap("800B") == 800e9
    assert parse_cap("1.2M") == 1.2e6
    assert pd.isna(parse_cap("--"))

def test_load_universe_filters(tmp_path):
    csv = tmp_path / "etoro.csv"
    pd.DataFrame({
        "TKR": ["AAPL", "PENNY", "SMALL", "7203.T", "MSFT"],
        "PRC": [200.0, 0.5, 50.0, 3000.0, 400.0],
        "CAP": ["3.5T", "10B", "100M", "40T", "2.5T"],
    }).to_csv(csv, index=False)
    u = load_universe(str(csv), min_price=1.0, min_cap_usd=5e8)
    assert u == ["AAPL", "MSFT"]  # PENNY (price), SMALL (cap), 7203.T (non-USD) excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: trade_modules.v3`.

- [ ] **Step 3: Write minimal implementation**

```python
# trade_modules/v3/__init__.py
"""Trading Model v3 — validation-first pipeline (Phase 1: price-spine gate)."""
```

```python
# trade_modules/v3/universe.py
import re
import pandas as pd

US_TICKER = re.compile(r"^[A-Z]+$")  # US/USD listing = no exchange suffix
_SUFFIX = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3, "k": 1e3}

def parse_cap(v) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return float("nan")
    s = str(v).strip().replace(",", "")
    m = re.match(r"^([0-9.]+)\s*([TBMKk]?)$", s)
    if not m:
        return float("nan")
    return float(m.group(1)) * _SUFFIX.get(m.group(2), 1.0)

def load_universe(etoro_csv_path: str, min_price: float = 1.0, min_cap_usd: float = 5e8) -> list[str]:
    df = pd.read_csv(etoro_csv_path, na_values=["--"])
    tkr = df["TKR"].astype(str)
    mask_us = tkr.str.match(US_TICKER)
    price = pd.to_numeric(df["PRC"], errors="coerce")
    cap = df["CAP"].map(parse_cap)
    keep = mask_us & (price > min_price) & (cap >= min_cap_usd)
    return sorted(df.loc[keep, "TKR"].dropna().astype(str).unique().tolist())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_universe.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/__init__.py trade_modules/v3/universe.py tests/unit/trade_modules/test_v3_universe.py
git commit -m "feat(v3): USD universe loader with price/cap filters"
```

---

### Task 2: EUR price loader

**Files:**
- Create: `trade_modules/v3/prices.py`
- Test: `tests/unit/trade_modules/test_v3_prices.py`

**Interfaces:**
- Consumes: `trade_modules.riskfirst.prices.fetch_prices(tickers, period="2y") -> DataFrame` (dates × tickers, Close).
- Produces: `to_eur(usd_close: pd.DataFrame, eurusd: pd.Series) -> pd.DataFrame` — EUR wide frame. `load_eur_close(tickers: list[str], period: str = "2y", fetch=fetch_prices) -> pd.DataFrame` (the `fetch` seam is injectable for tests).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/trade_modules/test_v3_prices.py
import pandas as pd, numpy as np
from trade_modules.v3.prices import to_eur, load_eur_close

def test_to_eur_divides_by_usd_per_eur():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    usd = pd.DataFrame({"AAPL": [110.0, 121.0, 132.0]}, index=idx)
    eurusd = pd.Series([1.10, 1.10, 1.10], index=idx)  # USD per EUR
    eur = to_eur(usd, eurusd)
    assert np.allclose(eur["AAPL"].values, [100.0, 110.0, 120.0])

def test_load_eur_close_uses_fetch_seam():
    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    def fake_fetch(tickers, period="2y"):
        if tickers == ["EURUSD=X"]:
            return pd.DataFrame({"EURUSD=X": [1.25, 1.25]}, index=idx)
        return pd.DataFrame({"AAPL": [125.0, 250.0]}, index=idx)
    eur = load_eur_close(["AAPL"], fetch=fake_fetch)
    assert np.allclose(eur["AAPL"].values, [100.0, 200.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_prices.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# trade_modules/v3/prices.py
import pandas as pd
from trade_modules.riskfirst.prices import fetch_prices

def to_eur(usd_close: pd.DataFrame, eurusd: pd.Series) -> pd.DataFrame:
    fx = eurusd.reindex(usd_close.index).ffill().bfill()
    return usd_close.div(fx, axis=0)

def load_eur_close(tickers: list[str], period: str = "2y", fetch=fetch_prices) -> pd.DataFrame:
    usd = fetch(list(tickers), period=period)
    fx_df = fetch(["EURUSD=X"], period=period)
    eurusd = fx_df["EURUSD=X"] if "EURUSD=X" in fx_df.columns else fx_df.iloc[:, 0]
    eur = to_eur(usd, eurusd)
    return eur.dropna(how="all")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_prices.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/prices.py tests/unit/trade_modules/test_v3_prices.py
git commit -m "feat(v3): EUR price loader (USD close / EURUSD)"
```

---

### Task 3: Forward-return labels + market-demeaning

**Files:**
- Create: `trade_modules/v3/labels.py`
- Test: `tests/unit/trade_modules/test_v3_labels.py`

**Interfaces:**
- Produces:
  - `forward_returns(eur_close: pd.DataFrame, asof_dates: list, horizons: list[int]) -> pd.DataFrame` — long `[as_of, ticker, horizon, fwd_ret]`; strictly forward (`iloc[i+h]`).
  - `demean_by_date(fwd: pd.DataFrame) -> pd.DataFrame` — adds `net_alpha` = `fwd_ret` minus the per-(as_of, horizon) cross-sectional mean.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/trade_modules/test_v3_labels.py
import pandas as pd, numpy as np
from trade_modules.v3.labels import forward_returns, demean_by_date

def _close():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    return pd.DataFrame({"A": [100,101,102,103,104,105],
                         "B": [100,100,100,100,100,100]}, index=idx, dtype=float)

def test_forward_returns_are_strictly_forward():
    c = _close(); asof = c.index[0]
    fwd = forward_returns(c, [asof], [2])
    a = fwd[(fwd.ticker=="A") & (fwd.horizon==2)]["fwd_ret"].iloc[0]
    assert np.isclose(a, 102/100 - 1)
    # no row uses data at/before as_of for the future leg
    assert (fwd["horizon"] == 2).all()

def test_forward_returns_skips_when_no_future_bar():
    c = _close()
    fwd = forward_returns(c, [c.index[5]], [2])  # last bar, no +2
    assert fwd.empty

def test_demean_by_date():
    c = _close(); asof = c.index[0]
    fwd = demean_by_date(forward_returns(c, [asof], [2]))
    grp = fwd[fwd.horizon==2]
    assert np.isclose(grp["net_alpha"].sum(), 0.0, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_labels.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# trade_modules/v3/labels.py
import numpy as np
import pandas as pd

def forward_returns(eur_close: pd.DataFrame, asof_dates: list, horizons: list[int]) -> pd.DataFrame:
    idx = eur_close.index
    rows = []
    for asof in asof_dates:
        if asof not in idx:
            continue
        i = idx.get_loc(asof)
        base = eur_close.iloc[i]
        for h in horizons:
            j = i + h
            if j >= len(idx):
                continue
            fr = (eur_close.iloc[j] / base) - 1.0
            for tkr, r in fr.dropna().items():
                rows.append({"as_of": asof, "ticker": tkr, "horizon": int(h), "fwd_ret": float(r)})
    return pd.DataFrame(rows, columns=["as_of", "ticker", "horizon", "fwd_ret"])

def demean_by_date(fwd: pd.DataFrame) -> pd.DataFrame:
    out = fwd.copy()
    out["net_alpha"] = out.groupby(["as_of", "horizon"])["fwd_ret"].transform(lambda s: s - s.mean())
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_labels.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/labels.py tests/unit/trade_modules/test_v3_labels.py
git commit -m "feat(v3): forward-return labels + market-demeaning"
```

---

### Task 4: Cross-sectional IC + summary

**Files:**
- Modify: `trade_modules/v3/labels.py` (append)
- Test: `tests/unit/trade_modules/test_v3_labels.py` (append)

**Interfaces:**
- Produces:
  - `cross_sectional_ic(scores: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> pd.Series` — Spearman rank IC per `as_of` (index = as_of). `scores` cols `[as_of, ticker, score]`.
  - `ic_summary(ic: pd.Series) -> dict` — `{n, mean_ic, t_stat, hit_rate}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/trade_modules/test_v3_labels.py
from trade_modules.v3.labels import cross_sectional_ic, ic_summary

def test_cross_sectional_ic_perfect_rank():
    asof = pd.Timestamp("2026-01-01")
    scores = pd.DataFrame({"as_of":[asof]*3, "ticker":["A","B","C"], "score":[1.0,2.0,3.0]})
    fwd = pd.DataFrame({"as_of":[asof]*3, "ticker":["A","B","C"], "horizon":[5,5,5],
                        "fwd_ret":[0.01,0.02,0.03]})
    ic = cross_sectional_ic(scores, fwd, 5)
    assert np.isclose(ic.loc[asof], 1.0)

def test_ic_summary_stats():
    ic = pd.Series([0.1, 0.1, 0.1, 0.1])
    s = ic_summary(ic)
    assert s["n"] == 4 and np.isclose(s["mean_ic"], 0.1) and s["hit_rate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_labels.py -k "ic" -v`
Expected: FAIL with `ImportError: cannot import name 'cross_sectional_ic'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to trade_modules/v3/labels.py
def cross_sectional_ic(scores: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> pd.Series:
    f = fwd[fwd["horizon"] == horizon].merge(scores, on=["as_of", "ticker"], how="inner")
    def _ic(g):
        if len(g) < 3:
            return np.nan
        return g["score"].corr(g["fwd_ret"], method="spearman")
    return f.groupby("as_of").apply(_ic).dropna()

def ic_summary(ic: pd.Series) -> dict:
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    sd = float(ic.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sd / np.sqrt(n)) if (n > 1 and sd and sd > 0) else float("nan")
    hit = float((ic > 0).mean()) if n else float("nan")
    return {"n": n, "mean_ic": mean, "t_stat": t, "hit_rate": hit}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_labels.py -v`
Expected: PASS (all label tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/labels.py tests/unit/trade_modules/test_v3_labels.py
git commit -m "feat(v3): cross-sectional rank IC + summary stats"
```

---

### Task 5: Price-spine sleeves (12‑1 momentum + low-vol)

**Files:**
- Create: `trade_modules/v3/spine.py`
- Test: `tests/unit/trade_modules/test_v3_spine.py`

**Interfaces:**
- Produces:
  - `momentum_12_1(eur_close, asof, skip=21, lookback=252) -> pd.Series` (z-scored cross-section; empty if warmup missing).
  - `low_vol(eur_close, asof, lookback=252) -> pd.Series` (z-scored `-vol`).
  - `spine_scores(eur_close, asof_dates, w_mom=0.5, w_lv=0.5) -> pd.DataFrame` `[as_of, ticker, score]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/trade_modules/test_v3_spine.py
import pandas as pd, numpy as np
from trade_modules.v3.spine import momentum_12_1, low_vol, spine_scores

def _trend_close(n=300):
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    up = 100 * (1.001 ** np.arange(n))      # steady uptrend
    flat = 100 * np.ones(n)
    down = 100 * (0.999 ** np.arange(n))
    return pd.DataFrame({"UP": up, "FLAT": flat, "DOWN": down}, index=idx)

def test_momentum_ranks_uptrend_highest():
    c = _trend_close(); asof = c.index[-1]
    m = momentum_12_1(c, asof)
    assert m["UP"] > m["FLAT"] > m["DOWN"]

def test_momentum_empty_without_warmup():
    c = _trend_close(); asof = c.index[10]  # < lookback
    assert momentum_12_1(c, asof).empty

def test_low_vol_prefers_stable():
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    calm = 100 + np.cumsum(rng.normal(0, 0.05, 300))
    wild = 100 + np.cumsum(rng.normal(0, 2.0, 300))
    c = pd.DataFrame({"CALM": calm, "WILD": wild}, index=idx)
    lv = low_vol(c, c.index[-1])
    assert lv["CALM"] > lv["WILD"]

def test_spine_scores_shape():
    c = _trend_close(); dates = [c.index[-1]]
    s = spine_scores(c, dates)
    assert set(s.columns) == {"as_of", "ticker", "score"}
    assert len(s) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_spine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# trade_modules/v3/spine.py
import numpy as np
import pandas as pd

def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

def momentum_12_1(eur_close: pd.DataFrame, asof, skip: int = 21, lookback: int = 252) -> pd.Series:
    idx = eur_close.index
    if asof not in idx:
        return pd.Series(dtype=float)
    i = idx.get_loc(asof)
    if i - lookback < 0:
        return pd.Series(dtype=float)
    mom = (eur_close.iloc[i - skip] / eur_close.iloc[i - lookback]) - 1.0
    return _z(mom.dropna())

def low_vol(eur_close: pd.DataFrame, asof, lookback: int = 252) -> pd.Series:
    idx = eur_close.index
    if asof not in idx:
        return pd.Series(dtype=float)
    i = idx.get_loc(asof)
    if i - lookback < 0:
        return pd.Series(dtype=float)
    rets = eur_close.iloc[i - lookback:i + 1].pct_change().iloc[1:]
    vol = rets.std(ddof=0)
    return _z(-vol.dropna())

def spine_scores(eur_close: pd.DataFrame, asof_dates, w_mom: float = 0.5, w_lv: float = 0.5) -> pd.DataFrame:
    rows = []
    for asof in asof_dates:
        both = pd.concat(
            [momentum_12_1(eur_close, asof).rename("mom"),
             low_vol(eur_close, asof).rename("lv")], axis=1).dropna()
        if both.empty:
            continue
        score = _z(w_mom * both["mom"] + w_lv * both["lv"])
        for tkr, sc in score.items():
            rows.append({"as_of": asof, "ticker": tkr, "score": float(sc)})
    return pd.DataFrame(rows, columns=["as_of", "ticker", "score"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_spine.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/spine.py tests/unit/trade_modules/test_v3_spine.py
git commit -m "feat(v3): price-spine sleeves (12-1 momentum + low-vol)"
```

---

### Task 6: Gate runner (IC gate + harness DSR)

**Files:**
- Create: `trade_modules/v3/validate_spine.py`
- Test: `tests/unit/trade_modules/test_v3_validate_spine.py`

**Interfaces:**
- Consumes: `validation.harness.evaluate`, `validation.ic_decay.compute_ic_decay`, `labels.*`.
- Produces:
  - `build_rows(scores, fwd, horizons, top_q=0.2, regime=None) -> list[dict]` — long-book (top quantile) harness rows with `net_alpha`.
  - `run_gate(scores, fwd, horizons, n_trials=2, min_obs=10, ic_min=0.02, t_min=3.0, hit_min=0.55) -> dict` — `{ic, ic_decay, harness, primary_ic_pass, dsr_pass, gate_pass}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/trade_modules/test_v3_validate_spine.py
import pandas as pd, numpy as np
from trade_modules.v3.validate_spine import build_rows, run_gate

def _strong_signal(n_dates=40, n_names=30, horizon=5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="W")
    srows, frows = [], []
    for d in dates:
        s = rng.normal(0, 1, n_names)
        for k in range(n_names):
            tkr = f"T{k}"
            srows.append({"as_of": d, "ticker": tkr, "score": float(s[k])})
            # forward return strongly increasing in score + noise
            frows.append({"as_of": d, "ticker": tkr, "horizon": horizon,
                          "fwd_ret": float(0.02 * s[k] + rng.normal(0, 0.01))})
    return pd.DataFrame(srows), pd.DataFrame(frows)

def test_build_rows_shape_and_keys():
    scores, fwd = _strong_signal()
    rows = build_rows(scores, fwd, [5], top_q=0.2)
    assert rows and set(["ticker", "signal_date", "horizon", "net_alpha", "signal"]).issubset(rows[0])
    assert all(r["signal"] == "spine" for r in rows)

def test_run_gate_passes_ic_on_strong_signal():
    scores, fwd = _strong_signal()
    v = run_gate(scores, fwd, [5], n_trials=2, min_obs=5)
    assert v["primary_ic_pass"] is True
    assert v["ic"][5]["mean_ic"] > 0.5

def test_run_gate_rejects_noise():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2025-01-01", periods=40, freq="W")
    srows, frows = [], []
    for d in dates:
        for k in range(30):
            srows.append({"as_of": d, "ticker": f"T{k}", "score": float(rng.normal())})
            frows.append({"as_of": d, "ticker": f"T{k}", "horizon": 5, "fwd_ret": float(rng.normal(0, 0.01))})
    v = run_gate(pd.DataFrame(srows), pd.DataFrame(frows), [5], n_trials=2, min_obs=5)
    assert v["primary_ic_pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_validate_spine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# trade_modules/v3/validate_spine.py
import pandas as pd
from trade_modules.validation.harness import evaluate
from trade_modules.validation.ic_decay import compute_ic_decay
from trade_modules.v3.labels import cross_sectional_ic, ic_summary, demean_by_date

def build_rows(scores, fwd, horizons, top_q: float = 0.2, regime: dict | None = None) -> list[dict]:
    fwd = demean_by_date(fwd)
    rows = []
    for h in horizons:
        f = fwd[fwd["horizon"] == h].merge(scores, on=["as_of", "ticker"], how="inner")
        for asof, g in f.groupby("as_of"):
            thr = g["score"].quantile(1.0 - top_q)
            longs = g[g["score"] >= thr]
            for _, r in longs.iterrows():
                row = {"ticker": str(r["ticker"]),
                       "signal_date": str(pd.Timestamp(asof).date()),
                       "horizon": int(h),
                       "net_alpha": float(r["net_alpha"]),
                       "signal": "spine"}
                if regime is not None:
                    row["regime"] = regime.get(asof, "NA")
                rows.append(row)
    return rows

def run_gate(scores, fwd, horizons, *, n_trials: int = 2, min_obs: int = 10,
             ic_min: float = 0.02, t_min: float = 3.0, hit_min: float = 0.55,
             regime: dict | None = None) -> dict:
    rows = build_rows(scores, fwd, horizons, regime=regime)
    report = evaluate(rows, family_key="signal", n_trials=n_trials,
                      min_obs=min_obs, horizons=tuple(horizons))
    ic = {h: ic_summary(cross_sectional_ic(scores, fwd, h)) for h in horizons}
    decay = compute_ic_decay({h: ic[h]["mean_ic"] for h in horizons})
    ph = horizons[0]
    primary_ic_pass = bool(
        ic[ph]["mean_ic"] >= ic_min and ic[ph]["t_stat"] >= t_min and ic[ph]["hit_rate"] >= hit_min
    )
    dsr_pass = bool(report.get("overall", {}).get("passed", False))
    return {"ic": ic, "ic_decay": decay, "harness": report,
            "primary_ic_pass": primary_ic_pass, "dsr_pass": dsr_pass,
            "gate_pass": bool(primary_ic_pass and dsr_pass)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/trade_modules/test_v3_validate_spine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add trade_modules/v3/validate_spine.py tests/unit/trade_modules/test_v3_validate_spine.py
git commit -m "feat(v3): spine gate runner (cross-sectional IC + harness DSR)"
```

---

### Task 7: End-to-end gate script (the decision)

**Files:**
- Create: `scripts/v3_spine_gate.py`
- Test: none (thin CLI glue over tested modules; verified by manual run).

**Interfaces:**
- Consumes: all of `trade_modules.v3.*`.
- Produces: a JSON verdict written to `~/Downloads/<UTC-stamp>_v3_spine_gate.json` and a printed PASS/FAIL summary.

- [ ] **Step 1: Write the script**

```python
# scripts/v3_spine_gate.py
"""Phase-1 decisive gate: does the price spine (12-1 momentum + low-vol) show
positive OOS cross-sectional IC + pass the DSR gate, on the USD sub-universe, in EUR?
Run: .venv/bin/python scripts/v3_spine_gate.py
"""
import json
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

from trade_modules.v3.universe import load_universe
from trade_modules.v3.prices import load_eur_close
from trade_modules.v3.spine import spine_scores
from trade_modules.v3.labels import forward_returns
from trade_modules.v3.validate_spine import run_gate

ETORO_CSV = "yahoofinance/output/etoro.csv"
HORIZONS = [5, 21, 63]

def month_end_rebalances(index) -> list:
    s = pd.Series(index=index, data=1)
    return list(s.resample("ME").last().dropna().index.intersection(index))

def main() -> None:
    tickers = load_universe(ETORO_CSV)
    print(f"USD universe: {len(tickers)} names")
    eur = load_eur_close(tickers, period="2y")
    eur = eur.dropna(axis=1, thresh=int(0.6 * len(eur)))  # keep names with ≥60% history
    print(f"priced (EUR) names: {eur.shape[1]}, bars: {eur.shape[0]}")

    rebal = [d for d in month_end_rebalances(eur.index)]
    rebal = [d for d in rebal if eur.index.get_loc(d) >= 252]  # momentum warmup
    print(f"rebalance dates (warmed up): {len(rebal)}")

    scores = spine_scores(eur, rebal)
    fwd = forward_returns(eur, rebal, HORIZONS)
    verdict = run_gate(scores, fwd, HORIZONS, n_trials=2, min_obs=10)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    out = os.path.expanduser(f"~/Downloads/{stamp}_v3_spine_gate.json")
    with open(out, "w") as fh:
        json.dump({"ic": verdict["ic"], "ic_decay": verdict["ic_decay"],
                   "primary_ic_pass": verdict["primary_ic_pass"],
                   "dsr_pass": verdict["dsr_pass"], "gate_pass": verdict["gate_pass"],
                   "harness_overall": verdict["harness"].get("overall", {})}, fh, indent=2, default=str)

    print("\n=== PRICE-SPINE GATE VERDICT ===")
    for h in HORIZONS:
        s = verdict["ic"][h]
        print(f"  h={h:>3}d  IC={s['mean_ic']:+.4f}  t={s['t_stat']:+.2f}  hit={s['hit_rate']:.0%}  n={s['n']}")
    print(f"  primary_ic_pass={verdict['primary_ic_pass']}  dsr_pass={verdict['dsr_pass']}  "
          f"GATE_PASS={verdict['gate_pass']}")
    print(f"  report: {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate end-to-end**

Run: `.venv/bin/python scripts/v3_spine_gate.py`
Expected: prints per-horizon IC table + `GATE_PASS=<bool>` and a report path. (Network required for yfinance; run where prices are reachable, e.g. the VPS if the NBG Mac blocks yfinance.)

- [ ] **Step 3: Interpret + record (do not silently pass)**

- If `primary_ic_pass=True`: the spine has genuine cross-sectional edge → this is the honest near-term go-live candidate; proceed to Plan #2 (combiner + risk backbone + fundamental sleeves).
- If `False`: **stop and report** — the price spine alone lacks OOS edge; reconsider sleeve construction / horizons / universe before adding fundamentals. Note whether the harness failed on `MIN_REGIMES` (short/mono-regime window) vs genuine low IC — the `harness_overall.reasons` distinguishes them.

- [ ] **Step 4: Commit**

```bash
git add scripts/v3_spine_gate.py
git commit -m "feat(v3): end-to-end price-spine validation gate script"
```

---

## Self-Review

**Spec coverage (Phase 1 slice of spec §10):** universe/tradability gate → Task 1; EUR PIT prices → Task 2; forward-return labels + cross-sectional IC (spec §7 primary gate) → Tasks 3–4; price spine sleeves (12‑1 momentum + low-vol, spec §6b M4/R1-R3) → Task 5; harness DSR/PBO reuse + combined gate → Task 6; decisive run + honest verdict → Task 7. Deferred by design (own plans): git PIT fundamental panel, fundamental/analyst sleeves, combiner (§4.3), regime/Polymarket (§8), risk gate + construction + execution (§4.2 stages 6–10), the debate retest (§6). Deferred tasks #40–#42 (census cohorts, TipRanks/news IC, accruals) are not part of the price-spine gate.

**Placeholder scan:** none — every step has runnable code + exact commands.

**Type consistency:** `scores` DataFrame `[as_of, ticker, score]` and `fwd` `[as_of, ticker, horizon, fwd_ret(+net_alpha)]` are consistent across Tasks 3–6; `evaluate(...)` is called with `list[dict]` rows carrying `signal_date/horizon/net_alpha/signal` exactly as the harness expects.

**Known limitations to carry into interpretation:** (1) USD-only sub-universe (multi-currency EUR conversion deferred); (2) ~2y price history → the harness may flag `MIN_REGIMES` on a mono-regime window — that is informative, not a bug; (3) `fetch_prices` network dependency (run on the VPS if the Mac blocks yfinance); (4) monthly rebalance is a starting choice — horizons {5,21,63} let us see IC decay before committing a holding period.

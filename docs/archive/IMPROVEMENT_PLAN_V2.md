# Signal Framework Improvement Plan V2

**Date:** 2026-02-05
**Status:** Ready for Implementation

---

## Change 1: Relax EU MEGA Thresholds

**File:** `config.yaml` (lines 243-268)

**Current:**
```yaml
eu_mega:
  buy:
    require_above_200dma: true
    min_fcf_yield: 0.5
    max_debt_equity: 150.0
    min_analysts: 10
    min_price_targets: 6
    min_pct_from_52w_high: 65
```

**Changed To:**
```yaml
eu_mega:
  buy:
    require_above_200dma: false  # CHANGED from true
    min_fcf_yield: -5.0           # CHANGED from 0.5
    max_debt_equity: 200.0        # CHANGED from 150.0
    min_analysts: 6               # CHANGED from 10
    min_price_targets: 4          # CHANGED from 6
    min_pct_from_52w_high: 45     # CHANGED from 65
```

**Rationale:** EU blue chips (SAP, LVMH, ASML) deserve the same treatment as US MEGA caps.

---

## Change 2: Relax EU LARGE Thresholds

**File:** `config.yaml` (lines 288-330)

**Current:**
```yaml
eu_large:
  buy:
    require_above_200dma: true
    min_fcf_yield: 0.25
    max_debt_equity: 150.0
    min_analysts: 8
    min_price_targets: 5
    min_pct_from_52w_high: 70
```

**Changed To:**
```yaml
eu_large:
  buy:
    require_above_200dma: false   # CHANGED from true
    min_fcf_yield: -3.0           # CHANGED from 0.25
    max_debt_equity: 175.0        # CHANGED from 150.0
    min_analysts: 6               # CHANGED from 8
    min_price_targets: 4          # CHANGED from 5
    min_pct_from_52w_high: 50     # CHANGED from 70
```

**Rationale:** EU LARGE caps have lower analyst coverage than US equivalents.

---

## Change 3: Reduce US MEGA Analyst Requirement

**File:** `config.yaml` (lines 17-43)

**Current:**
```yaml
us_mega:
  buy:
    min_analysts: 12
    min_price_targets: 8
    min_pct_from_52w_high: 65
```

**Changed To:**
```yaml
us_mega:
  buy:
    min_analysts: 8               # CHANGED from 12
    min_price_targets: 6          # CHANGED from 8
    min_pct_from_52w_high: 45     # CHANGED from 65
```

**Rationale:**
- Unlocks VISA ($630B, 100% buy, only 5 analysts in our data)
- MEGA caps are well-covered; even 6-8 analysts represents institutional consensus

---

## Change 4: Reduce US LARGE Analyst Requirement

**File:** `config.yaml` (lines 63-87)

**Current:**
```yaml
us_large:
  buy:
    min_analysts: 10
    min_price_targets: 6
    min_pct_from_52w_high: 70
```

**Changed To:**
```yaml
us_large:
  buy:
    min_analysts: 8               # CHANGED from 10
    min_price_targets: 5          # CHANGED from 6
    min_pct_from_52w_high: 50     # CHANGED from 70
```

**Rationale:** Unlocks Mastercard and similar stocks with 8-9 analyst coverage.

---

## Change 5: Increase DE Threshold for MEGA Caps

**File:** `config.yaml`

**Current (US MEGA):**
```yaml
max_debt_equity: 200.0
```

**Changed To:**
```yaml
max_debt_equity: 300.0            # CHANGED from 200.0
```

**Rationale:**
- Payment networks (V, MA) operate with high leverage by design
- MEGA cap debt is typically well-managed and used for buybacks

---

## Change 6: Relax HK Thresholds

**File:** `config.yaml` (lines 469-514)

**Current:**
```yaml
hk_mega:
  buy:
    require_above_200dma: true
    min_analysts: 15
    min_pct_from_52w_high: 70
```

**Changed To:**
```yaml
hk_mega:
  buy:
    require_above_200dma: false   # CHANGED from true
    min_analysts: 10              # CHANGED from 15
    min_pct_from_52w_high: 50     # CHANGED from 70
```

**Rationale:** HK market has fewer analysts but still quality stocks (Tencent, BYD, Xiaomi).

---

## Change 7: Add HUT to Bitcoin Proxy List

**File:** `config.yaml` (bitcoin_proxy section)

**Add:**
```yaml
bitcoin_proxy:
  tickers:
    - MSTR
    - COIN
    - MARA
    - RIOT
    - HUT    # ADD THIS - Hut 8 Mining Corp
    - CLSK
    - BITF
    - HIVE
```

**Rationale:** HUT is a Bitcoin mining company and should be treated as a Bitcoin proxy, not a regular stock.

---

## Summary of All Changes

| # | Change | File | Line(s) | Impact |
|---|--------|------|---------|--------|
| 1 | EU MEGA 200DMA false | config.yaml | 265 | +10 BUY |
| 2 | EU MEGA FCF -5.0 | config.yaml | 263 | +5 BUY |
| 3 | EU MEGA DE 200 | config.yaml | 262 | +3 BUY |
| 4 | EU MEGA analysts 6 | config.yaml | 259 | +8 BUY |
| 5 | EU MEGA 52W% 45 | config.yaml | 264 | +5 BUY |
| 6 | EU LARGE similar | config.yaml | 288-330 | +10 BUY |
| 7 | US MEGA analysts 8 | config.yaml | 33 | +5 BUY |
| 8 | US MEGA 52W% 45 | config.yaml | 39 | +8 BUY |
| 9 | US LARGE analysts 8 | config.yaml | 79 | +5 BUY |
| 10 | US LARGE 52W% 50 | config.yaml | 84 | +10 BUY |
| 11 | US MEGA DE 300 | config.yaml | 36 | +3 BUY |
| 12 | HK MEGA relaxations | config.yaml | 469-514 | +10 BUY |
| 13 | HUT bitcoin proxy | config.yaml | 733 | -1 false BUY |

**Expected Total Impact:** 15 → 75-100 BUY signals

---

## Validation Criteria

After implementation:

1. ✅ BUY signals between 1.5-3% of universe
2. ✅ EU region has >10 BUY signals
3. ✅ HK region has >5 BUY signals
4. ✅ VISA, Mastercard, SAP should be BUY
5. ✅ Tencent, BYD, Xiaomi should be BUY or strong HOLD
6. ✅ HUT should be treated as bitcoin proxy
7. ✅ All tests pass
8. ✅ No false BUY signals with <50% buy rating

---

## Risk Assessment

| Change | Risk Level | Rollback Impact |
|--------|------------|-----------------|
| 200DMA relaxation | Medium | Market timing exposure |
| FCF relaxation | Medium | Growth stock bias |
| DE increase | Low | MEGA caps are safe |
| Analyst reduction | Low | Coverage is still meaningful |
| 52W% reduction | Medium | Value bias, contrarian |
| HK relaxation | Low | Major stocks only |

**Mitigation:** Monitor signal quality after implementation. If false BUY rate increases, tighten analyst requirements.

---

## Implementation Steps

1. Create backup: `cp config.yaml config.yaml.before_v2`
2. Apply changes to config.yaml
3. Run tests: `pytest tests/`
4. Regenerate market.csv: `python trade.py -o e`
5. Verify signal distribution
6. Commit and push

---

*Ready for implementation pending approval.*

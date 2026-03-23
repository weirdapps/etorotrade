# Signal Framework Improvement Plan

**Date:** 2026-02-04
**Status:** Ready for Implementation

---

## Change 1: Fix "TRUST" ETF Misclassification

**File:** `yahoofinance/utils/data/asset_type_utils.py`

**Problem:** "TRUST" in company name triggers ETF classification, but banks/financial companies use "TRUST" in their names.

**Affected Stocks:** NTRS, STB.L, and potentially other financial services companies.

**Code Change:**

```python
# Add to known_non_etfs set (around line 138)
known_non_etfs = {
    'JUP.L',   # Jupiter Fund Management
    'BLK',     # BlackRock
    'TROW',    # T. Rowe Price
    'BEN',     # Franklin Resources
    'IVZ',     # Invesco
    'SEIC',    # SEI Investments
    'AMG',     # Affiliated Managers Group
    'JHG',     # Janus Henderson
    'VCTR',    # Victory Capital
    'APAM',    # Artisan Partners
    'VRTS',    # Virtus Investment Partners
    # ADD THESE:
    'NTRS',    # Northern Trust Corp
    'BNY',     # Bank of New York Mellon
    'STB.L',   # Secure Trust Bank
    'TFC',     # Truist Financial
}

# Add financial service keywords to management_indicators (around line 158)
management_indicators = [
    'MANAGEMENT', 'MANAGERS', 'ASSET MANAGEMENT', 'FUND MANAGEMENT',
    'INVESTMENT MANAGEMENT', 'CAPITAL MANAGEMENT', 'WEALTH MANAGEMENT',
    # ADD THESE:
    'BANK', 'BANCORP', 'FINANCIAL SERVICES', 'TRUST CORP', 'TRUST BANK',
]
```

**Rationale:** Banks and financial services companies often have "TRUST" in their names but are not ETFs. Adding explicit exclusions and pattern matching prevents misclassification.

---

## Change 2: Fix VET Ticker Collision

**File:** `yahoofinance/utils/data/asset_type_utils.py`

**Problem:** VET (Vermilion Energy) collides with VET (Vechain cryptocurrency).

**Code Change:**

```python
def _is_crypto_asset(ticker: str, company_name: Optional[str] = None) -> bool:
    """Check if ticker represents a cryptocurrency."""
    # Crypto tickers typically end with -USD
    if ticker.endswith('-USD'):
        return True

    # Known crypto tickers without -USD suffix
    known_crypto = {
        'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'LINK',
        'XLM', 'DOGE', 'SOL', 'HBAR', 'MATIC', 'AVAX', 'ATOM',
        'ALGO', 'VET', 'FIL', 'THETA', 'TRX', 'EOS', 'XMR',
        'DASH', 'ZEC', 'NEO', 'QTUM', 'ONT', 'IOTA', 'XTZ'
    }

    if ticker in known_crypto:
        # ADD THIS CHECK: If company name suggests a non-crypto business, it's not crypto
        if company_name:
            company_upper = company_name.upper()
            non_crypto_indicators = [
                'ENERGY', 'OIL', 'GAS', 'PETROLEUM', 'MINING', 'MINERALS',
                'INC', 'CORP', 'LTD', 'LIMITED', 'PLC', 'LLC',
                'BANK', 'FINANCIAL', 'INSURANCE', 'MANUFACTURING',
            ]
            if any(indicator in company_upper for indicator in non_crypto_indicators):
                return False
        return True

    return False
```

**Rationale:** Cryptocurrency tickers like VET (Vechain) can collide with stock tickers. By checking the company name for business indicators, we can distinguish between them.

---

## Change 3: Relax 200DMA Requirement for MEGA/LARGE Caps

**File:** `config.yaml`

**Problem:** 212 high-quality stocks blocked by 200DMA requirement, including MSFT, V, MA.

**Code Change:**

```yaml
us_mega:
  buy:
    # ... existing criteria ...
    require_above_200dma: false  # CHANGED from true
    # NEW: Score adjustment instead of blocking
    dma_score_penalty: 10  # Reduce score by 10 points if below 200DMA

us_large:
  buy:
    # ... existing criteria ...
    require_above_200dma: false  # CHANGED from true
    dma_score_penalty: 10

# Keep true for mid/small caps where momentum is more important
us_mid:
  buy:
    require_above_200dma: true

us_small:
  buy:
    require_above_200dma: true
```

**Rationale:**
- MEGA/LARGE caps are well-covered with reliable fundamental data
- Academic research (Lakonishok 1994) shows value in buying quality below moving averages
- Market-wide corrections can push quality stocks below 200DMA temporarily
- Smaller caps need momentum confirmation due to higher volatility

---

## Change 4: Relax FCF Yield Threshold

**File:** `config.yaml`

**Problem:** 66 stocks blocked including BABA due to strict FCF requirement.

**Code Change:**

```yaml
us_mega:
  buy:
    min_fcf_yield: -5.0  # CHANGED from 0.5 - allow moderately negative FCF for growth

us_large:
  buy:
    min_fcf_yield: -3.0  # CHANGED from 0.25 - some flexibility for growth companies

# Keep stricter for smaller caps
us_mid:
  buy:
    min_fcf_yield: 0.0  # Keep at break-even

us_small:
  buy:
    min_fcf_yield: 0.0  # Keep at break-even
```

**Rationale:**
- Large tech/growth companies often have negative FCF during expansion phases
- BABA's -1.6% FCF is due to cloud infrastructure investment, not operational issues
- Academic research values FCF, but thresholds should be sector-aware

---

## Change 5: Increase Debt/Equity Threshold for MEGA Caps

**File:** `config.yaml`

**Problem:** AVGO (Broadcom) blocked with 96% buy rating due to DE > 150%.

**Code Change:**

```yaml
us_mega:
  buy:
    max_debt_equity: 200.0  # CHANGED from 150.0

us_large:
  buy:
    max_debt_equity: 175.0  # CHANGED from 150.0
```

**Rationale:**
- MEGA caps like Broadcom carry higher debt post-acquisitions (VMware, Symantec)
- These are strategic acquisitions with strong cash flow to service debt
- Interest coverage ratio matters more than absolute leverage for large caps

---

## Change 6: Update Signal Logic for Asset Type (Optional)

**File:** `trade_modules/analysis/signals.py`

**Problem:** Need to pass company_name to _is_crypto_asset for ticker collision fix.

**Code Change:**

```python
# Around line 626, update the classify_asset_type call to ensure company_name is passed
if classify_asset_type:
    company_name = company_col.loc[idx] if idx in company_col.index and not pd.isna(company_col.loc[idx]) else None
    asset_type = classify_asset_type(ticker, cap_values.loc[idx], company_name)
```

This should already be in place, but verify the function signature includes company_name.

---

## Implementation Order

### Phase 1: Critical Bug Fixes (Immediate)
1. Change 1: Fix TRUST ETF misclassification
2. Change 2: Fix VET ticker collision

### Phase 2: Threshold Adjustments (After Bug Fixes)
3. Change 3: Relax 200DMA for MEGA/LARGE
4. Change 4: Relax FCF threshold
5. Change 5: Increase DE threshold

### Phase 3: Testing
6. Run full test suite: `pytest tests/`
7. Regenerate market.csv: `python trade.py -o m`
8. Verify signal distribution improvement

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| BUY signals | 13 (0.2%) | ~75-150 (1.5-3%) |
| False BUY signals | 3 | 0 |
| MSFT status | HOLD | BUY |
| BABA status | HOLD | BUY |
| AVGO status | HOLD | BUY |
| NTRS status | BUY (wrong) | HOLD |

---

## Validation Criteria

After implementation:
1. No stocks with <50% buy rating should have BUY signal
2. No stocks with <10% upside should have BUY signal
3. All known ETF patterns still correctly classified
4. All crypto patterns still correctly classified (with company name check)
5. Signal distribution in reasonable range (1-3% BUY, 10-20% SELL, 30-50% HOLD)

---

## Risk Assessment

| Change | Risk Level | Rollback Impact |
|--------|------------|-----------------|
| TRUST fix | Low | Easy revert |
| VET fix | Low | Easy revert |
| 200DMA relaxation | Medium | Could add risk in bear markets |
| FCF relaxation | Medium | Could include cash-burning companies |
| DE increase | Low | Minor impact |

**Mitigation:** Implement changes incrementally and validate signal quality after each phase.

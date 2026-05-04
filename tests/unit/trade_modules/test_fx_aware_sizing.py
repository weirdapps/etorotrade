"""
M8: FX-aware sizing — CIO v36 Empirical Refoundation.

Account is EUR-home (Greek bank exec). Today the sizer is FX-blind:
- max_position_usd=22500 in config.yaml ignores EUR-USD risk
- ~70% of book in USD positions × 8% EURUSD vol = ~560bps unhedged FX
- HK and JP positions add similar uncounted FX vol

M8 adds two pieces:
1. currency_for_ticker(ticker) — infer currency from ticker suffix
2. fx_vol_multiplier(currency, ref_currency='EUR', stock_vol_annual=0.20)
   — returns size scaling that keeps total vol constant when adding FX
   exposure on top of stock vol.

Math: σ_total² = σ_stock² + σ_FX². To preserve total vol budget when
adding FX layer, scale the position by σ_stock / σ_total.
"""

import pytest


class TestCurrencyForTicker:
    @pytest.mark.parametrize(
        "ticker, expected",
        [
            ("AAPL", "USD"),
            ("MSFT", "USD"),
            ("BTC-USD", "USD"),
            ("0700.HK", "HKD"),
            ("2333.HK", "HKD"),
            ("6758.T", "JPY"),
            ("SAP.DE", "EUR"),
            ("ABI.BR", "EUR"),
            ("RHM.DE", "EUR"),
            ("GLE.PA", "EUR"),
            ("NOVO-B.CO", "EUR"),  # CO = Copenhagen, DKK actually but EUR-pegged-ish
            ("PRU.L", "GBP"),
            ("UNK", "USD"),  # default
        ],
    )
    def test_infers_currency_from_suffix(self, ticker, expected):
        from trade_modules.fx_sizing import currency_for_ticker

        assert currency_for_ticker(ticker) == expected


class TestFxVolMultiplier:
    def test_eur_position_returns_unity(self):
        from trade_modules.fx_sizing import fx_vol_multiplier

        # No FX layer when stock currency == reference currency
        assert fx_vol_multiplier("EUR", ref_currency="EUR") == 1.0

    def test_usd_position_returns_less_than_one(self):
        from trade_modules.fx_sizing import fx_vol_multiplier

        m = fx_vol_multiplier("USD", ref_currency="EUR", stock_vol_annual=0.20)
        # σ_stock=0.20, σ_FX=0.08 → scale = 0.20/sqrt(0.04+0.0064) ≈ 0.928
        assert 0.92 <= m <= 0.94

    def test_hkd_pegged_to_usd_similar_haircut(self):
        from trade_modules.fx_sizing import fx_vol_multiplier

        m_usd = fx_vol_multiplier("USD", ref_currency="EUR", stock_vol_annual=0.20)
        m_hkd = fx_vol_multiplier("HKD", ref_currency="EUR", stock_vol_annual=0.20)
        # HKD pegged to USD → similar EUR/HKD vol → similar haircut (within 1pp)
        assert abs(m_usd - m_hkd) < 0.02

    def test_jpy_higher_vol_larger_haircut(self):
        from trade_modules.fx_sizing import fx_vol_multiplier

        m_usd = fx_vol_multiplier("USD", ref_currency="EUR", stock_vol_annual=0.20)
        m_jpy = fx_vol_multiplier("JPY", ref_currency="EUR", stock_vol_annual=0.20)
        # JPY has higher EUR-pair vol → larger haircut
        assert m_jpy < m_usd

    def test_high_vol_stock_smaller_relative_fx_impact(self):
        """A high-vol stock makes FX layer relatively less important."""
        from trade_modules.fx_sizing import fx_vol_multiplier

        m_low_vol = fx_vol_multiplier("USD", stock_vol_annual=0.15)
        m_high_vol = fx_vol_multiplier("USD", stock_vol_annual=0.40)
        # When stock vol is high, FX vol is a smaller share of total vol
        # → less haircut
        assert m_high_vol > m_low_vol


class TestEnrichAppliesFxAdjustment:
    def test_eur_stock_no_haircut(self, monkeypatch):
        # Disable conviction clamp interaction — focus on FX
        from trade_modules import conviction_sizer
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", True)

        conc = [{"ticker": "SAP.DE", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        enrich_with_position_sizes(
            conc,
            portfolio_value=400_000,
            fx_aware=True,
        )
        # EUR ticker → fx_multiplier = 1.0 → no haircut on USD-denominated calc
        assert "fx_currency" in conc[0]
        assert conc[0]["fx_currency"] == "EUR"
        assert conc[0].get("fx_multiplier") == 1.0

    def test_usd_stock_gets_fx_haircut(self, monkeypatch):
        from trade_modules import conviction_sizer
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        monkeypatch.setattr(conviction_sizer, "CONVICTION_CLAMP_TO_UNITY", True)

        conc_eur = [{"ticker": "SAP.DE", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]
        conc_usd = [{"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"}]

        enrich_with_position_sizes(conc_eur, portfolio_value=400_000, fx_aware=True)
        enrich_with_position_sizes(conc_usd, portfolio_value=400_000, fx_aware=True)

        # USD position size < EUR position size due to FX vol layer
        assert conc_usd[0]["suggested_size_usd"] < conc_eur[0]["suggested_size_usd"]
        assert conc_usd[0]["fx_multiplier"] < 1.0

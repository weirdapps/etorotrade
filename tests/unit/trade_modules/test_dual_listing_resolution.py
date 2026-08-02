"""Dual-listing / eToro original-market ticker resolution.

Regression tests for the bug where held dual-listed holdings resolved to the
wrong market: Roche (ROP.ZU) / Novartis (NOVN.ZU) / Investor AB (INVEB.ST) came
back as dead '--' rows because their eToro suffix was never mapped to a Yahoo
symbol, and cross-listed US tickers leaked the eToro *display* form (ASML.NV)
into the *data-fetch* path.

Contract:
  * display ticker  = the eToro original-market symbol, verbatim
  * data-fetch ticker = a live Yahoo Finance symbol (may differ from display)
  * the two must stay consistent with scripts/refresh_etoro_universe._SUFFIX_REMAP
"""

from yahoofinance.utils import ticker_mappings as tm


class TestDataFetchForHeldSwissAndNordicListings:
    """eToro Swiss (.ZU) and Nordic (.ST share-class) holdings must fetch live."""

    def test_roche_zu_fetches_via_swiss_suffix(self):
        # eToro ROP.ZU = Roche; Yahoo carries it under .SW (same .ZU->.SW rule the
        # universe build already uses: refresh_etoro_universe NESN.ZU -> NESN.SW).
        assert tm.get_data_fetch_ticker("ROP.ZU") == "ROP.SW"

    def test_novartis_zu_fetches_via_swiss_suffix(self):
        assert tm.get_data_fetch_ticker("NOVN.ZU") == "NOVN.SW"

    def test_investor_ab_stockholm_gets_share_class_hyphen(self):
        # eToro INVEB.ST = Investor AB ser. B; Yahoo symbol is INVE-B.ST.
        assert tm.get_data_fetch_ticker("INVEB.ST") == "INVE-B.ST"


class TestCrossListedFetchDoesNotLeakDisplayForm:
    """A US/ADR base must never resolve to the eToro .NV display form for fetching."""

    def test_bare_asml_resolves_to_live_yahoo_symbol(self):
        # Bug: get_data_fetch_ticker('ASML') returned 'ASML.NV' (dead on Yahoo).
        assert tm.get_data_fetch_ticker("ASML") == "ASML.AS"

    def test_asml_nv_fetches_amsterdam(self):
        assert tm.get_data_fetch_ticker("ASML.NV") == "ASML.AS"


class TestDisplayIsEtoroOriginalMarket:
    """Display must be the eToro original-market ticker, unchanged."""

    def test_roche_display_stays_zu(self):
        assert tm.get_display_ticker("ROP.ZU") == "ROP.ZU"

    def test_newmont_display_stays_bare(self):
        assert tm.get_display_ticker("NEM") == "NEM"


class TestReverseMappingIsCollisionFree:
    """Many US tickers map to one home listing; the reverse must pick the live US ticker."""

    def test_shell_reverse_is_shel_not_dead_rds_class(self):
        # dual_listed_mappings has SHEL, RDS.A, RDS.B all -> SHEL.L; naive dict
        # inversion collapsed SHEL.L -> RDS.B (a dead ticker).
        assert tm.get_us_ticker("SHEL.L") == "SHEL"

    def test_jd_reverse_is_jd_not_jd_us(self):
        assert tm.get_us_ticker("9618.HK") == "JD"


class TestGeographyForEuropeanSuffixes:
    """Swiss/Nordic suffixes are European, not US, for risk multipliers."""

    def test_swiss_zu_is_europe(self):
        assert tm.get_ticker_geography("ROP.ZU") == "EU"

    def test_stockholm_st_is_europe(self):
        assert tm.get_ticker_geography("INVEB.ST") == "EU"

"""Unit tests for scripts/refresh_etoro_universe.py."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

# Load the script as a module via importlib (avoids reportMissingImports;
# matches the pattern used by tests/unit/scripts/test_validate_brief.py).
_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "scripts" / "refresh_etoro_universe.py"
)
_spec = importlib.util.spec_from_file_location("refresh_etoro_universe", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
refresh_etoro_universe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh_etoro_universe)

# Public functions exposed by the module — aliased here so test bodies stay terse.
extract_symbol = refresh_etoro_universe.extract_symbol
is_stock_or_etf = refresh_etoro_universe.is_stock_or_etf
is_etorian_alias = refresh_etoro_universe.is_etorian_alias
dedupe_by_symbol = refresh_etoro_universe.dedupe_by_symbol
build_exchange_map = refresh_etoro_universe.build_exchange_map
normalize_to_yahoo = refresh_etoro_universe.normalize_to_yahoo

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "etoro_bulk_sample.json"


@pytest.fixture
def bulk_data():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def find(bulk_data, instrument_id):
    """Return the fixture item with given InstrumentID."""
    for item in bulk_data["InstrumentDisplayDatas"]:
        if item["InstrumentID"] == instrument_id:
            return item
    raise KeyError(instrument_id)


class TestExtractSymbol:
    def test_extracts_nasdaq_symbol(self, bulk_data):
        item = find(bulk_data, 1001)  # Apple
        assert extract_symbol(item) == "aapl"

    def test_extracts_xetra_symbol_with_suffix(self, bulk_data):
        item = find(bulk_data, 2001)  # SAP.DE
        assert extract_symbol(item) == "sap.de"

    def test_extracts_hk_symbol_with_padding(self, bulk_data):
        item = find(bulk_data, 2002)  # 0700.HK
        assert extract_symbol(item) == "0700.hk"

    def test_extracts_lse_symbol(self, bulk_data):
        item = find(bulk_data, 2004)  # AZN.L
        assert extract_symbol(item) == "azn.l"

    def test_returns_none_when_images_empty(self, bulk_data):
        item = find(bulk_data, 7001)  # MalformedStock, Images: []
        assert extract_symbol(item) is None

    def test_returns_none_when_no_market_avatars_path(self, bulk_data):
        item = find(bulk_data, 7002)  # NoMarketAvatarStock
        assert extract_symbol(item) is None

    def test_returns_none_when_images_key_missing(self):
        item = {"InstrumentID": 9999, "InstrumentDisplayName": "X"}
        assert extract_symbol(item) is None


class TestIsStockOrEtf:
    def test_stock_passes(self, bulk_data):
        assert is_stock_or_etf(find(bulk_data, 1001))  # Apple, type 5

    def test_etf_passes(self, bulk_data):
        assert is_stock_or_etf(find(bulk_data, 3001))  # SPY, type 6

    def test_forex_filtered(self, bulk_data):
        assert not is_stock_or_etf(find(bulk_data, 1))  # EUR/USD, type 1

    def test_crypto_filtered(self, bulk_data):
        assert not is_stock_or_etf(find(bulk_data, 4001))  # Bitcoin, type 10

    def test_commodity_filtered(self, bulk_data):
        assert not is_stock_or_etf(find(bulk_data, 5001))  # Gold, type 2

    def test_index_filtered(self, bulk_data):
        assert not is_stock_or_etf(find(bulk_data, 6001))  # S&P 500, type 4

    def test_missing_type_id_filtered(self):
        assert not is_stock_or_etf({"InstrumentID": 1})


class TestIsEtorianAlias:
    def test_etorian_prefix_flagged(self, bulk_data):
        assert is_etorian_alias(find(bulk_data, 610))  # ETORIAN610

    def test_normal_name_not_flagged(self, bulk_data):
        assert not is_etorian_alias(find(bulk_data, 1001))  # Apple

    def test_missing_name_not_flagged(self):
        assert not is_etorian_alias({"InstrumentID": 1})


class TestDedupeBySymbol:
    def test_removes_duplicate_symbol_keeps_first(self):
        rows = [
            {"symbol": "AAPL", "company": "Apple", "exchange": "NASDAQ"},
            {"symbol": "MSFT", "company": "Microsoft", "exchange": "NASDAQ"},
            {"symbol": "AAPL", "company": "Apple Duplicate", "exchange": "NASDAQ"},
        ]
        result = dedupe_by_symbol(rows)
        assert len(result) == 2
        assert result[0]["company"] == "Apple"  # First wins
        assert result[1]["symbol"] == "MSFT"

    def test_preserves_order_of_unique_rows(self):
        rows = [
            {"symbol": "B", "company": "Bee"},
            {"symbol": "A", "company": "Ay"},
            {"symbol": "C", "company": "See"},
        ]
        result = dedupe_by_symbol(rows)
        assert [r["symbol"] for r in result] == ["B", "A", "C"]

    def test_empty_list(self):
        assert dedupe_by_symbol([]) == []


class TestBuildExchangeMap:
    def test_builds_from_cross_reference(self, bulk_data, tmp_path):
        # Seed current input/etoro.csv with known suffix mappings
        csv_path = tmp_path / "etoro.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "company", "price", "exchange"])
            w.writerow(["aapl", "Apple", "150", ""])       # NASDAQ → no suffix → "" in 'exchange'
            w.writerow(["msft", "Microsoft", "300", ""])
            w.writerow(["sap.de", "SAP", "150", "DE"])
            w.writerow(["0700.hk", "Tencent", "440", "HK"])
            w.writerow(["azn.l", "AstraZeneca", "120", "L"])

        mapping = build_exchange_map(bulk_data, str(csv_path))

        assert mapping[4] == ""    # NASDAQ (AAPL, MSFT both confirm)
        assert mapping[5] == "DE"  # XETRA
        assert mapping[9] == "HK"  # Hong Kong
        assert mapping[7] == "L"   # LSE

    def test_unknown_exchange_not_in_map(self, bulk_data, tmp_path):
        csv_path = tmp_path / "etoro.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "company", "price", "exchange"])
            w.writerow(["aapl", "Apple", "150", ""])

        mapping = build_exchange_map(bulk_data, str(csv_path))
        assert 9999 not in mapping  # The ExchangeID 9999 from fixture has no cross-reference

    def test_missing_csv_returns_empty_map(self, bulk_data, tmp_path):
        mapping = build_exchange_map(bulk_data, str(tmp_path / "does-not-exist.csv"))
        assert mapping == {}


MAPPING = {4: "", 5: "DE", 6: "AS", 7: "L", 9: "HK"}


class TestNormalizeToYahoo:
    def test_nasdaq_no_suffix(self):
        sym, unmapped = normalize_to_yahoo("aapl", 4, MAPPING)
        assert sym == "AAPL"
        assert unmapped is False

    def test_xetra_with_de_suffix_already_in_symbol(self):
        # The bulk endpoint returns "sap.de" — we should NOT double-append
        sym, unmapped = normalize_to_yahoo("sap.de", 5, MAPPING)
        assert sym == "SAP.DE"
        assert unmapped is False

    def test_xetra_without_suffix_in_symbol(self):
        # Defensive: if symbol has no suffix, we add it
        sym, unmapped = normalize_to_yahoo("bmw", 5, MAPPING)
        assert sym == "BMW.DE"
        assert unmapped is False

    def test_hk_padding_4_digits(self):
        sym, unmapped = normalize_to_yahoo("700.hk", 9, MAPPING)
        assert sym == "0700.HK"
        assert unmapped is False

    def test_hk_already_padded(self):
        sym, unmapped = normalize_to_yahoo("0700.hk", 9, MAPPING)
        assert sym == "0700.HK"
        assert unmapped is False

    def test_hk_5_digit_symbol_left_alone(self):
        # Some HK tickers are 5 digits (e.g. 09988.HK for Alibaba)
        sym, unmapped = normalize_to_yahoo("9988.hk", 9, MAPPING)
        assert sym == "9988.HK"  # No padding for 4+ digit base; keep as-is uppercased
        assert unmapped is False

    def test_lse_with_l_suffix(self):
        sym, unmapped = normalize_to_yahoo("azn.l", 7, MAPPING)
        assert sym == "AZN.L"
        assert unmapped is False

    def test_unmapped_exchange_returns_unmapped_true(self):
        sym, unmapped = normalize_to_yahoo("wxyz", 9999, MAPPING)
        assert sym == "WXYZ"  # no suffix, just uppercased
        assert unmapped is True

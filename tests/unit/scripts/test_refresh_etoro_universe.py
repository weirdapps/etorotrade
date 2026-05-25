"""Unit tests for scripts/refresh_etoro_universe.py."""

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

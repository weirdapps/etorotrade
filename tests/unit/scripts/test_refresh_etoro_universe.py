"""Unit tests for scripts/refresh_etoro_universe.py."""

import csv
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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
is_etorian_alias = refresh_etoro_universe.is_etorian_alias
normalize_symbol = refresh_etoro_universe.normalize_symbol
fix_share_classes = refresh_etoro_universe.fix_share_classes
dedupe_by_symbol = refresh_etoro_universe.dedupe_by_symbol
fetch_page = refresh_etoro_universe.fetch_page
fetch_all_assets = refresh_etoro_universe.fetch_all_assets
write_universe_csv = refresh_etoro_universe.write_universe_csv
write_delta_log = refresh_etoro_universe.write_delta_log
get_credentials = refresh_etoro_universe.get_credentials
main = refresh_etoro_universe.main
MIN_INSTRUMENTS_THRESHOLD = refresh_etoro_universe.MIN_INSTRUMENTS_THRESHOLD
DISCOVER_URL = refresh_etoro_universe.DISCOVER_URL

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "etoro_bulk_sample.json"


@pytest.fixture
def sample_response():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def find(response, instrument_id):
    """Return the fixture item with given instrumentId."""
    for item in response["items"]:
        if item["instrumentId"] == instrument_id:
            return item
    raise KeyError(instrument_id)


class TestIsEtorianAlias:
    def test_etorian_symbol_flagged(self, sample_response):
        assert is_etorian_alias(find(sample_response, 610))  # symbol "ETORIAN610"

    def test_etorian_displayname_flagged(self):
        assert is_etorian_alias({"symbol": "X", "displayName": "ETORIAN999"})

    def test_etorian_symbol_with_normal_name(self):
        assert is_etorian_alias({"symbol": "ETORIAN999", "displayName": "Random Filler"})

    def test_normal_item_not_flagged(self, sample_response):
        assert not is_etorian_alias(find(sample_response, 1001))  # Apple

    def test_missing_fields_not_flagged(self):
        assert not is_etorian_alias({"instrumentId": 1})


class TestNormalizeSymbol:
    def test_us_stock_no_change(self):
        assert normalize_symbol("AAPL") == "AAPL"

    def test_strips_us_suffix(self):
        assert normalize_symbol("STX.US") == "STX"

    def test_strips_us_suffix_lowercase(self):
        assert normalize_symbol("cvx.us") == "CVX"

    def test_drops_rth_variant(self):
        assert normalize_symbol("STX.RTH") is None

    def test_hk_5_digit_to_4(self):
        assert normalize_symbol("00001.HK") == "0001.HK"

    def test_hk_4_digit_unchanged(self):
        assert normalize_symbol("0700.HK") == "0700.HK"

    def test_hk_5_digit_no_leading_zeros(self):
        assert normalize_symbol("09988.HK") == "9988.HK"

    def test_de_suffix_unchanged(self):
        assert normalize_symbol("SAP.DE") == "SAP.DE"

    def test_brk_class_share_unchanged(self):
        assert normalize_symbol("BRK.B") == "BRK.B"

    def test_novo_dash_unchanged(self):
        assert normalize_symbol("NOVO-B.CO") == "NOVO-B.CO"

    def test_asx_remapped_to_ax(self):
        assert normalize_symbol("XYZ.ASX") == "XYZ.AX"

    def test_zu_remapped_to_sw(self):
        assert normalize_symbol("NESN.ZU") == "NESN.SW"

    def test_nv_remapped_to_as(self):
        assert normalize_symbol("HAVAS.NV") == "HAVAS.AS"

    def test_lsb_remapped_to_ls(self):
        assert normalize_symbol("CA366.LSB") == "CA366.LS"

    def test_drops_delisted(self):
        assert normalize_symbol("BLMZ.DELISTED") is None

    def test_drops_test(self):
        assert normalize_symbol("DUCO.TEST") is None

    def test_drops_cvr(self):
        assert normalize_symbol("SURF.CVR") is None

    def test_drops_numeric_suffix(self):
        assert normalize_symbol("DRM.15255") is None

    def test_drops_dup(self):
        assert normalize_symbol("EXH1.DUP10606") is None

    def test_drops_call_put(self):
        assert normalize_symbol("TSLA.CALL1") is None
        assert normalize_symbol("TSLA.PUT2") is None

    def test_keeps_short_numeric_suffix(self):
        assert normalize_symbol("6758.T") == "6758.T"


class TestFixShareClasses:
    def test_ab_pair_gets_hyphen(self):
        rows = [
            {"symbol": "KINVA.ST", "company": "Kinnevik AB ser. A"},
            {"symbol": "KINVB.ST", "company": "Kinnevik AB ser. B"},
            {"symbol": "AAPL", "company": "Apple"},
        ]
        result = fix_share_classes(rows)
        assert result[0]["symbol"] == "KINV-A.ST"
        assert result[1]["symbol"] == "KINV-B.ST"
        assert result[2]["symbol"] == "AAPL"  # unaffected

    def test_already_hyphenated_left_alone(self):
        rows = [
            {"symbol": "ASSA-B.ST", "company": "ASSA ABLOY AB ser. B"},
            {"symbol": "ASSA-A.ST", "company": "ASSA ABLOY AB ser. A"},
        ]
        result = fix_share_classes(rows)
        assert result[0]["symbol"] == "ASSA-B.ST"
        assert result[1]["symbol"] == "ASSA-A.ST"

    def test_single_class_with_keyword_gets_hyphen(self):
        rows = [
            {"symbol": "EKTAB.ST", "company": "Elekta AB Ser. B"},
        ]
        result = fix_share_classes(rows)
        assert result[0]["symbol"] == "EKTA-B.ST"

    def test_false_positive_no_keyword_no_pair(self):
        rows = [
            {"symbol": "DNB.OL", "company": "DNB Bank ASA"},
        ]
        result = fix_share_classes(rows)
        assert result[0]["symbol"] == "DNB.OL"  # unchanged

    def test_non_scandi_suffix_ignored(self):
        rows = [
            {"symbol": "TESTB.DE", "company": "Test Ser. B"},
        ]
        result = fix_share_classes(rows)
        assert result[0]["symbol"] == "TESTB.DE"  # non-scandi, unchanged

    def test_single_char_base_ignored(self):
        rows = [
            {"symbol": "AB.ST", "company": "AB Volvo"},
        ]
        result = fix_share_classes(rows)
        assert result[0]["symbol"] == "AB.ST"  # base "A" too short → leave alone


class TestDedupeBySymbol:
    def test_removes_duplicate_keeps_first(self):
        rows = [
            {"symbol": "AAPL", "company": "Apple"},
            {"symbol": "MSFT", "company": "Microsoft"},
            {"symbol": "AAPL", "company": "Apple Duplicate"},
        ]
        result = dedupe_by_symbol(rows)
        assert len(result) == 2
        assert result[0]["company"] == "Apple"
        assert result[1]["symbol"] == "MSFT"

    def test_preserves_order(self):
        rows = [{"symbol": s} for s in ["B", "A", "C"]]
        assert [r["symbol"] for r in dedupe_by_symbol(rows)] == ["B", "A", "C"]

    def test_empty_list(self):
        assert dedupe_by_symbol([]) == []


class TestGetCredentials:
    def test_env_vars_take_priority(self):
        with patch.dict(os.environ, {"ETORO_API_KEY": "env-api", "ETORO_USER_KEY": "env-user"}):
            api, user = get_credentials()
        assert api == "env-api"
        assert user == "env-user"

    def test_keychain_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(refresh_etoro_universe.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="kc-secret\n")
            api, user = get_credentials()
        assert api == "kc-secret"
        assert user == "kc-secret"

    def test_raises_when_both_missing(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(refresh_etoro_universe.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            with pytest.raises(RuntimeError, match="Missing credentials"):
                get_credentials()


class TestFetchPage:
    def test_success_first_try(self):
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"page": 1, "totalItems": 100, "items": [{"symbol": "AAPL"}]},
            )
            result = fetch_page("Stocks", 1, "api", "user")
            assert result["items"] == [{"symbol": "AAPL"}]

    def test_uses_correct_params(self):
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"items": []},
            )
            fetch_page("ETF", 3, "api-k", "user-k", page_size=500)
            call = mock_get.call_args
            assert call.args[0] == DISCOVER_URL
            assert call.kwargs["params"]["assetClass"] == "ETF"
            assert call.kwargs["params"]["page"] == 3
            assert call.kwargs["params"]["pageSize"] == 500
            assert call.kwargs["headers"]["x-api-key"] == "api-k"
            assert call.kwargs["headers"]["x-user-key"] == "user-k"
            assert "User-Agent" in call.kwargs["headers"]
            assert "x-request-id" in call.kwargs["headers"]

    def test_retries_on_500(self):
        responses = [
            MagicMock(status_code=500, text="boom"),
            MagicMock(status_code=200, json=lambda: {"items": [{"symbol": "X"}]}),
        ]
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get, \
             patch.object(refresh_etoro_universe.time, "sleep"):
            mock_get.side_effect = responses
            result = fetch_page("Stocks", 1, "k", "u")
            assert result["items"][0]["symbol"] == "X"
            assert mock_get.call_count == 2

    def test_raises_after_max_retries(self):
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get, \
             patch.object(refresh_etoro_universe.time, "sleep"):
            mock_get.return_value = MagicMock(status_code=500, text="boom")
            with pytest.raises(RuntimeError, match="all 3 attempts failed"):
                fetch_page("Stocks", 1, "k", "u", max_retries=3)
            assert mock_get.call_count == 3


class TestFetchAllAssets:
    def test_paginates_through_stocks_and_etfs(self):
        # Simulate: Stocks has 3 items total, page_size=2 → 2 pages
        # ETF has 1 item total, page_size=2 → 1 page
        stock_responses = [
            {"page": 1, "totalItems": 3, "items": [{"symbol": "A"}, {"symbol": "B"}]},
            {"page": 2, "totalItems": 3, "items": [{"symbol": "C"}]},
        ]
        etf_responses = [
            {"page": 1, "totalItems": 1, "items": [{"symbol": "SPY"}]},
        ]
        all_responses = iter(stock_responses + etf_responses)

        def fake_fetch_page(asset_class, page, api_key, user_key, page_size=2, **kw):
            return next(all_responses)

        with patch.object(refresh_etoro_universe, "fetch_page", side_effect=fake_fetch_page), \
             patch.object(refresh_etoro_universe.time, "sleep"):
            result = fetch_all_assets("k", "u", page_size=2)

        assert [r["symbol"] for r in result] == ["A", "B", "C", "SPY"]

    def test_empty_page_stops_pagination(self):
        # First page empty — should stop after 1 call per asset class (2 total)
        empty_resp = {"page": 1, "totalItems": 0, "items": []}
        call_count = [0]

        def fake_fetch_page(*args, **kwargs):
            call_count[0] += 1
            return empty_resp

        with patch.object(refresh_etoro_universe, "fetch_page", side_effect=fake_fetch_page), \
             patch.object(refresh_etoro_universe.time, "sleep"):
            result = fetch_all_assets("k", "u")

        assert result == []
        assert call_count[0] == 2  # 1 call per asset class (Stocks + ETF)


class TestWriteUniverseCsv:
    def test_writes_expected_columns(self, tmp_path):
        path = tmp_path / "etoro.csv"
        rows = [
            {"symbol": "AAPL", "company": "Apple", "exchange": "Nasdaq"},
            {"symbol": "SAP.DE", "company": "SAP SE", "exchange": "FRA"},
        ]
        write_universe_csv(rows, str(path))
        content = path.read_text()
        assert content.startswith("symbol,company,exchange")
        assert "AAPL,Apple,Nasdaq" in content
        assert "SAP.DE,SAP SE,FRA" in content

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "etoro.csv"
        path.write_text("old")
        write_universe_csv([{"symbol": "X", "company": "Y", "exchange": ""}], str(path))
        assert "old" not in path.read_text()
        assert "X,Y," in path.read_text()

    def test_atomic_no_tmp_remnant(self, tmp_path):
        path = tmp_path / "etoro.csv"
        write_universe_csv([{"symbol": "X", "company": "Y", "exchange": ""}], str(path))
        assert path.exists()
        assert not (tmp_path / "etoro.csv.tmp").exists()


class TestWriteDeltaLog:
    def test_writes_expected_fields(self, tmp_path):
        path = tmp_path / "log.json"
        write_delta_log(
            path=str(path),
            new_symbols=["N1", "N2"],
            removed_symbols=["O1"],
            total_count=5000,
        )
        data = json.loads(path.read_text())
        assert data["total_count"] == 5000
        assert data["new_count"] == 2
        assert data["removed_count"] == 1
        assert "N1" in data["sample_new"]
        assert "O1" in data["sample_removed"]
        assert "timestamp" in data

    def test_truncates_to_50(self, tmp_path):
        path = tmp_path / "log.json"
        write_delta_log(
            path=str(path),
            new_symbols=[f"S{i}" for i in range(200)],
            removed_symbols=[],
            total_count=5000,
        )
        data = json.loads(path.read_text())
        assert data["new_count"] == 200
        assert len(data["sample_new"]) == 50


class TestMain:
    def test_end_to_end_with_fixture(self, sample_response, tmp_path, monkeypatch):
        # Seed credentials
        monkeypatch.setenv("ETORO_API_KEY", "test-api")
        monkeypatch.setenv("ETORO_USER_KEY", "test-user")

        output_csv = tmp_path / "etoro.csv"
        log_path = tmp_path / "log.json"

        # Pad to 1000+ items to clear the safety threshold (filler will be deduped or filtered)
        padded = {
            "page": 1, "pageSize": 2000, "totalItems": 2000,
            "items": sample_response["items"] + [
                {"instrumentId": 100000 + i, "symbol": f"PAD{i}", "displayName": f"Padding {i}",
                 "assetClass": "Stocks", "exchangeName": "Nasdaq"}
                for i in range(1000)
            ],
        }

        def fake_fetch_all_assets(api_key, user_key, page_size=1000):
            return padded["items"]

        with patch.object(refresh_etoro_universe, "fetch_all_assets", side_effect=fake_fetch_all_assets):
            exit_code = main(output_csv_path=str(output_csv), delta_log_path=str(log_path))

        assert exit_code == 0

        content = output_csv.read_text()
        assert content.startswith("symbol,company,exchange")
        # Includes
        assert "AAPL,Apple" in content
        assert "MSFT,Microsoft" in content
        assert "SAP.DE,SAP SE" in content
        assert "0700.HK,Tencent Holdings" in content
        assert "SPY,SPDR S&P 500" in content
        assert "BRK.A,Berkshire Hathaway" in content
        assert "BRK.B,Berkshire Hathaway B" in content
        # ETORIAN excluded
        assert "ETORIAN610" not in content
        assert "ETORIAN999" not in content
        # Empty-symbol row excluded
        assert "MissingSymbol" not in content
        # Dedupe — only one AAPL
        assert content.count("AAPL,") == 1

    def test_aborts_when_below_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ETORO_API_KEY", "test-api")
        monkeypatch.setenv("ETORO_USER_KEY", "test-user")

        def fake_fetch(*args, **kwargs):
            return [{"symbol": "X", "displayName": "X", "assetClass": "Stocks", "exchangeName": "N"}] * 10

        with patch.object(refresh_etoro_universe, "fetch_all_assets", side_effect=fake_fetch):
            exit_code = main(
                output_csv_path=str(tmp_path / "etoro.csv"),
                delta_log_path=str(tmp_path / "log.json"),
            )
        assert exit_code == 1
        assert not (tmp_path / "etoro.csv").exists()

    def test_aborts_when_credentials_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ETORO_API_KEY", raising=False)
        monkeypatch.delenv("ETORO_USER_KEY", raising=False)
        with patch.object(refresh_etoro_universe.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            exit_code = main(
                output_csv_path=str(tmp_path / "etoro.csv"),
                delta_log_path=str(tmp_path / "log.json"),
            )
        assert exit_code == 1

    def test_min_threshold_is_1000(self):
        assert MIN_INSTRUMENTS_THRESHOLD == 1000

"""
Test suite for trade_modules.portfolio_reconciler module.

Tests portfolio reconciliation between live eToro holdings and signal-generated
portfolio.csv, covering credential retrieval, API fetching, symbol normalization,
CSV reading, position aggregation, and the main reconcile_portfolio orchestrator.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from trade_modules.portfolio_reconciler import (
    ETORO_TO_YAHOO,
    _aggregate_positions,
    _curl_json,
    _fetch_instrument_metadata,
    _fetch_live_portfolio,
    _get_etoro_credentials,
    _normalize_etoro_symbol,
    _read_portfolio_csv,
    reconcile_portfolio,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_credentials():
    """Return a pair of fake API credentials."""
    return ("fake-public-key", "fake-user-key")


@pytest.fixture
def sample_positions():
    """Create realistic eToro position dicts (two AAPL lots + one MSFT)."""
    return [
        {
            "instrumentId": 1001,
            "investmentPct": 0.10,
            "netProfit": 50.0,
            "openRate": 170.0,
        },
        {
            "instrumentId": 1001,
            "investmentPct": 0.05,
            "netProfit": 20.0,
            "openRate": 175.0,
        },
        {
            "instrumentId": 1002,
            "investmentPct": 0.08,
            "netProfit": -10.0,
            "openRate": 330.0,
        },
    ]


@pytest.fixture
def sample_metadata():
    """Instrument metadata keyed by instrument ID."""
    return {
        1001: {
            "instrumentID": 1001,
            "symbolFull": "AAPL",
            "instrumentDisplayName": "Apple Inc.",
        },
        1002: {
            "instrumentID": 1002,
            "symbolFull": "MSFT",
            "instrumentDisplayName": "Microsoft Corp.",
        },
    }


@pytest.fixture
def portfolio_csv_content():
    """CSV content matching the etorotrade portfolio.csv output format."""
    return (
        "TKR,NAME,BS,PRC,UP%\n"
        "AAPL,Apple Inc.,BUY,175.00,12.5\n"
        "MSFT,Microsoft Corp.,HOLD,330.00,8.0\n"
        "GOOGL,Alphabet Inc.,SELL,140.00,-2.0\n"
    )


# ---------------------------------------------------------------------------
# _get_etoro_credentials
# ---------------------------------------------------------------------------


class TestGetEtoroCredentials:
    """Tests for macOS Keychain credential retrieval."""

    def test_success(self, monkeypatch):
        """Successful credential retrieval returns (public_key, user_key)."""
        call_count = 0

        def fake_run(args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.stdout = "pub-key\n" if call_count == 1 else "usr-key\n"
            result.returncode = 0
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        pub, usr = _get_etoro_credentials()
        assert pub == "pub-key"
        assert usr == "usr-key"
        assert call_count == 2

    def test_keychain_failure_raises(self, monkeypatch):
        """CalledProcessError from security binary raises RuntimeError."""

        def fail_run(args, **kwargs):
            raise subprocess.CalledProcessError(44, "security")

        monkeypatch.setattr(subprocess, "run", fail_run)
        with pytest.raises(RuntimeError, match="Failed to get eToro credentials"):
            _get_etoro_credentials()

    def test_timeout_raises(self, monkeypatch):
        """TimeoutExpired from security binary raises RuntimeError."""

        def timeout_run(args, **kwargs):
            raise subprocess.TimeoutExpired("security", 5)

        monkeypatch.setattr(subprocess, "run", timeout_run)
        with pytest.raises(RuntimeError, match="Failed to get eToro credentials"):
            _get_etoro_credentials()


# ---------------------------------------------------------------------------
# _curl_json
# ---------------------------------------------------------------------------


class TestCurlJson:
    """Tests for the curl-based JSON fetcher."""

    def test_success_returns_parsed_json(self, monkeypatch):
        """Successful curl returns parsed JSON dict."""
        payload = {"positions": [{"instrumentId": 1}]}

        def fake_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(payload)
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        data = _curl_json("https://example.com/api", "key", "user")
        assert data == payload

    def test_nonzero_return_code_raises(self, monkeypatch):
        """Non-zero curl exit code raises RuntimeError."""

        def fail_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 22
            result.stderr = "HTTP 404"
            return result

        monkeypatch.setattr(subprocess, "run", fail_run)
        with pytest.raises(RuntimeError, match="eToro API request failed"):
            _curl_json("https://example.com/api", "key", "user")

    def test_invalid_json_raises(self, monkeypatch):
        """Invalid JSON body raises json.JSONDecodeError."""

        def bad_json_run(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = "NOT JSON"
            return result

        monkeypatch.setattr(subprocess, "run", bad_json_run)
        with pytest.raises(json.JSONDecodeError):
            _curl_json("https://example.com/api", "key", "user")

    def test_passes_correct_headers(self, monkeypatch):
        """Verify API key / user key / request-id headers are sent."""
        captured_args = {}

        def capture_run(args, **kwargs):
            captured_args["args"] = args
            result = MagicMock()
            result.returncode = 0
            result.stdout = "{}"
            return result

        monkeypatch.setattr(subprocess, "run", capture_run)
        _curl_json("https://example.com", "my-api-key", "my-user-key")

        args = captured_args["args"]
        # Find header values by position after -H flags
        header_values = []
        for i, a in enumerate(args):
            if a == "-H" and i + 1 < len(args):
                header_values.append(args[i + 1])

        assert any("X-API-KEY: my-api-key" in h for h in header_values)
        assert any("X-USER-KEY: my-user-key" in h for h in header_values)
        assert any("X-REQUEST-ID:" in h for h in header_values)


# ---------------------------------------------------------------------------
# _fetch_live_portfolio
# ---------------------------------------------------------------------------


class TestFetchLivePortfolio:
    """Tests for the live portfolio fetch wrapper."""

    def test_returns_positions_list(self, monkeypatch):
        """Returns the 'positions' list from API response."""
        positions = [{"instrumentId": 1}, {"instrumentId": 2}]

        def fake_curl(url, api_key, user_key):
            assert "portfolio/live" in url
            return {"positions": positions}

        monkeypatch.setattr("trade_modules.portfolio_reconciler._curl_json", fake_curl)
        result = _fetch_live_portfolio("k", "u")
        assert result == positions

    def test_missing_positions_key_returns_empty(self, monkeypatch):
        """Missing 'positions' key returns empty list."""

        def fake_curl(url, api_key, user_key):
            return {"something_else": 42}

        monkeypatch.setattr("trade_modules.portfolio_reconciler._curl_json", fake_curl)
        result = _fetch_live_portfolio("k", "u")
        assert result == []


# ---------------------------------------------------------------------------
# _fetch_instrument_metadata
# ---------------------------------------------------------------------------


class TestFetchInstrumentMetadata:
    """Tests for instrument metadata resolution."""

    def test_empty_ids_returns_empty(self):
        """Empty instrument_ids list returns empty dict without API call."""
        result = _fetch_instrument_metadata([], "k", "u")
        assert result == {}

    def test_parses_instrument_display_datas(self, monkeypatch):
        """Correctly parses instrumentDisplayDatas keyed by instrumentID."""
        api_response = {
            "instrumentDisplayDatas": [
                {"instrumentID": 10, "symbolFull": "AAPL"},
                {"instrumentID": 20, "symbolFull": "MSFT"},
            ]
        }

        def fake_curl(url, api_key, user_key):
            assert "instrumentIds=10,20" in url
            return api_response

        monkeypatch.setattr("trade_modules.portfolio_reconciler._curl_json", fake_curl)
        result = _fetch_instrument_metadata([10, 20], "k", "u")
        assert 10 in result
        assert 20 in result
        assert result[10]["symbolFull"] == "AAPL"

    def test_deduplicates_ids(self, monkeypatch):
        """Duplicate instrument IDs are deduplicated in the request."""
        captured_url = {}

        def fake_curl(url, api_key, user_key):
            captured_url["url"] = url
            return {"instrumentDisplayDatas": []}

        monkeypatch.setattr("trade_modules.portfolio_reconciler._curl_json", fake_curl)
        _fetch_instrument_metadata([5, 5, 3, 3], "k", "u")
        assert "instrumentIds=3,5" in captured_url["url"]

    def test_skips_entries_without_instrument_id(self, monkeypatch):
        """Entries missing instrumentID are skipped."""
        api_response = {
            "instrumentDisplayDatas": [
                {"symbolFull": "ORPHAN"},  # no instrumentID
                {"instrumentID": 99, "symbolFull": "OK"},
            ]
        }

        def fake_curl(url, api_key, user_key):
            return api_response

        monkeypatch.setattr("trade_modules.portfolio_reconciler._curl_json", fake_curl)
        result = _fetch_instrument_metadata([99], "k", "u")
        assert len(result) == 1
        assert 99 in result

    def test_empty_display_datas_returns_empty(self, monkeypatch):
        """Missing instrumentDisplayDatas key returns empty dict."""

        def fake_curl(url, api_key, user_key):
            return {}

        monkeypatch.setattr("trade_modules.portfolio_reconciler._curl_json", fake_curl)
        result = _fetch_instrument_metadata([1], "k", "u")
        assert result == {}


# ---------------------------------------------------------------------------
# _normalize_etoro_symbol
# ---------------------------------------------------------------------------


class TestNormalizeEtoroSymbol:
    """Tests for eToro→Yahoo Finance symbol normalization."""

    @pytest.mark.parametrize(
        "etoro_sym, expected",
        [
            ("BTC", "BTC-USD"),
            ("ETH", "ETH-USD"),
            ("SOL", "SOL-USD"),
            ("DOGE", "DOGE-USD"),
        ],
    )
    def test_crypto_mapping(self, etoro_sym, expected):
        """Known crypto symbols are mapped to Yahoo Finance '-USD' format."""
        assert _normalize_etoro_symbol(etoro_sym) == expected

    def test_crypto_case_insensitive(self):
        """Crypto lookup is case-insensitive (input is uppercased)."""
        assert _normalize_etoro_symbol("btc") == "BTC-USD"
        assert _normalize_etoro_symbol("Eth") == "ETH-USD"

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        assert _normalize_etoro_symbol("  AAPL  ") == "AAPL"

    def test_empty_string_returns_empty(self):
        """Empty string is returned as-is."""
        assert _normalize_etoro_symbol("") == ""

    def test_none_returns_none(self):
        """None input returns None (falsy guard)."""
        assert _normalize_etoro_symbol(None) is None

    def test_regular_stock_passthrough(self):
        """Non-crypto symbols pass through (uppercased)."""
        # Patch out the optional normalize_ticker import so we test the fallback
        with patch.dict("sys.modules", {"yahoofinance.utils.data.ticker_utils": None}):
            result = _normalize_etoro_symbol("aapl")
            assert result == "AAPL"

    def test_all_crypto_entries_covered(self):
        """Every entry in ETORO_TO_YAHOO is exercised."""
        for etoro, yahoo in ETORO_TO_YAHOO.items():
            assert _normalize_etoro_symbol(etoro) == yahoo

    def test_normalize_ticker_import_error_fallback(self):
        """When normalize_ticker import fails, symbol is returned uppercased."""
        with patch.dict("sys.modules", {"yahoofinance.utils.data.ticker_utils": None}):
            # Force the ImportError path
            assert _normalize_etoro_symbol("TSLA") == "TSLA"


# ---------------------------------------------------------------------------
# _read_portfolio_csv
# ---------------------------------------------------------------------------


class TestReadPortfolioCsv:
    """Tests for reading portfolio.csv files."""

    def test_reads_valid_csv(self, tmp_path, portfolio_csv_content):
        """Reads a well-formed portfolio.csv into a dict keyed by TKR."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(portfolio_csv_content)

        result = _read_portfolio_csv(str(csv_file))
        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result
        assert result["AAPL"]["NAME"] == "Apple Inc."
        assert result["GOOGL"]["BS"] == "SELL"

    def test_missing_file_returns_empty(self, tmp_path):
        """Non-existent file returns empty dict."""
        result = _read_portfolio_csv(str(tmp_path / "nonexistent.csv"))
        assert result == {}

    def test_empty_csv_header_only(self, tmp_path):
        """CSV with header but no data rows returns empty dict."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("TKR,NAME,BS,PRC,UP%\n")

        result = _read_portfolio_csv(str(csv_file))
        assert result == {}

    def test_skips_empty_ticker_rows(self, tmp_path):
        """Rows with blank TKR field are skipped."""
        content = "TKR,NAME\nAAPL,Apple\n,NoTicker\n  ,Whitespace\n"
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(content)

        result = _read_portfolio_csv(str(csv_file))
        assert len(result) == 1
        assert "AAPL" in result

    def test_ticker_whitespace_stripped(self, tmp_path):
        """Leading/trailing whitespace in TKR column is stripped."""
        content = "TKR,NAME\n  AAPL  ,Apple\n"
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(content)

        result = _read_portfolio_csv(str(csv_file))
        assert "AAPL" in result

    def test_missing_tkr_column_returns_empty(self, tmp_path):
        """CSV without TKR column produces empty dict (all rows have blank TKR)."""
        content = "SYMBOL,NAME\nAAPL,Apple\n"
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(content)

        result = _read_portfolio_csv(str(csv_file))
        # DictReader returns "" for missing keys, which is falsy -> skipped
        assert result == {}


# ---------------------------------------------------------------------------
# _aggregate_positions
# ---------------------------------------------------------------------------


class TestAggregatePositions:
    """Tests for eToro position aggregation by resolved symbol."""

    def test_aggregates_multiple_lots(self, sample_positions, sample_metadata):
        """Two AAPL lots are merged; MSFT stays separate."""
        result = _aggregate_positions(sample_positions, sample_metadata)

        assert "AAPL" in result
        assert "MSFT" in result

        aapl = result["AAPL"]
        assert aapl["num_positions"] == 2
        # invest_pct = (0.10 + 0.05) * 100 = 15.0
        assert aapl["invest_pct"] == 15.0
        # total_profit = 50 + 20 = 70
        assert aapl["total_profit"] == 70.0
        # weighted avg open = (170*0.10 + 175*0.05) / 0.15 = 25.75/0.15 = 171.6667
        assert abs(aapl["avg_open_rate"] - 171.6667) < 0.01

        msft = result["MSFT"]
        assert msft["num_positions"] == 1
        assert msft["invest_pct"] == 8.0
        assert msft["total_profit"] == -10.0

    def test_empty_positions(self):
        """Empty position list returns empty dict."""
        result = _aggregate_positions([], {})
        assert result == {}

    def test_missing_metadata_uses_fallback_symbol(self):
        """Position with no metadata uses 'ID:<instrumentId>' as symbol."""
        positions = [{"instrumentId": 9999, "investmentPct": 0.05, "netProfit": 0, "openRate": 100}]
        result = _aggregate_positions(positions, {})

        # Fallback: raw_symbol = "ID:9999", normalized passes through
        assert len(result) == 1
        key = list(result.keys())[0]
        assert "9999" in key

    def test_zero_invest_pct_avg_open_is_zero(self):
        """When total investmentPct is 0, avg_open_rate is 0."""
        positions = [{"instrumentId": 1, "investmentPct": 0, "netProfit": 0, "openRate": 100}]
        metadata = {1: {"instrumentID": 1, "symbolFull": "TEST", "instrumentDisplayName": "Test"}}
        result = _aggregate_positions(positions, metadata)

        assert result["TEST"]["avg_open_rate"] == 0.0

    def test_missing_fields_default_to_zero(self):
        """Positions missing numeric fields default to 0 via .get()."""
        positions = [{"instrumentId": 1}]  # no investmentPct, netProfit, openRate
        metadata = {1: {"instrumentID": 1, "symbolFull": "BARE", "instrumentDisplayName": "Bare"}}
        result = _aggregate_positions(positions, metadata)

        bare = result["BARE"]
        assert bare["invest_pct"] == 0.0
        assert bare["total_profit"] == 0.0
        assert bare["avg_open_rate"] == 0.0
        assert bare["num_positions"] == 1

    def test_crypto_symbol_normalized(self):
        """Crypto symbols are normalized through ETORO_TO_YAHOO map."""
        positions = [
            {"instrumentId": 5, "investmentPct": 0.02, "netProfit": 100, "openRate": 60000}
        ]
        metadata = {5: {"instrumentID": 5, "symbolFull": "BTC", "instrumentDisplayName": "Bitcoin"}}
        result = _aggregate_positions(positions, metadata)

        assert "BTC-USD" in result
        assert result["BTC-USD"]["raw_etoro_symbol"] == "BTC"

    def test_result_has_expected_keys(self, sample_positions, sample_metadata):
        """Each aggregated entry has all expected keys."""
        result = _aggregate_positions(sample_positions, sample_metadata)
        expected_keys = {
            "symbol",
            "raw_etoro_symbol",
            "name",
            "instrument_id",
            "num_positions",
            "invest_pct",
            "avg_open_rate",
            "total_profit",
        }
        for entry in result.values():
            assert set(entry.keys()) == expected_keys


# ---------------------------------------------------------------------------
# reconcile_portfolio (orchestrator)
# ---------------------------------------------------------------------------


class TestReconcilePortfolio:
    """Tests for the main reconcile_portfolio function."""

    @pytest.fixture
    def mock_api(self, monkeypatch, sample_positions, sample_metadata):
        """Patch credentials + API calls; return the metadata for assertions."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: sample_positions,
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: sample_metadata,
        )
        return sample_metadata

    def test_in_sync_portfolio(self, tmp_path, mock_api):
        """When CSV and live match exactly, has_drift is False."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text(
            "TKR,NAME,BS,PRC,UP%\nAAPL,Apple,BUY,175,10\nMSFT,Microsoft,HOLD,330,5\n"
        )

        # Patch report output to tmp_path
        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["has_drift"] is False
        assert result["live_count"] == 2
        assert result["csv_count"] == 2
        assert result["matched"] == 2
        assert result["new_positions"] == []
        assert result["closed_positions"] == []
        assert "in sync" in result["summary"]

    def test_new_position_detected(self, tmp_path, mock_api):
        """Position on eToro but not in CSV is flagged as NEW."""
        csv_file = tmp_path / "portfolio.csv"
        # Only AAPL in CSV — MSFT is "new" on eToro
        csv_file.write_text("TKR,NAME,BS,PRC,UP%\nAAPL,Apple,BUY,175,10\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["has_drift"] is True
        assert len(result["new_positions"]) == 1
        assert result["new_positions"][0]["symbol"] == "MSFT"
        assert "NEW" in result["summary"]

    def test_closed_position_detected(self, tmp_path, mock_api):
        """Position in CSV but not on eToro is flagged as CLOSED."""
        csv_file = tmp_path / "portfolio.csv"
        # GOOGL in CSV but not in live eToro
        csv_file.write_text(
            "TKR,NAME,BS,PRC,UP%\n"
            "AAPL,Apple,BUY,175,10\n"
            "MSFT,Microsoft,HOLD,330,5\n"
            "GOOGL,Alphabet,SELL,140,-2\n"
        )

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["has_drift"] is True
        assert len(result["closed_positions"]) == 1
        assert result["closed_positions"][0]["symbol"] == "GOOGL"
        assert result["closed_positions"][0]["signal"] == "SELL"
        assert "CLOSED" in result["summary"]

    def test_both_new_and_closed(self, tmp_path, mock_api):
        """Simultaneous new + closed positions are both reported."""
        csv_file = tmp_path / "portfolio.csv"
        # AAPL matches, MSFT missing from CSV (new), GOOGL missing from eToro (closed)
        csv_file.write_text(
            "TKR,NAME,BS,PRC,UP%\nAAPL,Apple,BUY,175,10\nGOOGL,Alphabet,SELL,140,-2\n"
        )

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["has_drift"] is True
        assert len(result["new_positions"]) == 1
        assert len(result["closed_positions"]) == 1
        assert result["matched"] == 1
        assert "NEW" in result["summary"]
        assert "CLOSED" in result["summary"]

    def test_empty_csv_all_positions_new(self, tmp_path, mock_api):
        """Empty CSV means all live positions are NEW."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME,BS,PRC,UP%\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["csv_count"] == 0
        assert result["live_count"] == 2
        assert result["matched"] == 0
        assert len(result["new_positions"]) == 2
        assert result["has_drift"] is True

    def test_missing_csv_file(self, tmp_path, mock_api):
        """Non-existent CSV path treated as empty portfolio."""
        nonexistent = str(tmp_path / "does_not_exist.csv")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(nonexistent)

        assert result["csv_count"] == 0
        assert result["portfolio_csv_date"] == ""
        assert result["has_drift"] is True

    def test_default_csv_path_uses_expanduser(self, tmp_path, monkeypatch):
        """When no path given, default uses os.path.expanduser."""

        # Patch expanduser to point at tmp_path for both the CSV default and report dir
        def fake_expanduser(path):
            if "portfolio.csv" in path:
                return str(tmp_path / "portfolio.csv")
            return str(tmp_path / "reports")

        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler.os.path.expanduser", fake_expanduser
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: [],
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: {},
        )

        result = reconcile_portfolio()  # no argument
        assert result["live_count"] == 0

    def test_report_json_written(self, tmp_path, mock_api):
        """Reconciliation report is saved to the output directory."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\nAAPL,Apple\nMSFT,Microsoft\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        report_path = report_dir / "committee" / "reports" / "reconciliation.json"
        # The expanduser mock returns report_dir, but the code appends committee/reports
        # Let's check the actual directory the code writes to
        actual_report = tmp_path / "reports" / "reconciliation.json"
        # The code does: output_dir = expanduser("~/.weirdapps-trading/committee/reports")
        # With our mock, expanduser returns str(report_dir) for any input
        # So output_dir = str(report_dir), output_path = report_dir/reconciliation.json
        assert actual_report.exists()
        saved = json.loads(actual_report.read_text())
        assert saved["has_drift"] == result["has_drift"]
        assert saved["live_count"] == result["live_count"]

    def test_result_has_all_expected_keys(self, tmp_path, mock_api):
        """Result dict contains all documented keys."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\nAAPL,Apple\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        expected_keys = {
            "timestamp",
            "portfolio_csv_date",
            "live_count",
            "csv_count",
            "matched",
            "new_positions",
            "closed_positions",
            "summary",
            "has_drift",
        }
        assert set(result.keys()) == expected_keys

    def test_csv_mtime_populated(self, tmp_path, mock_api):
        """portfolio_csv_date is populated when the file exists."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\nAAPL,Apple\nMSFT,Microsoft\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        # Should be a non-empty date string
        assert result["portfolio_csv_date"] != ""
        # Rough format check: YYYY-MM-DD HH:MM
        assert len(result["portfolio_csv_date"]) >= 10

    def test_timestamp_is_iso_format(self, tmp_path, mock_api):
        """Result timestamp is valid ISO format."""
        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\nAAPL,Apple\nMSFT,Microsoft\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        from datetime import datetime

        # Should not raise
        datetime.fromisoformat(result["timestamp"])


class TestReconcilePortfolioNoLivePositions:
    """Edge case: eToro returns zero positions."""

    def test_all_csv_positions_closed(self, tmp_path, monkeypatch):
        """When eToro has zero positions, all CSV holdings appear as CLOSED."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: [],
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: {},
        )

        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME,BS\nAAPL,Apple,BUY\nMSFT,Microsoft,HOLD\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["live_count"] == 0
        assert result["csv_count"] == 2
        assert len(result["closed_positions"]) == 2
        assert result["has_drift"] is True

    def test_both_empty(self, tmp_path, monkeypatch):
        """Empty CSV + empty eToro = no drift."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: [],
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: {},
        )

        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert result["has_drift"] is False
        assert "in sync" in result["summary"]


class TestReconcileSummaryFormatting:
    """Tests for summary string construction edge cases."""

    def test_new_only_summary(self, tmp_path, monkeypatch):
        """Summary with only new positions does not mention CLOSED."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        positions = [{"instrumentId": 1, "investmentPct": 0.1, "netProfit": 0, "openRate": 100}]
        metadata = {
            1: {"instrumentID": 1, "symbolFull": "NEW_STOCK", "instrumentDisplayName": "New"}
        }
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: positions,
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: metadata,
        )

        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert "NEW" in result["summary"]
        assert "CLOSED" not in result["summary"]

    def test_closed_only_summary(self, tmp_path, monkeypatch):
        """Summary with only closed positions does not mention NEW."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: [],
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: {},
        )

        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\nAAPL,Apple\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        assert "CLOSED" in result["summary"]
        assert "NEW" not in result["summary"]


class TestNewPositionFields:
    """Verify the shape of items in new_positions and closed_positions."""

    def test_new_position_shape(self, tmp_path, monkeypatch):
        """Each new position dict has symbol/name/raw_etoro_symbol/invest_pct/num_positions/total_profit."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        positions = [{"instrumentId": 1, "investmentPct": 0.05, "netProfit": 25.0, "openRate": 150}]
        metadata = {1: {"instrumentID": 1, "symbolFull": "NVDA", "instrumentDisplayName": "Nvidia"}}
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: positions,
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: metadata,
        )

        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        new = result["new_positions"][0]
        assert new["symbol"] == "NVDA"
        assert new["name"] == "Nvidia"
        assert new["raw_etoro_symbol"] == "NVDA"
        assert new["invest_pct"] == 5.0
        assert new["num_positions"] == 1
        assert new["total_profit"] == 25.0

    def test_closed_position_shape(self, tmp_path, monkeypatch):
        """Each closed position dict has symbol/name/signal/last_price/upside."""
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._get_etoro_credentials",
            lambda: ("pk", "uk"),
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_live_portfolio",
            lambda ak, uk: [],
        )
        monkeypatch.setattr(
            "trade_modules.portfolio_reconciler._fetch_instrument_metadata",
            lambda ids, ak, uk: {},
        )

        csv_file = tmp_path / "portfolio.csv"
        csv_file.write_text("TKR,NAME,BS,PRC,UP%\nTSLA,Tesla,BUY,250.00,15.0\n")

        report_dir = tmp_path / "reports"
        with patch(
            "trade_modules.portfolio_reconciler.os.path.expanduser", return_value=str(report_dir)
        ):
            result = reconcile_portfolio(str(csv_file))

        closed = result["closed_positions"][0]
        assert closed["symbol"] == "TSLA"
        assert closed["name"] == "Tesla"
        assert closed["signal"] == "BUY"
        assert closed["last_price"] == "250.00"
        assert closed["upside"] == "15.0"


if __name__ == "__main__":
    pytest.main([__file__])

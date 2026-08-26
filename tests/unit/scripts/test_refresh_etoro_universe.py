"""Unit tests for scripts/refresh_etoro_universe.py."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load the script as a module via importlib (avoids reportMissingImports;
# matches the pattern used by tests/unit/scripts/test_validate_brief.py).
_SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "refresh_etoro_universe.py"
_spec = importlib.util.spec_from_file_location("refresh_etoro_universe", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
refresh_etoro_universe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refresh_etoro_universe)

# Public functions exposed by the module — aliased here so test bodies stay terse.
is_etorian_alias = refresh_etoro_universe.is_etorian_alias
normalize_symbol = refresh_etoro_universe.normalize_symbol
fix_share_classes = refresh_etoro_universe.fix_share_classes
dedupe_by_symbol = refresh_etoro_universe.dedupe_by_symbol
fetch_all_instruments = refresh_etoro_universe.fetch_all_instruments
write_universe_csv = refresh_etoro_universe.write_universe_csv
write_delta_log = refresh_etoro_universe.write_delta_log
get_credentials = refresh_etoro_universe.get_credentials
main = refresh_etoro_universe.main
MIN_INSTRUMENTS_THRESHOLD = refresh_etoro_universe.MIN_INSTRUMENTS_THRESHOLD
INSTRUMENTS_URL = refresh_etoro_universe.INSTRUMENTS_URL
is_junk_symbol = refresh_etoro_universe.is_junk_symbol
bloomberg_shape = refresh_etoro_universe.bloomberg_shape
audit_symbols = refresh_etoro_universe.audit_symbols
report_audit = refresh_etoro_universe.report_audit
_UNKNOWN_SUFFIX_FAIL_THRESHOLD = refresh_etoro_universe._UNKNOWN_SUFFIX_FAIL_THRESHOLD

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

    def test_ch_is_a_china_adr_and_is_stripped_not_remapped(self):
        """``.CH`` is eToro's marker for a Chinese ADR on NYSE/Nasdaq, so the bare US ticker
        IS the listing. It is NOT Switzerland — eToro spells Zurich ``.ZU`` (see the test
        above) and Yahoo spells it ``.SW``.

        Until 2026-08-11 this generator had no ``.CH`` rule at all, so the seven affected
        names reached ``yahoofinance/input/etoro.csv`` still suffixed (QIHU.CH, JMEI.CH,
        CYOU.CH, QUNR.CH, TCOM.CH, ATHM.CH, GSOL.CH), where ConfigManager's own table then
        mapped them to ``.SW`` — a Swiss symbol that does not exist for any of them.
        """
        assert normalize_symbol("TCOM.CH") == "TCOM"  # Trip.com Group ADR, Nasdaq
        assert normalize_symbol("ATHM.CH") == "ATHM"  # Autohome ADR, NYSE
        assert normalize_symbol("qunr.ch") == "QUNR"  # case-insensitive, like .US above
        # ...and the real Swiss path is untouched
        assert normalize_symbol("NESN.ZU") == "NESN.SW"

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
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(refresh_etoro_universe.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="kc-secret\n")
            api, user = get_credentials()
        assert api == "kc-secret"
        assert user == "kc-secret"

    def test_raises_when_both_missing(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(refresh_etoro_universe.subprocess, "run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            with pytest.raises(RuntimeError, match="Missing credentials"):
                get_credentials()


class TestFetchAllInstruments:
    def test_success_filters_stocks_and_etfs(self):
        api_response = {
            "instrumentDisplayDatas": [
                {
                    "instrumentID": 1,
                    "symbolFull": "EURUSD",
                    "instrumentDisplayName": "EUR/USD",
                    "instrumentTypeID": 1,
                    "exchangeID": 1,
                },
                {
                    "instrumentID": 1001,
                    "symbolFull": "AAPL",
                    "instrumentDisplayName": "Apple",
                    "instrumentTypeID": 5,
                    "exchangeID": 4,
                    "distributionType": 5,
                },
                {
                    "instrumentID": 2001,
                    "symbolFull": "SPY",
                    "instrumentDisplayName": "SPDR S&P 500",
                    "instrumentTypeID": 6,
                    "exchangeID": 4,
                    "distributionType": 5,
                },
            ]
        }
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: api_response)
            result = fetch_all_instruments("api", "user")
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["displayName"] == "Apple"
        assert result[0]["exchangeName"] == "Nasdaq"
        assert result[1]["symbol"] == "SPY"

    def test_uses_correct_headers(self):
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: {"instrumentDisplayDatas": []}
            )
            fetch_all_instruments("api-k", "user-k")
            call = mock_get.call_args
            assert call.args[0] == INSTRUMENTS_URL
            assert call.kwargs["headers"]["x-api-key"] == "api-k"
            assert call.kwargs["headers"]["x-user-key"] == "user-k"
            assert "User-Agent" in call.kwargs["headers"]
            assert "x-request-id" in call.kwargs["headers"]

    def test_retries_on_500(self):
        responses = [
            MagicMock(status_code=500, text="boom"),
            MagicMock(
                status_code=200,
                json=lambda: {
                    "instrumentDisplayDatas": [
                        {
                            "instrumentID": 1,
                            "symbolFull": "X",
                            "instrumentDisplayName": "X",
                            "instrumentTypeID": 5,
                            "exchangeID": 4,
                            "distributionType": 5,
                        },
                    ]
                },
            ),
        ]
        with (
            patch.object(refresh_etoro_universe.requests, "get") as mock_get,
            patch.object(refresh_etoro_universe.time, "sleep"),
        ):
            mock_get.side_effect = responses
            result = fetch_all_instruments("k", "u")
            assert result[0]["symbol"] == "X"
            assert mock_get.call_count == 2

    def test_raises_after_max_retries(self):
        with (
            patch.object(refresh_etoro_universe.requests, "get") as mock_get,
            patch.object(refresh_etoro_universe.time, "sleep"),
        ):
            mock_get.return_value = MagicMock(status_code=500, text="boom")
            with pytest.raises(RuntimeError, match="all 3 attempts failed"):
                fetch_all_instruments("k", "u", max_retries=3)
            assert mock_get.call_count == 3


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

        # Pad to 1000+ items to clear the safety threshold
        padded_items = sample_response["items"] + [
            {
                "instrumentId": 100000 + i,
                "symbol": f"PAD{i}",
                "displayName": f"Padding {i}",
                "assetClass": "Stocks",
                "exchangeName": "Nasdaq",
            }
            for i in range(1000)
        ]

        def fake_fetch(api_key, user_key, **kw):
            return padded_items

        with patch.object(refresh_etoro_universe, "fetch_all_instruments", side_effect=fake_fetch):
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
            return [{"symbol": "X", "displayName": "X", "exchangeName": "N"}] * 10

        with patch.object(refresh_etoro_universe, "fetch_all_instruments", side_effect=fake_fetch):
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


# ---------------------------------------------------------------------------
# 2026-08-11 suffix audit. Every case below was measured against the live eToro
# catalogue or probed against Yahoo before being written down; the bar counts in
# the docstrings are 1-month `yf.download` results from that date.
# ---------------------------------------------------------------------------


class TestVenueSuffixRemaps:
    """eToro's spelling of an exchange -> Yahoo's spelling of the same exchange."""

    def test_bloomberg_milan_and_london_codes(self):
        """.IM/.LN were remapped at FETCH time but not here, so the universe carried the
        eToro spelling. Probed: 28IA.IM 0 bars / 28IA.MI 22; SHLD.LN 0 / SHLD.L 22."""
        assert normalize_symbol("28IA.IM") == "28IA.MI"
        assert normalize_symbol("SHLD.LN") == "SHLD.L"

    def test_existing_venue_remaps_still_hold(self):
        assert normalize_symbol("XYZ.ASX") == "XYZ.AX"
        assert normalize_symbol("NESN.ZU") == "NESN.SW"
        assert normalize_symbol("HAVAS.NV") == "HAVAS.AS"
        assert normalize_symbol("CA366.LSB") == "CA366.LS"


class TestHongKongPadsBothWays:
    """The pad was one-sided (`len(base) > 4`), so only 5-digit codes were touched and every
    short code stayed unfetchable. Probed: 3.HK 0 bars / 0003.HK 21; 288.HK 0 / 0288.HK 21."""

    def test_short_codes_are_padded(self):
        assert normalize_symbol("3.HK") == "0003.HK"  # Hong Kong & China Gas
        assert normalize_symbol("12.HK") == "0012.HK"  # Henderson Land
        assert normalize_symbol("288.HK") == "0288.HK"  # WH Group

    def test_long_codes_still_trimmed_and_4_digit_untouched(self):
        assert normalize_symbol("00175.HK") == "0175.HK"
        assert normalize_symbol("0700.HK") == "0700.HK"
        assert normalize_symbol("9988.HK") == "9988.HK"

    def test_non_numeric_hk_stem_is_left_alone(self):
        """zfill must not be applied to a non-numeric stem."""
        assert normalize_symbol("ANT.HK") == "ANT.HK"


class TestDoubledDotIsCollapsedNotStripped:
    """YU..L is eToro writing an LSE TIDM that ends in a period. Probed: YU.L and RE.L both
    return 22 bars and neither collides with an existing row."""

    def test_collapse(self):
        assert normalize_symbol("YU..L") == "YU.L"
        assert normalize_symbol("RE..L") == "RE.L"

    def test_never_stripped_to_the_bare_stem(self):
        """FA..L -> FA and IX..L -> IX would merge into unrelated US tickers."""
        assert normalize_symbol("FA..L") == "FA.L"
        assert normalize_symbol("IX..L") == "IX.L"


class TestJunkSymbolCatchesWhatSuffixRulesCannot:
    def test_cvr_marker_on_the_stem_side_of_the_dot(self):
        """_is_junk_suffix reads only after the last dot, so it caught US.CVR1 and missed
        CVR.THS — the same marker on the other side."""
        assert normalize_symbol("CVR.THS") is None
        assert normalize_symbol("CVR.AVDL") is None
        assert normalize_symbol("US.CVR1") is None

    def test_short_numeric_suffix(self):
        """The old rule required len > 3, so WRTS.APRN.15 slipped through.

        Asserted on ``_is_junk_suffix`` DIRECTLY and on a stem that is not itself junk. Going
        through normalize_symbol("WRTS.APRN.15") passes either way, because WRTS is in
        _JUNK_STEMS and short-circuits first — a mutation reinstating `len(s) > 3` survived
        that version of this test, which is the definition of a test passing for the wrong
        reason.
        """
        assert refresh_etoro_universe._is_junk_suffix(".15")
        assert refresh_etoro_universe._is_junk_suffix(".20")
        assert refresh_etoro_universe._is_junk_suffix(".999")
        assert refresh_etoro_universe._is_junk_suffix(".15255")
        assert normalize_symbol("AAPL.15") is None  # stem is NOT junk; only the rule saves it
        assert normalize_symbol("WRTS.APRN.15") is None

    def test_space_delimited_placeholders(self):
        for sym in ("LSE CVR", "MTN DUMMY CVR", "CEPU ESCROW ASSET", "OSLO CVR1", "ETOR 4 C40"):
            assert is_junk_symbol(sym), sym

    def test_lifecycle_markers_in_all_four_spellings(self):
        for sym in ("SNDK_OLD", "BMPS-OLD", "MRK.DE_OLD", "STJ.L OLD"):
            assert is_junk_symbol(sym), sym

    def test_old_marker_never_matches_a_real_ticker(self):
        """\\bOLD$ cannot fire inside a word — GOLD is not a lifecycle marker."""
        assert not is_junk_symbol("GOLD")
        assert not is_junk_symbol("GOLD.L")

    def test_dormant_and_drm_placeholders(self):
        assert is_junk_symbol("DORMANT11232")
        assert is_junk_symbol("DRM.14731")

    def test_xetra_duplicate_artifacts(self):
        assert normalize_symbol("CEBP.DE11") is None
        assert normalize_symbol("IQQ6.D11") is None
        assert normalize_symbol("ICGB.DE22") is None


class TestCorporateActionPlaceholderPredicate:
    """The company==stem conjunction is LOAD-BEARING. Measured on the live universe file: the
    conjunction selects 63 junk rows, a bare ^CA\\d+ prefix rule selects 71 — the eight extra
    are real companies."""

    def test_placeholder_is_dropped(self):
        assert is_junk_symbol("CA141.L", "CA141")
        assert is_junk_symbol("CA307.OL", "CA307")

    def test_real_companies_sharing_the_prefix_survive(self):
        assert not is_junk_symbol("CA21", "Royal Dutch  Shell")
        assert not is_junk_symbol("CA8", "Meredith Holdings Corp.")
        assert not is_junk_symbol("CA12908", "Everfuel A/S")
        assert not is_junk_symbol("CA1.DE", "Circus SE")


class TestScandinavianHyphenIsVenueSpecific:
    """Stockholm and Copenhagen hyphenate a share-class letter; Helsinki and Oslo do NOT.
    Probed 2026-08-11: KESKOA.HE 21 bars / KESKO-A.HE 0 · ODFB.OL 21 / ODF-B.OL 0 ·
    ASSA-B.ST 21 / ASSAB.ST 0 (the control that keeps .ST in the set)."""

    def _syms(self, rows):
        return {r["symbol"] for r in fix_share_classes(rows)}

    def test_helsinki_is_not_hyphenated(self):
        rows = [
            {"symbol": "KESKOA.HE", "company": "Kesko Oyj", "exchange": "Helsinki"},
            {"symbol": "KESKOB.HE", "company": "Kesko Oyj B", "exchange": "Helsinki"},
        ]
        assert self._syms(rows) == {"KESKOA.HE", "KESKOB.HE"}

    def test_oslo_is_not_hyphenated(self):
        rows = [
            {"symbol": "VENDA.OL", "company": "Vend Marketplaces ASA", "exchange": "Oslo"},
            {"symbol": "VENDB.OL", "company": "Vend Marketplaces ASA", "exchange": "Oslo"},
        ]
        assert self._syms(rows) == {"VENDA.OL", "VENDB.OL"}

    def test_stockholm_and_copenhagen_still_are(self):
        rows = [
            {"symbol": "ASSAA.ST", "company": "ASSA ABLOY ser. A", "exchange": "Stockholm"},
            {"symbol": "ASSAB.ST", "company": "ASSA ABLOY ser. B", "exchange": "Stockholm"},
            {"symbol": "NOVOA.CO", "company": "Novo Nordisk A", "exchange": "Copenhagen"},
            {"symbol": "NOVOB.CO", "company": "Novo Nordisk B", "exchange": "Copenhagen"},
        ]
        assert self._syms(rows) == {"ASSA-A.ST", "ASSA-B.ST", "NOVO-A.CO", "NOVO-B.CO"}


class TestSuffixAudit:
    """The guard the module was missing. A suffix in none of the ruling sets used to be written
    to the universe in silence — that is how .CH stayed mis-mapped for months."""

    def _row(self, sym, company="X Corp", exchange="Nasdaq"):
        return {"symbol": sym, "company": company, "exchange": exchange}

    def test_ruled_suffixes_are_silent(self):
        rows = [
            self._row(s)
            for s in (
                "AAPL",
                "VOD.L",
                "SAP.DE",
                "RRL.ASX",
                "NOVN.ZU",
                "TCOM.CH",
                "AAPL.EUR",
                "STX.US",
                "CVR.THS",
                "BRK.B",
            )
        ]
        unknown, _ = audit_symbols(rows)
        assert unknown == {}

    def test_an_unruled_suffix_is_reported(self):
        rows = [self._row("ABC.XYZ")]
        unknown, _ = audit_symbols(rows)
        assert list(unknown) == [".XYZ"]
        assert unknown[".XYZ"] == ["ABC.XYZ"]

    def test_below_threshold_warns_but_does_not_abort(self):
        rows = [self._row(f"A{i}.XYZ") for i in range(_UNKNOWN_SUFFIX_FAIL_THRESHOLD - 1)]
        unknown, review = audit_symbols(rows)
        assert report_audit(unknown, review) is False

    def test_at_threshold_aborts(self):
        """A new exchange arrives in bulk. That must stop the refresh, not scroll past."""
        rows = [self._row(f"A{i}.XYZ") for i in range(_UNKNOWN_SUFFIX_FAIL_THRESHOLD)]
        unknown, review = audit_symbols(rows)
        assert report_audit(unknown, review) is True

    def test_the_dot_ch_regression_would_now_be_caught(self):
        """The whole point. Strip .CH out of the ruling sets and the audit must shout."""
        original = refresh_etoro_universe._STRIP_SUFFIXES
        refresh_etoro_universe._STRIP_SUFFIXES = (".US",)
        try:
            rows = [self._row(s) for s in ("TCOM.CH", "ATHM.CH", "QUNR.CH")]
            unknown, review = audit_symbols(rows)
            assert ".CH" in unknown
            assert report_audit(unknown, review) is True
        finally:
            refresh_etoro_universe._STRIP_SUFFIXES = original

    def test_junk_never_reaches_the_audit(self):
        unknown, review = audit_symbols([self._row("DORMANT11232"), self._row("LSE CVR")])
        assert unknown == {}
        assert review == {}

    def test_bloomberg_form_is_reviewed_not_dropped_and_not_unknown(self):
        """SRT3 GY is Sartorius Vorzug — a real security in Bloomberg notation. It needs a
        per-venue remap, which is a reviewed decision, so it warns rather than vanishing."""
        assert bloomberg_shape("SRT3 GY")
        assert bloomberg_shape("HAWK US")
        assert not bloomberg_shape("MTN DUMMY CVR")
        unknown, review = audit_symbols([self._row("SRT3 GY", "SARTORIUS AG-VORZUG", "FRA")])
        assert unknown == {}
        assert any("bloomberg" in k for k in review)

    def test_bare_symbol_on_a_non_us_venue_is_reviewed(self):
        """A bare symbol means a US listing. When eToro says the venue is London, passthrough
        would aim a US ticker at a foreign company — report it, never guess."""
        _, review = audit_symbols([self._row("SJNK", "SPDR Bloomberg Short Term", "LSE")])
        assert any("non-US venue" in k for k in review)

    def test_bare_us_symbols_are_not_reviewed(self):
        for venue in ("Nasdaq", "NYSE", "CBOE", "Chicago Board Options Exchange", "OTC Markets"):
            _, review = audit_symbols([self._row("AAPL", "Apple Inc", venue)])
            assert review == {}, venue


class TestCurrencyLineShapeDecidesStripVsDrop:
    """Both shapes exist in the live file and they need opposite handling."""

    def test_venue_qualified_stem_is_stripped_and_recovered(self):
        """GLB.L.GBX / KSP.L.GBX have NO bare GLB.L / KSP.L row, so stripping the pence unit
        is the only thing that produces the London listing at all."""
        assert normalize_symbol("GLB.L.GBX", "Glanbia Plc") == "GLB.L"
        assert normalize_symbol("KSP.L.GBX", "Kingspan Group Plc") == "KSP.L"

    def test_bare_stem_is_dropped_not_collided(self):
        """All ten .EUR rows are Frankfurt duplicates of a Nasdaq row that already exists.
        Stripping would collide, and dedupe could keep the one labelled FRA."""
        assert normalize_symbol("AAPL.EUR", "Apple") is None
        assert normalize_symbol("NVDA.EUR", "NVIDIA Corporation") is None

    def test_a_currency_word_inside_a_real_ticker_is_untouched(self):
        assert normalize_symbol("EURO.L", "Euromoney") == "EURO.L"
        assert normalize_symbol("IEUR", "iShares Core MSCI Europe") == "IEUR"


class TestPlaceholderCodesBeyondTheCAFamily:
    """The row's NAME is its own internal code. Same shape as CA141, different prefixes."""

    def test_ipo_and_ca_ops_placeholders_are_dropped(self):
        assert normalize_symbol("IPO56.L", "IPO56") is None
        assert normalize_symbol("IPO100.L", "IPO100") is None
        assert normalize_symbol("IPO108.AX", "IPO108") is None
        assert normalize_symbol("CA.OPS31.L", "CA.Ops31") is None
        assert normalize_symbol("IPO1", "IPO1") is None

    def test_ip_group_is_not_a_placeholder(self):
        """IPO.L is IP Group PLC. A prefix rule on the SYMBOL would delete it; the rule keys
        on the company name being the code, so it survives."""
        assert normalize_symbol("IPO.L", "IP Group PLC") == "IPO.L"


class TestOpenableFilter:
    """eToro's ``/market-data/instruments`` returns everything it DISPLAYS, which is a superset
    of what it will let an account OPEN. Measured 2026-08-26: 2,555 of 14,332 stocks+ETFs are a
    close-only block — holdable and sellable, not buyable — and they are indistinguishable by
    symbol, name or exchange. They reached the model as BUY recommendations that could not be
    placed; three of one day's ten buys were dead this way.

    ``distributionType`` separates them cleanly. Validated against eToro's own account-eligibility
    endpoint at n=400: ``None`` -> 199/200 answered ``allow_open=False``; ``5`` -> 162/200 True.
    """

    def test_the_marker_is_the_KEYS_ABSENCE_not_its_value(self):
        """Measured on the live payload: the close-only block OMITS ``distributionType``; every
        openable row carries it as 5. The first version of this function tested the VALUE and
        dropped nothing at all, because ``dict.get`` reports absent and null identically."""
        assert refresh_etoro_universe.is_openable({"distributionType": 5}) is True
        assert refresh_etoro_universe.is_openable({}) is False
        assert refresh_etoro_universe.is_openable({"symbolFull": "ACL.ASX"}) is False

    def test_an_UNRECOGNISED_value_is_kept(self):
        """It asks whether eToro said anything at all, so a third value appearing tomorrow reads
        as openable rather than silently shrinking the universe."""
        assert refresh_etoro_universe.is_openable({"distributionType": 7}) is True
        assert refresh_etoro_universe.is_openable({"distributionType": "whatever"}) is True

    def test_an_explicit_NULL_is_still_openable_and_that_is_deliberate(self):
        """eToro does not currently send null — it omits the key. If it ever starts, that is a
        NEW shape nobody has measured, and the permissive direction is the safe one: the
        fail-closed control is the per-order broker refusal, not this generator."""
        assert refresh_etoro_universe.is_openable({"distributionType": None}) is True

    def test_fetch_drops_the_close_only_rows(self):
        api_response = {
            "instrumentDisplayDatas": [
                {
                    "instrumentID": 1001,
                    "symbolFull": "AAPL",
                    "instrumentDisplayName": "Apple",
                    "instrumentTypeID": 5,
                    "exchangeID": 4,
                    "distributionType": 5,
                },
                {
                    "instrumentID": 1001690,
                    "symbolFull": "ACL.ASX",
                    "instrumentDisplayName": "Australian Clinical Labs",
                    "instrumentTypeID": 5,
                    "exchangeID": 31,
                },  # no distributionType: the real shape
            ]
        }
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: api_response)
            result = fetch_all_instruments("api", "user")
        assert [r["symbol"] for r in result] == ["AAPL"], (
            "ACL.ASX is real, listed and displayed — and eToro answered allow_open=False for it"
        )

    def test_it_REFUSES_rather_than_writing_an_empty_universe(self):
        """If every row fails the test, the field changed meaning — eToro did not delist its
        whole catalogue. Raising keeps the last good CSV on disk."""
        api_response = {
            "instrumentDisplayDatas": [
                {
                    "instrumentID": i,
                    "symbolFull": f"X{i}",
                    "instrumentDisplayName": "x",
                    "instrumentTypeID": 5,
                    "exchangeID": 4,
                }
                for i in range(5)
            ]
        }
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: api_response)
            with pytest.raises(RuntimeError, match="excluded ALL"):
                fetch_all_instruments("api", "user")

    def test_it_REFUSES_an_implausibly_large_drop(self):
        """~18% is the measured shape. A ceiling catches the field inverting, which would gut the
        universe without emptying it — the failure a floor on the absolute count cannot see."""
        # A PLAUSIBLE CATALOGUE SIZE, because the ceiling is a ratio and a ratio of two rows
        # is noise. `MIN_INSTRUMENTS_THRESHOLD` is what refuses a payload smaller than this.
        rows = [
            {
                "instrumentID": i,
                "symbolFull": f"X{i}",
                "instrumentDisplayName": "x",
                "instrumentTypeID": 5,
                "exchangeID": 4,
                **({} if i < 900 else {"distributionType": 5}),
            }
            for i in range(1000)
        ]
        with patch.object(refresh_etoro_universe.requests, "get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200, json=lambda: {"instrumentDisplayDatas": rows}
            )
            with pytest.raises(RuntimeError, match="above the"):
                fetch_all_instruments("api", "user")

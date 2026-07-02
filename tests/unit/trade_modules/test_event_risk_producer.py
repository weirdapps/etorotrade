"""Unit tests for scripts/event_risk_producer.py — pure functions only (no IO).

Tests cover:
  classify_event_risk — any-warning mode, block_types mode, string vs dict entries,
                        upper-case + sort + dedup output, empty-list skip.
  write_event_risk    — round-trip correctness, empty list, creates missing dirs,
                        no stray .tmp file left behind.
"""

from __future__ import annotations

import json

from scripts.event_risk_producer import classify_event_risk, write_event_risk

# ---------------------------------------------------------------------------
# classify_event_risk — any-warning mode (block_types=None)
# ---------------------------------------------------------------------------


def test_classify_any_warning_flags_non_empty():
    warnings = {
        "AAPL": [{"type": "Lawsuit", "date": "2024-01-01", "detail": "class action"}],
        "MSFT": [],
        "TSLA": [{"type": "Regulatory", "date": "2024-02-01", "detail": "SEC probe"}],
    }
    result = classify_event_risk(warnings)
    assert result == ["AAPL", "TSLA"]


def test_classify_any_warning_skips_empty_list():
    warnings = {"NVDA": [], "GOOG": []}
    result = classify_event_risk(warnings)
    assert result == []


def test_classify_any_warning_returns_sorted_uppercase():
    warnings = {
        "tsla": [{"type": "Lawsuit"}],
        "aapl": [{"type": "SEC"}],
        "msft": [{"type": "Probe"}],
    }
    result = classify_event_risk(warnings)
    assert result == ["AAPL", "MSFT", "TSLA"]


def test_classify_any_warning_deduplicates_tickers():
    # Ticker appears with mixed case keys — both refer to same ticker
    warnings = {
        "AAPL": [{"type": "Lawsuit"}],
        "aapl": [{"type": "SEC"}],
    }
    result = classify_event_risk(warnings)
    assert result == ["AAPL"]


def test_classify_any_warning_handles_empty_dict():
    assert classify_event_risk({}) == []


def test_classify_any_warning_handles_none_warnings_value():
    # None as warnings value treated as absent / no warnings
    warnings = {"AAPL": None, "TSLA": [{"type": "SEC"}]}
    result = classify_event_risk(warnings)
    assert result == ["TSLA"]


# ---------------------------------------------------------------------------
# classify_event_risk — block_types mode
# ---------------------------------------------------------------------------


def test_classify_block_types_matches_case_insensitive():
    warnings = {
        "AAPL": [{"type": "LAWSUIT", "detail": "class action"}],
        "MSFT": [{"type": "Regulatory", "detail": "fine"}],
        "TSLA": [{"type": "EarningsMiss", "detail": "not material"}],
    }
    result = classify_event_risk(warnings, block_types=["lawsuit", "regulatory"])
    assert result == ["AAPL", "MSFT"]


def test_classify_block_types_substring_match():
    # "regul" should match "Regulatory"
    warnings = {
        "AAPL": [{"type": "RegulatoryAction", "detail": "fine"}],
        "MSFT": [{"type": "Lawsuit", "detail": "claim"}],
    }
    result = classify_event_risk(warnings, block_types=["regul"])
    assert result == ["AAPL"]


def test_classify_block_types_non_matching_type_not_flagged():
    warnings = {
        "AAPL": [{"type": "EarningsMiss"}],
    }
    result = classify_event_risk(warnings, block_types=["lawsuit"])
    assert result == []


def test_classify_block_types_multiple_warnings_one_match_flags_ticker():
    warnings = {
        "AAPL": [
            {"type": "EarningsMiss"},
            {"type": "Lawsuit", "detail": "class action"},
        ],
    }
    result = classify_event_risk(warnings, block_types=["lawsuit"])
    assert result == ["AAPL"]


def test_classify_block_types_robust_to_string_entries():
    # Entries may be plain strings instead of dicts
    warnings = {
        "AAPL": ["lawsuit filed", "regulatory probe"],
        "MSFT": ["earnings miss"],
    }
    result = classify_event_risk(warnings, block_types=["lawsuit"])
    assert result == ["AAPL"]


def test_classify_block_types_dict_warningtype_field():
    # TipRanks may use 'warningType' or 'name' instead of 'type'
    warnings = {
        "AAPL": [{"warningType": "Litigation", "detail": "class action"}],
    }
    result = classify_event_risk(warnings, block_types=["litigation"])
    assert result == ["AAPL"]


def test_classify_block_types_dict_name_field():
    warnings = {
        "TSLA": [{"name": "SEC Investigation", "detail": "securities fraud"}],
    }
    result = classify_event_risk(warnings, block_types=["sec"])
    assert result == ["TSLA"]


def test_classify_block_types_empty_block_types_flags_nothing():
    # An empty list of block_types should flag nothing (all non-matching)
    warnings = {
        "AAPL": [{"type": "Lawsuit"}],
    }
    result = classify_event_risk(warnings, block_types=[])
    assert result == []


# ---------------------------------------------------------------------------
# write_event_risk — round-trip, empty, parent dirs, no .tmp leftovers
# ---------------------------------------------------------------------------


def test_write_event_risk_round_trip(tmp_path):
    out = tmp_path / "event_risk.json"
    tickers = ["TSLA", "AAPL", "MSFT"]
    returned_path = write_event_risk(tickers, path=str(out))
    assert returned_path == str(out)
    data = json.loads(out.read_text())
    assert data == sorted({"AAPL", "MSFT", "TSLA"})


def test_write_event_risk_empty_list(tmp_path):
    out = tmp_path / "empty.json"
    write_event_risk([], path=str(out))
    data = json.loads(out.read_text())
    assert data == []


def test_write_event_risk_none_tickers(tmp_path):
    out = tmp_path / "none.json"
    write_event_risk(None, path=str(out))
    data = json.loads(out.read_text())
    assert data == []


def test_write_event_risk_creates_missing_parent_dirs(tmp_path):
    out = tmp_path / "deep" / "nested" / "dir" / "event_risk.json"
    write_event_risk(["AAPL"], path=str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert data == ["AAPL"]


def test_write_event_risk_no_tmp_file_left_behind(tmp_path):
    out = tmp_path / "event_risk.json"
    write_event_risk(["AAPL"], path=str(out))
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_write_event_risk_deduplicates_and_uppercases(tmp_path):
    out = tmp_path / "event_risk.json"
    write_event_risk(["tsla", "TSLA", "aapl"], path=str(out))
    data = json.loads(out.read_text())
    assert data == ["AAPL", "TSLA"]


def test_write_event_risk_returns_path_written(tmp_path):
    out = tmp_path / "event_risk.json"
    result = write_event_risk(["AAPL"], path=str(out))
    assert result == str(out)

"""Unit tests for scripts/v3_fundamentals_update.py — the catch-up (self-healing
incremental) --since computation used by the scheduled daily fundamentals sync.

Loaded via importlib to match tests/unit/scripts/test_refresh_etoro_universe.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT_PATH = Path(__file__).parent.parent.parent.parent / "scripts" / "v3_fundamentals_update.py"
_spec = importlib.util.spec_from_file_location("v3_fundamentals_update", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
v3_fundamentals_update = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3_fundamentals_update)

_catch_up_since = v3_fundamentals_update._catch_up_since
append_records = v3_fundamentals_update.append_records


def _seed(store: str, last_datekey: str) -> None:
    append_records(
        pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "datekey": last_datekey,
                    "reportperiod": "2026-03-31",
                    "assets": 1.0,
                }
            ]
        ),
        store_path=store,
    )


class TestCatchUpSince:
    def test_uses_store_last_minus_buffer(self, tmp_path):
        store = str(tmp_path / "f.parquet")
        _seed(store, "2026-07-20")
        got = _catch_up_since(buffer_days=90, store_path=store)
        assert got == (pd.Timestamp("2026-07-20") - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    def test_default_buffer_heals_multiweek_outage(self, tmp_path):
        store = str(tmp_path / "f.parquet")
        _seed(store, "2026-07-20")
        # The default buffer must reach well before the last datekey so a missed run
        # (node offline for weeks) is filled on the next run — not left as a hole.
        got = pd.Timestamp(_catch_up_since(store_path=store))
        assert got < pd.Timestamp("2026-07-20") - pd.Timedelta(days=30)

    def test_empty_store_returns_backfill_default(self, tmp_path):
        store = str(tmp_path / "none.parquet")
        assert _catch_up_since(store_path=store) == "2015-01-01"

    def test_empty_store_custom_backfill(self, tmp_path):
        store = str(tmp_path / "none.parquet")
        assert _catch_up_since(backfill_since="2010-06-01", store_path=store) == "2010-06-01"


class TestMainCatchUp:
    def test_catch_up_overrides_since_from_store(self, monkeypatch):
        """`--catch-up --from-universe` fetches the universe at the store-derived since."""
        monkeypatch.setenv("NDL_API_KEY", "k")
        captured: dict = {}

        def fake_fetch(key, tickers, since):
            captured["since"] = since
            captured["tickers"] = tickers
            return pd.DataFrame()

        monkeypatch.setattr(v3_fundamentals_update, "_universe_tickers", lambda: ["AAPL", "MSFT"])
        monkeypatch.setattr(
            v3_fundamentals_update,
            "store_coverage",
            lambda *a, **k: {
                "last": "2026-07-20",
                "n_rows": 1,
                "n_tickers": 1,
                "first": "2016-01-01",
            },
        )
        monkeypatch.setattr(v3_fundamentals_update, "_fetch", fake_fetch)
        monkeypatch.setattr(v3_fundamentals_update, "append_records", lambda df: 0)
        monkeypatch.setattr(
            "sys.argv", ["v3_fundamentals_update.py", "--from-universe", "--catch-up"]
        )

        v3_fundamentals_update.main()

        assert captured["tickers"] == ["AAPL", "MSFT"]
        assert captured["since"] == (pd.Timestamp("2026-07-20") - pd.Timedelta(days=90)).strftime(
            "%Y-%m-%d"
        )

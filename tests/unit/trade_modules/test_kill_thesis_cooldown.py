"""
M10: Kill-thesis 4-week cooldown — CIO v36 Empirical Refoundation.

Today: kill_thesis triggers a -15 conviction penalty (committee_synthesis
line 3121-ish). Position sizing then re-enters at next run with a smaller
multiplier (0.85 → 0.75) — this is a leak. The kill-thesis is supposed to
mean "we were materially wrong about this name" but the system can re-buy
within days at 88% size.

M10: After a kill-thesis fires on TICKER, set a per-ticker cooldown for
28 days. During cooldown, position size for that ticker is forced to 0
regardless of conviction.

Cooldown state lives in a JSONL log so it persists across runs:
~/.weirdapps-trading/committee/kill_thesis_cooldowns.jsonl
"""

import json
from datetime import datetime, timedelta


class TestRecordKillThesis:
    def test_records_trigger_with_timestamp(self, tmp_path):
        from trade_modules.kill_thesis_cooldown import record_kill_thesis

        log = tmp_path / "kt.jsonl"
        record_kill_thesis(
            "AAPL",
            reason="VIX>40 triggered",
            log_path=log,
            now=datetime(2026, 5, 1, 12, 0, 0),
        )
        with open(log) as f:
            entries = [json.loads(line) for line in f]
        assert len(entries) == 1
        assert entries[0]["ticker"] == "AAPL"
        assert entries[0]["timestamp"].startswith("2026-05-01")
        assert "VIX>40" in entries[0]["reason"]


class TestIsInCooldown:
    def test_no_log_means_not_in_cooldown(self, tmp_path):
        from trade_modules.kill_thesis_cooldown import is_in_cooldown

        assert is_in_cooldown("AAPL", log_path=tmp_path / "missing.jsonl") is False

    def test_recent_trigger_blocks_for_28_days(self, tmp_path):
        from trade_modules.kill_thesis_cooldown import (
            is_in_cooldown,
            record_kill_thesis,
        )

        log = tmp_path / "kt.jsonl"
        trigger = datetime(2026, 5, 1, 12, 0, 0)
        record_kill_thesis("AAPL", reason="x", log_path=log, now=trigger)

        # Day 1 after trigger → still in cooldown
        assert (
            is_in_cooldown(
                "AAPL",
                log_path=log,
                now=trigger + timedelta(days=1),
            )
            is True
        )
        # Day 27 → still in cooldown
        assert (
            is_in_cooldown(
                "AAPL",
                log_path=log,
                now=trigger + timedelta(days=27),
            )
            is True
        )
        # Day 28 → still in cooldown (boundary)
        assert (
            is_in_cooldown(
                "AAPL",
                log_path=log,
                now=trigger + timedelta(days=28),
            )
            is True
        )
        # Day 29 → cooldown expired
        assert (
            is_in_cooldown(
                "AAPL",
                log_path=log,
                now=trigger + timedelta(days=29),
            )
            is False
        )

    def test_other_ticker_not_blocked(self, tmp_path):
        from trade_modules.kill_thesis_cooldown import (
            is_in_cooldown,
            record_kill_thesis,
        )

        log = tmp_path / "kt.jsonl"
        record_kill_thesis(
            "AAPL",
            reason="x",
            log_path=log,
            now=datetime(2026, 5, 1),
        )
        assert (
            is_in_cooldown(
                "MSFT",
                log_path=log,
                now=datetime(2026, 5, 5),
            )
            is False
        )

    def test_latest_trigger_resets_cooldown_window(self, tmp_path):
        """If kill-thesis re-fires, cooldown resets from the latest trigger."""
        from trade_modules.kill_thesis_cooldown import (
            is_in_cooldown,
            record_kill_thesis,
        )

        log = tmp_path / "kt.jsonl"
        record_kill_thesis(
            "AAPL",
            reason="x",
            log_path=log,
            now=datetime(2026, 5, 1),
        )
        record_kill_thesis(
            "AAPL",
            reason="y",
            log_path=log,
            now=datetime(2026, 5, 20),
        )
        # 28 days from May 1 = May 29 (would be expired)
        # 28 days from May 20 = June 17 (still in cooldown on June 1)
        assert (
            is_in_cooldown(
                "AAPL",
                log_path=log,
                now=datetime(2026, 6, 1),
            )
            is True
        )
        # June 18 → expired
        assert (
            is_in_cooldown(
                "AAPL",
                log_path=log,
                now=datetime(2026, 6, 18),
            )
            is False
        )


class TestEnrichRespectsCooldown:
    """enrich_with_position_sizes should zero out positions in cooldown."""

    def test_cooldown_zeroes_buy_position(self, tmp_path, monkeypatch):
        from trade_modules.committee_synthesis import enrich_with_position_sizes
        from trade_modules.kill_thesis_cooldown import record_kill_thesis

        # Set the log path used by synthesis to our tmp file
        log = tmp_path / "kt.jsonl"
        record_kill_thesis(
            "AAPL",
            reason="x",
            log_path=log,
            now=datetime(2026, 5, 1),
        )

        monkeypatch.setattr(
            "trade_modules.committee_synthesis.KILL_THESIS_COOLDOWN_LOG",
            log,
        )

        conc = [
            {"ticker": "AAPL", "action": "BUY", "conviction": 70, "market_cap": "MEGA"},
        ]
        # Provide explicit "now" via env (synthesis must support it for tests)
        monkeypatch.setattr(
            "trade_modules.committee_synthesis._cooldown_now",
            lambda: datetime(2026, 5, 5),
        )
        enrich_with_position_sizes(conc, portfolio_value=400_000)

        assert conc[0]["suggested_size_usd"] == 0
        assert conc[0].get("kill_thesis_cooldown") is True

    def test_no_cooldown_allows_normal_sizing(self, tmp_path, monkeypatch):
        from trade_modules.committee_synthesis import enrich_with_position_sizes

        log = tmp_path / "empty.jsonl"
        monkeypatch.setattr(
            "trade_modules.committee_synthesis.KILL_THESIS_COOLDOWN_LOG",
            log,
        )

        conc = [
            {"ticker": "MSFT", "action": "BUY", "conviction": 70, "market_cap": "MEGA"},
        ]
        enrich_with_position_sizes(conc, portfolio_value=400_000)
        assert conc[0]["suggested_size_usd"] > 0
        assert conc[0].get("kill_thesis_cooldown") is not True

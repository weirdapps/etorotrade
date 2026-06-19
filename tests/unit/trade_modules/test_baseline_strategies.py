"""Tests for baseline_strategies.py — baseline return computation and stats."""

import json
import tempfile
from pathlib import Path

import numpy as np

from trade_modules.baseline_strategies import _stats, compute_baselines


class TestStats:
    """Tests for _stats helper function."""

    def test_empty_returns(self):
        result = _stats([], "empty")
        assert result["name"] == "empty"
        assert result["count"] == 0

    def test_single_return(self):
        result = _stats([5.0], "single")
        assert result["count"] == 1
        assert result["mean_return"] == 5.0
        assert result["median_return"] == 5.0
        assert result["std_return"] == 0.0
        assert result["hit_rate"] == 100.0
        assert result["min_return"] == 5.0
        assert result["max_return"] == 5.0
        assert result["sharpe_proxy"] == 0.0  # ddof=1 undefined for n=1

    def test_known_returns(self):
        returns = [10.0, -5.0, 15.0, -2.0, 8.0]
        result = _stats(returns, "known")
        arr = np.array(returns)
        assert result["count"] == 5
        assert result["mean_return"] == round(float(np.mean(arr)), 2)
        assert result["median_return"] == round(float(np.median(arr)), 2)
        assert result["hit_rate"] == 60.0  # 3 out of 5 positive
        assert result["min_return"] == -5.0
        assert result["max_return"] == 15.0

    def test_all_negative_returns(self):
        returns = [-3.0, -7.0, -1.0]
        result = _stats(returns, "neg")
        assert result["hit_rate"] == 0.0
        assert result["mean_return"] < 0

    def test_all_positive_returns(self):
        returns = [1.0, 2.0, 3.0]
        result = _stats(returns, "pos")
        assert result["hit_rate"] == 100.0
        assert result["mean_return"] == 2.0


class TestComputeBaselines:
    """Tests for compute_baselines function."""

    def test_nonexistent_file_returns_no_data(self):
        result = compute_baselines(Path("/nonexistent/signal_log.jsonl"))
        assert result["status"] == "no_data"
        assert result["baselines"] == {}

    def test_empty_file_returns_no_data(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            tmp_path = Path(f.name)
        try:
            result = compute_baselines(tmp_path)
            assert result["status"] == "no_data"
        finally:
            tmp_path.unlink()

    def test_valid_entries_produce_baselines(self):
        entries = [
            {
                "signal": "B",
                "return_7d": 3.0,
                "return_30d": 8.0,
                "pct_52w_high": 0.95,
                "exret": 12.0,
            },
            {
                "signal": "B",
                "return_7d": -1.0,
                "return_30d": 5.0,
                "pct_52w_high": 0.90,
                "exret": 10.0,
            },
            {
                "signal": "S",
                "return_7d": -4.0,
                "return_30d": -6.0,
                "pct_52w_high": 0.60,
                "exret": 2.0,
            },
            {
                "signal": "H",
                "return_7d": 1.0,
                "return_30d": 2.0,
                "pct_52w_high": 0.80,
                "exret": 5.0,
            },
            {
                "signal": "B",
                "return_7d": 2.0,
                "return_30d": 4.0,
                "pct_52w_high": 0.85,
                "exret": 8.0,
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            tmp_path = Path(f.name)
        try:
            result = compute_baselines(tmp_path, horizons=(7, 30))
            assert result["status"] == "ok"
            assert "T+7" in result["baselines"]
            assert "T+30" in result["baselines"]

            t7 = result["baselines"]["T+7"]
            assert t7["buy_signals"]["count"] == 3
            assert t7["sell_signals"]["count"] == 1
            assert t7["hold_signals"]["count"] == 1
            assert t7["all_universe"]["count"] == 5
        finally:
            tmp_path.unlink()

    def test_missing_return_fields_skipped(self):
        """Entries without the return field for a horizon are excluded."""
        entries = [
            {"signal": "B", "return_7d": 3.0},  # no return_30d
            {"signal": "B", "return_30d": 5.0},  # no return_7d
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            tmp_path = Path(f.name)
        try:
            result = compute_baselines(tmp_path, horizons=(7, 30))
            assert result["status"] == "ok"
            assert result["baselines"]["T+7"]["buy_signals"]["count"] == 1
            assert result["baselines"]["T+30"]["buy_signals"]["count"] == 1
        finally:
            tmp_path.unlink()

    def test_invalid_json_lines_skipped(self):
        """Malformed JSON lines should be skipped without error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"signal": "B", "return_7d": 3.0}\n')
            f.write("not valid json\n")
            f.write('{"signal": "S", "return_7d": -2.0}\n')
            tmp_path = Path(f.name)
        try:
            result = compute_baselines(tmp_path, horizons=(7,))
            assert result["status"] == "ok"
            assert result["baselines"]["T+7"]["all_universe"]["count"] == 2
        finally:
            tmp_path.unlink()

"""
M11: Per-modifier T+30 calibrator tests.

The calibrator script reads concordance history + price data, computes
Spearman ρ(modifier_value, T+30_alpha) per modifier, and classifies each
modifier as PREDICTIVE / WEAK / SHADOW / DROP / INSUFFICIENT_DATA.

The current synthesis pipeline declares 22 active modifiers based on
literature review (committee_synthesis.py:138-161). The calibrator
replaces that with empirical evidence from realized 30-day forward
returns on accumulated concordance history.
"""

import json

import pytest


@pytest.fixture
def tmp_history(tmp_path):
    """Create three days of synthetic concordance history."""
    hist_dir = tmp_path / "history"
    hist_dir.mkdir()

    # Synthetic data: a "good" modifier perfectly correlated with future α,
    # a "noise" modifier with no relationship, and an "inverted" modifier.
    rows_by_date = {
        "2026-01-01": [
            {
                "ticker": "AAA",
                "conviction_waterfall": {"good_mod": +5, "noise_mod": +3, "inv_mod": +5},
            },
            {
                "ticker": "BBB",
                "conviction_waterfall": {"good_mod": -5, "noise_mod": -3, "inv_mod": -5},
            },
            {
                "ticker": "CCC",
                "conviction_waterfall": {"good_mod": +5, "noise_mod": +3, "inv_mod": -5},
            },
            {
                "ticker": "DDD",
                "conviction_waterfall": {"good_mod": -5, "noise_mod": -3, "inv_mod": +5},
            },
        ],
    }
    for date, rows in rows_by_date.items():
        with open(hist_dir / f"concordance-{date}.json", "w") as f:
            json.dump({"date": date, "concordance": rows}, f)

    return hist_dir


class TestExtractModifierObservations:
    def test_extracts_modifier_values_per_ticker_per_date(self, tmp_history):
        from scripts.calibrate_modifiers_t30 import extract_modifier_observations

        obs = extract_modifier_observations(tmp_history)
        # 4 stocks × 3 modifiers = 12 observations
        assert len(obs) == 12
        # Each observation has (ticker, date, modifier, value)
        sample = obs[0]
        assert "ticker" in sample
        assert "date" in sample
        assert "modifier" in sample
        assert "value" in sample

    def test_skips_concordance_without_waterfall(self, tmp_path):
        from scripts.calibrate_modifiers_t30 import extract_modifier_observations

        hist_dir = tmp_path / "history"
        hist_dir.mkdir()
        # Concordance entries with no conviction_waterfall field (legacy format)
        with open(hist_dir / "concordance-2026-01-01.json", "w") as f:
            json.dump(
                {
                    "date": "2026-01-01",
                    "concordance": [
                        {"ticker": "AAA", "conviction": 60},  # no waterfall
                    ],
                },
                f,
            )

        assert extract_modifier_observations(hist_dir) == []

    def test_handles_dict_format_concordance(self, tmp_path):
        """Some files store concordance as {ticker: {...}} not list."""
        from scripts.calibrate_modifiers_t30 import extract_modifier_observations

        hist_dir = tmp_path / "history"
        hist_dir.mkdir()
        with open(hist_dir / "concordance-2026-01-01.json", "w") as f:
            json.dump(
                {
                    "date": "2026-01-01",
                    "concordance": {
                        "AAA": {"conviction_waterfall": {"m1": 3}},
                        "BBB": {"conviction_waterfall": {"m1": -3}},
                    },
                },
                f,
            )

        obs = extract_modifier_observations(hist_dir)
        assert len(obs) == 2
        assert {o["ticker"] for o in obs} == {"AAA", "BBB"}


class TestClassifyVerdict:
    def test_strong_signal_predictive(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict

        # |ρ| ≥ 0.10 with significance and large n → PREDICTIVE
        assert classify_verdict(rho=+0.15, p_value=0.001, n=200) == "PREDICTIVE"
        assert classify_verdict(rho=-0.20, p_value=0.0001, n=300) == "PREDICTIVE"

    def test_weak_but_significant_is_weak(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict

        # 0.05 ≤ |ρ| < 0.10 with significance → WEAK
        assert classify_verdict(rho=+0.07, p_value=0.01, n=200) == "WEAK"
        assert classify_verdict(rho=-0.06, p_value=0.04, n=300) == "WEAK"

    def test_no_significance_is_shadow(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict

        # |ρ| < 0.05 OR p > 0.05 → SHADOW
        assert classify_verdict(rho=+0.03, p_value=0.40, n=200) == "SHADOW"
        assert classify_verdict(rho=+0.20, p_value=0.30, n=200) == "SHADOW"

    def test_insufficient_n_is_insufficient_data(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict

        # n < 30 → INSUFFICIENT_DATA regardless of ρ
        assert classify_verdict(rho=+0.50, p_value=0.001, n=10) == "INSUFFICIENT_DATA"

    def test_nan_is_drop(self):
        from scripts.calibrate_modifiers_t30 import classify_verdict

        # NaN ρ means the modifier fired identical Δ on every stock
        # (no rank discrimination). DROP from production immediately.
        assert classify_verdict(rho=float("nan"), p_value=1.0, n=100) == "DROP"


class TestAnalyzeModifier:
    def test_perfect_correlation_yields_high_rho(self):
        from scripts.calibrate_modifiers_t30 import analyze_modifier

        # values and alphas perfectly rank-correlated → ρ ≈ +1
        values = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0] * 10
        alphas = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 10  # monotonic
        result = analyze_modifier("good_mod", values, alphas)
        assert result["n"] == 60
        assert result["spearman_rho"] > 0.9
        assert result["verdict"] == "PREDICTIVE"

    def test_constant_modifier_value_yields_nan_dropped(self):
        from scripts.calibrate_modifiers_t30 import analyze_modifier

        # All modifier values identical → no rank discrimination → DROP
        values = [3.0] * 100
        alphas = [-2.0, -1.0, 0.0, 1.0, 2.0] * 20
        result = analyze_modifier("constant_mod", values, alphas)
        assert result["verdict"] == "DROP"


class TestMain:
    def test_main_writes_json_with_per_modifier_verdicts(self, tmp_path, monkeypatch):
        """End-to-end: synthetic history + synthetic forward returns → JSON."""
        from scripts import calibrate_modifiers_t30 as mod

        # Build synthetic history where good_mod tracks alpha exactly
        hist_dir = tmp_path / "history"
        hist_dir.mkdir()

        for i, date in enumerate(["2026-01-01", "2026-01-02", "2026-01-03"]):
            rows = []
            for j in range(40):
                ticker = f"T{j:03d}"
                # Synthetic: good_mod is +j-20 (range -20..+19),
                # noise_mod is random (we'll set to constant +3 → DROP),
                # alpha is good_mod + small noise so ρ(good, α) ≈ 1
                rows.append(
                    {
                        "ticker": ticker,
                        "conviction_waterfall": {
                            "good_mod": float(j - 20),
                            "noise_mod": 3.0,  # constant → NaN ρ → DROP
                        },
                    }
                )
            with open(hist_dir / f"concordance-{date}.json", "w") as f:
                json.dump({"date": date, "concordance": rows}, f)

        # Stub the alpha lookup to mirror good_mod values exactly
        def fake_alpha_lookup(observations, **_kwargs):
            out = {}
            for o in observations:
                if o["modifier"] == "good_mod":
                    out[(o["ticker"], o["date"])] = o["value"]  # perfect rank match
                else:
                    out.setdefault((o["ticker"], o["date"]), o["value"])
            return out

        monkeypatch.setattr(mod, "compute_alpha_lookup", fake_alpha_lookup)

        out_path = tmp_path / "calibration.json"
        mod.main(history_dir=hist_dir, output_path=out_path)

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)

        assert "modifiers" in data
        assert "good_mod" in data["modifiers"]
        assert "noise_mod" in data["modifiers"]
        # good_mod should be PREDICTIVE
        assert data["modifiers"]["good_mod"]["verdict"] == "PREDICTIVE"
        # noise_mod constant value → DROP
        assert data["modifiers"]["noise_mod"]["verdict"] == "DROP"

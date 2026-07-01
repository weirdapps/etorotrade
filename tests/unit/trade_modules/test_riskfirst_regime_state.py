import json

from trade_modules.riskfirst.regime_state import (
    DEFAULT_STATE_PATH,
    resolve_regime_multiplier,
    update_history,
)


def test_update_history_overwrites_same_day():
    h = update_history([], "risk_on", "2026-07-01")
    h = update_history(h, "crisis", "2026-07-01")  # same day overwrites
    assert h == [["2026-07-01", "crisis"]]


def test_update_history_appends_new_day_and_trims():
    h = []
    for i in range(12):
        h = update_history(h, "neutral", f"2026-07-{i + 1:02d}", max_history=10)
    assert len(h) == 10
    assert h[-1][0] == "2026-07-12"


def test_resolve_first_run_falls_back_to_neutral(tmp_path):
    sp = str(tmp_path / "state.json")
    mult, detail = resolve_regime_multiplier(
        state_path=sp,
        persistence_days=2,
        regime_fn=lambda: "crisis",
        today="2026-07-01",
    )
    assert detail["raw_regime"] == "crisis"
    assert detail["confirmed_regime"] == "neutral"  # only 1 day, < persistence
    assert mult == 0.90
    assert json.load(open(sp))["history"][-1] == ["2026-07-01", "crisis"]


def test_resolve_confirms_after_persistence(tmp_path):
    sp = str(tmp_path / "state.json")
    resolve_regime_multiplier(
        state_path=sp, persistence_days=2, regime_fn=lambda: "crisis", today="2026-07-01"
    )
    mult, detail = resolve_regime_multiplier(
        state_path=sp,
        persistence_days=2,
        regime_fn=lambda: "crisis",
        today="2026-07-02",
    )
    assert detail["confirmed_regime"] == "crisis"
    assert mult == 0.40


def test_resolve_corrupt_state_recovers(tmp_path):
    sp = tmp_path / "state.json"
    sp.write_text("{ not json")
    mult, detail = resolve_regime_multiplier(
        state_path=str(sp),
        persistence_days=2,
        regime_fn=lambda: "neutral",
        today="2026-07-01",
    )
    assert mult == 0.90  # cold start after recovery


def test_default_state_path_is_str():
    assert isinstance(DEFAULT_STATE_PATH, str) and DEFAULT_STATE_PATH


def test_resolve_creates_nested_state_dir(tmp_path):
    sp = str(tmp_path / "sub" / "dir" / "state.json")
    mult, _ = resolve_regime_multiplier(
        state_path=sp, persistence_days=2, regime_fn=lambda: "neutral", today="2026-07-01"
    )
    import os as _os

    assert _os.path.exists(sp)

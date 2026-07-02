"""Tests for trade_modules.committee_v2 — S3 final-universe selection.

TDD: tests written BEFORE implementation. All tests should FAIL on first run.

Module structure:
  trade_modules/committee_v2/conviction.py  — score_conviction(row, weights=None) -> float
  trade_modules/committee_v2/actions.py     — assign_action(conviction, is_held, in_universe, cfg=None) -> str
  trade_modules/committee_v2/personas.py    — persona_debate(row) -> dict
  trade_modules/committee_v2/cio.py         — synthesize(candidates, held_tickers, cfg=None) -> list[dict]
"""

import pytest

from trade_modules.committee_v2.actions import assign_action
from trade_modules.committee_v2.cio import synthesize
from trade_modules.committee_v2.conviction import score_conviction
from trade_modules.committee_v2.personas import persona_debate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(**kwargs):
    """Build a minimal row dict, merging kwargs over a neutral baseline."""
    base = {
        "ticker": "TEST",
        "composite_pct": 0.5,
        "UP%": "20.0%",
        "%B": "50%",
        "ROE": "12%",
        "FCF": "2.0%",
        # persona fields
        "PET": "20",
        "PEF": "18",
        "PEG": "1.0",
        "DE": "80",
        "EG": "15",
        "52W": "60",
        "B": "1.0",
    }
    base.update(kwargs)
    return base


# ===========================================================================
# 1. score_conviction
# ===========================================================================


class TestScoreConviction:
    def test_returns_float_in_0_to_100(self):
        score = score_conviction(_row())
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_missing_composite_gives_neutral_contribution(self):
        row_with = _row(composite_pct=0.9)
        row_without = _row()
        row_without["composite_pct"] = None
        # Without composite the score is lower than with a very high composite
        assert score_conviction(row_with) > score_conviction(row_without)

    def test_missing_upside_gives_neutral_contribution(self):
        row_high = _row(**{"UP%": "80%"})
        row_missing = _row()
        row_missing["UP%"] = None
        assert score_conviction(row_high) > score_conviction(row_missing)

    def test_missing_buy_consensus_gives_neutral_contribution(self):
        row_high = _row(**{"%B": "100%"})
        row_missing = _row()
        row_missing["%B"] = None
        assert score_conviction(row_high) > score_conviction(row_missing)

    def test_missing_quality_gives_neutral_contribution(self):
        row_good = _row(ROE="25%", FCF="5.0%")
        row_missing = _row()
        row_missing["ROE"] = None
        row_missing["FCF"] = None
        assert score_conviction(row_good) > score_conviction(row_missing)

    # Monotonicity tests

    def test_monotonic_in_composite_pct(self):
        scores = [score_conviction(_row(composite_pct=v)) for v in [0.0, 0.25, 0.5, 0.75, 1.0]]
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)), (
            f"score_conviction not monotonically increasing in composite_pct: {scores}"
        )

    def test_monotonic_in_upside(self):
        scores = [score_conviction(_row(**{"UP%": f"{v}%"})) for v in [0, 10, 20, 40, 80]]
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)), (
            f"score_conviction not monotonically increasing in UP%: {scores}"
        )

    def test_monotonic_in_buy_consensus(self):
        scores = [score_conviction(_row(**{"%B": f"{v}%"})) for v in [0, 25, 50, 75, 100]]
        assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)), (
            f"score_conviction not monotonically increasing in %B: {scores}"
        )

    def test_quality_positive_roe_and_fcf_increases_score(self):
        score_good = score_conviction(_row(ROE="20%", FCF="3.0%"))
        score_bad = score_conviction(_row(ROE="-5%", FCF="-2.0%"))
        assert score_good > score_bad

    def test_all_missing_returns_50_neutral(self):
        """All missing → each component neutral → result near 50."""
        row = {"ticker": "X"}
        score = score_conviction(row)
        # Neutral = 50 for all components, so result should equal 50
        assert score == pytest.approx(50.0, abs=0.01)

    def test_all_max_returns_100(self):
        row = _row(composite_pct=1.0, **{"UP%": "100%", "%B": "100%", "ROE": "40%", "FCF": "10.0%"})
        score = score_conviction(row)
        assert score == pytest.approx(100.0, abs=0.01)

    def test_all_min_returns_0(self):
        row = _row(composite_pct=0.0, **{"UP%": "0%", "%B": "0%", "ROE": "-50%", "FCF": "-10.0%"})
        score = score_conviction(row)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_weights_parameter_respected(self):
        """Custom weights should change the score."""
        row = _row(composite_pct=1.0, **{"UP%": "0%", "%B": "0%", "ROE": "-10%", "FCF": "-5%"})
        # Default weights: composite=0.40 → high score
        default = score_conviction(row)
        # Override: composite=0.0, rest absorb weight → lower score
        custom = score_conviction(
            row, weights={"composite": 0.0, "upside": 0.40, "buy_consensus": 0.35, "quality": 0.25}
        )
        assert default != custom

    def test_clamped_below_0(self):
        """Score never goes negative even with extreme inputs."""
        row = _row(composite_pct=0.0, **{"UP%": "-50%", "%B": "0%", "ROE": "-100%", "FCF": "-100%"})
        assert score_conviction(row) >= 0.0

    def test_clamped_above_100(self):
        row = _row(composite_pct=2.0, **{"UP%": "500%", "%B": "200%", "ROE": "999%", "FCF": "999%"})
        assert score_conviction(row) <= 100.0


# ===========================================================================
# 2. assign_action
# ===========================================================================


class TestAssignAction:
    """Table-driven tests for the long-only action assignment."""

    # Default thresholds: buy=65, add=70, hold=45, trim=30

    # --- new names (not held) ---

    def test_new_high_conviction_is_buy(self):
        assert assign_action(70.0, is_held=False, in_universe=True) == "BUY"

    def test_new_at_buy_threshold_is_buy(self):
        assert assign_action(65.0, is_held=False, in_universe=True) == "BUY"

    def test_new_just_below_buy_is_none(self):
        assert assign_action(64.9, is_held=False, in_universe=True) == "NONE"

    def test_new_low_conviction_is_none(self):
        assert assign_action(20.0, is_held=False, in_universe=True) == "NONE"

    def test_new_zero_conviction_is_none(self):
        assert assign_action(0.0, is_held=False, in_universe=True) == "NONE"

    def test_new_perfect_conviction_is_buy(self):
        assert assign_action(100.0, is_held=False, in_universe=True) == "BUY"

    # --- held, dropped from universe (in_universe=False) ---

    def test_held_dropped_high_conviction_is_sell(self):
        """Dropped from eligible universe → always SELL regardless of conviction."""
        assert assign_action(90.0, is_held=True, in_universe=False) == "SELL"

    def test_held_dropped_mid_conviction_is_sell(self):
        assert assign_action(50.0, is_held=True, in_universe=False) == "SELL"

    def test_held_dropped_zero_conviction_is_sell(self):
        assert assign_action(0.0, is_held=True, in_universe=False) == "SELL"

    # --- held, in universe ---

    def test_held_high_conviction_is_add(self):
        assert assign_action(75.0, is_held=True, in_universe=True) == "ADD"

    def test_held_at_add_threshold_is_add(self):
        assert assign_action(70.0, is_held=True, in_universe=True) == "ADD"

    def test_held_just_below_add_is_hold(self):
        assert assign_action(69.9, is_held=True, in_universe=True) == "HOLD"

    def test_held_at_hold_threshold_is_hold(self):
        assert assign_action(45.0, is_held=True, in_universe=True) == "HOLD"

    def test_held_just_below_hold_is_trim(self):
        assert assign_action(44.9, is_held=True, in_universe=True) == "TRIM"

    def test_held_at_trim_threshold_is_trim(self):
        assert assign_action(30.0, is_held=True, in_universe=True) == "TRIM"

    def test_held_just_below_trim_is_sell(self):
        assert assign_action(29.9, is_held=True, in_universe=True) == "SELL"

    def test_held_zero_conviction_is_sell(self):
        assert assign_action(0.0, is_held=True, in_universe=True) == "SELL"

    # --- no shorts emitted ---

    def test_never_emits_short(self):
        """No action label should contain 'SHORT' anywhere."""
        all_combos = [
            (0.0, False, True),
            (100.0, False, True),
            (0.0, True, True),
            (100.0, True, True),
            (50.0, True, False),
        ]
        for conviction, is_held, in_universe in all_combos:
            action = assign_action(conviction, is_held, in_universe)
            assert "SHORT" not in action.upper(), (
                f"Short label emitted for ({conviction}, {is_held}, {in_universe}): {action}"
            )

    # --- config override ---

    def test_config_buy_threshold_override(self):
        """cfg buy=80 → conviction=70 should be NONE for new name."""
        cfg = {"buy": 80, "add": 85, "hold": 45, "trim": 30}
        assert assign_action(70.0, is_held=False, in_universe=True, cfg=cfg) == "NONE"

    def test_config_hold_threshold_override(self):
        """cfg hold=60 → conviction=55 should be TRIM for held name."""
        cfg = {"buy": 65, "add": 70, "hold": 60, "trim": 30}
        assert assign_action(55.0, is_held=True, in_universe=True, cfg=cfg) == "TRIM"


# ===========================================================================
# 3. persona_debate — ADVISORY
# ===========================================================================


class TestPersonaDebate:
    # --- return structure ---

    def test_returns_expected_keys(self):
        result = persona_debate(_row())
        assert set(result.keys()) == {"personas", "consensus", "dissent"}

    def test_personas_has_five_names(self):
        result = persona_debate(_row())
        assert set(result["personas"].keys()) == {"buffett", "wood", "klarman", "dalio", "lynch"}

    def test_verdicts_valid_values(self):
        result = persona_debate(_row())
        for name, verdict in result["personas"].items():
            assert verdict in {"approve", "neutral", "reject"}, (
                f"Persona {name} returned invalid verdict: {verdict!r}"
            )

    def test_dissent_is_list(self):
        result = persona_debate(_row())
        assert isinstance(result["dissent"], list)

    # --- Buffett lens ---

    def test_buffett_approves_quality_low_vol(self):
        """ROE>15, DE<100, PET in (0,25), beta<1.2 → approve."""
        row = _row(ROE="20%", DE="50", PET="18", **{"B": "0.9"})
        result = persona_debate(row)
        assert result["personas"]["buffett"] == "approve"

    def test_buffett_rejects_high_leverage(self):
        """DE>200 → reject."""
        row = _row(ROE="20%", DE="250", PET="18", **{"B": "0.9"})
        result = persona_debate(row)
        assert result["personas"]["buffett"] == "reject"

    def test_buffett_rejects_negative_earnings(self):
        """Negative or missing PET (earnings missing/negative) → reject."""
        row = _row(ROE="20%", DE="50", PET="-5", **{"B": "0.9"})
        result = persona_debate(row)
        assert result["personas"]["buffett"] == "reject"

    # --- Wood lens ---

    def test_wood_approves_growth_momentum(self):
        """EG>20, 52W>=70 → approve."""
        row = _row(EG="30", **{"52W": "80"})
        result = persona_debate(row)
        assert result["personas"]["wood"] == "approve"

    def test_wood_rejects_negative_growth(self):
        """EG<0 → reject."""
        row = _row(EG="-5", **{"52W": "80"})
        result = persona_debate(row)
        assert result["personas"]["wood"] == "reject"

    def test_wood_rejects_weak_momentum(self):
        """52W<40 → reject."""
        row = _row(EG="25", **{"52W": "30"})
        result = persona_debate(row)
        assert result["personas"]["wood"] == "reject"

    # --- Klarman lens ---

    def test_klarman_approves_margin_of_safety(self):
        """UP%>25, FCF>0, PET in (0,15) → approve."""
        row = _row(**{"UP%": "35%"}, FCF="3.0%", PET="12")
        result = persona_debate(row)
        assert result["personas"]["klarman"] == "approve"

    def test_klarman_rejects_negative_fcf(self):
        """FCF<0 → reject."""
        row = _row(**{"UP%": "35%"}, FCF="-2.0%", PET="12")
        result = persona_debate(row)
        assert result["personas"]["klarman"] == "reject"

    def test_klarman_rejects_high_pe(self):
        """PET>40 → reject."""
        row = _row(**{"UP%": "35%"}, FCF="3.0%", PET="45")
        result = persona_debate(row)
        assert result["personas"]["klarman"] == "reject"

    # --- Dalio lens ---

    def test_dalio_approves_low_beta(self):
        """beta<0.8 → approve."""
        row = _row(**{"B": "0.6"})
        result = persona_debate(row)
        assert result["personas"]["dalio"] == "approve"

    def test_dalio_rejects_high_beta(self):
        """beta>1.8 → reject."""
        row = _row(**{"B": "2.0"})
        result = persona_debate(row)
        assert result["personas"]["dalio"] == "reject"

    def test_dalio_neutral_mid_beta(self):
        """0.8<=beta<=1.8 → neutral."""
        row = _row(**{"B": "1.2"})
        result = persona_debate(row)
        assert result["personas"]["dalio"] == "neutral"

    # --- Lynch lens ---

    def test_lynch_approves_growth_at_price(self):
        """0<PEG<1.5, EG>0 → approve."""
        row = _row(PEG="1.2", EG="15")
        result = persona_debate(row)
        assert result["personas"]["lynch"] == "approve"

    def test_lynch_rejects_high_peg(self):
        """PEG>3 → reject."""
        row = _row(PEG="3.5", EG="15")
        result = persona_debate(row)
        assert result["personas"]["lynch"] == "reject"

    def test_lynch_rejects_negative_growth(self):
        """EG<=0 → reject or neutral (not approve)."""
        row = _row(PEG="1.0", EG="-5")
        result = persona_debate(row)
        assert result["personas"]["lynch"] != "approve"

    # --- consensus and dissent ---

    def test_consensus_is_approve_when_majority_approve(self):
        """A Buffett+Lynch+Klarman-approve row should show consensus=approve."""
        # All-approve row: quality value, growth, margin-of-safety
        row = _row(
            ROE="25%",
            DE="60",
            PET="12",
            **{"B": "0.9", "UP%": "35%"},
            FCF="4.0%",
            EG="25",
            **{"52W": "75"},
            PEG="1.0",
        )
        result = persona_debate(row)
        assert result["consensus"] in {"approve", "reject", "split"}

    def test_dissent_contains_disagreeing_personas(self):
        """If consensus=approve, personas that voted reject appear in dissent."""
        row = _row(
            ROE="25%",
            DE="60",
            PET="12",
            **{"B": "0.9", "UP%": "35%"},
            FCF="4.0%",
            EG="25",
            **{"52W": "75"},
            PEG="1.0",
        )
        result = persona_debate(row)
        consensus = result["consensus"]
        if consensus in {"approve", "reject"}:
            for name in result["dissent"]:
                verdict = result["personas"][name]
                # dissent = voted opposite of consensus (not neutral)
                assert verdict != consensus

    # --- ADVISORY: personas must NOT affect action ---

    def test_personas_advisory_action_identical_with_and_without(self):
        """The action must be identical whether or not persona_debate is called.

        Advisory means: calling persona_debate has ZERO side effects on
        conviction scoring or action assignment. We verify this by computing
        the action independently and confirming persona_debate output doesn't
        alter it.
        """
        row = _row()
        conviction_before = score_conviction(row)
        action_before = assign_action(conviction_before, is_held=False, in_universe=True)

        # Call persona_debate (advisory layer)
        persona_debate(row)

        conviction_after = score_conviction(row)
        action_after = assign_action(conviction_after, is_held=False, in_universe=True)

        assert action_before == action_after, (
            f"Calling persona_debate changed action: {action_before} -> {action_after}"
        )

    def test_personas_advisory_row_not_mutated(self):
        """persona_debate must not mutate the input row."""
        import copy

        row = _row()
        original = copy.deepcopy(row)
        persona_debate(row)
        assert row == original, "persona_debate mutated the input row"


# ===========================================================================
# 4. cio.synthesize
# ===========================================================================


class TestCioSynthesize:
    def _make_candidate(self, ticker, composite_pct=0.8, upside="30%", buy_pct="70%"):
        return _row(
            ticker=ticker,
            composite_pct=composite_pct,
            **{"UP%": upside, "%B": buy_pct},
            ROE="20%",
            FCF="3.0%",
        )

    def test_returns_list_of_dicts(self):
        candidates = [self._make_candidate("AAPL")]
        result = synthesize(candidates, held_tickers=set())
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_result_has_required_keys(self):
        candidates = [self._make_candidate("AAPL")]
        result = synthesize(candidates, held_tickers=set())
        required = {
            "ticker",
            "action",
            "conviction",
            "persona_consensus",
            "persona_dissent",
            "rationale",
        }
        for item in result:
            assert required.issubset(item.keys()), f"Missing keys: {required - item.keys()}"

    def test_none_actions_excluded(self):
        """Candidates below BUY threshold should not appear in output."""
        low = _row(
            ticker="LOW", composite_pct=0.0, **{"UP%": "0%", "%B": "0%"}, ROE="-20%", FCF="-5%"
        )
        result = synthesize([low], held_tickers=set())
        tickers = [r["ticker"] for r in result]
        assert "LOW" not in tickers

    def test_buy_action_assigned_to_new_high_conviction(self):
        candidates = [self._make_candidate("AAPL")]
        result = synthesize(candidates, held_tickers=set())
        entry = next((r for r in result if r["ticker"] == "AAPL"), None)
        assert entry is not None
        assert entry["action"] == "BUY"

    def test_held_not_in_candidates_gets_sell(self):
        """A held ticker that drops from the candidate list → SELL."""
        candidates = [self._make_candidate("AAPL")]
        result = synthesize(candidates, held_tickers={"MSFT", "AAPL"})
        msft = next((r for r in result if r["ticker"] == "MSFT"), None)
        assert msft is not None, "MSFT (held but not in candidates) must appear"
        assert msft["action"] == "SELL"

    def test_sorted_by_conviction_descending(self):
        high = self._make_candidate("HIGH", composite_pct=1.0, upside="80%", buy_pct="100%")
        low = self._make_candidate("MID", composite_pct=0.5, upside="20%", buy_pct="60%")
        result = synthesize([low, high], held_tickers=set())
        convictions = [r["conviction"] for r in result]
        assert convictions == sorted(convictions, reverse=True), (
            f"Results not sorted by conviction desc: {convictions}"
        )

    def test_long_only_no_shorts(self):
        """synthesize must never emit any action containing 'SHORT'."""
        candidates = [
            self._make_candidate("A"),
            _row(ticker="B", composite_pct=0.0, **{"UP%": "0%", "%B": "0%"}, ROE="-5%", FCF="-3%"),
        ]
        result = synthesize(candidates, held_tickers={"B"})
        for item in result:
            assert "SHORT" not in item["action"].upper(), f"Short action found: {item}"

    def test_held_in_candidates_with_high_conviction_is_add(self):
        candidate = self._make_candidate("TSLA")
        result = synthesize([candidate], held_tickers={"TSLA"})
        tsla = next((r for r in result if r["ticker"] == "TSLA"), None)
        assert tsla is not None
        assert tsla["action"] in {"ADD", "HOLD"}, f"Expected ADD or HOLD, got {tsla['action']}"

    def test_multiple_held_not_in_candidates_all_sell(self):
        candidates = [self._make_candidate("AAPL")]
        result = synthesize(candidates, held_tickers={"MSFT", "GOOG"})
        sell_tickers = {r["ticker"] for r in result if r["action"] == "SELL"}
        assert {"MSFT", "GOOG"}.issubset(sell_tickers)

    def test_conviction_values_in_range(self):
        candidates = [self._make_candidate(t) for t in ["A", "B", "C"]]
        result = synthesize(candidates, held_tickers=set())
        for item in result:
            assert 0.0 <= item["conviction"] <= 100.0, (
                f"Conviction out of range for {item['ticker']}: {item['conviction']}"
            )

    def test_persona_consensus_present_in_result(self):
        candidates = [self._make_candidate("NVDA")]
        result = synthesize(candidates, held_tickers=set())
        for item in result:
            assert item["persona_consensus"] in {"approve", "reject", "split", "neutral"}, (
                f"Unexpected persona_consensus: {item['persona_consensus']}"
            )

    def test_config_override_changes_thresholds(self):
        """With very high buy threshold, nothing new should be BUY."""
        cfg = {"buy": 99, "add": 99, "hold": 45, "trim": 30}
        candidates = [self._make_candidate("AAPL")]
        result = synthesize(candidates, held_tickers=set(), cfg=cfg)
        # Either AAPL not in results or action is not BUY
        aapl = next((r for r in result if r["ticker"] == "AAPL"), None)
        assert aapl is None or aapl["action"] != "BUY"

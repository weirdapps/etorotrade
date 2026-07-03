"""Acceptance tests for the S0 validation harness — "the referee".

These encode what a real-money VALIDATION HARNESS MUST do. They were written
FIRST (TDD) and MUST fail on the buggy harness, proving the defects, then pass
once the referee is fixed.

A false PASS here greenlights a broken strategy with real capital, so every
assertion below is a safety property, not a nicety:

  1. must-PASS   — a synthetic clear-edge dataset passes with high DSR + positive OOS alpha.
  2. must-FAIL   (noise)    — alpha ~ N(0, sigma), no edge, fails.
  3. must-FAIL   (constant) — a zero-variance alpha series must NOT get DSR 1.0 (C1 regression).
  4. no-masking  — one great family + one broken family => overall FAIL (I1).
  5. oos-computed — OOS is really computed when the span allows, and explicitly
                    marked not-computed (with a reason) when the span is too short (C4).

The synthetic-data generators are deliberately verbose and self-contained so the
data shape is auditable.
"""

from __future__ import annotations

import datetime

import numpy as np

from trade_modules.validation.harness import evaluate

# Horizons present in the synthetic data. Keep the max small so a reasonable
# date span yields >= 3 walk-forward folds after the embargo.
_HORIZONS = (7, 30, 60)
_PRIMARY = 30


def _make_family_rows(
    *,
    family: str,
    n_dates: int,
    tickers_per_date: int,
    alpha_mean: float,
    alpha_std: float,
    start: datetime.date,
    step_days: int,
    regimes: tuple[str, ...],
    seed: int,
) -> list[dict]:
    """Build a self-consistent block of rows for ONE signal family.

    Rows are emitted for every horizon in ``_HORIZONS`` so the harness's
    primary-horizon filter (h=30) has data AND the walk-forward embargo is
    driven by a realistic max horizon (60), not a 250-day hardcode.

    The alpha at the primary horizon is drawn N(alpha_mean, alpha_std); other
    horizons scale it so IC/persistence is well-defined but the headline stats
    are governed by h=30.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    dates = [start + datetime.timedelta(days=step_days * i) for i in range(n_dates)]
    for di, d in enumerate(dates):
        regime = regimes[di % len(regimes)]
        for t in range(tickers_per_date):
            ticker = f"{family}_T{t:03d}"
            base_alpha = float(rng.normal(alpha_mean, alpha_std))
            for h in _HORIZONS:
                # Scale non-primary horizons so a persistence signal exists but
                # h=30 stays the anchor. h=30 uses base_alpha verbatim.
                scale = {7: 0.6, 30: 1.0, 60: 1.3}[h]
                alpha_h = base_alpha * scale
                rows.append(
                    {
                        "signal": family,
                        "tier": "large",
                        "region": "us",
                        "signal_date": d.isoformat(),
                        "ticker": ticker,
                        "horizon": h,
                        "alpha": alpha_h,
                        # net_alpha = gross minus a small cost; keeps sign for a clear edge.
                        "net_alpha": alpha_h - 0.02,
                        "future_price": 100.0,
                        "price_at_signal": 90.0,
                        "regime": regime,
                    }
                )
    return rows


def _clear_edge_dataset() -> list[dict]:
    """Two consistent families, each with a strong, persistent positive net edge.

    Design targets:
      - >= 300 obs per family at the primary horizon,
      - >= 2 regimes including a stress regime ("crisis"),
      - date span wide enough (with a 60-day embargo) for >= 3 walk-forward folds,
      - low cross-family Sharpe dispersion => small estimated var_sr => honest high DSR.
    """
    start = datetime.date(2022, 1, 3)
    # 160 dates * 3 tickers = 480 rows/family at h=30 (>= 300); step 4 days => ~640-day span.
    fam_a = _make_family_rows(
        family="A",
        n_dates=160,
        tickers_per_date=3,
        alpha_mean=0.55,
        alpha_std=0.9,
        start=start,
        step_days=4,
        regimes=("risk_on", "neutral", "crisis"),
        seed=101,
    )
    fam_b = _make_family_rows(
        family="B",
        n_dates=160,
        tickers_per_date=3,
        alpha_mean=0.50,
        alpha_std=0.9,
        start=start,
        step_days=4,
        regimes=("risk_on", "neutral", "crisis"),
        seed=202,
    )
    return fam_a + fam_b


# ---------------------------------------------------------------------------
# 1. must-PASS — a clear, honest edge
# ---------------------------------------------------------------------------
class TestMustPassClearEdge:
    def setup_method(self):
        self.rows = _clear_edge_dataset()
        self.result = evaluate(self.rows)

    def test_overall_passed_true(self):
        assert self.result["overall"]["passed"] is True, (
            f"Clear-edge dataset must PASS; reasons={self.result['overall'].get('reasons')}"
        )

    def test_dsr_high(self):
        dsr = self.result["overall"]["dsr"]
        assert dsr is not None and dsr >= 0.95, f"Expected high DSR, got {dsr}"

    def test_oos_alpha_positive(self):
        # At least one material family must report a computed, positive OOS alpha.
        fams = self.result["families"]
        material = {f: s for f, s in fams.items() if not s.get("insufficient_data")}
        assert material, "Expected material families"
        computed_positive = []
        for f, s in material.items():
            oos = s.get("oos")
            if isinstance(oos, dict) and oos.get("computed"):
                oa = oos.get("oos_alpha")
                if oa is not None:
                    computed_positive.append(oa > 0)
        assert computed_positive, "Expected at least one family with computed OOS"
        assert any(computed_positive), "Expected positive OOS alpha in a clear-edge family"

    def test_at_least_three_folds_somewhere(self):
        # The clear-edge span must yield >= 3 OOS folds in at least one family.
        fams = self.result["families"]
        fold_counts = [
            s.get("oos", {}).get("n_folds", 0)
            for s in fams.values()
            if isinstance(s.get("oos"), dict)
        ]
        assert fold_counts and max(fold_counts) >= 3, (
            f"Expected >= 3 walk-forward folds somewhere, got {fold_counts}"
        )


# ---------------------------------------------------------------------------
# 2. must-FAIL — pure noise, no edge
# ---------------------------------------------------------------------------
class TestMustFailNoise:
    def setup_method(self):
        start = datetime.date(2022, 1, 3)
        self.rows = _make_family_rows(
            family="NOISE",
            n_dates=160,
            tickers_per_date=3,
            alpha_mean=0.0,
            alpha_std=1.0,
            start=start,
            step_days=4,
            regimes=("risk_on", "neutral", "crisis"),
            seed=999,
        )
        self.result = evaluate(self.rows)

    def test_passed_false(self):
        assert self.result["overall"]["passed"] is False

    def test_reasons_non_empty(self):
        assert len(self.result["overall"]["reasons"]) > 0

    def test_dsr_not_high(self):
        dsr = self.result["overall"]["dsr"]
        # Noise must not clear the significance hurdle.
        assert dsr is None or dsr < 0.95, f"Noise must not pass DSR, got {dsr}"


# ---------------------------------------------------------------------------
# 3. must-FAIL — CONSTANT alpha (zero variance). C1 regression.
# ---------------------------------------------------------------------------
class TestMustFailConstant:
    def setup_method(self):
        # Every alpha identical => zero variance. A naive Sharpe = mu/0 -> inf
        # -> DSR 1.0 would be a catastrophic false PASS. Must be treated as no-edge.
        start = datetime.date(2022, 1, 3)
        self.rows = _make_family_rows(
            family="CONST",
            n_dates=160,
            tickers_per_date=3,
            alpha_mean=0.05,
            alpha_std=0.0,  # <-- zero variance
            start=start,
            step_days=4,
            regimes=("risk_on", "neutral", "crisis"),
            seed=7,
        )
        self.result = evaluate(self.rows)

    def test_passed_false(self):
        assert self.result["overall"]["passed"] is False, (
            "A zero-variance (constant) alpha series must NOT pass — C1 regression."
        )

    def test_family_dsr_not_one(self):
        fam = self.result["families"].get("CONST", {})
        dsr = fam.get("dsr")
        assert dsr is None or dsr < 0.95, (
            f"Constant series must never yield a high/1.0 DSR, got {dsr}"
        )

    def test_family_sharpe_not_huge(self):
        fam = self.result["families"].get("CONST", {})
        sharpe = fam.get("sharpe")
        # Zero-variance must map to Sharpe None or 0 — never a blow-up.
        assert sharpe is None or abs(sharpe) < 1e-6, f"Expected ~0/None Sharpe, got {sharpe}"


# ---------------------------------------------------------------------------
# 4. no-masking — one great family must NOT rescue a broken family
# ---------------------------------------------------------------------------
class TestNoMasking:
    def setup_method(self):
        start = datetime.date(2022, 1, 3)
        great = _make_family_rows(
            family="GREAT",
            n_dates=160,
            tickers_per_date=3,
            alpha_mean=0.6,
            alpha_std=0.9,
            start=start,
            step_days=4,
            regimes=("risk_on", "neutral", "crisis"),
            seed=11,
        )
        # Broken family: strongly NEGATIVE net alpha, plenty of obs (material).
        broken = _make_family_rows(
            family="BROKEN",
            n_dates=160,
            tickers_per_date=3,
            alpha_mean=-0.5,
            alpha_std=0.9,
            start=start,
            step_days=4,
            regimes=("risk_on", "neutral", "crisis"),
            seed=22,
        )
        self.result = evaluate(great + broken)

    def test_overall_fails(self):
        assert self.result["overall"]["passed"] is False, (
            "A broken material family must not be masked by a great one (I1)."
        )

    def test_both_families_material(self):
        fams = self.result["families"]
        assert not fams["GREAT"].get("insufficient_data")
        assert not fams["BROKEN"].get("insufficient_data")


# ---------------------------------------------------------------------------
# 5. oos-computed — OOS truly computed when span allows; explicit reason when not
# ---------------------------------------------------------------------------
class TestOosComputed:
    def test_oos_computed_when_span_exceeds_embargo(self):
        rows = _clear_edge_dataset()
        result = evaluate(rows)
        fams = result["families"]
        material = {f: s for f, s in fams.items() if not s.get("insufficient_data")}
        assert material
        for f, s in material.items():
            oos = s.get("oos")
            assert isinstance(oos, dict), f"family {f}: oos must be a dict, got {oos!r}"
            assert oos.get("computed") is True, f"family {f}: oos should be computed, got {oos}"
            assert oos.get("oos_alpha") is not None, f"family {f}: oos_alpha should be set"

    def test_oos_not_computed_marked_with_reason_on_short_span(self):
        # All rows on a SINGLE date => span (0 days) < embargo => 0 folds.
        # OOS must be explicitly not-computed with a reason, never silently None.
        rng = np.random.default_rng(3)
        rows: list[dict] = []
        for t in range(60):  # 60 tickers, one date, >= min_obs at h=30
            base = float(rng.normal(0.4, 0.9))
            for h in _HORIZONS:
                rows.append(
                    {
                        "signal": "SHORT",
                        "tier": "large",
                        "region": "us",
                        "signal_date": "2022-06-01",
                        "ticker": f"S_T{t:03d}",
                        "horizon": h,
                        "alpha": base,
                        "net_alpha": base - 0.02,
                        "future_price": 100.0,
                        "price_at_signal": 90.0,
                        "regime": "risk_on",
                    }
                )
        result = evaluate(rows)
        fam = result["families"].get("SHORT", {})
        assert not fam.get("insufficient_data"), "SHORT family should be material at h=30"
        oos = fam.get("oos")
        assert isinstance(oos, dict), f"oos must be a dict, got {oos!r}"
        assert oos.get("computed") is False, f"oos should be not-computed, got {oos}"
        assert oos.get("reason"), f"oos must carry an explicit reason, got {oos}"

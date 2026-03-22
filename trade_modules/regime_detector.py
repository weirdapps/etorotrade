"""
Multi-Factor Market Regime Detector

Replaces simple VIX-threshold regime classification with a multi-factor
model that considers VIX level, VIX term structure, equity momentum,
and market breadth to produce a richer regime classification.

Regime Labels:
    RISK_ON    — Low vol, positive momentum, healthy breadth
    NEUTRAL    — Mixed signals, no dominant trend
    RISK_OFF   — Elevated vol, negative momentum
    CRISIS     — Extreme vol, VIX backwardation, significant drawdown

Integration:
    - Called from signal_tracker.log_signal() to tag each signal with regime
    - Enhances vix_regime_provider with multi-factor context
    - Feeds into backtest_engine for regime-stratified analysis
    - Informs conviction_sizer for regime-aware position sizing
"""

import logging
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Multi-factor market regime classification."""
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"


# Regime score ranges: higher = more risk-on
REGIME_SCORE_RANGES = {
    MarketRegime.CRISIS: (0, 20),
    MarketRegime.RISK_OFF: (20, 40),
    MarketRegime.NEUTRAL: (40, 60),
    MarketRegime.RISK_ON: (60, 100),
}

# Position sizing multipliers per regime (extends vix_regime_provider)
REGIME_POSITION_MULTIPLIERS = {
    MarketRegime.RISK_ON: 1.00,
    MarketRegime.NEUTRAL: 1.00,
    MarketRegime.RISK_OFF: 0.75,
    MarketRegime.CRISIS: 0.50,
}

# Feature weights for composite score
FEATURE_WEIGHTS = {
    "vix_score": 0.30,       # VIX level percentile (inverted)
    "term_structure": 0.15,  # VIX/VIX3M ratio
    "spy_momentum": 0.25,   # SPY returns (20d and 60d blend)
    "spy_drawdown": 0.15,   # Distance from 52-week high
    "vix_trend": 0.15,      # VIX directional change
}

# Cache
_regime_cache: Optional[Dict[str, Any]] = None
_regime_cache_timestamp: Optional[datetime] = None
_regime_lock = threading.Lock()
_REGIME_CACHE_TTL_MINUTES = 30


class RegimeDetector:
    """
    Multi-factor market regime classifier.

    Computes a regime score (0-100) from five market factors
    and maps it to a regime label.
    """

    def __init__(self, lookback_days: int = 252):
        """
        Args:
            lookback_days: Trading days for percentile calculations.
        """
        self.lookback_days = lookback_days
        self._market_data: Optional[Dict[str, Any]] = None

    def fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch all required market data for regime detection.

        Fetches: VIX, VIX3M (term structure), SPY (momentum/drawdown).
        """
        import yfinance as yf

        data: Dict[str, Any] = {}
        period = f"{self.lookback_days + 30}d"  # Extra buffer

        # VIX
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period=period)
            if not vix_hist.empty:
                data["vix_current"] = float(vix_hist["Close"].iloc[-1])
                data["vix_history"] = vix_hist["Close"].values.astype(float)
                data["vix_5d_ago"] = float(vix_hist["Close"].iloc[-6]) if len(vix_hist) >= 6 else data["vix_current"]
            else:
                logger.warning("VIX data empty")
                return {}
        except Exception as e:
            logger.warning("Failed to fetch VIX: %s", e)
            return {}

        # VIX3M (3-month VIX for term structure)
        try:
            vix3m = yf.Ticker("^VIX3M")
            vix3m_hist = vix3m.history(period="5d")
            if not vix3m_hist.empty:
                data["vix3m_current"] = float(vix3m_hist["Close"].iloc[-1])
            else:
                data["vix3m_current"] = None
        except Exception:
            data["vix3m_current"] = None

        # SPY
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period=period)
            if not spy_hist.empty:
                data["spy_current"] = float(spy_hist["Close"].iloc[-1])
                data["spy_history"] = spy_hist["Close"].values.astype(float)
                data["spy_52w_high"] = float(spy_hist["Close"].max())
            else:
                logger.warning("SPY data empty")
                return {}
        except Exception as e:
            logger.warning("Failed to fetch SPY: %s", e)
            return {}

        self._market_data = data
        return data

    def compute_features(
        self, data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute regime features from market data.

        Returns dict of feature name -> score (each 0-100, higher = more risk-on).
        """
        if data is None:
            data = self._market_data
        if not data:
            return {}

        features: Dict[str, float] = {}

        # 1. VIX Level Score (inverted: low VIX = high score)
        vix = data["vix_current"]
        vix_history = data.get("vix_history", np.array([vix]))

        # Percentile rank (inverted: low VIX percentile = high risk-on score)
        if len(vix_history) >= 20:
            vix_percentile = float(np.percentile(
                np.searchsorted(np.sort(vix_history), vix_history),
                np.searchsorted(np.sort(vix_history), vix) / len(vix_history) * 100,
            )) if len(vix_history) > 1 else 50.0
            # Simpler: what % of historical readings are below current
            vix_percentile = float(np.sum(vix_history < vix) / len(vix_history) * 100)
        else:
            vix_percentile = 50.0

        # Invert: high VIX percentile = low risk-on score
        features["vix_score"] = round(100.0 - vix_percentile, 1)

        # Also map absolute VIX to a score for robustness
        if vix < 13:
            vix_abs_score = 90.0
        elif vix < 16:
            vix_abs_score = 75.0
        elif vix < 20:
            vix_abs_score = 60.0
        elif vix < 25:
            vix_abs_score = 45.0
        elif vix < 30:
            vix_abs_score = 30.0
        elif vix < 40:
            vix_abs_score = 15.0
        else:
            vix_abs_score = 5.0

        # Blend percentile and absolute (50/50)
        features["vix_score"] = round(
            0.5 * features["vix_score"] + 0.5 * vix_abs_score, 1
        )

        # 2. VIX Term Structure
        # VIX/VIX3M < 1.0 = contango (normal, risk-on)
        # VIX/VIX3M > 1.0 = backwardation (stress, risk-off)
        vix3m = data.get("vix3m_current")
        if vix3m and vix3m > 0:
            ratio = vix / vix3m
            # Map ratio to score: 0.7 -> 90, 1.0 -> 50, 1.3 -> 10
            term_score = max(0, min(100, 50 + (1.0 - ratio) * 150))
            features["term_structure"] = round(term_score, 1)
        else:
            features["term_structure"] = 50.0  # Neutral if unavailable

        # 3. SPY Momentum (blended 20d and 60d returns)
        spy_history = data.get("spy_history", np.array([]))
        if len(spy_history) >= 21:
            spy_20d_return = float(
                (spy_history[-1] - spy_history[-21]) / spy_history[-21] * 100
            )
        else:
            spy_20d_return = 0.0

        if len(spy_history) >= 61:
            spy_60d_return = float(
                (spy_history[-1] - spy_history[-61]) / spy_history[-61] * 100
            )
        else:
            spy_60d_return = 0.0

        # Blend: 60% weight on 20d, 40% on 60d
        blended_return = 0.6 * spy_20d_return + 0.4 * spy_60d_return

        # Map to score: -10% -> 10, 0% -> 50, +10% -> 90
        momentum_score = max(0, min(100, 50 + blended_return * 5))
        features["spy_momentum"] = round(momentum_score, 1)

        # 4. SPY Drawdown from 52-week high
        spy_current = data.get("spy_current", 0)
        spy_52w_high = data.get("spy_52w_high", spy_current)
        if spy_52w_high > 0:
            drawdown_pct = (spy_current - spy_52w_high) / spy_52w_high * 100
            # Map: 0% -> 85, -5% -> 60, -10% -> 35, -20% -> 10
            drawdown_score = max(0, min(100, 85 + drawdown_pct * 4))
            features["spy_drawdown"] = round(drawdown_score, 1)
        else:
            features["spy_drawdown"] = 50.0

        # 5. VIX Trend (5-day directional change)
        vix_5d_ago = data.get("vix_5d_ago", vix)
        if vix_5d_ago > 0:
            vix_change_pct = (vix - vix_5d_ago) / vix_5d_ago * 100
            # Map: -20% -> 85 (VIX falling = good), +20% -> 15 (VIX rising = bad)
            trend_score = max(0, min(100, 50 - vix_change_pct * 1.75))
            features["vix_trend"] = round(trend_score, 1)
        else:
            features["vix_trend"] = 50.0

        return features

    def classify(
        self,
        features: Optional[Dict[str, float]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify current market regime from features.

        Returns dict with:
            regime: MarketRegime label
            score: Composite score 0-100
            features: Individual feature scores
            description: Human-readable summary
        """
        if features is None:
            if data is None:
                data = self.fetch_market_data()
            features = self.compute_features(data)

        if not features:
            return {
                "regime": MarketRegime.NEUTRAL.value,
                "score": 50.0,
                "features": {},
                "description": "Regime data unavailable — defaulting to NEUTRAL",
            }

        # Compute weighted composite score
        composite = 0.0
        total_weight = 0.0
        for feature_name, weight in FEATURE_WEIGHTS.items():
            if feature_name in features:
                composite += features[feature_name] * weight
                total_weight += weight

        if total_weight > 0:
            composite = composite / total_weight
        else:
            composite = 50.0

        composite = round(composite, 1)

        # Classify based on score ranges
        regime = MarketRegime.NEUTRAL
        for r, (low, high) in REGIME_SCORE_RANGES.items():
            if low <= composite < high:
                regime = r
                break
        if composite >= 60:
            regime = MarketRegime.RISK_ON

        # Override: VIX backwardation + high VIX = CRISIS regardless of score
        vix_current = (data or self._market_data or {}).get("vix_current", 0)
        term_struct = features.get("term_structure", 50)
        if vix_current > 30 and term_struct < 30:
            regime = MarketRegime.CRISIS
            composite = min(composite, 15.0)

        # Description
        descriptions = {
            MarketRegime.RISK_ON: "Risk-on environment — low volatility, positive momentum",
            MarketRegime.NEUTRAL: "Neutral regime — mixed signals, no dominant trend",
            MarketRegime.RISK_OFF: "Risk-off environment — elevated volatility, caution warranted",
            MarketRegime.CRISIS: "Crisis mode — extreme volatility, defensive positioning",
        }

        return {
            "regime": regime.value,
            "score": composite,
            "features": features,
            "vix": vix_current,
            "position_multiplier": REGIME_POSITION_MULTIPLIERS[regime],
            "description": descriptions[regime],
            "timestamp": datetime.now().isoformat(),
        }


# ==============================
# Module-level API (cached)
# ==============================


def _is_cache_valid() -> bool:
    """Check if regime cache is still valid."""
    if _regime_cache_timestamp is None:
        return False
    return datetime.now() - _regime_cache_timestamp < timedelta(
        minutes=_REGIME_CACHE_TTL_MINUTES
    )


def get_current_regime() -> str:
    """
    Get current market regime label.

    Primary API for signal_tracker.log_signal() integration.
    Cached with 30-minute TTL.

    Returns:
        Regime label: "risk_on", "neutral", "risk_off", or "crisis"
    """
    global _regime_cache, _regime_cache_timestamp

    with _regime_lock:
        if _is_cache_valid() and _regime_cache is not None:
            return _regime_cache["regime"]

    try:
        detector = RegimeDetector()
        data = detector.fetch_market_data()
        if not data:
            return MarketRegime.NEUTRAL.value

        features = detector.compute_features(data)
        result = detector.classify(features, data)

        with _regime_lock:
            _regime_cache = result
            _regime_cache_timestamp = datetime.now()

        logger.info(
            "Regime: %s (score=%.1f, VIX=%.1f)",
            result["regime"], result["score"], result.get("vix", 0),
        )

        return result["regime"]

    except Exception as e:
        logger.warning("Regime detection failed: %s", e)
        return MarketRegime.NEUTRAL.value


def get_regime_detail() -> Dict[str, Any]:
    """
    Get detailed regime analysis with feature breakdown.

    Returns full result dict from RegimeDetector.classify().
    """
    global _regime_cache, _regime_cache_timestamp

    with _regime_lock:
        if _is_cache_valid() and _regime_cache is not None:
            return _regime_cache.copy()

    try:
        detector = RegimeDetector()
        data = detector.fetch_market_data()
        if not data:
            return {"regime": "neutral", "score": 50.0, "features": {}}

        features = detector.compute_features(data)
        result = detector.classify(features, data)

        with _regime_lock:
            _regime_cache = result
            _regime_cache_timestamp = datetime.now()

        return result

    except Exception as e:
        logger.warning("Regime detail failed: %s", e)
        return {"regime": "neutral", "score": 50.0, "features": {}}


def get_regime_position_multiplier() -> float:
    """
    Get position sizing multiplier for current regime.

    Drop-in replacement for vix_regime_provider.get_position_size_multiplier()
    with richer signal.
    """
    regime_str = get_current_regime()
    try:
        regime = MarketRegime(regime_str)
    except ValueError:
        regime = MarketRegime.NEUTRAL
    return REGIME_POSITION_MULTIPLIERS[regime]


def invalidate_cache() -> None:
    """Force regime cache invalidation (for testing)."""
    global _regime_cache_timestamp
    with _regime_lock:
        _regime_cache_timestamp = None

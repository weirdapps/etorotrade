"""
Financial Sentiment Analyzer (finBERT)

Scores news headline sentiment per ticker using ProsusAI/finBERT,
a BERT model fine-tuned on financial text. Produces a sentiment score
from -1.0 (very negative) to +1.0 (very positive) per ticker.

Integration: Called from signal_tracker.log_signal() to enrich each
signal entry with sentiment data for forward validation.

Architecture:
- Model loaded lazily on first use, cached in memory
- Per-ticker sentiment cached daily (file + memory)
- Graceful degradation: returns None if model/news unavailable
"""

import json
import logging
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache paths
OUTPUT_DIR = Path(__file__).parent.parent / "yahoofinance" / "output"
SENTIMENT_CACHE_PATH = OUTPUT_DIR / ".sentiment_cache.json"

# Module-level state
_model = None
_tokenizer = None
_model_lock = threading.Lock()
_model_load_attempted = False

# Memory cache: {ticker: {score, headline_count, timestamp}}
_sentiment_cache: Dict[str, Dict[str, Any]] = {}
_cache_loaded = False
_cache_lock = threading.Lock()

# Performance limits
MAX_HEADLINES = 10  # Per ticker
MAX_TOKENS = 128  # Per headline
BATCH_SIZE = 16  # Headlines per inference batch
CACHE_TTL_HOURS = 18  # Cache validity


def _load_model() -> bool:
    """
    Load finBERT model and tokenizer. Called once, thread-safe.

    Returns True if model loaded successfully.
    """
    global _model, _tokenizer, _model_load_attempted

    with _model_lock:
        if _model_load_attempted:
            return _model is not None

        _model_load_attempted = True

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = "ProsusAI/finbert"
            logger.info("Loading finBERT model...")

            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            _model.eval()

            # Use MPS (Apple Silicon) if available, else CPU
            if torch.backends.mps.is_available():
                _model = _model.to("mps")
                logger.info("finBERT loaded on MPS (Apple Silicon)")
            else:
                logger.info("finBERT loaded on CPU")

            return True

        except Exception as e:
            logger.warning("Failed to load finBERT: %s", e)
            _model = None
            _tokenizer = None
            return False


def _load_cache() -> None:
    """Load daily sentiment cache from disk."""
    global _sentiment_cache, _cache_loaded

    with _cache_lock:
        if _cache_loaded:
            return
        _cache_loaded = True

        if not SENTIMENT_CACHE_PATH.exists():
            return

        try:
            with open(SENTIMENT_CACHE_PATH, "r") as f:
                data = json.load(f)

            # Only use cache from today (within TTL)
            cache_date = data.get("date", "")
            if cache_date == date.today().isoformat():
                _sentiment_cache = data.get("tickers", {})
                logger.info(
                    "Loaded sentiment cache: %d tickers from %s",
                    len(_sentiment_cache), cache_date,
                )
            else:
                logger.debug("Sentiment cache expired (date: %s)", cache_date)
        except Exception as e:
            logger.debug("Failed to load sentiment cache: %s", e)


def _save_cache() -> None:
    """Persist sentiment cache to disk."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "date": date.today().isoformat(),
            "generated_at": datetime.now().isoformat(),
            "ticker_count": len(_sentiment_cache),
            "tickers": _sentiment_cache,
        }
        with open(SENTIMENT_CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.debug("Failed to save sentiment cache: %s", e)


def _fetch_news(ticker: str) -> List[str]:
    """
    Fetch recent news headlines for a ticker via yfinance.

    Returns list of headline strings (up to MAX_HEADLINES).
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return []

        headlines = []
        for item in news[:MAX_HEADLINES]:
            # yfinance >=1.2.0 nests title under 'content'
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
            if title and len(title) > 10:
                headlines.append(title)

        return headlines

    except Exception as e:
        logger.debug("Failed to fetch news for %s: %s", ticker, e)
        return []


def _score_headlines(headlines: List[str]) -> List[Tuple[float, str]]:
    """
    Score headlines using finBERT.

    Returns list of (score, label) tuples where:
    - score: -1.0 to +1.0
    - label: "positive", "negative", or "neutral"
    """
    if not headlines or _model is None or _tokenizer is None:
        return []

    try:
        import torch

        device = next(_model.parameters()).device
        results = []

        # Process in batches
        for i in range(0, len(headlines), BATCH_SIZE):
            batch = headlines[i:i + BATCH_SIZE]

            inputs = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # finBERT classes: [positive, negative, neutral]
            labels = ["positive", "negative", "neutral"]

            for j in range(len(batch)):
                prob_positive = float(probs[j][0])
                prob_negative = float(probs[j][1])
                prob_neutral = float(probs[j][2])

                # Composite score: positive - negative (range -1 to +1)
                score = prob_positive - prob_negative

                # Determine dominant label
                max_idx = int(probs[j].argmax())
                label = labels[max_idx]

                results.append((round(score, 4), label))

        return results

    except Exception as e:
        logger.debug("finBERT scoring failed: %s", e)
        return []


def _aggregate_sentiment(
    scores: List[Tuple[float, str]],
) -> Dict[str, Any]:
    """
    Aggregate per-headline scores into a single ticker sentiment.

    Uses confidence-weighted mean: headlines with strong polarity
    (far from 0) contribute more than neutral ones.
    """
    if not scores:
        return {"score": None, "label": "unknown", "headline_count": 0}

    raw_scores = [s[0] for s in scores]
    labels = [s[1] for s in scores]

    # Confidence-weighted mean: weight by |score| to emphasize strong signals
    weights = [abs(s) + 0.1 for s in raw_scores]  # +0.1 floor
    weighted_sum = sum(s * w for s, w in zip(raw_scores, weights))
    total_weight = sum(weights)

    aggregate_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    # Dominant label
    label_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    dominant_label = max(label_counts, key=label_counts.get)

    return {
        "score": aggregate_score,
        "label": dominant_label,
        "headline_count": len(scores),
        "positive_pct": round(label_counts["positive"] / len(scores) * 100, 1),
        "negative_pct": round(label_counts["negative"] / len(scores) * 100, 1),
        "neutral_pct": round(label_counts["neutral"] / len(scores) * 100, 1),
    }


def get_ticker_sentiment(ticker: str) -> Optional[float]:
    """
    Get sentiment score for a ticker.

    Primary API for integration with signal_tracker.log_signal().

    Returns:
        Sentiment score from -1.0 (very negative) to +1.0 (very positive),
        or None if sentiment cannot be determined.
    """
    _load_cache()

    # Check memory cache
    cached = _sentiment_cache.get(ticker)
    if cached is not None:
        return cached.get("score")

    # Load model if not yet loaded
    if not _load_model():
        return None

    # Fetch and score
    headlines = _fetch_news(ticker)
    if not headlines:
        # Cache the miss to avoid repeated lookups
        _sentiment_cache[ticker] = {
            "score": None,
            "label": "no_news",
            "headline_count": 0,
            "timestamp": datetime.now().isoformat(),
        }
        return None

    scores = _score_headlines(headlines)
    result = _aggregate_sentiment(scores)
    result["timestamp"] = datetime.now().isoformat()

    # Cache result
    _sentiment_cache[ticker] = result

    # Periodically save cache (every 50 new entries)
    if len(_sentiment_cache) % 50 == 0:
        _save_cache()

    return result.get("score")


def get_ticker_sentiment_detail(ticker: str) -> Dict[str, Any]:
    """
    Get detailed sentiment breakdown for a ticker.

    Returns full detail including headline count, label distribution,
    and aggregate score.
    """
    _load_cache()

    cached = _sentiment_cache.get(ticker)
    if cached is not None:
        return cached

    # Compute if not cached
    get_ticker_sentiment(ticker)
    return _sentiment_cache.get(ticker, {"score": None, "label": "unavailable"})


def pre_warm(tickers: List[str]) -> Dict[str, Any]:
    """
    Pre-compute sentiment for a batch of tickers.

    Called before signal generation to warm the cache.
    Skips tickers already in today's cache.

    Args:
        tickers: List of ticker symbols to analyze.

    Returns:
        Summary dict with counts and timing.
    """
    _load_cache()

    if not _load_model():
        return {"status": "model_unavailable", "computed": 0}

    start = datetime.now()
    computed = 0
    skipped = 0

    for ticker in tickers:
        if ticker in _sentiment_cache:
            skipped += 1
            continue

        get_ticker_sentiment(ticker)
        computed += 1

    _save_cache()

    elapsed = (datetime.now() - start).total_seconds()

    summary = {
        "status": "complete",
        "total_tickers": len(tickers),
        "computed": computed,
        "skipped_cached": skipped,
        "elapsed_seconds": round(elapsed, 1),
        "avg_per_ticker": round(elapsed / max(computed, 1), 2),
    }

    logger.info(
        "Sentiment pre-warm: %d computed, %d cached, %.1fs elapsed",
        computed, skipped, elapsed,
    )

    return summary


def get_sentiment_summary() -> Dict[str, Any]:
    """Get summary statistics of cached sentiment data."""
    _load_cache()

    if not _sentiment_cache:
        return {"total": 0}

    scores = [
        v["score"] for v in _sentiment_cache.values()
        if v.get("score") is not None
    ]

    if not scores:
        return {"total": len(_sentiment_cache), "with_score": 0}

    import numpy as np

    return {
        "total": len(_sentiment_cache),
        "with_score": len(scores),
        "mean_score": round(float(np.mean(scores)), 4),
        "median_score": round(float(np.median(scores)), 4),
        "positive_count": sum(1 for s in scores if s > 0.1),
        "negative_count": sum(1 for s in scores if s < -0.1),
        "neutral_count": sum(1 for s in scores if -0.1 <= s <= 0.1),
    }


def flush_cache() -> None:
    """Force-save the current sentiment cache to disk."""
    _save_cache()

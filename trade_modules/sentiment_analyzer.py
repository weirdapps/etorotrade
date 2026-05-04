"""
Financial Sentiment Analyzer (finBERT) — Multi-Source

Scores news headline sentiment per ticker using ProsusAI/finBERT,
a BERT model fine-tuned on financial text. Produces a sentiment score
from -1.0 (very negative) to +1.0 (very positive) per ticker.

News Sources (in priority order):
    1. Premium Financial News — Google News RSS filtered to 14 elite outlets:
       Reuters, Bloomberg, CNBC, Financial Times, Wall Street Journal,
       Barron's, CNN Business, TheStreet, TipRanks, Investor's Business Daily,
       Benzinga, Motley Fool, MarketWatch, Seeking Alpha
    2. Finviz              — free, no key, ~100 headlines (aggregates major outlets)
    3. Google News (broad)  — free, no key, ~100 headlines (all sources)
    4. NewsAPI.org          — requires NEWS_API_KEY env var, ~60 per query
    5. Seeking Alpha RSS    — free, no key, ~30 analysis articles
    6. Yahoo Finance        — via yfinance, ~10 per ticker (shared rate limit)

Integration: Called from signal_tracker.log_signal() to enrich each
signal entry with sentiment data for forward validation.

Architecture:
- Model loaded lazily on first use, cached in memory
- Per-ticker sentiment cached daily (file + memory)
- Multi-source headlines aggregated and deduplicated
- Graceful degradation: returns None if model/news unavailable
"""

import json
import logging
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

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
_sentiment_cache: dict[str, dict[str, Any]] = {}
_cache_loaded = False
_cache_lock = threading.Lock()

# Performance limits
MAX_HEADLINES = 20  # Per ticker (across all sources, after dedup)
MAX_TOKENS = 128  # Per headline
BATCH_SIZE = 16  # Headlines per inference batch
CACHE_TTL_HOURS = 18  # Cache validity
_REQUEST_TIMEOUT = 8  # Seconds per source request

# Premium financial news domains for filtered Google News queries
_PREMIUM_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "ft.com",
    "wsj.com",
    "barrons.com",
    "cnn.com",
    "thestreet.com",
    "tipranks.com",
    "investors.com",
    "benzinga.com",
    "fool.com",
    "marketwatch.com",
    "seekingalpha.com",
]


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
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
            with open(SENTIMENT_CACHE_PATH) as f:
                data = json.load(f)

            # Only use cache from today (within TTL)
            cache_date = data.get("date", "")
            if cache_date == date.today().isoformat():
                _sentiment_cache = data.get("tickers", {})
                logger.info(
                    "Loaded sentiment cache: %d tickers from %s",
                    len(_sentiment_cache),
                    cache_date,
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


# ==============================
# Multi-Source News Fetchers
# ==============================


def _normalize_headline(text: str) -> str:
    """Normalize headline for deduplication comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _deduplicate_headlines(
    tagged_headlines: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Remove near-duplicate headlines across sources.

    Uses word-overlap ratio: if >60% of words match between two
    headlines, the later one is dropped (keeps first occurrence).
    """
    if not tagged_headlines:
        return []

    seen_normalized: list[set] = []
    unique: list[tuple[str, str]] = []

    for headline, source in tagged_headlines:
        norm = _normalize_headline(headline)
        words = set(norm.split())

        if not words or len(words) < 3:
            continue

        is_dup = False
        for seen_words in seen_normalized:
            overlap = len(words & seen_words)
            smaller = min(len(words), len(seen_words))
            if smaller > 0 and overlap / smaller > 0.60:
                is_dup = True
                break

        if not is_dup:
            seen_normalized.append(words)
            unique.append((headline, source))

    return unique


def _parse_google_news_rss(content: bytes) -> list[tuple[str, str]]:
    """Parse Google News RSS XML into (title, source_label) tuples."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(content)
    headlines = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        if title_el is not None and title_el.text:
            title = title_el.text.strip()
            if len(title) > 10:
                # Google News appends " - Source Name"
                # Extract source for labeling
                source_label = "google_news"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    source_suffix = parts[-1].lower()
                    for domain in _PREMIUM_DOMAINS:
                        domain_name = domain.replace(".com", "")
                        if domain_name in source_suffix:
                            source_label = domain_name
                            break
                headlines.append((title, source_label))
    return headlines


def _fetch_premium_news(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from premium financial sources via Google News RSS.

    Single HTTP call filtered to 14 elite financial outlets:
    Reuters, Bloomberg, CNBC, FT, WSJ, Barron's, CNN, TheStreet,
    TipRanks, IBD, Benzinga, Motley Fool, MarketWatch, Seeking Alpha.
    """
    try:
        import requests

        # Build OR query for all premium domains
        site_filter = " OR ".join(f"site:{d}" for d in _PREMIUM_DOMAINS)
        query = f"{ticker}+stock+({site_filter})"
        url = f"https://news.google.com/rss/search?" f"q={query}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []

        headlines = _parse_google_news_rss(r.content)
        return headlines[:30]

    except Exception as e:
        logger.debug("Premium news fetch failed for %s: %s", ticker, e)
        return []


def _fetch_google_news(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from Google News RSS (broad, all sources).

    Free, no API key needed, excellent coverage (~100 results).
    Complements the premium filter with smaller outlets and blogs.
    """
    try:
        import requests

        query = f"{ticker}+stock"
        url = f"https://news.google.com/rss/search?" f"q={query}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []

        headlines = _parse_google_news_rss(r.content)
        return headlines[:30]

    except Exception as e:
        logger.debug("Google News fetch failed for %s: %s", ticker, e)
        return []


def _fetch_finviz_news(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from Finviz stock page.

    Free, no API key needed. Parses the news table from the
    stock quote page. Aggregates Reuters, Bloomberg, MarketWatch,
    Barron's, WSJ, Benzinga, etc.
    """
    try:
        import requests

        url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        r = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []

        # Extract headlines from news table links
        pattern = r'class="tab-link-news"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, r.text)

        headlines = []
        for title in matches:
            title = title.strip()
            # Unescape HTML entities
            title = title.replace("&amp;", "&").replace("&quot;", '"')
            title = title.replace("&#39;", "'").replace("&lt;", "<")
            title = title.replace("&gt;", ">")
            if len(title) > 10:
                headlines.append((title, "finviz"))

        return headlines[:30]

    except Exception as e:
        logger.debug("Finviz fetch failed for %s: %s", ticker, e)
        return []


def _fetch_seekingalpha_news(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from Seeking Alpha RSS feed.

    Free, no API key needed. Returns ~30 recent articles
    including in-depth analysis pieces and news.
    """
    try:
        import xml.etree.ElementTree as ET

        import requests

        url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
        r = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []

        root = ET.fromstring(r.content)
        headlines = []

        # Try RSS format
        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
                if len(title) > 10:
                    headlines.append((title, "seeking_alpha"))

        # Try Atom format if RSS found nothing
        if not headlines:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                if title_el is not None and title_el.text:
                    title = title_el.text.strip()
                    if len(title) > 10:
                        headlines.append((title, "seeking_alpha"))

        return headlines[:20]

    except Exception as e:
        logger.debug("Seeking Alpha fetch failed for %s: %s", ticker, e)
        return []


def _fetch_newsapi_news(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from NewsAPI.org.

    Requires NEWS_API_KEY environment variable. Free tier allows
    100 requests/day with articles up to 1 month old.
    """
    try:
        import requests
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.environ.get("NEWS_API_KEY", "")
        if not api_key:
            return []

        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker}+stock&language=en&sortBy=publishedAt"
            f"&pageSize=15&apiKey={api_key}"
        )
        r = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []

        data = r.json()
        if data.get("status") != "ok":
            return []

        headlines = []
        for article in data.get("articles", []):
            title = article.get("title", "")
            source_name = article.get("source", {}).get("name", "newsapi")
            if title and len(title) > 10 and title != "[Removed]":
                headlines.append((title, f"newsapi:{source_name.lower()}"))

        return headlines[:15]

    except Exception as e:
        logger.debug("NewsAPI fetch failed for %s: %s", ticker, e)
        return []


def _fetch_cnbc_rss(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from CNBC RSS feed (top finance/investing).

    Free, no API key. General financial news — not ticker-specific
    but provides broad market context.
    """
    try:
        import xml.etree.ElementTree as ET

        import requests

        # CNBC finance RSS
        url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
        r = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []

        root = ET.fromstring(r.content)
        headlines = []
        ticker_upper = ticker.upper()

        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
                # Only keep if it mentions the ticker or company
                if len(title) > 10 and ticker_upper in title.upper():
                    headlines.append((title, "cnbc"))

        return headlines[:10]

    except Exception as e:
        logger.debug("CNBC RSS fetch failed for %s: %s", ticker, e)
        return []


def _fetch_yfinance_news(ticker: str) -> list[tuple[str, str]]:
    """
    Fetch headlines from Yahoo Finance via yfinance.

    No API key needed but shares rate limit with stock data fetches.
    Returns ~10 headlines per ticker.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return []

        headlines = []
        for item in news[:10]:
            # yfinance >=1.2.0 nests title under 'content'
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
            if title and len(title) > 10:
                headlines.append((title, "yfinance"))

        return headlines

    except Exception as e:
        logger.debug("yfinance news fetch failed for %s: %s", ticker, e)
        return []


# Source registry: (fetcher_function, name)
# Order matters — higher priority / higher quality sources first.
# Premium filtered sources come before broad to win dedup priority.
_NEWS_SOURCES = [
    (_fetch_premium_news, "premium"),
    (_fetch_finviz_news, "finviz"),
    (_fetch_google_news, "google_news"),
    (_fetch_newsapi_news, "newsapi"),
    (_fetch_seekingalpha_news, "seeking_alpha"),
    (_fetch_cnbc_rss, "cnbc"),
    (_fetch_yfinance_news, "yfinance"),
]


def _fetch_news(ticker: str) -> list[str]:
    """
    Fetch and deduplicate news headlines from multiple sources.

    Tries all available sources, aggregates headlines, removes
    near-duplicates, and returns up to MAX_HEADLINES unique headlines.

    Returns list of headline strings.
    """
    all_tagged: list[tuple[str, str]] = []
    sources_hit: list[str] = []

    for fetcher, source_name in _NEWS_SOURCES:
        try:
            results = fetcher(ticker)
            if results:
                all_tagged.extend(results)
                sources_hit.append(f"{source_name}:{len(results)}")
        except Exception as e:
            logger.debug("Source %s failed for %s: %s", source_name, ticker, e)

    if not all_tagged:
        return []

    # Deduplicate across sources
    unique = _deduplicate_headlines(all_tagged)

    if sources_hit:
        logger.debug(
            "News for %s: %d raw -> %d unique from [%s]",
            ticker,
            len(all_tagged),
            len(unique),
            ", ".join(sources_hit),
        )

    # Return just the headline text, capped
    return [headline for headline, _source in unique[:MAX_HEADLINES]]


def _fetch_news_detailed(ticker: str) -> tuple[list[str], dict[str, int]]:
    """
    Like _fetch_news but also returns source breakdown.

    Returns:
        (headlines, source_counts) where source_counts maps
        source name to number of unique headlines contributed.
    """
    all_tagged: list[tuple[str, str]] = []

    for fetcher, _source_name in _NEWS_SOURCES:
        try:
            results = fetcher(ticker)
            if results:
                all_tagged.extend(results)
        except Exception:
            pass

    if not all_tagged:
        return [], {}

    unique = _deduplicate_headlines(all_tagged)
    capped = unique[:MAX_HEADLINES]

    # Count contributions per source
    source_counts: dict[str, int] = {}
    for _headline, source in capped:
        source_counts[source] = source_counts.get(source, 0) + 1

    return [h for h, _s in capped], source_counts


# ==============================
# finBERT Scoring
# ==============================


def _score_headlines(headlines: list[str]) -> list[tuple[float, str]]:
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
            batch = headlines[i : i + BATCH_SIZE]

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
    scores: list[tuple[float, str]],
    source_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
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
    weighted_sum = sum(s * w for s, w in zip(raw_scores, weights, strict=False))
    total_weight = sum(weights)

    aggregate_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    # Dominant label
    label_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    dominant_label = max(label_counts, key=label_counts.get)

    result = {
        "score": aggregate_score,
        "label": dominant_label,
        "headline_count": len(scores),
        "positive_pct": round(label_counts["positive"] / len(scores) * 100, 1),
        "negative_pct": round(label_counts["negative"] / len(scores) * 100, 1),
        "neutral_pct": round(label_counts["neutral"] / len(scores) * 100, 1),
    }

    if source_counts:
        result["sources"] = source_counts
        result["source_count"] = len(source_counts)

    return result


# ==============================
# Public API
# ==============================


def get_ticker_sentiment(ticker: str) -> float | None:
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

    # Fetch from all sources and score
    headlines, source_counts = _fetch_news_detailed(ticker)
    if not headlines:
        # Cache the miss to avoid repeated lookups
        _sentiment_cache[ticker] = {
            "score": None,
            "label": "no_news",
            "headline_count": 0,
            "sources": {},
            "source_count": 0,
            "timestamp": datetime.now().isoformat(),
        }
        return None

    scores = _score_headlines(headlines)
    result = _aggregate_sentiment(scores, source_counts)
    result["timestamp"] = datetime.now().isoformat()

    # Cache result
    _sentiment_cache[ticker] = result

    # Periodically save cache (every 50 new entries)
    if len(_sentiment_cache) % 50 == 0:
        _save_cache()

    return result.get("score")


def get_ticker_sentiment_detail(ticker: str) -> dict[str, Any]:
    """
    Get detailed sentiment breakdown for a ticker.

    Returns full detail including headline count, label distribution,
    source breakdown, and aggregate score.
    """
    _load_cache()

    cached = _sentiment_cache.get(ticker)
    if cached is not None:
        return cached

    # Compute if not cached
    get_ticker_sentiment(ticker)
    return _sentiment_cache.get(ticker, {"score": None, "label": "unavailable"})


def pre_warm(tickers: list[str]) -> dict[str, Any]:
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
        computed,
        skipped,
        elapsed,
    )

    return summary


def get_sentiment_summary() -> dict[str, Any]:
    """Get summary statistics of cached sentiment data."""
    _load_cache()

    if not _sentiment_cache:
        return {"total": 0}

    scores = [v["score"] for v in _sentiment_cache.values() if v.get("score") is not None]

    if not scores:
        return {"total": len(_sentiment_cache), "with_score": 0}

    import numpy as np

    # Source diversity stats
    all_sources: dict[str, int] = {}
    for v in _sentiment_cache.values():
        for src, count in v.get("sources", {}).items():
            all_sources[src] = all_sources.get(src, 0) + count

    return {
        "total": len(_sentiment_cache),
        "with_score": len(scores),
        "mean_score": round(float(np.mean(scores)), 4),
        "median_score": round(float(np.median(scores)), 4),
        "positive_count": sum(1 for s in scores if s > 0.1),
        "negative_count": sum(1 for s in scores if s < -0.1),
        "neutral_count": sum(1 for s in scores if -0.1 <= s <= 0.1),
        "source_totals": all_sources,
    }


def flush_cache() -> None:
    """Force-save the current sentiment cache to disk."""
    _save_cache()

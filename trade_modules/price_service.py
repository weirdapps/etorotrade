"""
Centralized Price Service for Backtesting

Single source of truth for historical price data across all backtest
modules. Features:
- Batch yfinance downloads (no per-ticker calls)
- Persistent parquet cache
- Trading-day-aware indexing
- Regional benchmark support
- Consistent error handling
"""

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Regional benchmarks: region -> benchmark ETF ticker
REGION_BENCHMARKS: dict[str, str] = {
    "us": "SPY",
    "eu": "EXS1.DE",  # iShares Core DAX (STOXX proxy)
    "uk": "ISF.L",  # iShares Core FTSE 100
    "hk": "2800.HK",  # Tracker Fund of Hong Kong
    "default": "SPY",
}

DEFAULT_CACHE_DIR = Path.home() / ".weirdapps-trading" / "price_cache"

# How far behind `end_date` a cached column may fall before it is refetched.
#
# Deliberately not zero. `end_date` is routinely a Saturday, every venue here
# keeps its own holiday calendar, and Hong Kong and Xetra do not close on the
# same days as New York. Demanding coverage to the exact day would refetch the
# entire universe on every call, which is the opposite failure to the one below.
# Five days spans a long weekend without spanning a market week.
CACHE_STALE_TOLERANCE_DAYS = 5


def _stale_columns(
    cached: pd.DataFrame | None,
    end_date: str,
    tolerance_days: int = CACHE_STALE_TOLERANCE_DAYS,
) -> set[str]:
    """Cached tickers whose prices stop before `end_date` minus the tolerance.

    A column that exists is not a column that is current, and until 2026-09-13
    this service could not tell the two apart: `missing` was computed purely
    from set membership, so a ticker was fetched exactly once and then never
    again however old its data became.

    The production cache showed what that costs. backtest_prices.parquet was
    (216, 3556) with 3129 columns holding no valid price after 2026-05-27 and
    394 columns entirely empty, while its mtime tracked the most recent weekly
    run: the file was rewritten every week and its contents had not moved in
    three and a half months. The backtest on top of it evaluated two tickers out
    of 106.

    An all-NaT column has no last_valid_index and reads as stale, which is the
    right answer: we hold no price for it, so "covered" would be a lie.
    """
    if cached is None or cached.empty:
        return set()
    try:
        cutoff = pd.Timestamp(end_date) - pd.Timedelta(days=tolerance_days)
    except (ValueError, TypeError):
        # An unparseable end_date must not silently disable the check; treat
        # everything as stale and let the download decide.
        return set(cached.columns)
    stale = set()
    for col in cached.columns:
        last = cached[col].last_valid_index()
        if last is None or pd.Timestamp(last) < cutoff:
            stale.add(col)
    return stale


def _apply_data_fetch_substitutions(
    tickers: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Translate held-side tickers to yfinance-friendly substitutes.

    Some portfolio holdings (e.g. LYXGRE.DE on Xetra) are not indexed by
    Yahoo Finance, but the same security trades on another exchange under
    a different symbol (LYXGRE.DE → GRE.PA on Paris). The canonical map
    lives on ConfigManager.data_fetch_substitutions; this helper applies it
    transparently around batch fetches and returns a reverse map so callers
    can rename result columns back to the held-side symbol.

    Returns:
        (fetch_tickers, reverse_map) where reverse_map[fetch] = original.
    """
    try:
        from trade_modules.config_manager import get_config

        subs = get_config().data_fetch_substitutions
    except Exception:
        return list(tickers), {}
    if not subs:
        return list(tickers), {}
    fetch_tickers: list[str] = []
    reverse: dict[str, str] = {}
    for t in tickers:
        sub = subs.get(t.upper()) if t else None
        if sub:
            fetch_tickers.append(sub)
            reverse[sub] = t
        else:
            fetch_tickers.append(t)
    return fetch_tickers, reverse


class PriceService:
    """
    Centralized price fetcher with caching and trading-day indexing.

    Usage:
        svc = PriceService()
        prices = svc.get_prices(["AAPL", "MSFT"], "2026-01-01", "2026-04-01")
        # prices is a DataFrame: index=DatetimeIndex (trading days), columns=tickers
        ret = svc.trading_day_return(prices, "AAPL", "2026-01-15", horizon=7)
    """

    def __init__(
        self,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        default_benchmark: str = "SPY",
    ):
        self.cache_dir = cache_dir
        self.default_benchmark = default_benchmark
        self._price_cache: pd.DataFrame | None = None

    def get_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        include_benchmark: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical close prices for tickers.

        Always includes the default benchmark (SPY) unless disabled.
        Uses parquet cache to avoid redundant downloads.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date YYYY-MM-DD.
            end_date: End date YYYY-MM-DD.
            include_benchmark: If True, include default benchmark.

        Returns:
            DataFrame with DatetimeIndex and one column per ticker.
        """
        if not tickers:
            return pd.DataFrame()

        all_tickers = list(set(tickers))
        if include_benchmark and self.default_benchmark not in all_tickers:
            all_tickers.append(self.default_benchmark)

        # Add regional benchmarks for non-US tickers
        for bm in REGION_BENCHMARKS.values():
            if bm != self.default_benchmark and bm not in all_tickers:
                all_tickers.append(bm)

        # Check cache. Absent OR stale both count as missing: see _stale_columns.
        cached = self._load_cache()
        if cached is not None:
            cached_tickers = set(cached.columns)
            stale = _stale_columns(cached, end_date)
            missing = [t for t in all_tickers if t not in cached_tickers or t in stale]
            refreshed = [t for t in missing if t in cached_tickers]
            if refreshed:
                # Say it out loud. The whole failure was silent: a frozen cache
                # and a healthy one produce identical logs and an identical
                # exit code, and the weekly job reported status "complete" on
                # two tickers out of 106 for months.
                logger.info(
                    "Refetching %d cached ticker(s) with no prices within %dd of %s",
                    len(refreshed),
                    CACHE_STALE_TOLERANCE_DAYS,
                    end_date,
                )
        else:
            missing = all_tickers

        # Fetch missing tickers
        if missing:
            new_data = self._download_prices(missing, start_date, end_date)
            if cached is not None and not new_data.empty:
                prices = pd.concat([cached, new_data], axis=1)
                # keep="last", and it is load-bearing. `cached` is concatenated
                # first, so pandas' default keep="first" kept the STALE column
                # and discarded the download that had just been made to replace
                # it. A refetch changed nothing, which is why fixing the
                # staleness check alone would have left the cache frozen: the
                # right requests would have been issued and every answer thrown
                # away. Found 2026-09-13, alongside the membership bug above.
                prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
            elif not new_data.empty:
                prices = new_data
            else:
                prices = cached if cached is not None else pd.DataFrame()

            # Update cache
            if not prices.empty:
                self._save_cache(prices)
        else:
            prices = cached if cached is not None else pd.DataFrame()

        # Ensure DatetimeIndex
        if not prices.empty and not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)

        self._price_cache = prices
        return prices

    def get_benchmark(self, region: str | None = None) -> str:
        """Return the appropriate benchmark ticker for a region."""
        if region is None:
            return self.default_benchmark
        return REGION_BENCHMARKS.get(region, self.default_benchmark)

    def trading_day_return(
        self,
        prices: pd.DataFrame,
        ticker: str,
        signal_date: str,
        horizon: int,
    ) -> float | None:
        """
        Compute return at T+horizon TRADING DAYS from signal_date.

        Uses index-based offset on the trading calendar (not calendar days).

        Args:
            prices: DataFrame from get_prices().
            ticker: Ticker symbol.
            signal_date: Signal date YYYY-MM-DD.
            horizon: Number of trading days forward.

        Returns:
            Percentage return, or None if insufficient data.
        """
        if ticker not in prices.columns:
            return None

        ticker_prices = prices[ticker].dropna()
        if ticker_prices.empty:
            return None

        sig_ts = pd.Timestamp(signal_date)
        future_dates = ticker_prices.index[ticker_prices.index >= sig_ts]
        if len(future_dates) <= horizon:
            return None

        base_price = float(ticker_prices.loc[future_dates[0]])
        future_price = float(ticker_prices.loc[future_dates[horizon]])

        if base_price <= 0:
            return None

        return (future_price - base_price) / base_price * 100

    def trading_day_alpha(
        self,
        prices: pd.DataFrame,
        ticker: str,
        signal_date: str,
        horizon: int,
        region: str | None = None,
    ) -> float | None:
        """
        Compute alpha (stock return minus benchmark return) at T+horizon.

        Uses region-appropriate benchmark.

        Args:
            prices: DataFrame from get_prices().
            ticker: Ticker symbol.
            signal_date: Signal date YYYY-MM-DD.
            horizon: Number of trading days forward.
            region: Stock region for benchmark selection.

        Returns:
            Alpha percentage, or None if insufficient data.
        """
        stock_return = self.trading_day_return(prices, ticker, signal_date, horizon)
        if stock_return is None:
            return None

        benchmark = self.get_benchmark(region)
        bm_return = self.trading_day_return(prices, benchmark, signal_date, horizon)
        if bm_return is None:
            # Fall back to default benchmark
            bm_return = self.trading_day_return(
                prices, self.default_benchmark, signal_date, horizon
            )
        if bm_return is None:
            return None

        return stock_return - bm_return

    def _download_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        batch_size: int = 500,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Batch-download close prices via yfinance with retry/backoff."""
        import yfinance as yf

        fetch_tickers, reverse_map = _apply_data_fetch_substitutions(tickers)

        frames = []
        for i in range(0, len(fetch_tickers), batch_size):
            batch = fetch_tickers[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        batch,
                        start=start_date,
                        end=end_date,
                        group_by="ticker",
                        threads=True,
                        progress=False,
                        auto_adjust=True,
                    )
                    if data.empty:
                        break

                    if len(batch) == 1:
                        close = data[["Close"]].rename(columns={"Close": batch[0]})
                    else:
                        if isinstance(data.columns, pd.MultiIndex):
                            close = data.xs("Close", axis=1, level=1)
                        else:
                            close = data[["Close"]]
                    frames.append(close)
                    break  # success
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.info(
                            "Batch %d-%d attempt %d failed, retrying in %ds: %s",
                            i,
                            i + len(batch),
                            attempt + 1,
                            wait,
                            e,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "Failed to fetch batch %d-%d after %d attempts: %s",
                            i,
                            i + len(batch),
                            max_retries,
                            e,
                        )

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        # Rename substituted columns back to the held-side symbol so callers
        # see the ticker they asked for (e.g. LYXGRE.DE, not GRE.PA).
        if reverse_map:
            result = result.rename(columns=reverse_map)
        return result

    def _load_cache(self) -> pd.DataFrame | None:
        """Load cached prices from parquet."""
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / "backtest_prices.parquet"
        if not cache_path.exists():
            return None
        try:
            df = pd.read_parquet(cache_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.debug("Cache load failed: %s", e)
            return None

    def _save_cache(self, prices: pd.DataFrame) -> None:
        """Save prices to parquet cache."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / "backtest_prices.parquet"
        try:
            prices.to_parquet(cache_path)
        except Exception as e:
            logger.debug("Cache save failed: %s", e)

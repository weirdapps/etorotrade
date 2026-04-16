"""
Committee Backtesting Framework

CIO Legacy Review Finding D1: Systematic backtesting of conviction parameters
against historical committee data and forward returns.

Without backtesting, every parameter in the synthesis engine is an informed
opinion. With backtesting, every parameter is evidence. This module provides:

1. Historical concordance loading from committee archives
2. Forward return computation at T+7, T+30, T+90 horizons
3. Parameter sweep framework for key thresholds
4. Hit rate, information ratio, and return metrics
5. Summary reports for parameter calibration

Usage:
    backtester = CommitteeBacktester()
    backtester.load_history(committee_log_dir)
    results = backtester.evaluate_performance()
    sweep = backtester.sweep_parameter("buy_floor", [50, 55, 60])
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_COMMITTEE_LOG_DIR = Path.home() / ".weirdapps-trading" / "committee"


class CommitteeBacktester:
    """
    Backtesting framework for committee conviction parameters.

    Loads historical concordance data, computes forward returns, and
    evaluates parameter choices against realized outcomes.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
    ):
        self.log_dir = log_dir or DEFAULT_COMMITTEE_LOG_DIR
        self.history: List[Dict[str, Any]] = []
        self.forward_returns: Dict[str, Dict[str, float]] = {}

    def load_history(
        self,
        log_dir: Optional[Path] = None,
        min_entries: int = 5,
    ) -> int:
        """
        Load historical committee concordance data.

        Scans for concordance.json files in the committee log directory
        and builds a timeline of recommendations.

        Args:
            log_dir: Directory containing committee output files.
            min_entries: Minimum entries required for meaningful analysis.

        Returns:
            Number of historical entries loaded.
        """
        directory = log_dir or self.log_dir
        if not directory.is_dir():
            logger.warning("Committee log directory not found: %s", directory)
            return 0

        entries = []

        # Scan for concordance files (various naming patterns)
        # Prioritize history/ subdirectory (dated archives from generate_report_from_files)
        history_dir = directory / "history"
        has_history = history_dir.is_dir() and any(history_dir.glob("concordance-*.json"))
        patterns = ["concordance*.json", "synthesis*.json", "*committee*.json"]
        for pattern in patterns:
            for fpath in directory.glob(f"**/{pattern}"):
                # Skip undated concordance.json only if dated history exists
                if (has_history and fpath.name == "concordance.json"
                        and fpath.parent == directory):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("Skipping %s: %s", fpath.name, exc)
                    continue

                # Skip bare list format (no metadata to extract)
                if isinstance(data, list):
                    data = {"concordance": data}

                # Extract date from filename or data
                date_str = data.get("date")
                if not date_str:
                    # Try to extract from filename
                    for part in fpath.stem.split("-"):
                        if len(part) == 10 and part.count("-") == 2:
                            date_str = part
                            break
                    if not date_str:
                        # Use file modification time
                        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                        date_str = mtime.strftime("%Y-%m-%d")

                # Extract concordance entries
                concordance = data.get("concordance", [])
                if not concordance:
                    # Try alternate key
                    concordance = data.get("stocks", {})
                # Normalize dict format to list of dicts
                if isinstance(concordance, dict):
                    concordance = [
                        dict(v, ticker=k) if isinstance(v, dict) else {"ticker": k}
                        for k, v in concordance.items()
                    ]

                if concordance:
                    entries.append({
                        "date": date_str,
                        "file": str(fpath),
                        "concordance": concordance,
                        "version": data.get("version", "unknown"),
                    })

        # Deduplicate by date (keep the entry with most concordance data)
        by_date: Dict[str, Dict] = {}
        for entry in entries:
            d = entry["date"]
            if d not in by_date or len(entry["concordance"]) > len(by_date[d]["concordance"]):
                by_date[d] = entry
        entries = sorted(by_date.values(), key=lambda e: e["date"])
        self.history = entries

        logger.info(
            "Loaded %d committee snapshots (%s to %s)",
            len(entries),
            entries[0]["date"] if entries else "N/A",
            entries[-1]["date"] if entries else "N/A",
        )
        return len(entries)

    def compute_forward_returns(
        self,
        price_fetcher=None,
        price_service=None,
        horizons: Tuple[int, ...] = (7, 30, 90),
        benchmark: str = "SPY",
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Compute forward returns and alpha for historical recommendations.

        For each (ticker, date) recommendation, computes realized returns
        and alpha vs benchmark at the given horizons.

        Supports two backends:
        - price_service (preferred): Uses trading-day offsets via PriceService.
          Batch-downloads prices once and computes T+N in trading days.
        - price_fetcher (legacy): Uses calendar-day offsets via timedelta.
          Kept for backward compatibility.

        Args:
            price_fetcher: (Legacy) Callable(ticker, date_str) -> Optional[float].
                Uses calendar-day offsets. Deprecated in favor of price_service.
            price_service: PriceService instance with get_prices(),
                trading_day_return(), and trading_day_alpha() methods.
                Uses trading-day offsets (correct behavior).
            horizons: Tuple of forward-looking days (default: 7, 30, 90).
            benchmark: Benchmark ticker for alpha computation (default "SPY").

        Returns:
            Dict of "ticker:date" -> {"T+7": float, "T+7_alpha": float, ...}
        """
        if price_service is not None:
            return self._compute_returns_with_service(
                price_service, horizons, benchmark
            )
        if price_fetcher is not None:
            return self._compute_returns_with_fetcher(
                price_fetcher, horizons, benchmark
            )

        logger.info(
            "No price_service or price_fetcher provided — forward returns "
            "will be empty. Pass a PriceService for trading-day accuracy."
        )
        return {}

    def _compute_returns_with_service(
        self,
        price_service,
        horizons: Tuple[int, ...],
        benchmark: str,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Compute forward returns using PriceService (trading-day offsets).

        Batch-downloads all required price data in a single call, then
        computes T+N returns where N is measured in trading days.
        """
        if not self.history:
            return {}

        # Collect all unique tickers and determine date range
        all_tickers = set()
        all_dates = []
        for entry in self.history:
            date_str = entry["date"]
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                all_dates.append(dt)
            except (ValueError, TypeError):
                continue
            for stock in entry["concordance"]:
                ticker = stock.get("ticker", "")
                if ticker:
                    all_tickers.add(ticker)

        if not all_tickers or not all_dates:
            return {}

        # Add benchmark to the ticker set
        all_tickers.add(benchmark)

        # Calculate date range: earliest date to latest + buffer for max horizon
        start_date = min(all_dates)
        max_horizon = max(horizons)
        # Buffer: trading days * 1.5 + 10 calendar days for weekends/holidays
        buffer_days = int(max_horizon * 1.5 + 10)
        end_date = max(all_dates) + timedelta(days=buffer_days)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Batch download all prices in one call
        try:
            prices = price_service.get_prices(
                sorted(all_tickers),
                start_str,
                end_str,
            )
        except Exception as exc:
            logger.error("PriceService.get_prices() failed: %s", exc)
            return {}

        # Compute returns for each (ticker, date) pair
        results: Dict[str, Dict[str, Optional[float]]] = {}

        for entry in self.history:
            date_str = entry["date"]
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            for stock in entry["concordance"]:
                ticker = stock.get("ticker", "")
                if not ticker:
                    continue

                region = stock.get("region")
                key = f"{ticker}:{date_str}"
                returns = {}

                for h in horizons:
                    ret = price_service.trading_day_return(prices, ticker, dt, h)
                    if ret is not None:
                        returns[f"T+{h}"] = round(ret, 2)
                        alpha = price_service.trading_day_alpha(
                            prices, ticker, dt, h, region=region
                        )
                        returns[f"T+{h}_alpha"] = (
                            round(alpha, 2) if alpha is not None else None
                        )
                    else:
                        returns[f"T+{h}"] = None
                        returns[f"T+{h}_alpha"] = None

                if any(
                    v is not None
                    for k, v in returns.items()
                    if not k.endswith("_alpha")
                ):
                    results[key] = returns

        self.forward_returns = results
        logger.info(
            "Computed forward returns (PriceService) for %d recommendations",
            len(results),
        )
        return results

    def _compute_returns_with_fetcher(
        self,
        price_fetcher,
        horizons: Tuple[int, ...],
        benchmark: str,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Compute forward returns using legacy price_fetcher (calendar-day offsets).

        This is the original implementation kept for backward compatibility.
        Note: uses timedelta(days=h) which counts calendar days, not trading days.
        Prefer _compute_returns_with_service() for accurate results.
        """
        results: Dict[str, Dict[str, Optional[float]]] = {}

        # Cache benchmark prices per (date, horizon) to avoid redundant fetches
        benchmark_cache: Dict[str, Optional[float]] = {}

        for entry in self.history:
            date_str = entry["date"]
            try:
                base_date = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            # Fetch benchmark base price once per date
            bm_base_key = f"{benchmark}:{date_str}"
            if bm_base_key not in benchmark_cache:
                benchmark_cache[bm_base_key] = price_fetcher(benchmark, date_str)
            bm_base_price = benchmark_cache[bm_base_key]

            for stock in entry["concordance"]:
                ticker = stock.get("ticker", "")
                if not ticker:
                    continue

                key = f"{ticker}:{date_str}"

                # Get base price
                base_price = price_fetcher(ticker, date_str)
                if base_price is None or base_price <= 0:
                    continue

                returns = {}
                for h in horizons:
                    target_date = (base_date + timedelta(days=h)).strftime("%Y-%m-%d")
                    target_price = price_fetcher(ticker, target_date)
                    if target_price is not None and target_price > 0:
                        ret = (target_price - base_price) / base_price * 100
                        returns[f"T+{h}"] = round(ret, 2)

                        # Compute alpha vs benchmark
                        bm_target_key = f"{benchmark}:{target_date}"
                        if bm_target_key not in benchmark_cache:
                            benchmark_cache[bm_target_key] = price_fetcher(
                                benchmark, target_date
                            )
                        bm_target_price = benchmark_cache[bm_target_key]
                        if (bm_base_price and bm_base_price > 0
                                and bm_target_price and bm_target_price > 0):
                            bm_ret = (bm_target_price - bm_base_price) / bm_base_price * 100
                            returns[f"T+{h}_alpha"] = round(ret - bm_ret, 2)
                        else:
                            returns[f"T+{h}_alpha"] = None
                    else:
                        returns[f"T+{h}"] = None
                        returns[f"T+{h}_alpha"] = None

                if any(v is not None for k, v in returns.items() if not k.endswith('_alpha')):
                    results[key] = returns

        self.forward_returns = results
        logger.info("Computed forward returns for %d recommendations", len(results))
        return results

    def evaluate_performance(
        self,
        horizon: str = "T+30",
    ) -> Dict[str, Any]:
        """
        Evaluate committee performance by action group.

        For each action (BUY, ADD, HOLD, TRIM, SELL), computes:
        - Count of recommendations
        - Hit rate (% that moved in the right direction, absolute)
        - Alpha hit rate (% that beat/trailed benchmark, regime-neutral)
        - Average return and alpha at the specified horizon
        - Best and worst outcomes

        Args:
            horizon: Forward return horizon to evaluate (default "T+30").

        Returns:
            Performance summary dict.
        """
        action_returns: Dict[str, List[float]] = {}
        action_alphas: Dict[str, List[float]] = {}
        action_details: Dict[str, List[Dict]] = {}

        alpha_key = f"{horizon}_alpha"

        for entry in self.history:
            date_str = entry["date"]
            for stock in entry["concordance"]:
                ticker = stock.get("ticker", "")
                action = stock.get("verdict", stock.get("action", "HOLD"))
                conviction = stock.get("conviction", 50)
                key = f"{ticker}:{date_str}"

                ret_data = self.forward_returns.get(key, {})
                ret = ret_data.get(horizon)
                if ret is None:
                    continue

                alpha = ret_data.get(alpha_key)

                action_returns.setdefault(action, []).append(ret)
                if alpha is not None:
                    action_alphas.setdefault(action, []).append(alpha)
                action_details.setdefault(action, []).append({
                    "ticker": ticker,
                    "date": date_str,
                    "conviction": conviction,
                    "return": ret,
                    "alpha": alpha,
                })

        summary: Dict[str, Any] = {}
        for action, returns in action_returns.items():
            if not returns:
                continue

            # Absolute hit rate: BUY/ADD positive, SELL/TRIM negative
            if action in ("BUY", "ADD"):
                hits = sum(1 for r in returns if r > 0)
            elif action in ("SELL", "TRIM"):
                hits = sum(1 for r in returns if r < 0)
            else:
                hits = sum(1 for r in returns if abs(r) < 5)  # HOLD: stable

            hit_rate = hits / len(returns) * 100

            # Alpha hit rate: regime-neutral (did it beat the benchmark?)
            alphas = action_alphas.get(action, [])
            if alphas:
                if action in ("BUY", "ADD"):
                    alpha_hits = sum(1 for a in alphas if a > 0)
                elif action in ("SELL", "TRIM"):
                    alpha_hits = sum(1 for a in alphas if a < 0)
                else:
                    alpha_hits = sum(1 for a in alphas if abs(a) < 2)
                alpha_hit_rate = round(alpha_hits / len(alphas) * 100, 1)
                avg_alpha = round(sum(alphas) / len(alphas), 2)
            else:
                alpha_hit_rate = None
                avg_alpha = None

            avg_return = sum(returns) / len(returns)
            sorted_returns = sorted(returns)

            summary[action] = {
                "count": len(returns),
                "hit_rate": round(hit_rate, 1),
                "alpha_hit_rate": alpha_hit_rate,
                "avg_return": round(avg_return, 2),
                "avg_alpha": avg_alpha,
                "median_return": round(sorted_returns[len(sorted_returns) // 2], 2),
                "best": round(sorted_returns[-1], 2),
                "worst": round(sorted_returns[0], 2),
                "positive_pct": round(
                    sum(1 for r in returns if r > 0) / len(returns) * 100, 1
                ),
            }

        return {
            "horizon": horizon,
            "total_recommendations": sum(
                s["count"] for s in summary.values()
            ),
            "actions": summary,
            "history_entries": len(self.history),
        }

    def sweep_parameter(
        self,
        param_name: str,
        values: List[Any],
        horizon: str = "T+30",
    ) -> List[Dict[str, Any]]:
        """
        Test different values for a conviction parameter.

        Re-evaluates historical concordance data with modified parameters
        to find the optimal value. This is the core calibration function.

        Currently supports:
        - "buy_floor": BUY signal conviction floor (default 55)
        - "hold_cap": HOLD signal conviction cap (default 70)
        - "bonus_cap": Maximum bonus points (default 20)
        - "penalty_cap": Maximum penalty points (default 25)
        - "risk_buy_weight": Risk manager BUY weight (default 1.2)

        Args:
            param_name: Name of the parameter to sweep.
            values: List of values to test.
            horizon: Forward return horizon (default "T+30").

        Returns:
            List of {value, hit_rate, avg_return, count} for each parameter value.
        """
        if not self.history or not self.forward_returns:
            logger.warning(
                "No history or forward returns loaded — "
                "call load_history() and compute_forward_returns() first"
            )
            return []

        from trade_modules.committee_synthesis import (
            count_agent_votes,
            determine_base_conviction,
        )

        results = []

        for value in values:
            buy_returns = []
            sell_returns = []

            for entry in self.history:
                date_str = entry["date"]
                for stock in entry["concordance"]:
                    ticker = stock.get("ticker", "")
                    signal = stock.get("signal", "H")
                    conviction = stock.get("conviction", 50)
                    bull_pct = stock.get("bull_pct", 50)
                    fund_score = stock.get("fund_score", 50)
                    excess_exret = stock.get("excess_exret", 0)
                    bear_ratio = stock.get("bear_weight", 2.5) / (
                        stock.get("bull_weight", 2.5) + stock.get("bear_weight", 2.5)
                    )

                    # Re-compute base with modified parameter
                    if param_name == "buy_floor":
                        if signal == "B":
                            # Override the floor
                            new_base = determine_base_conviction(
                                bull_pct, signal, fund_score, excess_exret, bear_ratio,
                            )
                            new_base = max(new_base, value)  # Apply test floor
                            new_conviction = (
                                new_base
                                + stock.get("bonuses", 0)
                                - stock.get("penalties", 0)
                            )
                        else:
                            new_conviction = conviction
                    elif param_name == "hold_cap":
                        if signal == "H":
                            agent_base = int(30 + (bull_pct / 100) * 50)
                            new_base = min(agent_base, value)
                            new_conviction = (
                                new_base
                                + stock.get("bonuses", 0)
                                - stock.get("penalties", 0)
                            )
                        else:
                            new_conviction = conviction
                    else:
                        new_conviction = conviction

                    # Would this change the action?
                    key = f"{ticker}:{date_str}"
                    ret_data = self.forward_returns.get(key, {})
                    ret = ret_data.get(horizon)
                    if ret is None:
                        continue

                    if signal == "B" and new_conviction >= 55:
                        buy_returns.append(ret)
                    elif signal == "S" and new_conviction >= 60:
                        sell_returns.append(ret)

            buy_hit = (
                sum(1 for r in buy_returns if r > 0) / len(buy_returns) * 100
                if buy_returns else 0
            )
            sell_hit = (
                sum(1 for r in sell_returns if r < 0) / len(sell_returns) * 100
                if sell_returns else 0
            )
            buy_avg = (
                sum(buy_returns) / len(buy_returns)
                if buy_returns else 0
            )

            results.append({
                "param": param_name,
                "value": value,
                "buy_count": len(buy_returns),
                "buy_hit_rate": round(buy_hit, 1),
                "buy_avg_return": round(buy_avg, 2),
                "sell_count": len(sell_returns),
                "sell_hit_rate": round(sell_hit, 1),
            })

        return results

    def generate_calibration_report(self) -> Dict[str, Any]:
        """
        Generate calibration report with walk-forward validation.

        Splits history 70/30 into train/test. Evaluates on both halves
        independently so calibration decisions can be validated.
        """
        from trade_modules.backtest_stats import walk_forward_split

        perf_7d = self.evaluate_performance("T+7")
        perf_30d = self.evaluate_performance("T+30")

        sweeps = {}
        for param, values in [
            ("buy_floor", [45, 50, 55, 60, 65]),
            ("hold_cap", [60, 65, 70, 75, 80]),
        ]:
            sweep_result = self.sweep_parameter(param, values)
            if sweep_result:
                sweeps[param] = sweep_result

        # Walk-forward validation
        walk_forward = {"train_entries": 0, "test_entries": 0}
        if len(self.history) >= 4:
            train, test = walk_forward_split(self.history, train_ratio=0.7)
            walk_forward["train_entries"] = len(train)
            walk_forward["test_entries"] = len(test)
            walk_forward["train_date_range"] = (
                f"{train[0]['date']} to {train[-1]['date']}" if train else ""
            )
            walk_forward["test_date_range"] = (
                f"{test[0]['date']} to {test[-1]['date']}" if test else ""
            )

            # Evaluate performance on test set only
            if test and self.forward_returns:
                test_returns = {}
                for entry in test:
                    for stock in entry["concordance"]:
                        key = f"{stock.get('ticker', '')}:{entry['date']}"
                        if key in self.forward_returns:
                            test_returns[key] = self.forward_returns[key]

                # Temporarily swap for evaluation
                original_returns = self.forward_returns
                original_history = self.history
                self.forward_returns = test_returns
                self.history = test

                test_perf = self.evaluate_performance("T+30")
                walk_forward["test_performance_30d"] = test_perf

                # Restore
                self.forward_returns = original_returns
                self.history = original_history

        # Recommendations
        recommendations = []
        if perf_30d.get("actions", {}).get("BUY", {}).get("hit_rate", 0) < 50:
            recommendations.append(
                "BUY hit rate below 50% at T+30 — consider raising buy_floor"
            )
        if perf_30d.get("actions", {}).get("SELL", {}).get("hit_rate", 0) < 60:
            recommendations.append(
                "SELL hit rate below 60% at T+30 — review SELL criteria"
            )

        buy_data = perf_30d.get("actions", {}).get("BUY", {})
        if buy_data.get("avg_return", 0) < 0:
            recommendations.append(
                "BUY average return is negative at T+30 — system is not "
                "selecting winners; review conviction scoring"
            )

        # Check for walk-forward divergence
        if walk_forward.get("test_performance_30d"):
            test_buy = walk_forward["test_performance_30d"].get(
                "actions", {}
            ).get("BUY", {})
            full_buy = perf_30d.get("actions", {}).get("BUY", {})
            if (
                test_buy.get("hit_rate", 0) > 0
                and full_buy.get("hit_rate", 0) > 0
            ):
                divergence = full_buy["hit_rate"] - test_buy["hit_rate"]
                if divergence > 15:
                    recommendations.append(
                        f"OVERFITTING WARNING: Full-sample BUY hit rate "
                        f"({full_buy['hit_rate']:.0f}%) is {divergence:.0f}pp "
                        f"higher than out-of-sample ({test_buy['hit_rate']:.0f}%). "
                        f"Parameter tuning may not generalize."
                    )

        if not recommendations:
            recommendations.append(
                "All metrics within acceptable ranges"
            )

        return {
            "generated_at": datetime.now().isoformat(),
            "history_entries": len(self.history),
            "forward_returns_computed": len(self.forward_returns),
            "performance_7d": perf_7d,
            "performance_30d": perf_30d,
            "parameter_sweeps": sweeps,
            "walk_forward": walk_forward,
            "recommendations": recommendations,
        }


    def backfill_from_synthesis(
        self,
        synthesis_path: Optional[Path] = None,
    ) -> int:
        """
        Backfill historical concordance from existing synthesis.json files.

        Scans for synthesis.json (and synthesis_previous.json) and creates
        dated concordance archives in history/ for backtesting.

        Returns:
            Number of entries backfilled.
        """
        directory = self.log_dir
        history_dir = directory / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        # Find all synthesis files
        synth_files = list(directory.glob("**/synthesis*.json"))
        for fpath in synth_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            concordance = data.get("concordance", [])
            if not concordance:
                continue

            # Get date — from data, or from filename, or from mtime
            date_str = None
            if isinstance(concordance, list) and concordance:
                # Check if synthesis has a date key
                date_str = data.get("date")
            if not date_str:
                mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                date_str = mtime.strftime("%Y-%m-%d")

            archive_path = history_dir / f"concordance-{date_str}.json"
            if archive_path.exists():
                continue  # Don't overwrite existing archives

            archive_data = {
                "date": date_str,
                "version": data.get("version", "backfilled"),
                "regime": data.get("regime", "UNKNOWN"),
                "concordance": concordance,
            }
            try:
                with open(archive_path, "w") as f:
                    json.dump(archive_data, f, indent=2)
                count += 1
                logger.info("Backfilled %s from %s", archive_path.name, fpath.name)
            except OSError:
                pass

        # Also extract from concordance.json if it has a date
        conc_path = directory / "concordance.json"
        if conc_path.exists():
            try:
                with open(conc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    return count  # Bare list format — no date metadata to backfill
                date_str = data.get("date")
                stocks = data.get("stocks", {})
                if date_str and stocks:
                    archive_path = history_dir / f"concordance-{date_str}.json"
                    if not archive_path.exists():
                        concordance = [
                            {"ticker": k, **v} for k, v in stocks.items()
                        ]
                        archive_data = {
                            "date": date_str,
                            "version": "backfilled-concordance",
                            "concordance": concordance,
                        }
                        with open(archive_path, "w") as f:
                            json.dump(archive_data, f, indent=2)
                        count += 1
            except (json.JSONDecodeError, OSError):
                pass

        return count


def run_backtest(
    log_dir: Optional[Path] = None,
    horizon: str = "T+30",
    fetch_prices: bool = True,
    use_price_service: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run a full backtest.

    Args:
        log_dir: Committee output directory.
        horizon: Forward return horizon (default "T+30").
        fetch_prices: Whether to fetch prices at all.
        use_price_service: If True, uses PriceService (trading-day offsets).
            If False, falls back to legacy yfinance_price_fetcher (calendar days).
    """
    bt = CommitteeBacktester(log_dir=log_dir)

    # Step 1: Backfill
    backfilled = bt.backfill_from_synthesis()
    if backfilled:
        logger.info("Backfilled %d concordance archives", backfilled)

    # Step 2: Load history
    loaded = bt.load_history()
    if loaded < 2:
        return {
            "status": "insufficient_data",
            "history_entries": loaded,
            "message": (
                f"Only {loaded} committee snapshot(s) found. Need at least 2 "
                "for meaningful backtesting."
            ),
        }

    # Step 3: Forward returns
    if fetch_prices:
        if use_price_service:
            from trade_modules.price_service import PriceService
            svc = PriceService()
            bt.compute_forward_returns(price_service=svc)
        else:
            bt.compute_forward_returns(price_fetcher=yfinance_price_fetcher)

    if not bt.forward_returns:
        return {
            "status": "no_returns",
            "history_entries": loaded,
            "message": (
                f"Loaded {loaded} snapshots but could not compute forward "
                "returns. Some recommendations may be too recent for "
                f"{horizon} evaluation."
            ),
        }

    # Step 4: Calibration report
    report = bt.generate_calibration_report()
    report["status"] = "complete"
    return report


def yfinance_price_fetcher(
    ticker: str,
    date_str: str,
    _cache: Optional[Dict] = None,
) -> Optional[float]:
    """
    Default price fetcher using yfinance (CIO v6.0 G1).

    Without a concrete price fetcher, the backtesting framework is a skeleton
    that has never been used with real data. This function makes backtesting
    operational by providing a yfinance-based implementation with caching.

    Uses a module-level cache dict to avoid redundant API calls — multiple
    date lookups for the same ticker reuse the same price history.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        date_str: Date string in YYYY-MM-DD format.
        _cache: Optional external cache dict. If None, uses module-level cache.

    Returns:
        Closing price on or near the given date, or None if unavailable.
    """
    if _cache is None:
        if not hasattr(yfinance_price_fetcher, "_price_cache"):
            yfinance_price_fetcher._price_cache = {}
        _cache = yfinance_price_fetcher._price_cache

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not available for price fetching")
        return None

    # Fetch and cache full history per ticker
    if ticker not in _cache:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty:
                _cache[ticker] = {}
            else:
                _cache[ticker] = {
                    d.strftime("%Y-%m-%d"): float(p)
                    for d, p in hist["Close"].items()
                }
        except Exception as exc:
            logger.debug("Failed to fetch history for %s: %s", ticker, exc)
            _cache[ticker] = {}

    prices = _cache.get(ticker, {})
    if not prices:
        return None

    # Exact match first
    if date_str in prices:
        return prices[date_str]

    # Nearest date within 5 calendar days (handles weekends/holidays)
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

    best_date = None
    best_delta = 999
    for d_str in prices:
        try:
            d = datetime.strptime(d_str, "%Y-%m-%d")
            delta = abs((d - target).days)
            if delta < best_delta and delta <= 5:
                best_delta = delta
                best_date = d_str
        except ValueError:
            continue

    return prices.get(best_date) if best_date else None


def evaluate_recent(
    current_prices: Dict[str, float],
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate the most recent committee run against current market prices.

    CIO v12.0 P1: Closes the feedback loop by computing realized returns
    for the most recent concordance, suitable for inclusion in the next
    committee report's context. Unlike run_backtest() which fetches prices
    for all historical snapshots, this only needs current prices (which
    the committee skill already has from portfolio.csv).

    Args:
        current_prices: Dict of ticker -> current price (from portfolio.csv PRC column).
        log_dir: Committee output directory. Defaults to ~/.weirdapps-trading/committee.

    Returns:
        Dict with per-action hit rates, avg returns, and per-stock details.
        Returns {"status": "no_history"} if no previous concordance exists.
    """
    directory = log_dir or DEFAULT_COMMITTEE_LOG_DIR

    # Load most recent concordance
    conc_path = directory / "concordance.json"
    if not conc_path.exists():
        return {"status": "no_history"}

    try:
        with open(conc_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"status": "no_history"}

    # Normalize format
    if isinstance(prev, list):
        entries = prev
        date_str = "unknown"
    elif isinstance(prev, dict) and "stocks" in prev:
        date_str = prev.get("date", "unknown")
        entries = [dict(v, ticker=k) for k, v in prev["stocks"].items()]
    elif isinstance(prev, dict):
        date_str = prev.get("date", "unknown")
        entries = prev.get("concordance", [])
    else:
        return {"status": "no_history"}

    if not entries:
        return {"status": "no_history"}

    # Compute returns
    action_returns: Dict[str, List[Dict]] = {}
    for entry in entries:
        ticker = entry.get("ticker", "")
        if not ticker:
            continue
        prev_price = entry.get("price", 0)
        curr_price = current_prices.get(ticker, 0)
        if not prev_price or prev_price <= 0 or not curr_price or curr_price <= 0:
            continue

        ret_pct = round((curr_price - prev_price) / prev_price * 100, 2)
        action = entry.get("action", "HOLD")
        conviction = entry.get("conviction", 50)

        action_returns.setdefault(action, []).append({
            "ticker": ticker,
            "conviction": conviction,
            "prev_price": prev_price,
            "curr_price": curr_price,
            "return_pct": ret_pct,
        })

    # Compute per-action stats
    summary = {}
    for action, details in action_returns.items():
        returns = [d["return_pct"] for d in details]
        if not returns:
            continue

        if action in ("BUY", "ADD"):
            hits = sum(1 for r in returns if r > 0)
        elif action in ("SELL", "TRIM"):
            hits = sum(1 for r in returns if r < 0)
        else:
            hits = sum(1 for r in returns if abs(r) < 5)

        hit_rate = hits / len(returns) * 100
        avg_ret = sum(returns) / len(returns)

        # Sort by absolute return for best/worst
        details.sort(key=lambda d: d["return_pct"])

        summary[action] = {
            "count": len(returns),
            "hit_rate": round(hit_rate, 1),
            "avg_return": round(avg_ret, 2),
            "best": details[-1] if details else None,
            "worst": details[0] if details else None,
        }

    return {
        "status": "complete",
        "prev_committee_date": date_str,
        "total_evaluated": sum(s["count"] for s in summary.values()),
        "actions": summary,
    }

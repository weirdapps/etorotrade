"""
Portfolio Risk Analysis Module

Provides portfolio-level risk metrics that complement individual stock signals.
Uses historical returns from yfinance to calculate correlations.

P0 Improvement - Implemented from HEDGE_FUND_REVIEW.md recommendations.

A portfolio manager should never look at stocks in isolation. Position sizing
without correlation context is incomplete risk management.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioRiskAnalyzer:
    """
    Analyze portfolio-level risk metrics.

    Provides:
    - Correlation matrix calculation from historical returns
    - Sector concentration tracking
    - Concentration risk warnings
    - Portfolio beta calculation
    - High correlation pair identification
    """

    # Default configuration
    DEFAULT_MAX_SECTOR_CONCENTRATION = 0.25  # 25% max per sector
    DEFAULT_CORRELATION_THRESHOLD = 0.70  # Flag pairs above 70% correlation
    DEFAULT_LOOKBACK_DAYS = 252  # 1 year of trading days

    def __init__(
        self,
        max_sector_concentration: float = DEFAULT_MAX_SECTOR_CONCENTRATION,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ):
        """
        Initialize the portfolio risk analyzer.

        Args:
            max_sector_concentration: Maximum allowed sector weight (0-1)
            correlation_threshold: Minimum correlation to flag (0-1)
            lookback_days: Number of trading days for correlation calculation
        """
        self.max_sector_concentration = max_sector_concentration
        self.correlation_threshold = correlation_threshold
        self.lookback_days = lookback_days
        self._price_cache: Dict[str, pd.Series] = {}

    def _get_historical_prices(
        self, ticker: str, period_days: int
    ) -> Optional[pd.Series]:
        """
        Get historical closing prices for a ticker.

        Args:
            ticker: Stock ticker symbol
            period_days: Number of days of history to fetch

        Returns:
            Series of closing prices indexed by date, or None if unavailable
        """
        # Check cache first
        cache_key = f"{ticker}_{period_days}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{period_days}d")

            if hist.empty or len(hist) < 20:
                logger.debug(f"Insufficient history for {ticker}: {len(hist)} days")
                return None

            prices = hist["Close"]
            self._price_cache[cache_key] = prices
            return prices

        except Exception as e:
            logger.debug(f"Failed to get history for {ticker}: {e}")
            return None

    def calculate_correlation_matrix(
        self, tickers: List[str], period: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlations using historical returns.

        Args:
            tickers: List of ticker symbols
            period: Number of trading days for correlation calc (default: lookback_days)

        Returns:
            DataFrame with pairwise correlations, empty if insufficient data
        """
        if not tickers or len(tickers) < 2:
            return pd.DataFrame()

        period = period or self.lookback_days
        returns_data: Dict[str, pd.Series] = {}

        for ticker in tickers:
            prices = self._get_historical_prices(ticker, period)
            if prices is not None and len(prices) > 20:
                # Calculate daily returns
                returns = prices.pct_change().dropna()
                if len(returns) > 10:
                    returns_data[ticker] = returns

        if len(returns_data) < 2:
            logger.debug(
                f"Insufficient data for correlation: {len(returns_data)}/{len(tickers)} tickers"
            )
            return pd.DataFrame()

        # Align dates across all tickers
        returns_df = pd.DataFrame(returns_data)

        # Drop rows with any NaN (misaligned dates)
        returns_df = returns_df.dropna()

        if len(returns_df) < 20:
            logger.debug(f"Insufficient aligned data: {len(returns_df)} rows")
            return pd.DataFrame()

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def get_sector_concentration(
        self, portfolio_df: pd.DataFrame, sector_col: str = "sector"
    ) -> Dict[str, float]:
        """
        Calculate sector weights as percentage of portfolio.

        Args:
            portfolio_df: DataFrame with position data
            sector_col: Column name containing sector info

        Returns:
            Dict mapping sector -> percentage weight (0-1)
        """
        if portfolio_df.empty:
            return {}

        if sector_col not in portfolio_df.columns:
            # Try common variations
            for col in ["SECTOR", "Sector", "sector", "GICS_SECTOR"]:
                if col in portfolio_df.columns:
                    sector_col = col
                    break
            else:
                logger.debug(f"No sector column found in DataFrame")
                return {}

        # Filter out None/NaN sectors
        valid_df = portfolio_df[portfolio_df[sector_col].notna()].copy()

        if valid_df.empty:
            return {}

        # Use market cap for weighting if available
        weight_col = None
        for col in ["market_cap", "CAP", "MARKET_CAP", "cap"]:
            if col in valid_df.columns:
                weight_col = col
                break

        if weight_col:
            # Parse market cap if it's a string
            if valid_df[weight_col].dtype == object:
                from .analysis.tiers import _parse_market_cap

                valid_df["_weight"] = valid_df[weight_col].apply(_parse_market_cap)
            else:
                valid_df["_weight"] = valid_df[weight_col]

            sector_weights = valid_df.groupby(sector_col)["_weight"].sum()
            total = sector_weights.sum()

            if total > 0:
                return (sector_weights / total).to_dict()

        # Equal weight fallback
        return valid_df[sector_col].value_counts(normalize=True).to_dict()

    def flag_concentration_risks(
        self, portfolio_df: pd.DataFrame, sector_col: str = "sector"
    ) -> List[str]:
        """
        Return warnings for concentrated positions.

        Args:
            portfolio_df: DataFrame with position data
            sector_col: Column name containing sector info

        Returns:
            List of warning strings for concentrated sectors
        """
        warnings = []
        sector_weights = self.get_sector_concentration(portfolio_df, sector_col)

        for sector, weight in sorted(
            sector_weights.items(), key=lambda x: x[1], reverse=True
        ):
            if weight > self.max_sector_concentration:
                warnings.append(
                    f"CONCENTRATION WARNING: {sector} at {weight*100:.1f}% "
                    f"(max: {self.max_sector_concentration*100:.0f}%)"
                )

        return warnings

    def calculate_portfolio_beta(
        self, portfolio_df: pd.DataFrame, benchmark: str = "SPY"
    ) -> Optional[float]:
        """
        Calculate weighted average portfolio beta.

        Args:
            portfolio_df: DataFrame with 'ticker' and 'beta' columns
            benchmark: Benchmark ticker (default SPY)

        Returns:
            Portfolio beta or None if cannot calculate
        """
        # Find beta column
        beta_col = None
        for col in ["beta", "BETA", "Beta"]:
            if col in portfolio_df.columns:
                beta_col = col
                break

        if beta_col is None:
            return None

        # Filter valid beta values
        valid = portfolio_df[
            portfolio_df[beta_col].notna()
            & (portfolio_df[beta_col] > 0)
            & (portfolio_df[beta_col] < 10)  # Sanity check
        ].copy()

        if valid.empty:
            return None

        # Use market cap weighting if available
        weight_col = None
        for col in ["market_cap", "CAP", "MARKET_CAP", "cap"]:
            if col in valid.columns:
                weight_col = col
                break

        if weight_col:
            # Parse market cap if string
            if valid[weight_col].dtype == object:
                from .analysis.tiers import _parse_market_cap

                valid["_weight"] = valid[weight_col].apply(_parse_market_cap)
            else:
                valid["_weight"] = valid[weight_col]

            total_weight = valid["_weight"].sum()
            if total_weight > 0:
                weights = valid["_weight"] / total_weight
                return float((valid[beta_col] * weights).sum())

        # Equal weight fallback
        return float(valid[beta_col].mean())

    def identify_high_correlation_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find stock pairs with correlation above threshold.

        Args:
            correlation_matrix: Correlation matrix DataFrame
            threshold: Minimum correlation to flag (default: correlation_threshold)

        Returns:
            List of (ticker1, ticker2, correlation) tuples, sorted by correlation desc
        """
        threshold = threshold or self.correlation_threshold
        pairs: List[Tuple[str, str, float]] = []

        if correlation_matrix.empty:
            return pairs

        tickers = correlation_matrix.columns.tolist()

        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i + 1 :]:
                corr = correlation_matrix.loc[ticker1, ticker2]
                if abs(corr) >= threshold:
                    pairs.append((ticker1, ticker2, float(corr)))

        # Sort by absolute correlation, descending
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    # Tier-specific expected volatility (annualized)
    TIER_EXPECTED_VOLATILITY = {
        "MEGA": 0.15,
        "LARGE": 0.20,
        "MID": 0.25,
        "SMALL": 0.30,
        "MICRO": 0.35,
        # V/G/B tier system mapping
        "V": 0.175,   # Value (~MEGA/LARGE blend)
        "G": 0.25,    # Growth (~MID)
        "B": 0.325,   # Bets (~SMALL/MICRO blend)
    }

    def calculate_effective_concentration(
        self, tickers: List[str], period: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate effective number of independent positions using correlations.

        Uses the diversification ratio: effective_positions = n^2 / sum(all correlations).

        Args:
            tickers: List of ticker symbols
            period: Number of trading days for correlation calc

        Returns:
            Dict with effective_positions, diversification_ratio, correlated_clusters
        """
        result: Dict[str, Any] = {
            "effective_positions": len(tickers),
            "diversification_ratio": 1.0,
            "correlated_clusters": [],
        }

        if len(tickers) < 2:
            return result

        corr_matrix = self.calculate_correlation_matrix(tickers, period=period)
        if corr_matrix.empty:
            return result

        n = len(corr_matrix)
        # Sum of all pairwise correlations (including diagonal = 1.0 for each)
        total_correlation = corr_matrix.values.sum()

        if total_correlation > 0:
            effective = (n * n) / total_correlation
            result["effective_positions"] = round(effective, 1)
            result["diversification_ratio"] = round(effective / n, 2)

        # Find correlated clusters
        result["correlated_clusters"] = self.flag_correlation_clusters(
            corr_matrix, min_cluster_size=3, threshold=0.75
        )

        return result

    def flag_correlation_clusters(
        self,
        correlation_matrix: pd.DataFrame,
        min_cluster_size: int = 3,
        threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Find groups of 3+ stocks with mutual correlation above threshold.

        Args:
            correlation_matrix: Pairwise correlation DataFrame
            min_cluster_size: Minimum stocks to form a cluster
            threshold: Minimum correlation to consider stocks related

        Returns:
            List of cluster dicts with tickers, avg_correlation, combined_weight_warning
        """
        if correlation_matrix.empty or len(correlation_matrix) < min_cluster_size:
            return []

        tickers = correlation_matrix.columns.tolist()
        clusters: List[Dict[str, Any]] = []
        used_tickers: set = set()

        # For each ticker, find all tickers correlated above threshold
        neighbor_map: Dict[str, set] = {}
        for ticker in tickers:
            neighbors = set()
            for other in tickers:
                if other != ticker:
                    if abs(correlation_matrix.loc[ticker, other]) >= threshold:
                        neighbors.add(other)
            neighbor_map[ticker] = neighbors

        # Find clusters: groups where all members are mutually correlated
        for ticker in tickers:
            if ticker in used_tickers:
                continue
            candidates = neighbor_map[ticker] | {ticker}
            # Filter to only those mutually correlated with each other
            cluster_tickers = [ticker]
            for candidate in sorted(candidates - {ticker}):
                # Check if candidate is correlated with all current cluster members
                is_mutual = all(
                    abs(correlation_matrix.loc[candidate, member]) >= threshold
                    for member in cluster_tickers
                )
                if is_mutual:
                    cluster_tickers.append(candidate)

            if len(cluster_tickers) >= min_cluster_size:
                # Calculate average pairwise correlation within cluster
                corr_sum = 0.0
                pair_count = 0
                for i, t1 in enumerate(cluster_tickers):
                    for t2 in cluster_tickers[i + 1:]:
                        corr_sum += abs(correlation_matrix.loc[t1, t2])
                        pair_count += 1

                avg_corr = corr_sum / pair_count if pair_count > 0 else 0.0

                clusters.append({
                    "tickers": cluster_tickers,
                    "avg_correlation": round(avg_corr, 2),
                    "combined_weight_warning": (
                        f"{len(cluster_tickers)} stocks acting as ~1 position "
                        f"(avg correlation {avg_corr:.0%})"
                    ),
                })
                used_tickers.update(cluster_tickers)

        return clusters

    def check_drawdowns(
        self,
        portfolio_df: pd.DataFrame,
        tier_col: str = "tier",
    ) -> List[Dict[str, Any]]:
        """
        Check for stocks with drawdowns exceeding tier-expected volatility.

        Uses the 52-week high percentage to measure drawdown, then compares
        against expected volatility for each tier.

        Args:
            portfolio_df: DataFrame with portfolio positions
            tier_col: Column name containing tier classification

        Returns:
            List of dicts with ticker, drawdown_pct, tier, expected_vol, severity
        """
        alerts: List[Dict[str, Any]] = []

        if portfolio_df.empty:
            return alerts

        # Find the 52-week high column
        high_col = None
        for col in ["52W", "pct_52w_high", "52w", "PCT_52W_HIGH"]:
            if col in portfolio_df.columns:
                high_col = col
                break

        if high_col is None:
            return alerts

        # Find ticker column
        tkr_col = None
        for col in ["TKR", "ticker", "TICKER", "symbol"]:
            if col in portfolio_df.columns:
                tkr_col = col
                break

        if tkr_col is None:
            return alerts

        # Determine tier for each stock
        cap_col = None
        for col in [tier_col, "tier", "TIER", "CAP", "cap", "market_cap"]:
            if col in portfolio_df.columns:
                cap_col = col
                break

        for _, row in portfolio_df.iterrows():
            ticker = row.get(tkr_col)
            high_val = row.get(high_col)

            if pd.isna(ticker) or pd.isna(high_val):
                continue

            # Parse 52W value - it's percentage of 52-week high (e.g., 72 means at 72% of high)
            try:
                pct_of_high = float(high_val)
            except (ValueError, TypeError):
                continue

            # Drawdown = how far below the 52-week high (100 - pct_of_high)
            drawdown_pct = (100 - pct_of_high) / 100.0  # Convert to decimal

            if drawdown_pct <= 0:
                continue  # At or above 52-week high

            # Determine tier and expected volatility
            tier = self._resolve_tier(row, cap_col)
            expected_vol = self.TIER_EXPECTED_VOLATILITY.get(tier, 0.25)

            # Determine severity
            if drawdown_pct > 2.0 * expected_vol:
                severity = "CRITICAL"
            elif drawdown_pct > 1.5 * expected_vol:
                severity = "WARNING"
            elif drawdown_pct > 1.0 * expected_vol:
                severity = "WATCH"
            else:
                continue  # Within normal range

            alerts.append({
                "ticker": ticker,
                "drawdown_pct": round(drawdown_pct * 100, 1),
                "tier": tier,
                "expected_vol": round(expected_vol * 100, 1),
                "severity": severity,
            })

        # Sort by severity (CRITICAL first) then drawdown
        severity_order = {"CRITICAL": 0, "WARNING": 1, "WATCH": 2}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x["drawdown_pct"]))

        return alerts

    def _resolve_tier(self, row: pd.Series, cap_col: Optional[str]) -> str:
        """Resolve the tier for a row based on available columns."""
        if cap_col and cap_col in ("tier", "TIER"):
            tier_val = row.get(cap_col)
            if not pd.isna(tier_val) and str(tier_val) in self.TIER_EXPECTED_VOLATILITY:
                return str(tier_val)

        # Try to determine from market cap
        cap_val = row.get(cap_col) if cap_col else None
        if cap_val is not None and not pd.isna(cap_val):
            try:
                if isinstance(cap_val, str):
                    from .analysis.tiers import _parse_market_cap
                    cap_num = _parse_market_cap(cap_val)
                else:
                    cap_num = float(cap_val)

                if cap_num >= 500_000_000_000:
                    return "MEGA"
                elif cap_num >= 100_000_000_000:
                    return "LARGE"
                elif cap_num >= 10_000_000_000:
                    return "MID"
                elif cap_num >= 2_000_000_000:
                    return "SMALL"
                else:
                    return "MICRO"
            except (ValueError, TypeError):
                pass

        return "MID"  # Default fallback

    def get_risk_summary(
        self, portfolio_df: pd.DataFrame, ticker_col: str = "ticker"
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for a portfolio.

        Args:
            portfolio_df: DataFrame with portfolio positions
            ticker_col: Column name containing ticker symbols

        Returns:
            Dictionary with risk metrics and warnings
        """
        summary: Dict[str, Any] = {
            "portfolio_beta": None,
            "sector_concentration": {},
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "correlation_matrix_available": False,
            "effective_concentration": None,
            "correlation_clusters": [],
            "drawdown_alerts": [],
        }

        if portfolio_df.empty:
            return summary

        # Find ticker column
        for col in ["ticker", "TICKER", "Ticker", "symbol", "SYMBOL", "TKR"]:
            if col in portfolio_df.columns:
                ticker_col = col
                break

        # Calculate portfolio beta
        summary["portfolio_beta"] = self.calculate_portfolio_beta(portfolio_df)

        # Calculate sector concentration
        summary["sector_concentration"] = self.get_sector_concentration(portfolio_df)

        # Check for concentration risks
        summary["concentration_warnings"] = self.flag_concentration_risks(portfolio_df)

        # Calculate correlations if we have tickers
        if ticker_col in portfolio_df.columns:
            tickers = portfolio_df[ticker_col].dropna().unique().tolist()
            if len(tickers) >= 2:
                corr_matrix = self.calculate_correlation_matrix(tickers[:50])  # Limit
                if not corr_matrix.empty:
                    summary["correlation_matrix_available"] = True
                    summary["high_correlation_pairs"] = (
                        self.identify_high_correlation_pairs(corr_matrix)
                    )

            # Effective concentration
            if len(tickers) >= 2:
                summary["effective_concentration"] = (
                    self.calculate_effective_concentration(tickers[:50])
                )
                summary["correlation_clusters"] = (
                    summary["effective_concentration"].get("correlated_clusters", [])
                )

        # Drawdown alerts
        summary["drawdown_alerts"] = self.check_drawdowns(portfolio_df)

        return summary

    def format_risk_report(self, summary: Dict[str, Any]) -> List[str]:
        """
        Format risk summary as human-readable report lines.

        Args:
            summary: Risk summary from get_risk_summary()

        Returns:
            List of formatted report lines
        """
        lines: List[str] = []

        # Portfolio beta
        if summary.get("portfolio_beta") is not None:
            beta = summary["portfolio_beta"]
            risk_level = (
                "LOW" if beta < 0.8 else "MODERATE" if beta < 1.2 else "HIGH"
            )
            lines.append(f"Portfolio Beta: {beta:.2f} ({risk_level} volatility)")

        # Concentration warnings
        for warning in summary.get("concentration_warnings", []):
            lines.append(warning)

        # High correlation pairs
        pairs = summary.get("high_correlation_pairs", [])
        if pairs:
            lines.append(f"High Correlation Pairs ({len(pairs)} found):")
            for ticker1, ticker2, corr in pairs[:5]:  # Top 5
                lines.append(f"  {ticker1} <-> {ticker2}: {corr:.2f}")
            if len(pairs) > 5:
                lines.append(f"  ... and {len(pairs) - 5} more")

        # Effective concentration
        eff_conc = summary.get("effective_concentration")
        if eff_conc:
            eff_pos = eff_conc.get("effective_positions", 0)
            div_ratio = eff_conc.get("diversification_ratio", 0)
            lines.append(
                f"Effective Independent Positions: {eff_pos} "
                f"(diversification ratio: {div_ratio:.0%})"
            )

        # Correlation clusters
        clusters = summary.get("correlation_clusters", [])
        if clusters:
            lines.append(f"Correlation Clusters ({len(clusters)} found):")
            for cluster in clusters:
                tickers_str = ", ".join(cluster["tickers"])
                lines.append(
                    f"  [{tickers_str}] avg corr: {cluster['avg_correlation']:.2f} "
                    f"- {cluster['combined_weight_warning']}"
                )

        # Drawdown alerts
        alerts = summary.get("drawdown_alerts", [])
        if alerts:
            lines.append(f"Drawdown Alerts ({len(alerts)} found):")
            for alert in alerts:
                lines.append(
                    f"  {alert['severity']}: {alert['ticker']} "
                    f"down {alert['drawdown_pct']:.1f}% from 52W high "
                    f"(tier {alert['tier']}, expected vol {alert['expected_vol']:.0f}%)"
                )

        return lines


# Convenience functions for easy access


def analyze_portfolio_risk(portfolio_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to analyze portfolio risk.

    Args:
        portfolio_df: DataFrame with portfolio positions

    Returns:
        Risk summary dictionary
    """
    analyzer = PortfolioRiskAnalyzer()
    return analyzer.get_risk_summary(portfolio_df)


def get_concentration_warnings(
    portfolio_df: pd.DataFrame, max_concentration: float = 0.25
) -> List[str]:
    """
    Get sector concentration warnings for a portfolio.

    Args:
        portfolio_df: DataFrame with portfolio positions
        max_concentration: Maximum allowed sector weight (0-1)

    Returns:
        List of warning strings
    """
    analyzer = PortfolioRiskAnalyzer(max_sector_concentration=max_concentration)
    return analyzer.flag_concentration_risks(portfolio_df)


def get_high_correlation_stocks(
    tickers: List[str], threshold: float = 0.70
) -> List[Tuple[str, str, float]]:
    """
    Find highly correlated stock pairs.

    Args:
        tickers: List of ticker symbols
        threshold: Minimum correlation to flag

    Returns:
        List of (ticker1, ticker2, correlation) tuples
    """
    analyzer = PortfolioRiskAnalyzer(correlation_threshold=threshold)
    corr_matrix = analyzer.calculate_correlation_matrix(tickers)
    return analyzer.identify_high_correlation_pairs(corr_matrix)

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

        # Find clusters: groups where members are mostly mutually correlated
        # CIO v3 F8: Relaxed from strict mutual (all pairs) to 2/3 majority
        # to capture hub-and-spoke patterns (e.g., AAPL correlated with all
        # FAANG but META-GOOGL weaker)
        for ticker in tickers:
            if ticker in used_tickers:
                continue
            candidates = neighbor_map[ticker] | {ticker}
            # Filter to those correlated with majority of current cluster members
            cluster_tickers = [ticker]
            for candidate in sorted(candidates - {ticker}):
                if len(cluster_tickers) == 1:
                    # First addition only needs correlation with seed ticker
                    if abs(correlation_matrix.loc[candidate, cluster_tickers[0]]) >= threshold:
                        cluster_tickers.append(candidate)
                else:
                    # Subsequent additions need correlation with ≥2/3 of cluster
                    mutual_count = sum(
                        abs(correlation_matrix.loc[candidate, member]) >= threshold
                        for member in cluster_tickers
                    )
                    required = max(1, int(len(cluster_tickers) * 0.67))
                    if mutual_count >= required:
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
        previous_drawdowns: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check for stocks with drawdowns exceeding tier-expected volatility.

        Uses the 52-week high percentage to measure drawdown, then compares
        against expected volatility for each tier.

        CIO v3 F12: When previous_drawdowns are provided, stocks that have
        recovered >20% of their drawdown get severity downgraded (e.g.,
        CRITICAL → WARNING). This prevents over-reaction on recovering positions.

        Args:
            portfolio_df: DataFrame with portfolio positions
            tier_col: Column name containing tier classification
            previous_drawdowns: Optional dict mapping ticker -> previous drawdown_pct
                (from a prior check_drawdowns call). Used for recovery tracking.

        Returns:
            List of dicts with ticker, drawdown_pct, tier, expected_vol, severity,
            and optionally recovery_pct, previous_drawdown_pct
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

            alert = {
                "ticker": ticker,
                "drawdown_pct": round(drawdown_pct * 100, 1),
                "tier": tier,
                "expected_vol": round(expected_vol * 100, 1),
                "severity": severity,
            }

            # CIO v3 F12: Recovery tracking
            if previous_drawdowns and ticker in previous_drawdowns:
                prev_dd = previous_drawdowns[ticker]
                current_dd = alert["drawdown_pct"]
                if prev_dd > 0 and current_dd < prev_dd:
                    recovery_pct = round((prev_dd - current_dd) / prev_dd * 100, 1)
                    alert["previous_drawdown_pct"] = prev_dd
                    alert["recovery_pct"] = recovery_pct
                    # Downgrade severity if recovered >20% of drawdown
                    if recovery_pct >= 20:
                        downgrade_map = {
                            "CRITICAL": "WARNING",
                            "WARNING": "WATCH",
                        }
                        new_severity = downgrade_map.get(severity)
                        if new_severity:
                            alert["severity"] = new_severity
                            alert["recovery_note"] = (
                                f"Downgraded from {severity}: "
                                f"recovered {recovery_pct:.0f}% of drawdown"
                            )

            alerts.append(alert)

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

    def calculate_portfolio_var(
        self,
        portfolio_df: pd.DataFrame,
        horizon_days: int = 1,
        confidence_95: bool = True,
        confidence_99: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate Portfolio Value at Risk (VaR) using parametric method.

        Uses the existing correlation matrix and individual stock volatilities
        to compute portfolio-level VaR at 95% and 99% confidence levels.

        Args:
            portfolio_df: DataFrame with portfolio positions
            horizon_days: Time horizon for VaR calculation (default: 1 day)
            confidence_95: Include 95% VaR (default: True)
            confidence_99: Include 99% VaR (default: True)

        Returns:
            Dict with var_95, var_99, var_95_pct, var_99_pct, portfolio_vol, effective_positions
        """
        result: Dict[str, Any] = {
            "var_95": None,
            "var_99": None,
            "var_95_pct": None,
            "var_99_pct": None,
            "cvar_95_pct": None,  # CIO v3 F7: Expected Shortfall
            "cvar_99_pct": None,
            "portfolio_vol": None,
            "effective_positions": None,
            "var_alert": False,
        }

        if portfolio_df.empty:
            return result

        # Find ticker column
        tkr_col = None
        for col in ["TKR", "ticker", "TICKER", "symbol"]:
            if col in portfolio_df.columns:
                tkr_col = col
                break

        if tkr_col is None:
            return result

        tickers = portfolio_df[tkr_col].dropna().unique().tolist()
        if len(tickers) < 2:
            return result

        # CIO Review v3 F6: Use 252-day window for correlation (captures full
        # market cycles) but 60-day window for volatility (recent vol estimate).
        # Previously both used 60-day, which understated tail risk.
        corr_matrix = self.calculate_correlation_matrix(tickers, period=252)
        if corr_matrix.empty:
            # Fallback to shorter period if insufficient history
            corr_matrix = self.calculate_correlation_matrix(tickers, period=60)
        if corr_matrix.empty:
            return result

        # Calculate individual stock volatilities from recent returns (60-day)
        returns_data: Dict[str, pd.Series] = {}
        for ticker in corr_matrix.columns:
            prices = self._get_historical_prices(ticker, period_days=90)
            if prices is not None and len(prices) > 20:
                returns = prices.pct_change().dropna()
                if len(returns) > 10:
                    returns_data[ticker] = returns

        if len(returns_data) < 2:
            return result

        # Align returns and calculate volatilities using recent 60 days
        returns_df = pd.DataFrame(returns_data).dropna()
        if len(returns_df) < 20:
            return result

        # Use last 60 days for volatility estimate (recent market conditions)
        recent_returns = returns_df.tail(60) if len(returns_df) > 60 else returns_df
        volatilities = recent_returns.std()

        # Determine weights (equal weights if position sizes not available)
        weights_dict = {}
        weight_col = None
        for col in ["SZ", "size", "position_size", "market_cap", "CAP"]:
            if col in portfolio_df.columns:
                weight_col = col
                break

        if weight_col:
            # Use position sizes as weights
            for _, row in portfolio_df.iterrows():
                ticker = row.get(tkr_col)
                if ticker in corr_matrix.columns:
                    try:
                        size_val = row.get(weight_col)
                        if pd.notna(size_val):
                            if isinstance(size_val, str):
                                # Parse percentage like "2.5%"
                                size_val = float(size_val.strip('%')) / 100.0
                            weights_dict[ticker] = float(size_val)
                    except (ValueError, TypeError):
                        pass

        if not weights_dict:
            # Equal weights fallback
            n = len(corr_matrix)
            weights_dict = {ticker: 1.0 / n for ticker in corr_matrix.columns}

        # Normalize weights to sum to 1
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v / total_weight for k, v in weights_dict.items()}

        # Align weights with correlation matrix
        tickers_aligned = [t for t in corr_matrix.columns if t in weights_dict]
        if len(tickers_aligned) < 2:
            return result

        weights = np.array([weights_dict[t] for t in tickers_aligned])
        vols = volatilities[tickers_aligned].values
        corr_aligned = corr_matrix.loc[tickers_aligned, tickers_aligned].values

        # Calculate covariance matrix
        # Cov = diag(vol) @ Corr @ diag(vol)
        vol_diag = np.diag(vols)
        cov_matrix = vol_diag @ corr_aligned @ vol_diag

        # Portfolio variance: w^T @ Cov @ w
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # Annualize for reporting
        portfolio_vol_annual = portfolio_vol * np.sqrt(252)
        result["portfolio_vol"] = round(portfolio_vol_annual, 4)

        # Scale volatility by horizon
        portfolio_vol_horizon = portfolio_vol * np.sqrt(horizon_days)

        # VaR calculation: VaR = z * σ * sqrt(horizon/252)
        # Note: We already have daily vol, so we scale by sqrt(horizon_days)
        # For a 1-day horizon with daily vol, this is just z * daily_vol

        if confidence_95:
            z_95 = 1.645
            var_95_pct = z_95 * portfolio_vol_horizon
            result["var_95_pct"] = round(var_95_pct * 100, 2)
            # If we had portfolio value, we'd calculate dollar VaR
            # For now, report as percentage
            result["var_95"] = result["var_95_pct"]

        if confidence_99:
            z_99 = 2.326
            var_99_pct = z_99 * portfolio_vol_horizon
            result["var_99_pct"] = round(var_99_pct * 100, 2)
            result["var_99"] = result["var_99_pct"]

        # CIO Review v3 F7: Expected Shortfall (CVaR/ES)
        # Average loss beyond VaR threshold — more informative than VaR alone
        portfolio_returns = returns_df[tickers_aligned].dot(weights)
        if len(portfolio_returns) >= 20:
            if confidence_95:
                var_threshold_95 = np.percentile(portfolio_returns, 5)
                tail_95 = portfolio_returns[portfolio_returns <= var_threshold_95]
                if len(tail_95) > 0:
                    result["cvar_95_pct"] = round(abs(tail_95.mean()) * 100, 2)

            if confidence_99:
                var_threshold_99 = np.percentile(portfolio_returns, 1)
                tail_99 = portfolio_returns[portfolio_returns <= var_threshold_99]
                if len(tail_99) > 0:
                    result["cvar_99_pct"] = round(abs(tail_99.mean()) * 100, 2)

        # Alert if VaR exceeds 12% threshold
        if result["var_95_pct"] and result["var_95_pct"] > 12.0:
            result["var_alert"] = True

        # Effective positions (for context)
        eff_conc = self.calculate_effective_concentration(tickers_aligned, period=60)
        result["effective_positions"] = eff_conc.get("effective_positions")

        return result

    def get_drawdown_actions(
        self, drawdown_alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on drawdown alerts.

        Connects drawdown severity to portfolio re-evaluation decisions:
        - CRITICAL: Force sell review with tightened thresholds
        - WARNING: Flag for review and close monitoring
        - WATCH: Monitor without immediate action

        Args:
            drawdown_alerts: Output from check_drawdowns()

        Returns:
            List of action dicts with ticker, severity, action, and recommendation
        """
        actions: List[Dict[str, Any]] = []

        for alert in drawdown_alerts:
            ticker = alert.get("ticker")
            severity = alert.get("severity")
            drawdown_pct = alert.get("drawdown_pct")

            action_dict: Dict[str, Any] = {
                "ticker": ticker,
                "severity": severity,
                "drawdown_pct": drawdown_pct,
            }

            if severity == "CRITICAL":
                action_dict["action"] = "FORCE_SELL_REVIEW"
                action_dict["recommendation"] = (
                    f"Immediate review required. Consider tightening sell thresholds "
                    f"by 20% (multiply by 0.8) for this position. Drawdown exceeds "
                    f"2x expected volatility."
                )
                action_dict["threshold_adjustment"] = 0.8

            elif severity == "WARNING":
                action_dict["action"] = "REVIEW"
                action_dict["recommendation"] = (
                    f"Position under stress. Review fundamentals and analyst changes. "
                    f"Drawdown exceeds 1.5x expected volatility. Watch closely for "
                    f"deterioration to CRITICAL."
                )
                action_dict["threshold_adjustment"] = None

            elif severity == "WATCH":
                action_dict["action"] = "MONITOR"
                action_dict["recommendation"] = (
                    f"Normal volatility range exceeded. Monitor for trend continuation. "
                    f"No immediate action required."
                )
                action_dict["threshold_adjustment"] = None

            else:
                # Unknown severity
                action_dict["action"] = "UNKNOWN"
                action_dict["recommendation"] = "Review manually."
                action_dict["threshold_adjustment"] = None

            actions.append(action_dict)

        return actions

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
            "portfolio_var": None,
            "drawdown_actions": [],
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

        # Portfolio VaR
        summary["portfolio_var"] = self.calculate_portfolio_var(portfolio_df)

        # Drawdown actions
        if summary["drawdown_alerts"]:
            summary["drawdown_actions"] = self.get_drawdown_actions(
                summary["drawdown_alerts"]
            )

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

        # Portfolio VaR
        var_data = summary.get("portfolio_var")
        if var_data and var_data.get("var_95_pct") is not None:
            var_95 = var_data["var_95_pct"]
            var_99 = var_data.get("var_99_pct")
            portfolio_vol = var_data.get("portfolio_vol")

            var_line = f"Portfolio VaR: {var_95:.2f}% at 95% confidence"
            if var_99:
                var_line += f", {var_99:.2f}% at 99%"
            if portfolio_vol:
                var_line += f" (annual vol: {portfolio_vol*100:.1f}%)"
            lines.append(var_line)

            # Expected Shortfall (CIO v3 F7)
            cvar_95 = var_data.get("cvar_95_pct")
            cvar_99 = var_data.get("cvar_99_pct")
            if cvar_95 is not None:
                cvar_line = f"Expected Shortfall: {cvar_95:.2f}% at 95%"
                if cvar_99 is not None:
                    cvar_line += f", {cvar_99:.2f}% at 99%"
                lines.append(cvar_line)

            # Alert if VaR exceeds threshold
            if var_data.get("var_alert"):
                lines.append(
                    f"  WARNING: VaR exceeds 12% threshold - portfolio at elevated risk"
                )

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

        # Drawdown actions
        actions = summary.get("drawdown_actions", [])
        if actions:
            lines.append(f"Drawdown Actions ({len(actions)} recommendations):")
            for action in actions:
                lines.append(
                    f"  {action['ticker']} ({action['severity']}): "
                    f"{action['action']}"
                )
                lines.append(f"    → {action['recommendation']}")
                if action.get("threshold_adjustment"):
                    lines.append(
                        f"    → Suggested threshold multiplier: "
                        f"{action['threshold_adjustment']:.1f}"
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

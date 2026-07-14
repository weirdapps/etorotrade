"""v3 portfolio construction bridge + report-only risk gate.

Turns the v3 combiner output (a scored frame carrying ``conviction`` /
``eligible`` / ``sector`` / ``price`` / ``beta`` / ``country``) into a
risk-first book by calling the :mod:`trade_modules.riskfirst` construction
PRIMITIVES directly, so that conviction participates in sizing:

  * dual-listing dedup — never hold both listings of one company; the
    home (mother-market) listing is kept (Change 3);
  * a conviction-tilt blend that sits between Equal-Risk-Contribution (ERC)
    weights and conviction-proportional weights, so the #1-conviction name is
    no longer handed the smallest weight by pure risk parity (Change 1);
  * single-name / USD-bloc / sector caps in the same order the riskfirst engine
    applies them;
  * regime deployment — the final capped book is scaled to a ``gross_target``
    fraction of capital (85-95% by regime), replacing fractional Kelly
    (Change 2);
  * an empirical shrunk-covariance estimate from the supplied price history,
    falling back to a single-factor beta covariance whenever a selected name
    lacks usable price history;
  * a pre-gate report-only assessment — parametric-normal CVaR(95%), net beta,
    and the effective number of bets — kept for oversight (it only sets flags);
    and
  * a Phase 5A HARD risk gate (:mod:`trade_modules.v3.risk_gate`) that ENFORCES
    the vol ceiling and concentration caps on the deployed book (caps to
    convergence -> tail de-weight to the ceiling -> gross-cut fallback). The
    returned book is the gated book; ``diagnostics["gate"]`` holds its report.

Pure orchestration: no network access and no ``yahoofinance.core.config``
import (module-level or otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.riskfirst.construct import apply_name_cap, cap_groups, erc_weights
from trade_modules.riskfirst.covariance import single_factor_cov
from trade_modules.riskfirst.fx import USD_BLOC, cap_bloc, currency_of
from trade_modules.riskfirst.prices import daily_returns, shrunk_cov
from trade_modules.v3.risk_gate import _Z_ES_95, apply_risk_gate

# Exchange-suffix -> company domicile country, for dual-listing (mother-market)
# dedup. A bare ticker (no suffix) or an unmapped suffix is treated as US.
_SUFFIX_COUNTRY = {
    ".PA": "France",
    ".DE": "Germany",
    ".L": "United Kingdom",
    ".AS": "Netherlands",
    ".SW": "Switzerland",
    ".MI": "Italy",
    ".MC": "Spain",
    ".OL": "Norway",
    ".ST": "Sweden",
    ".CO": "Denmark",
    ".HE": "Finland",
    ".IR": "Ireland",
    ".BR": "Belgium",
    ".LS": "Portugal",
    ".VI": "Austria",
    ".WA": "Poland",
    ".AT": "Greece",
    ".HK": "Hong Kong",
    ".T": "Japan",
    ".KS": "South Korea",
    ".KQ": "South Korea",
    ".TW": "Taiwan",
    ".SS": "China",
    ".SZ": "China",
    ".SI": "Singapore",
    ".NS": "India",
    ".BO": "India",
    ".AX": "Australia",
    ".TO": "Canada",
    ".V": "Canada",
    ".SA": "Brazil",
    ".MX": "Mexico",
    ".BA": "Argentina",
    ".SN": "Chile",
}
_HOME_COUNTRY = "United States"  # bare / unmapped suffix

# Company/listing country -> economic region, for the geographic concentration cap.
# Kept consistent with the report's allocation bars (both driven by ``region_of``).
_COUNTRY_REGION = {
    "United States": "North America",
    "Canada": "North America",
    "France": "Europe",
    "Germany": "Europe",
    "United Kingdom": "Europe",
    "Netherlands": "Europe",
    "Switzerland": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Norway": "Europe",
    "Sweden": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Ireland": "Europe",
    "Belgium": "Europe",
    "Portugal": "Europe",
    "Austria": "Europe",
    "Poland": "Europe",
    "Greece": "Europe",
    "Hong Kong": "Asia-Pacific",
    "Japan": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "Taiwan": "Asia-Pacific",
    "China": "Asia-Pacific",
    "Singapore": "Asia-Pacific",
    "India": "Asia-Pacific",
    "Australia": "Asia-Pacific",
    "Brazil": "Latin America",
    "Mexico": "Latin America",
    "Argentina": "Latin America",
    "Chile": "Latin America",
}
_DEFAULT_REGION = "North America"  # bare/unmapped US listing


def region_of(ticker: str) -> str:
    """Economic region for a ticker, from its LISTING venue (exchange suffix).

    Domicile drives dual-listing dedup elsewhere; the geographic cap and the
    report's allocation bars both use the trading venue via this one function,
    so the two never diverge. Crypto (``-USD`` / hyphen) -> "Crypto/Global".
    Bare or unmapped US-style tickers -> North America.
    """
    t = str(ticker or "").upper()
    if t.endswith("-USD") or t.endswith("-EUR"):  # crypto (BTC-USD); not BRK-B
        return "Crypto/Global"
    # Own suffix lookup (not _suffix_of, which drops single-letter suffixes as share
    # classes): a mapped single-letter exchange suffix like .T (Tokyo) / .L (London)
    # / .V (Canada) IS a real venue. Unmapped suffixes (.B share class) fall to US.
    country = (
        _SUFFIX_COUNTRY.get("." + t.rsplit(".", 1)[-1], _HOME_COUNTRY)
        if "." in t
        else _HOME_COUNTRY
    )
    return _COUNTRY_REGION.get(country, _DEFAULT_REGION)


# _Z_ES_95 imported from risk_gate (single definition shared across both modules).
# Report-only risk-gate thresholds.
_BETA_BAND = (0.3, 1.1)
_MIN_EFFECTIVE_BETS = 12
# Minimum overlapping return observations before the empirical cov is trusted.
_MIN_COV_OBS = 60
# Single-factor fallback assumptions (mirror engine.select_and_construct defaults).
_MARKET_VOL = 0.18
_IDIO_VOL = 0.25


def _make_cov_fn(prices: pd.DataFrame, scored: pd.DataFrame):
    """Build ``cov_fn(selected) -> cov`` for the engine.

    Prefers the empirical shrunk covariance from ``prices``; falls back to the
    single-factor beta covariance whenever any selected name lacks usable price
    history (missing column, empty frame, or too few overlapping observations).
    The returned matrix is aligned to the order of ``selected``.
    """

    def cov_fn(selected) -> np.ndarray:
        sel = list(selected)
        have_cols = (
            prices is not None and not prices.empty and all(t in prices.columns for t in sel)
        )
        if have_cols:
            rets = daily_returns(prices[sel]).dropna(how="any")
            if len(rets) >= _MIN_COV_OBS:
                return shrunk_cov(rets)
        betas = pd.to_numeric(scored.reindex(sel)["beta"], errors="coerce").fillna(1.0).to_numpy()
        return single_factor_cov(betas, _MARKET_VOL, _IDIO_VOL)

    return cov_fn


def _sector_exposures(weights: pd.Series, sector_labels: dict) -> dict:
    """Aggregate (absolute) portfolio weight per uppercased sector label."""
    out: dict[str, float] = {}
    for tkr, w in weights.items():
        if w <= 1e-12:
            continue
        lab = sector_labels.get(tkr, "UNKNOWN")
        out[lab] = out.get(lab, 0.0) + float(w)
    return out


def _empty_result(cvar_budget: float) -> dict:
    """A well-formed all-cash result for a degenerate (no eligible names) input."""
    return {
        "weights": pd.Series(dtype=float),
        "gross": 0.0,
        "cash": 1.0,
        "usd_bloc": 0.0,
        "sector_exposures": {},
        "selected": [],
        "diagnostics": {
            "cvar_95_risk_book": 0.0,
            "cvar_95_deployed": 0.0,
            "cvar_budget": cvar_budget,
            "net_beta": 0.0,
            "net_beta_band": _BETA_BAND,
            "effective_bets": 0.0,
            "port_vol": 0.0,
            "gross_risk": 0.0,
            "binding": {
                "cvar": False,
                "net_beta": False,
                "effective_bets": False,
                "deployment_capped": False,
                "name_cap": False,
                "sector_cap": False,
                "usd_bloc_cap": False,
                "vol_over_target": False,
            },
        },
    }


def _root_of(ticker: str) -> str:
    """Ticker with the exchange suffix stripped (split on ``.``), uppercased.

    ``ABVX.PA`` -> ``ABVX``; ``SAP.DE`` -> ``SAP``; bare ``AAPL`` -> ``AAPL``.
    """
    return str(ticker).split(".")[0].upper()


def _suffix_of(ticker: str) -> str:
    """Uppercased exchange suffix (``.PA``), or ``""`` for a bare ticker.

    Single-letter suffixes (``.A``, ``.B``, ``.C``) are share-class designators,
    NOT exchange suffixes — they are excluded from the country map so that
    ``BRK.A`` and ``BRK.B`` are both classified as United States (bare/unmapped)
    rather than treated as foreign-exchange listings.
    """
    t = str(ticker)
    if "." not in t:
        return ""
    suffix = "." + t.rsplit(".", 1)[-1]
    upper = suffix.upper()
    # Single-letter suffix: share-class designator, not an exchange suffix.
    if len(upper) == 2:  # "." + exactly one char
        return ""
    return upper


def _listing_country(ticker: str) -> str:
    """Domicile the listing's exchange belongs to (bare/unmapped -> United States)."""
    return _SUFFIX_COUNTRY.get(_suffix_of(ticker), _HOME_COUNTRY)


def _norm_name(value) -> str:
    """Normalize a company name for equality: lowercase, punctuation -> space, collapse."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().replace(".", " ").replace(",", " ").split())


def _country_of(row) -> str:
    """The domicile ``country`` field of a scored row ("" when missing/NaN)."""
    if "country" not in row.index:
        return ""
    val = row["country"]
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip() if isinstance(val, str) else ""


def _same_company(a: str, b: str, scored: pd.DataFrame) -> bool:
    """True when two same-root listings are genuinely the same company.

    Requires matching root (already grouped) AND (normalized-equal name OR the
    same non-empty domicile country). Different companies sharing a root stay.
    """
    ra, rb = scored.loc[a], scored.loc[b]
    name_a, name_b = _norm_name(ra.get("name")), _norm_name(rb.get("name"))
    if name_a and name_b and name_a == name_b:
        return True
    ca, cb = _country_of(ra), _country_of(rb)
    return bool(ca and cb and ca == cb)


def _adv_of(row) -> float:
    """Best-available average dollar volume / volume for a tie-break (NaN when absent)."""
    for col in ("adv_usd", "avg_volume"):
        if col in row.index:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                return float(val)
    return float("nan")


def _tiebreak(candidates: list[str], scored: pd.DataFrame) -> str:
    """Keep the highest-ADV listing; fall back to deterministic (sorted first)."""
    if len(candidates) == 1:
        return candidates[0]
    advs = {c: _adv_of(scored.loc[c]) for c in candidates}
    if any(np.isfinite(v) for v in advs.values()):
        return max(candidates, key=lambda c: (advs[c] if np.isfinite(advs[c]) else -np.inf, c))
    return sorted(candidates)[0]


def _pick_mother(members: list[str], scored: pd.DataFrame) -> str:
    """Pick the mother/home-market listing to keep among same-company listings."""
    # Country-aware (preferred): keep the listing whose exchange matches the
    # company's domicile country.
    domiciles = [_country_of(scored.loc[m]) for m in members]
    domicile = next((c for c in domiciles if c), "")
    if domicile:
        matches = [m for m in members if _listing_country(m) == domicile]
        if matches:
            return _tiebreak(matches, scored)
    # Fallback (country missing / no exchange matches): prefer a suffixed home
    # listing over a bare US ADR; among suffixed keep highest ADV else sorted.
    suffixed = [m for m in members if _suffix_of(m)]
    if suffixed:
        return _tiebreak(suffixed, scored)
    return _tiebreak(members, scored)


def dedup_dual_listings(scored: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate listings of one company, keeping the mother-market listing.

    Groups tickers by root (suffix-stripped). Within a root group, only listings
    that are genuinely the SAME company — matching root AND (normalized-equal
    ``name`` OR same domicile ``country``) — are deduped; different companies
    sharing a root are all kept. For each same-company cluster the home-market
    listing is kept (see :func:`_pick_mother`) and the rest are dropped. Row
    order is preserved.
    """
    if scored is None or scored.empty:
        return scored

    tickers = list(scored.index)
    groups: dict[str, list[str]] = {}
    for t in tickers:
        groups.setdefault(_root_of(t), []).append(t)

    drop: set = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        # Union-find over the group: connect same-company pairs into clusters.
        parent = {m: m for m in members}

        def find(x, _parent=parent):
            while _parent[x] != x:
                _parent[x] = _parent[_parent[x]]
                x = _parent[x]
            return x

        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if _same_company(a, b, scored):
                    parent[find(a)] = find(b)

        clusters: dict[str, list[str]] = {}
        for m in members:
            clusters.setdefault(find(m), []).append(m)

        for cluster in clusters.values():
            if len(cluster) < 2:
                continue  # lone listing (distinct company) — keep it
            keep = _pick_mother(cluster, scored)
            drop.update(m for m in cluster if m != keep)

    if not drop:
        return scored
    return scored.loc[[t for t in tickers if t not in drop]]


def build_portfolio(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_n: int = 20,
    conviction_weight: float = 0.5,
    target_vol: float = 0.12,
    name_cap: float = 0.08,
    sector_cap: float = 0.25,
    usd_bloc_cap: float = 0.60,
    region_cap: float = 0.65,
    gross_target: float = 0.90,
    cvar_budget: float = 0.25,
    vol_ceiling: float = 0.18,
) -> dict:
    """Construct a risk-first book from a v3 scored frame + price history.

    Pipeline: dedup dual listings -> select ``top_n`` by conviction ->
    conviction-tilt blend of ERC and conviction weights -> name / USD-bloc /
    sector caps -> deploy to ``gross_target``.

    Args:
        scored: Output of :func:`trade_modules.v3.combine.compute_scores` — must
            carry ``conviction``, ``eligible``, ``sector``, ``price`` and
            ``beta`` columns (and optionally ``country`` / ``name`` for
            dual-listing dedup), indexed by ticker.
        prices: Daily closes (dates x tickers) for the covariance estimate.
        top_n: Number of top-conviction names to select.
        conviction_weight: Blend λ in ``(1-λ)·w_erc + λ·w_conv`` (Change 1).
            ``λ=0`` reproduces pure ERC; ``λ=1`` is conviction-proportional.
            ``w_conv[i] = max(conviction_i, 0) / Σ max(conviction_j, 0)`` over
            the selected names (uniform when all convictions are ≤ 0).
        target_vol: Annualised volatility budget used as a report-only flag.
            Sets ``binding["vol_over_target"]`` when the risk-book vol exceeds
            this threshold. The book is NEVER shrunk to meet it (vol/CVaR
            enforcement is deferred to Phase 5).
        name_cap / sector_cap / usd_bloc_cap: Binding concentration caps applied
            (on the gross=1.0 risk book) in the same order the riskfirst engine
            uses; the sector cap uses an uppercased ``SECTOR``.
        gross_target: Fraction of capital deployed (Change 2, replaces Kelly).
            The final capped book is scaled so ``sum(weights) == gross_target``
            and ``cash == 1 - gross_target``. The runner sets it per regime
            (85% risk_off / 90% neutral / 95% risk_on).
        cvar_budget: Annual parametric-normal CVaR(95%) ceiling for the risk
            book. A breach only sets ``binding["cvar"]`` — it NEVER shrinks the
            book (the CVaR flag stays report-only; hard VOL enforcement is done by
            the Phase 5A gate below).
        vol_ceiling: HARD annualised vol ceiling enforced by the Phase 5A risk
            gate (:func:`trade_modules.v3.risk_gate.apply_risk_gate`). After the
            conviction-tilt + caps + ``gross_target`` deployment, the book is run
            through the gate, which (1) drives all caps to convergence, (2)
            de-weights the worst tail-risk names until vol ≤ ceiling, and (3) as a
            fallback cuts gross below ``gross_target`` if de-weighting is
            exhausted. The RETURNED book is the GATED book; ``diagnostics["gate"]``
            carries the gate report and ``diagnostics`` still keeps the pre-gate
            assessment (``binding`` / ``port_vol`` / ``cvar_95_*``).

    Returns:
        ``{weights, gross, cash, usd_bloc, sector_exposures, selected,
        diagnostics}``. ``diagnostics`` holds:

        *Risk-book metrics* (the fully-invested, gross=1.0 sleeve):
        ``cvar_95_risk_book`` (= 2.063·``port_vol``, annual parametric-normal
        CVaR), ``net_beta``, ``effective_bets`` (= 1/Σwᵢ² on invested
        proportions), ``port_vol``, ``gross_risk``.

        *Deployed-capital metric*: ``cvar_95_deployed`` — CVaR of the book at
        the actual ``gross_target`` (≈ ``gross_target · cvar_95_risk_book``).

        CVaR / net-beta / effective-bets are reported for oversight only; none
        of them shrink the book.
    """
    # Change 3: drop duplicate listings (keep the mother market) before anything.
    scored = dedup_dual_listings(scored)

    if "eligible" in scored.columns:
        elig = scored[scored["eligible"].astype(bool)].copy()
    else:
        elig = scored.copy()
    if elig.empty:
        return _empty_result(cvar_budget)

    # Attach the uppercased SECTOR column the sector cap reads.
    sub = elig.copy()
    sub["SECTOR"] = sub["sector"].astype("string").str.upper()
    sector_labels = sub["SECTOR"].astype(str).to_dict()

    # --- selection: top_n by conviction (drop NaN, descending) ---
    conv_ranked = (
        pd.to_numeric(sub["conviction"], errors="coerce").dropna().sort_values(ascending=False)
    )
    selected = list(conv_ranked.index[:top_n])
    if not selected:
        return _empty_result(cvar_budget)

    cov = np.asarray(_make_cov_fn(prices, sub)(selected), dtype=float)

    # --- Change 1: conviction-tilt blend between ERC and conviction weights ---
    w_erc = erc_weights(cov)  # sum = 1
    conv_sel = (
        pd.to_numeric(sub.reindex(selected)["conviction"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy()
    )
    conv_total = float(conv_sel.sum())
    w_conv = (
        conv_sel / conv_total if conv_total > 0 else np.full(len(selected), 1.0 / len(selected))
    )
    lam = min(1.0, max(0.0, float(conviction_weight)))
    w = (1.0 - lam) * w_erc + lam * w_conv
    w_sum = float(w.sum())
    if w_sum > 0:
        w = w / w_sum  # renormalize to sum = 1

    # --- caps: name -> USD-bloc -> sector -> region (single-name reasserted between) ---
    is_bloc = np.array([currency_of(t) in USD_BLOC for t in selected])
    sec_arr = sub.reindex(selected)["SECTOR"].astype(str).to_numpy()
    region_arr = np.array([region_of(t) for t in selected])
    w = apply_name_cap(w, name_cap)
    w = cap_bloc(w, is_bloc, usd_bloc_cap)
    w = apply_name_cap(w, name_cap)  # keep single-name cap after bloc redistribution
    w = cap_groups(w, sec_arr, sector_cap)
    w = apply_name_cap(w, name_cap)
    w = cap_groups(w, region_arr, region_cap)
    w = apply_name_cap(w, name_cap)

    gross_risk = float(w.sum())  # capped risk-book gross (≈ 1.0)

    # --- report-only risk gate on the fully-invested (gross=1.0) risk book ---
    w_shape = w / gross_risk if gross_risk > 0 else w
    port_vol = float(np.sqrt(max(float(w_shape @ cov @ w_shape), 0.0)))
    cvar_95_risk_book = _Z_ES_95 * port_vol
    cvar_binding = bool(cvar_95_risk_book > cvar_budget)  # report only — never shrinks

    # Shape metrics on invested proportions (scale-invariant across the dials).
    if float(w_shape.sum()) > 0:
        effective_bets = float(1.0 / np.sum(w_shape**2))
    else:
        effective_bets = 0.0
    betas_sel = (
        pd.to_numeric(sub.reindex(selected)["beta"], errors="coerce")
        .fillna(1.0)  # consistent with cov_fn's fillna(1.0) for missing betas
        .to_numpy()
    )
    net_beta = float(np.dot(w_shape, betas_sel))

    # --- Change 2: deploy the risk book to gross_target (replaces Kelly) ---
    # FIX 1: use a scale factor that NEVER re-inflates a cap-limited book.
    # When caps forced gross_risk < gross_target (e.g. all-USD bloc with no
    # non-bloc receiver), the capped weights stay as-is (scale=1) and the
    # book stays at gross_risk < gross_target (deployment_capped flag set).
    # When caps did not limit deployment (gross_risk >= gross_target), we
    # scale uniformly DOWN to gross_target — uniform scaling preserves all
    # cap fractions unchanged.
    gross_target = min(1.0, max(0.0, float(gross_target)))
    scale = min(gross_target / gross_risk, 1.0) if gross_risk > 0 else 0.0
    w_final = w * scale  # sum = gross_risk * scale <= gross_target
    deployed = float(w_final.sum())
    cvar_95_deployed = _Z_ES_95 * float(np.sqrt(max(float(w_final @ cov @ w_final), 0.0)))

    # FIX 2: verify cap exposures on the deployed book (fractions of deployed).
    # w_shape = w / gross_risk = w_final / deployed (scale factor cancels), so
    # it is the shape of the deployed book regardless of the deployment target.
    _TOL = 1e-6
    if deployed > 0:
        _max_name = float(w_shape.max())
        sec_arr = sub.reindex(selected)["SECTOR"].astype(str).to_numpy()
        _sector_totals: dict[str, float] = {}
        for _wt, _sec in zip(w_shape, sec_arr, strict=True):
            _sector_totals[_sec] = _sector_totals.get(_sec, 0.0) + float(_wt)
        _max_sector = max(_sector_totals.values(), default=0.0)
        _usd_frac = float(
            sum(
                float(_wt)
                for _wt, _t in zip(w_shape, selected, strict=True)
                if currency_of(_t) in USD_BLOC
            )
        )
        _region_totals: dict[str, float] = {}
        for _wt, _t in zip(w_shape, selected, strict=True):
            _rg = region_of(_t)
            _region_totals[_rg] = _region_totals.get(_rg, 0.0) + float(_wt)
        _max_region = max(_region_totals.values(), default=0.0)
        _name_cap_breach = bool(_max_name > name_cap + _TOL)
        _sector_cap_breach = bool(_max_sector > sector_cap + _TOL)
        _usd_bloc_breach = bool(_usd_frac > usd_bloc_cap + _TOL)
        _region_cap_breach = bool(_max_region > region_cap + _TOL)
    else:
        _name_cap_breach = _sector_cap_breach = _usd_bloc_breach = False
        _region_cap_breach = False
        _max_region = 0.0

    diagnostics = {
        # Risk-book metrics: the fully-invested, gross=1.0 sleeve.
        "cvar_95_risk_book": cvar_95_risk_book,
        # Deployed-capital metric: CVaR at the actual deployment fraction.
        "cvar_95_deployed": cvar_95_deployed,
        "cvar_budget": cvar_budget,
        "net_beta": net_beta,
        "net_beta_band": _BETA_BAND,
        "effective_bets": effective_bets,
        "port_vol": port_vol,
        "gross_risk": gross_risk,
        "max_region": _max_region,
        "binding": {
            "cvar": cvar_binding,
            "net_beta": not (_BETA_BAND[0] <= net_beta <= _BETA_BAND[1]),
            "effective_bets": effective_bets < _MIN_EFFECTIVE_BETS,
            # FIX 1: deployment was cap-limited below gross_target.
            "deployment_capped": bool(gross_risk < gross_target - _TOL),
            # FIX 2: residual cap breaches on the deployed book (surfaced, never silent).
            "name_cap": _name_cap_breach,
            "sector_cap": _sector_cap_breach,
            "usd_bloc_cap": _usd_bloc_breach,
            "region_cap": _region_cap_breach,
            # FIX 3: vol budget flag (report-only; never shrinks the book).
            "vol_over_target": bool(port_vol > target_vol),
        },
    }

    # --- Phase 5A: HARD risk gate (blocking enforcement) on the deployed book ---
    # The diagnostics above are the PRE-gate assessment (report-only). The gate
    # now ENFORCES the vol ceiling + caps: caps-to-convergence -> tail de-weight
    # to the vol ceiling -> gross-cut fallback. The RETURNED book is the gated one.
    gated_series, gate_diag = apply_risk_gate(
        pd.Series(w_final, index=selected),
        cov,
        sectors=sub.reindex(selected)["SECTOR"].astype(str).to_numpy(),
        currencies=[currency_of(t) for t in selected],
        betas=betas_sel,
        conviction=pd.to_numeric(sub.reindex(selected)["conviction"], errors="coerce"),
        vol_ceiling=vol_ceiling,
        name_cap=name_cap,
        sector_cap=sector_cap,
        usd_bloc_cap=usd_bloc_cap,
    )
    diagnostics["gate"] = gate_diag

    # Post-gate region clamp: the risk gate is region-blind and its tail-deweight /
    # redistribution can re-concentrate a region above the cap. Only when a region
    # genuinely exceeds the cap do we re-assert it on the gated book (gross-preserving;
    # excess redistributes to under-cap names) + re-assert the name cap. In production
    # the book sits under the 65% region cap, so this is skipped and the gated book is
    # returned untouched (no perturbation of the gate's vol/name enforcement).
    gated_arr = gated_series.to_numpy()
    _gsum = float(gated_arr.sum())
    if _gsum > 0:
        _reg_tot: dict[str, float] = {}
        for _wt, _t in zip(gated_arr, selected, strict=True):
            _reg_tot[region_of(_t)] = _reg_tot.get(region_of(_t), 0.0) + float(_wt)
        if max(_reg_tot.values(), default=0.0) > region_cap * _gsum + 1e-9:
            gated_arr = cap_groups(gated_arr, region_arr, region_cap * _gsum)
            gated_arr = apply_name_cap(gated_arr, name_cap * _gsum)
            gated_series = pd.Series(gated_arr, index=selected)

    full = pd.Series(0.0, index=sub.index)
    full.loc[selected] = gated_series.to_numpy()
    gross = float(full.sum())

    usd_bloc = float(sum(v for t, v in full.items() if currency_of(t) in USD_BLOC))
    # Region exposures + breach on the FINAL gated book. The region cap is enforced
    # pre-gate (cap_groups in the cap sequence); the gate only reduces concentration
    # (tail de-weight + uniform gross-cut), so region stays within cap. We recompute
    # the breach flag on the gated book so the report reflects the truth, not the
    # pre-gate assessment.
    region_exposures: dict[str, float] = {}
    for t, v in full.items():
        if v > 1e-12:
            rg = region_of(t)
            region_exposures[rg] = region_exposures.get(rg, 0.0) + float(v)
    if gross > 0:
        _max_region_final = max(region_exposures.values(), default=0.0) / gross
        diagnostics["max_region"] = _max_region_final
        diagnostics["binding"]["region_cap"] = bool(_max_region_final > region_cap + _TOL)
    return {
        "weights": full,
        "gross": gross,
        "cash": max(0.0, 1.0 - gross),
        "usd_bloc": usd_bloc,
        "sector_exposures": _sector_exposures(full, sector_labels),
        "region_exposures": region_exposures,
        "selected": selected,
        "diagnostics": diagnostics,
    }

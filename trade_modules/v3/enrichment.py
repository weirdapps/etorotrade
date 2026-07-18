"""BUILD (2026-07-19): two-stage scoring — bound the yfinance enrichment cost.

④ expanded the scored universe ~25x (~2,580 names), but ``enrich_features`` fetches
per-ticker yfinance ``.info`` (~4s each) -> a full-universe run is ~3 hours and
throttle-prone: not production-viable.

Two-stage fix:
  1. ``panel_prescore`` — score the FULL coverage-gated universe on the panel-native
     factors only (PET/ROE/FCF/52W/B/AM/EG + offline sector), NO network (~0.3s).
     Conviction itself is eligibility-gated (needs price history), so we rank by the
     weighted cluster-z mean, which is populated for every name.
  2. ``select_enrichment_set`` — full-enrich (yfinance .info + prices) only holdings +
     analyst candidates + the top coverage names by pre-score, bounded by ``cap``.

This keeps ④'s selection BREADTH (the pre-rank sees all 2,580, so momentum names the
old analyst gate excluded can surface) while restoring old-report runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trade_modules.v3.combine import CLUSTER_WEIGHTS, CLUSTERS, compute_scores
from trade_modules.v3.constants import MIN_FACTOR_COVERAGE
from trade_modules.v3.features import enrich_features
from trade_modules.v3.universe import load_universe


def panel_prescore(etoro_csv_path, universe, sector_map=None) -> pd.Series:
    """Cheap, network-free panel-only conviction proxy for ranking.

    Enriches ``universe`` with NO-OP info/price/accruals fetchers (panel-native
    factors + offline sector only), then returns the per-ticker weighted mean of the
    six cluster-z columns (``CLUSTER_WEIGHTS``, renormalized per-row over the clusters
    present). Higher = stronger on the panel factors. NaN where no cluster is present.
    """
    feats = enrich_features(
        universe,
        etoro_csv_path,
        info_fetch=lambda _ts: {},
        price_fetch=lambda _ts, period="2y", **_k: pd.DataFrame(),
        accruals_fetch=lambda _ts: {},
        sector_map=sector_map,
    )
    scored = compute_scores(feats, sector_neutral=True)
    cluster_names = list(CLUSTERS.keys())
    zmat = scored[cluster_names]
    w = pd.Series(CLUSTER_WEIGHTS)[cluster_names]
    wmat = pd.DataFrame(
        np.tile(w.to_numpy(), (len(zmat), 1)), index=zmat.index, columns=cluster_names
    ).where(zmat.notna())
    wsum = wmat.sum(axis=1)
    return (zmat * wmat).sum(axis=1) / wsum.where(wsum > 0)


def select_enrichment_set(
    etoro_csv_path,
    holdings,
    candidates,
    *,
    cap: int = 500,
    sector_map=None,
    min_factor_coverage: int = MIN_FACTOR_COVERAGE,
) -> list[str]:
    """Bounded set to fully enrich: holdings + candidates + top coverage names.

    ``holdings`` and ``candidates`` are ALWAYS included (held names are never dropped;
    analyst candidates kept through the transition). Remaining budget up to ``cap`` is
    filled with the highest panel-pre-score names from the coverage-gated universe.
    De-duped; holdings/candidates first, then top coverage names.
    """
    universe = load_universe(etoro_csv_path, min_factor_coverage=min_factor_coverage)
    always = list(dict.fromkeys([str(t) for t in holdings] + [str(t) for t in candidates]))
    always_up = {t.upper() for t in always}

    prescore = panel_prescore(etoro_csv_path, universe, sector_map=sector_map)
    ranked = [str(t) for t in prescore.dropna().sort_values(ascending=False).index]
    top = [t for t in ranked if t.upper() not in always_up]

    slots = max(0, int(cap) - len(always))
    return list(dict.fromkeys(always + top[:slots]))

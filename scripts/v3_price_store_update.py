"""BUILD ⑤ (2026-07-19): grow the versioned append-only price store.

Fetches the current US universe's EUR adjusted closes and appends them to
``trade_modules.v3.price_store`` (``~/.weirdapps-trading/v3_price_store.parquet``).
Run daily (VPS / network allowed) so the store accumulates history AND retains the
bars of names that later leave the universe — the survivorship fix a backtest needs.

    .venv/bin/python scripts/v3_price_store_update.py

No module-level yahoofinance.core.config import (lazy imports in main()).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ETORO_CSV = "yahoofinance/output/etoro.csv"


def main() -> None:
    from trade_modules.v3.price_store import append_bars, store_coverage  # noqa: PLC0415
    from trade_modules.v3.prices import load_eur_close  # noqa: PLC0415
    from trade_modules.v3.universe import load_universe  # noqa: PLC0415

    tickers = load_universe(ETORO_CSV)
    print(f"universe: {len(tickers)} names")

    eur = load_eur_close(tickers, period="5y")
    added = append_bars(eur)
    cov = store_coverage()
    print(
        f"appended {added} new bars -> store now {cov['n_rows']:,} rows, "
        f"{cov['n_tickers']:,} tickers, {cov['n_dates']:,} dates "
        f"({cov['first']} .. {cov['last']})"
    )


if __name__ == "__main__":
    main()

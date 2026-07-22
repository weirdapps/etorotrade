#!/usr/bin/env bash
# Generate and email the daily Trading Model v3 "Factor Snapshot".
#
# Successor to the v2 committee brief (run-committee-brief.sh): reads the live
# eToro account + master daily signals, runs the v3 overlay pipeline, and emails
# the HTML report via outlook-cli. Repo-tracked; the systemd unit ExecStart
# points at this file so the repo path IS the runtime.
set -euo pipefail

source ~/.config/nbg-vertex/env 2>/dev/null || true
source ~/.config/etoro/env 2>/dev/null || true
export PATH="$HOME/.local/bin:$HOME/.pyenv/shims:$HOME/scripts:$PATH"

REPO="$HOME/SourceCode/etorotrade"
cd "$REPO"

# Stay on master with today's committed signals (daily-signals.yml -> etoro.csv/buy.csv).
git pull --ff-only --quiet 2>/dev/null \
  || echo "WARN: etorotrade pull skipped (local changes or diverged)"

# 1) Live eToro account snapshot -> per-position P/L (keys from ~/.config/etoro/env).
.venv/bin/python scripts/v3_account_snapshot.py 2>&1 \
  || echo "WARN: account snapshot failed - report falls back to portfolio.csv weights"

# 2) Overlay report with the locked decision-support config.
#    V3_FLOOR_CORE=1 (owner 2026-07-22): the equal-risk base sizes EVERY name to ~3.4%,
#    which trims the held mega-cap core (NVDA 8.9%, GOOG 8.7%, MSFT 7.5%, …) down to 3.4%.
#    The floor holds each held mega-cap at its CURRENT weight (<=10%/name) so the core
#    keeps its size; funded by trimming non-core. NOTE: cap_exempt ALONE did NOT protect
#    size (it only exempts the vol lever, not the equal-weight base) — the winners still
#    fell to 3.4%. The floor is what preserves mega-cap exposure.
#    V3_USE_PRICE_STORE=1 (owner 2026-07-22): read prices from the append-only price
#    store (refreshed daily by refresh-prices) + live-fetch only names it misses — so a
#    per-run yfinance throttle no longer unscores core holdings (the "-" bug).
V3_FLOOR_CORE=1 V3_NONCORE_SELL_FLOOR=-0.5 V3_CAP_MODE=cap_ordered \
V3_USD_BLOC_CAP=0.65 V3_VOL_CEILING=0.35 V3_USE_PRICE_STORE=1 \
  .venv/bin/python scripts/v3_overlay_report.py

# 3) Email the scheduled Factor Snapshot. BODY = the compact Outlook-safe summary
#    (regime, buy/keep/sell counts, deployment, vol, one-line action table). The FULL
#    browser Factor Snapshot — IDENTICAL to the trio's attachment (conviction heatmap +
#    per-stock factor cards + trade levels) — rides along as an HTML ATTACHMENT: it
#    can't be the email body because its modern CSS breaks the Outlook sanitizer.
#    (Owner choice 2026-07-22: "attach + short summary".) The other two of the trio
#    (v3_pipeline_map.py + v3_master_taxonomy.py) stay on-demand.
SUMMARY="$(ls -t "$HOME/Downloads/"*_v3_overlay_summary.html 2>/dev/null | head -1 || true)"
SNAPSHOT="$(ls -t "$HOME/Downloads/"*_v3_overlay_report.html 2>/dev/null | head -1 || true)"
if [ -z "${SUMMARY:-}" ] || [ -z "${SNAPSHOT:-}" ]; then
  echo "ERROR: no v3 overlay summary/report produced" >&2
  exit 1
fi

SUBJECT="trading model v3 · factor snapshot $(TZ=Europe/Athens date '+%Y-%m-%d %H:%M')"

~/scripts/outlook-cli send-mail \
  --to dimitrios.plessas@nbg.gr \
  --subject "$SUBJECT" \
  --html "$SUMMARY" \
  --attach "$SNAPSHOT" \
  --send-now --no-signature

echo "OK: emailed summary $(basename "$SUMMARY") + attached $(basename "$SNAPSHOT")"

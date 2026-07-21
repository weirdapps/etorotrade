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
V3_FLOOR_CORE=0 V3_NONCORE_SELL_FLOOR=-0.5 V3_CAP_MODE=cap_ordered \
V3_USD_BLOC_CAP=0.65 V3_VOL_CEILING=0.35 \
  .venv/bin/python scripts/v3_overlay_report.py

# 3) Email the freshest report — the Outlook-safe EMAIL edition (table-based,
#    inline styles), not the browser HTML which the Outlook body sanitizer breaks.
#    This edition is now at FULL info-parity with the browser attachment (conviction
#    heatmap + full per-stock factor cards + trade levels). It is the ONLY thing this
#    scheduled job mails. The other two of the trio are on-demand (sent WITH the snapshot
#    only when the owner asks for "all three files"): the pipeline map (v3_pipeline_map.py)
#    and the MASTER TAXONOMY (v3_master_taxonomy.py — factors x dimensions x weights;
#    replaced the old factor-backtest file as the 3rd member 2026-07-21).
REPORT="$(ls -t "$HOME/Downloads/"*_v3_overlay_email.html 2>/dev/null | head -1 || true)"
if [ -z "${REPORT:-}" ]; then
  echo "ERROR: no v3 overlay email report produced" >&2
  exit 1
fi

SUBJECT="trading model v3 · factor snapshot $(TZ=Europe/Athens date '+%Y-%m-%d %H:%M')"

~/scripts/outlook-cli send-mail \
  --to dimitrios.plessas@nbg.gr \
  --subject "$SUBJECT" \
  --html "$REPORT" \
  --send-now --no-signature

echo "OK: emailed $(basename "$REPORT") ($(wc -c < "$REPORT" | tr -d ' ') bytes)"

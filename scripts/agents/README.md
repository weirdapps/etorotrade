# Committee Agent Scripts (CIO v36+)

These are the canonical, version-controlled agent scripts that produce JSON
in `~/.weirdapps-trading/committee/reports/`. They were moved here from
`~/.weirdapps-trading/committee/scripts/` on 2026-05-04 (CIO v36 N2) for:

- Version control + history
- CI test coverage
- Reproducible runs
- Single source of truth (no parallel copies drifting)

The external location at `~/.weirdapps-trading/committee/scripts/` is kept
as symlinks pointing back here so existing orchestrators continue to work.

| Script | Source | Output | Status |
|---|---|---|---|
| `fundamental_analysis.py` | yfinance + Piotroski calc | `reports/fundamental.json` | active |
| `technical_analysis.py` | yfinance + RSI/MACD/ADX | `reports/technical.json` | active |
| `macro_analyst.py` | yfinance VIX + sector ETFs | `reports/macro.json` | NEW v36 |
| `census_analyst.py` | etoro_census archive (auto-latest) | `reports/census.json` | active (v36 patched) |
| `fetch_news_events.py` | yfinance.news (real, often empty) | `reports/news.json` | active (v36 placeholders stripped) |
| `opportunity_scanner.py` | etoro.csv signal data | `reports/opportunity.json` | active |
| `analyze_priority_candidates.py` | filter top opportunities | sub-output | helper |
| `generate_panw_report.py` | per-stock deep dive | sub-output | template |
| `run_v36_synthesis.py` | synthesis + HTML orchestrator | `reports/synthesis.json` + HTML | NEW v36 |

To regenerate the external symlinks:

```bash
cd ~/.weirdapps-trading/committee/scripts/
for f in ~/SourceCode/etorotrade/scripts/agents/*.py; do
    name=$(basename "$f")
    rm -f "$name"
    ln -s "$f" "$name"
done
```

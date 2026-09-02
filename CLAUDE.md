# etorotrade

Market analysis and portfolio management tool that aggregates Yahoo Finance analyst data into BUY/SELL/HOLD signals with position sizing.

## Tech Stack

- Python >=3.10, <4.0 (CI matrix: 3.10, 3.11, 3.12)
- Key deps: `yfinance`, `yahooquery`, `pandas`, `numpy`, `pydantic`, `aiohttp`. **Versions are NOT restated here** — read `pyproject.toml`. This line used to carry pins and drifted to `yfinance==1.5.2` while the project was on 1.6.0, which is what a second copy of a version number always does.
- Lockfiles: `requirements-lock.txt` (prod), `requirements-dev-lock.txt` (prod+dev), `requirements-smoketest-lock.txt`. A fourth, `requirements-universe-lock.txt`, is hand-maintained and NOT produced by `relock.sh`.
- `pandas` is deliberately capped below 3.0: pandas 3 needs Python >=3.11 while this project still supports 3.10, and its new default `str` dtype silently no-ops the `dtype == "object"` guards in the signal path. See issue #213.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install --only-binary :all: --require-hashes -r requirements-dev-lock.txt
# Or: bash scripts/dev/setup.sh (also installs pre-commit hooks)
```

Do NOT use `poetry install` to install deps — CI uses `pip --require-hashes`. Poetry is only needed when **changing** deps (`scripts/dev/relock.sh` regenerates all lockfiles).

## Running Tests

```bash
pytest tests/                          # all tests
pytest tests/unit/                     # unit only (fast)
pytest tests/integration/             # requires network
pytest -m "not slow"                  # skip slow tests
pytest --cov=yahoofinance --cov=trade_modules --cov-report=html
```

## Running the Tool

```bash
python trade.py                        # interactive mode
python trade.py -o p                  # analyze portfolio
python trade.py -o m                  # market screening
python trade.py -o t -t b             # buy signals
python trade.py -o t -t s             # sell signals
python trade.py -o i -t AAPL,MSFT    # specific tickers
python trade.py -o b                  # backtest validation
python trade.py -o p -pv 50000        # portfolio with $50k value
```

## Code Organization

```text
trade.py                    # entry point
trade_modules/              # trading logic (signals, sizing, committee, risk)
  trade_engine.py           # main orchestration
  config_manager.py         # ConfigManager + ticker substitution map
  analysis_engine.py        # signal generation
  committee_*.py            # CIO committee workflow
yahoofinance/               # data layer
  api/providers/            # Yahoo Finance + async providers
  analysis/                 # stock/market analyzers
  core/                     # DI container, errors, config
  utils/                    # trade criteria, async helpers
scripts/                    # standalone analysis scripts
  dev/                      # relock.sh, setup.sh
  analyze_geography.py      # ETF geographic exposure
  analyze_industry.py       # sector allocation
config.yaml                 # buy/sell thresholds per tier/region
yahoofinance/input/         # portfolio.csv goes here
yahoofinance/output/        # CSV + HTML reports land here
```

## Key Conventions

- Signal column is `BS` in CSV output: `B`=BUY, `H`=HOLD, `S`=SELL, `I`=INCONCLUSIVE
- 5-tier market cap system: MEGA (≥$500B), LARGE ($100B–$500B), MID ($10B–$100B), SMALL ($2B–$10B); below $2B → INCONCLUSIVE
- Regional thresholds: US (baseline), EU (.L/.PA/.AS), HK (.HK)
- Line length: 100 chars. Type hints on new code. `ruff` is the only formatter and linter (`ruff format` + `ruff check`, import order via the `I` rule); black, isort and flake8 were removed 2026-09-03
- Thresholds live in `config.yaml` — do not hardcode in callers

## Known Gotchas

**Lockfile pattern**: The committed `requirements-*-lock.txt` files are the source of truth for CI. After editing `pyproject.toml`, run `scripts/dev/relock.sh` to regenerate `poetry.lock` + all three lockfiles, then commit all five together.

**portfolio.csv conflict**: `yahoofinance/output/portfolio.csv` is written by both the daily-signals GH Action and local runs — a persistent `UU` merge conflict. The worktree usually holds the freshest local data. Pull before running if you need the latest CI-generated signals.

**Ticker substitutions**: eToro tickers don't always match yfinance. Check `ConfigManager.data_fetch_substitutions` in `trade_modules/config_manager.py` before assuming a symbol is unavailable. Add new mismatches there — callers pick it up automatically. Known: `LYXGRE.DE` → `GRE.PA`.

**Census time-series tests**: Some tests use hardcoded dates that will trigger date-drift failures. See project memory `etorotrade_ci_pattern.md` in `claude-config/project-memory/etorotrade/`.

**Portfolio performance**: Never calculate monthly returns from yfinance weighted averages — use the eToro tradeinfo API (`period=CurrMonth`). yfinance systematically undershoots (April 2026: +8.5% yfinance vs +12.43% actual).

**mypy**: Lenient config — `trade_modules.*`, `scripts.*`, and two yahoofinance modules are fully parked (`ignore_errors = true`). mypy runs but doesn't block CI.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push/PR/nightly | Test matrix (3.10–3.12), bandit, safety, flake8, mypy, coverage → SonarCloud |
| `daily-signals.yml` | daily ~22:00 UTC | Runs full signal pipeline, commits output CSVs |
| `weekly-backtest.yml` | weekly | T+7/T+30 backtest validation |
| `weekly-universe-refresh.yml` | weekly | Refreshes the ~4,000-ticker universe |
| `deps-refresh.yml` | weekly | Dependabot-style lockfile refresh |
| `sonarcloud.yml` | push/PR | Quality gate (separate from CI) |

SonarCloud project: `weirdapps_etorotrade`. SessionStart diagnostic is disabled (`hooks.json` emptied) — re-apply after plugin updates.

## eToro API

Canonical domain: `https://www.etoro.com/api/public/v1`
Legacy alias: `https://www.etoro.com/api/public/v1` (works but not canonical)
Auth: X-API-KEY + X-USER-KEY (regular, not PERSONAL) + X-REQUEST-ID (UUID) + User-Agent

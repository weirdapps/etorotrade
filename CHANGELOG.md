# Changelog

## [2025-01-18] - PP Sell Criterion Update

### Changed
- **Trading Criteria**: Updated sell criterion for PP (Past Performance/Price Performance) from <=25% to <=35%
  - File: `yahoofinance/core/trade_criteria_config.py`
  - Previous: `SELL_MAX_PRICE_PERFORMANCE = -25.0` (Sell if PP < -25%)
  - Current: `SELL_MAX_PRICE_PERFORMANCE = -35.0` (Sell if PP < -35%)
  - Impact: Stocks with PP between -35% and -25% will no longer trigger automatic sell recommendations

### Fixed
- **Portfolio Display**: Fixed EG (Earnings Growth) and PP (Past Performance) columns not showing in portfolio analysis
  - Resolved column mapping issues in `yahoofinance/presentation/console.py`
  - Fixed data preservation through `_add_position_size_column` method
  - Ensured proper display of EG and PP values in terminal output

### Technical Changes
- Enhanced 12-month trailing price performance calculation
- Improved confidence score calculation to prevent qualified stocks from being filtered out
- Fixed CLI silent failure issues in trade analysis
- Updated column filtering and display logic for consistent CSV and terminal output

### Testing
- Verified PP criterion change with multiple test cases:
  - CVE (PP = -29.92%): Changed from S (SELL) to I (INCONCLUSIVE)
  - UNH (PP = -47.99%): Remains S (SELL) as expected
  - ELV (PP = -39.13%): Remains S (SELL) as expected
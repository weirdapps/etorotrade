# EG and PP Columns Display Fix Summary

## Issue
When running `trade p n` (or `trade p e`), the EG (Earnings Growth) and PP (Past Performance) columns were showing `--` instead of actual values, even though the portfolio.csv file contained the `earnings_growth` and `twelve_month_performance` data.

## Root Cause
The issue was in the `_add_position_size_column` method in `/Users/plessas/SourceCode/etorotrade/yahoofinance/presentation/console.py`. This method was:

1. **Overwriting existing data**: The EG and PP columns were correctly populated during the `_format_dataframe` step, but the `_add_position_size_column` method was unconditionally recreating these columns.

2. **Looking for wrong column names**: The method was searching for `earnings_growth` and `twelve_month_performance` columns, but by the time it ran, these had already been renamed to `EG` and `PP` in the formatting step.

3. **Data type handling**: The method assumed string values with `%` symbols but was receiving numeric values.

## Solution
Updated the `_add_position_size_column` method to:

1. **Check if columns already exist with valid data**: Only recreate EG and PP columns if they don't exist or contain only null/`--` values.

2. **Handle multiple data sources**: Look for data in both the original column names (`earnings_growth`, `twelve_month_performance`) and the renamed columns (`EG`, `PP`).

3. **Proper data type handling**: Handle both string values (with `%` symbols) and numeric values correctly.

## Files Modified
- `/Users/plessas/SourceCode/etorotrade/yahoofinance/presentation/console.py` - Lines 1068-1109

## Code Changes
```python
# Before: Always overwrote EG and PP columns
df['EG'] = earnings_growths
df['PP'] = three_month_perfs

# After: Only create columns if they don't exist with valid data
if 'EG' not in df.columns or df['EG'].isna().all() or (df['EG'] == '--').all():
    # Create EG column with proper data handling
    
if 'PP' not in df.columns or df['PP'].isna().all() or (df['PP'] == '--').all():
    # Create PP column with proper data handling
```

## Testing
Created comprehensive tests that verified:
1. API data contains correct `earnings_growth` and `twelve_month_performance` values
2. Column mapping works correctly in `_format_dataframe`
3. Data is preserved through the `_add_position_size_column` method
4. Final display shows correct EG and PP values

## Result
The EG and PP columns now display correctly:
- **AMZN**: EG = 27.0, PP = 21.84
- **NVDA**: EG = 13.1, PP = 42.91
- **MSFT**: EG = 17.2, PP = 17.09
- **GOOGL**: EG = 46.2, PP = 3.82
- **AAPL**: EG = 15.5, PP = -5.88

The fix ensures that the EG (Earnings Growth) and PP (Past Performance) columns are now properly visible in the terminal display when running `trade p n` or `trade p e` commands.
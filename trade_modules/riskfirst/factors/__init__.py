"""riskfirst factors — each module exposes ``compute(df) -> pd.Series``.

FACTOR CONTRACT
---------------
- Input ``df``: a pandas DataFrame indexed by ticker, columns from the processed
  universe (etoro.csv schema). No I/O — pure function.
- Output: a pd.Series indexed to df, HIGHER = MORE ATTRACTIVE. NaN where data
  is absent (never raises on missing columns).
- Each factor winsorizes its sub-metrics and z-scores cross-sectionally.
"""

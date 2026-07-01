"""riskfirst factors — each module exposes ``compute(df) -> pd.Series``.

FACTOR CONTRACT
---------------
- Input ``df``: a pandas DataFrame indexed by ticker, columns from the processed
  universe (etoro.csv). Relevant columns and meanings:
    CAP  market cap (string e.g. "1.2T"/"500B"/"3.4M" — parse with
         trade_modules.analysis.tiers._parse_market_cap)
    PRC  price            UP%  analyst upside %      %B   analyst buy %
    AM   analyst momentum (pp)   B    beta            52W  % of 52-week high
    PET  P/E trailing     PEF  P/E forward            P/S  price/sales
    PEG  PEG ratio        DV   dividend yield %       SI   short interest %
    EG   earnings growth %      ROE  return on equity %   DE   debt/equity
    FCF  free-cash-flow yield %
- Output: a pandas Series of z-scores ALIGNED to df.index, where HIGHER = MORE
  ATTRACTIVE for that factor, built with trade_modules.riskfirst.stats.zscore
  (NaN-safe, winsorized). Missing inputs -> NaN for that name (never raise).

Factors (snapshot-based; momentum/lowvol use documented proxies until a
price-history covariance is wired):
    value     cheap: earnings yield (1/PET, 1/PEF), FCF yield, 1/(P/S)
    quality   profitable & sound: ROE up, leverage (DE) down, FCF up, EG up
    momentum  proxy: 52W proximity-to-high + analyst momentum AM
    lowvol    proxy: low beta (B) -> invert
    size      small-cap tilt: -log(market cap)
"""

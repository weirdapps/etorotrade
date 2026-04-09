#!/usr/bin/env python3
"""
Opportunity Scanner for Investment Committee - Iran Crisis Focus
Date: 2026-04-08

Screens the full eToro signal universe for opportunities aligned with:
- Defense sector (escalation plays)
- Energy sector (oil supply disruption)
- Gold/precious metals (safe haven)
- Quality stocks oversold in panic (contrarian value)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import yfinance as yf
from typing import Dict, List

# EXCLUDED TICKERS - Current portfolio holdings
EXCLUDED = {
    '6758.T', '0700.HK', 'NVDA', 'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSM',
    'META', 'NOVO-B.CO', 'WMT', 'LLY', 'JPM', 'MU', 'AMD', 'BAC', 'PLTR',
    'PG', 'UNH', 'C', 'TMO', 'SAP.DE', 'ANET', 'SCHW', 'DTE.DE', 'ABI.BR',
    'RHM.DE', 'NU', 'GLE.PA', 'PRU.L', 'MSTR', 'GLD', 'LYXGRE.DE',
    'BTC-USD', 'ETH-USD'
}

# Sector classifications for Iran crisis
DEFENSE_KEYWORDS = ['AERO', 'DEFENSE', 'AVIATION', 'LOCKHEED', 'RAYTHEON', 'NORTHROP',
                    'BAE', 'BOEING', 'AIRBUS', 'RHEINMETALL', 'KONGSBERG', 'SAAB']
ENERGY_KEYWORDS = ['OIL', 'PETROLEUM', 'ENERGY', 'EXXON', 'SHELL', 'CHEVRON', 'BP',
                   'TOTAL', 'EQUINOR', 'CNOOC', 'PETRO']
GOLD_KEYWORDS = ['GOLD', 'NEWMONT', 'BARRICK', 'PRECIOUS', 'METAL', 'MINING']
SAFE_HAVEN_SECTORS = ['Utilities', 'Consumer Staples', 'Healthcare']


def load_signals(file_path: str) -> pd.DataFrame:
    """Load signal CSV with proper column types."""
    df = pd.read_csv(file_path)

    # Convert percentage strings to floats, handling '--' and missing values
    pct_cols = ['UP%', '%B', 'DV', 'SI', 'FCF']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip('%').replace('--', np.nan).replace('nan', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle numeric columns with '--'
    num_cols = ['PET', 'PEF', 'P/S', 'PEG', 'B', 'ROE', 'DE', 'EG', 'PP', 'PRC', 'TGT', '#T', '#A', '52W']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_census(file_path: str) -> pd.DataFrame:
    """Load census data and extract holding percentages."""
    with open(file_path, 'r') as f:
        census_data = json.load(f)

    holdings = []
    # Try both old and new structure
    investors = census_data.get('investors') or census_data.get('popular_investors', [])

    for investor in investors:
        username = investor.get('userName') or investor.get('username', '')
        for holding in investor.get('portfolio', {}).get('positions', []):
            # Normalize ticker/symbol
            ticker = holding.get('instrumentId') or holding.get('ticker') or holding.get('symbol', '')
            if ticker:
                holdings.append({
                    'ticker': ticker,
                    'investor': username,
                    'weight': holding.get('netPositionPercent', 0) or holding.get('weight', 0)
                })

    if not holdings:
        return pd.DataFrame(columns=['ticker', 'census_pct', 'census_count'])

    df = pd.DataFrame(holdings)

    # Aggregate by ticker
    census_agg = df.groupby('ticker').agg({
        'weight': 'mean',
        'investor': 'count'
    }).reset_index()
    census_agg.columns = ['ticker', 'census_pct', 'census_count']

    return census_agg


def classify_sector(name: str, ticker: str) -> str:
    """Classify stock into macro-relevant sector for Iran crisis."""
    name_upper = str(name).upper()
    ticker_upper = str(ticker).upper()

    # Defense
    if any(kw in name_upper for kw in DEFENSE_KEYWORDS):
        return 'Defense'

    # Energy
    if any(kw in name_upper for kw in ENERGY_KEYWORDS):
        return 'Energy'

    # Gold/Precious Metals
    if any(kw in name_upper for kw in GOLD_KEYWORDS):
        return 'Gold'

    # Safe haven sectors (need better classification - placeholder)
    return 'Other'


def criterion_a_signal_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Criterion A: Strong BUY signals - analyst consensus + upside."""
    mask = (
        (df['BS'] == 'B') &
        (df['%B'] >= 60) &  # At least 60% buy recommendation
        (df['UP%'] >= 15) &  # At least 15% upside
        (df['#A'] >= 5)      # Minimum 5 analysts
    )
    df['crit_a'] = mask
    return df


def criterion_b_value_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Criterion B: Value + Quality - reasonable valuations with strong fundamentals."""
    mask = (
        (df['PEF'] > 0) & (df['PEF'] < 25) &  # Forward PE < 25
        (df['ROE'] > 12) &                     # ROE > 12%
        (df['DE'] < 100)                       # Debt/Equity < 100%
    )
    df['crit_b'] = mask
    return df


def criterion_c_iran_aligned(df: pd.DataFrame) -> pd.DataFrame:
    """Criterion C: Iran-aligned sectors - Defense, Energy, Gold, Safe Havens."""
    df['sector'] = df.apply(lambda x: classify_sector(x['NAME'], x['TKR']), axis=1)

    mask = df['sector'].isin(['Defense', 'Energy', 'Gold'])
    df['crit_c'] = mask
    return df


def criterion_d_census_momentum(df: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """Criterion D: Census momentum - held by popular investors."""
    # NOTE: Census uses numeric instrumentId, not ticker symbols
    # Skipping census matching for now - would need ticker-to-instrumentId mapping
    df['census_pct'] = 0
    df['census_count'] = 0
    df['crit_d'] = False  # Disable census criterion temporarily
    return df


def criterion_e_contrarian_deep_value(df: pd.DataFrame) -> pd.DataFrame:
    """Criterion E: Contrarian deep value - oversold quality."""
    mask = (
        (df['52W'] < 70) &      # Below 70% of 52-week high (oversold)
        (df['UP%'] > 25) &       # High upside potential
        (df['ROE'] > 15) &       # Strong ROE
        (df['BS'] == 'B')        # BUY signal
    )
    df['crit_e'] = mask
    return df


def calculate_opportunity_score(row: pd.Series) -> float:
    """
    Weighted scoring system for ranking opportunities.

    Weights:
    - Criterion A (Signal Strength): 20
    - Criterion B (Value/Quality): 15
    - Criterion C (Iran-aligned): 30 (HIGH - this is the crisis alpha)
    - Criterion D (Census): 10
    - Criterion E (Contrarian): 25

    Max score: 100
    """
    score = 0

    if row.get('crit_a', False):
        score += 20
    if row.get('crit_b', False):
        score += 15
    if row.get('crit_c', False):
        score += 30
    if row.get('crit_d', False):
        score += 10
    if row.get('crit_e', False):
        score += 25

    # Bonus adjustments
    if row.get('UP%', 0) > 30:
        score += 5
    if row.get('sector') == 'Defense':
        score += 10  # Extra weight for defense in Iran crisis
    if row.get('sector') == 'Energy':
        score += 8   # Oil supply disruption theme
    if row.get('sector') == 'Gold':
        score += 5   # Safe haven

    return min(score, 100)


def get_yfinance_details(ticker: str) -> Dict:
    """Fetch insider activity and quick technical check from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Insider activity
        insider_pct = info.get('heldPercentInsiders', 0) * 100
        institutional_pct = info.get('heldPercentInstitutions', 0) * 100

        # Technical: RSI approximation via 52W position
        rsi_proxy = info.get('fiftyTwoWeekLow', 0)

        return {
            'insider_pct': round(insider_pct, 1),
            'institutional_pct': round(institutional_pct, 1),
            'insider_activity': f"Insiders: {insider_pct:.1f}%, Institutions: {institutional_pct:.1f}%",
            'technical_quick': 'Oversold' if rsi_proxy < 0.3 else 'Neutral'
        }
    except Exception as e:
        return {
            'insider_pct': 0,
            'institutional_pct': 0,
            'insider_activity': 'Data unavailable',
            'technical_quick': 'N/A'
        }


def analyze_sector_gaps(df: pd.DataFrame, excluded: set) -> List[str]:
    """Identify sector gaps in current portfolio."""
    gaps = []

    # Check defense exposure
    defense_count = df[df['sector'] == 'Defense']['TKR'].nunique()
    if defense_count > 0:
        gaps.append(f"Defense: {defense_count} opportunities (portfolio has RHM.DE only)")

    # Check energy exposure beyond basics
    energy_count = df[df['sector'] == 'Energy']['TKR'].nunique()
    if energy_count > 0:
        gaps.append(f"Energy: {energy_count} opportunities (portfolio has no pure energy)")

    # Check gold/precious metals
    gold_count = df[df['sector'] == 'Gold']['TKR'].nunique()
    if gold_count > 0:
        gaps.append(f"Gold/Metals: {gold_count} opportunities (portfolio has GLD only)")

    return gaps


def main():
    print("=" * 80)
    print("OPPORTUNITY SCANNER - IRAN CRISIS FOCUS")
    print("Date: 2026-04-08")
    print("=" * 80)

    # Paths
    signals_path = Path('/Users/plessas/SourceCode/etorotrade/yahoofinance/output/etoro.csv')
    census_path = Path('/Users/plessas/SourceCode/etoro_census/archive/data/etoro-data-2026-04-08-03-00.json')
    output_path = Path('/Users/plessas/.weirdapps-trading/committee/reports/opportunities.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n1. Loading signal universe: {signals_path}")
    df = load_signals(str(signals_path))
    print(f"   Loaded {len(df)} stocks")

    print(f"\n2. Loading census data: {census_path}")
    census = load_census(str(census_path))
    print(f"   Loaded {len(census)} census holdings")

    # Exclude current portfolio
    print(f"\n3. Excluding {len(EXCLUDED)} current holdings")
    df = df[~df['TKR'].isin(EXCLUDED)].copy()
    print(f"   {len(df)} stocks remaining")

    # Apply criteria
    print("\n4. Applying screening criteria...")
    df = criterion_a_signal_strength(df)
    print(f"   Criterion A (Signal Strength): {df['crit_a'].sum()} passed")

    df = criterion_b_value_quality(df)
    print(f"   Criterion B (Value/Quality): {df['crit_b'].sum()} passed")

    df = criterion_c_iran_aligned(df)
    print(f"   Criterion C (Iran-aligned sectors): {df['crit_c'].sum()} passed")

    df = criterion_d_census_momentum(df, census)
    print(f"   Criterion D (Census momentum): {df['crit_d'].sum()} passed")

    df = criterion_e_contrarian_deep_value(df)
    print(f"   Criterion E (Contrarian value): {df['crit_e'].sum()} passed")

    # Filter: at least 1 criterion
    candidates = df[df[['crit_a', 'crit_b', 'crit_c', 'crit_d', 'crit_e']].any(axis=1)].copy()
    print(f"\n5. Total candidates (any criterion): {len(candidates)}")

    # Calculate scores
    print("\n6. Calculating opportunity scores...")
    candidates['opportunity_score'] = candidates.apply(calculate_opportunity_score, axis=1)

    # Rank
    candidates = candidates.sort_values('opportunity_score', ascending=False)

    # Top 15 for deep dive
    top_15 = candidates.head(15)
    print(f"\n7. Top 15 opportunities identified")

    # Deep dive with yfinance (top 10 only to save time)
    print("\n8. Running deep dive on top 10 (yfinance lookup)...")
    enriched = []
    for idx, row in top_15.head(10).iterrows():
        ticker = row['TKR']
        print(f"   Fetching {ticker}...")

        yf_data = get_yfinance_details(ticker)

        # Criteria matched
        matched = []
        if row.get('crit_a'): matched.append('A')
        if row.get('crit_b'): matched.append('B')
        if row.get('crit_c'): matched.append('C')
        if row.get('crit_d'): matched.append('D')
        if row.get('crit_e'): matched.append('E')

        # Macro fit explanation
        macro_fit = ""
        if row['sector'] == 'Defense':
            macro_fit = "Defense sector benefits from Iran escalation; increased military spending likely"
        elif row['sector'] == 'Energy':
            macro_fit = "Oil supply disruption risk from Iran conflict; energy prices supportive"
        elif row['sector'] == 'Gold':
            macro_fit = "Safe haven demand in geopolitical uncertainty"
        else:
            macro_fit = "Quality stock oversold in risk-off environment"

        # Risk flags
        risk_flags = []
        if row.get('B', 0) > 1.5:
            risk_flags.append('High beta - volatile')
        if row.get('DE', 0) > 80:
            risk_flags.append('Elevated debt levels')
        if row.get('PEF', 0) > 20:
            risk_flags.append('Premium valuation')

        enriched.append({
            'rank': len(enriched) + 1,
            'ticker': ticker,
            'name': row['NAME'],
            'sector': row['sector'],
            'price': float(row.get('PRC', 0)),
            'target': float(row.get('TGT', 0)),
            'upside_pct': float(row.get('UP%', 0)),
            'exret': row.get('EXR', ''),
            'buy_pct': float(row.get('%B', 0)),
            'pe_trailing': float(row.get('PET', 0)) if pd.notna(row.get('PET')) else None,
            'pe_forward': float(row.get('PEF', 0)) if pd.notna(row.get('PEF')) else None,
            'beta': float(row.get('B', 0)),
            'signal': row['BS'],
            'criteria_matched': matched,
            'opportunity_score': int(row['opportunity_score']),
            'census_holding_pct': float(row.get('census_pct', 0)),
            'insider_activity': yf_data['insider_activity'],
            'technical_quick': yf_data['technical_quick'],
            'macro_fit': macro_fit,
            'why_compelling': f"{row.get('UP%', 0):.1f}% upside, {row.get('%B', 0):.0f}% analyst BUY, {row['sector']} sector aligns with Iran crisis themes",
            'risk_flags': risk_flags
        })

    # Add remaining 5 without yfinance deep dive
    for idx, row in top_15.iloc[10:].iterrows():
        matched = []
        if row.get('crit_a'): matched.append('A')
        if row.get('crit_b'): matched.append('B')
        if row.get('crit_c'): matched.append('C')
        if row.get('crit_d'): matched.append('D')
        if row.get('crit_e'): matched.append('E')

        macro_fit = ""
        if row['sector'] == 'Defense':
            macro_fit = "Defense sector benefits from Iran escalation"
        elif row['sector'] == 'Energy':
            macro_fit = "Oil supply disruption risk"
        elif row['sector'] == 'Gold':
            macro_fit = "Safe haven demand"
        else:
            macro_fit = "Quality oversold"

        enriched.append({
            'rank': len(enriched) + 1,
            'ticker': row['TKR'],
            'name': row['NAME'],
            'sector': row['sector'],
            'price': float(row.get('PRC', 0)),
            'target': float(row.get('TGT', 0)),
            'upside_pct': float(row.get('UP%', 0)),
            'exret': row.get('EXR', ''),
            'buy_pct': float(row.get('%B', 0)),
            'pe_trailing': float(row.get('PET', 0)) if pd.notna(row.get('PET')) else None,
            'pe_forward': float(row.get('PEF', 0)) if pd.notna(row.get('PEF')) else None,
            'beta': float(row.get('B', 0)),
            'signal': row['BS'],
            'criteria_matched': matched,
            'opportunity_score': int(row['opportunity_score']),
            'census_holding_pct': float(row.get('census_pct', 0)),
            'insider_activity': 'Not fetched',
            'technical_quick': 'Not fetched',
            'macro_fit': macro_fit,
            'why_compelling': f"{row.get('UP%', 0):.1f}% upside, {row['sector']} sector",
            'risk_flags': []
        })

    # Sector gap analysis
    print("\n9. Analyzing sector gaps...")
    sector_gaps = analyze_sector_gaps(candidates, EXCLUDED)

    # Output JSON
    output = {
        'analyst': 'opportunity_scanner',
        'timestamp': datetime.now().isoformat(),
        'macro_context': 'Iran war sentiment shift overnight - risk-off environment with defense/energy/gold opportunities',
        'screening_stats': {
            'universe_size': len(df) + len(EXCLUDED),
            'after_exclusions': len(df),
            'passed_criteria_a': int(df['crit_a'].sum()),
            'passed_criteria_b': int(df['crit_b'].sum()),
            'passed_criteria_c': int(df['crit_c'].sum()),
            'passed_criteria_d': int(df['crit_d'].sum()),
            'passed_criteria_e': int(df['crit_e'].sum()),
            'unique_candidates': len(candidates),
            'top_ranked': len(enriched)
        },
        'top_opportunities': enriched,
        'sector_gaps': sector_gaps,
        'census_hidden_gems': [],  # TODO: Add low-coverage, high-conviction picks
        'contrarian_picks': []      # TODO: Add quality oversold names
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n10. Output written to: {output_path}")
    print("\n" + "=" * 80)
    print("TOP 15 OPPORTUNITIES")
    print("=" * 80)
    for opp in enriched:
        print(f"\n{opp['rank']:2}. {opp['ticker']:10} {opp['name'][:40]:40}")
        print(f"    Score: {opp['opportunity_score']}/100 | Sector: {opp['sector']}")
        print(f"    Upside: {opp['upside_pct']:.1f}% | Buy%: {opp['buy_pct']:.0f}% | Beta: {opp['beta']:.1f}")
        print(f"    Macro: {opp['macro_fit']}")
        print(f"    Criteria: {', '.join(opp['criteria_matched'])}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

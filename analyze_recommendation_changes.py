#!/usr/bin/env python3
"""Analyze trading recommendation changes after criteria updates."""

import pandas as pd
from typing import Dict, Any, Tuple, List

# Original criteria (before changes)
class OriginalCriteria:
    MIN_ANALYST_COUNT = 5
    MIN_PRICE_TARGETS = 5
    
    # SELL criteria
    SELL_MAX_UPSIDE = 5.0
    SELL_MIN_BUY_PERCENTAGE = 65.0
    SELL_MIN_FORWARD_PE = 50.0
    SELL_MIN_PEG = 3.0
    SELL_MIN_SHORT_INTEREST = 2.0  # Original: 2.0%
    SELL_MIN_BETA = 3.0
    SELL_MAX_EXRET = 0.05
    
    # BUY criteria
    BUY_MIN_UPSIDE = 20.0  # Original: 20%
    BUY_MIN_BUY_PERCENTAGE = 85.0  # Original: 85%
    BUY_MIN_BETA = 0.25
    BUY_MAX_BETA = 2.5
    BUY_MIN_FORWARD_PE = 0.5
    BUY_MAX_FORWARD_PE = 45.0  # Original: 45
    BUY_MAX_PEG = 2.5
    BUY_MAX_SHORT_INTEREST = 1.5  # Original: 1.5%
    BUY_MIN_EXRET = 0.15  # Original: 15%
    BUY_MIN_MARKET_CAP = 500_000_000

# New criteria (after changes)
class NewCriteria:
    MIN_ANALYST_COUNT = 5
    MIN_PRICE_TARGETS = 5
    
    # SELL criteria
    SELL_MAX_UPSIDE = 5.0
    SELL_MIN_BUY_PERCENTAGE = 65.0
    SELL_MIN_FORWARD_PE = 50.0
    SELL_MIN_PEG = 3.0
    SELL_MIN_SHORT_INTEREST = 2.5  # Changed: 2.5%
    SELL_MIN_BETA = 3.0
    SELL_MAX_EXRET = 0.05
    
    # BUY criteria
    BUY_MIN_UPSIDE = 15.0  # Changed: 15%
    BUY_MIN_BUY_PERCENTAGE = 75.0  # Changed: 75%
    BUY_MIN_BETA = 0.25
    BUY_MAX_BETA = 2.5
    BUY_MIN_FORWARD_PE = 0.5
    BUY_MAX_FORWARD_PE = 55.0  # Changed: 55
    BUY_MAX_PEG = 2.5
    BUY_MAX_SHORT_INTEREST = 2.0  # Changed: 2.0%
    BUY_MIN_EXRET = 0.10  # Changed: 10%
    BUY_MIN_MARKET_CAP = 500_000_000
    
    # New PE difference constraint
    BUY_MAX_PE_EXPANSION = 10  # PEF - PET <= 10


def get_numeric_value(value: Any) -> float:
    """Convert various value formats to float."""
    if value is None or pd.isna(value):
        return None
        
    if isinstance(value, str):
        if value == "--" or not value.strip():
            return None
        try:
            return float(value.replace("%", "").replace(",", ""))
        except:
            return None
    
    try:
        return float(value)
    except:
        return None


def parse_market_cap(cap_str: str) -> float:
    """Parse market cap string to numeric value."""
    if cap_str == "--" or not cap_str or pd.isna(cap_str):
        return None
    
    try:
        cap_str = str(cap_str).upper().strip()
        if cap_str.endswith('T'):
            return float(cap_str[:-1]) * 1_000_000_000_000
        elif cap_str.endswith('B'):
            return float(cap_str[:-1]) * 1_000_000_000
        elif cap_str.endswith('M'):
            return float(cap_str[:-1]) * 1_000_000
        else:
            return float(cap_str)
    except:
        return None


def check_confidence(row: Dict, criteria_class) -> bool:
    """Check if stock has sufficient analyst coverage."""
    analyst_count = get_numeric_value(row.get("# T"))
    total_ratings = get_numeric_value(row.get("# A"))
    
    if analyst_count is None or total_ratings is None:
        return False
    
    return (analyst_count >= criteria_class.MIN_ANALYST_COUNT and 
            total_ratings >= criteria_class.MIN_PRICE_TARGETS)


def check_sell_criteria(row: Dict, criteria_class) -> Tuple[bool, str]:
    """Check if stock meets SELL criteria."""
    # Check each sell condition
    upside = get_numeric_value(row.get("UPSIDE"))
    if upside is not None and upside < criteria_class.SELL_MAX_UPSIDE:
        return True, f"Low upside ({upside:.1f}%)"
    
    buy_pct = get_numeric_value(row.get("% BUY"))
    if buy_pct is not None and buy_pct < criteria_class.SELL_MIN_BUY_PERCENTAGE:
        return True, f"Low buy% ({buy_pct:.1f}%)"
    
    pef = get_numeric_value(row.get("PEF"))
    pet = get_numeric_value(row.get("PET"))
    
    if pef is not None and pet is not None:
        if pef > 0 and pet > 0 and (pef - pet) > 0.5:
            return True, f"Worsening PE"
    
    if pef is not None:
        if pef < 0:
            return True, f"Negative PE"
        elif pef > criteria_class.SELL_MIN_FORWARD_PE:
            return True, f"High PE ({pef:.1f})"
    
    peg = get_numeric_value(row.get("PEG"))
    if peg is not None and peg >= criteria_class.SELL_MIN_PEG:
        return True, f"High PEG ({peg:.1f})"
    
    si = get_numeric_value(row.get("SI"))
    if si is not None and si > criteria_class.SELL_MIN_SHORT_INTEREST:
        return True, f"High SI ({si:.1f}%)"
    
    beta = get_numeric_value(row.get("BETA"))
    if beta is not None and beta > criteria_class.SELL_MIN_BETA:
        return True, f"High beta ({beta:.1f})"
    
    exret = get_numeric_value(row.get("EXRET"))
    if exret is not None:
        if exret > 1.0:
            exret = exret / 100
        if exret < criteria_class.SELL_MAX_EXRET:
            return True, f"Low EXRET ({exret*100:.1f}%)"
    
    return False, None


def check_buy_criteria(row: Dict, criteria_class) -> Tuple[bool, str]:
    """Check if stock meets BUY criteria."""
    # ALL conditions must be met
    upside = get_numeric_value(row.get("UPSIDE"))
    if upside is None or upside < criteria_class.BUY_MIN_UPSIDE:
        return False, f"Upside ({upside:.1f}%)" if upside else "Upside N/A"
    
    buy_pct = get_numeric_value(row.get("% BUY"))
    if buy_pct is None or buy_pct < criteria_class.BUY_MIN_BUY_PERCENTAGE:
        return False, f"Buy% ({buy_pct:.1f}%)" if buy_pct else "Buy% N/A"
    
    beta = get_numeric_value(row.get("BETA"))
    if beta is None:
        return False, "Beta N/A"
    if beta <= criteria_class.BUY_MIN_BETA or beta > criteria_class.BUY_MAX_BETA:
        return False, f"Beta ({beta:.1f})"
    
    pef = get_numeric_value(row.get("PEF"))
    if pef is None:
        return False, "PEF N/A"
    if not (criteria_class.BUY_MIN_FORWARD_PE < pef <= criteria_class.BUY_MAX_FORWARD_PE):
        return False, f"PEF ({pef:.1f})"
    
    # Check PE difference for new criteria
    if hasattr(criteria_class, 'BUY_MAX_PE_EXPANSION'):
        pet = get_numeric_value(row.get("PET"))
        if pet is not None and pet > 0:
            pe_diff = pef - pet
            if pe_diff > criteria_class.BUY_MAX_PE_EXPANSION:
                return False, f"PE expansion ({pe_diff:.1f})"
    else:
        # Original logic: PEF < PET or PET <= 0
        pet = get_numeric_value(row.get("PET"))
        if pet is not None:
            pe_improving = pef < pet
            is_growth = pet <= 0
            if not (pe_improving or is_growth):
                return False, f"PE not improving"
    
    # Optional criteria
    peg = get_numeric_value(row.get("PEG"))
    if peg is not None and peg >= criteria_class.BUY_MAX_PEG:
        return False, f"PEG ({peg:.1f})"
    
    si = get_numeric_value(row.get("SI"))
    if si is not None and si > criteria_class.BUY_MAX_SHORT_INTEREST:
        return False, f"SI ({si:.1f}%)"
    
    market_cap = parse_market_cap(row.get("CAP"))
    if market_cap is None or market_cap < criteria_class.BUY_MIN_MARKET_CAP:
        return False, f"Market cap"
    
    exret = get_numeric_value(row.get("EXRET"))
    if exret is None:
        return False, "EXRET N/A"
    if exret > 1.0:
        exret = exret / 100
    if exret < criteria_class.BUY_MIN_EXRET:
        return False, f"EXRET ({exret*100:.1f}%)"
    
    return True, None


def calculate_action(row: Dict, criteria_class) -> Tuple[str, str]:
    """Calculate trading action for a stock."""
    # Check confidence first
    if not check_confidence(row, criteria_class):
        return "I", "Low confidence"
    
    # Check SELL criteria
    is_sell, sell_reason = check_sell_criteria(row, criteria_class)
    if is_sell:
        return "S", sell_reason
    
    # Check BUY criteria
    is_buy, buy_reason = check_buy_criteria(row, criteria_class)
    if is_buy:
        return "B", "All criteria met"
    
    # Default to HOLD
    return "H", buy_reason or "Default"


def main():
    # Read market data
    print("Reading market.csv...")
    df = pd.read_csv("/Users/plessas/SourceCode/etorotrade/yahoofinance/output/market.csv")
    
    print(f"Analyzing {len(df)} assets...")
    
    changes = []
    
    for idx, row in df.iterrows():
        # Get current ACT value
        current_act = row.get("ACT", "")
        
        # Calculate with original criteria
        old_act, old_reason = calculate_action(row, OriginalCriteria)
        
        # Calculate with new criteria
        new_act, new_reason = calculate_action(row, NewCriteria)
        
        # Check if recommendation changed
        if old_act != new_act:
            changes.append({
                'Ticker': row['TICKER'],
                'Name': row['COMPANY'],
                'Old': old_act,
                'New': new_act,
                'Old Reason': old_reason,
                'New Reason': new_reason,
                'UPSIDE': row.get('UPSIDE'),
                '% BUY': row.get('% BUY'),
                'BETA': row.get('BETA'),
                'PEF': row.get('PEF'),
                'PET': row.get('PET'),
                'SI': row.get('SI'),
                'EXRET': row.get('EXRET'),
                'CAP': row.get('CAP')
            })
    
    # Display results
    print(f"\nFound {len(changes)} assets with changed recommendations:\n")
    
    # Group by change type
    change_summary = {}
    for change in changes:
        key = f"{change['Old']} → {change['New']}"
        if key not in change_summary:
            change_summary[key] = []
        change_summary[key].append(change)
    
    # Print summary
    for change_type, assets in sorted(change_summary.items()):
        print(f"\n{change_type}: {len(assets)} assets")
        print("-" * 80)
        
        for asset in assets[:10]:  # Show first 10 of each type
            print(f"{asset['Ticker']:8} {asset['Name'][:30]:30} | Reason: {asset['New Reason']}")
            print(f"         UPSIDE: {asset['UPSIDE']:>6} | %BUY: {asset['% BUY']:>5} | EXRET: {asset['EXRET']:>6} | SI: {asset['SI']:>5}")
        
        if len(assets) > 10:
            print(f"         ... and {len(assets) - 10} more")
    
    # Save detailed results
    if changes:
        changes_df = pd.DataFrame(changes)
        changes_df.to_csv("/Users/plessas/SourceCode/etorotrade/recommendation_changes.csv", index=False)
        print(f"\nDetailed changes saved to recommendation_changes.csv")


if __name__ == "__main__":
    main()
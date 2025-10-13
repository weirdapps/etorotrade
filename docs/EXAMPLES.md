# Real-World Examples
## eToro Trade Analysis Tool

This guide provides practical examples of how to use the tool in different scenarios.

---

## Table of Contents

1. [Getting Started with eToro](#getting-started-with-etoro)
2. [Weekly Portfolio Review](#weekly-portfolio-review)
3. [Researching a New Stock](#researching-a-new-stock)
4. [Finding Buy Opportunities](#finding-buy-opportunities)
5. [Sector-Specific Analysis](#sector-specific-analysis)
6. [Portfolio Rebalancing](#portfolio-rebalancing)
7. [Geographic Exposure Analysis](#geographic-exposure-analysis)

---

## Getting Started with eToro

**Scenario:** You just opened an eToro account and bought your first stocks. You want to analyze your portfolio.

**Steps:**

1. **Export your portfolio from eToro:**
   - Log in to eToro web platform
   - Navigate to Portfolio
   - Click the "⋯" (more) menu
   - Select "Export to Excel" or "Download"
   - Save the file

2. **Prepare the file:**
   ```bash
   # Create input directory if it doesn't exist
   mkdir -p yahoofinance/input

   # Move your downloaded file
   mv ~/Downloads/portfolio.csv yahoofinance/input/portfolio.csv
   ```

3. **Run your first analysis:**
   ```bash
   python trade.py -o p
   ```

4. **Understand the output:**
   - Console shows color-coded signals:
     - 🟢 **B** = BUY (analyst consensus positive)
     - 🔴 **S** = SELL (analyst consensus negative)
     - ⚪ **H** = HOLD (fairly valued)
     - ⚫ **I** = INCONCLUSIVE (insufficient analyst coverage)

   - Key columns to watch:
     - **UPSIDE**: Potential gain to analyst target price
     - **%BUY**: Percentage of analysts recommending buy
     - **EXRET**: Expected return (upside × %buy / 100)

5. **Review the HTML report:**
   ```bash
   # On Mac
   open yahoofinance/output/portfolio.html

   # On Windows
   start yahoofinance\output\portfolio.html

   # On Linux
   xdg-open yahoofinance/output/portfolio.html
   ```

**What to expect:**
- First run takes 30-60 seconds for a 10-stock portfolio
- Subsequent runs are faster (48-hour cache)
- Some stocks may show INCONCLUSIVE if they're small-cap with limited analyst coverage

---

## Weekly Portfolio Review

**Scenario:** You review your eToro portfolio every Sunday to decide which positions to hold, add to, or exit.

**Workflow:**

1. **Monday morning: Export fresh portfolio data**
   ```bash
   # Download from eToro, save to input directory
   # (eToro data updates overnight)
   ```

2. **Run portfolio analysis:**
   ```bash
   python trade.py -o p
   ```

3. **Review SELL signals:**
   - Look for positions marked **S** (SELL)
   - Check if UPSIDE is negative or very low
   - Review %BUY percentage (low = bearish analyst sentiment)

4. **Check position sizes:**
   - Tool suggests position sizes based on market cap tier
   - Compare "SIZE" column to your actual investment
   - Over-weighted positions may need trimming

5. **Document your decisions:**
   ```bash
   # Save this week's report
   cp yahoofinance/output/portfolio.html ~/Documents/portfolio_2025-01-13.html
   ```

**Example interpretation:**

```
TICKER  COMPANY        UPSIDE  %BUY   BS   EXRET  INTERPRETATION
AAPL    Apple Inc      15.2%   76%    B    11.5   Strong buy - consider adding
MSFT    Microsoft      12.8%   82%    B    10.5   Strong buy - hold or add
NVDA    NVIDIA         -5.2%   45%    S    -2.3   Sell signal - consider exit
TSLA    Tesla          8.5%    52%    H    4.4    Hold - fairly valued
```

**Decision framework:**
- **Multiple SELL signals:** Review fundamentals, consider reducing position
- **Strong BUY with low allocation:** Consider increasing position
- **HOLD signals:** Keep current allocation, monitor quarterly

---

## Researching a New Stock

**Scenario:** You heard about a stock on social media and want to research it before buying.

**Example: Researching PLTR (Palantir)**

1. **Quick analysis:**
   ```bash
   python trade.py -o i -t PLTR
   ```

2. **What to look for:**

   **✅ Good signs:**
   - UPSIDE > 15%
   - %BUY > 60%
   - Analyst count (#A) > 10
   - PEG ratio < 2.0
   - Beta < 1.5 (less volatile)

   **⚠️ Warning signs:**
   - UPSIDE < 0% (price above analyst targets)
   - %BUY < 40% (bearish sentiment)
   - Few analysts (#A < 5) = less reliable
   - PEG > 3.0 (expensive vs growth)
   - High short interest (SI > 10%)

3. **Compare with similar stocks:**
   ```bash
   # Research multiple data analytics companies
   python trade.py -o i -t PLTR,SNOW,DDOG,MDB
   ```

4. **Review the output:**
   ```
   TICKER  UPSIDE  %BUY  PEG   BETA   BS   INTERPRETATION
   PLTR    25.3%   68%   2.1   1.8    B    High growth, analyst support
   SNOW    18.5%   72%   1.8   1.3    B    Better valuation
   DDOG    12.2%   65%   2.5   1.4    B    Lower upside
   MDB     -2.1%   55%   3.2   1.6    S    Overvalued vs peers
   ```

5. **Make informed decision:**
   - Read analyst reports for context
   - Check recent earnings results
   - Review company fundamentals beyond this tool
   - **Remember:** This tool provides data, not investment advice

---

## Finding Buy Opportunities

**Scenario:** You have $10,000 to invest and want to find stocks with strong analyst support.

**Method 1: Market-wide screening (large dataset)**

```bash
# Screen all eToro available stocks
python trade.py -o m
```

**What happens:**
- Analyzes 5,544 available stocks on eToro
- Takes 15-20 minutes
- Generates `yahoofinance/output/market.csv`

**Then filter for buy signals:**
```bash
# View buy opportunities
python trade.py -o t -t b
```

Output shows stocks meeting BUY criteria for each market cap tier.

**Method 2: Focused screening (faster)**

```bash
# Screen specific large-cap tech stocks
python trade.py -o i -t AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,AMD,INTC,CRM

# Screen banking sector
python trade.py -o i -t JPM,BAC,WFC,C,GS,MS,BLK,SCHW

# Screen healthcare
python trade.py -o i -t JNJ,UNH,PFE,ABBV,TMO,DHR,BMY,LLY
```

**Interpreting results:**

1. **Filter by signal:**
   - Focus on **B** (BUY) signals
   - Consider **H** (HOLD) if you want established positions

2. **Sort by EXRET** (Expected Return):
   - Higher EXRET = better risk/reward ratio
   - EXRET > 10% is strong opportunity

3. **Check fundamentals:**
   - PEG < 2.0 (reasonable valuation)
   - Beta < 2.0 (moderate volatility)
   - Dividend yield > 0% (bonus income)

4. **Diversification:**
   - Don't buy all tech stocks
   - Mix market cap tiers (MEGA, LARGE, MID)
   - Consider geographic exposure

**Example portfolio allocation:**

```
TICKER  CAP    UPSIDE  %BUY  EXRET  SIZE($)  WHY
MSFT    MEGA   12.8%   82%   10.5   $12,500  Stable large-cap leader
ROKU    MID    28.3%   70%   19.8   $7,500   High growth potential
DDOG    MID    22.1%   68%   15.0   $7,500   Strong tech fundamentals

Total: $27,500 suggested allocation for $10,000 → Scale down proportionally
```

**Actual allocation for $10,000:**
- MSFT: $4,500 (45%)
- ROKU: $2,750 (27.5%)
- DDOG: $2,750 (27.5%)

---

## Sector-Specific Analysis

**Scenario:** You believe the AI sector will grow and want to invest in AI-focused companies.

**AI/Technology stocks:**
```bash
python trade.py -o i -t NVDA,AMD,AVGO,QCOM,ASML,TSM,INTC,MU,ARM,SMCI
```

**Cloud infrastructure:**
```bash
python trade.py -o i -t AMZN,MSFT,GOOGL,IBM,ORCL,CRM,SNOW,DDOG,NET,CFLT
```

**Semiconductor equipment:**
```bash
python trade.py -o i -t ASML,LRCX,AMAT,KLAC,ENTG
```

**Analysis approach:**

1. **Identify sector leaders:**
   - Highest market cap
   - Best analyst ratings (%BUY)
   - Most analyst coverage (#A)

2. **Find undervalued opportunities:**
   - High UPSIDE percentage
   - Low PEG ratio
   - BUY signal

3. **Assess risk:**
   - High beta = more volatile
   - High PEG = expensive vs growth
   - Low analyst count = less certain

4. **Diversify within sector:**
   - Mix of MEGA and MID cap
   - Different sub-sectors (chips vs software)
   - Geographic diversity (US vs Taiwan vs Netherlands)

---

## Portfolio Rebalancing

**Scenario:** Your portfolio has grown unevenly. Some positions are now over-weighted, others under-weighted.

**Step 1: Analyze current portfolio**
```bash
python trade.py -o p
```

**Step 2: Review position sizes**

Look at the SIZE column - this is the tool's suggested position size based on:
- Market capitalization tier
- Expected return (EXRET)
- Risk factors (beta, earnings growth)

**Example current vs. suggested:**

```
TICKER  CURRENT($)  SUGGESTED($)  ACTION
AAPL    $15,000    $12,500       Trim $2,500
MSFT    $12,000    $12,500       Add $500
NVDA    $20,000    $10,000       Trim $10,000 (overweight!)
ROKU    $2,000     $7,500        Add $5,500
TSLA    $8,000     SELL          Exit position
```

**Step 3: Identify sells:**
```bash
# Check which holdings have SELL signals
python trade.py -o t -t s
```

**Step 4: Find replacements:**
```bash
# Find new buy opportunities NOT in your portfolio
python trade.py -o t -t b
```

**Step 5: Execute rebalancing:**

1. **Exit SELL positions:**
   - Sell stocks with **S** signal
   - Trim over-weighted positions

2. **Add to under-weighted:**
   - Increase positions with **B** signal
   - Add to under-weighted holdings

3. **Add new positions:**
   - From buy opportunities list
   - Choose different sectors for diversification

---

## Geographic Exposure Analysis

**Scenario:** You want to understand if your portfolio is too US-focused and diversify internationally.

**Step 1: Run geographic analysis**
```bash
python scripts/analyze_geography.py
```

**What it shows:**
- Breakdown by country/region
- Percentage of portfolio in each geography
- ETF holdings transparency (if you own ETFs)

**Example output:**
```
Geographic Exposure Analysis:
United States:    75.5%
Europe:           15.2%
Asia:             8.3%
Other:            1.0%
```

**Step 2: Add international stocks**

```bash
# European stocks
python trade.py -o i -t ASML.AS,SAP,NOVO-B.CO,MC.PA,OR.PA

# Asian stocks
python trade.py -o i -t TSM,BABA,TCEHY,SMSN.IL
```

**Step 3: Consider international ETFs**

```bash
# Analyze international ETFs
python trade.py -o i -t EFA,VGK,VWO,EEM,IEMG
```

**Balancing considerations:**
- **US-heavy** (>80%): Consider international diversification
- **Geography-specific risk**: Currency fluctuations, regulations
- **Different market caps**: Emerging markets often smaller cap
- **Tax implications**: International stocks may have withholding tax

---

## Tips for All Scenarios

### Best Practices

1. **Regular analysis:**
   - Weekly for active traders
   - Monthly for long-term investors
   - After major market events

2. **Combine with other research:**
   - Read earnings reports
   - Follow company news
   - Check SEC filings
   - Review industry trends

3. **Don't over-trade:**
   - Signals change with analyst updates
   - Consider transaction costs
   - Tax implications of frequent selling

4. **Position sizing discipline:**
   - Follow suggested SIZE guidelines
   - Don't over-allocate to single stock
   - Maximum 10-15% per position

5. **Risk management:**
   - Diversify across sectors
   - Mix market cap tiers
   - Consider geographic exposure
   - Keep some cash reserves

### What This Tool Does NOT Do

- ❌ Predict short-term price movements
- ❌ Time the market
- ❌ Account for your personal risk tolerance
- ❌ Consider your tax situation
- ❌ Replace comprehensive financial planning

### What This Tool DOES Do

- ✅ Aggregates analyst consensus
- ✅ Calculates expected returns systematically
- ✅ Applies tier-specific criteria consistently
- ✅ Suggests position sizes based on fundamentals
- ✅ Provides data-driven framework for decisions

---

*Last Updated: October 2025*

**Remember:** This tool provides analysis only, not investment advice. All investment decisions are your own responsibility. Always conduct your own research and consider consulting with qualified financial advisors before making investment decisions.

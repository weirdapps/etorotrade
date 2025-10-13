# Frequently Asked Questions (FAQ)
## eToro Trade Analysis Tool

Quick answers to common questions about using the tool.

---

## General Questions

### Is this investment advice?

**No.** This tool provides systematic analysis of publicly available analyst data. It is:
- ✅ A research tool that organizes data
- ✅ A framework for evaluating securities
- ✅ An aggregator of analyst consensus

It is NOT:
- ❌ Investment advice
- ❌ A recommendation to buy or sell
- ❌ A guarantee of returns
- ❌ A substitute for professional financial advice

**All investment decisions are your own responsibility.** You should conduct your own research and consider consulting with qualified financial advisors before making investment decisions.

### How accurate are the trading signals?

**Accuracy depends on multiple factors:**

1. **Analyst consensus quality:**
   - Analysts can be wrong
   - Past performance ≠ future results
   - Consensus lags market events

2. **Data coverage:**
   - Stocks with 20+ analysts = more reliable
   - Stocks with <5 analysts = less reliable
   - INCONCLUSIVE signals indicate insufficient data

3. **Market conditions:**
   - Bull markets may have more BUY signals
   - Bear markets may have more SELL signals
   - Tool reflects analyst sentiment, not market timing

4. **Timeframe:**
   - Analysts typically target 12-month price targets
   - Not designed for day trading or short-term speculation

**Historical performance:** This tool is designed for systematic evaluation, not prediction. No signal accuracy guarantee is provided or implied.

### Do I need an eToro account to use this tool?

**No.** The tool works for any investor:

**With eToro account:**
- Export portfolio CSV from eToro
- Analyze your actual holdings
- Get position-specific signals

**Without eToro account:**
- Use manual ticker input
- Screen market opportunities
- Research individual stocks

**Commands for non-eToro users:**
```bash
# Analyze specific stocks
python trade.py -o i -t AAPL,MSFT,GOOGL

# Generate buy opportunities list
python trade.py -o t -t b
```

The eToro integration simply provides a convenient CSV format for portfolio imports.

### How often should I run this analysis?

**Recommended frequency:**

- **Active traders:** Weekly
  - Monday morning after weekend news
  - Track signal changes over time
  - Monitor SELL signals for exits

- **Long-term investors:** Monthly
  - First of the month review
  - Quarterly earnings season
  - After major market events

- **Before buying:** Always
  - Research individual stocks before purchase
  - Compare multiple alternatives
  - Check for SELL signals before entering

**Cache considerations:**
- Tool uses 48-hour cache
- Fresh analyst data available after 48 hours
- Run too frequently = same results

### Can I use this for day trading?

**Not recommended.** This tool is designed for:
- ✅ Fundamental analysis
- ✅ Medium to long-term investing (weeks to months)
- ✅ Position-based strategies

It is NOT designed for:
- ❌ Day trading (intraday movements)
- ❌ Technical analysis (chart patterns)
- ❌ High-frequency trading
- ❌ Options or derivatives

**Why:** Analyst consensus and price targets are 12-month forward-looking estimates, not short-term price predictions.

---

## Technical Questions

### What data sources does the tool use?

**Primary source:** Yahoo Finance API via yfinance library
- Analyst recommendations (20+ investment banks)
- Price targets (mean analyst target)
- Financial metrics (P/E, PEG, Beta, etc.)
- Market data (price, market cap, etc.)

**Supplementary source:** YahooQuery API
- Used when yfinance data incomplete
- Primarily for PEG ratios
- Fills gaps in missing metrics

**Data quality:**
- Real-time market prices
- Analyst consensus updated regularly
- 48-hour cache to reduce API load

### How is the BUY/SELL signal calculated?

**Signal generation process:**

1. **Data requirements:**
   - Minimum **4 analysts** with recommendations
   - Minimum **4 price targets**
   - If insufficient: Signal = **INCONCLUSIVE**

2. **Market cap classification:**
   - Categorize into 5 tiers: MEGA, LARGE, MID, SMALL, MICRO
   - Apply region-specific adjustments (US/EU/HK)

3. **Criteria evaluation:**
   - Each tier has specific thresholds in `config.yaml`
   - Metrics evaluated: UPSIDE, %BUY, EXRET, PEG, Beta, PE ratios

4. **Signal determination:**
   - **SELL:** ANY sell condition met (conservative approach)
   - **BUY:** ALL buy conditions met (strict criteria)
   - **HOLD:** Between BUY and SELL thresholds
   - **INCONCLUSIVE:** Insufficient analyst data

**Example for MEGA-cap US stocks:**
```yaml
BUY requirements (ALL must be true):
- UPSIDE ≥ 8%
- %BUY ≥ 55%
- EXRET ≥ 5.0
- Beta: 0.5 - 2.0
- PEF/PET ratio improving

SELL triggers (ANY can trigger):
- UPSIDE < 5%
- %BUY < 45%
- EXRET < 3.0
- PEF/PET ratio deteriorating
```

### Can I customize the trading criteria?

**Yes.** The tool is highly configurable:

**File to modify:** `config.yaml`

**What you can customize:**
- Tier-specific thresholds (UPSIDE, %BUY, EXRET minimums)
- Market cap tier boundaries
- Beta ranges
- PEG ratio limits
- Forward P/E limits
- Regional adjustments

**Example modification:**
```yaml
tiers:
  MEGA:
    US:
      min_upside: 8.0        # Change to 10.0 for stricter
      min_buy_percentage: 55 # Change to 60 for stricter
      min_exret: 5.0         # Change to 7.0 for stricter
```

**⚠️ Warning:**
- Modifying criteria changes signals significantly
- Stricter = fewer BUY signals
- Looser = more BUY signals (but lower quality)
- Keep backup of original config

**Reset to defaults:**
```bash
# Restore original config.yaml from repository
git checkout config.yaml
```

### Why do some stocks show INCONCLUSIVE?

**Reason:** Insufficient analyst coverage.

**Requirements for signal:**
- Minimum 4 analysts
- Minimum 4 price targets

**Common causes:**

1. **Small-cap stocks:**
   - Market cap < $2B often lack coverage
   - Too small for major investment banks
   - Solution: Stick to mid/large-cap stocks

2. **New IPOs:**
   - Recently public companies
   - Analysts building coverage over time
   - Wait 6-12 months after IPO

3. **International stocks:**
   - Some foreign stocks have limited US analyst coverage
   - European/Asian stocks may show INCONCLUSIVE
   - Not a data error, reflects actual coverage gap

4. **Niche industries:**
   - Specialized sectors
   - Limited analyst expertise
   - Manual research required

**This is NOT a bug** - the tool correctly identifies when data is insufficient for reliable signals.

### How long does analysis take?

**Expected times:**

| Tickers | Time | Use Case |
|---------|------|----------|
| 1-10 | 5-10 sec | Individual stock research |
| 10-50 | 30-60 sec | Typical portfolio |
| 50-100 | 2-3 min | Large portfolio |
| 100-500 | 10-15 min | Market screening |
| 5,544 | 15-20 min | Full eToro market |

**Factors affecting speed:**
- Network connection speed
- First run vs cached run (48hr cache)
- Yahoo Finance API response time
- Number of concurrent requests (max 15)

**Speeding up analysis:**
- Run analysis once, review cached data within 48 hours
- Analyze in smaller batches
- Use faster internet connection

---

## Portfolio & Strategy Questions

### What position size should I use?

**Tool provides SIZE suggestions based on:**
- Market capitalization tier
- Expected return (EXRET)
- Risk factors (Beta, earnings growth)

**Example base sizes:**
```
MEGA-cap:  $12,500 base
LARGE-cap: $10,000 base
MID-cap:   $7,500 base
SMALL-cap: $5,000 base
MICRO-cap: $2,500 base
```

**Then adjusted for:**
- Higher EXRET = larger position
- Higher Beta = smaller position (risk adjustment)
- Strong earnings growth = larger position

**⚠️ These are suggestions, NOT requirements:**
- Scale to your portfolio size
- Consider your risk tolerance
- Maximum 10-15% per position recommended
- Maintain diversification

**Example for $50,000 portfolio:**
```
Tool suggests $12,500 for AAPL (25% of portfolio)
Better allocation: $5,000-$7,500 (10-15%)
```

### Should I sell everything that shows SELL?

**Not necessarily.** Consider:

1. **Check why it's SELL:**
   - Negative UPSIDE? (price above target)
   - Low analyst support? (%BUY low)
   - Deteriorating fundamentals? (PEF/PET ratio)

2. **Your investment timeline:**
   - Short-term holder? Consider selling
   - Long-term investor? May hold through volatility

3. **Tax implications:**
   - Short-term capital gains tax
   - Wash sale rules
   - Consider timing of sales

4. **Transaction costs:**
   - Trading fees
   - Spread costs
   - Impact on portfolio

5. **Alternative opportunities:**
   - Are there better BUY opportunities?
   - What would you buy instead?

**Approach:** Use SELL signals as research triggers, not automatic sell orders.

### How do I diversify my portfolio?

**Use multiple tool features:**

1. **Sector diversification:**
   ```bash
   # Check sector allocation
   python scripts/analyze_industry.py
   ```

2. **Geographic diversification:**
   ```bash
   # Check geographic exposure
   python scripts/analyze_geography.py
   ```

3. **Market cap diversification:**
   - Mix MEGA, LARGE, and MID cap stocks
   - Avoid all small-cap or all mega-cap
   - Different tiers = different risk/return profiles

4. **Signal diversification:**
   - Not all BUY signals are equal
   - High EXRET ≠ guaranteed returns
   - Spread risk across multiple opportunities

**Example balanced portfolio:**
```
40% MEGA-cap (stability)
30% LARGE-cap (growth + stability)
20% MID-cap (growth potential)
10% SMALL-cap (higher risk/reward)
```

### What if the market crashes?

**During bear markets:**

1. **More SELL signals are normal:**
   - Analysts adjust targets downward
   - %BUY percentages drop
   - Tool reflects deteriorating sentiment

2. **Tool is not a market timer:**
   - Doesn't predict crashes
   - Doesn't know when to "get out"
   - Reflects analyst consensus, which can be wrong

3. **Use as data point:**
   - Rising SELL signals = caution
   - Declining EXRET = weakening outlook
   - Consider, don't blindly follow

4. **Opportunity identification:**
   - Quality stocks on sale may still show BUY
   - High UPSIDE during crashes = potential
   - Distinguishes fundamentals from panic

**Remember:** This tool doesn't replace a comprehensive investment strategy or risk management plan.

---

## Troubleshooting & Support

### Where can I get help?

**Documentation:**
- [User Guide](USER_GUIDE.md) - Getting started
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues
- [Examples](EXAMPLES.md) - Real-world scenarios
- [Technical Guide](TECHNICAL.md) - For developers

**Community support:**
- [GitHub Issues](https://github.com/weirdapps/etorotrade/issues)
- Search existing issues first
- Provide details when creating new issues

**Before asking for help:**
1. Read relevant documentation
2. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Search GitHub issues for similar problems
4. Note your Python version and OS

### Is this tool being actively maintained?

**Current status:**
- ✅ Open source on GitHub
- ✅ Regular updates
- ✅ Bug fixes applied
- ✅ Community contributions welcome

**Check latest updates:**
```bash
# Pull latest changes
git pull origin master
```

**Version information:**
- Check commit history on GitHub
- Review [CI/CD.md](CI_CD.md) for pipeline status
- SonarCloud badges in README show code quality

### Can I contribute to the project?

**Yes!** Contributions welcome:

**Ways to contribute:**
1. Report bugs via GitHub Issues
2. Suggest features
3. Submit pull requests
4. Improve documentation
5. Share usage examples

**Before contributing:**
- Read [TECHNICAL.md](TECHNICAL.md) for technical architecture
- Run tests: `pytest tests/`
- Follow PEP 8 style: `flake8`
- Include type hints
- Update documentation

**Pull request requirements:**
- All tests must pass
- Code quality checks pass
- Documentation updated
- Clear description of changes

---

## Legal & Disclaimers

### Is this legal to use?

**Yes.** The tool:
- Uses publicly available data
- Accesses free Yahoo Finance APIs
- Applies systematic analysis criteria
- Provides research, not regulated advice

**Important notes:**
- You're responsible for your investment decisions
- Tool doesn't provide regulated investment advice
- No guarantees of accuracy or profitability
- Past performance ≠ future results

### What's the license?

**MIT License**

- ✅ Free to use
- ✅ Free to modify
- ✅ Free to distribute
- ✅ Commercial use allowed

**Requirements:**
- Include original license file
- Include copyright notice

**No warranty:**
- Provided "as is"
- No guarantee of merchantability
- No liability for losses

See [LICENSE](../LICENSE) file for full terms.

### Can I use this for my business?

**Yes, under MIT License:**
- ✅ Internal investment research
- ✅ Portfolio management tools
- ✅ Client reporting (with proper disclaimers)
- ✅ Educational purposes

**⚠️ Important:**
- If providing investment advice, ensure proper licensing
- Comply with financial regulations in your jurisdiction
- Include appropriate disclaimers
- Don't misrepresent tool capabilities

**Recommended disclaimer:**
> "Analysis powered by open-source tool. This is research data, not investment advice. All investment decisions are the client's responsibility."

### Who created this tool?

**Created by:** [plessas](https://www.etoro.com/people/plessas) - eToro Popular Investor

**Purpose:** Help investors make data-driven decisions by aggregating analyst data into a single, systematic framework.

**Philosophy:**
- Transparency over black boxes
- Data-driven over emotional
- Systematic over arbitrary
- Educational over automated

---

## Still Have Questions?

**Documentation resources:**
- [README.md](../README.md) - Overview and quick start
- [USER_GUIDE.md](USER_GUIDE.md) - Detailed usage instructions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
- [EXAMPLES.md](EXAMPLES.md) - Real-world scenarios
- [TECHNICAL.md](TECHNICAL.md) - Technical architecture
- [POSITION_SIZING.md](POSITION_SIZING.md) - Methodology details

**Can't find your answer?**
1. Search [GitHub Issues](https://github.com/weirdapps/etorotrade/issues)
2. Create new issue with:
   - Clear question
   - What you tried
   - Expected vs actual behavior
   - Python version, OS

---

*Last Updated: October 2025*

**Disclaimer:** This tool provides analysis only, not investment advice. All investment decisions are your own responsibility. Always conduct your own research and consider consulting with qualified financial advisors before making investment decisions.

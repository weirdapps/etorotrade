# etorotrade 🚀 Smart Investment Analysis Made Simple

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=coverage)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=weirdapps_etorotrade&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=weirdapps_etorotrade)

**Turn market chaos into clear, actionable investment decisions in seconds**

![eToro Trade Analysis Tool](docs/assets/etorotrade.png)

Stop second-guessing your trades. etorotrade analyzes your portfolio and the market using institutional-grade algorithms, delivering BUY/SELL/HOLD recommendations based on real analyst consensus and proven financial metrics. What Wall Street pays millions for, now at your fingertips.

## 💡 What Can It Do For You?

### 📈 **Get Instant Trade Recommendations**
```bash
python trade.py -o t -t b    # Find BUY opportunities NOW
python trade.py -o t -t s    # Identify SELL signals
```
Get clear BUY/SELL/HOLD signals based on:
- 🎯 Analyst consensus from major banks
- 📊 Price targets and upside potential  
- 💰 Smart position sizing based on risk
- 🔄 Momentum and earnings growth

### 🌍 **Understand Your True Portfolio Exposure**
```bash
python scripts/analyze_geography.py   # Where is your money really?
python scripts/analyze_industry.py    # What sectors are you betting on?
```
See through ETFs to understand:
- Geographic exposure (including ETF holdings)
- Sector allocation with ETF transparency
- Crypto, commodities, and derivatives breakdown
- Risk concentration warnings

### 🎨 **Multiple Ways to Analyze**
- **Portfolio Mode** - Analyze your eToro holdings
- **Market Mode** - Screen the entire market
- **Trade Mode** - Get specific BUY/SELL recommendations
- **Manual Mode** - Analyze specific tickers

## 🚀 Get Started in 2 Minutes

### 1️⃣ **Installation**
```bash
git clone https://github.com/weirdapps/etorotrade
cd etorotrade
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ **Add Your Portfolio** (Optional)
Place your eToro portfolio export in `yahoofinance/input/portfolio.csv`

### 3️⃣ **Run Your First Analysis**
```bash
python trade.py    # Interactive mode - easiest to start!
```

## 🎮 How to Use

### 🔥 **Quick Commands for Power Users**
```bash
# Find what to BUY right now
python trade.py -o t -t b

# Check if you should SELL anything  
python trade.py -o t -t s

# Analyze your portfolio risk
python trade.py -o p -t e

# Deep dive into geography & sectors
python scripts/analyze_geography.py
python scripts/analyze_industry.py

# Analyze specific stocks
python trade.py -o i -t AAPL,MSFT,NVDA
```

### 🎯 **Interactive Mode** (Beginner Friendly)
Just run `python trade.py` and choose:
- **P** → Portfolio Analysis 
- **M** → Market Screening
- **T** → Trade Recommendations  
- **I** → Input Specific Tickers

## 📖 Understanding the Output

### 🎯 **Trade Signals Explained**
When you see a recommendation, here's what it means:

| Signal | What It Means | Action |
|--------|--------------|--------|
| **BUY** 🟢 | Strong analyst consensus + good upside | Consider opening position |
| **SELL** 🔴 | Overvalued or declining fundamentals | Consider taking profits |
| **HOLD** 🟡 | Fair value, wait for better entry | Keep existing position |
| **INCONCLUSIVE** ⚪ | Mixed signals | Do more research |

### 📊 **Key Metrics Decoded**
- **ACT** - Action recommendation (B/S/H/I)
- **UPSIDE** - Potential gain to analyst price target
- **EXRET** - Expected return (probability-weighted)
- **SIZE** - Suggested position size in USD
- **PP** - 3-month price performance
- **EG** - Earnings growth year-over-year

### 🎨 **Market Cap Tiers**
The system now uses a sophisticated 5-tier classification system based on market capitalization:
- **MEGA** 🏆 - Mega-cap champions (≥$500B market cap)
- **LARGE** 💎 - Large-cap established ($100B-$500B)
- **MID** 🚀 - Mid-cap growth ($10B-$100B)
- **SMALL** 🌱 - Small-cap opportunities ($2B-$10B)
- **MICRO** ⚡ - Micro-cap speculative (<$2B)

Each tier has tailored buy/sell criteria optimized for its risk-return profile.

## 🌟 Real-World Usage & Validation

I personally use this system for my eToro investment decisions with real money:

👉 **[@plessas on eToro](https://www.etoro.com/people/plessas)** - See it in action

The system has helped me:
- 📈 Identify winning trades before they pop
- 🛡️ Avoid overvalued hype stocks
- ⚖️ Maintain balanced portfolio allocation
- 🎯 Size positions based on risk/reward

## ⚡ Why It's Fast & Reliable

- **Lightning Fast**: Analyzes 100+ stocks in seconds
- **Real-Time Data**: Yahoo Finance + YahooQuery APIs
- **Smart Caching**: Reduces API calls, speeds up analysis
- **Error Resilient**: Automatic retries and fallbacks
- **Production Ready**: Comprehensive test coverage, CI/CD pipeline
- **5-Tier Analysis**: Sophisticated market cap based classification
- **Geographic Awareness**: Region-specific criteria (US/EU/HK)
- **YAML Configuration**: Flexible, externalized trading thresholds

## ⚙️ Configuration (Optional)

### Trading Thresholds
The system uses a sophisticated configuration in `config.yaml` that defines:
- **5 Market Cap Tiers**: MEGA, LARGE, MID, SMALL, MICRO
- **3 Geographic Regions**: US, EU, HK
- **Tier-specific criteria**: Each tier has optimized buy/sell thresholds
- **Portfolio value-based sizing**: Dynamic position sizing based on portfolio size

### Portfolio Input
Export your eToro portfolio to `yahoofinance/input/portfolio.csv`:
```csv
symbol,totalInvestmentPct,totalNetProfitPct,instrumentDisplayName
AAPL,5.2,12.5,Apple Inc
MSFT,4.8,8.3,Microsoft Corp
```

You can also specify portfolio value for position sizing:
```bash
python trade.py -o p -pv 50000  # $50,000 portfolio
```

### Environment Variables (Optional)
```bash
# .env file for eToro API (if available)
ETORO_API_KEY=your-key
ETORO_USER_KEY=your-key
```

## 📂 Project Structure

```
etorotrade/
├── trade.py                 # Main entry point
├── scripts/
│   ├── analyze_geography.py # Geographic exposure analysis
│   └── analyze_industry.py  # Sector allocation analysis
├── trade_modules/           # Trading logic
├── yahoofinance/           # Data & analysis
└── tools/
    ├── lint.sh             # Code quality checks
    └── cleanup.sh          # Clean temp files
```

## 🧪 For Developers

```bash
# Run tests
pytest tests/

# Check code quality
./tools/lint.sh

# Clean up
./tools/cleanup.sh
```

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Detailed usage instructions
- **[Technical Docs](docs/CLAUDE.md)** - Architecture details
- **[Position Sizing](docs/POSITION_SIZING.md)** - How positions are calculated

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with modern software engineering practices for professional investment analysis**
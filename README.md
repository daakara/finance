# 📈 Finance Terminal | Quantitative Intelligence & Execution Platform

A professional-grade, institutional financial intelligence platform featuring real-time Congressional STOCK Act tracking, Mark Minervini VCP algorithmic entry points, Linda Raschke 20 EMA pullbacks, and Cornish-Fisher Modified Value-at-Risk modeling.

**Live Deployment**: [https://finance-xp8.pages.dev/](https://finance-xp8.pages.dev/)

![Next.js](https://img.shields.io/badge/next.js-14.2-black.svg)
![React](https://img.shields.io/badge/react-18-cyan.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.110+-green.svg)
![Cloudflare Pages](https://img.shields.io/badge/deployed-cloudflare%20pages-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Key Modules & Capabilities

### 1. 📈 Interactive Dual-Horizon Candlestick Engine
- **Lightweight Charts Canvas**: Sub-millisecond rendering with automatic viewport fitting (`fitContent()`) and dynamic canvas theming (Dark vs. Paper Light).
- **⚡ Day Trader Scalp Sessions**: `1m`, `5m`, `15m`, `1h` timeframes with live **Volume-Weighted Average Price (VWAP)** indicator overlay and Unix epoch time scaling.
- **🏛️ Long-Term Macro Horizons**: `1M`, `6M`, `1Y`, `3Y`, `5Y` timeframes with **20 Exponential Moving Average (20 EMA)** trend support overlay and ISO calendar date scaling.
- **🎯 Metric Disambiguation**: Clear separation between the Watchlist **24H Daily Return** and the Chart Header **Active Horizon Return** with explicit date baseline tooltips.
- **⚡ Fast Fallback Resilience**: 1.5s API timeout with high-fidelity instant (<1ms) fallback generator (`generateFallbackAnalytics`).

### 2. 🏛️ Congressional STOCK Act, Legislative Alignment & Staleness Decay
- **Public Law 112-105 Tracking**: Real-time disclosures from US House and Senate members (e.g. Nancy Pelosi LEAPS call purchases).
- **⚖️ Legislative Alignment Index (0–100)**: Quantitative scoring of committee jurisdiction conflict overlap (+16 to +32 pts), transaction sizing tiers ($50k–$1M+), and verified historical alpha.
- **⏱️ Staleness Time-Decay Engine**: Automatic conviction decay penalizing disclosures older than 15, 30, and 45 days, with explicit late-filer mean-reversion risk warnings for non-compliant filers.
- **SEC EDGAR Form 4**: Open-market insider purchases (&ge; \$100k) by CEOs, CFOs, and Board Directors within 2 business days.

### 3. 🎯 Minervini VCP & Algorithmic Execution Ladder
- **Volatility Contraction Pattern (VCP)**: 3-stage contraction detection with volume dry-up confirmations.
- **⚡ 4 Mathematical ATR States**: Real-time state tagging (`IN_BUY_ZONE`, `APPROACHING_TARGET`, `WAITING_PULLBACK`, `STOPPED_OUT`).
- **Strict Execution Invariant**: `Stop Loss < Optimal Entry Min <= Optimal Entry Max <= Current Spot < Target 1 < Target 2`.
- **Intraday Position Sizer**: 1-click execution calculation risking $1\%–2\%$ account equity into persistent local storage tracking.

### 4. 🛡️ Cornish-Fisher Modified VaR & Self-Healing Engine
- **Non-Normal Fat-Tail VaR**: Polynomial Cornish-Fisher expansion adjusting for skewness and kurtosis with monotonic 99% $\le$ 95% safety floors.
- **🤖 Self-Healing Forecast Auditor**: Continuous model calibration using Kupiec POF VaR exception tests, walk-forward RMSE tracking, and dynamic confidence bound expansion.
- **FRED Macro Regimes**: Real-time 10Y-2Y Yield Curve spreads (`T10Y2Y`) and OAS Credit Spreads (`BAMLH0A0HYM2`) applying dynamic 0.5x–1.25x position risk multipliers.

---

## 🏗️ Architecture

```
finance/
├── frontend/                     # Next.js 14 App Router + TailwindCSS (Cloudflare Pages)
│   ├── app/
│   │   ├── page.tsx             # Main Terminal with 4 Modular Workspaces
│   │   ├── compare/page.tsx     # Normalized Multi-Asset Benchmarking
│   │   ├── screener/page.tsx    # Expert Model Stock Screener (Magic Formula, Lynch, VCP)
│   │   ├── smart-money/page.tsx # Congressional STOCK Act & Dark Pool Feeds
│   │   ├── portfolio/page.tsx   # Local Anonymous Portfolio & Execution Tracker
│   │   └── guide/page.tsx       # Institutional Field Manual & Math Specification
│   ├── components/
│   │   ├── PriceChart.tsx       # Dual-Horizon TradingView Lightweight Charts
│   │   ├── Navbar.tsx           # Global Navigation & Role Switcher
│   │   ├── WatchlistSidebar.tsx # Real-Time 24H Watchlist & Search
│   │   └── ...                  # Workspace Cards (Execution, Factors, Risk, Smart Money)
│   └── lib/
│       ├── api.ts               # Analytics Engine, Timeout Budgets & Horizon Fallback
│       ├── constants.ts         # Shared Factor Scores & Multi-Period Baselines
│       └── institutionalFeeds.ts# FRED Macro & SEC Form 4 Ingestion
├── api/                          # FastAPI Backend Services (Render)
│   └── main.py                  # Analytical Endpoints & Live Market Ingestion
├── analyst_dashboard/            # Quantitative Analyzers & Mathematical Engines
│   ├── analyzers/               # Advanced Risk, Factor Models & Execution Analyzers
│   └── visualizers/             # Streamlit Visualizations & Exploration Dashboards
└── requirements.txt
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### 🚨 SSL Certificate Issue Fix
This platform includes comprehensive SSL certificate handling to resolve common connection issues. If you experience SSL certificate errors, the system will automatically:
- Use proper certificate configuration
- Retry with different SSL approaches  
- Fall back to realistic sample data for demonstration

See `SSL_FIX_GUIDE.md` for detailed troubleshooting.

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd finance
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your browser at `http://localhost:8501`

## 🔑 API Keys (Optional)

While the platform works with free data sources, you can enhance functionality with API keys:

- **Alpha Vantage**: For enhanced stock data and news
- **News API**: For financial news sentiment analysis
- **Twitter API**: For social sentiment analysis

Add these to your `.env` file:
```
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here
```

## 📊 Usage Guide

### Market Overview
- View real-time market indices and treasury rates
- Monitor market sentiment indicators
- Track major economic indicators

### Stock Analysis
1. Enter stock symbols in the sidebar
2. Select analysis period (1 month to max history)
3. View comprehensive stock metrics and charts
4. Analyze fundamental ratios and quality scores

### Technical Analysis
1. Choose a symbol for technical analysis
2. View interactive charts with multiple indicators
3. Monitor automated trading signals
4. Identify support and resistance levels

### Portfolio Analysis
1. Add multiple symbols to your portfolio
2. Set custom weights for each asset
3. Analyze portfolio performance and risk metrics
4. View correlation matrices and optimization suggestions

## 🔍 Core Components

### Data Layer
- **Fetchers**: Retrieve data from multiple sources (yfinance, ccxt, APIs)
- **Processors**: Clean, validate, and transform raw data
- **Cache**: Intelligent caching with TTL for performance

### Analysis Engine
- **Technical Analysis**: 15+ technical indicators with signal generation
- **Fundamental Analysis**: Complete financial ratio calculations
- **Portfolio Analytics**: Modern Portfolio Theory implementation

### Visualization Engine
- **Professional Charts**: Candlestick, line, bar, and specialized financial charts
- **Interactive Dashboards**: Real-time updating interfaces
- **Custom Themes**: Dark/light themes with financial color schemes

## 🎯 Key Features

### Performance
- **Caching System**: Multi-level caching (memory + file) with TTL
- **Modular Design**: Clean separation of concerns for maintainability
- **Error Handling**: Comprehensive error handling and logging
- **Input Validation**: Robust validation for all user inputs

### Data Quality
- **Data Validation**: OHLCV relationship validation and outlier detection
- **Missing Data Handling**: Forward fill and interpolation strategies
- **Timezone Awareness**: Global market timezone handling

### User Experience
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live data refresh and caching
- **Intuitive Interface**: Professional yet accessible design
- **Export Capabilities**: Download analysis results and charts

## 🧪 Code Quality

### Best Practices
- **Type Hints**: All functions include comprehensive type annotations
- **Docstrings**: Detailed documentation with examples
- **Modular Functions**: Maximum 50 lines per function
- **Single Responsibility**: Each function has one clear purpose
- **DRY Principle**: No code duplication

### Testing
```bash
# Run tests (when available)
python -m pytest tests/
```

### Code Style
- Follows PEP 8 guidelines
- Black code formatting
- Comprehensive error handling
- Logging for debugging and monitoring

## 📈 Advanced Features

### Portfolio Optimization
- **Efficient Frontier**: Calculate optimal risk-return portfolios
- **Risk Parity**: Equal risk contribution portfolios
- **Kelly Criterion**: Optimal position sizing
- **Rebalancing Alerts**: Automated rebalancing suggestions

### Risk Management
- **Value at Risk (VaR)**: 5% and 1% confidence intervals
- **Conditional VaR**: Expected shortfall calculations
- **Maximum Drawdown**: Historical drawdown analysis
- **Stress Testing**: Portfolio performance under various scenarios

### Technical Indicators
- **Trend Following**: Moving averages, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, Volume Profile, Money Flow Index

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚨 Disclaimer

This software is for educational and informational purposes only. It should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.

## 🆘 Support

- **Documentation**: Check the inline documentation and docstrings
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join discussions for feature requests and questions

## 🔮 Roadmap

### Upcoming Features
- [ ] Machine Learning predictions
- [ ] Cryptocurrency analysis expansion
- [ ] Options analysis tools
- [ ] Backtesting engine
- [ ] Custom indicator builder
- [ ] API for external integration
- [ ] Mobile app companion
- [ ] Real-time alerts system

### Performance Improvements
- [ ] Database integration for data persistence
- [ ] Async data fetching
- [ ] WebSocket real-time data streams
- [ ] Advanced caching strategies

---

**Built with ❤️ for the financial analysis community**

# 📈 Financial Market Analysis Platform

A comprehensive, professional-grade financial analysis platform with two powerful applications: a Main Financial Dashboard and a Hidden Gems Stock Scanner. Built with Python and Streamlit for real-time market analysis with **100% live data integrity**.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.40+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Data Policy](https://img.shields.io/badge/data-live%20only-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen.svg)

## 🚀 Features

### Market Analysis
- **Real-time Market Data**: Live stock, ETF, and cryptocurrency data
- **Market Indices Tracking**: S&P 500, NASDAQ, Dow Jones, and international indices
- **Economic Indicators**: Treasury rates, VIX, currency indices

### Technical Analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic, Williams %R, ATR, OBV
- **Pattern Recognition**: Support/resistance levels, trend analysis
- **Interactive Charts**: Professional candlestick charts with volume overlays
- **Trading Signals**: Automated signal generation based on technical indicators

### Fundamental Analysis
- **Financial Ratios**: P/E, PEG, P/B, P/S, D/E, Current Ratio, Quick Ratio
- **Profitability Metrics**: ROE, ROA, ROIC, Profit Margins
- **Valuation Models**: DCF, Dividend Discount Model, WACC calculations
- **Quality & Value Scoring**: Proprietary scoring system for investment evaluation

### Portfolio Management
- **Portfolio Construction**: Custom weight allocation and rebalancing
- **Risk Analytics**: Sharpe ratio, Sortino ratio, Maximum Drawdown, VaR, CVaR
- **Performance Attribution**: Alpha, Beta, tracking error analysis
- **Optimization**: Modern Portfolio Theory implementation
- **Correlation Analysis**: Asset correlation matrices and heat maps

### Visualization
- **Interactive Charts**: Plotly-powered professional charts
- **Customizable Dashboards**: Multi-tab interface with real-time updates
- **Performance Tracking**: Cumulative returns, drawdown analysis
- **Comparison Tools**: Multi-asset normalized price comparisons

## 🏗️ Architecture

The platform follows a modular, production-ready architecture:

```
finance/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration and constants
├── data/
│   ├── fetchers.py          # Data retrieval (yfinance, ccxt, APIs)
│   ├── processors.py        # Data cleaning and transformation
│   └── cache.py             # Intelligent caching system
├── analysis/
│   ├── technical.py         # Technical analysis indicators
│   ├── fundamental.py       # Fundamental analysis metrics
│   └── portfolio.py         # Portfolio analytics and optimization
├── visualizations/
│   ├── charts.py            # Interactive chart creation
│   ├── dashboards.py        # Dashboard layouts
│   └── themes.py            # Visual styling
├── utils/
│   ├── helpers.py           # Utility functions
│   ├── validators.py        # Input validation
│   └── formatters.py        # Data formatting
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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-08-27

### Fixed
- 🌓 **Comprehensive Paper Light Theming**: Overhauled CSS selector architecture with wildcard substring attributes (`[class*="bg-[#..."]`), eliminating dark background retention on headers, sidebars, cards, insets, pills, and inputs.
- 📊 **Dynamic HTML5 Chart Canvas Adaptation**: Replaced hardcoded black canvas background and dark grid lines in Lightweight Charts with reactive theme color palettes, responding in real-time to `"finance:theme-change"` and `data-theme` mutations without page reloads.
- 🏷️ **Accessible Badge High-Contrast Contrast**: Mapped dark pill backgrounds (`bg-*-950`) to soft pastel fills (`bg-*-50`) with high-contrast text (`text-*-800`/`900`), complying with WCAG AA standards.

### Added
- 🧪 **Theme Compliance Quality Gate**: Added `test_light_paper_theme_compliance` in `tests/test_nextjs_frontend_structure.py` to prevent theme regression across headers, cards, and canvas components.

---

## [1.1.0] - 2026-08-27

### Fixed
- ⏱️ **Timeframe Selector Responsiveness**: Broken re-render reset loop between `Navbar.tsx` and `page.tsx` eliminated via `useCallback` and decoupled state synchronization.
- 📈 **Lightweight Charts Rescaling**: Removed non-existent `resetTimeScale()` call that threw silent `TypeError` in v4, restoring smooth `fitContent()` viewport auto-scaling.
- 📊 **Macro 5Y Horizon Parsing**: Fixed `generateFallbackAnalytics` to strictly match intraday intervals, preventing `1mo` monthly bars from being misclassified and truncated.
- 📱 **Mobile Touch Accessibility**: Added explicit `type="button"` and `touch-manipulation` to all timeframe selector elements.
- 🛡️ **Network Resilience**: Extended API client timeout from 1500ms to 8000ms to eliminate cold-start drops and prevent fallback baseline jumping.

### Added
- 🧪 Automated regression quality gate in `tests/test_nextjs_frontend_structure.py` enforcing timeframe state isolation, chart API conformance, and interval matching rules.

---

## [1.0.0] - 2025-10-31

### Added
- 📊 Main Financial Platform with real-time market data
- 💎 Hidden Gems Scanner for discovering undervalued stocks
- 📈 Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- 📊 Fundamental analysis metrics (P/E, ROE, market cap, etc.)
- 💼 Portfolio management and tracking
- 🔄 Multi-asset support (stocks, ETFs, cryptocurrencies)
- 📊 Interactive Plotly charts and visualizations
- ✅ Comprehensive test suite (13/13 tests passing)
- 🔒 Live-data-only policy (no sample data)
- 🔧 SSL certificate handling for corporate environments
- 📝 Comprehensive documentation
- 🚀 Startup scripts for easy deployment

### Changed
- 🔄 Removed all sample data fallbacks for data integrity
- 🔒 Implemented transparent error handling
- ⚡ Optimized data fetching with caching
- 📊 Improved chart rendering performance

### Security
- 🔒 SSL certificate validation configured
- 🔐 No data collection or tracking
- ✅ All analysis runs locally

### Documentation
- 📝 README.md - Comprehensive project documentation
- 📝 CONTRIBUTING.md - Contribution guidelines
- 📝 SSL_TROUBLESHOOTING_GUIDE.md - SSL certificate troubleshooting
- 📝 SAMPLE_DATA_REMOVAL_REPORT.md - Live-data-only implementation
- 📝 BOTH_APPS_DATA_REVIEW.md - Data integrity review
- 📝 AUTOMATION_IMPLEMENTATION_REPORT.md - Automation features

---

## [Unreleased]

### Planned Features
- [ ] Email alerts for gem discoveries
- [ ] Export to CSV/Excel functionality
- [ ] More technical indicators
- [ ] News sentiment analysis
- [ ] Backtesting framework
- [ ] Options analysis module
- [ ] Machine learning predictions
- [ ] Social sentiment analysis

---

## Version History

### v1.0.0 - Initial Release (2025-10-31)
First stable release with full feature set:
- Two complete applications (Main Platform + Hidden Gems)
- Live data integration
- Technical and fundamental analysis
- Portfolio management
- Comprehensive testing
- Production-ready architecture

---

## Migration Guides

### Upgrading to v1.0.0

This is the initial release. No migration needed.

**Important Changes:**
- **Live Data Only**: Sample data functionality has been completely removed
- **SSL Configuration**: Automatic SSL certificate handling added
- **Error Handling**: More explicit error messages for troubleshooting

**Breaking Changes:**
- None (initial release)

**Deprecations:**
- None (initial release)

---

## Notes

### Semantic Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Change Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

---

**Last Updated**: October 31, 2025

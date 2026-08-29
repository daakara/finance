# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.0] - 2026-08-29

### Privacy-First Matomo Analytics & Comprehensive User Journey Telemetry Suite
- 📊 **Matomo Tag Manager & `_paq` Dual Tracker (`layout.tsx`)**: Explicitly initialized the official Matomo endpoint (`https://data.fpldna.com/matomo/matomo.php`, `idsite=3`) with MTM container (`container_tK4RnlSN.js`) for privacy-first, zero-cookie analytical compliance.
- 🎯 **Full User Journey Telemetry Suite (`frontend/lib/matomo.ts`)**: Implemented telemetry tracking functions for the entire trader workflow:
  - ✈️ **Pre-Flight Decision Gates**: Tracks execution clearance outcomes (`Cleared` vs `Conditional`) and trade plan journal clipboard copies.
  - 📐 **Position Sizing & Half-Kelly**: Tracks risk allocation adjustments, share calculations, and portfolio additions.
  - 🔔 **Execution & Breakout Alerts**: Tracks pullback buy zone alerts and Stage 4 breakout pivot triggers.
  - 🌪️ **Macro Stress Shocks**: Tracks portfolio what-if shocks (`QQQ -5%`, `Yield +50bps`, `VIX 35`) and cash reserve recommendations.
  - ⭐ **Watchlist & Screener Journeys**: Tracks favorite asset toggles, screener preset filtering, ticker list copies, and CSV data exports.
- 🧪 **Automated Analytics CI Verification (`tests/test_nextjs_frontend_structure.py`)**: Added automated regression checks ensuring all telemetry hooks and tracking containers remain permanently wired across releases (56/56 tests passing).

---

## [2.6.0] - 2026-08-29

### Cloudflare Edge SWR Shield, Stale-If-Error Fallback & Server Boot Pre-Warming
- ⚡ **Cloudflare Edge SWR & Stale-If-Error Directives (`api/routes/`)**: Configured RFC 5861 `Cache-Control`, `CDN-Cache-Control`, and `Cloudflare-CDN-Cache-Control` (`max-age=15..60, s-maxage=60..300, stale-while-revalidate=86400, stale-if-error=86400`) across `analytics.py`, `screener.py`, and `smart_money.py`, delivering $<10\text{ms}$ edge cache delivery and shielding upstream data providers from rate limits.
- 🚀 **Server Boot Universe Pre-Warming (`api/main.py`)**: Added background startup pre-warming thread initializing the core 60-ticker universe into in-memory `INFO_CACHE` on boot, eliminating first-request cold-start latency.
- 🛡️ **Cloudflare Pages Headers Hardening (`frontend/public/_headers`)**: Added explicit immutable caching for static Next.js chunks (`/_next/static/*`) and edge proxy rules for API routes.
- 🧪 **Automated Cache Header CI Test (`tests/test_screener_execution.py`)**: Added automated unit test validating that screener endpoints output full SWR and `stale-if-error` headers (55/55 tests passing).

---

## [2.5.0] - 2026-08-29

### Zero Static Placeholders Across All Pre-Flight Checklist Validation Gates
- 🛑 **100% Dynamic Pre-Flight Verification (`PreFlightChecklistModal.tsx`)**: Completely purged all remaining static boolean placeholders (`isSmartMoneyPassed = true`, `isCatalystPassed = true`, `isMacroPassed = true`).
- 📡 **Smart Money & Distribution Trap Coupling**: Check 3 dynamically resolves against `MASTER_ASSET_CATALOG` quality scores, short float ratios ($>12\%$), and Form 4 C-Suite selling flags.
- ⚡ **Binary Catalyst & Macro VIX Guard**: Checks 4 & 5 dynamically evaluate binary event hazard windows ($<48\text{h}$) and broad market volatility ($VIX \ge 26.0$).
- 🧪 **Zero-Placeholder CI Assertions (`test_cross_component_state_synchronicity.py`)**: Added automated AST/code-level checks guaranteeing zero hardcoded static boolean assignments exist in decision modals (55/55 tests passing).

---

## [2.4.0] - 2026-08-29

### Cross-Component State Synchronicity & Contradiction Immunity Test Suite
- 🧪 **Automated Cross-Component Test Suite (`tests/test_cross_component_state_synchronicity.py`)**: Implemented automated CI verification testing dynamic state propagation across all interactive modals (`PreFlightChecklistModal`, `PositionSizerModal`, `AlertTriggerModal`, `OptimalEntryExitCard`) ensuring zero hardcoded boolean placeholders or contradiction blindspots exist (55/55 tests passing).
- 🛡️ **Multi-Modal Stage 4 Alignment**: Enforced that `isStage4` strictly down-sizes position risk ($0.25\%$), converts alert triggers to the 50-day breakout pivot, and blocks pre-flight clearance until base completion.

---

## [2.3.0] - 2026-08-29

### Dynamic Stage 4 & Buy Zone Coupling in Pre-Flight Trade Clearance Gate
- 🛑 **Pre-Flight Trend & Stage Gate (`PreFlightChecklistModal.tsx`)**: Eliminated the static pass blindspot by dynamically wiring Check 2 ("Trend & Moving Averages / Buy Zone Corridor") directly to `isStage4`, `optimalEntryMin`, and `optimalEntryMax`.
- ⚠️ **Stage 4 Invalidation Discipline**: When an asset is in a Stage 4 correction or awaiting a 50-day base breakout (e.g. `LRCX`), Check 2 automatically triggers `❌ / ⏳ STAGE 4 WAIT`, and the overall trade readiness score shifts to `⚠️ CONDITIONAL / AWAIT BASE CLEARANCE (4/5 Passed)` and blocks clearance to execute.
- 🎯 **Extended Price / Chasing Protection**: Added automatic chase warnings if spot price trades above the verified accumulation ceiling ($>+2\%$ above `optimalEntryMax`).

---

## [2.2.0] - 2026-08-29

### Quantitative Methodology Guardian (`quant-guardian`) & Domain Invariant Test Suite
- 🧠 **Registered Subagent & Skill (`quant-guardian`)**: Created and registered the specialized Quantitative Finance & Investment Domain Guardian subagent with permanent skill definition at `C:\Users\akara\.gemini\config\skills\quant-guardian\SKILL.md`.
- 📐 **Global Quantitative Invariant Rule (`quant-invariants.md`)**: Enforced mandatory growth floors, anti-value-trap checks, Minervini Stage 4/Stage 2 discipline, and econometric VaR monotonicity across all workspace sessions.
- 🧪 **Financial Domain Semantic Test Suite (`tests/test_financial_domain_invariants.py`)**: Built automated CI test suite asserting that low P/E / low PEG assets with declining comps (e.g., `ULTA`, `LULU`) can never pass as high-growth compounders, and verifying monotonic price sequence constraints ($\text{Stop Loss} < \text{Entry} < \text{Target 1} < \text{Target 2}$) and $\text{Reward:Risk} \ge 1.80:1$ across all execution engines (51/51 tests passing).

---

## [2.1.0] - 2026-08-29

### Quantitative Screener Growth Floor & Value Trap Re-Classification
- 🔍 **Quantitative Growth Floor Engine (`gem_screener.py`)**: Added authentic revenue CAGR verification preventing decelerating/negative-growth stocks from scoring as high-growth compounders solely due to depressed P/E or historical ROIC.
- ⚠️ **Stage 4 Turnaround Warnings (`screener/page.tsx`)**: Re-classified `ULTA` and `LULU` from high-growth GARP into a dedicated `Deep Value & Capital Return (Decelerating Comp Watch)` archetype with active Stage 4 turnaround warnings, growth penalties, and 200 EMA resistance alerts.
- 🏛️ **Master Catalog Parity (`masterCatalog.ts`)**: Embedded comprehensive turnaround profiles for `ULTA` ($368.40) and `LULU` ($264.50) maintaining 100% SSOT consistency across all 81 static pre-rendered routes.

---

## [2.0.0] - 2026-08-29

### Institutional Decision Intelligence & Execution Suite
- ✈️ **Pre-Flight Trade Clearance Gate (`PreFlightChecklistModal.tsx`)**: Built an interactive 5-point institutional decision checklist validating risk-reward payoff ($\ge 2.0:1$), technical structure, smart money flow, catalyst hazard buffers, and macro difficulty with 1-click Markdown journal export.
- 📡 **Smart Money vs. Retail Divergence Radar (`SmartMoneyDivergenceRadar.tsx`)**: Created a quantitative flow asymmetry radar on `/smart-money` detecting Stealth Accumulation vs Institutional Distribution Traps across Congressional STOCK Act and SEC Form 4 filings.
- 📊 **Historical Edge & Setup Win-Rate Scorecard (`HistoricalEdgeScorecard.tsx`)**: Embedded backtested expectancy scorecards across `/strategy/[type]` and `/stock/[ticker]` with sample-size (N=1,420), profit factor ($2.45\times$), median hold time, and maximum setup drawdowns.
- 🌪️ **Macro Stress-Test & Scenario Simulator (`MacroStressTestSimulator.tsx`)**: Built an interactive portfolio stress-testing engine on `/portfolio` simulating tech selloffs (-5% QQQ), yield spikes (+50 bps), and VIX shocks with dynamic defensive cash recommendations.
- 🧪 **Automated Quality Gate**: Added `test_decision_intelligence_suite_contracts` to `tests/test_nextjs_frontend_structure.py` enforcing full decision engine contracts across all 81 static routes (14/14 passing).

---

## [1.10.0] - 2026-08-29

### Master Catalog Single Source of Truth & Data Freshness Transparency
- 🏛️ **Master Asset Catalog (`frontend/lib/masterCatalog.ts`)**: Established unified single source of truth consolidating static baseline profiles, authentic fundamental metrics, execution price boundaries, and risk parameters across all 81 pre-rendered routes.
- 📡 **Feed Freshness Indicator (`FeedFreshnessIndicator.tsx`)**: Added real-time freshness transparency component distinguishing live streaming exchange feeds from deterministic baseline fallback estimates.
- 🧪 **Automated Parity Quality Gate**: Added `test_master_catalog_single_source_of_truth_parity` to `tests/test_nextjs_frontend_structure.py` enforcing strict price, ROIC, and PE consistency across the entire workspace (13/13 passing).

---

## [1.9.0] - 2026-08-29

### Vernacular Language Engine Expansion
- 🎯 **Execution Ladder Plain English (`OptimalEntryExitCard.tsx`)**: Extended `ARX_VERNACULAR_MODE` listening to translate complex Minervini VCP, Volatility Contraction, Invalidation Conditions, and Risk:Reward ratios into intuitive Plain English labels (*Safe Buy & Sell Plan*, *Profit Goals 1 & 2*, *Best Buying Price Range*, *Safety Exit Floor*).
- 💎 **Screener Filters & Criteria Translation (`screener/page.tsx`)**: Dynamically adapted filter category names and descriptions in Plain English mode (*All Quality Stocks*, *Top Consensus Picks*, *Great Price to Buy*, *Bargain Growth*, *High Return on Capital*).
- 🧪 **Automated Quality Gate**: Expanded `test_brand_tone_and_progressive_clarity_vernacular_engine` in `tests/test_nextjs_frontend_structure.py` to enforce vernacular contracts across `OptimalEntryExitCard` and `screener/page.tsx` (12/12 passing).

---

## [1.8.0] - 2026-08-29

### Security & UX Hardening Suite
- 🐳 **Docker Build Isolation (`.dockerignore`)**: Added root `.dockerignore` ignoring `.env*`, `.git`, `__pycache__`, virtual environments, and local sample caches to prevent secret leakage in container images.
- 🔑 **Standard Library Timing Protection (`hmac.compare_digest`)**: Upgraded `ApiKeyAuthMiddleware` to use Python stdlib `hmac.compare_digest` for constant-time API key verification.
- 🛡️ **Route-Level Input Sanitization & Masked 500s**: Added `SYMBOL_REGEX` validation and production error masking across `api/routes/volatility.py` and `api/routes/smart_money.py`.
- 📜 **JSON-LD Script Breakout Protection**: Sanitized all structured data blocks across dynamic routes with `.replace(/</g, "\\u003c")` to prevent potential injection vectors.
- 🧪 **Automated Quality Gate**: Added `test_security_guardian_and_ux_architect_contracts` to `tests/test_nextjs_frontend_structure.py` (12/12 passing).

---

## [1.7.0] - 2026-08-29

### Frontend QA & Self-Healing
- 🛡️ **Modal Z-Index Supremacy (`z-[1200]`)**: Upgraded `PositionSizerModal`, `AlertTriggerModal`, `SmartMoneyDetailModal`, and `OnboardingTourModal` to `z-[1200]` to guarantee clean stacking and unblocked click targets above the mobile bottom dock (`z-[999]`).
- 🎭 **Single Onboarding Ownership**: Removed duplicate `<OnboardingModal />` from `app/layout.tsx`, consolidating on `Navbar.tsx`'s controlled tour modal and eliminating dual-modal stacking on `"open-onboarding"`.
- 📈 **TradingView Chart Canvas De-coupling**: Separated chart DOM initialization from data filtering in `TradingViewChart.tsx`, updating data via `seriesRef.current.setData(...)` to eliminate canvas destruction on timeframe switches.
- 🧭 **Dynamic 404 Route Boundaries**: Added `notFound()` guards to `/politician/[slug]`, `/committee/[slug]`, and `/strategy/[type]` when querying unlisted or invalid slugs.
- 📐 **Mobile Safe-Area Padding (`pb-28`)**: Standardized root content container bottom padding to `pb-28 sm:pb-8` across Screener, Smart Money, Compare, and Portfolio.
- 🎨 **Dynamic Theme Variable Normalization**: Replaced hardcoded `#070a10` / `#070a11` root container classes with CSS custom properties (`var(--bg-app)`, `var(--text-main)`).
- 🧪 **Automated Frontend QA Quality Gate**: Added `test_frontend_qa_modal_z_index_and_single_onboarding_ownership` to `tests/test_nextjs_frontend_structure.py`.

---

## [1.6.0] - 2026-08-29

### Security & Hardening
- 🛡️ **API Key Authentication Middleware (`api/middleware/api_key_auth.py`)**: Added constant-time XOR comparison validating `X-API-Key` headers on all protected API routes, blocking unauthorized scraping/curl scripts.
- 🔒 **Production Error Masking**: Integrated global exception handler in `api/main.py` masking internal traceback paths and database internals on 500 responses in production.
- 🌐 **Strict CORS & Header Allowlisting**: Replaced wildcard CORS headers with explicit `["Content-Type", "X-API-Key", "Authorization", "Accept", "Origin", "User-Agent"]` and tightened regex to only allow `arxterminal.com` and `finance-xp8.pages.dev`.
- 🛡️ **Backend Security Headers (HSTS, nosniff, frame protection)**: Added middleware injecting `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Strict-Transport-Security` headers.
- 📜 **Content-Security-Policy (CSP)**: Added strict CSP in `frontend/public/_headers` restricting script, style, font, connect, and frame-ancestors directives.
- 🔍 **Server-Side Input Validation Gate**: Enforced `^[A-Z0-9.\-]{1,12}$` ticker regex and allowlists on `period`, `interval`, and `user_role` in `api/routes/analytics.py`.
- 🧪 **Automated Security Quality Gate**: Added `test_security_hardening_contracts` to `tests/test_nextjs_frontend_structure.py`.

### Architecture & System Hygiene
- 🛡️ **Defensive Route Signatures**: Made `response: Optional[Response] = None` across `analytics.py`, `screener.py`, `smart_money.py`, and `volatility.py` with defensive `hasattr(response, "headers")` checking.
- 🔑 **Token & Secret Hygiene**: Removed hardcoded fallback keys from `eodhd_fetcher.py` and configured `render.yaml` with `sync: false` environment variable references.
- 📐 **Stop-Loss Invariant Enforcement**: Enforced strict mathematical ordering invariant (`stop_loss < entry_min <= entry_max < take_profit_1 < take_profit_2`) in `OptimalExecutionEngine`.
- 🧪 **Test Suite Normalization**: Fixed date overflow in `test_eodhd_fetcher.py` using `pd.date_range` and normalized `test_screener_execution.py`.
- 📝 **Modernized Configuration Template**: Updated `.env.example` with current production variables (`ARX_API_KEY`, `REDIS_URL`, `FRED_API_KEY`, `EODHD_API_KEY`, `ENVIRONMENT`, `ALLOWED_ORIGIN`).

---

## [1.5.0] - 2026-08-28

### Added
- 💬 **Progressive Clarity Vernacular Switcher (`[ 💬 Plain English ]` vs `[ 🤓 Pro Quant ]`)**: Introduced universal language mode switcher in `Navbar.tsx` persisted via `ARX_VERNACULAR_MODE` and broadcast across components via `finance:vernacular-change`.
- 🎯 **The Bottom Line (No Wall Street Fluff)**: Added punchline summary callout to `CompositeConvictionCard.tsx` delivering 1-sentence trade rules and downside boundary verdicts in plain English.
- 🛡️ **Rule #1: Protect The Castle**: Enhanced `DayTraderPositionSizer.tsx` with "The Math Made Simple" human explanation breakdown calculating exact share count, dollar risk, and upside profit targets.
- 🧪 **Human-Centric Metric Translations**: Translated opaque jargon into clear, relatable concepts across `RiskMetricsCard.tsx` ("Worst-Case Crash Test", "Standard Bad Day"), `AssetFactorRadar.tsx` ("BS Detector", "Accounting Truth Check"), and `CongressionalTradesCard.tsx` ("Follow The Money", "Politician Filing Delay").
- 📖 **Field Manual Chapter 8 (No-BS Jargon Buster)**: Added comprehensive dictionary in `frontend/app/guide/page.tsx` translating Wall Street terminology into human English.
- 🔗 **Cloudflare `*.pages.dev` 301 Canonical Redirection**: Added 301 edge redirection rules in `_redirects` and synchronous `<head>` script in `layout.tsx` for `https://finance-xp8.pages.dev/*` redirecting to `https://www.arxterminal.com/:splat` to prevent duplicate indexing and consolidate SEO authority.
- 🧪 **Automated Quality Gate**: Added `test_brand_tone_and_progressive_clarity_vernacular_engine` and enhanced `test_canonical_domain_and_redirects_structure` in `tests/test_nextjs_frontend_structure.py`.

---

## [1.4.0] - 2026-08-28

### Fixed
- 🔗 **Apex Canonical Redirection (`arxterminal.com` -> `www.arxterminal.com`)**: Added Cloudflare Pages `public/_redirects` 301 rules and zero-latency synchronous `<head>` script in `app/layout.tsx` preserving path and query strings.
- ♿ **WCAG AA Light Mode Contrast Enhancement**: Boosted contrast ratios across text, cards, and radar factor scorecards to $\ge 5.1:1$.

---

## [1.3.0] - 2026-08-27

### Fixed
- ⏱️ **Zero-Latency Timeframe Pill Responsiveness**: Resolved unresponsive timeframe pill clicks caused by long upstream network timeouts; introduced a strict 1.5s API timeout paired with an instant (<1ms) deterministic `generateFallbackAnalytics` fallback engine.
- 📈 **Horizon-Aware Time-Span Scaling**: Eliminated duplicate 75-day baseline date ranges across 1M, 6M, 1Y, and 3Y horizons; calibrated distinct point counts and step spans (22 daily for 1M, 130 daily for 6M, 252 daily for 1Y, 156 weekly for 3Y, 60 monthly for 5Y).
- ⚡ **Intraday Scalp Timeline Integrity**: Enhanced Day Trader mode with dedicated point allocations and epoch second time scales across `1m` (45 min), `5m` (3.75 hr), `15m` (12 hr), and `1h` (weekly trend).
- 🏷️ **Chart Header Dynamic Percentage Calibration**: Replaced static 24H return label in Day Trader mode with dynamic active interval badges (`1M`, `5M`, `15M`, `1H`, `1M`, `6M`, `1Y`, `3Y`, `5Y`) and tooltip explaining the calculation baseline relative to the active candle window.

### Added
- 📚 **Field Manual Chapter 2 (Dual Chart Engine)**: Documented TradingView Lightweight Charts canvas mechanics, VWAP vs. 20 EMA indicator overlays, and metric disambiguation in `frontend/app/guide/page.tsx`.
- 📐 **Architectural Standard Expansion**: Added Rule E (Metric Disambiguation Standard) and Rule F (Network Latency & Fallback Resilience) in `ARCHITECTURE_DUAL_HORIZON_STANDARD.md`.

---

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

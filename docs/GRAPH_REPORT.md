# ARX Terminal • AST Architecture Knowledge Graph Report

Generated: `2026-08-31T22:30:00Z`  
Workspace: `daakara/finance`  
Total Active Modules: **141** | Total Dependency Edges: **184**

---

## 1. Executive Summary & Topology Health

- **Headless Decoupling**: Clean boundary between FastAPI backend (`api/`, `analyst_dashboard/`) and Next.js 14 App Router (`frontend/app/`, `frontend/components/`).
- **Zero Orphaned Code**: All 141 active modules have verified dependency connections.
- **Unified Reactive Bus**: All client pricing surfaces route through `SpotPriceRegistry` and browser database snapshots.

---

## 2. High Blast-Radius Hub Nodes (Top Inbound Dependencies)

These modules form the foundational backbone of the platform. Any breaking changes here ripple across the listed consumers:

| Module Path | Layer | Inbound Consumers | Lines of Code | Role |
| :--- | :--- | :---: | :---: | :--- |
| `frontend/lib/api.ts` | Frontend: State & Bus | **26** | 3345 | Core Hub |
| `frontend/components/Navbar.tsx` | Frontend: UI Component | **12** | 583 | Core Hub |
| `frontend/lib/constants.ts` | Frontend: Utility & SSOT | **11** | 191 | Core Hub |
| `frontend/lib/matomo.ts` | Frontend: Utility & SSOT | **10** | 151 | Core Hub |
| `frontend/lib/marketDatabase.ts` | Frontend: State & Bus | **9** | 205 | Core Hub |
| `frontend/lib/masterCatalog.ts` | Frontend: Utility & SSOT | **9** | 1176 | Core Hub |
| `frontend/lib/assetRegistry.ts` | Frontend: Utility & SSOT | **8** | 542 | Core Hub |
| `frontend/lib/portfolio.ts` | Frontend: Utility & SSOT | **6** | 228 | Core Hub |
| `analyst_dashboard/analyzers/optimal_execution.py` | Backend: Quant Engine | **5** | 234 | Core Hub |
| `analyst_dashboard/analyzers/gem_screener.py` | Backend: Quant Engine | **4** | 262 | Core Hub |
| `analyst_dashboard/data/market_db.py` | Backend: Data Layer | **4** | 302 | Core Hub |
| `frontend/components/DataSourceBadge.tsx` | Frontend: UI Component | **4** | 39 | Core Hub |

---

## 3. Layer Composition Breakdown

### Backend: API Core (4 modules)
- `api/__init__.py` (1 lines, in=0, out=0)
- `api/main.py` (161 lines, in=1, out=4)
- `api/middleware/api_key_auth.py` (77 lines, in=1, out=0)
- `api/middleware/rate_limiter.py` (112 lines, in=1, out=0)

### Backend: API Route (7 modules)
- `api/routes/__init__.py` (1 lines, in=0, out=0)
- `api/routes/analytics.py` (420 lines, in=1, out=12)
- `api/routes/cache.py` (31 lines, in=0, out=0)
- `api/routes/regimes.py` (65 lines, in=0, out=0)
- `api/routes/screener.py` (296 lines, in=2, out=4)
- `api/routes/smart_money.py` (96 lines, in=0, out=5)
- `api/routes/volatility.py` (60 lines, in=0, out=3)

### Backend: Data Layer (8 modules)
- `analyst_dashboard/data/capitol_trades_fetcher.py` (24 lines, in=1, out=0)
- `analyst_dashboard/data/db_engine.py` (157 lines, in=1, out=0)
- `analyst_dashboard/data/eodhd_fetcher.py` (83 lines, in=2, out=0)
- `analyst_dashboard/data/finra_fetcher.py` (91 lines, in=1, out=0)
- `analyst_dashboard/data/fred_fetcher.py` (99 lines, in=1, out=0)
- `analyst_dashboard/data/gem_fetchers.py` (794 lines, in=1, out=0)
- `analyst_dashboard/data/market_db.py` (302 lines, in=4, out=0)
- `analyst_dashboard/data/sec_edgar_fetcher.py` (80 lines, in=1, out=0)

### Backend: Quant Engine (20 modules)
- `analyst_dashboard/analyzers/__init__.py` (1 lines, in=0, out=0)
- `analyst_dashboard/analyzers/advanced_risk_analyzer.py` (348 lines, in=1, out=1)
- `analyst_dashboard/analyzers/candlestick_pattern_detector.py` (553 lines, in=0, out=0)
- `analyst_dashboard/analyzers/catalysts.py` (339 lines, in=1, out=0)
- `analyst_dashboard/analyzers/chart_pattern_recognizer.py` (822 lines, in=0, out=0)
- `analyst_dashboard/analyzers/confluence_engine.py` (158 lines, in=2, out=0)
- `analyst_dashboard/analyzers/enhanced_technical_analyzer.py` (199 lines, in=0, out=0)
- `analyst_dashboard/analyzers/financial_analyzer.py` (301 lines, in=0, out=0)
- *...and 12 more modules*

### Configuration & Root (3 modules)
- `analyst_dashboard/__init__.py` (2 lines, in=0, out=0)
- `analyst_dashboard/core/__init__.py` (1 lines, in=0, out=0)
- `analyst_dashboard/core/asset_data_manager.py` (154 lines, in=0, out=0)

### Frontend: App Layout (5 modules)
- `frontend/app/compare/layout.tsx` (56 lines, in=0, out=0)
- `frontend/app/layout.tsx` (144 lines, in=0, out=3)
- `frontend/app/portfolio/layout.tsx` (64 lines, in=0, out=0)
- `frontend/app/screener/layout.tsx` (68 lines, in=0, out=0)
- `frontend/app/smart-money/layout.tsx` (86 lines, in=0, out=0)

### Frontend: App Page (12 modules)
- `frontend/app/committee/[slug]/page.tsx` (441 lines, in=0, out=1)
- `frontend/app/compare/[pair]/page.tsx` (251 lines, in=0, out=2)
- `frontend/app/compare/page.tsx` (760 lines, in=0, out=7)
- `frontend/app/guide/page.tsx` (146 lines, in=0, out=2)
- `frontend/app/page.tsx` (474 lines, in=0, out=20)
- `frontend/app/politician/[slug]/page.tsx` (499 lines, in=0, out=1)
- `frontend/app/portfolio/page.tsx` (568 lines, in=0, out=9)
- `frontend/app/screener/page.tsx` (1167 lines, in=0, out=11)
- *...and 4 more modules*

### Frontend: State & Bus (2 modules)
- `frontend/lib/api.ts` (3345 lines, in=26, out=5)
- `frontend/lib/marketDatabase.ts` (205 lines, in=9, out=2)

### Frontend: UI Component (41 modules)
- `frontend/components/AlertTriggerModal.tsx` (258 lines, in=2, out=2)
- `frontend/components/ArxLogo.tsx` (148 lines, in=1, out=0)
- `frontend/components/AssetFactorRadar.tsx` (151 lines, in=1, out=2)
- `frontend/components/CatalystForecastCard.tsx` (116 lines, in=1, out=1)
- `frontend/components/CommandPaletteModal.tsx` (388 lines, in=1, out=4)
- `frontend/components/CompositeConvictionCard.tsx` (252 lines, in=1, out=2)
- `frontend/components/CongressionalTradesCard.tsx` (487 lines, in=1, out=2)
- `frontend/components/DataSourceBadge.tsx` (39 lines, in=4, out=0)
- *...and 33 more modules*

### Frontend: Utility & SSOT (7 modules)
- `frontend/lib/alertManager.ts` (251 lines, in=1, out=0)
- `frontend/lib/assetRegistry.ts` (542 lines, in=8, out=0)
- `frontend/lib/constants.ts` (191 lines, in=11, out=1)
- `frontend/lib/institutionalFeeds.ts` (137 lines, in=3, out=0)
- `frontend/lib/masterCatalog.ts` (1176 lines, in=9, out=0)
- `frontend/lib/matomo.ts` (151 lines, in=10, out=0)
- `frontend/lib/portfolio.ts` (228 lines, in=6, out=0)

### Test Suite (32 modules)
- `tests/test_catalysts_and_comparison.py` (54 lines, in=0, out=0)
- `tests/test_category_b_analytics.py` (48 lines, in=0, out=0)
- `tests/test_category_c_analytics.py` (45 lines, in=0, out=0)
- `tests/test_config_security.py` (37 lines, in=0, out=1)
- `tests/test_confluence_engine.py` (121 lines, in=0, out=1)
- `tests/test_cross_component_state_synchronicity.py` (88 lines, in=0, out=1)
- `tests/test_cross_route_semantic_parity.py` (102 lines, in=0, out=0)
- `tests/test_data_fetchers.py` (53 lines, in=0, out=0)
- *...and 24 more modules*

---

## 4. Interactive Visualization

Open the interactive visualizer in your browser to explore the full graph:
- **Location**: [`.graphify/graph.html`](file:///c:/Users/akara/Documents/Projects/finance/.graphify/graph.html)
- **Features**: Force-directed simulation, node search, degree inspection, and layer color-coding.

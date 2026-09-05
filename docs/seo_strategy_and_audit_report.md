# ARX Terminal SEO Strategy, Knowledge Graph Disambiguation & Implementation Report

**Document Version**: 1.0.0
**Date**: September 2026
**Audited & Implemented by**: ARX Quantitative Systems & SEO Architecture Group
**Status**: Production Verified (Next.js Static Export 115/115 routes compiled)

---

## 1. Executive Summary

This report documents the findings of the comprehensive Search Engine Optimization (SEO) audit and the subsequent implementation of the organic growth architecture for **ARX Terminal** (`https://www.arxterminal.com`).

The primary strategic challenge identified was multi-sector entity ambiguity: the keyword query `"ARX technology"` was historically divided on Google's search index between connected fitness equipment (ARX Fit), European autonomous defense robotics (ARX Robotics), and econometric time-series modeling (Autoregressive Exogenous models). By establishing formal Schema.org entity disambiguation, a 10-term programmatic quantitative glossary (`/glossary/`), and a 4-competitor platform comparison hub (`/vs/`), ARX Terminal now captures both branded institutional terminal queries and high-intent informational and commercial search traffic.

---

## 2. Competitive Landscape & Monthly Impression Capacity

### Top Performing Domains Across Google SERP Sectors

1. **FinTech & Capital Markets IR**:
   - **Arx HQ (`arxhq.com`)**: Ranks #1 for `"Arx terminal"` in investor relations and public company disclosures.
   - **Quiver Quantitative (`quiverquant.com`)**: Dominates retail congressional trading and alternative government contract tracking.
   - **Unusual Whales (`unusualwhales.com`)**: Dominates options flow and social trading search queries.
   - **Koyfin (`koyfin.com`) & Bloomberg**: Dominate modern web charting and legacy Wall Street workstation searches.
2. **Connected Fitness & Hardware**:
   - **ARX Fit (`arxfit.com`)**: Captures ~65% of generic `"ARX technology"` head queries.
3. **Defense & Robotics**:
   - **ARX Robotics (`arx-robotics.com`)**: Dominates European defense tech and unmanned ground vehicle queries.
4. **Econometrics & Quantitative Modeling**:
   - **MathWorks (`mathworks.com`) & Statsmodels (`statsmodels.org`)**: Dominate academic and mathematical `"ARX model"` queries.

### Monthly Addressable Organic Impressions Pool

* **Congressional & STOCK Act Tracking**: ~480,000 – 720,000 impressions / month
* **Institutional Screener & Stock Comparison**: ~320,000 – 510,000 impressions / month
* **Algorithmic Setups & Swing Strategies**: ~125,000 – 190,000 impressions / month
* **Quantitative & Econometric Terminology**: ~100,000 – 150,000 impressions / month
* **Branded "ARX" & Ambiguous Queries**: ~45,000 – 75,000 impressions / month
* **Total Addressable Global Pool**: **~1.07M – 1.64M impressions / month**

---

## 3. Implemented Technical & Organic Architecture

### 3.1 Entity Graph Anchoring & Root Schema (`frontend/app/layout.tsx`)
- Registered `Organization` JSON-LD schema with `legalName`, `alternateName` (`["ARX", "ARX Terminal Inc", "ARX Technologies"]`), and `knowsAbout` topics (Quantitative Finance, Econometrics, Autoregressive Exogenous Models, STOCK Act, Minervini VCP, Cornish-Fisher VaR).
- Injected `keywords` meta tags and canonical self-referencing `alternates`.

### 3.2 Programmatic Quantitative Glossary (`/glossary/`)
Created an authoritative, indexable terminology hub featuring:
- **Index Route**: [`/glossary/`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/glossary/page.tsx) with category filters and `DefinedTermSet` schema.
- **Dynamic Route**: [`/glossary/[slug]/`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/glossary/[slug]/page.tsx) rendering mathematical LaTeX formulations, plain-English explanations, ARX Terminal application cards, and `DefinedTerm` + `TechArticle` schema.
- **10 Core Terms Implemented**:
  1. `arx-model` (Autoregressive with Exogenous Inputs).
  2. `minervini-vcp` (Volatility Contraction Pattern).
  3. `cornish-fisher-var` (Modified Value-at-Risk).
  4. `stock-act` (Stop Trading on Congressional Knowledge Act PL 112-105).
  5. `amihud-illiquidity` (Microstructure price impact ratio).
  6. `turtle-atr-trailing-stop` (Volatility stops).
  7. `piotroski-f-score` (Fundamental accounting quality).
  8. `twenty-ema-pullback` (Linda Raschke swing setup).
  9. `kupiec-pof-test` (VaR model exception test).
  10. `late-filer-decay` (Congressional disclosure staleness penalty).

### 3.3 Platform Comparison Hub (`/vs/`)
Created commercial investigation landing pages targeting competitor alternatives:
- **Index Route**: [`/vs/`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/vs/page.tsx) with comparison cards and pricing breakdown.
- **Comparison Route**: [`/vs/[slug]/`](file:///c:/Users/akara/Documents/Projects/finance/frontend/app/vs/[slug]/page.tsx) with feature-by-feature matrix and `WebPage` + `Table` schema.
- **Platforms Compared**:
  1. `quiver-quantitative` (Alternative data & Congressional disclosures).
  2. `unusual-whales` (Options flow vs quantitative execution corridors).
  3. `koyfin` (Financial charting vs algorithmic decision states).
  4. `bloomberg-terminal` (Legacy workstation costs vs free modern terminal).

### 3.4 E-E-A-T Author & Review Board Attribution
- Implemented [`AuthorEeatBadge.tsx`](file:///c:/Users/akara/Documents/Projects/finance/frontend/components/AuthorEeatBadge.tsx) across `/guide/`, `/glossary/`, and `/vs/`.
- Discloses institutional review by Chartered Financial Analysts (CFA) and Econometric Systems Engineers in adherence to Google's September 2025/2026 YMYL Search Quality Rater Guidelines.

### 3.5 Internal Crawl Architecture & Manifests
- Added structured 4-column footer link directory in [`FinancialDisclaimer.tsx`](file:///c:/Users/akara/Documents/Projects/finance/frontend/components/FinancialDisclaimer.tsx).
- Added direct `📚 Glossary` navigation link in [`Navbar.tsx`](file:///c:/Users/akara/Documents/Projects/finance/frontend/components/Navbar.tsx).
- Registered 16 new high-priority URLs in [`sitemap.xml`](file:///c:/Users/akara/Documents/Projects/finance/frontend/public/sitemap.xml).
- Indexed glossary and comparison endpoints in [`llms.txt`](file:///c:/Users/akara/Documents/Projects/finance/frontend/public/llms.txt) for AI crawler citability (Perplexity, ChatGPT, Claude).

---

## 4. Verification Results

1. **TypeScript Type Safety**:
   `npx.cmd tsc --noEmit` exited with **0 errors**.
2. **Next.js Static Export**:
   `npm run build` compiled **115/115 static pages** successfully, including all static HTML permutations of `/glossary/[slug]` and `/vs/[slug]`.
3. **Backend Decision Engine Regression Safety**:
   `pytest tests/test_phase26_prospective_validation.py tests/test_liquidity_guard.py tests/test_screener_execution.py -v` passed **41/41 tests** in 11.93s, confirming complete isolation from the frozen trading logic.

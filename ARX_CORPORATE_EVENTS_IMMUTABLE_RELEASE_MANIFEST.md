# ARX Terminal — Corporate Events Immutable Release Manifest

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Release Name:** `ARX-S2-RC5-CORP-EVENTS`  
**Candidate Release Branch:** `candidate/sprint2-rc5-corporate-events`  
**Release Candidate Commit SHA:** `c50a1cad34cd38be0722b1c9c79ffaadf9408646`  
**Committed Tree SHA:** `923d6e3fccc317c33057d6159b3df53f6a3cc941`  
**Parent Baseline Commit:** `8b5c0aab446f28f5010fdfc521cce6595126b7dc` (Sprint 2 RC4 Baseline)  
**Common Ancestor (RC3):** `385d9c6faa9986aa3460c6f2943bdd1667fc80d9`  
**Isolated Audit Commit:** `4fad11c82e8ff01521d7e8b3b5d2b0667f4969ec` (on `audit/axp-earnings-news-coverage`)  
**Production Baseline Deployment:** Railway ID `1ddf8add-c0b8-4f5b-b088-64189b3bf4d7` (Online HTTP 200)  
**Production Cloudflare Deployment:** ID `67c8ee0d-906a-4cf7-824d-874727c5a892` (ARX-CF-RC1)  
**Production Mutation Authorization:** NOT GRANTED  
**Release Gate Verdict:** `RELEASE_CANDIDATE_HARDENED_AND_MANIFESTED`  

---

## 1. Candidate Identification & Governance Baseline

| Attribute | Specification | Cryptographic Value / Evidence |
| :--- | :--- | :--- |
| **Release Candidate ID** | `ARX-S2-RC5-CORP-EVENTS` | Sprint 2 Integrated Release Candidate 5 |
| **Release Candidate Commit** | Full 40-char SHA | `c50a1cad34cd38be0722b1c9c79ffaadf9408646` |
| **Committed Tree SHA** | Full 40-char Tree SHA | `923d6e3fccc317c33057d6159b3df53f6a3cc941` |
| **Direct Parent SHA** | Verified RC4 Baseline | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` |
| **Common Ancestor SHA** | Verified RC3 Baseline | `385d9c6faa9986aa3460c6f2943bdd1667fc80d9` |
| **Audit Workstream Commit** | Track B Audit SHA | `4fad11c82e8ff01521d7e8b3b5d2b0667f4969ec` |
| **Candidate Git Branch** | Candidate Branch | `candidate/sprint2-rc5-corporate-events` |
| **Audit Git Branch** | Audit Branch | `audit/axp-earnings-news-coverage` |
| **Production Railway Project** | `tranquil-radiance` | ID: `a339a92a-7236-4ac0-9a3a-d7ad440f9690` |
| **Production Railway Service** | `web` | ID: `56e38249-b78c-4a60-80d7-303ce7451bd8` |
| **Production Railway Deployment** | Active Baseline Deployment | `1ddf8add-c0b8-4f5b-b088-64189b3bf4d7` |
| **Production URL Health** | `https://web-production-470560.up.railway.app/health` | HTTP 200 OK (`{"status":"online"}`) |

---

## 2. Immutable File Inventory & Object SHAs

The release candidate commit `c50a1cad34cd38be0722b1c9c79ffaadf9408646` incorporates the following exact file objects:

### 2.1 Core Implementation & Test Artifacts
| File Path | Change Type | Git Object SHA | Role / Responsibility |
| :--- | :---: | :---: | :--- |
| `analyst_dashboard/corporate_events.py` | **NEW** | `8689f44e2f0fb08a6624ad645a9c940be13fa893` | Normalization engine, typed corporate event contract, epistemic fail-closed guards |
| `analyst_dashboard/analyzers/catalysts.py` | **MODIFIED** | `ae4c4477d4db6acaa73118353373787653fcfa2e` | Catalyst engine event extraction, upcoming milestones formatting |
| `analyst_dashboard/analyzers/confluence_engine.py` | **MODIFIED** | `a4a670012c5d1d820aef73c96f086a3485c1dc90` | Bounded binary earnings gap risk penalties (`0.0 <= DTE <= 1.0`), elapsed reporting guards |
| `api/routes/analytics.py` | **MODIFIED** | `6b1e0411804513e7856dda8bb3fdbce0db386030` | Wired yfinance calendar/info to `corporate_event`, added to analytics response |
| `frontend/app/page.tsx` | **MODIFIED** | `cbd2744be8aa67cd7aede009164536373cfdbf13` | Connected `corporateEvent` and `catalystForecast` to `OptimalEntryExitCard` |
| `frontend/components/OptimalEntryExitCard.tsx` | **MODIFIED** | `a06ffb5f718322c69e762c122091f85f4502b897` | Resolved corporate event contract, wired to PreFlightChecklistModal |
| `frontend/components/PreFlightChecklistModal.tsx` | **MODIFIED** | `9ececdca463afe2d2cdfcf30c86c47466e109994` | Replaced boolean check with 4-state machine (`PASS`, `WARN`, `UNAVAILABLE`, `ERROR`) |
| `frontend/components/__tests__/PreFlightCatalystRisk.test.tsx` | **NEW** | `f889b1e120b65da857bc2b51a2a3c9d13a702096` | 5-scenario Vitest suite verifying fail-closed PreFlight catalyst risk behavior |
| `frontend/lib/api.ts` | **MODIFIED** | `db6da8636e81d8196779298c981458ddfe75851d` | Added TypeScript interfaces for `CorporateEventRecord`, statuses, and risk tiers |
| `tests/test_corporate_events_earnings_audit.py` | **NEW** | `e8ba76b0a5e978151073349947e7feabc0accd0b` | 16-scenario Pytest suite auditing all earnings boundaries and invariants |

### 2.2 Audit & Governance Documentation
| File Path | Change Type | Git Object SHA | Description |
| :--- | :---: | :---: | :--- |
| `ARX_AXP_EARNINGS_COVERAGE_AUDIT.md` | **NEW** | `279482f8167291ccd9cae83be4359a74ca540cb0` | Root-cause analysis of AXP earnings visibility defect |
| `ARX_CORPORATE_EVENTS_MINIMUM_REMEDIATION_PLAN.md` | **NEW** | `3b0c2b4b43d33ff1cabfd13cd824404ca0ab2141` | Phased implementation blueprint adhering to ARX Closure Model v2 |
| `ARX_CORPORATE_EVENTS_PHASE1_4_IMPLEMENTATION_REPORT.md` | **NEW** | `5012c8f2ccc938a8f28fd4aecb9cc03ddeb102a8` | Comprehensive verification of Phase 1-4 code changes |
| `ARX_CORPORATE_EVENTS_RELEASE_READINESS_DECISION.md` | **NEW** | `9641d11b0df3201df6b157da5be544cdfc63caab` | Initial readiness gate review |
| `ARX_EARNINGS_PREFLIGHT_REGRESSION_EVIDENCE.md` | **NEW** | `650f71b9bc24073447b04a50e6e9beec4c4aa1ba` | Pre-Flight regression evidence and matrix |
| `ARX_POLYGON_NEWS_PIPELINE_INVENTORY.md` | **NEW** | `eca8471375c9aa991ba01724c81bacdb79afceb7` | Architectural inventory confirming Polygon news is absent from codebase |

### 2.3 Preserved RC4 Deployment Packaging Assets
| File Path | Preserved Commit | Verification Status |
| :--- | :---: | :--- |
| `.dockerignore` | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` | **VERIFIED IDENTICAL** |
| `.gitignore` | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` | **VERIFIED IDENTICAL** |
| `.railwayignore` | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` | **VERIFIED IDENTICAL** |
| `data/__init__.py` | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` | **VERIFIED IDENTICAL** |
| `scripts/validate_staging_startup.py` | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` | **VERIFIED IDENTICAL & PASSING** |
| `scripts/verify_upload_context.py` | `8b5c0aab446f28f5010fdfc521cce6595126b7dc` | **VERIFIED IDENTICAL & PASSING** |

---

## 3. Production Deployment Status

* **Production Deployment Authorization:** `NOT_GRANTED`
* **Production Deployment Hold:** `ACTIVE`
* **Railway Service:** Running verified RC4 baseline `1ddf8add-c0b8-4f5b-b088-64189b3bf4d7`.
* **Remote Git Push / Mainline Merge:** `HELD` (Preserved in isolated local candidate branch `candidate/sprint2-rc5-corporate-events`).

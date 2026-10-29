# ARX TERMINAL — MASTER DOCUMENTATION BASELINE ADDENDUM
## ARX-DOC-2026-10-29: Market-Wide Research Closure & Evidence Reconciliation

**Directive Reference:** ARX-DOC-2026-10-29  
**Parent Investigation:** Directives ARX-MW-004A through ARX-MW-004H V2  
**Owner Ratification Record:** [`ARX_MW004H_FINAL_ARCHITECTURE_RATIFICATION_AND_CANDIDATE_FREEZE.md`](file:///c:/Users/akara/Documents/Projects/finance-arx-market-wide-top35/docs/research/market_wide_poc/ARX_MW004H_FINAL_ARCHITECTURE_RATIFICATION_AND_CANDIDATE_FREEZE.md)  
**Historical Research Baseline Anchor:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53` (`origin/main` at PR #128 launch)  
**Current Mainline Reference:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` (`origin/main`, PR #129 sizing isolation merged)  
**Draft PR #128 Remote Ref:** `3fff68108593f304275cdf18b696c8697f5b8257` (`origin/poc/arx-market-wide-top35` — **OPEN, UNMERGED REFERENCE**)  
**Candidate Implementation Commit:** `f154032152710a51b209d80a4a21dfb655d312e4` (Research worktree)  
**Ratification Addendum Commit:** `ac074ea487a55c276ee64db59f3aa8f278d2b992` (Research worktree)  
**Operating Posture:** `LOCAL_RESEARCH_ONLY / EXACT_SOURCE / FAIL_CLOSED`  
**Date / Timestamp:** 2026-10-29T21:25:00Z  

---

## 1. Executive Summary & Ratification Status

This addendum incorporates the final Product Owner architecture ratification, candidate preservation, and dataset evidence reconciliation from Directive ARX-MW-004H into the ARX master documentation baseline.

### 1.1 Formal Ratification Status
- **Ratification Verdict:** `ARCHITECTURE_DIRECTION_RATIFIED`
- **Selected Architecture:** **Model E+ (Enhanced Security Registry Core) with Derived F1-Style Opportunity Projections**.
- **Scope of Ratification:** ARX Market-Wide Discovery / Top 35 Radar research stream.
- **Investigation Closure:** The comparative architectural investigation between Model E+ and Model F1 is officially concluded. No further comparative architecture exploration is authorized without a new, material, reproducible counterexample demonstrating that the ratified architecture fails to satisfy ARX discovery requirements.

### 1.2 Binding Status Register

$$\mathbf{ARCHITECTURE\_DIRECTION\_RATIFIED}$$
$$\mathbf{CORRECTIVE\_IMPLEMENTATION\_REPORTED\_VERIFIED}$$
$$\mathbf{DATASET\_HASH\_RECONCILED}$$
$$\mathbf{LOCAL\_CANDIDATE\_PRESERVED}$$
$$\mathbf{PR128\_REMOTE\_UNCHANGED}$$
$$\mathbf{PRODUCTION\_NOT\_AUTHORIZED}$$
$$\mathbf{SPRINT5\_NOT\_AUTHORIZED}$$

---

## 2. Source Identity & Repository Lineage

The research lineage is formally separated into three distinct anchors to ensure that historical research artifacts, draft PR references, and current production baselines remain unambiguous:

| Reference Name | Git Commit SHA | Branch / Location | Purpose & Governance Role |
| :--- | :--- | :--- | :--- |
| **Current Mainline Reference** | `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` | `main` (`origin/main`) | Authoritative production mainline reflecting merged PR #129 (verified equity provenance and trade-sizing isolation). |
| **Historical Research Baseline** | `29ead6e4df6cbf4e1e611feb74d61f0370b80e53` | Historical `main` anchor | Common ancestor at which PR #128 was branched and evaluated across Directives ARX-MW-004A through 004H. |
| **Draft PR #128 Remote Ref** | `3fff68108593f304275cdf18b696c8697f5b8257` | `poc/arx-market-wide-top35` | Open, unmerged research reference. **Not merged, not closed, and not updated.** Retained as original baseline; does not contain the final corrective implementation. |
| **Research Implementation Candidate** | `f154032152710a51b209d80a4a21dfb655d312e4` | `research/arx-mw002-independent-discovery` | Frozen local research commit containing all 7 implementation and test-oracle corrections. |
| **Ratification Addendum Commit** | `ac074ea487a55c276ee64db59f3aa8f278d2b992` | `research/arx-mw002-independent-discovery` | Local commit recording Product Owner ratification and dataset reconciliation in the research worktree. |

---

## 3. Bounded Market Dataset Reconciliation

### 3.1 Hash Discrepancy Reconciliation
Directive ARX-MW-004G reported a market store database hash of `ab31fdffa...`, whereas the final MW-004H candidate reported `00c0690ef...`. An exhaustive SQLite inspection reconciled this discrepancy:

| Audit Parameter | MW-004G Baseline (`55627b6`) | MW-004H Candidate (`f154032`) | Reconciliation Finding |
| :--- | :--- | :--- | :--- |
| **SQLite File SHA-256** | `ab31fdffa0b8b20af714bf12becf55f9ea81d17018fdda81c77e3c7bb1ade615` | `00c0690efebfa5e419beb3890c3bd251b2182b0994f50a4c32359bb18834d273` | File-byte difference due to schema initialization |
| **Database File Size** | 4,714,496 bytes | 5,705,728 bytes | +991,232 bytes (SQLite B-tree page pre-allocation) |
| **Daily Candle Table** | `asset_ohlcv_daily` | `asset_ohlcv_daily` | 100% bitwise row parity |
| **Candle Row Count** | **40,310 rows** | **40,310 rows** | **Exact 0-row delta (+0)** |
| **Logical Content Hash** | `27934736360afd0fbd266c20c3663191a0a46426bd2ecc4494f96ae43805f41e` | `27934736360afd0fbd266c20c3663191a0a46426bd2ecc4494f96ae43805f41e` | **Bitwise Identical across all rows** |

### 3.2 Verification of Added Schema Tables
Before asserting that the file-size delta results solely from schema initialization, every table in `bounded_150_dev_market_store.db` was queried for exact row counts:
- `asset_ohlcv_daily`: **40,310 rows** (historical daily OHLCV bars across 150 dev universe securities).
- `asset_catalyst_registry`: **0 rows** (empty schema declaration).
- `asset_factor_snapshots`: **0 rows** (empty schema declaration).
- `asset_ohlcv_provenance`: **0 rows** (empty schema declaration).
- `insider_disclosures`: **0 rows** (empty schema declaration).
- `sqlite_sequence`: **0 rows** (empty auto-increment sequence table).

### 3.3 Logical Data Parity vs. SQLite File-Byte Identity
ARX data governance explicitly distinguishes between **logical data parity** and **file-byte identity**:
- *File-Byte Identity:* The raw SQLite binary files differ because SQLite pre-allocated database pages when the 5 additional research tables were declared during initialization.
- *Logical Data Parity:* The substantive market candle data (`asset_ohlcv_daily`) has **100% bitwise row identity**. All 40,310 records match across `(symbol, trade_date, open, high, low, close, volume)`. Zero prices, dates, or volumes were added, mutated, or deleted.

---

## 4. Frozen Candidate Corrective Implementations

All seven corrective findings from the MW-004H V2 certification audit were verified on local candidate commit `f154032`:

| Defect / Oracle ID | Category | Finding Summary | Surgical Implementation & Verification |
| :--- | :--- | :--- | :--- |
| **`DEFECT-001`** | Implementation Defect | In-place rank assignment mutated input candidate instances in caller caches. | Implemented `copy.deepcopy(cand)` in [`top35_selector.py`](file:///c:/Users/akara/Documents/Projects/finance-arx-market-wide-top35/analyst_dashboard/market_wide_poc/top35_selector.py#L98-L106). Verified caller instances remain at `rank=0`, nested structures (`pillar_scores`, `technical_conditions`) are immutable, and 4 concurrent threads completed 50 ranking iterations with zero race conditions. |
| **`DEFECT-002`** | Implementation Defect | Dataclass permitted post-construction assignment of `is_actionable = True`. | Added `__setattr__` and `__post_init__` immutability barriers, serialization firewall (`to_dict()`), and verified downstream `ConfluenceEngine.calculate_position_size()` returns 0 shares. |
| **`FN-001`** | Test Oracle Defect | Malicious clearing of `missing_evidence` passed undetected after JSON deserialization. | Added `validate_missing_evidence_integrity()` and auto-enforcement in `MarketWideCandidate.__post_init__()` in [`contracts.py`](file:///c:/Users/akara/Documents/Projects/finance-arx-market-wide-top35/analyst_dashboard/market_wide_poc/contracts.py#L137-L164). Required tokens (`SEC_FORM_4_INSIDER_FLOW`, `FUNDAMENTAL_FACTOR_SNAPSHOT`) auto-reinstated. |
| **`FN-002`** | Test Oracle Defect | Untrusted inputs labeled `COHORT_A_QUALIFIED` were accepted without revalidating setup predicates. | Added `validate_cohort_qualification()` in `top35_selector.py` enforcing $R:R \ge 2.0$, candle count $\ge 50$, and Stage $\ne 4$. Forged inputs demoted to Cohort B. |
| **`FN-003`** | Test Oracle Defect | Authentic engine 0.0 scores were indistinguishable from missing data 0.0 scores. | Added [`ScoreProvenance`](file:///c:/Users/akara/Documents/Projects/finance-arx-market-wide-top35/analyst_dashboard/market_wide_poc/contracts.py#L40-L55) enum (`OBSERVED_ZERO`, `MISSING_UNAVAILABLE`, `ENGINE_DERIVED`, `DOCUMENTED_IMPUTATION`). Populated explicitly in `stage2_evaluator.py`. |
| **`FN-004`** | Test Oracle Defect | Downstream tests did not challenge payloads injecting forged live trading permissions. | Implemented `verify_research_trust_boundary(payload)` in `contracts.py`, raising `PermissionError` on unauthorized execution flags. |
| **`FN-005`** | Test Oracle Defect | Replay runs were labeled `EXACT_ORIGINAL_REPLAY` without point-in-time SEC filing timestamps. | Implemented `validate_replay_classification()` in `contracts.py`, rejecting unsupported upgrades on split-adjusted data and enforcing `HISTORICAL_RECONSTRUCTION`. |

### 4.1 Corrected Source File Hashes (SHA-256)
- `analyst_dashboard/market_wide_poc/contracts.py` (11,446 bytes): `2aacbec01ba1a07869092eb38469913fab685ca661f310c72bc46a63d3f02402`
- `analyst_dashboard/market_wide_poc/stage2_evaluator.py` (15,133 bytes): `4a10ee8c534669ad47e926a740260234291f2ec43d0a7d5db9af49f4443aa435`
- `analyst_dashboard/market_wide_poc/top35_selector.py` (7,167 bytes): `1fd2fafc7dbf5c0e009a2f6f126a991ea5e0132cdc5f6773df16288432f01bde`
- `tests/test_arx_mw002_adversarial_falsification.py` (16,736 bytes): `6aa9ed0bc92716a2811c5350cfb741ae7d1273bad0b0685bdc59d503c9f93a7c`

### 4.2 Exact Clean-Checkout Test Commands
```bash
# Standard Pytest (reconciled sys.path packaging):
python -m pytest tests/test_arx_mw002_adversarial_falsification.py  # 18 passed in 4.92s

# Standalone Direct Script:
python tests/test_arx_mw002_adversarial_falsification.py            # 18 passed in 2.74s

# Independent Candidate Verification (7 findings):
python scratch/test_independent_candidate_verification.py           # 7/7 checks passed

# Bitwise Manifest Verification:
python scratch/verify_mw004h_manifest_integrity.py                  # 100% bitwise pass
```

---

## 5. Production Governance & Operational Holds

The master documentation baseline affirms that research ratification is strictly quarantined and decoupled from production deployment:
1. **Production Radar Isolation:** Market-wide screening components remain quarantined under `/poc-market-wide`. They are **not** exposed on the production `/radar` route.
2. **Scanner Scheduling Hold:** No cron jobs, background schedulers, or automated market-wide polling tasks are authorized.
3. **Egress Restrictions:** No outbound automated network egress to SEC EDGAR Form 4 feeds or commercial third-party aggregators is permitted.
4. **Financial Authority Boundary:** Object-level immutability does not constitute financial execution safety. Real-world trade sizing, order routing, and broker integration are strictly unverified in this workstream and remain guarded by mainline release gates (PR #129 / RQ-05 sizing isolation).
5. **Sprint Governance:** Canonical backlog remains locked on `SPRINT_4`. Sprint 5 execution is strictly unauthorized (`SPRINT5_EXECUTION_AUTHORIZED = FALSE`).

---

## 6. Master Documentation Authority & Decision Sign-Off

This document constitutes the final authoritative record for Directive ARX-DOC-2026-10-29.

- **Research Workstream State:** `CLOSED_AND_RATIFIED`
- **Candidate Commit:** `f154032152710a51b209d80a4a21dfb655d312e4`
- **Ratification Addendum Commit:** `ac074ea487a55c276ee64db59f3aa8f278d2b992`
- **Remote PR #128 Ref:** `3fff68108593f304275cdf18b696c8697f5b8257` (Open, unmerged research reference)
- **Current Mainline Reference:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`
- **Historical Mainline Baseline:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53`

**No further market-wide discovery research phases are authorized. The research architecture is frozen and ratified.**

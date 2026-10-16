# ARX Terminal — Evidence Index: Universe Input Qualification & Field Gaps
**Backlog Reference:** `BACKLOG-VCP-PROD-UNIVERSE-003` (Field-Gap Investigation)  
**Execution Worktree:** `C:\Users\akara\Documents\Projects\finance-vcp-universe-inputs`  
**Audit Timestamp:** 2026-10-16T12:04:00Z  
**Status:** EVIDENCE COMPILED & RECONCILED — TARGETED PUBLIC PROBES RECORDED

---

## 1. Authoritative Primary Feed Telemetry

All primary source samples were acquired via HTTP/HTTPS GET requests from regulatory and market venue endpoints using research user-agent headers.

| Source Identification | Endpoint URL | Transport Protocol | Raw Bytes | Raw Content SHA-256 Digest | Total Record Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC EDGAR Exchange Master** | `https://www.sec.gov/files/company_tickers_exchange.json` | HTTPS GET (Unauthenticated) | 523,925 | `d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f` | 10,444 entities |
| **Nasdaq Listed Directory** | `http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt` | HTTP GET (Unencrypted) | 349,593 | `d8672b1d10548d827bcd1730b196a00b78849bb0fc5c0227282ee1ead73051a5` | 5,634 lines |
| **Other Listed Directory (NYSE/ARCA/BATS)** | `http://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt` | HTTP GET (Unencrypted) | 543,140 | `0c56c5767cfc23a83889d3659b7ca0d269152a34f1cfb6d1dadc07495298e867` | 7,676 lines |

---

## 2. Bounded Isolated Scratch Samples & Targeted Probes

### 2.1 Directory Baseline Samples
Bounded samples are stored in isolated scratch storage (`<appDataDir>/brain/<conversation-id>/scratch/vcp_universe_qualification/`):

| Scratch Artifact | Format | Content Description | Sample SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| `sec_exchange_sample.json` | JSON | 50 top market-cap entities from SEC EDGAR exchange directory | `06fd1ce96a78eb4195f60e6ea158e5af40290d8816a06917dfe6a26401e70f94` |
| `nasdaq_listed_sample.txt` | Pipe-delimited | 40-row head + 3-row tail from `nasdaqlisted.txt` (alphabetical A...) | `02866c0ab25e1d6318565d4717d205d16f94d2bcde2c2fa73b6c2b0b7ec6eec9` |
| `other_listed_sample.txt` | Pipe-delimited | 40-row head + 3-row tail from `otherlisted.txt` (alphabetical A...) | `bedca81527c7093fafc53adbb83389a59cc8fee9d579a47c02f03d5190ba28df` |
| `sample_provenance_manifest.json` | JSON | Machine-readable provenance index linking raw feeds to sample digests | Stored in scratch |

### 2.2 Targeted Public Evidence Probes (Field Gap Investigation)
Stored in `<appDataDir>/brain/<conversation-id>/scratch/vcp_universe_qualification/field_gap_probes/`:

| Probe Identifier | Target & Research Question | Endpoint URL | Retrieved (UTC) | Status / Bytes | Raw SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PROBE_SEC_SUBMISSIONS_MULTICLASS`** | Alphabet Inc. (CIK 0001652044) multi-share class CIK test (`GOOGL`, `GOOG`, `GOOGM`, `GOOGN`) | `https://data.sec.gov/submissions/CIK0001652044.json` | 2026-10-16T10:04:08Z | 200 OK / 152,401 | `648cea84d824309ba96f4c7750c4dbcefb4208a3d4cb80fb359c19bcae72d24b` |
| **`PROBE_OPENFIGI_DISAMBIGUATION`** | Disambiguation benchmark: Multi-class (`GOOG`/`GOOGL`), Common Stock (`AAPL`), ETP (`SPY`), REIT (`O`) | `https://api.openfigi.com/v3/mapping` | 2026-10-16T10:04:09Z | 200 OK / 1,328 | `d197d627eb720a2d52528148e6462744ce586ec44f2fb1b86e0c65507f3fe1d4` |
| **`PROBE_NASDAQ_TRADE_HALTS`** | Real-time US equity regulatory trading halts & intraday suspension events | `http://www.nasdaqtrader.com/rss.aspx?feed=tradehalts` | 2026-10-16T10:04:10Z | 200 OK / 17,456 | `0406e8fa111a0d708fc23fefc6655c6352934ff24ec6a536ff48a66ee5c6d36e` |

---

## 3. Sample Overlap Findings & Truncation Limitations

- **SEC & Nasdaq Directory Sample Overlap:** Exactly 1 ticker (`AAPL`).
- **SEC & Other Directory Sample Overlap:** Exactly 1 ticker (`ABBV`).
- **Unmatched SEC Tickers in Bounded Sample Slice:** 48 of 50 (96.0% unmatched in this specific slice).
- **Explanation:** The SEC scratch sample contains mega-cap issues, while the directory samples were extracted alphabetically from the start of the files (AAAP, AACG, A, AA...).
- **Methodological Boundary:** This 96% unmatched rate is a property of the **truncated sample slice**, not a measure of full-market directory join coverage.

---

## 4. Offline Research Model Execution Telemetry

Execution telemetry of `validate_inputs_offline.py` (Illustrative Research Model):

```text
============================================================================
ARX TERMINAL -- ILLUSTRATIVE OFFLINE UNIVERSE INPUT RESEARCH MODEL
Governance Status: ILLUSTRATIVE RESEARCH MODEL -- NOT AN AUTHORITATIVE CONTRACT TEST
Contract Reference: 6c9b5b00de26545239616ac46f78b35a343ee3dd
Worktree: finance-vcp-universe-inputs
Sample Directory: C:\Users\akara\.gemini\antigravity\brain\1f9bcc84-ac95-4b1d-baf9-b4d4d76f13c7\scratch\vcp_universe_qualification
============================================================================
--- PART 1: Synthetic Rule Logic Tests (Unit Fixtures) ---
  [PASS] Fixture NVDA (Synthetic): ELIGIBLE (PASSED all rules R01-R07)
  [PASS] Fixture SPY (Synthetic): INELIGIBLE under R02 (INELIGIBLE_SECURITY_TYPE)
  [PASS] Fixture TCEHY (Synthetic): INELIGIBLE under R04 (INELIGIBLE_EXCHANGE)
  [PASS] Fixture CPRX (Synthetic): INELIGIBLE under R07 (INELIGIBLE_DELISTED)
  [PASS] Fixture Missing ID (Synthetic): UNRESOLVED (UNRESOLVED_CRITICAL_IDENTITY)

--- PART 2: Actual Source Sample Audit & Overlap Analysis ---
Loaded scratch sample records:
  SEC Sample Entities:           50
  Nasdaq Listed Sample Rows:     43
  Other Listed Sample Rows:      43

Sample Overlap Findings (Truncated Sample Slice):
  SEC & Nasdaq Directory Overlap:   1 (['AAPL'])
  SEC & Other Directory Overlap:    1 (['ABBV'])
  Total Matched SEC Tickers:        2 of 50
  Unmatched SEC Tickers:            48 of 50 (96.0% unmatched in this slice)
  SAMPLE SLICE NOTE: The SEC sample contains mega-cap issues, while directory samples were
                     extracted alphabetically from the top of files (A...). This 96% unmatched
                     figure is a property of the truncated sample slice, not full market coverage.

Issuer CIK Cardinality Audit:
  Total Distinct CIKs in SEC Sample: 50
  CIKs with Multiple Listed Tickers: 0
  PROVENANCE NOTE: In full EDGAR master, multiple share classes (e.g. GOOG & GOOGL for CIK 0001652044)
                   share an issuer CIK. CIK identifies an issuer, not a unique security line.

Source-Audit Certification Conclusion:
  Total Evaluated Records:             50
  Records Certified as Fully Verified: 0
  Records That Could Not Be Certified: 50 (100% of sample)
  Missing Primary Source Fields:
    - security_type_absent     : 50 records missing empirical proof
    - currency_absent          : 50 records missing empirical proof
    - primary_listing_absent   : 50 records missing empirical proof

  GOVERNANCE FINDING:
  - Unknown evidence must NOT be mapped to an authoritative 'INELIGIBLE' contract decision.
  - Rather, these 50 records COULD NOT BE CERTIFIED because primary feeds lack explicit
    security_type, currency, and primary_listing fields.
  - Resolution does not require a commercial vendor: open standards (e.g. OpenFIGI) or approved
    normalization rules may resolve individual gaps upon Product Owner review.

--- PART 3: Simulated Readiness & Session-Guard Arithmetic ---
1. Source Population Accounting Closure:
   Source Count (S):           100
   Eligible (E):               80
   Ineligible (I):             18
   Unresolved (U):             2
   Closure Verified:           100 == 80 + 18 + 2

2. Universe Build Readiness Partition Closure:
   Eligible Population (E):    80
   Data Ready (DR):            72
   Data Unavailable (DU):      8
   Readiness Closure Verified: 80 == 72 + 8 + 0
   Scannable Universe (|DR|):  72

3. Session Guard Execution Coverage Check (session_guard.py#L602-L689):
   Population Checked:         Scannable Membership (|DR| = 72)
   Denominator:                72
   Exclusions:                 0
   Required Threshold:         100.0% floor
   Scenario A (Full Readiness): 72/72 ready -> coverage=100.0% -> Eligible=True
   Scenario B (1 Stale Symbol): 71/72 ready -> coverage=98.6% -> Eligible=False (REJECTED: INCOMPLETE_UNIVERSE_COVERAGE)

   RECONCILIATION SUMMARY:
   - At build time, E is partitioned into DR and DU. DU symbols are excluded from scannable universe.
   - At execution time, session_guard evaluates DR only, requiring available_count == |DR| (100% floor).
   - There is no conflict: DU is not checked at runtime; DR must have zero missing symbols at runtime.

--- Side-Effect & Policy Invariance Check ---
  Production SQLite writes attempted: 0
  Local SQLite writes attempted:      0
  External network requests made:     0
  Scanner dispatch triggers fired:    0
  Status: PURE IN-MEMORY VERIFICATION COMPLETED SUCCESSFULLY
============================================================================
```

---

## 5. Worktree Git Provenance & Isolation

| Repository Entity | Path / Identifier | Commit SHA / Reference |
| :--- | :--- | :--- |
| **Base Mainline Commit** | `C:\Users\akara\Documents\Projects\finance-vcp-universe-inputs` | `787c1977b0071f2b31c4ad3d36a7f872147ad59e` (PR #4) |
| **Base Tree SHA** | `finance-vcp-universe-inputs` | `809c260d6f8ee53ed27b13183df44c79adefa2fd` |
| **Worktree A Reference** | `fix/vcp-universe-wiring-reintroduction` | `6c9b5b00de26545239616ac46f78b35a343ee3dd` (Tree: `3ef8ead46c34754724cfdc5e906af66271fa2ed2`) |
| **Production Reference (PR #5)** | `origin/main` / PR #5 | `2eb032b2aec18ac4fea0f654eee8fb6c028466fe` (Zero diff on `analyst_dashboard/universe/` and `analyst_dashboard/vcp/session_guard.py`) |
| **Worktree B Reference** | `research/data-gov-discovery` | `0d7cd818da352c0c81c0126f06f4f362d613aec1` |
| **Worktree C (Active)** | `research/vcp-production-universe-inputs` | Active Research Worktree |

---

## 6. Implementation Package Engineering Specifications

The implementation package comprises four formal engineering specifications authored in `docs/research/vcp-production-universe-inputs/`:

| Document | Purpose & Scope | Governance Status |
| :--- | :--- | :--- |
| [`POLICY_DECISION_PROPOSALS.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/POLICY_DECISION_PROPOSALS.md) | Structured policy proposals for Primary Listing (R06 - UNRESOLVED), Currency (R05), Identity (R00), and Classification (R02). | `UNRESOLVED / PROPOSED_FOR_PO_REVIEW` |
| [`INPUT_VALIDATION_SPEC.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/INPUT_VALIDATION_SPEC.md) | Fail-closed input validation boundary, F01–F12 field specifications, temporal evidence requirements, default-filling vulnerability mitigation. | Complete Specification |
| [`PUBLICATION_TOOLING_SPEC.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/PUBLICATION_TOOLING_SPEC.md) | CLI publication tool architecture (`arx_publish_universe.py`), verified `UniverseStore` capabilities, triggers, forward-only revocation, read-only handoff. | Complete Specification |
| [`IMPLEMENTATION_PLAN.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/IMPLEMENTATION_PLAN.md) | Gap-to-implementation mapping, 6-task roadmap, 12 expanded acceptance scenarios, partitioned effort estimate (14–20 engineering hours decoupled from PO calendar and vendor onboarding). | Ready for PO Decision |


# ARX Terminal — SEC PDS Daily EOD Closure Contract (Epoch 001)

**Document Identifier:** `SEC_PDS_DAILY_EOD_CLOSURE_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. PDS Source-Day Identity & Scope

1. **Source Business Day Definition**:
   - `PDS_BUSINESS_DAY_ID`: Canonical calendar day `YYYY-MM-DD` in `America/New_York`.
   - `DISSEMINATION_WINDOW`: 06:00:00 Eastern Time to ~22:00:00 Eastern Time.
   - Day boundaries strictly follow the SEC EDGAR dissemination schedule. UTC timestamps and daylight saving transitions are ARX representation concerns only.

2. **Permitted Authority Claim**:
   If and only if the exact-set closure theorem passes, the authorized claim is:
   - `POPULATION_AUTHORITY = AUTHORITATIVE_CLOSED_INTERVAL`
   - `INTERVAL_CLOSURE_AUTHORITY = SOURCE_NATIVE_TERMINAL_MARKER`
   - `AUTHORITY_SCOPE = PDS_DISSEMINATION_POPULATION_FOR_ONE_EDGAR_BUSINESS_DAY`

3. **Mandatory Non-Claims**:
   - `REAL_WORLD_EVENT_COMPLETENESS = NO` (Filing omissions prior to SEC acceptance cannot be ruled out).
   - `HISTORICAL_FINALITY = NO` (SEC may process retroactive PAC deletions or amendments on future days).
   - `EARLIEST_PUBLIC_AVAILABILITY_EXACT = NO` (PDS delivery time does not establish public web publication time).
   - `MARKET_WIDE_SMART_MONEY_COVERAGE = NO` (Applies only to the governed universe).

---

## 2. Exact-Set Reconciliation Theorem

Let $D$ be an official EDGAR business day.
1. **$E(D)$ (EOD Expected Set)**: The lossless set of filenames parsed from `dissem2_MM-DD-YYYY.EOD`:
   $$E(D) = \{ f \mid f \in \text{parse}(\text{raw\_EOD}(D)) \}$$
   Diagnostics required:
   - `EOD_DUPLICATE_FILENAME_COUNT = 0`
   - `EOD_MALFORMED_ENTRY_COUNT = 0`
   - `EOD_UNKNOWN_ENTRY_TYPE_COUNT = 0`

2. **$A(D)$ (PDS Accounted Set)**: The set of filenames for which ARX possesses a valid raw payload from live PDS or qualified recovery:
   $$A(D) = \{ a.\text{filename} \mid a \in \text{accounted\_artifacts}, a.\text{source\_day} = D \}$$

3. **Reconciliation Conditions**:
   - $E(D) \setminus A(D) = \emptyset$ (`MISSING_PDS_FILES = 0`)
   - $A(D) \setminus E(D) = \emptyset$ (`UNEXPECTED_PDS_FILES = 0`)
   - $\text{hash}(E(D)) == \text{hash}(A(D))$
   - $\text{SAME\_FILENAME\_DIFFERENT\_BYTES} = 0$
   - $\text{UNCLASSIFIED\_CROSS\_DAY\_RETRANSMISSIONS} = 0$

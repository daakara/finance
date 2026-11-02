# ARX Terminal — SM-DATA-GOV-001 Policy
# Smart Money Evidence Lifecycle & Historical Integrity Governance

**Document ID:** `SM-DATA-GOV-001-POL`  
**Version:** 1.0  
**Effective Date:** 2026-11-02  
**Status:** Approved & Enforced  
**Workstream:** `SM-DATA-001`  
**Scope:** SEC Form 4 and Form 4/A Evidence Lifecycle, Historical Replay, Point-in-Time Integrity, and Backup Custody

---

## 1. Governing Principle

ARX shall preserve sufficient evidence to reproduce historical Smart Money assessments while preventing stale, corrected, incomplete, or retrospectively discovered information from being misrepresented as contemporaneous investment evidence.

### Core Invariants:
1. **Decoupled Governance:** Retention, analytical eligibility, source freshness, and legal recordkeeping are independently governed.
2. **Deterministic Reproducibility:** Historical Smart Money scores and analytical confluence inputs for any historical timestamp $T$ must be deterministically reproducible from canonical evidence.
3. **No Silent Re-scoring:** No retention policy, archival offload, or schema migration shall silently alter production Smart Money scores or confluence weights.

---

## 2. Evidence Classification and Retention

All Smart Money evidence is categorized into three strict lifecycle tiers:

| Class | Evidence Types | Retention Policy | Storage & Protection |
|---|---|---|---|
| **Class A: Canonical Filing Evidence** | Raw SEC Form 4/4/A XML, cryptographic SHA-256 hashes, source URLs, acceptance timestamps, normalized transactions | Preserve Indefinitely | PostgreSQL `form4_raw_filings` & `form4_parsed_transactions`. Protected by database trigger `abort_immutable_record()` (append-only) and fail-closed deletion prohibition (`PermissionError: CANONICAL_DELETION_PROHIBITED`). |
| **Class B: Derived Analytical Features** | Directional convictions, buy/sell ratios, aggregated volume metrics, confluence inputs | Recreate deterministically or cache with parser version tag | Ephemeral / reconstructible from Class A canonical evidence. Cache entries tagged with `parser_version`. |
| **Class C: Audit Trail & Proof Receipts** | Pre/post-ingestion backup receipts, restore drill receipts, advisory lock logs, ingestion run receipts | Preserve Indefinitely | Cryptographic JSON receipts in `.secrets/` and `sm_audit_events`. Append-only with SHA-256 manifests. |

### Destruction Protection Protocol
Destructive deletion of canonical Smart Money evidence is strictly prohibited. Because no trusted legal retention or cryptographic deletion authority framework is currently configured in the repository, all deletion requests fail closed unconditionally with `PermissionError: CANONICAL_DELETION_PROHIBITED`. Neither null tokens, arbitrary strings, nor prefix-formatted tokens can authorize deletion. Table `form4_raw_filings` is permanently append-only via database trigger `abort_immutable_record()`.

---

## 3. Point-in-Time Integrity & Replay

To eliminate future lookahead bias and ensure rigorous quantitative backtesting, ARX supports two explicit point-in-time historical replay modes:

### Replay Modes:
1. **`PUBLIC_AVAILABLE` (Dissemination Truth):**
   - Cutoff filter: `av.public_availability_at <= :as_of_watermark`
   - Represents information that was officially accessible to market participants at watermark $W$.
2. **`ARX_OBSERVED` (Contemporaneous Knowledge Truth):**
   - Cutoff filter:
     * PostgreSQL: `COALESCE(av.recorded_at, f.created_at) <= :as_of_watermark`
     * SQLite: `created_at IS NOT NULL AND created_at <= :as_of_watermark` (never substitutes `acceptance_datetime`)
   - Represents evidence that ARX's automated infrastructure had actually ingested, validated, and recorded by watermark $W$.
   - Eliminates lookahead bias where an SEC filing was published after-hours (e.g., Friday 21:00 UTC) but only ingested by ARX on Monday 09:00 UTC. Replaying at Saturday noon under `ARX_OBSERVED` strictly excludes the filing.

### Timestamp Integrity:
Missing, ambiguous, or unverified acceptance timestamps shall NEVER be fabricated, assumed, or defaulted to midnight UTC. Such records are explicitly marked with `availability_uncertain: True` on the analytical observation.

---

## 4. Freshness, Coverage, and Abstention Governance

Smart Money analytical endpoints and feed payloads must maintain absolute epistemic honesty:

### Mandatory Telemetry Indicators:
All Form 4 feed responses must expose:
- `latest_filing_date`: Date of the most recent filed Form 4.
- `latest_successful_sec_source_check`: Timestamp of the most recent successful SEC source check.
- `latest_successful_ingestion`: Timestamp of the most recent successful local database ingestion.
- `coverage_status`: Formal issuer status (`SUPPORTED_SEC_ISSUER`, `UNVERIFIED_ISSUER`, `FOREIGN_FPI_EXEMPT`, `EXEMPT_ASSET`).
- `evidence_status`: Evidence state (`VERIFIED`, `NO_OBSERVATIONS`, `PIPELINE_PENDING`, `INAPPLICABLE`).
- `unresolved_failures_count`: Integer count of active unhandled parser or fetch failures.
- `replay_mode`: Active replay mode (`PUBLIC_AVAILABLE` or `ARX_OBSERVED`).
- `as_of_watermark`: Active cutoff watermark.

### Prohibition on Misrepresenting Incomplete Coverage:
When an issuer is supported but continuous ingestion is offline or paused under governed operational hold (`HOLD_SEC_001`), the system SHALL NOT emit `"0 insider transactions filed in trailing 90 days"` or report neutral buying conviction. It must emit `PIPELINE_PENDING` / `PIPELINE_HELD`, `direction: UNVERIFIED`, and explanation referencing `HOLD_SEC_001`.

---

## 5. Amendment, Supersession, and Restatement Integrity

1. **Coexistence:** Original filings (Form 4) and amendments (Form 4/A) must coexist in `form4_raw_filings`. Neither is overwritten or purged.
2. **Linkage:** All supersession relationships are tracked in `sm_amendments` (`amendment_accession`, `base_accession`, `resolution_status`, `amendment_number`).
3. **Transaction-Level Reconciliation & Point-in-Time Supersession (SM-DATA-GOV-001.3):**
   - **No Future Leakage:** An amendment only supersedes an original filing or transaction if the amendment itself was available/observed at or before the evaluation watermark $W$. A future amendment (e.g. filed Monday) MUST NOT suppress an original filing on an earlier historical evaluation date (e.g. Saturday replay).
   - **Replay Mode Consistency:**
     - In `PUBLIC_AVAILABLE` replay mode, amendments supersede only if `am_av.public_availability_at <= W`.
     - In `ARX_OBSERVED` replay mode, amendments supersede only if `COALESCE(am_av.recorded_at, am_f.created_at) <= W`.
   - **Partial Corrections:** Partial amendments (`PARTIAL_CORRECTION`) supersede only matching line items (by `reporting_owner_name`, `transaction_date`, `transaction_code`). Un-amended line items in the original filing remain active.
   - **Amendment Chains:** In multi-stage amendment chains (e.g. Amendment 1 followed by Amendment 2), an earlier amendment is only superseded by a subsequent amendment if that subsequent amendment was available at or before $W$.
   - **Ambiguous Amendments:** Unresolved or ambiguous amendments fail closed and are excluded from positive buying conviction until adjudicated.

---

## 6. Parser Versioning & Reproducibility

1. **Deterministic Parsing:** Given raw filing XML, the Form 4 parser (`Form4XmlParser`) produces identical normalized transaction records deterministically.
2. **Schema V2 Invariants:** Every transaction record must contain the 5 canonical classification fields:
   - `is_derivative` (BOOLEAN)
   - `signal_type` (SmartMoneySignalType)
   - `acquired_disposed` ('A' or 'D')
   - `is_positive_buying_conviction` (BOOLEAN)
   - `is_open_market_purchase_evidence` (BOOLEAN)
3. **Parser Version Migration:** Any parser modification requires incrementing the parser version identifier and validating full backward reproducibility against the regression corpus.

---

## 7. Recovery, Backup, and Storage Governance

1. **Automated Backups:** Full PostgreSQL backups (`pg_dump`) must precede schema migrations and ingestion cycles.
2. **Cryptographic Verification:** Every backup produces a tamper-evident receipt recording dump SHA-256, manifest SHA-256, byte count, and execution timestamp.
3. **Recovery Verification:** Backups are regularly validated via automated restore rehearsals to isolated disposable databases (`RPO < 1 hour`, `RTO < 15 minutes`).
4. **Storage Projections:** With current ingestion footprint (~5.2 KB per filing, ~1 MB/year for 10-issuer core universe), direct PostgreSQL storage is optimal. Archival offloading or table partitioning is deferred until database size exceeds 50 GB.

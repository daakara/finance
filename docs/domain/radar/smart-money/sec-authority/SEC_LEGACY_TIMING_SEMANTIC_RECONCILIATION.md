# ARX Terminal — Smart Money Legacy Timing Semantic Reconciliation (Epoch 001)

**Document Identifier:** `SEC_LEGACY_TIMING_SEMANTIC_RECONCILIATION`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Qualification Engineer  

---

## 1. Audit of Historical `public_availability_at` References

An audit of the repository establishes that existing Smart Money code paths populated `public_availability_at` from EDGAR `<acceptanceDateTime>`:
- `analyst_dashboard/smart_money/form4_models.py` lines 68-73:
  `self.public_availability_at = self.acceptance_datetime`
- `analyst_dashboard/smart_money/policy_bundle.py` line 20:
  `"KNOWLEDGE_TIME: Exact public_availability_at derived from acceptanceDateTime."`
- `analyst_dashboard/smart_money/form4_parser.py` line 284
- `analyst_dashboard/smart_money/form4_ingestion.py` line 813
- `analyst_dashboard/smart_money/incremental_engine.py` line 143

---

## 2. Epistemic Reconciliation Findings

1. **Source of Truth**:
   The value stored in `public_availability_at` is the SEC's internal system acceptance datetime.

2. **Semantic Reclassification**:
   - `LEGACY_FIELD`: `public_availability_at`
   - `LEGACY_VALUE_SOURCE`: `SEC_ACCEPTANCE_DATETIME`
   - `RECONCILED_SEMANTIC`: `sec_acceptance_at`
   - `HISTORICAL_FIRST_PUBLIC_AVAILABLE_AT`: `UNKNOWN_UNLESS_SEPARATELY_OBSERVED`

3. **Non-Destructive Preservation Policy**:
   - **`LEGACY_ROWS_MUTATED = NO`**: Historical raw SQLite evidence rows (`sm_source_availability_ledger`, `form4_raw_filings`) are **not** rewritten.
   - Point-in-time quantitative research must treat legacy timestamps as an acceptance lower bound ($	au_{\text{public}} \ge \tau_{\text{accept}}$), not as verified instant public knowledge time.

# ARX Terminal — Legacy PIT Consumer Audit & Remediation (Epoch 001)

**Document Identifier:** `LEGACY_PUBLIC_AVAILABILITY_CONSUMER_AUDIT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent Historical PIT Reconciliation Engineer  
**Operating Mode:** `RECONSTRUCT_ACTUAL_STATE` + `NO_RETROACTIVE_KNOWLEDGE` + `APPEND_ONLY`

---

## 1. Audit Scope and Identified Code Paths

An exhaustive audit of all repository code consuming `public_availability_at` was conducted across `analyst_dashboard/smart_money/**`, `tests/**`, and data engines:

| Code Path | Symbol / Function | Legacy Field | Actual Underlying Semantic | Consumer Classification | Remediation Status |
|---|---|---|---|---|---|
| `analyst_dashboard/smart_money/form4_models.py` | `SmartMoneyTransaction.__post_init__` | `public_availability_at` | `SEC_ACCEPTANCE_DATETIME` | `PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT` | REMEDIATED |
| `analyst_dashboard/smart_money/form4_parser.py` | `Form4XmlParser.parse_filing` | `public_availability_at` | `SEC_ACCEPTANCE_DATETIME` | `ACCEPTANCE_SEMANTICS_CORRECT` | REMEDIATED |
| `analyst_dashboard/smart_money/form4_ingestion.py` | `query_smart_money_transactions_as_of` | `l.public_availability_at` | `SEC_ACCEPTANCE_DATETIME` | `PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT` | REMEDIATED |
| `analyst_dashboard/smart_money/incremental_engine.py` | `compute_source_max_timestamp` / `check_late_arrival` | `sm_source_availability_ledger.public_availability_at` | `SEC_ACCEPTANCE_DATETIME` | `ACCEPTANCE_SEMANTICS_CORRECT` | REMEDIATED |
| `analyst_dashboard/smart_money/policy_bundle.py` | `PolicyBundleAuthority.POLICIES` | `KNOWLEDGE_TIME text definition` | `SEC_ACCEPTANCE_DATETIME` | `PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT` | REMEDIATED |
| `analyst_dashboard/smart_money/certification_engine.py` | `SmartMoneyCertificationEngine._check_source_gap_integrity` | `sm_source_availability_ledger.public_availability_at` | `SEC_ACCEPTANCE_DATETIME` | `ACCEPTANCE_SEMANTICS_CORRECT` | REMEDIATED |
| `analyst_dashboard/smart_money/form4_scanner.py` | `evaluate_issuer_transactions` | `transaction_date` (not `public_availability_at`) | `FORM4_TRANSACTION_DATE` | `ACCEPTANCE_SEMANTICS_CORRECT` | REMEDIATED |

---

## 2. Classification Summary

- **Total Consumers Audited:** 7
- **`ACCEPTANCE_SEMANTICS_CORRECT`:** 4
- **`PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT`:** 3
- **`AMBIGUOUS`:** 0
- **`DEAD_CODE`:** 0

---

## 3. Epistemic Defect Identification & Backtest Safety

### Defect Finding
Historical code paths in `form4_models.py` and `policy_bundle.py` equated EDGAR `<acceptanceDateTime>` with `public_availability_at` and referred to it as "exact knowledge time". In reality:
$$\tau_{\text{acceptance}} < \tau_{\text{public\_web\_dissemination}}$$

### Governed Invariants
1. **`LEGACY_ACCEPTANCE_USED_AS_EXACT_PUBLIC_TIME = YES`** (in legacy prose and default property initialization).
2. **`HISTORICAL_PIT_CONTAMINATION_RISK = YES`**: Historical backtests evaluating signals at $\tau_{\text{acceptance}}$ without adding a conservative dissemination delay suffer from look-ahead contamination.
3. **`HISTORICAL_STRICT_PIT_EVALUATION = NOT_AUTHORIZED`**: No historical Smart Money backtest may claim strict certified point-in-time public knowledge timing unless evaluated against prospective public observation evidence or under a governed conservative lag policy.
4. **`LEGACY_ROWS_MUTATED = NO`**: Historical raw SQLite evidence rows (`sm_source_availability_ledger`, `form4_raw_filings`) are **not** rewritten or modified.
5. **`HISTORICAL_EXACT_FIRST_PUBLIC_TIME = UNKNOWN_UNLESS_SEPARATELY_OBSERVED`**: For historical filings, public observation fields remain `UNKNOWN` rather than being fabricated from acceptance timestamps.

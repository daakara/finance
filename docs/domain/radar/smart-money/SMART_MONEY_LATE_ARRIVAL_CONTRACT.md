# ARX TERMINAL — SMART MONEY WAVE 2C
## LATE-ARRIVAL FORENSICS & WATERMARK BREACH CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-LATE-ARRIVAL-001
VERSION: 2.1.0
STATUS: RATIFIED
GOVERNANCE: FAIL_CLOSED
BREACH_RULE: WATERMARK_INTEGRITY_BREACH
```

---

### 1. Invariant Definition

A filing is defined as **late-arriving** if its SEC public availability timestamp precedes the current processing epoch:

$$\text{public\_availability\_at} < T_{\text{ingest}}$$

---

### 2. Event Classification

Two mutually exclusive categories are enforced:

1. **`LATE_WITHIN_UNCERTIFIED_INTERVAL`**:
   - Condition: $W_{\text{certified}} < \text{public\_availability\_at} < T_{\text{ingest}}$
   - Impact: Ingested normally into historical cache and ledger. Does not violate certified completeness because that interval was never certified.
2. **`LATE_BEYOND_CERTIFIED_WATERMARK` (INTEGRITY BREACH)**:
   - Condition: $\text{public\_availability\_at} \le W_{\text{certified}}$
   - Classification: `WATERMARK_INTEGRITY_BREACH`
   - Mandatory System Actions:
     1. Durable record appended to `sm_late_arrival_events` with `is_integrity_breach = 1`.
     2. `has_active_integrity_breach()` evaluates to `True`.
     3. Global certified watermark advancement is **FROZEN** (all subsequent `certify_watermark_advancement` calls raise `RuntimeError`).
     4. Downstream consumer readiness transitions immediately to `DATA_INCOMPLETE`.
     5. Full audit alert triggered for partition gap forensic review.

---

### 3. Breach Table Schema (`sm_late_arrival_events`)

- `integrity_event_id`: UUID
- `detected_at`: UTC timestamp of detection
- `accession_number`: SEC filing accession
- `issuer_cik`: Issuer CIK
- `public_availability_at`: Exact SEC acceptance datetime
- `affected_watermark_receipt`: ID of the watermark receipt breached
- `affected_interval`: ISO8601 interval between `public_availability_at` and `W_certified`
- `event_type`: `LATE_BEYOND_CERTIFIED_WATERMARK`
- `is_integrity_breach`: `1` (Boolean True)

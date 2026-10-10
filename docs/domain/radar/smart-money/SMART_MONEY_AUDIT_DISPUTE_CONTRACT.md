# ARX Terminal — Smart Money Audit Dispute & SLA Governance Contract

**Contract ID:** `SM_AUDIT_DISPUTE_V1`  
**Engine Implementation:** `analyst_dashboard/smart_money/audit_dispute_engine.py`  
**Status:** `FROZEN_LOCAL_ENGINEERING_AUTHORITY`

---

## 1. Governance Architecture & Purpose

When independent auditors, verification algorithms, or forensic audits diverge from an issued certification receipt, the system enters an **Audit Dispute**.

Dispute governance enforces two ironclad principles:
1. **The SLA Certification Firewall**: Process timers, deadlines, and SLA breaches exist solely for engineering and organizational accountability. They have strictly **ZERO** mathematical or certification authority. A timeout can never advance a watermark, seal a receipt, or paper over a proof gap.
2. **Disputed Receipt Non-Reuse Invariant**: Once a receipt is disputed, it is permanently tainted. Post-resolution advancement requires an entirely fresh candidate cycle, independent reconstruction, and a new `CertificationReceiptV2` sequence number.

---

## 2. Four Orthogonal State Machines

Disputes are tracked along four independent state vectors:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DisputeState (Lifecycle)                                 │
│ DETECTED -> TRIAGE -> ROOT_CAUSE -> RECONCILIATION ->       │
│ REAUDIT -> GOVERNANCE_ESCALATED -> GOVERNANCE_HOLD ->       │
│ RESOLVED                                                    │
├─────────────────────────────────────────────────────────────┤
│ 2. SlaState (Process Deadlines)                             │
│ WITHIN_TRIAGE_SLA -> TRIAGE_OVERDUE (4h) ->                 │
│ ROOT_CAUSE_OVERDUE (24h) -> RECONCILIATION_OVERDUE (48h) -> │
│ GOVERNANCE_ESCALATED -> MAX_SLA_EXCEEDED (120h)             │
├─────────────────────────────────────────────────────────────┤
│ 3. EscalationLevel (Authority Tier)                         │
│ L0_ENGINEERING -> L1_TECHNICAL_ADJUDICATION ->              │
│ L2_GOVERNANCE_CLOSURE -> L3_GOVERNANCE_HOLD                 │
├─────────────────────────────────────────────────────────────┤
│ 4. CertificationDisputeState (Completeness Impact)          │
│ LAST_UNDISPUTED_ACTIVE (Candidate Blocked)                  │
│ PRIOR_CERTIFICATION_UNDER_INTEGRITY_REVIEW                 │
│ PRIOR_CERTIFICATION_INVALIDATED                            │
│ POST_RESOLUTION_CERTIFICATION_PENDING                       │
│ NEW_RECEIPT_CERTIFIED                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. SLA Deadlines & Escalation Tiers

Dispute resolution deadlines are measured relative to `detected_at`:

| Phase | Duration | SLA State on Breach | Escalation Level |
|---|---|---|---|
| **Triage** | 4 hours | `TRIAGE_OVERDUE` | `L1_TECHNICAL_ADJUDICATION` |
| **Root Cause** | 24 hours | `ROOT_CAUSE_OVERDUE` | `L1_TECHNICAL_ADJUDICATION` |
| **Reconciliation** | 48 hours | `RECONCILIATION_OVERDUE` | `L2_GOVERNANCE_CLOSURE` |
| **Max Threshold** | 120 hours | `MAX_SLA_EXCEEDED` | `L3_GOVERNANCE_HOLD` |

### Escalation Hierarchy
- **L0 (Engineering)**: Initial triage, log inspection, and trace comparison.
- **L1 (Technical Adjudication)**: Senior architecture review of parser discrepancies or timing edge cases.
- **L2 (Governance Closure)**: Formal arbitration ruling and decision on code remediation.
- **L3 (Governance Hold)**: Complete freeze on pipeline advancement until governance board signs off.

---

## 4. Minimal Counterexample Recording

Every opened dispute must register a minimal counterexample:
```json
{
  "first_divergent_step": "Step 12: Accession reconciliation mismatch",
  "minimal_counterexample": {
    "issuer_cik": "0000320193",
    "accession_number": "0000320193-26-000123",
    "expected_present": true,
    "registered_present": false,
    "delta": 1
  }
}
```
All dispute state changes are appended to `sm_audit_dispute_history` with sequence IDs and timestamps.

# TRANSACTION LOG CONSENSUS COHERENCE CONTRACT (Theorem C / FND-ARCH-014)

## 1. AUTHORITY MODEL & LEDGER PRIMACY

This contract establishes the formal relationship between the authoritative write-ahead transaction log (`transaction_log`) and consensus checkpoints (`authoritative_epoch` / consensus states):

1. **Transaction Log as Authoritative History**:
   - The Write-Ahead Log (`transaction_log`) is the sole primary, immutable source of truth for all state-transition events.
   - Every transaction sequence number (`log_sequence`) is strictly monotonic and linked via cryptographic entry hashes:
     $$\text{entry\_hash} = \text{SHA256}(\text{prev\_entry\_hash} : \text{canonical\_payload})$$

2. **Consensus Checkpoints as Derived Projections**:
   - Checkpoints and consensus records are derived materialized views over authoritative log intervals.
   - A consensus checkpoint asserts that all events up to sequence $K$ have been durably replicated and anchored.

## 2. MANDATORY COHERENCE INVARIANTS

For every accepted non-genesis consensus checkpoint $(K, H_{\text{checkpoint}})$:

$$\exists \, \text{row} \in \text{transaction\_log} \quad \text{such that} \quad \text{log\_sequence} = K \quad \land \quad \text{entry\_hash} = H_{\text{checkpoint}}$$

### Strict Fail-Closed Verification Rules:
1. **Exact Sequence Existence**:
   - If no row exists in `transaction_log` where `log_sequence == checkpoint_sequence`, verification MUST fail immediately with `ConsensusLogDivergenceError`.
   - Never infer future sequences or assume validity when the log has not yet reached the checkpoint sequence ($\text{max\_seq} < K$).
2. **Exact Hash Parity**:
   - If the row exists but $\text{entry\_hash} \neq H_{\text{checkpoint}}$, verification MUST fail with `ConsensusLogDivergenceError`.
3. **Zero State Mutation on Failure**:
   - When verification fails, the database transaction must execute `ROLLBACK`.
   - The authoritative latest checkpoint sequence and hash in `authoritative_epoch` MUST NOT be updated.

## 3. ADVERSARIAL MATRIX & FAILURE MODES

| Condition | Verification Verdict | State Mutation | Exception Raised |
| :--- | :--- | :--- | :--- |
| Log sequence $K$ does not exist in WAL | REJECTED | None (Rollback) | `ConsensusLogDivergenceError` |
| Empty WAL log attempting checkpoint $K \ge 1$ | REJECTED | None (Rollback) | `ConsensusLogDivergenceError` |
| Log sequence $K$ exists, but hash diverges | REJECTED | None (Rollback) | `ConsensusLogDivergenceError` |
| Attempted checkpoint sequence $\le \text{latest\_seq}$ | REJECTED | None (Rollback) | `GlobalRollbackError` |
| Exact log sequence $K$ exists and hash matches | ACCEPTED | Head Updated | None |

## 4. INVARIANTS

```yaml
UNDETECTED_LOG_CONSENSUS_DIVERGENCE: 0
MISSING_WAL_ROW_CHECKPOINT_ACCEPTED: 0
CHECKPOINT_FAILURE_STATE_MUTATION: 0
FAIL_CLOSED_VERIFICATION_ENFORCED: true
```

# ARX TERMINAL — SMART MONEY WAVE 2C
## PARTITION WATERMARKS & CERTIFIED COMPLETENESS CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-WATERMARK-001
VERSION: 2.1.0
STATUS: RATIFIED
GOVERNANCE: FAIL_CLOSED
WATERMARK_RULE: MINIMUM_PARTITION_FRONTIER
```

---

### 1. Watermark Architecture

Completeness in ARX Terminal Smart Money is governed by a strict two-stage watermark hierarchy:

```
[SEC EDGAR Submissions]
          │
          ▼
   [Partition Discovery] ─── (Per CIK 1..106)
          │
          ▼
   Candidate Complete Frontier ── (Blocked by Open Gaps in Partition)
          │
          ▼
   Atomic Global Consolidation: MIN(Partition Frontiers across all 106)
          │
          ▼
   Certified Completeness Watermark (Immutable Watermark Receipt Sealed)
```

---

### 2. Minimum Partition Frontier Invariant

$$W_{\text{global}} = \min_{p \in P} \left( F_p \right)$$

where:
- $P$ is the full set of 106 scannable universe members.
- $F_p$ is the effective certified frontier for partition $p$.
- If any single partition is delayed, blocked by a rate limit, or suffers an open gap at timestamp $T$, the entire global watermark is pinned at $\le T$.
- No partition may advance beyond its earliest blocking source gap.

---

### 3. Watermark Receipts (`sm_watermark_receipts`)

Every advancement of the certified completeness watermark must produce an immutable audit receipt containing:
- `receipt_id`: `RCPT-{cycle_id}-{timestamp}`
- `previous_certified_watermark`: Monotonically precedes new watermark.
- `new_certified_watermark`: Target verified frontier.
- `partition_frontier_hash`: SHA256 of canonical JSON array of all 106 partition frontiers.
- `open_gap_set_hash`: SHA256 of all active blocking gaps across the universe.
- `source_ledger_hash`: SHA256 of `sm_source_availability_ledger` state at certification time.
- `corpus_state_hash`: SHA256 of `form4_raw_filings` + `form4_parsed_transactions` state.
- `certified_at`: UTC timestamp of certification transaction.
- `is_sealed`: Immutable Boolean flag (`1`).

---

### 4. Non-Monotonic Rejection

Any attempt to advance the certified watermark backward or out-of-order raises a fatal `RuntimeError` and aborts the database transaction.

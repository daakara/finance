# ARX Terminal — Public Availability Clock Authority Contract (Epoch 001)

**Document Identifier:** `PUBLIC_AVAILABILITY_CLOCK_AUTHORITY_CONTRACT`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent Point-in-Time Integrity Auditor & SEC Observation Authority  
**Operating Mode:** `CLOCK_AWARE` + `FAIL_CLOSED`

---

## 1. Principles of Clock Authority

Point-in-time quantitative evidence is invalid without verifiable clock authority. Local wall-clock measurements carry uncertainty due to OS scheduling, NTP offset, and network jitter.

Every observation record binds:
- `CLOCK_SOURCE`: Underlying system time service (`OS_DISCIPLINED_TIME_SERVICE`).
- `CLOCK_SYNC_METHOD`: Synchronization mechanism (`WINDOWS_TIME_SYNC_OR_MONOTONIC_ANCHOR`).
- `CLOCK_OBSERVED_AT`: UTC timestamp of clock sampling.
- `CLOCK_OFFSET_MS`: Estimated drift against reference standard.
- `CLOCK_UNCERTAINTY_MS`: Maximum estimated uncertainty interval $\pm \Delta t$.
- `CLOCK_STATUS`: Governed classification state.

---

## 2. Clock Status Classification Matrix

| Status | Uncertainty Bound ($\Delta t$) | Ingestion Permitted | Certified Knowledge Timestamp |
|---|---|---|---|
| `SYNCHRONIZED` | $\Delta t \le 100 \text{ ms}$ | YES | YES |
| `DEGRADED` | $100 \text{ ms} < \Delta t \le 500 \text{ ms}$ | YES | YES (Annotated `DEGRADED`) |
| `UNTRUSTED` | $\Delta t > 500 \text{ ms}$ or leap/jump | YES (Presence only) | **BLOCKED** (`TIMESTAMP_AUTHORITY_BLOCKED`) |

---

## 3. Observation Time & Conservative Knowledge Derivation

For every response completed at $t_{\text{comp}}$ with uncertainty $\Delta t$:
$$\tau_{\text{earliest}} = t_{\text{comp}} - \Delta t$$
$$\tau_{\text{conservative}} = t_{\text{comp}} + \Delta t$$
$$\tau_{\text{latest}} = t_{\text{comp}} + \Delta t$$

### Hard Invariant
ARX Terminal did not possess the complete response before $t_{\text{comp}}$. Therefore:
$$\tau_{\text{knowledge}} \ge t_{\text{comp}} + \Delta t$$
Request start time ($t_{\text{start}}$) is **strictly prohibited** as a knowledge timestamp.

---

## 4. Clock-Interval Ordering Rules

When comparing two asynchronous observation events $A$ and $B$:
- If $\tau_{\text{latest}}(A) < \tau_{\text{earliest}}(B) \implies \text{A\_PRECEDES\_B}$.
- If $\tau_{\text{latest}}(B) < \tau_{\text{earliest}}(A) \implies \text{B\_PRECEDES\_A}$.
- If $[\tau_{\text{earliest}}(A), \tau_{\text{latest}}(A)]$ overlaps $[\tau_{\text{earliest}}(B), \tau_{\text{latest}}(B)] \implies \textbf{INDETERMINATE}$.

No order winner is forced when uncertainty intervals overlap. Timestamp averaging or median interpolation is strictly prohibited.

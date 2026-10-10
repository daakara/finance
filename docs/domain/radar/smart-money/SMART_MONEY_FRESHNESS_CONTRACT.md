# ARX TERMINAL — SMART MONEY WAVE 2C
## FRESHNESS AUTHORITY & READINESS SEMANTICS CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-FRESHNESS-001
VERSION: 2.1.0
STATUS: RATIFIED
GOVERNANCE: DERIVED_FROM_CERTIFIED_WATERMARK
```

---

### 1. Freshness Derivation Rule

Data freshness is strictly computed from `GLOBAL_CERTIFIED_COMPLETENESS_WATERMARK`, NOT from wall-clock time, ingestion time, or isolated filing timestamps:

$$\text{lag\_seconds} = \max \left( 0.0, T_{\text{current}} - W_{\text{certified}} \right)$$

---

### 2. Staleness Tiers

| Tier | Lag Threshold | Consumer Status | UI Presentation |
|---|---|---|---|
| `FRESH` | $\text{lag} \le 86,400\text{s}$ (24h) | `AVAILABLE` or `NO_MATCHES` | Normal display with freshness badge |
| `STALE` | $86,400\text{s} < \text{lag} \le 259,200\text{s}$ (72h) | `AVAILABLE` or `NO_MATCHES` | Warning badge: Stale data |
| `DATA_INCOMPLETE` | Active Gaps > 0 OR Active Breaches > 0 | `DATA_INCOMPLETE` | Pipeline incomplete warning slate |
| `UNAVAILABLE` | $\text{lag} > 259,200\text{s}$ (72h) OR No Watermark | `PIPELINE_PENDING` | Fail-closed neutral slate |

---

### 3. Readiness Status Precedence

1. **Active Integrity Breach**: If `active_breach_count > 0` $\rightarrow$ strictly `DATA_INCOMPLETE`.
2. **Active Source Gaps**: If `open_gap_count > 0` $\rightarrow$ strictly `DATA_INCOMPLETE`.
3. **No Watermark or Uninitialized**: $\rightarrow$ strictly `PIPELINE_PENDING`.
4. **Lag > 72h**: $\rightarrow$ `DATA_INCOMPLETE` or `PIPELINE_PENDING`.
5. **Clean Completeness**:
   - Matches found: `AVAILABLE`.
   - Zero matches found: `NO_MATCHES` (truthful zero, distinct from pipeline pending).

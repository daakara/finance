# ARX TERMINAL — SMART MONEY WAVE 2C
## SOURCE GAP LEDGER & GAP FRONTIER CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-GAP-001
VERSION: 2.1.0
STATUS: RATIFIED
GOVERNANCE: FAIL_CLOSED
RULE: GAP_FRONTIER_RULE
```

---

### 1. Gap Taxonomy (`GapType`)

Source gaps are strictly categorized into an exhaustive closed set:

| Gap Type | Description | Initial Status | Blocking Frontier |
|---|---|---|---|
| `DISCOVERY_FAILURE` | Inability to poll company submissions list | `OPEN` | `window_start` |
| `FETCH_FAILURE` | HTTP 5xx or connection error retrieving XML | `OPEN` | `source_event_time` if known, else `window_start` |
| `PARSE_FAILURE` | Malformed XML or unparseable schema | `OPEN` | `source_event_time` |
| `RATE_LIMITED` | HTTP 429 response from SEC EDGAR | `OPEN` | `window_start` |
| `TIMEOUT` | Request timed out exceeding SLA | `OPEN` | `window_start` |
| `UNKNOWN_SOURCE_WINDOW` | Unresolvable gap between polling intervals | `OPEN` | `window_start` |
| `AMENDMENT_UNRESOLVED` | Form 4/A references missing historical accession | `OPEN` | `source_event_time` |
| `SOURCE_INTEGRITY_CONFLICT` | Accession payload hash mismatch detected | `OPEN` | `source_event_time` |

---

### 2. Gap Frontier Rule

$$W_{\text{candidate}}(p) = \min \left( \text{cursor}(p), \min_{g \in G_p^{\text{active}}} \text{blocking\_frontier}(g) \right)$$

where:
- $G_p^{\text{active}}$ is the set of all gaps in partition $p$ with status `OPEN` or `RETRYING`.
- Unresolved gaps unconditionally clamp the candidate complete frontier.
- Discovery of subsequent filings at $T_2 > T_{\text{gap}}$ DOES NOT clear or skip the gap.

---

### 3. Gap Lifecycle

```
[Issue Detected]
       │
       ▼
   Gap Registered (OPEN, attempt_count = 1)
       │
       ├── Backoff & Retry ──► (RETRYING, attempt_count += 1)
       │                              │
       │                              ▼
       │                    (Retry Exhaustion >= 3)
       │                              │
       ▼                              ▼
  [Resolution]               [Persistent Blocker]
(RESOLVED, audit reason)   (Remains RETRYING/OPEN)
```

1. **Resolution Authority**: Gaps may only be marked `RESOLVED` when the source filing is successfully ingested and parsed, or explicitly discharged by an authorized audit procedure with recorded `resolution_authority` and `resolution_reason`.
2. **Terminal Exclusion**: A gap cannot be silently dropped. Any terminal exclusion requires non-repudiable audit recording.

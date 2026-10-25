# Post-Rotation Runtime Health & Stability Specification

## 1. Post-Rotation Observation Protocol

Following the secret update and revocation, the rehearsal environment is observed for runtime stability:

| Health Indicator | Target Invariant | Method of Verification | Expected State |
| :--- | :--- | :--- | :--- |
| **Worker Process State** | Single active daemon, zero crash loops | `railway deployment list` / `ps aux` | `OWNER_LOCK_ACQUIRED` |
| **Scheduler Liveness** | Background thread running | Container startup logs | `interval=21600s / 6.0h` |
| **Backup Freshness** | Verified recovery point < 12h | `evaluate_freshness()` | `is_fresh: true`, `FRESH` |
| **Authentication Errors** | Zero 401/403 R2 exceptions | `railway logs -s smart-money-worker-rehearsal` | Clean execution |
| **Primary Advisory Lock** | Held exclusively by active worker | `pg_locks` query for `401007311509093371` | 1 lock granted |
| **Backup Advisory Lock** | Released upon pass completion | `pg_locks` query for `872055449083353674` | 0 locks held |
| **Web Service Liveness** | HTTP status 200 OK | `curl http://127.0.0.1:8000/health` | `{"status":"online"}` |
| **Production Isolation** | Zero production mutations | Railway environment audit | `11cad5e1-...` unchanged |

---

## 2. Guardrails & Prohibitions

1. **Zero Fault Injection**:
   No synthetic staleness injection or gate tampering will be run against the post-rotation runtime.
2. **Zero Code Mutations**:
   Application codebase remains fixed at `1cc9fa886d28908270df4794e64536d4cb1948e2`.
3. **Rollback Trigger**:
   If the worker container enters a crash loop or repeatedly fails R2 authentication after rotation, secrets must be immediately reverted to the prior operational state while the token remains active.

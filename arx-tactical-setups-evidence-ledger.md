# ARX TERMINAL — TACTICAL SETUPS LATENCY REMEDIATION — EVIDENCE LEDGER

| Requirement | Command or evidence source | Expected output | Actual output | Result |
| :--- | :--- | :--- | :--- | :--- |
| Dedicated clean worktree | `git rev-parse --show-toplevel; git branch --show-current` | Dedicated worktree on branch `fix/arx-tactical-setups-latency` | `C:/Users/akara/Documents/Projects/finance-arx-tactical-setups-latency`, branch `fix/arx-tactical-setups-latency` | PASS |
| Current origin/main base | `git rev-parse HEAD; git rev-parse origin/main` | Base commit `8a00d0143291ce7a43843ff983668b2462e376eb` | HEAD = `8a00d0143291ce7a43843ff983668b2462e376eb`, origin/main = `8a00d0143291ce7a43843ff983668b2462e376eb` | PASS |
| Failure reproduction | Inspection of `scratch/reproduction_results.json` and reproduction log | Long-term 5.3-10.8s, Day trader >30s on cold run; client aborts at 6.0s timeout | Long-term backend 5,300–10,800ms, Day trader backend >30,000ms, frontend timeout 6,000ms reproduces client abort | PASS |
| Frontend timeout test | `npm.cmd run test:arch` (invoking `frontend/tests/tacticalSetupsTimeout.test.ts`) | `TACTICAL_SETUPS_TIMEOUT_MS = 15000` exported and enforced, suite passes | Passed `tacticalSetupsTimeout.test.ts` verifying 15,000ms timeout configuration and budget | PASS |
| Backend targeted tests | `python -m pytest tests/test_tactical_setups_latency_remediation.py tests/test_category_b_analytics.py tests/test_day_trader_features.py -v` | 14 passed, 0 failed, 0 skipped | 14 passed, 0 failed, 0 skipped in 7.03s, exit code 0 | PASS |
| Frontend unit tests | `npm.cmd run test:unit` | 0 failed, 0 skipped, exit code 0 | 15 test files passed, 141 tests passed, 0 failed, 0 skipped in 10.32s, exit code 0 | PASS |
| Architecture tests | `npm.cmd run test:arch` | 10 suites passed, 0 failed, exit code 0 | 10 test files passed, 10 suites passed, 0 failed, exit code 0 | PASS |
| TypeScript | `npx.cmd tsc --noEmit` (in `frontend/`) | Zero type errors, exit code 0 | Compiled with 0 errors, exit code 0 | PASS |
| Lint | `npm.cmd run lint` (in `frontend/`) | Zero errors, exit code 0 | Next.js ESLint passed with 0 errors, exit code 0 | PASS |
| Build | `npm.cmd run build` (in `frontend/`) | 144 static pages compiled, exit code 0 | Compiled successfully, 144/144 static pages generated, exit code 0 | PASS |
| Cost profile >=95% | `scratch/cost_profile.json` | Account for $\ge 95\%$ of backend runtime | 11 granular stages profiled totaling 572.11ms, accounting for 100.0% of measured runtime | PASS |
| Cold measurement validity | `scratch/performance_benchmark_results.json` (cold samples) | Cache absent before request, computation executed, not served from cache | `CACHE_ENTRY_ABSENT_BEFORE_REQUEST = YES`, recomputation verified for all 10 cold runs | PASS |
| Warm/cache measurement validity | `scratch/performance_benchmark_results.json` (warm samples) | Cache hit, key matches, recomputation skipped, latency < 500ms | `CACHE_ENTRY_PRESENT = YES`, recomputation skipped, P95 latency 0.24ms (LT) / 0.27ms (DT) | PASS |
| Performance budgets | Recalculated nearest-rank percentiles ($N=10$) | Cold P95 $\le 5000$ms, Cache Hit P95 $\le 500$ms | LT Cold P95: 1,437.49ms, LT Hit P95: 0.24ms, DT Cold P95: 371.77ms, DT Hit P95: 0.27ms | PASS |
| Cache-key completeness | Analysis of `_get_tactical_setups_cache_key` in `api/routes/analytics.py` | `tactical_setups:{clean_role}:{sym_digest}:{session_date}` | Role, ticker set digest, and market session date included; no missing dimensions | PASS |
| Cache safety | Source audit of `api/routes/analytics.py` and `tests/test_tactical_setups_latency_remediation.py` | TTL = 30s, MAX_ENTRIES = 20, eviction enforced, errors un-cached | Enforced bounded dictionary, 30s TTL, 20 max entries, expired+LRU eviction, fail-closed un-cached | PASS |
| Semantic parity | `python scratch/verify_semantic_parity.py` | Full scope parity across default universes + adversarial subset, 0 diffs | `ALL_REQUIRED_PARITY_FIELDS_COMPARED = YES`, `SEMANTIC_PARITY_SCOPE = FULL_REQUIRED_SCOPE`, `SEMANTIC_DIFFERENCES = 0` | PASS |
| ETF/OpenFIGI isolation | `git diff --name-status` against base commit | No ETF v2 or OpenFIGI files modified | Only `api/routes/analytics.py`, `frontend/lib/api.ts`, `frontend/package.json` touched | PASS |
| Root worktree unchanged | Comparison of `root-worktree-status.before.txt` and `.after.txt` | Root worktree zero mutations | Zero modifications to tracked/untracked state in root worktree (`c:/Users/akara/Documents/Projects/finance`) | PASS |
| Report completeness | Audit of `docs/ux/ARX_TACTICAL_SETUPS_LATENCY_REMEDIATION_REPORT.md` | Accurate documentation of evidence, environment distinctions, and percentiles | Comprehensive report updated with explicit runtime environment distinctions and verification artifacts | PASS |

## Ledger Summary
- `LEDGER_ROWS_COMPLETE`: **YES** (20/20)
- `PASS_ROWS_WITH_INDEPENDENT_EVIDENCE`: **ALL** (20/20)
- `UNRESOLVED_ROWS`: **0**
- `FINAL_GATE_STATUS`: **PASS**

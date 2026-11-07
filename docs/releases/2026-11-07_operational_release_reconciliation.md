# ARX — 2026-11-07 Production Reconciliation and Release Notes

**Documentation-only publication; no runtime change or hold release.**

## Delivered (verified GitHub / deployment identities)
| Release | Production merge SHA | Scope | Runtime |
|---|---|---|---|
| Radar PR #220 | `6f6770e0e72f5ca129feafdbee2c2e3932047191` | Dynamic ARX_US_EQUITIES only, fail-closed absence, Stage 2 subset filters, no curated UI selectors | Cloudflare canonical SUCCESS at this SHA; Railway deployed then superseded |
| Learning PR #221 | `c569e26b1f5059bd659ae68d418e03a6482a132f` | Passive prospective shadow capture, evaluation taxonomy, census | Railway deployed then superseded |
| Learning correction PR #222 | `83bca0b69d2d56d30b4d3157ff492fae260b7f3a` | Explicit census unavailability, authenticated access, safe dry-run settlement, auditable outcome finality | Railway web + worker SUCCESS on same SHA |

Reported acceptance evidence for Radar: 250 published members, 100 Stage 2, 0 VCP; 62 High R:R; 27 In Buy Zone; all default candidates non-actionable. Counts are for Nov 6 source sessions and are not guarantees of real-time recomputation.

## Not delivered / no release claims
- No automatic VCP natural scanning, recurring SEC egress, automatic settlement, model weight changes, Sprint 5 execution, or live trading.
- No verified clean prospective resolved sample. Historical global / AAPL counts are *unverified historical recommendations*, not calibration evidence.
- Cloudflare and Railway SHA mismatch after PR #222 is expected for backend-only changes; do not trigger redundant frontend release.
- PR #223 reproduced the exact file blobs merged by PR #222 and was closed unmerged with an explanatory comment.

## Follow-up
Canonical active backlog and implementation priority: [ARX active work register](../planning/ARX_ACTIVE_WORK_REGISTER_2026_11_07.md). Existing Oct 29 and earlier Nov 7 docs are historical snapshots and must not be silently rewritten into current evidence.

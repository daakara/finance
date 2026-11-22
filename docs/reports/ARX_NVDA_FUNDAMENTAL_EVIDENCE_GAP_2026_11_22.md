# NVDA fundamental evidence: diagnostic correction and integration blocker

Candidate only; not deployed. Production reference: `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.

The supplied read-only production inspection reports 493 NVDA factor rows, all
LIVE_RUNTIME/CURRENT_ONLY without filing or financial-period dates. Selected row
4344 has numerical scores but no qualifying filing lineage. No earlier qualifying
NVDA row was reported. These are operator-supplied observations, not a fresh SQL
inspection performed by this implementation task.

## Implemented first boundary

- Replace the fallback claim about regulator/exchange verification with a precise
  financial-evidence integration/metadata explanation. It propagates through the
  existing confluence pillar detail into the current UI, with no parallel renderer.
- Add explicit freshnessStatus, verificationBasis, filingAuthenticationStatus and
  missingEvidenceFields to the existing provenance contract.
- Missing evidence dates report UNKNOWN freshness. Legacy VERIFIED still means
  source/date metadata only; filing authentication is explicitly NOT_CHECKED.
- Preserve existing classification, scoring, weight, ETF and hold behavior. This
  change does not strengthen authentication or establish a verified NVDA pillar.

## Verification

Five offline unittest scenarios passed. Twenty baseline source/date comparisons
preserved evidence status, verification/synthetic/staleness flags, source and hash.
No runtime packages or external data were needed. Full release qualification is
pending; this receipt is not deployment acceptance.

## Source evidence blocker

The checkout search found NVDA security-reference metadata but no archived NVDA
10-K/10-Q company facts or raw filing documents suitable for deriving financial
metrics. The existing sourced-fundamentals ledger covers eight other companies.
A reference/ticker record and insider Form 4 bytes cannot establish financial
statement metrics. No SEC calls or other live harvesting were attempted.

Required continuation: inventory existing authenticated archives read-only for
NVDA CIK 0001045810. Verify issuer, accession, form, period, SEC acceptance/public
availability, content hash and custody against raw documents. If none exist,
report BLOCKED_SOURCE_EVIDENCE; any new collection needs a separately bounded
authorization outside the AAPL/NVDA Form 4 trial. Never relabel runtime records.

Only after source evidence is available should an isolated candidate integrate the
existing financial normalizer and metric derivations, preserve nulls and historical
records, prevent runtime estimates masking filing-backed factors, and enforce
authenticated lineage at the scoring boundary. Require adversarial source spoof,
wrong issuer/form/period, hash tampering, future availability and incomplete-metric
tests. Assess any score/coverage policy change separately; no reweighting is
authorized by this diagnostic patch.

Production SQL, NVDA archive custody, live UI, backup state and release acceptance
remain NOT_CHECKED in this task. Preserve trial ARX-TRIAL-2026-NVDA-AAPL-001,
snapshot SNAP_SM_1792939989_48f30db6 and all operational holds. No grants, queues,
SEC egress, production writes, deployments or configuration changes occurred.

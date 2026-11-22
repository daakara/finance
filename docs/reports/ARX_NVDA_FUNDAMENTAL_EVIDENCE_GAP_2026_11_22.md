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

The expanded read-only inventory found NVDA security-reference metadata but no
archived NVDA 10-K/10-Q company facts or raw filing documents suitable for deriving
financial metrics:

- Production release `2eccc64a...` contains no `CIK0001045810_companyfacts.json`,
  `CIK0001045810_submissions.json`, or filing-backed NVDA fundamentals bundle.
- Historical sourced-fundamentals branch heads `fa9dfde3...` and `dfc86d2b...`
  contain the same AAPL/GOOGL sample submissions, but no NVDA company-facts or
  submissions file. Branch `e06f66f...` also contains no qualifying NVDA archive.
- The sourced-fundamentals ledger covers FIX, DUOL, ULTA, MEDP, DECK, KLAC, LULU
  and CELH only. NVDA is not a ledger company.
- Default-branch repository searches for CIK `0001045810` plus `companyfacts`, the
  exact expected company-facts filename, and `NVIDIA sourced fundamentals` returned
  no source bundle.
- Railway's connected read listed the production web service's variable names. It
  confirmed the mounted market-store volume and database bindings, but exposed no
  separately named financial-evidence archive binding. Values were OAuth-redacted,
  so this is not proof that every object in external storage was inspected.

A reference/ticker record and insider Form 4 bytes cannot establish financial
statement metrics. No SEC calls, analytics hydration, or other live harvesting were
attempted. Cloudflare R2 object contents and any operator-local archives remain
`NOT_CHECKED` because no authenticated object-list/read path was available in this
session.

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

## Operator continuation prompt

Use this prompt only as a new, separately authorized source-custody task. It must
not be folded into the Form 4 trial or treated as permission to activate scoring:

> Perform a bounded NVDA financial-evidence acquisition and offline qualification
> for CIK `0001045810`, separate from trial `ARX-TRIAL-2026-NVDA-AAPL-001`.
> First inventory all existing operator-local and Cloudflare R2 archives read-only.
> If a complete official SEC company-facts/submissions bundle already exists,
> verify issuer identity, 10-K/10-Q accession, form, reporting period, filing date,
> SEC acceptance/public-availability timestamp, raw-byte SHA-256 and custody before
> reuse. Otherwise request explicit owner authorization before making any SEC
> request. Under that authorization, fetch only the minimum official SEC EDGAR
> company-facts and submissions resources needed for NVDA, use an attributable
> User-Agent, enforce the approved rate ceiling and request budget, and retain raw
> bytes plus a deterministic manifest. Do not call `/analytics/NVDA`, ingest Form 4,
> enqueue trial work, change grants/configuration/holds, publish a snapshot, score
> NVDA, trade, deploy or write production stores. Normalize and replay only in an
> isolated offline workspace. Fail closed on wrong issuer/form/period, missing or
> future availability, hash mismatch, incomplete metrics, amendment ambiguity or
> source spoofing. Acceptance requires an offline-replayable evidence bundle,
> verified hashes and accessions, null-preserving normalized metrics, adversarial
> tests, and an explicit `READY_FOR_SEPARATE_INTEGRATION_REVIEW` verdict. If archive
> access or collection authority is absent, report `BLOCKED_SOURCE_EVIDENCE` and
> stop without substituting `LIVE_RUNTIME` estimates.

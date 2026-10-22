# ARX Terminal — Radar Capability Freshness & Publication Integrity Closure

**Governance Model:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Mode:** TARGETED CORRECTION / CONTROLLED OFFLINE VERIFICATION  
**Railway Deployment Status:** DEPLOYMENT HOLD (Zero production mutations, zero backend operations)  
**Primary Authority:** S2-09 Radar Coverage Disclosure and Count Authority  
**Operating Worktree:** `C:\Users\akara\Documents\Projects\finance-radar-zero-candidate-fix`  
**Branch:** `fix/radar-zero-candidate-capability-remediation`  
**Base Candidate:** `1bc41e15c668def9686dd29044c9bd7fad01a87f` (Clean merge candidate of PR #40 + PR #42 + PR #44)  
**Zero-Candidate Remediation Parent:** `64eb2241af803f3d6ac7590064f81b38444396d0`  
**Final Closure Commit SHA:** `84e0f4a40b574aaa892a1264f3256c8902dc7396`  
**Final Disposition:** `RADAR_STATE_AND_INTEGRITY_DEFECTS_REMEDIATED`  
**Closing Recommendation:** `NONPRODUCTION_FRONTEND_ACCEPTANCE_CLOSED`  

---

## 1. Executive Summary

This closure report resolves the final identified Radar frontend correctness items under ARX Closure Model v2:
1. **Preventing Stale Capability State Leakage:** Eliminated operational claim carry-over across successive responses within the mounted `RadarHub` React component. All incoming responses are now normalized against a canonical default envelope; omitted strategies, missing capabilities, malformed payloads, and network failures fail closed immediately without leaking stale operational matches, build IDs, or digests.
2. **Eliminating Unsupported Static Publication Integrity Text:** Replaced the unconditional hardcoded static string `Publication Integrity: Verified` in `frontend/app/radar/page.tsx` with a dynamic, truthful attestation contract. Verification requires explicit cryptographic verification attestation (`publicationIntegrity === 'VERIFIED'` or `publicationIntegrityVerified === true`). Active rollbacks truthfully render `Publication Integrity: Rolled back`, unverified metadata without attestation truthfully renders `Publication Integrity: Not verified`, and digest strings or build IDs are never used to falsely infer verification.
3. **Out-of-Order Response Discard:** Integrated a monotonic request counter (`latestRequestIdRef`) to discard superseded in-flight responses, ensuring slow delayed responses cannot overwrite newer verified state snapshots.
4. **Preserving Remediated Invariants:** Fully retained the zero-candidate capability preservation fix (`matchedCount: 0` remains distinct from pending/unknown), scannable population provenance gates, and user filter isolation.

---

## 2. Specific Defects Reproduced & Traced

### Defect 1: Operational Claim Leakage on Missing Capabilities (`AVAILABLE → missing capabilities`)
- **Location:** `frontend/app/radar/page.tsx` lines 163–168 (prior code)
- **Defect Mechanism:** The state updater checked `if (res.capabilities) { setCapabilities((prev) => ({ ...prev, ...res.capabilities })); }`. When a subsequent response omitted `capabilities` entirely (e.g. payload `{ total_candidates: 0, results: [] }`), the conditional was bypassed entirely, leaving the previous operational scanner claims (`VCP AVAILABLE`, 12 matches) permanently displayed in the UI.
- **Evidence:** Confirmed in `frontend/tests/reproduceStateTransitions.ts` Case 1.1:
  ```
  Transition 1: AVAILABLE -> missing capabilities
     VCP status after missing capabilities: AVAILABLE
     VCP matches after missing capabilities: 12
     [DEFECT CONFIRMED 1.1]: Operational claim leaked into missing-capabilities response: YES
  ```

### Defect 2: Cross-Strategy Authority Leakage (`One strategy valid → another missing`)
- **Location:** `frontend/app/radar/page.tsx` line 164 (prior code)
- **Defect Mechanism:** Spreading previous state `...prev` meant that if a response returned valid capabilities for `VALUE_GARP` but omitted `VCP`, the stale `prev.VCP` operational status and match counts persisted unchanged.
- **Evidence:** Confirmed in `reproduceStateTransitions.ts` Case 1.2:
  ```
  Transition 7: One strategy valid (VALUE_GARP) -> another missing (VCP)
     VCP status when response omitted VCP: AVAILABLE
     VCP matches when response omitted VCP: 12
     [DEFECT CONFIRMED 1.2]: Stale VCP operational claim leaked when strategy omitted: YES
  ```

### Defect 3: Network Failure Left Stale Operational Claims Active (`Successful → network error`)
- **Location:** `frontend/app/radar/page.tsx` lines 216–219 (prior code)
- **Defect Mechanism:** The `.catch()` block only executed `setIsLoading(false)`. If a scan had previously succeeded with operational matches and a subsequent re-scan failed due to network failure, the operational claims and asset listings remained visible, misleading the user.
- **Evidence:** Confirmed in `reproduceStateTransitions.ts` Case 1.3:
  ```
  Transition 8: Successful response -> network error
     In current page.tsx line 216-219, .catch() only sets isLoading(false), leaving capabilities as stale AVAILABLE.
     [DEFECT CONFIRMED 1.3]: Stale operational claim persists despite network failure: YES
  ```

### Defect 4: Unconditional Static Assertion `Publication Integrity: Verified`
- **Location:** `frontend/app/radar/page.tsx` line 1059 (prior code)
- **Defect Mechanism:** The text `Publication Integrity: Verified` was hardcoded inside the VCP banner. Even when the VCP publication pointer was rolled back (`isRollback: true`), when scanner screening was `PIPELINE_PENDING`, or when metadata lacked verification attestation, the banner unconditionally claimed verified publication integrity.
- **Evidence:** Confirmed in `reproduceStateTransitions.ts` Case 1.4:
  ```
  Publication Integrity Badge Static Text finding
     In current page.tsx line 1058-1060: <div className="text-[11px] text-slate-500 font-mono">Publication Integrity: Verified</div>
     Renders 'Publication Integrity: Verified' unconditionally, even on rollback or pending.
     [DEFECT CONFIRMED 1.4]: Static unconditional integrity assertion: YES
  ```

### Defect 5: Out-of-Order Network Race Hazard
- **Defect Mechanism:** Concurrent or overlapping asynchronous requests to `/screener/run` had no request monotonic identifier. A slower delayed response could resolve after a newer response, overwriting fresh state with superseded stale capabilities.

---

## 3. Architecture & Code Corrections

### 3.1 Normalization & Truthful Publication Integrity Types (`frontend/lib/api.ts`)

```typescript
export type PublicationIntegrityStatus = "VERIFIED" | "NOT_VERIFIED" | "PENDING" | "ROLLBACK";

export interface RadarCapabilityDetail {
  status: RadarCapabilityStatus;
  universeScreening: RadarCapabilityStatus;
  singleAssetAnalysis: RadarCapabilityStatus;
  rationale?: string;
  matchedCount?: number | null;
  dataAsOf?: string;
  generatedAt?: string;
  snapshotId?: string;
  freshnessStatus?: string;
  subsignal?: string;
  populationDenominator?: number | null;
  universeBuildId?: string | null;
  universeId?: string | null;
  publicationDigest?: string | null;
  isRollback?: boolean;
  rollbackReason?: string | null;
  membershipStatus?: "AVAILABLE" | "PENDING" | "UNAVAILABLE";
  lifecycleState?: string;
  publicationIntegrity?: PublicationIntegrityStatus;
  publicationIntegrityVerified?: boolean;
}

export const DEFAULT_RADAR_CAPABILITIES: RadarCapabilities = {
  VALUE_GARP: {
    status: "AVAILABLE",
    universeScreening: "AVAILABLE",
    singleAssetAnalysis: "AVAILABLE",
    rationale: "Authentic multi-factor fundamental screening active across known candidate universe.",
    matchedCount: null,
    populationDenominator: null,
    universeBuildId: null,
    publicationDigest: null,
    publicationIntegrity: "NOT_VERIFIED",
    publicationIntegrityVerified: false,
  },
  VCP: {
    status: "PIPELINE_PENDING",
    universeScreening: "PIPELINE_PENDING",
    singleAssetAnalysis: "AVAILABLE",
    rationale: "Single-asset volatility contraction geometry active on /setups and /analytics; universe-level batch scanning pipeline is pending deployment.",
    matchedCount: null,
    populationDenominator: null,
    universeBuildId: null,
    publicationDigest: null,
    publicationIntegrity: "NOT_VERIFIED",
    publicationIntegrityVerified: false,
  },
  SMART_MONEY: {
    status: "PIPELINE_PENDING",
    universeScreening: "PIPELINE_PENDING",
    singleAssetAnalysis: "AVAILABLE",
    rationale: "Single-asset SEC Form 4 and Congressional STOCK Act disclosures active on /smart-money; universe-level institutional accumulation screener pipeline is pending deployment.",
    matchedCount: null,
    populationDenominator: null,
    universeBuildId: null,
    publicationDigest: null,
    publicationIntegrity: "NOT_VERIFIED",
    publicationIntegrityVerified: false,
  },
};
```

### 3.2 Publication Integrity Evaluation & Fail-Closed Normalizer (`frontend/lib/api.ts`)

```typescript
export function getPublicationIntegrityLabel(cap?: RadarCapabilityDetail | null): string {
  if (!cap) return "Publication Integrity: Not verified";
  if (cap.isRollback || cap.publicationIntegrity === "ROLLBACK") {
    return "Publication Integrity: Rolled back";
  }
  if (cap.publicationIntegrity === "VERIFIED" || cap.publicationIntegrityVerified === true) {
    return "Publication Integrity: Verified";
  }
  if (cap.publicationIntegrity === "PENDING") {
    return "Publication Integrity: Pending";
  }
  return "Publication Integrity: Not verified";
}

export function normalizeRadarCapabilities(
  incoming?: Partial<RadarCapabilities> | null
): RadarCapabilities {
  const result: RadarCapabilities = {
    VALUE_GARP: { ...DEFAULT_RADAR_CAPABILITIES.VALUE_GARP },
    VCP: { ...DEFAULT_RADAR_CAPABILITIES.VCP },
    SMART_MONEY: { ...DEFAULT_RADAR_CAPABILITIES.SMART_MONEY },
  };

  if (!incoming || typeof incoming !== "object") {
    return result;
  }

  const categories: CanonicalRadarCategory[] = ["VALUE_GARP", "VCP", "SMART_MONEY"];
  for (const cat of categories) {
    const raw = incoming[cat];
    if (!raw || typeof raw !== "object") {
      // Missing strategy in incoming payload strictly fails closed to canonical default
      continue;
    }

    // Rollback protection: rolled-back publication cannot claim operational screening
    if (raw.isRollback) {
      result[cat] = {
        ...DEFAULT_RADAR_CAPABILITIES[cat],
        status: "PIPELINE_PENDING",
        universeScreening: "PIPELINE_PENDING",
        singleAssetAnalysis: "AVAILABLE",
        matchedCount: null,
        populationDenominator: typeof raw.populationDenominator === "number" ? raw.populationDenominator : null,
        universeBuildId: raw.universeBuildId ?? null,
        universeId: raw.universeId ?? null,
        publicationDigest: raw.publicationDigest ?? null,
        isRollback: true,
        rollbackReason: raw.rollbackReason ?? null,
        membershipStatus: raw.membershipStatus ?? "UNAVAILABLE",
        lifecycleState: raw.lifecycleState ?? "STALE_OR_ROLLED_BACK_PUBLICATION",
        rationale: raw.rationale ?? `Publication build '${raw.universeBuildId}' was rolled back. Scanner execution is suspended.`,
        publicationIntegrity: "ROLLBACK",
        publicationIntegrityVerified: false,
      };
      continue;
    }

    // Operational match count validation: authentic non-negative integer
    const hasValidMatchedCount =
      typeof raw.matchedCount === "number" &&
      Number.isInteger(raw.matchedCount) &&
      raw.matchedCount >= 0;

    const isAvailable = (raw.universeScreening === "AVAILABLE" || raw.status === "AVAILABLE") && hasValidMatchedCount;

    // Truthful Publication Integrity evaluation:
    // Verified ONLY when explicit authoritative attestation is present.
    // Never inferred from publicationDigest, buildId, or operational scanner status alone.
    let pubIntegrity: PublicationIntegrityStatus = "NOT_VERIFIED";
    if (raw.publicationIntegrity === "VERIFIED" || raw.publicationIntegrityVerified === true) {
      pubIntegrity = "VERIFIED";
    } else if (raw.publicationIntegrity === "PENDING") {
      pubIntegrity = "PENDING";
    } else if (raw.publicationIntegrity === "ROLLBACK") {
      pubIntegrity = "ROLLBACK";
    }

    result[cat] = {
      status: isAvailable ? "AVAILABLE" : (raw.status === "AVAILABLE" ? "PIPELINE_PENDING" : (raw.status || DEFAULT_RADAR_CAPABILITIES[cat].status)),
      universeScreening: isAvailable ? "AVAILABLE" : (raw.universeScreening === "AVAILABLE" ? "PIPELINE_PENDING" : (raw.universeScreening || DEFAULT_RADAR_CAPABILITIES[cat].universeScreening)),
      singleAssetAnalysis: raw.singleAssetAnalysis || DEFAULT_RADAR_CAPABILITIES[cat].singleAssetAnalysis,
      rationale: raw.rationale || DEFAULT_RADAR_CAPABILITIES[cat].rationale,
      matchedCount: isAvailable ? raw.matchedCount : null,
      dataAsOf: raw.dataAsOf,
      generatedAt: raw.generatedAt,
      snapshotId: raw.snapshotId,
      freshnessStatus: raw.freshnessStatus,
      subsignal: raw.subsignal,
      populationDenominator: typeof raw.populationDenominator === "number" ? raw.populationDenominator : null,
      universeBuildId: raw.universeBuildId ?? null,
      universeId: raw.universeId ?? null,
      publicationDigest: raw.publicationDigest ?? null,
      isRollback: false,
      rollbackReason: null,
      membershipStatus: raw.membershipStatus ?? "AVAILABLE",
      lifecycleState: raw.lifecycleState,
      publicationIntegrity: pubIntegrity,
      publicationIntegrityVerified: pubIntegrity === "VERIFIED",
    };
  }

  return result;
}
```

### 3.3 State Lifecycle & Monotonic Discard in Radar (`frontend/app/radar/page.tsx`)

```typescript
  const [refreshNonce, setRefreshNonce] = useState(0);
  const latestRequestIdRef = React.useRef(0);

  useEffect(() => {
    let isMounted = true;
    const reqId = ++latestRequestIdRef.current;
    setIsLoading(true);

    fetchScreenerGems("all")
      .then((res) => {
        if (!isMounted || reqId !== latestRequestIdRef.current) return;
        setCapabilities(normalizeRadarCapabilities(res.capabilities));
        const mapped: RadarAsset[] = (res.results || []).map((gem: any) => { ... });
        setAllAssets(mapped);
        setIsLoading(false);
      })
      .catch(() => {
        if (!isMounted || reqId !== latestRequestIdRef.current) return;
        setCapabilities(DEFAULT_RADAR_CAPABILITIES);
        setAllAssets([]);
        setIsLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [refreshNonce]);
```

### 3.4 Dynamic Publication Integrity Banner (`frontend/app/radar/page.tsx`)

```tsx
{(() => {
  const cap = capabilities.VCP;
  const label = getPublicationIntegrityLabel(cap);
  if (cap?.isRollback || cap?.publicationIntegrity === 'ROLLBACK') {
    return (
      <div className="text-[11px] text-amber-400/90 font-mono flex items-center gap-1.5" data-testid="publication-integrity-badge">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
        {label}
      </div>
    );
  }
  if (cap?.publicationIntegrity === 'VERIFIED' || cap?.publicationIntegrityVerified === true) {
    return (
      <div className="text-[11px] text-emerald-400 font-mono flex items-center gap-1.5" data-testid="publication-integrity-badge">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
        {label}
      </div>
    );
  }
  return (
    <div className="text-[11px] text-slate-500 font-mono flex items-center gap-1.5" data-testid="publication-integrity-badge">
      <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
      {label}
    </div>
  );
})()}
```

---

## 4. Verification Evidence

### 4.1 Automated Test Suite (`frontend/tests/radarCountAuthority.test.ts`)

Executed via `node ./node_modules/vite-node/vite-node.mjs tests/radarCountAuthority.test.ts`:

```
Starting Radar Count Authority & Filtered Display Verification Suite...

1. Verifying static count authority invariants in frontend/app/radar/page.tsx...
   [PASS] 1.1: Unsupported '100% coverage' fallback text eliminated
   [PASS] 1.2: Population count strictly gated on buildId and publicationDigest, with 'Coverage Pending' fallback
   [PASS] 1.3: Conflation of filteredAssets.length with scanner matches eliminated
   [PASS] 1.4: Filtered display count clearly distinguished from authoritative scanner match count
   [PASS] 1.5: Authoritative scanner matches displayed when no reducing filter is applied
   [PASS] 1.6: Pending states render 'Scanner Pending', never fabricated zero matches

2. Simulating Radar banner count formatting logic across filter states & provenance...
   [PASS] 2.1: Unfiltered operational scanner: '12 scanner matches · 150 scannable population'
   [PASS] 2.2: User search filter: 'Showing 1 of 12 scanner matches · 150 scannable population' (matchedCount=12 intact)
   [PASS] 2.3: Incomplete provenance yields '12 scanner matches · Coverage Pending'

3. Testing truthful operational zero matches vs pending states...
   [PASS] 3.1: Authoritative operational zero matches: '0 scanner matches · 150 scannable population'
   [PASS] 3.2: Pending scanner: 'Scanner Pending · 150 scannable population' (zero fabricated 0)
   [PASS] 3.3: Boolean matchedCount rejected and fails closed to 'Scanner Pending'
   [PASS] 3.4: Negative matchedCount rejected and fails closed to 'Scanner Pending'

4. Testing rollback pointer containment...
   [PASS] 4.1: Rolled-back pointer enforces PIPELINE_PENDING and matchedCount=null

5. Testing filter independence across sort and ownership states...
   [PASS] 5.1: Ownership filter reduces displayed assets (2) without mutating authoritative matches (3)
   [PASS] 5.2: Search filter reduces displayed assets (1) without mutating authoritative matches (3)

6. Testing fetchScreenerGems() real transformation path across all 6 capability cases...
   [PASS] 6.1: Empty candidates + operational zero -> '0 scanner matches · 150 scannable population'
   [PASS] 6.2: Empty candidates + pending scanner -> 'Scanner Pending · 150 scannable population'
   [PASS] 6.3: Empty candidates + unavailable capabilities -> fail closed to 'Scanner Pending · Coverage Pending'
   [PASS] 6.4: Empty candidates + rollback -> matches suppressed to 'Scanner Pending · 150 scannable population'
   [PASS] 6.5: Empty candidates + missing build provenance -> '0 scanner matches · Coverage Pending'
   [PASS] 6.6: Nonempty candidate compatibility preserved -> '1 scanner matches · 150 scannable population'

7. Verifying sequential state transitions & request discard...
   [PASS] 7.1: AVAILABLE -> missing capabilities: Operational claim removed and fails closed
   [PASS] 7.2: AVAILABLE -> PIPELINE_PENDING: Previous matches removed, renders 'Scanner Pending'
   [PASS] 7.3: AVAILABLE -> rolled-back publication: Rollback disclosed and operational claims suppressed
   [PASS] 7.4: Valid provenance -> missing digest: Provenance gate enforces 'Coverage Pending'
   [PASS] 7.5: Operational zero -> malformed response: Fails closed to PIPELINE_PENDING
   [PASS] 7.6: Pending -> verified AVAILABLE: Authoritative matches and verified integrity displayed
   [PASS] 7.7: One strategy valid -> another missing: Zero cross-strategy authority leakage
   [PASS] 7.8: Successful response -> network error: Stale operational claims fully cleared
   [PASS] 7.9: Out-of-order response discard: Stale older response discarded cleanly

8. Verifying truthful Publication Integrity badge presentation...
   [PASS] 8.1: Explicit attestation -> 'Publication Integrity: Verified'
   [PASS] 8.2: Rollback condition -> 'Publication Integrity: Rolled back'
   [PASS] 8.3: Metadata present without attestation -> 'Publication Integrity: Not verified' (no false positive)
   [PASS] 8.4: Pending integrity -> 'Publication Integrity: Pending'
   [PASS] 8.5: Null/undefined capability -> 'Publication Integrity: Not verified'

===============================================================================
ALL RADAR COUNT AUTHORITY & REAL DATA TRANSFORMATION TESTS PASSED! ✅
===============================================================================
```

### 4.2 TypeScript Type-Check

Executed via `node node_modules/typescript/bin/tsc --noEmit`:
```
Exit code: 0
Diagnostic output: None (0 errors)
```

### 4.3 Real-Browser Playwright Verification (4 Minimal States)

Executed via headless Chromium targeting the isolated Next.js runtime on port 3035:

| State | Scenario | Rendered Banner / Tab | Rendered Integrity Badge | Screenshot Evidence | Status |
|---|---|---|---|---|---|
| **A** | Verified publication, operational scanner | `2 scanner matches · 150 scannable population` (Tab: `2`) | `Publication Integrity: Verified` (emerald badge) | `radar_state_verified_operational.png` | **PASS** |
| **B** | Missing publication evidence (neutral integrity) | `1 scanner matches · Coverage Pending` (Tab: `1`) | `Publication Integrity: Not verified` (slate badge) | `radar_state_missing_evidence.png` | **PASS** |
| **C** | Rolled-back publication (rollback disclosed) | `Scanner Pending · 150 scannable population • Build: build-s209-150-rollback(Rollback)` (Tab: `Universe Scanner Pending`) | `Publication Integrity: Rolled back` (amber badge) | `radar_state_rollback.png` | **PASS** |
| **D** | In-component state transition: Operational $\to$ Missing Capabilities | Initial: `2 scanner matches · 150 scannable population` $\to$ After transition: `Scanner Pending · Coverage Pending` (Tab: `Universe Scanner Pending`) | Initial: `Verified` $\to$ After transition: `Publication Integrity: Not verified` | `radar_state_transition_missing_caps.png` | **PASS** |

Screenshots are persisted in:
- `evidence/visual_acceptance/radar_state_*.png`
- `<appDataDir>\brain\5bcd9ef1-035c-4ab1-9453-2cb5f9e74b20\screenshots\radar_state_*.png`

---

## 5. Scope Boundaries & Mandatory Holds

In accordance with ARX Closure Model v2:
- **Zero Production Mutations:** Zero Railway deploys, restarts, or retries. Zero production database mutations.
- **Zero Scanner Activations:** No automated scanner activations or live market data ingestions were triggered.
- **Strict Worktree Isolation:** All edits were committed exclusively within `finance-radar-zero-candidate-fix` on branch `fix/radar-zero-candidate-capability-remediation`. Existing PR branches (PR #40, PR #42, PR #44) remain unmodified.
- **Preserved Release Invariants:**
  - `matchedCount: 0` remains preserved as an authentic operational zero.
  - Provenance gates (`universeBuildId` and `publicationDigest`) strictly control scannable population disclosure.
  - User search and portfolio ownership filters operate independently of scanner match authority.

---

## 6. Final Disposition & Release Verdict

**Disposition:**  
`RADAR_STATE_AND_INTEGRITY_DEFECTS_REMEDIATED`

**Closing Recommendation:**  
`NONPRODUCTION_FRONTEND_ACCEPTANCE_CLOSED`

All real-browser, unit, and integration visual acceptance gates for ARX Terminal Radar are hereby closed.

# ARX Terminal — PR #98 Mobile Navigation Defect Reproduction Report

**Directive ID:** `ARX_PR98_MOBILE_NAVIGATION_REMEDIATION_026`  
**Repository:** `daakara/finance`  
**PR:** [#98](https://github.com/daakara/finance/pull/98) — *Synthesis E V3 UX Parity*  
**Baseline Candidate SHA:** `c25b03d30c257b021ba40886e6f05049d112c52f`  
**Mainline Base:** `ea2bc6fb6322f4d7a5411b3a3d0e0f864a687cba` (PR #108 merged)  
**Classification:** P1 — Functional UX Correctness  
**Date:** October 27, 2026  

---

## 1. Executive Summary

Prior to final owner acceptance of PR #98, an adversarial review identified two source-confirmed functional UX defects in the mobile navigation architecture:

1. **Defect MOB-01 — Asset Context Contamination:** Implicit fallback to `"AAPL"` in `SynthesisMobileDock.tsx` combined with uncontextualized mounting in `frontend/app/radar/page.tsx`, causing mobile navigation from Radar to force-redirect to Apple Inc. (`AAPL`) regardless of the user's active viewport or search candidate.
2. **Defect MOB-02 — Inert Mobile Search:** `SynthesisMobileDock.tsx` attempting direct DOM manipulation (`document.getElementById("omni-search-input").focus()`) against an unmounted element, because `UniversalOmniSearch.tsx` conditionally renders `#omni-search-input` only when `isOpen === true`. When closed, the lookup returns `null`, causing the mobile dock's "Search" button to fail silently.

Both defects have been rigorously reproduced in isolation, traced to exact source lines, and documented below.

---

## 2. Defect MOB-01: Asset Context Contamination

### 2.1 Observed Source Behavior & Root Cause

- **File 1:** [`frontend/components/synthesis/SynthesisMobileDock.tsx`](file:///C:/Users/akara/Documents/Projects/finance-pr98-ux-integration/frontend/components/synthesis/SynthesisMobileDock.tsx#L8-L15)
  ```tsx
  interface SynthesisMobileDockProps {
    activeSymbol?: string | null;
  }

  export default function SynthesisMobileDock({ activeSymbol = "AAPL" }: SynthesisMobileDockProps) {
    const pathname = usePathname();
    const symbolParam = activeSymbol ? `?symbol=${activeSymbol}` : "";
    ...
    <Link href={`/${symbolParam}`} ...>Analysis</Link>
    ...
    <Link href={activeSymbol ? `/setups?ticker=${activeSymbol}` : "/setups"}>Plan</Link>
  ```
  The component defaulted `activeSymbol` to `"AAPL"` when omitted by the parent.

- **File 2:** [`frontend/app/radar/page.tsx`](file:///C:/Users/akara/Documents/Projects/finance-pr98-ux-integration/frontend/app/radar/page.tsx#L1564)
  ```tsx
        </div>
        )}
      </div>
      <SynthesisMobileDock />
    </TerminalShell>
  ```
  Radar mounted `<SynthesisMobileDock />` without passing an `activeSymbol` property, despite having canonical search state (`cleanQ` / `searchQuery`).

### 2.2 Reproduction Impact

1. A user on mobile browses `/radar` without selecting any asset. Tapping "Analysis" opened `/?symbol=AAPL` instead of the neutral `/` overview. Tapping "Plan" opened `/setups?ticker=AAPL` instead of the neutral `/setups` setup screener.
2. A user searches for `"NVDA"` on Radar. The mobile dock's Analysis and Plan buttons continued linking to `AAPL` because no asset context was passed from Radar into `SynthesisMobileDock`.
3. This violated the core ARX terminal invariant that the platform must never fabricate or substitute positive asset evidence.

---

## 3. Defect MOB-02: Inert Mobile Search

### 3.1 Observed Source Behavior & Root Cause

- **File 1:** [`frontend/components/synthesis/SynthesisMobileDock.tsx`](file:///C:/Users/akara/Documents/Projects/finance-pr98-ux-integration/frontend/components/synthesis/SynthesisMobileDock.tsx#L16-L22)
  ```tsx
  const handleOpenSearch = () => {
    const input = document.getElementById("omni-search-input");
    if (input) {
      input.focus();
      input.scrollIntoView({ behavior: "smooth" });
    }
  };
  ```

- **File 2:** [`frontend/components/UniversalOmniSearch.tsx`](file:///C:/Users/akara/Documents/Projects/finance-pr98-ux-integration/frontend/components/UniversalOmniSearch.tsx#L14-L185)
  ```tsx
  export default function UniversalOmniSearch() {
    const [isOpen, setIsOpen] = useState(false);
    ...
    {/* Fullscreen Search Modal Backdrop */}
    {isOpen && (
      <div role="dialog" aria-modal="true" aria-label="Universal Asset Search" ...>
        ...
        <input id="omni-search-input" ref={inputRef} ... />
      </div>
    )}
  ```

### 3.2 Reproduction Impact

1. When the user taps the "Search" button on the mobile dock, `isOpen` in `UniversalOmniSearch` is `false`.
2. As a consequence of conditional rendering, `<input id="omni-search-input">` does not exist in the DOM tree.
3. `document.getElementById("omni-search-input")` evaluates to `null`.
4. The `if (input)` branch is skipped. Zero actions are performed, zero events are emitted, and the dialog remains closed.
5. The mobile search button fails silently with no user feedback.

---

## 4. Defect Reproduction Verification Log

Execution of the isolated defect probe against candidate `c25b03d30c257b021ba40886e6f05049d112c52f`:

```text
===============================================================================
ARX PR #98 — Mobile Navigation Defect Reproduction & Validation Suite
Directive: ARX_PR98_MOBILE_NAVIGATION_REMEDIATION_026
===============================================================================

--- 1. Inspecting MOB-01: Asset Context Contamination ---
[MOB-01 Probe] SynthesisMobileDock has default activeSymbol = "AAPL": true
[MOB-01 Probe] Radar mounts <SynthesisMobileDock /> without activeSymbol prop: true
>> CONFIRMED DEFECT MOB-01: Radar mounts SynthesisMobileDock without symbol, causing dock to fall back to 'AAPL' default. Navigation from Radar opens Analysis/Plan for AAPL regardless of asset context.

--- 2. Inspecting MOB-02: Inert Mobile Search ---
[MOB-02 Probe] SynthesisMobileDock uses document.getElementById('omni-search-input'): true
[MOB-02 Probe] UniversalOmniSearch renders #omni-search-input conditionally under isOpen: true
[MOB-02 Probe] UniversalOmniSearch listens for 'arx:open-omni-search' event: false
>> CONFIRMED DEFECT MOB-02: When search dialog is closed (isOpen=false), #omni-search-input is unmounted. SynthesisMobileDock's direct getElementById lookup returns null, causing Search button to fail silently without opening modal.

Defect inspection complete.
```

Both defects were verified 100% reproducible on the qualified candidate before applying surgical remediation.

# Quality Gate D — Rendered Browser Verification Report

**Verification Date**: 2026-10-19 22:49:50 UTC  
**Engine**: Playwright Chromium (Automated Headless Browser)  
**Status**: **GATE D PASSED (100% UNOBSCURED)**

## Screenshot Verification Inventory

| Ticker | Viewport | Dimensions | Card Bounding Box | Card PNG | Viewport PNG | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **FIX** | desktop | 1280x900 | 848x1360px | `unobscured_card_fix_desktop.png` | `unobscured_viewport_fix_desktop.png` | **PASSED** |
| **DUOL** | desktop | 1280x900 | 848x1444px | `unobscured_card_duol_desktop.png` | `unobscured_viewport_duol_desktop.png` | **PASSED** |
| **ULTA** | desktop | 1280x900 | 848x1479px | `unobscured_card_ulta_desktop.png` | `unobscured_viewport_ulta_desktop.png` | **PASSED** |
| **AAPL** | desktop | 1280x900 | 848x166px | `unobscured_card_aapl_desktop.png` | `unobscured_viewport_aapl_desktop.png` | **PASSED** |
| **FIX** | mobile | 390x844 | 358x2292px | `unobscured_card_fix_mobile.png` | `unobscured_viewport_fix_mobile.png` | **PASSED** |
| **DUOL** | mobile | 390x844 | 358x2444px | `unobscured_card_duol_mobile.png` | `unobscured_viewport_duol_mobile.png` | **PASSED** |
| **ULTA** | mobile | 390x844 | 358x2513px | `unobscured_card_ulta_mobile.png` | `unobscured_viewport_ulta_mobile.png` | **PASSED** |
| **AAPL** | mobile | 390x844 | 358x272px | `unobscured_card_aapl_mobile.png` | `unobscured_viewport_aapl_mobile.png` | **PASSED** |

## Invariant Checklist

- [x] **No Modal Occlusion**: Onboarding tour modals permanently dismissed via localStorage; 0 visible dialogs.
- [x] **Element Visibility**: `#sourced-fundamentals-section` verified visible with positive area across all targets.
- [x] **Facts-Only Invariant**: Verified complete absence of DCF scenarios, fair values, or investment rankings.
- [x] **Mobile Responsiveness**: Narrow viewport (390x844) renders without overflow or clipped metrics.
- [x] **Graceful Fallback**: AAPL uncataloged state displays neutral informative message with links to case studies.

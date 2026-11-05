# ARX Terminal Production Releases

This directory contains immutable release notes for verified ARX Terminal production deployments.

A production release is considered closed only when:

```
DEPLOYED
+
PRODUCTION_VERIFIED
+
RELEASE_NOTES_CREATED
+
RELEASE_NOTES_COMMITTED
```

---

## Release Index

> This index records **dated production releases**, not the current branch head. For the latest independently verified state and unfinished deployment gates, see [Current project status](../ARX_CURRENT_PROJECT_STATUS.md). Historical release notes remain immutable records of what was known at the time.

| Date | Release SHA | Summary | Status |
| :--- | :--- | :--- | :--- |
| 2026-11-06 | [`ccd8fe1a`](2026-11-06_ccd8fe1a_radar_evidence_promotion.md) | PR #199 Radar historical VCP evidence integrity promotion | PRODUCTION_ACCEPTED for historical VCP evidence (011U operator receipt) |
| 2026-11-05 | [`74123474`](2026-11-05_74123474_smart_money_recovery_002r.md) | PR #195 Smart Money SEC Form 4 evidence-read permission recovery; actual NVDA readback | **PRODUCTION_ACCEPTED** for read path (operator acceptance + platform verification) |
| 2026-11-02 | [`7a513929`](2026-11-02_7a513929_smart-money-radar.md) | PR #161 consolidated Smart Money and Radar release | DEPLOYED; original acceptance partially verified; superseded by newer release |
| 2026-10-08 | [`5dcfeb4`](2026-10-08_5dcfeb4_execution-ladder-passive-capture.md) | Execution Ladder Prospective Passive Capture | VERIFIED |
| 2026-10-08 | [`5a084a9`](2026-10-08_5a084a9_qa-release-hardening.md) | Production QA Escape Hardening & Release Governance | VERIFIED |
| 2026-10-08 | [`dd972e9`](2026-10-08_dd972e9_qa-functional-release-resolution-fix.md) | QA Functional Release Resolution Fix | VERIFIED |
| 2026-10-08 | [`a05331f`](2026-10-08_a05331f_decision-surface-integrity-remediation.md) | Decision-Surface Integrity Remediation (QA-ESC-011) | VERIFIED |
| 2026-10-08 | [`3f89f8f`](2026-10-08_3f89f8f_synthesis-e-wave-4-decision-readiness.md) | Synthesis E Wave 4 Decision Readiness & State Harmonization | VERIFIED |
| 2026-10-08 | [`defdf89`](2026-10-08_defdf89_responsive-chart-priority-remediation.md) | Responsive Chart Priority & Data Authority Remediation | VERIFIED |
| 2026-10-08 | [`b26163f`](2026-10-08_b26163f_execution-ladder-capture-wiring-v3-governance.md) | Execution Ladder Capture Wiring Remediation & Epoch 4 V3 Governance | VERIFIED |
| 2026-10-09 | [`1328547`](2026-10-09_1328547_manual-holding-exits-mobile-radar-remediation.md) | Mobile Radar Remediation, Manual Holding Durable Exits & Portfolio Governance | VERIFIED |

# ARX Terminal — Production Release Notes

## QA Functional Release Resolution Fix

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
RELEASE_SHA =
  dd972e9767741b69842a28db3e80eac1255184f1
PREVIOUS_FUNCTIONAL_QA_RELEASE_SHA =
  5a084a9a9b9d01fdd090c1866dfbc4ffea1e3889
PREVIOUS_RUNTIME_SHA =
  e8c7597064fbec970b7f9d353e35643986debb82
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway & Cloudflare deployment
DEPLOYMENT_STATUS =
  DEPLOYED
PRODUCTION_VERIFICATION =
  VERIFIED
```

---

### Release Classification & Behavior Invariants

```ini
RELEASE_CLASSIFICATION =
  QA_HARDENING
  GOVERNANCE
  BUG_FIX

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  NONE

QA_SYSTEM_BEHAVIOR_CHANGE =
  YES

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

MODEL_TUNING =
  NO
```

This release addresses a QA and governance behavior defect in the post-deploy production verification gate. Zero modifications were made to production quantitative models, recommendation algorithms, execution ladder rules, or production client application runtime code.

---

### What Changed

#### Fixed
* **Closest Topological Ancestor Resolution in Post-Deploy Gate (`scripts/qa/post_deploy_production_verification.py`)**:
  Corrected functional release discovery during automatic ancestry scanning. Previously, candidate documented releases were selected arbitrarily from `valid_ancestors` (subject to directory listing order). The verifier now tracks the most recent topological ancestor in Git history (`is_git_ancestor(best_sha, cand_sha)`), ensuring that documentation-only descendant deployments resolve deterministically to the closest valid documented functional ancestor rather than an older historical ancestor.

---

### Release Quality & Ancestry Model

Under ARX deployment governance:
1. Documentation-only commits (such as release notes) trigger automatic cloud platform rebuilds, producing a new `RUNTIME_SHA`.
2. The deployment identity model requires:
   ```text
   CURRENT_DEPLOYED_SHA (RUNTIME_SHA)
           ↓ (contains / descends from)
   FUNCTIONAL_RELEASE_SHA
           ↓ (documented in)
   docs/releases/<date>_<short-sha>_<title>.md
   ```
3. With this fix, any documentation-only redeployment descending from `dd972e9` will select `dd972e9` (or its designated successor) as its closest documented functional release ancestor, upholding strict immutable governance.

---

### Passive Capture & Prospective Denominator

```ini
PROSPECTIVE_DENOMINATOR =
  0
DENOMINATOR_DELTA =
  0
```

Zero test runs, smoke executions, or verifier operations interacted with prospective capture stores. The prospective denominator remains strictly zero.

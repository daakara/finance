# Exception Record — EXC-001
**Title**: Frozen Engine Manifest Hash Mismatch  
**Classification**: PRE_EXISTING_UNRELATED  
**Severity**: LOW (Internal governance manifest hash check discrepancy)  
**Root Cause**: `verify_frozen_engine_manifest()` evaluates engine files against hashes pinned during Epoch 1 freeze prior to Sprint 2A/2B and price authority enhancements.  
**Impact**: Test assertion fails (`CORRUPTED != VERIFIED`), but active production engines are functional, correct, and protected by SQLite triggers.  
**Baseline Verification**: Reproduces identically at baseline `5a90b91`.  
**Candidate Independence**: Zero diff between candidate `74baf30` and baseline `5a90b91` on `FROZEN_ENGINE_MANIFEST.json` and engine files.  
**Status**: PENDING_PRODUCT_OWNER_APPROVAL  

# Exception Record — EXC-002
**Title**: Test Fixture is_actionable KeyError  
**Classification**: PRE_EXISTING_UNRELATED  
**Severity**: LOW (Unit test fixture expectation error)  
**Root Cause**: Unit test `test_prospective_extended_asset_never_emits_target_reached` in `tests/test_qa_escape_invariants.py` calls internal static method `OptimalExecutionEngine._enforce_execution_invariants()`, which normalizes execution parameters and sets `is_in_buy_zone`, but does not output `is_actionable` (which is produced at the higher orchestration/governance layer).  
**Impact**: Unit test raises `KeyError: 'is_actionable'`.  
**Safety Invariant Proof**: Core safety invariant passes unconditionally: spot >= TP1 evaluates strictly to `WAITING_PULLBACK` or `EXTENDED_ABOVE_BUY_ZONE`, never emitting `TARGET_REACHED`.  
**Baseline Verification**: Reproduces identically at baseline `5a90b91`.  
**Candidate Independence**: Zero diff between candidate `74baf30` and baseline `5a90b91` on `optimal_execution.py` or `test_qa_escape_invariants.py`.  
**Status**: PENDING_PRODUCT_OWNER_APPROVAL  

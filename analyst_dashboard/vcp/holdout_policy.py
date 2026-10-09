"""ARX VCP Holdout Precommitment Policy & Protocol.

Sprint 2B Terminal Semantics Correction + Internal Freeze Gate.
Governs holdout precommitment protocol, causal precedence invariants,
and historical non-retroactive-repairability of the current candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


HOLDOUT_PRECOMMITMENT_POLICY_ID: str = "ARX_VCP_HOLDOUT_PRECOMMITMENT_POLICY"
HOLDOUT_PRECOMMITMENT_POLICY_VERSION: str = "1.0.0"


class ProofMechanism(str, Enum):
    """Governed mechanisms for establishing cryptographic causal precedence."""
    SIGNED_REPOSITORY_COMMIT = "SIGNED_REPOSITORY_COMMIT"
    SIGNED_TAG = "SIGNED_TAG"
    IMMUTABLE_CI_ARTIFACT = "IMMUTABLE_CI_ARTIFACT"
    TRUSTED_TIMESTAMP = "TRUSTED_TIMESTAMP"
    EXTERNAL_NOTARIZATION = "EXTERNAL_NOTARIZATION"
    OTHER_GOVERNED_MECHANISM = "OTHER_GOVERNED_MECHANISM"


# Historical Invariants for Current Candidate:
CURRENT_CANDIDATE_HOLDOUT_PRECOMMITMENT: str = "NOT_ESTABLISHED"
CURRENT_CANDIDATE_PRECOMMITMENT_DEFECT: str = "HISTORICAL / NON_RETROACTIVELY_REPAIRABLE"
RETROACTIVE_TIMESTAMP_CAN_ESTABLISH_PRECOMMITMENT: bool = False
HOLDOUT_PRECOMMITMENT_REPAIR_FOR_CURRENT_CANDIDATE: str = "IMPOSSIBLE_RETROACTIVELY"
CURRENT_HOLDOUT_ENGINEERING_UTILITY: str = "PRESERVED"
CURRENT_HOLDOUT_PRECOMMITTED_PROSPECTIVE_AUTHORITY: str = "NOT_ESTABLISHED"
PRECOMMITMENT_PROOF_REQUIRES_CAUSAL_PRECEDENCE: bool = True


PRECOMMITMENT_EPOCH_PROTOCOL_STEPS: Tuple[str, ...] = (
    "1. CREATE_OR_SELECT_NEW_HOLDOUT_CASES",
    "2. ESTABLISH_EXPECTATIONS_INDEPENDENTLY_OF_CANDIDATE_IMPLEMENTATION",
    "3. FREEZE_HOLDOUT_MEMBERSHIP",
    "4. FREEZE_EXPECTED_PREDICATE_VECTORS_AND_FINAL_OUTCOMES",
    "5. CREATE_CANONICAL_HOLDOUT_COMMITMENT",
    "6. COMMIT_SIGN_TIMESTAMP_COMMITMENT_VIA_APPROVED_MECHANISM",
    "7. VERIFY_COMMITMENT_PREDATES_CANDIDATE_FUNCTIONAL_FREEZE",
    "8. IMPLEMENT_AND_FREEZE_CANDIDATE",
    "9. REVEAL_AND_EVALUATE_HOLDOUT_ONCE",
    "10. PRESERVE_REVEAL_ARTIFACT_AND_RESULTS_IMMUTABLY",
)


def verify_precommitment_ordering(commitment_timestamp: str, candidate_freeze_timestamp: str) -> bool:
    """Verifies strict causal precedence: holdout commitment must precede candidate freeze."""
    if not PRECOMMITMENT_PROOF_REQUIRES_CAUSAL_PRECEDENCE:
        return False
    if commitment_timestamp >= candidate_freeze_timestamp:
        raise ValueError(
            f"PRECOMMITMENT_CAUSAL_PRECEDENCE_VIOLATION: Commitment timestamp ({commitment_timestamp}) "
            f"does not precede candidate freeze timestamp ({candidate_freeze_timestamp})."
        )
    return True


def audit_candidate_precommitment(
    claimed_retroactive_timestamp: Optional[str] = None,
    claim_prospective_precommitted: bool = False,
) -> Dict[str, Any]:
    """Audits current candidate precommitment status and prevents invalid claims."""
    if claimed_retroactive_timestamp is not None:
        raise ValueError(
            "RETROACTIVE_TIMESTAMP_PROHIBITED: The current candidate's precommitment defect is "
            "HISTORICAL / NON_RETROACTIVELY_REPAIRABLE. A retroactive timestamp cannot manufacture past precommitment."
        )
    if claim_prospective_precommitted:
        raise ValueError(
            "INVALID_PROSPECTIVE_CLAIM: Current holdout cases preserve engineering conformance and regression utility, "
            "but prospective precommitted holdout authority is NOT_ESTABLISHED."
        )
    return {
        "candidate_holdout_precommitment": CURRENT_CANDIDATE_HOLDOUT_PRECOMMITMENT,
        "precommitment_defect": CURRENT_CANDIDATE_PRECOMMITMENT_DEFECT,
        "retroactive_timestamp_valid": RETROACTIVE_TIMESTAMP_CAN_ESTABLISH_PRECOMMITMENT,
        "repair_possible": HOLDOUT_PRECOMMITMENT_REPAIR_FOR_CURRENT_CANDIDATE,
        "engineering_utility": CURRENT_HOLDOUT_ENGINEERING_UTILITY,
        "prospective_authority": CURRENT_HOLDOUT_PRECOMMITTED_PROSPECTIVE_AUTHORITY,
    }


def compute_holdout_precommitment_policy_hash() -> str:
    """Computes deterministic hash over ARX_VCP_HOLDOUT_PRECOMMITMENT_POLICY."""
    spec = {
        "policy_id": HOLDOUT_PRECOMMITMENT_POLICY_ID,
        "version": HOLDOUT_PRECOMMITMENT_POLICY_VERSION,
        "mechanisms": [m.value for m in ProofMechanism],
        "protocol_steps": list(PRECOMMITMENT_EPOCH_PROTOCOL_STEPS),
        "causal_precedence_required": PRECOMMITMENT_PROOF_REQUIRES_CAUSAL_PRECEDENCE,
        "retroactive_timestamp_allowed": RETROACTIVE_TIMESTAMP_CAN_ESTABLISH_PRECOMMITMENT,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()

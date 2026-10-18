"""
scripts/generate_three_candidate_plan.py

Generates the canonical three-candidate identity registration plan artifact for
AAPL, GOOG, and GOOGL under the formally approved Product Owner architectural principle:
"An evidence-resolved internal security identity and supported listing association
may be registered independently of strategy eligibility and R06 admission status.
Such registration grants no conversion, admission, publication or scanner authority."

Outputs:
    evidence/identity_registration/three_candidate_registration_plan.json
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analyst_dashboard.universe.identity_proposal import build_three_candidate_proposals
from analyst_dashboard.universe.identity_adoption import (
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionAuthorization,
    IdentityAdoptionPlan,
)
from analyst_dashboard.universe.identity_persistence import SqliteIdentityRepository


def generate_plan() -> IdentityAdoptionPlan:
    # 1. Baseline target store: empty initialized repository snapshot
    repo = SqliteIdentityRepository(":memory:")
    target_state_digest = repo.compute_target_state_digest()

    # 2. Focus proposal ledger for AAPL, GOOG, GOOGL
    proposals = build_three_candidate_proposals()

    # 3. Explicit representation of approved architectural principle with UNAPPROVED exact plan
    auth = IdentityAdoptionAuthorization(
        is_authorized=False,
        authorized_by=None,
        authorization_timestamp_utc=None,
        scope="REFERENCE_IDENTITY_REGISTRATION",
        allow_reference_identity_registration=True,
        exact_plan_approval="ABSENT",
        application_authorized=False,
        production_target="PROPOSED_NOT_INITIALIZED",
        notes=(
            "PO approved architectural principle: evidence-resolved reference identity registration "
            "independently of strategy eligibility and R06 admission status. "
            "Exact candidate plan approval is ABSENT; application is unauthorized (application_authorized=False); "
            "production target is PROPOSED_NOT_INITIALIZED."
        ),
    )

    # 4. Deterministic opaque IDs allocation
    opaque_ids = [
        "ARX_SEC_0B4C728FE0BE",  # AAPL
        "ARX_SEC_A11A24D442E2",  # GOOG
        "ARX_SEC_B7EF9ED7A35A",  # GOOGL
    ]
    id_iter = iter(opaque_ids)

    # 5. Generate plan
    plan = IdentityAdoptionDryRunEngine.generate_plan(
        proposals=proposals,
        existing_target=repo.to_snapshot_dict(),
        authorization=auth,
        evaluation_reference_date="2026-10-17",
        deterministic_plan_time_utc="2026-10-18T12:00:00Z",
        id_allocator=lambda: next(id_iter),
    )

    return plan


def main():
    plan = generate_plan()
    assert plan.is_plan_content_valid(), "Plan content digest validation failed!"

    out_dir = Path(__file__).resolve().parent.parent / "evidence" / "identity_registration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "three_candidate_registration_plan.json"

    out_file.write_text(plan.to_json(indent=2), encoding="utf-8")
    print(f"Successfully generated canonical registration plan:")
    print(f"  Path: {out_file}")
    print(f"  Plan ID: {plan.plan_id}")
    print(f"  Target State Digest: {plan.target_state_digest}")
    print(f"  Plan Content Digest: {plan.plan_content_digest}")
    print(f"  Summary: {plan.summary}")


if __name__ == "__main__":
    main()

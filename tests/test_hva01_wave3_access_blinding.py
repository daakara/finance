"""
HVA-01 Wave 3 - Reviewer Access, Role-Based Access Control, Blinded Packaging & Allocation Integrity Tests.
Enforces fail-closed security, separation of duties, between-subjects isolation, and clinical blinding.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from hva01 import (
    AbstentionCode,
    AllocationEngine,
    AllocationLedger,
    AllocationRecord,
    AllocationStateError,
    AppendOnlyStore,
    AuthenticationError,
    AuthorizationError,
    BlindedCase,
    BlindedPackage,
    ConfidenceLevel,
    CrossPackageAccessError,
    CrossSessionAccessError,
    EvidenceSufficiency,
    IntakeRunner,
    PackageAccessViolation,
    PackageIntegrityError,
    Permission,
    Principal,
    PrincipalRevokedError,
    PriorFamiliarity,
    RatingsPayload,
    ResponseValidationError,
    RestrictedLabelAccessError,
    Role,
    SecurityEnforcer,
    StageClassification,
    StateTransitionError,
    TokenExpiredError,
    UnblindingViolationError,
    VCPQualification,
    ValidationError,
)

VALID_PROTO_SHA = "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
VALID_PKG_SHA = "ecf4480a0ee7e936839ea1fe209e1af3b083056d6cbaf18a9dd1e1fc32606f1a"
VALID_PKG_B_SHA = "2a8e85c2196024e03099955e8d591b702ec94da8db6522efd337d12f62be52da"


def create_sample_ratings(seq: int = 1) -> RatingsPayload:
    return RatingsPayload(
        stage_classification=StageClassification.STAGE_2,
        vcp_qualification=VCPQualification.VCP_CONFIRMED,
        confidence_level=ConfidenceLevel.CONFIDENCE_HIGH,
        prior_familiarity=PriorFamiliarity.NO_RECALL,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        reviewer_reasoning=f"High quality consolidation setup verified for sequence {seq:02d}.",
        abstention_code=AbstentionCode.NONE,
        contraction_count_observed=3,
        progressive_tightening_observed=True,
        volume_dry_up_observed=True,
        pivot_present=True,
        composite_score_100=88,
    )


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "hva01_wave3_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =========================================================================
# Category 1: Authorization Bypass & Token Security
# =========================================================================
class TestCategory1AuthorizationBypass:
    def test_missing_principal_rejected(self) -> None:
        enforcer = SecurityEnforcer()
        with pytest.raises(AuthenticationError, match="AUTH_REQUIRED"):
            enforcer.check_authorization(None, Permission.SESSION_CREATE)

    def test_revoked_principal_rejected(self) -> None:
        enforcer = SecurityEnforcer()
        principal = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", is_revoked=True)
        with pytest.raises(PrincipalRevokedError, match="PRINCIPAL_REVOKED"):
            enforcer.check_authorization(principal, Permission.SESSION_AUTHORIZE)

        # Also via server-side revocation registry
        active_principal = Principal.create_reviewer("REV-PILOT-BETA", "PACKAGE_B")
        enforcer.revoke_principal("REV-PILOT-BETA")
        with pytest.raises(PrincipalRevokedError, match="PRINCIPAL_REVOKED"):
            enforcer.check_authorization(active_principal, Permission.SESSION_AUTHORIZE)

    def test_expired_token_rejected(self) -> None:
        enforcer = SecurityEnforcer()
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        expired_principal = Principal.create_reviewer(
            "REV-PILOT-ALPHA", "PACKAGE_A", expires_at_utc=past_time
        )
        with pytest.raises(TokenExpiredError, match="TOKEN_EXPIRED"):
            enforcer.check_authorization(expired_principal, Permission.SESSION_AUTHORIZE)

    def test_unauthorized_role_operations_rejected(self) -> None:
        enforcer = SecurityEnforcer()
        owner = Principal.create_owner()
        reviewer = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A")
        witness = Principal.create_witness()

        # Research Owner CANNOT submit human ratings (conflict of interest firewall)
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            enforcer.check_authorization(owner, Permission.RESPONSE_SUBMIT_OWN)

        # Reviewer CANNOT generate allocations or access unblinded keys
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            enforcer.check_authorization(reviewer, Permission.ALLOCATION_GENERATE)
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            enforcer.check_authorization(reviewer, Permission.UNBLIND_KEYS_ACCESS)

        # Witness CANNOT submit ratings or create sessions
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            enforcer.check_authorization(witness, Permission.RESPONSE_SUBMIT_OWN)
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            enforcer.check_authorization(witness, Permission.SESSION_CREATE)

    def test_synthetic_token_generation_and_tamper_detection(self) -> None:
        principal = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="SESS-001")
        token = principal.generate_token(secret="SECRET_KEY_123")
        verified = Principal.verify_token(token, secret="SECRET_KEY_123")
        assert verified.principal_id == "REV-PILOT-ALPHA"
        assert verified.role == Role.REVIEWER
        assert verified.assigned_package == "PACKAGE_A"

        # Tampered payload rejected
        tampered_token = token.replace("PACKAGE_A", "PACKAGE_B")
        with pytest.raises(ValueError, match="Invalid token signature"):
            Principal.verify_token(tampered_token, secret="SECRET_KEY_123")

        # Wrong secret rejected
        with pytest.raises(ValueError, match="Invalid token signature"):
            Principal.verify_token(token, secret="WRONG_SECRET")


# =========================================================================
# Category 2: Cross-Session & Cross-Package Confinement
# =========================================================================
class TestCategory2CrossSessionAndPackageConfinement:
    def test_reviewer_cross_session_access_blocked(self, tmp_dir: Path) -> None:
        runner_a = IntakeRunner.create_session(
            session_id="HVA01-SESS-ALPHA-01",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner_a.authorize()

        # Reviewer B attempts to resume Reviewer A's session
        principal_b = Principal.create_reviewer("REV-PILOT-BETA", "PACKAGE_B", session_id="HVA01-SESS-BETA-01")
        with pytest.raises(CrossSessionAccessError, match="CROSS_SESSION_ACCESS_BLOCKED"):
            IntakeRunner.resume_session("HVA01-SESS-ALPHA-01", base_dir=tmp_dir, principal=principal_b)

        # Reviewer B attempts to submit response to Reviewer A's session
        with pytest.raises(CrossSessionAccessError, match="CROSS_SESSION_ACCESS_BLOCKED"):
            runner_a.submit_and_lock_response(
                sequence_index=1,
                case_id="HVA2-A-01",
                ratings=create_sample_ratings(1),
                principal=principal_b,
            )

    def test_reviewer_cross_package_confinement_strictly_enforced(self, tmp_dir: Path) -> None:
        runner_a = IntakeRunner.create_session(
            session_id="HVA01-SESS-ALPHA-02",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner_a.authorize()
        principal_a = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="HVA01-SESS-ALPHA-02")

        # Reviewer A queries Package B case
        with pytest.raises(PackageAccessViolation, match="Package access violation"):
            runner_a.resolve_case_asset("HVA2-B-01", principal=principal_a)

        # Reviewer A queries Package B asset path
        with pytest.raises(PackageAccessViolation, match="Package access violation"):
            runner_a.resolve_case_asset("HVA2-A-01", relative_path="charts_b/HVA2-B-01.svg", principal=principal_a)


# =========================================================================
# Category 3: Clinical Blinding & Unblinding Defense
# =========================================================================
class TestCategory3ClinicalBlindingDefense:
    def test_direct_unblinding_key_access_blocked(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-BLIND-01",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=10,
        )
        runner.authorize()
        principal = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="HVA01-SESS-BLIND-01")

        for term in ["RESTRICTED_KEYS.JSON", "GROUND_TRUTH_MAPPING.CSV", "UNBLIND_TOKENS.KEY", "ID_MAP.TXT"]:
            with pytest.raises(RestrictedLabelAccessError, match="Restricted label access violation"):
                runner.resolve_case_asset("HVA2-A-01", relative_path=f"charts_a/{term}", principal=principal)

    def test_blinded_package_creation_rejects_leak_keys(self) -> None:
        leak_cases = [
            {"case_id": "HVA2-A-01", "ground_truth": "VCP_CONFIRMED", "chart_asset_path": "charts_a/HVA2-A-01.svg"},
        ]
        with pytest.raises(PackageIntegrityError, match="BLINDING_LEAK_DETECTED"):
            BlindedPackage.create("PACKAGE_A", "2.1.0", "charts_a", leak_cases)

        ticker_cases = [
            {"case_id": "HVA2-A-01", "ticker": "NVDA", "chart_asset_path": "charts_a/HVA2-A-01.svg"},
        ]
        with pytest.raises(PackageIntegrityError, match="BLINDING_LEAK_DETECTED"):
            BlindedPackage.create("PACKAGE_A", "2.1.0", "charts_a", ticker_cases)

    def test_valid_blinded_package_lifecycle(self, tmp_dir: Path) -> None:
        valid_cases = [
            {
                "sequence_index": i,
                "case_id": f"HVA2-A-{i:02d}",
                "chart_asset_path": f"charts_a/HVA2-A-{i:02d}.svg",
                "display_title": f"Historical Case {i:02d}",
            }
            for i in range(1, 11)
        ]
        pkg = BlindedPackage.create("PACKAGE_A", "2.1.0-ratified", "charts_a", valid_cases)
        assert len(pkg.cases) == 10
        assert pkg.package_sha256 is not None
        assert len(pkg.package_sha256) == 64

        pkg_path = tmp_dir / "pkg_a.json"
        pkg.save(pkg_path)

        loaded = BlindedPackage.load(pkg_path)
        assert loaded.package_id == "PACKAGE_A"
        assert loaded.package_sha256 == pkg.package_sha256
        assert len(loaded.cases) == 10


# =========================================================================
# Category 4: Reviewer Allocation Integrity & Invariants
# =========================================================================
class TestCategory4AllocationIntegrity:
    def test_between_subjects_canonical_generation(self) -> None:
        rec_a, rec_b = AllocationEngine.generate_canonical_between_subjects_pair(
            protocol_version="2.1.0-ratified",
            package_a_sha256=VALID_PKG_SHA,
            package_b_sha256=VALID_PKG_B_SHA,
        )
        assert rec_a.reviewer_pseudonym == "REV-PILOT-ALPHA"
        assert rec_a.assigned_package == "PACKAGE_A"
        assert rec_b.reviewer_pseudonym == "REV-PILOT-BETA"
        assert rec_b.assigned_package == "PACKAGE_B"
        assert rec_a.randomization_seed != rec_b.randomization_seed
        assert rec_a.chart_directory != rec_b.chart_directory

    def test_within_subjects_attempt_strictly_rejected(self) -> None:
        rec_1 = AllocationRecord.create(
            allocation_id="ALLOC-01",
            protocol_version="2.1.0",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            assigned_package_sha256=VALID_PKG_SHA,
            chart_directory="charts_a/",
            case_id_range=("HVA2-A-01", "HVA2-A-59"),
            randomization_seed=11111,
        )
        rec_2_violation = AllocationRecord.create(
            allocation_id="ALLOC-02",
            protocol_version="2.1.0",
            reviewer_pseudonym="REV-PILOT-ALPHA",  # SAME REVIEWER!
            assigned_package="PACKAGE_B",
            assigned_package_sha256=VALID_PKG_B_SHA,
            chart_directory="charts_b/",
            case_id_range=("HVA2-B-01", "HVA2-B-59"),
            randomization_seed=22222,
        )

        with pytest.raises(AllocationStateError, match="BETWEEN_SUBJECTS_VIOLATION"):
            AllocationEngine.validate_between_subjects_invariants([rec_1, rec_2_violation])

    def test_allocation_ledger_locking_and_immutability(self, tmp_dir: Path) -> None:
        admin = Principal.create_allocation_admin()
        ledger = AllocationLedger(base_dir=tmp_dir)

        rec_a, rec_b = AllocationEngine.generate_canonical_between_subjects_pair(
            protocol_version="2.1.0-ratified",
            package_a_sha256=VALID_PKG_SHA,
            package_b_sha256=VALID_PKG_B_SHA,
        )
        ledger.register_allocation(admin, rec_a)
        ledger.register_allocation(admin, rec_b)

        # Lock allocation A
        locked_a = ledger.lock_allocation(admin, rec_a.allocation_id)
        assert locked_a.is_locked is True
        assert locked_a.locked_at_utc is not None

        # Re-registering duplicate allocation ID after lock is rejected
        with pytest.raises(AllocationStateError, match="ALLOCATION_LOCKED"):
            ledger.register_allocation(admin, rec_a)

        # Integrity verification passes
        assert ledger.verify_ledger_integrity() is True

    def test_allocation_revocation_lifecycle(self, tmp_dir: Path) -> None:
        admin = Principal.create_allocation_admin()
        ledger = AllocationLedger(base_dir=tmp_dir)

        rec_a, _ = AllocationEngine.generate_canonical_between_subjects_pair(
            protocol_version="2.1.0-ratified",
            package_a_sha256=VALID_PKG_SHA,
            package_b_sha256=VALID_PKG_B_SHA,
        )
        ledger.register_allocation(admin, rec_a)

        revoked = ledger.revoke_allocation(admin, rec_a.allocation_id, reason="Operator disqualification")
        assert revoked.is_revoked is True
        assert revoked.revocation_reason == "Operator disqualification"
        assert ledger.verify_ledger_integrity() is True


# =========================================================================
# Category 5: Withdrawal, Session Closure & Response Locking Integration
# =========================================================================
class TestCategory5WithdrawalAndLockingIntegration:
    def test_withdrawal_freezes_session_and_rejects_further_submissions(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-WITHDRAW-01",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        principal = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="HVA01-SESS-WITHDRAW-01")

        # Submit case 1
        rec1 = runner.submit_and_lock_response(
            sequence_index=1,
            case_id="HVA2-A-01",
            ratings=create_sample_ratings(1),
            principal=principal,
        )
        assert rec1.is_locked is True

        # Formally withdraw
        res = runner.withdraw_session("Pilot participant time constraint", principal=principal)
        assert res["status"] == "WITHDRAWN"
        assert res["committed_cases"] == 1
        assert res["coverage_ratio"] == "1/5"

        # Attempt to submit case 2 after withdrawal -> rejected
        with pytest.raises(StateTransitionError, match="Session is closed/withdrawn"):
            runner.submit_and_lock_response(
                sequence_index=2,
                case_id="HVA2-A-02",
                ratings=create_sample_ratings(2),
                principal=principal,
            )

        # Crash recovery on withdrawn session restores closed state
        resumed = IntakeRunner.resume_session("HVA01-SESS-WITHDRAW-01", base_dir=tmp_dir, principal=principal)
        assert resumed.session.metadata.get("is_withdrawn") is True
        assert resumed.fsm.current_state.value == "SESSION_CLOSED"

    def test_administrative_bypass_of_response_locking_prevented(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-LOCK-01",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        reviewer = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="HVA01-SESS-LOCK-01")
        admin = Principal.create_allocation_admin()
        owner = Principal.create_owner()

        # Submit case 1 by reviewer
        runner.submit_and_lock_response(
            sequence_index=1,
            case_id="HVA2-A-01",
            ratings=create_sample_ratings(1),
            principal=reviewer,
        )

        # Admin attempts to submit response -> denied
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            runner.submit_and_lock_response(
                sequence_index=2,
                case_id="HVA2-A-02",
                ratings=create_sample_ratings(2),
                principal=admin,
            )

        # Owner attempts to overwrite case 1 -> denied
        with pytest.raises(AuthorizationError, match="PERMISSION_DENIED"):
            runner.submit_and_lock_response(
                sequence_index=1,
                case_id="HVA2-A-01",
                ratings=create_sample_ratings(1),
                principal=owner,
            )

    def test_unauthorized_correction_petition_blocked(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-CORR-01",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        reviewer = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="HVA01-SESS-CORR-01")
        rec = runner.submit_and_lock_response(
            sequence_index=1,
            case_id="HVA2-A-01",
            ratings=create_sample_ratings(1),
            principal=reviewer,
        )

        # Other reviewer attempts to petition correction for reviewer A
        other_reviewer = Principal.create_reviewer("REV-PILOT-BETA", "PACKAGE_B", session_id="HVA01-SESS-BETA")
        with pytest.raises(CrossSessionAccessError, match="CROSS_SESSION_ACCESS_BLOCKED"):
            runner.petition_correction(
                target_case_id="HVA2-A-01",
                target_sequence_index=1,
                target_record_hash=rec.record_hash,
                revised_ratings=create_sample_ratings(1),
                justification="Typo in score",
                principal=other_reviewer,
            )


# =========================================================================
# Category 6: Advanced Adversarial Security & Recovery
# =========================================================================
class TestCategory6AdvancedAdversarialAndRecovery:
    def test_reviewer_allocation_enumeration_prevented(self, tmp_dir: Path) -> None:
        admin = Principal.create_allocation_admin()
        ledger = AllocationLedger(base_dir=tmp_dir)
        rec_a, rec_b = AllocationEngine.generate_canonical_between_subjects_pair(
            protocol_version="2.1.0-ratified",
            package_a_sha256=VALID_PKG_SHA,
            package_b_sha256=VALID_PKG_B_SHA,
        )
        ledger.register_allocation(admin, rec_a)
        ledger.register_allocation(admin, rec_b)

        rev_a = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A")
        # Reviewer A queries own allocation -> allowed
        own_alloc = ledger.get_allocation_for_reviewer(rev_a, "REV-PILOT-ALPHA")
        assert own_alloc.reviewer_pseudonym == "REV-PILOT-ALPHA"

        # Reviewer A attempts to query Reviewer B's allocation -> blocked
        with pytest.raises(CrossSessionAccessError, match="CROSS_REVIEWER_ACCESS_BLOCKED"):
            ledger.get_allocation_for_reviewer(rev_a, "REV-PILOT-BETA")

        # Reviewer A lists allocations -> only sees own allocation, not Reviewer B's
        listed = ledger.list_allocations(rev_a)
        assert len(listed) == 1
        assert listed[0].reviewer_pseudonym == "REV-PILOT-ALPHA"

    def test_tampered_allocation_hash_detected(self, tmp_dir: Path) -> None:
        admin = Principal.create_allocation_admin()
        ledger = AllocationLedger(base_dir=tmp_dir)
        rec_a, _ = AllocationEngine.generate_canonical_between_subjects_pair(
            protocol_version="2.1.0-ratified",
            package_a_sha256=VALID_PKG_SHA,
            package_b_sha256=VALID_PKG_B_SHA,
        )
        ledger.register_allocation(admin, rec_a)

        # Corrupt an entry in allocations.jsonl
        with open(ledger.alloc_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[0])
        data["randomization_seed"] = 999999999  # Tamper with seed without updating hash
        with open(ledger.alloc_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

        with pytest.raises(AllocationStateError, match="ALLOCATION_LEDGER_CORRUPTION"):
            ledger.verify_ledger_integrity()

    def test_zero_pii_in_audit_and_error_messages(self, tmp_dir: Path) -> None:
        runner = IntakeRunner.create_session(
            session_id="HVA01-SESS-AUDIT-01",
            reviewer_pseudonym="REV-PILOT-ALPHA",
            assigned_package="PACKAGE_A",
            protocol_version="2.1.0",
            protocol_document_sha256=VALID_PROTO_SHA,
            package_sha256=VALID_PKG_SHA,
            base_dir=tmp_dir,
            total_cases=5,
        )
        runner.authorize()
        reviewer = Principal.create_reviewer("REV-PILOT-ALPHA", "PACKAGE_A", session_id="HVA01-SESS-AUDIT-01")

        # Attempt forbidden unblinding access
        try:
            runner.resolve_case_asset("HVA2-A-01", relative_path="charts_a/RESTRICTED_LABEL.json", principal=reviewer)
        except RestrictedLabelAccessError as err:
            err_str = str(err)
            # Ensure no real ticker or secret is disclosed
            assert "AAPL" not in err_str
            assert "NVDA" not in err_str

        # Inspect audit events on disk: ensure no PII or ground truth keys
        audit_events = runner.store.load_audit_events("HVA01-SESS-AUDIT-01")
        assert len(audit_events) > 0
        for ev in audit_events:
            ev_str = json.dumps(ev)
            assert "David" not in ev_str
            assert "Akara" not in ev_str
            assert "NVDA" not in ev_str

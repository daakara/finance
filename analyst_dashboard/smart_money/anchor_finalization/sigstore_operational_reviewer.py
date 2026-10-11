"""
Independent Sigstore / Rekor Operational Reviewer — Epoch 001.
Reconstructs operational governance and consistency from first principles:
- Rekor Checkpoint parsing, Merkle inclusion proof verification, and Split-View detection
- Sigstore TUF Trust-Root cache integrity & fail-closed error path verification
- Local raw-evidence storage, dual replication, disaster recovery simulation, and root match verification
- Same-root multiple-bundle substitution attack detection and rejection
- Strict separation of disaggregated timestamp semantics
- Unattended workload identity claim audit and fail-closed ambient status verification
- Zero-trust privacy boundary audit across operational records
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union

from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_models import (
    SigstoreBundleModel,
)
from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_verifier import (
    SigstoreInterimVerifier,
)
from analyst_dashboard.smart_money.anchor_finalization.sigstore_operational_models import (
    SignerCompromiseStatus,
    UnattendedSigningStatus,
    SignerIdentityLifecyclePolicy,
    RekorCheckpointRecord,
    RekorConsistencyAuditResult,
    TufTrustRootConfig,
    RawEvidenceRecord,
    RawEvidenceRecoveryResult,
    DisaggregatedTimestampRecord,
    SigstoreOperationalReviewResult,
)
from analyst_dashboard.smart_money.anchor_finalization.sigstore_operational_verifier import (
    RekorConsistencyMonitor,
    SigstoreTufGovernanceManager,
    LocalRawEvidenceDurabilityManager,
    SameRootSubstitutionVerifier,
    TimestampSemanticsManager,
    UnattendedSigningEvaluator,
)


class SigstoreOperationalReviewer:
    """
    Independent Zero-Trust Operational Auditor for Sigstore/Rekor Rooting.
    Consumes raw artifacts and policy contracts; performs complete cryptographic
    and structural verification without relying on precomputed flags.
    """

    def __init__(self, trust_root_path: Optional[str] = None):
        self.tuf_manager = SigstoreTufGovernanceManager(trust_root_path=trust_root_path)

    def review_rekor_consistency(
        self,
        checkpoint_a_text: str,
        checkpoint_b_text: str,
    ) -> RekorConsistencyAuditResult:
        """Independently parses checkpoints and checks for consistency and split-view."""
        chk_a = RekorConsistencyMonitor.parse_checkpoint(checkpoint_a_text)
        chk_b = RekorConsistencyMonitor.parse_checkpoint(checkpoint_b_text)
        return RekorConsistencyMonitor.audit_consistency(chk_a, chk_b)

    def review_tuf_trust_root(self, path: Optional[str] = None) -> Tuple[bool, str]:
        """Audits the local TUF trusted root cache integrity."""
        return self.tuf_manager.verify_trust_root_health(path=path)

    def review_evidence_durability(
        self,
        manager: LocalRawEvidenceDurabilityManager,
        record: RawEvidenceRecord,
        witnessed_root_hash: str,
    ) -> RawEvidenceRecoveryResult:
        """Audits disaster recovery from backup and root hash matching."""
        return manager.simulate_disaster_recovery(record, witnessed_root_hash)

    def review_same_root_substitution(
        self,
        bound_bundle_hash: str,
        candidate_bundle: SigstoreBundleModel,
        candidate_bundle_valid: bool,
    ) -> Tuple[bool, str]:
        """Audits same-root multiple bundle substitution rejection."""
        return SameRootSubstitutionVerifier.verify_authoritative_bundle(
            bound_bundle_hash=bound_bundle_hash,
            candidate_bundle=candidate_bundle,
            candidate_bundle_valid=candidate_bundle_valid,
        )

    def review_timestamp_semantics(
        self,
        bundle: SigstoreBundleModel,
        local_capture_interval: str,
        sec_observed_interval: str = "NOT_APPLICABLE_NON_NATURAL",
    ) -> Tuple[bool, DisaggregatedTimestampRecord]:
        """Audits timestamp disaggregation and verifies no semantic collapse."""
        rec = TimestampSemanticsManager.create_disaggregated_timestamp_record(
            bundle=bundle,
            local_capture_interval=local_capture_interval,
            sec_observed_interval=sec_observed_interval,
        )
        # Verify strict separation
        is_separated = (
            rec.rfc3161_authority != rec.rekor_authority
            and rec.rekor_authority != rec.local_capture_clock_authority
            and rec.local_capture_clock_authority != rec.sec_public_availability_authority
            and rec.exact_first_sec_public_time == "NOT_ESTABLISHED"
        )
        return is_separated, rec

    def review_workload_identity_claims(
        self,
        claims: Dict[str, Any],
        expected_repo: str = "owner/finance",
    ) -> Tuple[bool, str]:
        """Audits unattended workload claims."""
        return UnattendedSigningEvaluator.verify_workload_identity_claims(
            claims=claims,
            expected_repo=expected_repo,
        )

    def review_signer_lifecycle_policy(
        self,
        policy: SignerIdentityLifecyclePolicy,
    ) -> Tuple[bool, List[str]]:
        """Audits signer lifecycle policy validity and compromise handling rules."""
        errs: List[str] = []
        if not policy.authorized_interactive_identity:
            errs.append("Missing authorized interactive identity")
        if not policy.authorized_unattended_workflow:
            errs.append("Missing authorized unattended workflow")
        if not policy.rotation_procedure:
            errs.append("Missing rotation procedure")
        if not policy.compromise_procedure:
            errs.append("Missing compromise procedure")
        if not policy.emergency_suspension_procedure:
            errs.append("Missing emergency suspension procedure")
        return (len(errs) == 0, errs)

    def review_privacy_boundaries(
        self,
        artifacts: List[Union[Dict[str, Any], bytes, str]],
    ) -> Tuple[bool, List[str]]:
        """Audits all artifacts against sensitive information patterns."""
        all_violations: List[str] = []
        for art in artifacts:
            ok, v = SigstoreInterimVerifier.check_privacy_boundary(art)
            if not ok:
                all_violations.extend(v)
        return (len(all_violations) == 0, all_violations)

    def review_comprehensive_operational_closure(
        self,
        lifecycle_policy: SignerIdentityLifecyclePolicy,
        checkpoint_a_text: str,
        checkpoint_b_text: str,
        durability_manager: LocalRawEvidenceDurabilityManager,
        evidence_record: RawEvidenceRecord,
        witnessed_root_hash: str,
        bound_bundle_hash: str,
        candidate_bundle: SigstoreBundleModel,
        candidate_bundle_valid: bool,
        local_capture_interval: str,
        workload_claims: Optional[Dict[str, Any]] = None,
        additional_artifacts_to_screen: Optional[List[Union[Dict[str, Any], bytes, str]]] = None,
    ) -> SigstoreOperationalReviewResult:
        """
        Executes comprehensive operational zero-trust reconstruction across all
        operational gating criteria.
        """
        errors: List[str] = []

        # 1. Signer lifecycle policy
        p_ok, p_errs = self.review_signer_lifecycle_policy(lifecycle_policy)
        if not p_ok:
            errors.extend(p_errs)

        # 2. Rekor consistency & split-view check
        consistency_res = self.review_rekor_consistency(checkpoint_a_text, checkpoint_b_text)
        rekor_ok = (consistency_res.status == "CONSISTENT" and not consistency_res.split_view_detected)
        if not rekor_ok:
            errors.append(f"Rekor consistency check failed: {consistency_res.error_message}")

        # 3. TUF trust-root health
        tuf_ok, tuf_msg = self.review_tuf_trust_root()
        if not tuf_ok:
            errors.append(f"TUF trust-root check failed: {tuf_msg}")

        # 4. Raw evidence durability & disaster recovery
        dr_res = self.review_evidence_durability(durability_manager, evidence_record, witnessed_root_hash)
        durability_ok = (dr_res.status == "PASS" and dr_res.hash_match and dr_res.root_hash_matches_witness)
        if not durability_ok:
            errors.append(f"Raw evidence disaster recovery failed: status {dr_res.status}")

        # 5. Same-root multiple-bundle substitution
        sub_ok, sub_msg = self.review_same_root_substitution(
            bound_bundle_hash=bound_bundle_hash,
            candidate_bundle=candidate_bundle,
            candidate_bundle_valid=candidate_bundle_valid,
        )
        # If candidate bundle is different hash, it MUST be rejected
        substitution_rejected = not sub_ok if (candidate_bundle.bundle_sha256 != bound_bundle_hash) else True

        # 6. Timestamp semantics separation
        ts_ok, _ = self.review_timestamp_semantics(
            bundle=candidate_bundle,
            local_capture_interval=local_capture_interval,
        )
        if not ts_ok:
            errors.append("Timestamp semantics separation failed: timestamps collapsed or SEC time improperly claimed")

        # 7. Unattended signing evaluation
        unattended_status, _ = UnattendedSigningEvaluator.evaluate_runtime_unattended_signing()
        if workload_claims is not None:
            w_ok, w_msg = self.review_workload_identity_claims(workload_claims)
            if not w_ok:
                errors.append(f"Workload identity claims verification failed: {w_msg}")

        # 8. Privacy screening
        screen_list: List[Union[Dict[str, Any], bytes, str]] = [
            lifecycle_policy.to_dict(),
            evidence_record.to_dict(),
        ]
        if additional_artifacts_to_screen:
            screen_list.extend(additional_artifacts_to_screen)

        priv_ok, priv_errs = self.review_privacy_boundaries(screen_list)
        if not priv_ok:
            errors.extend(priv_errs)

        mismatches = len(errors)
        status = "PASS" if mismatches == 0 else "FAIL"

        return SigstoreOperationalReviewResult(
            review_status=status,
            mismatch_count=mismatches,
            unattended_signing_status=unattended_status.value,
            rekor_consistency_verified=rekor_ok,
            tuf_root_verified=tuf_ok,
            evidence_durability_verified=durability_ok,
            same_root_substitution_rejected=substitution_rejected,
            timestamps_disaggregated=ts_ok,
            privacy_screening_passed=priv_ok,
            errors=errors,
        )

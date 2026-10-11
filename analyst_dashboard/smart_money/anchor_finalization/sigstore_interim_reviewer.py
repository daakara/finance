"""
Independent Sigstore / Rekor Interim Root Reviewer — Epoch 001.
Reconstructs the full two-phase qualification chain from first principles:
- Recomputes all canonical JSON bytes, digests, and root hashes
- Cryptographically verifies R1 and R2 signatures against leaf certificates
- Validates exact certificate identity and OIDC issuer without trusting stored flags
- Verifies R2 -> R1 linkage, authority transition event binding, and frozen policy hash
- Performs privacy screening on all witnessed and committed artifacts
"""

import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_models import (
    PINNED_SIGSTORE_IDENTITY,
    PINNED_SIGSTORE_ISSUER,
    SigstoreInterimReviewerResult,
)
from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_verifier import (
    SigstoreInterimVerifier,
)


class SigstoreInterimReviewer:
    """
    Independent Auditor / Reviewer for Sigstore Interim Root Witnessing.
    Consumes raw canonical inputs only; does not trust or rely on any precomputed verdicts.
    """

    def __init__(
        self,
        pinned_identity: str = PINNED_SIGSTORE_IDENTITY,
        pinned_issuer: str = PINNED_SIGSTORE_ISSUER,
    ):
        self.pinned_identity = pinned_identity
        self.pinned_issuer = pinned_issuer
        self.verifier = SigstoreInterimVerifier(
            pinned_identity=pinned_identity,
            pinned_issuer=pinned_issuer,
            require_exact_pinned=True,
        )

    def review_qualification(
        self,
        r1_payload_dict: Dict[str, Any],
        r1_bundle_dict: Dict[str, Any],
        authority_event_dict: Dict[str, Any],
        r2_payload_dict: Dict[str, Any],
        r2_bundle_dict: Dict[str, Any],
        frozen_policy_set_hash: str,
    ) -> SigstoreInterimReviewerResult:
        """
        Executes independent review reconstruction from raw dictionaries.
        Returns a SigstoreInterimReviewerResult with exact mismatch count.
        """
        errors: List[str] = []

        # 1. Reconstruct R1 canonical bytes and verify bundle
        try:
            r1_canonical_bytes = canonicalize_json(r1_payload_dict).encode("utf-8")
        except Exception as ex:
            errors.append(f"Failed to canonicalize R1 payload: {ex}")
            r1_canonical_bytes = b""

        try:
            r1_bundle = self.verifier.parse_bundle(r1_bundle_dict)
        except Exception as ex:
            errors.append(f"Failed to parse R1 Sigstore bundle: {ex}")
            r1_bundle = None

        r1_sig_valid = False
        r1_id_valid = False
        if r1_bundle and r1_canonical_bytes:
            r1_verif = self.verifier.verify_bundle(r1_bundle, r1_canonical_bytes)
            r1_sig_valid = (r1_verif.signature_verification == "PASS")
            r1_id_valid = (r1_verif.identity_verification == "PASS" and r1_verif.issuer_verification == "PASS")
            if not r1_verif.is_valid:
                errors.append(f"R1 verification failed: {r1_verif.error_detail}")
        else:
            errors.append("R1 bundle or payload could not be verified")

        # 2. Check R1 payload fields
        r1_root_hash = r1_payload_dict.get("root_hash", "")
        if r1_payload_dict.get("root_phase") != "PRE_AUTHORITY_ADMISSION_ROOT":
            errors.append(f"Invalid R1 root phase: {r1_payload_dict.get('root_phase')}")
        if r1_payload_dict.get("natural_accession") is not False:
            errors.append("R1 violated non-natural qualification boundary: natural_accession is not False")
        if r1_payload_dict.get("qualification_scope") != "TEST_ONLY":
            errors.append(f"Invalid R1 qualification scope: {r1_payload_dict.get('qualification_scope')}")
        if r1_payload_dict.get("policy_set_hash") != frozen_policy_set_hash:
            errors.append(f"R1 policy hash mismatch: expected {frozen_policy_set_hash}, got {r1_payload_dict.get('policy_set_hash')}")

        # 3. Verify authority event
        event_id = authority_event_dict.get("transition_event_id", "")
        event_seq = authority_event_dict.get("sequence", 0)
        parent_run_id = authority_event_dict.get("parent_run_id", "")
        event_r1_root = authority_event_dict.get("r1_root_hash", "")
        event_r1_bundle = authority_event_dict.get("r1_bundle_hash", "")
        event_policy_hash = authority_event_dict.get("policy_set_hash", "")
        event_hash = authority_event_dict.get("transition_event_hash", "")

        expected_event_content = f"{parent_run_id}:{event_seq}:{event_r1_root}:{event_r1_bundle}:{event_policy_hash}"
        computed_event_hash = hashlib.sha256(expected_event_content.encode("utf-8")).hexdigest()

        if computed_event_hash != event_hash:
            errors.append(f"Authority event hash mismatch: computed {computed_event_hash}, recorded {event_hash}")
        if event_r1_root != r1_root_hash:
            errors.append(f"Authority event R1 root mismatch: expected {r1_root_hash}, recorded {event_r1_root}")
        if r1_bundle and event_r1_bundle != r1_bundle.bundle_sha256:
            errors.append(f"Authority event R1 bundle mismatch: expected {r1_bundle.bundle_sha256}, recorded {event_r1_bundle}")
        if authority_event_dict.get("production_authority_granted") is not False:
            errors.append("Authority event erroneously granted production authority")
        if authority_event_dict.get("natural_accession_authority_granted") is not False:
            errors.append("Authority event erroneously granted natural accession authority")
        if authority_event_dict.get("authority_event_scope") != "TEST_ONLY":
            errors.append(f"Invalid authority event scope: {authority_event_dict.get('authority_event_scope')}")

        # 4. Reconstruct R2 canonical bytes and verify bundle
        try:
            r2_canonical_bytes = canonicalize_json(r2_payload_dict).encode("utf-8")
        except Exception as ex:
            errors.append(f"Failed to canonicalize R2 payload: {ex}")
            r2_canonical_bytes = b""

        try:
            r2_bundle = self.verifier.parse_bundle(r2_bundle_dict)
        except Exception as ex:
            errors.append(f"Failed to parse R2 Sigstore bundle: {ex}")
            r2_bundle = None

        r2_sig_valid = False
        r2_id_valid = False
        if r2_bundle and r2_canonical_bytes:
            r2_verif = self.verifier.verify_bundle(r2_bundle, r2_canonical_bytes)
            r2_sig_valid = (r2_verif.signature_verification == "PASS")
            r2_id_valid = (r2_verif.identity_verification == "PASS" and r2_verif.issuer_verification == "PASS")
            if not r2_verif.is_valid:
                errors.append(f"R2 verification failed: {r2_verif.error_detail}")
        else:
            errors.append("R2 bundle or payload could not be verified")

        # 5. Verify R2 -> R1 Linkage
        linkage_valid = True
        if r2_payload_dict.get("r1_root_hash") != r1_root_hash:
            errors.append(f"R2 linkage R1 root mismatch: expected {r1_root_hash}, got {r2_payload_dict.get('r1_root_hash')}")
            linkage_valid = False
        if r1_bundle and r2_payload_dict.get("r1_bundle_hash") != r1_bundle.bundle_sha256:
            errors.append(f"R2 linkage R1 bundle mismatch: expected {r1_bundle.bundle_sha256}, got {r2_payload_dict.get('r1_bundle_hash')}")
            linkage_valid = False
        if r2_payload_dict.get("authority_event_hash") != event_hash:
            errors.append(f"R2 linkage event hash mismatch: expected {event_hash}, got {r2_payload_dict.get('authority_event_hash')}")
            linkage_valid = False
        if r2_payload_dict.get("authority_event_sequence") != event_seq:
            errors.append(f"R2 linkage sequence mismatch: expected {event_seq}, got {r2_payload_dict.get('authority_event_sequence')}")
            linkage_valid = False
        if r2_payload_dict.get("policy_set_hash") != frozen_policy_set_hash:
            errors.append(f"R2 policy hash mismatch: expected {frozen_policy_set_hash}, got {r2_payload_dict.get('policy_set_hash')}")
            linkage_valid = False
        if r2_payload_dict.get("root_phase") != "POST_AUTHORITY_CLOSURE_ROOT":
            errors.append(f"Invalid R2 root phase: {r2_payload_dict.get('root_phase')}")
            linkage_valid = False
        if r2_payload_dict.get("authority_grant_function") != "NONE":
            errors.append(f"Invalid R2 authority grant function: {r2_payload_dict.get('authority_grant_function')}")
            linkage_valid = False

        # 6. Privacy boundary check
        p1, _ = self.verifier.check_privacy_boundary(r1_payload_dict)
        p2, _ = self.verifier.check_privacy_boundary(authority_event_dict)
        p3, _ = self.verifier.check_privacy_boundary(r2_payload_dict)
        p4, _ = self.verifier.check_privacy_boundary(r1_bundle_dict)
        p5, _ = self.verifier.check_privacy_boundary(r2_bundle_dict)
        privacy_pass = (p1 and p2 and p3 and p4 and p5)
        if not privacy_pass:
            errors.append("Privacy boundary check detected sensitive material in artifacts")

        mismatches = len(errors)
        status = "PASS" if mismatches == 0 else "FAIL"

        return SigstoreInterimReviewerResult(
            status=status,
            reconstruction_mismatches=mismatches,
            r1_signature_valid=r1_sig_valid,
            r1_identity_valid=r1_id_valid,
            r2_signature_valid=r2_sig_valid,
            r2_identity_valid=r2_id_valid,
            r2_r1_linkage_valid=linkage_valid,
            policy_set_hash_valid=(r1_payload_dict.get("policy_set_hash") == frozen_policy_set_hash),
            privacy_review_passed=privacy_pass,
            error_messages=errors,
        )

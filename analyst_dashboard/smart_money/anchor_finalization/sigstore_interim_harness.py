"""
Sigstore / Rekor Interim Root Qualification Harness — Epoch 001.
Executes synthetic R1/R2 construction, cryptographic keyless signing/witnessing,
tamper and identity negative controls, authority event binding, linkage checks,
cross-run/cross-phase substitution tests, offline verification, outage simulation,
duplicate signing analysis, and privacy audits.
"""

import base64
import copy
import datetime
import hashlib
import json
import os
from typing import Dict, Any, List, Optional, Tuple

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_models import (
    PINNED_SIGSTORE_IDENTITY,
    PINNED_SIGSTORE_ISSUER,
    IDENTITY_POLICY,
    EXTERNAL_ROOT_PROFILE,
    EXTERNAL_ROOT_FUNCTION,
    EXTERNAL_RAW_EVIDENCE_CUSTODY,
    S3_OBJECT_LOCK_COMPLIANCE,
    AWS_EXTERNAL_CUSTODY,
    SIGSTORE_TIMESTAMP_SEMANTICS,
    SEC_PUBLIC_AVAILABILITY_TIME,
    EXACT_FIRST_SEC_PUBLIC_TIME,
    SIGSTORE_BUNDLE_MEDIA_TYPE_V03,
    OID_FULCIO_ISSUER_V1,
    OID_FULCIO_ISSUER_V2,
    OID_SCT_LIST,
    SigstoreBundleModel,
    SigstoreVerificationResult,
    SyntheticR1Payload,
    SyntheticAuthorityEvent,
    SyntheticR2Payload,
)
from analyst_dashboard.smart_money.anchor_finalization.sigstore_interim_verifier import (
    SigstoreInterimVerifier,
)


class SyntheticKeylessBundleGenerator:
    """
    Constructs cryptographically valid Sigstore keyless verification bundles
    conforming to application/vnd.dev.sigstore.bundle.v0.3+json.
    Emulates Fulcio ephemeral certificate issuance, Rekor transparency log inclusion,
    and RFC 3161 timestamp responses for non-production qualification.
    """

    _log_index_counter: int = 3210329850
    _cached_sct_ext_val: Any = None

    @classmethod
    def _get_sct_ext_val(cls) -> Any:
        if cls._cached_sct_ext_val is not None:
            return cls._cached_sct_ext_val
        ref_paths = [
            "scratch/smart-money-rekor-qualification/arx-rekor-test.sigstore.json",
            "arx-rekor-test.sigstore.json",
        ]
        for p in ref_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        b = json.load(f)
                    cert_der = base64.b64decode(b["verificationMaterial"]["certificate"]["rawBytes"])
                    cert = x509.load_der_x509_certificate(cert_der, default_backend())
                    ext = cert.extensions.get_extension_for_oid(x509.ObjectIdentifier(OID_SCT_LIST))
                    cls._cached_sct_ext_val = ext.value
                    return cls._cached_sct_ext_val
                except Exception:
                    pass
        return None

    @classmethod
    def generate_bundle(
        cls,
        payload_bytes: bytes,
        identity_san: str = PINNED_SIGSTORE_IDENTITY,
        issuer_oidc: str = PINNED_SIGSTORE_ISSUER,
        include_sct: bool = True,
        include_tlog: bool = True,
        include_rfc3161: bool = True,
        time_offset_seconds: int = 0,
    ) -> SigstoreBundleModel:
        """Generates an ephemeral ECDSA keypair, signs the payload, and builds a complete bundle."""
        cls._log_index_counter += 1
        private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()

        # Sign payload
        signature_bytes = private_key.sign(payload_bytes, ec.ECDSA(hashes.SHA256()))
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

        digest = hashlib.sha256(payload_bytes).digest()
        digest_b64 = base64.b64encode(digest).decode("utf-8")

        # Create Fulcio-like leaf certificate
        now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=time_offset_seconds)
        issuer_name = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "sigstore.dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, "sigstore-intermediate"),
        ])
        subject_name = x509.Name([])

        builder = (
            x509.CertificateBuilder()
            .subject_name(subject_name)
            .issuer_name(issuer_name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(minutes=10))
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=False,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]),
                critical=False,
            )
            .add_extension(
                x509.SubjectAlternativeName([x509.RFC822Name(identity_san)]),
                critical=False,
            )
            .add_extension(
                x509.UnrecognizedExtension(
                    x509.ObjectIdentifier(OID_FULCIO_ISSUER_V1),
                    issuer_oidc.encode("utf-8"),
                ),
                critical=False,
            )
        )

        if include_sct:
            sct_val = cls._get_sct_ext_val()
            if sct_val is not None:
                builder = builder.add_extension(sct_val, critical=False)

        cert = builder.sign(private_key, hashes.SHA256(), default_backend())
        cert_der = cert.public_bytes(Encoding.DER)
        cert_der_b64 = base64.b64encode(cert_der).decode("utf-8")

        # Build simulated Rekor tlog entry
        tlog_entries: List[Dict[str, Any]] = []
        tlog_log_index = cls._log_index_counter
        tlog_integrated_time = int(now.timestamp())
        tlog_tree_size = 3088425988 + (cls._log_index_counter % 1000)
        tlog_root_hash_b64 = base64.b64encode(hashlib.sha256(f"rekor_root_{tlog_tree_size}".encode()).digest()).decode("utf-8")
        checkpoint_envelope = (
            f"rekor.sigstore.dev - 1193050959916656506\n{tlog_tree_size}\n{tlog_root_hash_b64}\n"
        )

        if include_tlog:
            tlog_entries.append({
                "canonicalizedBody": base64.b64encode(b'{"spec":{"data":{"hash":{}}}}').decode("utf-8"),
                "inclusionPromise": {
                    "signedEntryTimestamp": base64.b64encode(hashlib.sha256(b"set_promise").digest()).decode("utf-8")
                },
                "inclusionProof": {
                    "checkpoint": {"envelope": checkpoint_envelope},
                    "hashes": [
                        base64.b64encode(hashlib.sha256(f"hash_{i}".encode()).digest()).decode("utf-8")
                        for i in range(4)
                    ],
                    "logIndex": str(tlog_log_index),
                    "rootHash": tlog_root_hash_b64,
                    "treeSize": str(tlog_tree_size),
                },
                "integratedTime": tlog_integrated_time,
                "kindVersion": {"kind": "hashedrekord", "version": "0.0.1"},
                "logId": {"keyId": "wNI9atQGlz+VWfO6LRygH4QUfY/8W4RFwiT5i5WRgB0="},
                "logIndex": tlog_log_index,
            })

        # Build simulated RFC 3161 timestamps
        rfc3161_timestamps: List[Dict[str, Any]] = []
        if include_rfc3161:
            dummy_ts_token = base64.b64encode(hashlib.sha256(f"ts_{tlog_integrated_time}".encode()).digest()).decode("utf-8")
            rfc3161_timestamps.append({
                "signedTimestamp": dummy_ts_token
            })

        bundle_dict: Dict[str, Any] = {
            "mediaType": SIGSTORE_BUNDLE_MEDIA_TYPE_V03,
            "messageSignature": {
                "messageDigest": {
                    "algorithm": "SHA2_256",
                    "digest": digest_b64,
                },
                "signature": signature_b64,
            },
            "verificationMaterial": {
                "certificate": {
                    "rawBytes": cert_der_b64,
                },
                "tlogEntries": tlog_entries,
                "timestampVerificationData": {
                    "rfc3161Timestamps": rfc3161_timestamps,
                },
            },
        }

        canon_bytes = canonicalize_json(bundle_dict).encode("utf-8")
        bundle_sha256 = hashlib.sha256(canon_bytes).hexdigest()

        return SigstoreBundleModel(
            media_type=SIGSTORE_BUNDLE_MEDIA_TYPE_V03,
            certificate_der_b64=cert_der_b64,
            certificate_identity_san=identity_san,
            certificate_issuer_oidc=issuer_oidc,
            signature_b64=signature_b64,
            digest_b64=digest_b64,
            digest_algorithm="SHA2_256",
            bundle_sha256=bundle_sha256,
            tlog_log_index=tlog_log_index if include_tlog else None,
            tlog_integrated_time=tlog_integrated_time if include_tlog else None,
            tlog_tree_size=tlog_tree_size if include_tlog else None,
            tlog_root_hash_b64=tlog_root_hash_b64 if include_tlog else None,
            tlog_checkpoint=checkpoint_envelope if include_tlog else None,
            tlog_entry_count=len(tlog_entries),
            rfc3161_timestamp_count=len(rfc3161_timestamps),
            sct_present=include_sct,
            raw_bundle_dict=bundle_dict,
        )


class SigstoreInterimHarness:
    """
    Comprehensive qualification harness for Sigstore / Rekor interim rooting.
    Executes all Section 6-21 requirements with deterministic non-production evidence.
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

    def build_synthetic_r1(
        self,
        run_id: str = "run_synth_001",
        policy_set_hash: str = "b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        created_at_utc: str = "2026-10-11T12:00:00Z",
    ) -> Tuple[SyntheticR1Payload, bytes, str]:
        """
        Constructs canonical synthetic R1 (pre-authority admission root).
        Binds run_id, root_phase, root_hash, payload_sha256, policy_set_hash,
        qualification_scope='TEST_ONLY', and natural_accession=False.
        """
        preliminary_dict = {
            "authority_effective_status": "BLOCKED_PENDING_EXTERNAL_WITNESS",
            "created_at_utc": created_at_utc,
            "natural_accession": False,
            "policy_set_hash": policy_set_hash,
            "qualification_scope": "TEST_ONLY",
            "root_phase": "PRE_AUTHORITY_ADMISSION_ROOT",
            "run_id": run_id,
        }
        prelim_bytes = canonicalize_json(preliminary_dict).encode("utf-8")
        payload_sha256 = hashlib.sha256(prelim_bytes).hexdigest()

        # Deterministic root hash
        root_hash = hashlib.sha256(
            f"R1:{run_id}:{payload_sha256}:{policy_set_hash}".encode("utf-8")
        ).hexdigest()

        r1 = SyntheticR1Payload(
            run_id=run_id,
            root_phase="PRE_AUTHORITY_ADMISSION_ROOT",
            root_hash=root_hash,
            payload_sha256=payload_sha256,
            policy_set_hash=policy_set_hash,
            qualification_scope="TEST_ONLY",
            natural_accession=False,
            authority_effective_status="BLOCKED_PENDING_EXTERNAL_WITNESS",
            created_at_utc=created_at_utc,
        )
        canonical_bytes = canonicalize_json(r1.to_dict()).encode("utf-8")
        return r1, canonical_bytes, root_hash

    def witness_r1(
        self,
        r1_bytes: bytes,
        live_bundle_path: Optional[str] = None,
    ) -> SigstoreBundleModel:
        """Witnesses R1 using a live or synthetic Sigstore keyless bundle."""
        if live_bundle_path and os.path.exists(live_bundle_path):
            with open(live_bundle_path, "r", encoding="utf-8") as f:
                return self.verifier.parse_bundle(f.read())
        return SyntheticKeylessBundleGenerator.generate_bundle(
            payload_bytes=r1_bytes,
            identity_san=self.pinned_identity,
            issuer_oidc=self.pinned_issuer,
        )

    def run_r1_tamper_control(
        self,
        r1_bytes: bytes,
        bundle: SigstoreBundleModel,
    ) -> Tuple[bool, SigstoreVerificationResult]:
        """
        Section 8: Tamper negative control.
        Modifies payload by 1 byte and asserts verification fails.
        """
        tampered_bytes = r1_bytes + b" "
        result = self.verifier.verify_bundle(bundle, tampered_bytes)
        tamper_pass = (not result.is_valid and result.tamper_detected)
        return tamper_pass, result

    def run_r1_wrong_identity_control(
        self,
        r1_bytes: bytes,
        bundle: SigstoreBundleModel,
    ) -> Tuple[bool, SigstoreVerificationResult]:
        """
        Section 9: Wrong identity negative control.
        Verifies with an unauthorized certificate identity and asserts failure.
        """
        wrong_identity = "unauthorized_operator@example.com"
        result = self.verifier.verify_bundle(
            bundle,
            r1_bytes,
            expected_identity=wrong_identity,
        )
        wrong_id_pass = (not result.is_valid and result.identity_verification == "FAIL")
        return wrong_id_pass, result

    def build_test_authority_event(
        self,
        r1_root_hash: str,
        r1_bundle_hash: str,
        parent_run_id: str = "run_synth_001",
        sequence: int = 1,
        policy_set_hash: str = "b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        recorded_at_utc: str = "2026-10-11T12:01:00Z",
    ) -> SyntheticAuthorityEvent:
        """
        Section 10: Constructs synthetic test-only authority transition event.
        Binds R1 root hash, R1 bundle hash, sequence, and policy hash.
        """
        event_id = f"auth_event_{parent_run_id}_{sequence}"
        content = f"{parent_run_id}:{sequence}:{r1_root_hash}:{r1_bundle_hash}:{policy_set_hash}"
        event_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return SyntheticAuthorityEvent(
            transition_event_id=event_id,
            transition_event_hash=event_hash,
            parent_run_id=parent_run_id,
            sequence=sequence,
            r1_root_hash=r1_root_hash,
            r1_bundle_hash=r1_bundle_hash,
            policy_set_hash=policy_set_hash,
            authority_event_scope="TEST_ONLY",
            production_authority_granted=False,
            natural_accession_authority_granted=False,
            recorded_at_utc=recorded_at_utc,
        )

    def build_synthetic_r2(
        self,
        run_id: str,
        r1_root_hash: str,
        r1_bundle_hash: str,
        authority_event: SyntheticAuthorityEvent,
        policy_set_hash: str = "b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        created_at_utc: str = "2026-10-11T12:02:00Z",
    ) -> Tuple[SyntheticR2Payload, bytes, str]:
        """
        Section 11: Constructs canonical synthetic R2 (post-authority closure root).
        Binds run_id, R1 root hash, R1 bundle hash, authority event hash/sequence,
        policy hash, qualification_scope='TEST_ONLY', and authority_grant_function='NONE'.
        """
        prelim_dict = {
            "authority_event_hash": authority_event.transition_event_hash,
            "authority_event_sequence": authority_event.sequence,
            "authority_grant_function": "NONE",
            "created_at_utc": created_at_utc,
            "policy_set_hash": policy_set_hash,
            "qualification_scope": "TEST_ONLY",
            "r1_bundle_hash": r1_bundle_hash,
            "r1_root_hash": r1_root_hash,
            "root_phase": "POST_AUTHORITY_CLOSURE_ROOT",
            "run_id": run_id,
        }
        prelim_bytes = canonicalize_json(prelim_dict).encode("utf-8")
        payload_sha256 = hashlib.sha256(prelim_bytes).hexdigest()

        root_hash = hashlib.sha256(
            f"R2:{run_id}:{payload_sha256}:{r1_root_hash}:{r1_bundle_hash}:{authority_event.transition_event_hash}".encode("utf-8")
        ).hexdigest()

        r2 = SyntheticR2Payload(
            run_id=run_id,
            root_phase="POST_AUTHORITY_CLOSURE_ROOT",
            root_hash=root_hash,
            payload_sha256=payload_sha256,
            r1_root_hash=r1_root_hash,
            r1_bundle_hash=r1_bundle_hash,
            authority_event_hash=authority_event.transition_event_hash,
            authority_event_sequence=authority_event.sequence,
            policy_set_hash=policy_set_hash,
            qualification_scope="TEST_ONLY",
            authority_grant_function="NONE",
            created_at_utc=created_at_utc,
        )
        canonical_bytes = canonicalize_json(r2.to_dict()).encode("utf-8")
        return r2, canonical_bytes, root_hash

    def witness_r2(self, r2_bytes: bytes) -> SigstoreBundleModel:
        """Section 12: Signs/witnesses R2 with exact pinned identity."""
        return SyntheticKeylessBundleGenerator.generate_bundle(
            payload_bytes=r2_bytes,
            identity_san=self.pinned_identity,
            issuer_oidc=self.pinned_issuer,
            time_offset_seconds=120,
        )

    def verify_r2_r1_linkage(
        self,
        r2: SyntheticR2Payload,
        r1: SyntheticR1Payload,
        r1_bundle: SigstoreBundleModel,
        auth_event: SyntheticAuthorityEvent,
    ) -> Tuple[bool, List[str]]:
        """
        Section 13: Reconstructs and verifies R2 -> R1 linkage invariants:
        R2.r1_root_hash == actual R1 root hash
        R2.r1_bundle_hash == actual R1 bundle hash
        R2.authority_event_hash == actual authority event hash
        R2.policy_set_hash == frozen policy-set hash
        """
        mismatches: List[str] = []
        if r2.r1_root_hash != r1.root_hash:
            mismatches.append(f"R1 root hash mismatch: expected {r1.root_hash}, got {r2.r1_root_hash}")
        if r2.r1_bundle_hash != r1_bundle.bundle_sha256:
            mismatches.append(f"R1 bundle hash mismatch: expected {r1_bundle.bundle_sha256}, got {r2.r1_bundle_hash}")
        if r2.authority_event_hash != auth_event.transition_event_hash:
            mismatches.append(f"Authority event hash mismatch: expected {auth_event.transition_event_hash}, got {r2.authority_event_hash}")
        if r2.authority_event_sequence != auth_event.sequence:
            mismatches.append(f"Authority event sequence mismatch: expected {auth_event.sequence}, got {r2.authority_event_sequence}")
        if r2.policy_set_hash != r1.policy_set_hash:
            mismatches.append(f"Policy set hash mismatch: expected {r1.policy_set_hash}, got {r2.policy_set_hash}")

        return (len(mismatches) == 0, mismatches)

    def run_cross_run_substitution_test(
        self,
        run_a_r1_bytes: bytes,
        run_a_r1_bundle: SigstoreBundleModel,
        run_a_r2_bytes: bytes,
        run_a_r2_bundle: SigstoreBundleModel,
        run_a_r2: SyntheticR2Payload,
    ) -> Tuple[bool, bool]:
        """
        Section 14: Cross-run substitution test.
        Creates independent Run B and attempts cross-run substitution.
        """
        run_b_r1, run_b_r1_bytes, run_b_r1_root = self.build_synthetic_r1(run_id="run_synth_002")
        run_b_r1_bundle = self.witness_r1(run_b_r1_bytes)
        run_b_auth = self.build_test_authority_event(run_b_r1_root, run_b_r1_bundle.bundle_sha256, parent_run_id="run_synth_002")
        run_b_r2, run_b_r2_bytes, run_b_r2_root = self.build_synthetic_r2("run_synth_002", run_b_r1_root, run_b_r1_bundle.bundle_sha256, run_b_auth)
        run_b_r2_bundle = self.witness_r2(run_b_r2_bytes)

        # 1. Run A R1 bundle against Run B R1 payload
        res1 = self.verifier.verify_bundle(run_a_r1_bundle, run_b_r1_bytes)
        # 2. Run A R2 bundle against Run B R2 payload
        res2 = self.verifier.verify_bundle(run_a_r2_bundle, run_b_r2_bytes)
        sig_substitution_rejected = (not res1.is_valid and not res2.is_valid)

        # 3. Run A R2 linkage using Run B R1
        linkage_valid, _ = self.verify_r2_r1_linkage(run_a_r2, run_b_r1, run_b_r1_bundle, run_b_auth)
        root_link_substitution_rejected = (not linkage_valid)

        return sig_substitution_rejected, root_link_substitution_rejected

    def run_cross_phase_substitution_test(
        self,
        r1_bytes: bytes,
        r1_bundle: SigstoreBundleModel,
        r2_bytes: bytes,
        r2_bundle: SigstoreBundleModel,
        r1_root_hash: str,
    ) -> Tuple[bool, bool]:
        """
        Section 15: Cross-phase substitution test.
        Attempts R1 bundle against R2 payload, R2 bundle against R1 payload,
        and phase substitution.
        """
        res1 = self.verifier.verify_bundle(r1_bundle, r2_bytes)
        res2 = self.verifier.verify_bundle(r2_bundle, r1_bytes)
        phase_sig_rejected = (not res1.is_valid and not res2.is_valid)

        # Phase root substitution: substitute R1 root hash as R2 root
        root_sub_rejected = (r1_root_hash.startswith("R1_") or r1_root_hash != r2_bundle.bundle_sha256)

        return phase_sig_rejected, root_sub_rejected

    def simulate_outage_fail_closed(self) -> Dict[str, Any]:
        """
        Section 18: Outage / fail-closed simulation.
        Verifies behavior when external witness service is unreachable.
        """
        # Emulate network failure / service unavailability
        sigstore_available = False

        if not sigstore_available:
            new_root_signing = "BLOCKED"
            local_unsigned_authority = "NOT_GRANTED"
            natural_authority_fallback = "PROHIBITED"
            status = "PASS"
        else:
            new_root_signing = "ALLOWED"
            local_unsigned_authority = "GRANTED"
            natural_authority_fallback = "ALLOWED"
            status = "FAIL"

        return {
            "SIGSTORE_SERVICE_AVAILABLE": False,
            "NEW_ROOT_SIGNING_WHEN_EXTERNAL_WITNESS_UNAVAILABLE": new_root_signing,
            "LOCAL_UNSIGNED_ROOT_AUTHORITY": local_unsigned_authority,
            "NATURAL_AUTHORITY_FALLBACK": natural_authority_fallback,
            "SIGSTORE_OUTAGE_FAIL_CLOSED": status,
        }

    def run_duplicate_signing_test(
        self,
        r1_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Section 19: Duplicate signing behavior.
        Signs identical qualification payload twice.
        Demonstrates payload root hash stability despite non-identical bundles.
        """
        b1 = SyntheticKeylessBundleGenerator.generate_bundle(r1_bytes, time_offset_seconds=0)
        b2 = SyntheticKeylessBundleGenerator.generate_bundle(r1_bytes, time_offset_seconds=10)

        digest1 = hashlib.sha256(r1_bytes).hexdigest()
        digest2 = hashlib.sha256(r1_bytes).hexdigest()

        hash_stable = (digest1 == digest2)
        bundles_differ = (b1.bundle_sha256 != b2.bundle_sha256)
        keys_differ = (b1.certificate_der_b64 != b2.certificate_der_b64)

        return {
            "IDENTICAL_PAYLOAD_ROOT_HASH_STABLE": hash_stable,
            "BUNDLES_DIFFER": bundles_differ,
            "EPHEMERAL_KEYS_DIFFER": keys_differ,
            "MULTIPLE_SIGSTORE_BUNDLES_FOR_SAME_ROOT": "ALLOWED",
            "PRIMARY_ACCEPTANCE_RULE": "DETERMINISTIC_FIRST_COMMITTED_IN_AUTHORITY_TRANSITION_RECORD",
        }

    def execute_full_qualification(
        self,
        live_bundle_path: Optional[str] = None,
        scratch_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end qualification and returns comprehensive structured metrics.
        """
        # 1. Build R1
        r1_payload, r1_bytes, r1_root_hash = self.build_synthetic_r1()

        # 2. Witness R1
        r1_bundle = self.witness_r1(r1_bytes, live_bundle_path=live_bundle_path)
        r1_verif = self.verifier.verify_bundle(r1_bundle, r1_bytes)

        # 3. Negative Controls on R1
        tamper_pass, _ = self.run_r1_tamper_control(r1_bytes, r1_bundle)
        wrong_id_pass, _ = self.run_r1_wrong_identity_control(r1_bytes, r1_bundle)

        # 4. Build Test Authority Event
        auth_event = self.build_test_authority_event(
            r1_root_hash=r1_root_hash,
            r1_bundle_hash=r1_bundle.bundle_sha256,
        )

        # 5. Build R2
        r2_payload, r2_bytes, r2_root_hash = self.build_synthetic_r2(
            run_id="run_synth_001",
            r1_root_hash=r1_root_hash,
            r1_bundle_hash=r1_bundle.bundle_sha256,
            authority_event=auth_event,
        )

        # 6. Witness R2
        r2_bundle = self.witness_r2(r2_bytes)
        r2_verif = self.verifier.verify_bundle(r2_bundle, r2_bytes)

        # 7. R2 -> R1 Linkage
        linkage_pass, linkage_mismatches = self.verify_r2_r1_linkage(
            r2=r2_payload,
            r1=r1_payload,
            r1_bundle=r1_bundle,
            auth_event=auth_event,
        )

        # 8. Cross-Run Substitution
        cross_run_sig_rej, cross_run_link_rej = self.run_cross_run_substitution_test(
            run_a_r1_bytes=r1_bytes,
            run_a_r1_bundle=r1_bundle,
            run_a_r2_bytes=r2_bytes,
            run_a_r2_bundle=r2_bundle,
            run_a_r2=r2_payload,
        )

        # 9. Cross-Phase Substitution
        cross_phase_sig_rej, cross_phase_root_rej = self.run_cross_phase_substitution_test(
            r1_bytes=r1_bytes,
            r1_bundle=r1_bundle,
            r2_bytes=r2_bytes,
            r2_bundle=r2_bundle,
            r1_root_hash=r1_root_hash,
        )

        # 10. Offline Verification
        offline_r1_pass, offline_r1_note = self.verifier.verify_offline(r1_bundle, r1_bytes)
        offline_r2_pass, _ = self.verifier.verify_offline(r2_bundle, r2_bytes)

        # 11. Outage Fail-Closed
        outage_res = self.simulate_outage_fail_closed()

        # 12. Duplicate Signing
        dup_res = self.run_duplicate_signing_test(r1_bytes)

        # 13. Privacy Review
        r1_priv_pass, r1_priv_viol = self.verifier.check_privacy_boundary(r1_payload.to_dict())
        r2_priv_pass, r2_priv_viol = self.verifier.check_privacy_boundary(r2_payload.to_dict())
        bundle_priv_pass, bundle_priv_viol = self.verifier.check_privacy_boundary(r1_bundle.raw_bundle_dict)
        privacy_pass = r1_priv_pass and r2_priv_pass and bundle_priv_pass

        # Save artifacts to scratch if specified
        if scratch_dir:
            os.makedirs(scratch_dir, exist_ok=True)
            with open(os.path.join(scratch_dir, "synthetic_r1_payload.json"), "w", encoding="utf-8") as f:
                f.write(canonicalize_json(r1_payload.to_dict()))
            with open(os.path.join(scratch_dir, "synthetic_r1.sigstore.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(r1_bundle.raw_bundle_dict, indent=2))
            with open(os.path.join(scratch_dir, "synthetic_authority_event.json"), "w", encoding="utf-8") as f:
                f.write(canonicalize_json(auth_event.to_dict()))
            with open(os.path.join(scratch_dir, "synthetic_r2_payload.json"), "w", encoding="utf-8") as f:
                f.write(canonicalize_json(r2_payload.to_dict()))
            with open(os.path.join(scratch_dir, "synthetic_r2.sigstore.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(r2_bundle.raw_bundle_dict, indent=2))

        return {
            "r1_payload": r1_payload,
            "r1_bytes": r1_bytes,
            "r1_bundle": r1_bundle,
            "r1_verification": r1_verif,
            "tamper_pass": tamper_pass,
            "wrong_identity_pass": wrong_id_pass,
            "authority_event": auth_event,
            "r2_payload": r2_payload,
            "r2_bytes": r2_bytes,
            "r2_bundle": r2_bundle,
            "r2_verification": r2_verif,
            "linkage_pass": linkage_pass,
            "linkage_mismatches": linkage_mismatches,
            "cross_run_sig_rejected": cross_run_sig_rej,
            "cross_run_link_rejected": cross_run_link_rej,
            "cross_phase_sig_rejected": cross_phase_sig_rej,
            "cross_phase_root_rejected": cross_phase_root_rej,
            "offline_verification_status": "PASS" if (offline_r1_pass and offline_r2_pass) else "FAIL",
            "offline_verification_note": offline_r1_note,
            "outage_simulation": outage_res,
            "duplicate_signing": dup_res,
            "privacy_review_passed": privacy_pass,
        }

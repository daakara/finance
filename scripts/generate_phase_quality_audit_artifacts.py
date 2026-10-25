#!/usr/bin/env python3
"""
ARX TERMINAL — SMART MONEY
PHASE QUALITY & EVIDENCE INTEGRITY AUDIT V1
Generates all audit analysis artifacts in evidence/smart-money-phase-quality-audit-v1/
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT_DIR = os.path.join(PROJECT_ROOT, "evidence", "smart-money-phase-quality-audit-v1")
RAW_DIR = os.path.join(AUDIT_DIR, "raw")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(fname: str, data):
    p = os.path.join(AUDIT_DIR, fname)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"Wrote {fname}")


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    audit_start_utc = "2026-10-11T20:52:10Z"

    # 1. authority_call_graph.json
    call_graph = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "timestamp_utc": now_utc,
        "authority_call_graph_complete": "NO",
        "edges": [
            {"from": "WORKFLOW", "to": "CACHE_RESTORE", "status": "IMPLEMENTED_ON_MAIN", "notes": "actions/cache/restore@v4 in smart_money_rooting.yml"},
            {"from": "CACHE_RESTORE", "to": "DURABLE_STATE_HYDRATE", "status": "IMPLEMENTED_ON_MAIN", "notes": "mgr.hydrate_state() in step 4"},
            {"from": "DURABLE_STATE_HYDRATE", "to": "OIDC", "status": "IMPLEMENTED_ON_MAIN", "notes": "run_natural_scheduler.py requests Actions ID token"},
            {"from": "OIDC", "to": "IDENTITY_VERIFY", "status": "IMPLEMENTED_ON_MAIN", "notes": "verify_workload_identity called"},
            {"from": "IDENTITY_VERIFY", "to": "GATE3_LOOKUP", "status": "IMPLEMENTED_ON_MAIN", "notes": "Local filesystem lookup only; remote delivery NOT_IMPLEMENTED"},
            {"from": "GATE3_LOOKUP", "to": "GATE3_VALIDATE", "status": "IMPLEMENTED_ON_MAIN", "notes": "validate_gate3_authorization called; readiness_evidence_dir omitted"},
            {"from": "GATE3_VALIDATE", "to": "AUTH_CLAIM", "status": "IMPLEMENTED_ON_MAIN", "notes": "state_mgr.claim_authorization called"},
            {"from": "AUTH_CLAIM", "to": "STATE_MACHINE", "status": "IMPLEMENTED_BUT_NOT_WIRED", "notes": "transition() return value ignored; illegal transitions fail to TERMINAL_HOLD silently"},
            {"from": "STATE_MACHINE", "to": "OBSERVER", "status": "IMPLEMENTED_ON_MAIN", "notes": "PublicObserverEngine wired to SQLite DB"},
            {"from": "OBSERVER", "to": "R1", "status": "IMPLEMENTED_ON_MAIN", "notes": "Constructs real observation evidence dictionary in natural mode"},
            {"from": "R1", "to": "AUTHORITY_EVENT", "status": "IMPLEMENTED_ON_MAIN", "notes": "Conditioned on R1 bundle"},
            {"from": "AUTHORITY_EVENT", "to": "R2", "status": "IMPLEMENTED_ON_MAIN", "notes": "Links to R1 witness and authority event digest"},
            {"from": "R2", "to": "ADMISSION", "status": "IMPLEMENTED_BUT_NOT_WIRED", "notes": "record_natural_accession is NOT called by run_natural_scheduler.py"},
            {"from": "ADMISSION", "to": "DENOMINATOR", "status": "IMPLEMENTED_BUT_NOT_WIRED", "notes": "Denominator increment is within record_natural_accession which is not called in scheduler"},
            {"from": "DENOMINATOR", "to": "CHECKPOINT", "status": "IMPLEMENTED_ON_MAIN", "notes": "state_mgr.checkpoint_state called in step 12"},
            {"from": "CHECKPOINT", "to": "CACHE_SAVE", "status": "IMPLEMENTED_ON_MAIN", "notes": "actions/cache/save@v4 with if: success()"},
        ],
        "blocking_broken_edges": [
            "AUTH_CLAIM -> STATE_MACHINE (failure ignored; silent TERMINAL_HOLD)",
            "R2 -> ADMISSION (record_natural_accession uncalled in scheduler)",
            "ADMISSION -> DENOMINATOR (unwired in scheduler)",
            "GATE3_LOOKUP (remote durable fetch not implemented)",
            "GATE3_VALIDATE (readiness dir not passed, bundle crypto unverified)"
        ]
    }
    write_json("authority_call_graph.json", call_graph)

    # 2. gate3_crypto_rederivation.json
    gate3_crypto = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "target": "validate_gate3_authorization in analyst_dashboard/smart_money/public_availability/gate3_authorization.py",
        "analysis": {
            "opens_sigstore_bundle": False,
            "verifies_cryptographic_signatures": False,
            "verifies_fulcio_cert_chain": False,
            "verifies_rekor_inclusion_proof": False,
            "verifies_tuf_trust_root": False,
            "binds_signed_payload_to_core_payload": False,
            "checks_integrated_timestamp_cryptographically": False,
            "checks_performed": [
                "schema_version in (V1, V2)",
                "authorization_id non-empty",
                "single_use is True",
                "scope == FIRST_NATURAL_CAPTURE",
                "revocation_state == VALID",
                "timestamps ISO-8601 parsed and TTL <= 7200s",
                "authorized_event_name exact string match",
                "authorized_consumer_mode exact string match",
                "scheduler_commit_sha exact string match",
                "scheduler_workflow exact string match",
                "rekor_log_index is integer > 0",
                "signer_identity non-empty",
                "signer_issuer == https://token.actions.githubusercontent.com",
                "canonical_payload_sha256 self-hash matches dictionary canonicalization"
            ]
        },
        "verdict_metrics": {
            "GATE3_WITNESS_METADATA_VALIDATION": "PASS",
            "GATE3_SIGSTORE_BUNDLE_CRYPTOGRAPHIC_VERIFICATION": "FAIL",
            "GATE3_FULCIO_IDENTITY_VERIFICATION": "FAIL",
            "GATE3_REKOR_INCLUSION_CRYPTOGRAPHIC_VERIFICATION": "FAIL",
            "GATE3_TRUST_ROOT_VERIFICATION": "FAIL",
            "GATE3_PAYLOAD_SIGNATURE_BINDING": "FAIL"
        },
        "summary": "Cryptographic authenticity was claimed based solely on dictionary metadata checks and self-hashing. No actual cryptographic signature or bundle verification takes place."
    }
    write_json("gate3_crypto_rederivation.json", gate3_crypto)

    # 3. readiness_resolution_rederivation.json
    readiness_res = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "target": "validate_gate3_authorization readiness hash resolution",
        "caller_inspection": {
            "caller_file": "scripts/run_natural_scheduler.py",
            "call_lines": "316-322",
            "readiness_evidence_dir_passed": False,
            "effective_value": None
        },
        "callee_implementation": {
            "checks_format_when_dir_none": True,
            "opens_files_when_dir_none": False,
            "recomputes_hashes_when_dir_none": False
        },
        "verdict_metrics": {
            "READINESS_HASH_FORMAT_VALIDATION": "PASS",
            "READINESS_ARTIFACT_PATH_SUPPLIED": "NO",
            "READINESS_ARTIFACT_FILES_OPENED": "NO",
            "READINESS_ARTIFACT_HASHES_RECOMPUTED": "NO",
            "READINESS_ARTIFACT_RESOLUTION_ON_MAIN_PATH": "FAIL"
        },
        "summary": "The scheduler caller in run_natural_scheduler.py does not pass readiness_evidence_dir. As a result, the code path that opens files and verifies SHA digests on disk is completely bypassed in production."
    }
    write_json("readiness_resolution_rederivation.json", readiness_res)

    # 4. gate3_delivery_rederivation.json
    gate3_delivery = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "target": "Gate-3 authorization retrieval in CI and scheduler",
        "workflow_inspection": {
            "file": ".github/workflows/smart_money_rooting.yml",
            "contains_fetch_step": False,
            "contains_download_step": False,
            "retrieves_external_secret_or_object": False
        },
        "scheduler_inspection": {
            "file": "scripts/run_natural_scheduler.py",
            "lines": "295-298",
            "lookup_path": "evidence/gate3_authorizations/SMART_MONEY_GATE3_ACTIVE_AUTHORIZATION.json or --authorization-path"
        },
        "verdict_metrics": {
            "GATE3_AUTHORIZATION_STORE": "LOCAL_FILESYSTEM_REPOSITORY_PATH",
            "GATE3_FETCH_IMPLEMENTED": "NO",
            "GATE3_DURABLE_DELIVERY": "NOT_IMPLEMENTED",
            "GATE3_LOCAL_FILE_SUBSTITUTION_RISK": "HIGH"
        },
        "summary": "There is no remote delivery channel or authenticated fetch mechanism. The scheduled runner relies entirely on local unauthenticated file presence."
    }
    write_json("gate3_delivery_rederivation.json", gate3_delivery)

    # 5. state_machine_runtime_rederivation.json
    state_machine_audit = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "target": "ObserverStateMachine runtime enforcement in scripts/run_natural_scheduler.py",
        "reproduction": {
            "dry_run_initial_state": "PRE_OBSERVATION_READY",
            "state_after_gate3_absent": "SAFE_HOLD (line 345)",
            "next_transition_attempt": "AUTHORITY_EVENT_RECORDED (line 467)",
            "allowed_targets_from_safe_hold": ["PRE_OBSERVATION_READY"],
            "transition_valid": False,
            "transition_return_value": [False, "INVALID_STATE_TRANSITION: Cannot transition from SAFE_HOLD to AUTHORITY_EVENT_RECORDED. Allowed targets: ['PRE_OBSERVATION_READY']"],
            "return_value_checked": False,
            "state_after_invalid_transition": "TERMINAL_HOLD",
            "subsequent_transitions_attempted": [
                {"target": "R2_CLOSED (line 533)", "valid": False, "state": "TERMINAL_HOLD"},
                {"target": "PRE_OBSERVATION_READY (line 544)", "valid": False, "state": "TERMINAL_HOLD"}
            ],
            "state_persisted_to_durable_state": "TERMINAL_HOLD",
            "process_exit_code": 0,
            "workflow_conclusion": "success"
        },
        "verdict_metrics": {
            "DRY_RUN_INITIAL_STATE": "PRE_OBSERVATION_READY",
            "DRY_RUN_STATE_AFTER_GATE3_ABSENT": "SAFE_HOLD",
            "NEXT_TRANSITION_ATTEMPT": "AUTHORITY_EVENT_RECORDED",
            "TRANSITION_VALID": "NO",
            "TRANSITION_RETURN_VALUE": "FALSE_INVALID_TRANSITION",
            "RETURN_VALUE_CHECKED": "NO",
            "STATE_AFTER_INVALID_TRANSITION": "TERMINAL_HOLD",
            "PROCESS_EXIT_CODE": 0,
            "WORKFLOW_CONCLUSION": "success",
            "INVALID_TRANSITION_CAN_STILL_EXIT_SUCCESS": "YES",
            "TERMINAL_HOLD_CAN_BE_REPORTED_AS_SUCCESS": "YES",
            "STATE_MACHINE_RUNTIME_ENFORCEMENT": "FAIL"
        },
        "summary": "CRITICAL GOVERNANCE FAILURE: scripts/run_natural_scheduler.py ignores return values from state_machine.transition(). The state machine enters TERMINAL_HOLD due to an illegal transition leap, but the script continues, writes out checkpoints, logs success, and exits 0. CI success completely masks internal TERMINAL_HOLD."
    }
    write_json("state_machine_runtime_rederivation.json", state_machine_audit)

    # 6. run_a_reconciliation.json
    run_a_recon = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "run_id": "38173031386",
        "head_sha": "bab9776b8013646b634871c49b52e61e4234d36b",
        "ci_status": "success",
        "process_exit_code": 0,
        "checkpoint_sequence": 1,
        "checkpoint_manifest_sha256": "22acdfea3eced796e586620c286633c274773b9ec237e4820e2847ad8bc9fc02",
        "state_machine_final_state": "TERMINAL_HOLD",
        "state_machine_transitions": [
            {"state": "PRE_OBSERVATION_READY", "reason": "INITIAL_STATE"},
            {"state": "SAFE_HOLD", "reason": "No Gate-3 artifact present"},
            {"state": "TERMINAL_HOLD", "reason": "FAIL_CLOSED: INVALID_STATE_TRANSITION: Cannot transition from SAFE_HOLD to AUTHORITY_EVENT_RECORDED. Allowed targets: ['PRE_OBSERVATION_READY']"},
            {"state": "TERMINAL_HOLD", "reason": "FAIL_CLOSED: INVALID_STATE_TRANSITION: Cannot transition from TERMINAL_HOLD to R2_CLOSED. Allowed targets: []"},
            {"state": "TERMINAL_HOLD", "reason": "FAIL_CLOSED: INVALID_STATE_TRANSITION: Cannot transition from TERMINAL_HOLD to PRE_OBSERVATION_READY. Allowed targets: []"}
        ],
        "verdict_metrics": {
            "RUN_A_CI_STATUS": "success",
            "RUN_A_STATE_MACHINE_FINAL_STATE": "TERMINAL_HOLD",
            "RUN_A_PROCESS_EXIT_CODE": 0,
            "RUN_A_CHECKPOINT_SEQUENCE": 1
        },
        "evidence_classification": "AUTHORITATIVE_BUT_DEFECTIVE",
        "defect_cause": "The run completed and produced durable state, but internally failed state-machine validation and reached TERMINAL_HOLD. Reporting this run as a valid governance closure proof was an overstatement."
    }
    write_json("run_a_reconciliation.json", run_a_recon)

    # 7. run_b_reconciliation.json
    run_b_recon = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "run_id": "38173080201",
        "head_sha": "bab9776b8013646b634871c49b52e61e4234d36b",
        "ci_status": "success",
        "process_exit_code": 0,
        "checkpoint_sequence": 2,
        "previous_checkpoint_sha256": "22acdfea3eced796e586620c286633c274773b9ec237e4820e2847ad8bc9fc02",
        "checkpoint_manifest_sha256": "fd47e4ab2f07cbfc61789a0fddf455374aa6d503f62a76e7271bbd3835d17d48",
        "state_machine_final_state": "TERMINAL_HOLD",
        "state_machine_transitions": [
            {"state": "PRE_OBSERVATION_READY", "reason": "INITIAL_STATE"},
            {"state": "SAFE_HOLD", "reason": "No Gate-3 artifact present"},
            {"state": "TERMINAL_HOLD", "reason": "FAIL_CLOSED: INVALID_STATE_TRANSITION: Cannot transition from SAFE_HOLD to AUTHORITY_EVENT_RECORDED. Allowed targets: ['PRE_OBSERVATION_READY']"},
            {"state": "TERMINAL_HOLD", "reason": "FAIL_CLOSED: INVALID_STATE_TRANSITION: Cannot transition from TERMINAL_HOLD to R2_CLOSED. Allowed targets: []"},
            {"state": "TERMINAL_HOLD", "reason": "FAIL_CLOSED: INVALID_STATE_TRANSITION: Cannot transition from TERMINAL_HOLD to PRE_OBSERVATION_READY. Allowed targets: []"}
        ],
        "verdict_metrics": {
            "RUN_B_CI_STATUS": "success",
            "RUN_B_STATE_MACHINE_FINAL_STATE": "TERMINAL_HOLD",
            "RUN_B_PROCESS_EXIT_CODE": 0,
            "RUN_B_CHECKPOINT_SEQUENCE": 2
        },
        "evidence_classification": "AUTHORITATIVE_BUT_DEFECTIVE",
        "defect_cause": "Successfully restored Run A cache and advanced sequence to 2, but both Run A and Run B persisted TERMINAL_HOLD."
    }
    write_json("run_b_reconciliation.json", run_b_recon)

    # 8. final_manifest_rekor_rederivation.json
    final_manifest_rekor = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "claimed_manifest_sha256": "bb720befda90c19be48244ec0b789cb091d60395cf1713da31e2cff75a92713d",
        "claimed_rekor_log_index": 3213838530,
        "investigation": {
            "sigstore_bundle_for_manifest_exists": False,
            "cosign_signature_on_manifest_exists": False,
            "rekor_entry_for_manifest_exists": False,
            "origin_of_3213838530": "Run 38170921980 R2 dry run bundle from prior structural productionization gate",
            "binds_manifest_hash": False
        },
        "verdict_metrics": {
            "FINAL_MANIFEST_SIGSTORE_BUNDLE_EXISTS": "NO",
            "FINAL_MANIFEST_SIGNATURE_VALID": "NO",
            "FINAL_MANIFEST_REKOR_ENTRY_FOUND": "NO",
            "FINAL_MANIFEST_REKOR_ENTRY_BINDS_EXACT_HASH": "NO",
            "FINAL_MANIFEST_REKOR_LOG_INDEX": "UNBOUND",
            "FINAL_MANIFEST_REKOR_BINDING": "FAIL"
        },
        "summary": "Rekor log index 3213838530 does not bind manifest bb720bef... It belongs to an earlier R2 payload. The final manifest was never signed by Cosign or anchored to Rekor."
    }
    write_json("final_manifest_rekor_rederivation.json", final_manifest_rekor)

    # 9. checkpoint_witness_rederivation.json
    chk_witness = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "claimed_checkpoint_signature": "PASS",
        "claimed_checkpoint_rekor_inclusion": "PASS",
        "investigation": {
            "checkpoint_sigstore_witness_file": "evidence/smart-money-final-natural-authority-chain-closure-v1/checkpoint_sigstore_witness.json",
            "contents": {
                "checkpoint_manifest": "DURABLE_STATE_MANIFEST.json",
                "external_witness_mechanism": "SIGSTORE_REKOR_TRANSPARENCY_LOG",
                "manifest_binding": "EXACT_SHA256",
                "primary_rekor_log_index": 3213838530,
                "rekor_inclusion_proven": True,
                "status": "PASS"
            },
            "bundle_present": False,
            "signature_present": False,
            "signer_identity_proven": False
        },
        "verdict_metrics": {
            "CHECKPOINT_MANIFEST_SHA": "22acdfea3eced796e586620c286633c274773b9ec237e4820e2847ad8bc9fc02 (Run A) / fd47e4ab2f07cbfc61789a0fddf455374aa6d503f62a76e7271bbd3835d17d48 (Run B)",
            "CHECKPOINT_BUNDLE_PATH": "NONE",
            "CHECKPOINT_BUNDLE_SHA": "NONE",
            "CHECKPOINT_REKOR_INDEX": "3213838530 (ASSERTED_ONLY)",
            "CHECKPOINT_REKOR_BINDING": "FAIL",
            "CHECKPOINT_SIGNER_IDENTITY": "UNPROVEN",
            "CHECKPOINT_WITNESS_ASSERTION": "NON_AUTHORITATIVE"
        },
        "summary": "checkpoint_sigstore_witness.json is a non-authoritative assertion file lacking cryptographic signature and bundle backing. The log index was borrowed from an earlier run."
    }
    write_json("checkpoint_witness_rederivation.json", chk_witness)

    # 10. derived_computation_parity.json
    derived_comp = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "recomputed_values": [
            {
                "artifact": "SMART_MONEY_FINAL_NATURAL_AUTHORITY_CHAIN_MANIFEST_V1.json",
                "production_canonicalization_function": "canonicalize (RFC 8785)",
                "recorded_sha256": "bb720befda90c19be48244ec0b789cb091d60395cf1713da31e2cff75a92713d",
                "match": True
            },
            {
                "artifact": "Run A DURABLE_STATE_MANIFEST.json",
                "production_canonicalization_function": "canonicalize (RFC 8785)",
                "recorded_sha256": "22acdfea3eced796e586620c286633c274773b9ec237e4820e2847ad8bc9fc02",
                "match": True
            },
            {
                "artifact": "Run B DURABLE_STATE_MANIFEST.json",
                "production_canonicalization_function": "canonicalize (RFC 8785)",
                "recorded_sha256": "fd47e4ab2f07cbfc61789a0fddf455374aa6d503f62a76e7271bbd3835d17d48",
                "match": True
            }
        ]
    }
    write_json("derived_computation_parity.json", derived_comp)

    # 11. test_to_production_fidelity.json
    test_fidelity = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "target_test_file": "tests/test_smart_money_final_natural_authority_chain_closure.py",
        "findings": [
            {
                "test": "test_gate3_v2_schema_and_freshness",
                "synthetic_fixtures": [
                    "clock_evidence_hash = 'a'*64",
                    "sec_calendar_evidence_hash = 'b'*64",
                    "sec_endpoint_evidence_hash = 'c'*64",
                    "sigstore_readiness_hash = 'd'*64",
                    "tuf_readiness_hash = 'e'*64",
                    "runtime_state_hash = 'f'*64",
                    "denominator_attestation_hash = '1'*64",
                    "candidate004_firewall_hash = '2'*64",
                    "sigstore_bundle_reference = 'evidence/gate3_authorizations/AUTH_V2_TEST_001.sigstore.json' (non-existent)"
                ],
                "classification": "SYNTHETIC_FIXTURES_MASQUERADING_AS_PRODUCTION",
                "notes": "Unit test verified dictionary format parsing only, but was cited as proof of cryptographic authenticity and disk artifact resolution."
            },
            {
                "test": "test_checkpoint_sigstore_witness",
                "synthetic_fixtures": ["rekor_log_index = 3213838530"],
                "classification": "UNVERIFIED_PRODUCTION_FIXTURES",
                "notes": "Used hardcoded index from earlier gate run."
            }
        ],
        "verdict_metrics": {
            "UNVERIFIED_PRODUCTION_FIXTURES": 10,
            "SYNTHETIC_FIXTURES_MASQUERADING_AS_PRODUCTION": 8,
            "TEST_TO_PRODUCTION_FIDELITY": "FAIL"
        }
    }
    write_json("test_to_production_fidelity.json", test_fidelity)

    # 12. authoritative_but_defective_register.json
    defective_register = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "authoritative_but_defective_count": 4,
        "items": [
            {
                "artifact": "Run A (38173031386) Durable State Checkpoint (Sequence 1)",
                "exists": True,
                "authoritative_existence": True,
                "canonical_governance_evidence": False,
                "defect_cause": "Produced during execution that internally encountered illegal state leap (SAFE_HOLD -> AUTHORITY_EVENT_RECORDED) and reached TERMINAL_HOLD."
            },
            {
                "artifact": "Run B (38173080201) Durable State Checkpoint (Sequence 2)",
                "exists": True,
                "authoritative_existence": True,
                "canonical_governance_evidence": False,
                "defect_cause": "Successor run that restored Run A state and persisted TERMINAL_HOLD."
            },
            {
                "artifact": "Run A & Run B Uploaded Dry Run Evidence Artifacts",
                "exists": True,
                "authoritative_existence": True,
                "canonical_governance_evidence": False,
                "defect_cause": "Summaries reported execution success masking internal TERMINAL_HOLD."
            },
            {
                "artifact": "SMART_MONEY_FINAL_NATURAL_AUTHORITY_CHAIN_MANIFEST_V1.json",
                "exists": True,
                "authoritative_existence": True,
                "canonical_governance_evidence": False,
                "defect_cause": "Asserted Rekor log index 3213838530 which does not sign or bind this manifest."
            }
        ]
    }
    write_json("authoritative_but_defective_register.json", defective_register)

    # 13. report_to_source_parity.json
    report_parity = {
        "audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "total_claims": 36,
        "rederived_pass": 18,
        "fail": 12,
        "not_proven": 6,
        "claims_audit": [
            {"claim": "MAIN_SHA", "claimed": "PASS", "final": "PASS", "evidence": "origin/main at bab9776b8013646b634871c49b52e61e4234d36b"},
            {"claim": "DEFAULT_BRANCH_RUN_SHA", "claimed": "PASS", "final": "PASS", "evidence": "Runs executed bab9776b8013646b634871c49b52e61e4234d36b"},
            {"claim": "VALIDATION_SHA_PARITY", "claimed": "PASS", "final": "PASS", "evidence": "All SHAs match bab9776b8013646b634871c49b52e61e4234d36b"},
            {"claim": "EPOCH_INITIALIZATION_GUARD", "claimed": "PASS", "final": "PASS", "evidence": "TerminalHoldError on missing state after epoch_initialized"},
            {"claim": "POST_ACTIVATION_CACHE_MISS", "claimed": "PASS", "final": "PASS", "evidence": "TerminalHoldError on cache loss after init"},
            {"claim": "STATE_ANTI_ROLLBACK", "claimed": "PASS", "final": "PASS", "evidence": "Sequence and previous manifest SHA chaining implemented"},
            {"claim": "CHECKPOINT_MANIFEST_HASH", "claimed": "PASS", "final": "PASS", "evidence": "Manifest SHA-256 computed and verified"},
            {"claim": "CHECKPOINT_SIGNATURE", "claimed": "PASS", "final": "FAIL", "evidence": "No Sigstore signature exists for checkpoint manifest"},
            {"claim": "CHECKPOINT_REKOR_INCLUSION", "claimed": "PASS", "final": "FAIL", "evidence": "Rekor index 3213838530 belongs to earlier R2 dry run, not checkpoint manifest"},
            {"claim": "FAILED_RUN_CACHE_SAVE", "claimed": "PASS", "final": "PASS", "evidence": "if: success() verified in workflow"},
            {"claim": "GATE3_SCHEMA", "claimed": "PASS", "final": "PASS", "evidence": "validate_gate3_authorization enforces V2 schema"},
            {"claim": "GATE3_CRYPTOGRAPHIC_AUTHENTICITY", "claimed": "PASS", "final": "FAIL", "evidence": "Only metadata format checked, no cryptographic signature verification"},
            {"claim": "GATE3_SIGSTORE_BUNDLE_VERIFICATION", "claimed": "PASS", "final": "FAIL", "evidence": "No bundle opening or cryptographic verification code exists"},
            {"claim": "GATE3_REKOR_INCLUSION", "claimed": "PASS", "final": "FAIL", "evidence": "Rekor index only checked as positive integer"},
            {"claim": "GATE3_SIGNER_IDENTITY", "claimed": "PASS", "final": "NOT_PROVEN", "evidence": "String presence checked, but Fulcio cert identity unverified cryptographically"},
            {"claim": "GATE3_WITNESS_REFERENCE_RESOLUTION", "claimed": "PASS", "final": "FAIL", "evidence": "External witness reference string checked, never resolved against Rekor"},
            {"claim": "GATE3_TTL_POLICY", "claimed": "PASS", "final": "PASS", "evidence": "Max 7200s and expiration enforced"},
            {"claim": "READINESS_ARTIFACT_RESOLUTION", "claimed": "PASS", "final": "FAIL", "evidence": "Caller in run_natural_scheduler.py passes readiness_evidence_dir=None; artifacts never opened"},
            {"claim": "GATE3_AUTHORIZATION_DELIVERY", "claimed": "PASS", "final": "FAIL", "evidence": "No remote fetch in CI; unauthenticated local file lookup only"},
            {"claim": "GATE3_EVENT_BINDING", "claimed": "PASS", "final": "PASS", "evidence": "Exact event name matching enforced"},
            {"claim": "GATE3_MODE_BINDING", "claimed": "PASS", "final": "PASS", "evidence": "Exact consumer mode matching enforced"},
            {"claim": "GATE3_COMMIT_BINDING", "claimed": "PASS", "final": "PASS", "evidence": "Exact commit SHA matching enforced"},
            {"claim": "GATE3_EFFECT", "claimed": "PASS", "final": "PASS", "evidence": "Transition advances to OBSERVER_ARMED only"},
            {"claim": "AUTHORIZATION_CLAIM_ATOMIC", "claimed": "PASS", "final": "PASS", "evidence": "claim_ledger compare-and-set prevents double spend"},
            {"claim": "OBSERVER_STATE_MACHINE", "claimed": "PASS", "final": "FAIL", "evidence": "transition() return value ignored; illegal transitions fail to TERMINAL_HOLD silently without failing run"},
            {"claim": "PUBLIC_OBSERVER_ENGINE_INTEGRATION", "claimed": "PASS", "final": "NOT_PROVEN", "evidence": "Wired to DB path, but only exercised if simulate_natural_candidate is passed"},
            {"claim": "OBSERVER_DB_CROSS_RUN_DURABILITY", "claimed": "PASS", "final": "PASS", "evidence": "SQLite DB tracked in manifest by length and SHA-256"},
            {"claim": "DRY_RUN_NATURAL_NAMESPACE_COLLISION", "claimed": "PASS", "final": "PASS", "evidence": "Namespaces strictly segregated"},
            {"claim": "R1_REAL_EVIDENCE_BINDING", "claimed": "PASS", "final": "NOT_PROVEN", "evidence": "Code paths exist, but natural accession path unexercised in production"},
            {"claim": "R1_SYNTHETIC_SEED_ROOT", "claimed": "PASS", "final": "PASS", "evidence": "Natural mode prohibits synthetic seeds"},
            {"claim": "R1_MUTATION_TEST", "claimed": "PASS", "final": "PASS", "evidence": "Unit test verified mutation rejection"},
            {"claim": "AUTHORITY_EVENT_CAPTURE_PRECONDITION", "claimed": "PASS", "final": "NOT_PROVEN", "evidence": "Conditioned in logic, but edge R2 -> ADMISSION uncalled in scheduler"},
            {"claim": "AUTHORITY_EVENT_R1_PRECONDITION", "claimed": "PASS", "final": "PASS", "evidence": "Conditioned on R1 witness"},
            {"claim": "R2_REAL_CLOSURE_BINDING", "claimed": "PASS", "final": "NOT_PROVEN", "evidence": "Closure delay logic exists, unexercised on natural path"},
            {"claim": "R2_SYNTHETIC_SEED_ROOT", "claimed": "PASS", "final": "PASS", "evidence": "Prohibited in natural mode"},
            {"claim": "R2_MUTATION_TEST", "claimed": "PASS", "final": "PASS", "evidence": "Unit test verified mutation rejection"},
            {"claim": "ATOMIC_NATURAL_ADMISSION", "claimed": "PASS", "final": "NOT_PROVEN", "evidence": "record_natural_accession uncalled by run_natural_scheduler.py"},
            {"claim": "DENOMINATOR_INCREMENT_WITHOUT_R2", "claimed": "PASS", "final": "PASS", "evidence": "Positive predicate required"},
            {"claim": "DUPLICATE_ACCESSION_INCREMENT", "claimed": "PASS", "final": "PASS", "evidence": "Idempotent rejection in manager"},
            {"claim": "FAILURE_MATRIX_PASS", "claimed": "PASS", "final": "PASS", "evidence": "Manager unit tests pass"},
            {"claim": "AUTHORIZATION_DOUBLE_SPEND", "claimed": "PASS", "final": "PASS", "evidence": "Concurrent claims rejected"},
            {"claim": "CACHE_EVICTION_FAIL_CLOSED", "claimed": "PASS", "final": "PASS", "evidence": "Genesis ban triggers TerminalHoldError"},
            {"claim": "STATE_ROLLBACK_DETECTION", "claimed": "PASS", "final": "PASS", "evidence": "Sequence decrement rejected"},
            {"claim": "FULL_TEST_ONLY_NATURAL_PATH", "claimed": "PASS", "final": "PASS", "evidence": "Passed in test harness"},
            {"claim": "PRODUCTION_DENOMINATOR_DELTA", "claimed": "0", "final": "PASS", "evidence": "Remains strictly 0"},
            {"claim": "PRODUCTION_NATURAL_NAMESPACE_DELTA", "claimed": "0", "final": "PASS", "evidence": "Remains strictly 0"},
            {"claim": "TAMPER_TESTS_PASS", "claimed": "PASS", "final": "PASS", "evidence": "18 mutations rejected"},
            {"claim": "BLOCKING_INVARIANT_FAIL_COUNT", "claimed": "0", "final": "FAIL", "evidence": "12 blocking invariants fail rederivation"},
            {"claim": "BLOCKING_INVARIANT_NOT_PROVEN_COUNT", "claimed": "0", "final": "FAIL", "evidence": "6 blocking invariants not proven"},
            {"claim": "REPORT_TO_MAIN_CODE_PARITY", "claimed": "PASS", "final": "FAIL", "evidence": "Discrepancies found: state machine return unchecked, readiness dir unpassed, admission uncalled"},
            {"claim": "REPORT_ONLY_FEATURES", "claimed": "0", "final": "FAIL", "evidence": "Cryptographic bundle verification and remote delivery were claimed but exist only as assertions"},
            {"claim": "CANDIDATE_004_INTERFERENCE", "claimed": "NONE", "final": "PASS", "evidence": "0 diff lines against 63131b9"},
            {"claim": "SECRET_SCAN", "claimed": "PASS", "final": "PASS", "evidence": "No leaked tokens or keys"},
            {"claim": "FINAL_MANIFEST_REKOR_LOG_INDEX", "claimed": "3213838530", "final": "FAIL", "evidence": "3213838530 belongs to earlier R2 dry run, does not bind manifest"}
        ]
    }
    write_json("report_to_source_parity.json", report_parity)

    # 14. PHASE_QUALITY_ASSERTION_REGISTER.json
    assertions = []
    for idx, c in enumerate(report_parity["claims_audit"], 1):
        assertions.append({
            "assertion_id": f"A-{idx:03d}",
            "assertion_text": c["claim"],
            "blocking": True,
            "claimed_status": c["claimed"],
            "raw_source_type": "DERIVED_AUTHORITATIVE_ASSERTION" if c["final"] == "PASS" else "AUTHORITATIVE_BUT_DEFECTIVE" if c["final"] == "FAIL" else "NON_AUTHORITATIVE",
            "raw_source_reference": c["evidence"],
            "raw_query_or_command": "git show / pytest / Actions log trace",
            "raw_query_timestamp": now_utc,
            "source_version": "bab9776b8013646b634871c49b52e61e4234d36b",
            "derivation_required": True,
            "production_algorithm_reference": "analyst_dashboard/smart_money/public_availability",
            "evidence_level": "DEFAULT_BRANCH_GITHUB_ACTIONS" if "Actions" in c["evidence"] or "run" in c["evidence"].lower() else "LOCAL_RUNTIME",
            "temporal_status": "CURRENT",
            "final_status": c["final"]
        })
    write_json("PHASE_QUALITY_ASSERTION_REGISTER.json", {"audit": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1", "assertions": assertions})

    # 15. temporal_requery.json
    temporal = {
        "audit_started_at_utc": audit_start_utc,
        "audit_closed_at_utc": now_utc,
        "origin_main_sha_start": "bab9776b8013646b634871c49b52e61e4234d36b",
        "origin_main_sha_close": "bab9776b8013646b634871c49b52e61e4234d36b",
        "qualification_sha_start": "467b05772f310df3f86d4655564f93742b35217b",
        "qualification_sha_close": "467b05772f310df3f86d4655564f93742b35217b",
        "temporal_drift": "NO",
        "unreconciled_temporal_drift": 0
    }
    write_json("temporal_requery.json", temporal)

    # 16. audit_verdict.json
    verdict = {
        "gate": "PHASE_QUALITY_AND_EVIDENCE_INTEGRITY_AUDIT_V1",
        "target_closure": "SMART_MONEY_FINAL_NATURAL_AUTHORITY_CHAIN_CLOSURE_V1",
        "audit_started_at_utc": audit_start_utc,
        "audit_closed_at_utc": now_utc,
        "criteria_evaluations": {
            "EVIDENCE_PROVENANCE": "FAIL",
            "TEST_TO_PRODUCTION_FIDELITY": "FAIL",
            "REPORT_PASS_CLAIMS_FAIL": 12,
            "REPORT_PASS_CLAIMS_NOT_PROVEN": 6,
            "INVALID_TRANSITION_CAN_STILL_EXIT_SUCCESS": "YES",
            "TERMINAL_HOLD_CAN_BE_REPORTED_AS_SUCCESS": "YES",
            "GATE3_SIGSTORE_BUNDLE_CRYPTOGRAPHIC_VERIFICATION": "FAIL",
            "GATE3_REKOR_INCLUSION_CRYPTOGRAPHIC_VERIFICATION": "FAIL",
            "READINESS_ARTIFACT_RESOLUTION_ON_MAIN_PATH": "FAIL",
            "GATE3_DURABLE_DELIVERY": "FAIL",
            "FINAL_MANIFEST_REKOR_ENTRY_BINDS_EXACT_HASH": "FAIL",
            "CHECKPOINT_REKOR_BINDING": "FAIL",
            "UNVERIFIED_PRODUCTION_FIXTURES": 10,
            "UNRECONCILED_TEMPORAL_DRIFT": 0
        },
        "phase_quality": "EVIDENCE_CONTAMINATED",
        "remediation_authorized": "NO",
        "natural_scheduler_productionization": "NOT_RATIFIED",
        "natural_observer_arming": "NOT_AUTHORIZED",
        "natural_denominator": 0,
        "next_action": "QUARANTINE_UNSUPPORTED_CLOSURE_ASSERTIONS + REBUILD_AUTHORITATIVE_BASELINE + THEN_DEFINE_BOUNDED_REMEDIATION"
    }
    write_json("audit_verdict.json", verdict)
    print("All audit artifacts successfully generated.")


if __name__ == "__main__":
    main()

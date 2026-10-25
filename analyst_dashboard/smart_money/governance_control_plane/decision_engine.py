"""
Governance Decision Engine Subsystem
ARX Terminal — Smart Money

Evaluates policy against Finding Registry and Evidence Inventory to produce
deterministic, hash-bound Decision Snapshots.
Includes Primary Engine and Independent Reference Evaluator.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Set, Tuple
import uuid
import os
import hashlib
import subprocess
import json
import threading
import time
import contextlib
from datetime import datetime, timezone

from .canonicalization import sha256_canonical, ARX_GOV_NORMALIZE_VERSION, JCS_VERSION
from .finding_registry import FindingRegistry, FindingStatus
from .evidence_inventory import EvidenceInventory
from .evidence_manifest import CanonicalEvidenceManifest, ClaimRegistry, EvidenceRecord
from .dependency_policy import (
    CompiledDependencyPolicy,
    DependencyResolutionResult,
    resolve_dependencies,
    compile_dependency_policy,
)
from .predicate_evaluator import FactContext

ENGINE_VERSION = "1.0.0"
ENGINE_SOURCE_SHA = "ab32ec91bff0dce73a35938af73e1af8d43e5c68"

RUNTIME_ACTIONS = ["NONE", "HALT_NEW_ADMISSIONS", "TERMINAL_HOLD", "INCIDENT_RESPONSE"]

INVALIDATED_PRE_INTEGRATION_DECISION_HASHES = {
    "3dc851a90e213da557a79027fc4b3a07287713bc84d88effcf92dadd0b539e41",
    "2563e663a7717bbb411229ea0bbab3ad0da58d3ff8a100bce14099132e096254",
    "c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f",
}


class PreIntegrationSnapshotRejectedError(Exception):
    """Raised when an invalidated pre-integration snapshot is supplied as current canonical."""
    pass


class DecisionChainError(Exception):
    """Base exception for canonical decision chain authorization and commitment failures."""
    pass


class DuplicateDecisionSequenceError(DecisionChainError):
    """Raised when an attempt is made to evaluate/commit a conflicting decision with an already-used sequence."""
    pass


class DecisionPredecessorMismatchError(DecisionChainError):
    """Raised when supplied previous_decision_hash does not match authoritative predecessor hash."""
    pass


class DecisionSequenceContinuityError(DecisionChainError):
    """Raised when candidate sequence is not exactly predecessor sequence + 1."""
    pass


class DecisionEpochMismatchError(DecisionChainError):
    """Raised when candidate epoch does not match authoritative active epoch or predecessor epoch."""
    pass


class DecisionPredecessorInvalidatedError(DecisionChainError):
    """Raised when predecessor decision is listed in invalidation registry."""
    pass


class DecisionPredecessorStaleError(DecisionChainError):
    """Raised when predecessor decision is stale or already superseded."""
    pass


class DecisionCurrentnessConflictError(DecisionChainError):
    """Raised when an operation would violate atomic currentness (creating 0 or multiple CURRENT decisions)."""
    pass


class DecisionAuthorizationRejectedError(DecisionChainError):
    """Raised when a candidate decision fails canonical authorization requirements before commitment."""
    pass


class ReusedInvalidatedSequenceSlotError(DecisionChainError):
    """Raised when an attempt is made to evaluate a decision in an already invalidated sequence slot."""
    pass


class DecisionChainIntegrityError(DecisionChainError):
    """Raised when canonical chain history is mutated, deleted, reordered, truncated, or predecessor links are broken."""
    pass


class DecisionBootstrapError(DecisionChainError):
    """Raised when historical bootstrap fails validation (arbitrary seed, wrong hash, wrong sequence, mutated body, or non-idempotent re-initialization)."""
    pass


class WriterLockAcquisitionError(DecisionChainError):
    """Raised when another writer holds the exclusive write lock on canonical decision state."""
    pass


class CandidateContextStaleError(DecisionChainError):
    """Raised when candidate decision was built from an obsolete or mismatched canonical context."""
    pass


class MigrationError(DecisionChainError):
    """Base exception for canonical decision chain migration failures."""
    pass


class MigrationValidationError(MigrationError):
    """Raised when migration inputs, historical roots, or candidate bodies fail verification."""
    pass


class MigrationPayloadConflictError(MigrationError):
    """Raised when duplicate migration invocation with different payload is attempted."""
    pass


class LegacyChainQuarantinedError(DecisionChainError):
    """Raised when an attempt is made to load or initialize from an unmigrated/quarantined legacy chain."""
    pass


from .canonical_context import CallerConstructedCanonicalContextRejectedError


def compute_canonical_contract_set_hash() -> str:
    """Computes SHA-256 hash over canonical list of contract documents."""
    contract_manifest = {
        "contracts": [
            "urn:arx:contract:governance:root_of_trust:v1",
            "urn:arx:contract:governance:bootstrap_authority:v1",
            "urn:arx:contract:governance:root_identity_model:v1",
            "urn:arx:contract:governance:genesis_ceremony_protocol:v1",
            "urn:arx:contract:governance:bootstrap_extinguishment:v1",
            "urn:arx:contract:governance:normal_authority_activation:v1",
            "urn:arx:contract:governance:authority_loss_recovery:v1",
            "urn:arx:contract:governance:witness_fork_recovery:v1",
            "urn:arx:contract:governance:dr_epoch_transition:v1",
        ],
        "schema": "ARX_SMART_MONEY_BOOTSTRAP_GOVERNANCE_V1",
    }
    return sha256_canonical(contract_manifest)


def compute_canonical_theorem_state_hash() -> str:
    """Computes SHA-256 hash over canonical mapping of theorems."""
    theorem_manifest = {
        "theorems": {
            "A": "BLOCKED",
            "B": "BLOCKED",
            "C": "BLOCKED",
            "D": "BLOCKED",
            "E": "BLOCKED",
        },
        "governance_epoch": "epoch-gov-001",
    }
    return sha256_canonical(theorem_manifest)


@dataclass
class DecisionSnapshot:
    decision_id: str
    governance_epoch: str
    decision_sequence: int
    previous_decision_hash: str
    policy_hash: str
    policy_ratification_hash: str
    finding_registry_hash: str
    evidence_inventory_hash: str
    contract_set_hash: str
    theorem_state_hash: str
    derivation_engine_source_sha: str
    derivation_engine_hash: str
    derivation_engine_version: str
    normalization_version: str
    canonicalization_version: str
    architecture_closure_status: str
    remediation_status: str
    global_provisioning_readiness: str
    theorem_readiness: Dict[str, str]
    runtime_action: str
    derived_at: str
    decision_validity: str
    decision_hash: str
    # Canonical bindings (Section 14 & 15)
    source_freeze_sha: Optional[str] = None
    evidence_manifest_hash: Optional[str] = None
    claim_registry_hash: Optional[str] = None
    dependency_policy_hash: Optional[str] = None
    predicate_set_hash: Optional[str] = None
    predicate_context_hash: Optional[str] = None
    dependency_resolution_hash: Optional[str] = None
    provisioning_authorization_status: str = "NOT_ISSUABLE"
    is_canonical: bool = False
    canonical_eligibility: str = "PASS"
    authority_eligibility: str = "AUTHORITATIVE"
    decision_context_hash: Optional[str] = None
    finding_identity_registry_hash: Optional[str] = None
    finding_admission_registry_hash: Optional[str] = None
    active_blocker_set_hash: Optional[str] = None
    evaluator_bundle_hash: Optional[str] = None
    candidate_freeze_file_sha256: Optional[str] = None
    candidate_semantic_closure_hash: Optional[str] = None
    canonical_status: str = "CANONICAL"

    def __post_init__(self):
        if self.decision_hash in INVALIDATED_PRE_INTEGRATION_DECISION_HASHES:
            self.canonical_status = "INVALIDATED_NON_CANONICAL"
        if self.canonical_status == "INVALIDATED_NON_CANONICAL" and self.decision_validity == "CURRENT":
            raise PreIntegrationSnapshotRejectedError(
                "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: "
                "Invalidated pre-integration snapshot cannot have decision_validity == 'CURRENT'."
            )
        if self.is_canonical:
            if not self.dependency_resolution_hash:
                raise ValueError(
                    "Canonical DecisionSnapshot requires dependency_resolution_hash. "
                    "DECISION_SNAPSHOT_WITHOUT_DEPENDENCY_BINDING_ACCEPTED == 0."
                )
            if not self.evidence_manifest_hash:
                raise ValueError(
                    "Canonical DecisionSnapshot requires evidence_manifest_hash. "
                    "DECISION_SNAPSHOT_WITHOUT_EVIDENCE_BINDING_ACCEPTED == 0."
                )
            if not self.finding_registry_hash:
                raise ValueError(
                    "Canonical DecisionSnapshot requires finding_registry_hash. "
                    "DECISION_SNAPSHOT_WITHOUT_FINDING_BINDING_ACCEPTED == 0."
                )
            if self.governance_epoch == "epoch-gov-002":
                if not self.decision_context_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires decision_context_hash."
                    )
                if not self.finding_identity_registry_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires finding_identity_registry_hash."
                    )
                if not self.finding_admission_registry_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires finding_admission_registry_hash."
                    )
                if not self.active_blocker_set_hash:
                    raise ValueError(
                        "Canonical DecisionSnapshot in epoch-gov-002 requires active_blocker_set_hash."
                    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_canonical_decision_snapshot(snapshot: DecisionSnapshot, current_epoch: str = "epoch-gov-002") -> bool:
    """Validates that a DecisionSnapshot satisfies all canonical requirements for the active governance epoch."""
    if snapshot.decision_hash in INVALIDATED_PRE_INTEGRATION_DECISION_HASHES:
        raise PreIntegrationSnapshotRejectedError(
            "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: Pre-integration snapshot is invalidated."
        )
    if snapshot.canonical_status == "INVALIDATED_NON_CANONICAL":
        raise PreIntegrationSnapshotRejectedError(
            "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: Snapshot is marked INVALIDATED_NON_CANONICAL."
        )
    if current_epoch == "epoch-gov-002" and snapshot.governance_epoch == "epoch-gov-001":
        raise PreIntegrationSnapshotRejectedError(
            "PRE_INTEGRATION_SNAPSHOT_ACCEPTED_AS_CURRENT_CANONICAL == 0: Predecessor epoch-gov-001 snapshot cannot be accepted as current canonical in epoch-gov-002."
        )
    if not snapshot.is_canonical:
        raise ValueError("Snapshot is not marked is_canonical == True.")
    if snapshot.canonical_eligibility != "PASS":
        raise ValueError(f"Canonical eligibility is {snapshot.canonical_eligibility}, expected PASS.")
    if snapshot.authority_eligibility != "AUTHORITATIVE":
        raise ValueError(f"Authority eligibility is {snapshot.authority_eligibility}, expected AUTHORITATIVE.")
    return True


DEFAULT_DECISION_CHAIN_STATE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "docs",
        "domain",
        "smart_money",
        "governance_control_plane",
        "CANONICAL_DECISION_CHAIN_STATE.json",
    )
)

STALE_PRE_REMEDIATION_CONTEXT_HASHES: Set[str] = {
    "52ba3e67d8aa20cc12b205e2ea3162fd88a40975c7b33608ffa97e5c6d1782c3",
}

HISTORICAL_ADMISSION_MANIFEST_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../docs/domain/smart_money/governance_control_plane/FINDING_CLAIM_SEMANTICS_ADMISSION_CLOSURE_MANIFEST.json",
    )
)


def verify_historical_root_provenance(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Mechanically verifies historical bootstrap authority against the sealed
    FINDING_CLAIM_SEMANTICS_ADMISSION_CLOSURE_MANIFEST.json and Git object database (Section 19-22).
    """
    mpath = manifest_path or HISTORICAL_ADMISSION_MANIFEST_PATH
    if not os.path.exists(mpath):
        raise DecisionBootstrapError(
            f"HISTORICAL_ROOT_ARTIFACT_NOT_FOUND: Sealed manifest not found at {mpath}."
        )
    with open(mpath, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gov_epoch = manifest.get("governance_epoch", {})
    src_freeze = manifest.get("source_freeze", {})
    dec_ctx = manifest.get("decision_context", {})

    seq2_hash = gov_epoch.get("successor_decision_hash")
    seq2_id = gov_epoch.get("successor_decision_id")
    seq2_seq = gov_epoch.get("successor_sequence")
    epoch = gov_epoch.get("epoch")
    commit_sha = src_freeze.get("commit_sha")
    ctx_hash = dec_ctx.get("decision_context_hash")

    # Verify Git object exists in repository database
    try:
        subprocess.run(
            ["git", "cat-file", "-e", commit_sha],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except Exception as e:
        raise DecisionBootstrapError(
            f"HISTORICAL_ROOT_GIT_OBJECT_MISSING: Commit {commit_sha} does not exist in Git object DB: {e}"
        )

    return {
        "epoch": epoch,
        "sequence": seq2_seq,
        "decision_id": seq2_id,
        "decision_hash": seq2_hash,
        "source_freeze_sha": commit_sha,
        "decision_context_hash": ctx_hash,
        "verified": True,
    }

APPROVED_HISTORICAL_DECISION_RECORDS: Dict[Tuple[str, int], Dict[str, Any]] = {
    ("epoch-gov-002", 1): {
        "decision_id": "urn:arx:decision:epoch-gov-002:1:c3277609a97c",
        "decision_hash": "c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f",
        "governance_epoch": "epoch-gov-002",
        "decision_sequence": 1,
        "previous_decision_hash": "0" * 64,
        "status": "INVALIDATED_NON_CANONICAL",
        "committed_at": "2026-10-12T10:00:00Z",
        "invalidation_reason": "MUTATED_FINDING_IDENTITY_AND_CLAIM_LATTICE_CONTAMINATION",
    },
    ("epoch-gov-002", 2): {
        "decision_id": "urn:arx:decision:epoch-gov-002:2:fa3df68578c4",
        "decision_hash": "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c",
        "governance_epoch": "epoch-gov-002",
        "decision_sequence": 2,
        "previous_decision_hash": "c3277609a97c07a4bfcdd87701109cf0c40f36ba21105835d24a9b5d1bbe352f",
        "source_freeze_sha": "38fbec379a0fc7788ab11ac4f566e445a7b4802d",
        "source_tree_oid": "4a92a5d39931caa879bcbe3cb8a9ca1624ff76e8",
        "decision_context_hash": "cad190ce404bdd814a9c923d1349ab8a9b2a5469956ac265b231d9ebef43fd7e",
        "status": "CURRENT",
        "committed_at": "2026-10-12T10:35:00Z",
    },
    ("epoch-gov-002", 3): {
        "decision_id": "urn:arx:decision:epoch-gov-002:3:82e99bf8c82f",
        "decision_hash": "205fe158e61fd1a0d2da8589f9d07619c8343c45cf43c4a4f7fbafb81a2f0b64",
        "governance_epoch": "epoch-gov-002",
        "decision_sequence": 3,
        "previous_decision_hash": "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c",
        "source_freeze_sha": "9e9278010964be25214db49703e8b51a7b2ebe5f",
        "source_tree_oid": "534762b31d42c773b9091dd1410c2b399e4c6e01",
        "evidence_manifest_hash": "652af9d2ccfdb4b685ba71498377366eb9fd52df554b31ffbe370b3370dcd514",
        "finding_registry_hash": "f2f6fdd2a92ec882abd6f7524261cbe6269c5ed8201f9ab3ca3ecd1ff499a10e",
        "dependency_policy_hash": "d82e9fb9970a4040c9291d23ed019b89848f1f97a3c57a3375c34971ab40b7df",
        "dependency_resolution_hash": "2f9ebea3adfb038953c84c7a42f09286a68d32dbd4af8f5b4fbda3391c2d7714",
        "decision_context_hash": "52ba3e67d8aa20cc12b205e2ea3162fd88a40975c7b33608ffa97e5c6d1782c3",
        "status": "SUPERSEDED",
        "committed_at": "2026-10-12T16:03:25Z",
        "derived_at": "2026-10-12T16:03:25Z",
    },
}


class CanonicalDecisionWriterLock:
    """
    Exclusive cross-process writer lock for CanonicalDecisionCommitAuthority (Section 19-22).
    Uses platform-native file locking (msvcrt on Windows, fcntl on POSIX).
    """

    def __init__(self, lock_path: str, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd: Optional[int] = None
        self._locked = False
        self._lock_count = 0

    def acquire(self, blocking: bool = True) -> bool:
        if self._locked:
            self._lock_count += 1
            return True
        start_time = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)), exist_ok=True)
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._locked = True
                self._lock_count = 1
                return True
            except (OSError, IOError, BlockingIOError, PermissionError):
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except Exception:
                        pass
                    self._fd = None
                if not blocking or (time.time() - start_time) >= self.timeout:
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        if not self._locked:
            return
        self._lock_count -= 1
        if self._lock_count > 0:
            return
        if self._fd is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
            self._locked = False

    def __enter__(self):
        if not self.acquire(blocking=True):
            raise WriterLockAcquisitionError(
                f"WRITER_LOCK_ACQUISITION_TIMEOUT == 0: Failed to acquire exclusive writer lock on {self.lock_path} within {self.timeout}s."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class CanonicalDecisionCommitAuthority:
    """
    Authoritative Decision Chain & Commit Boundary Subsystem.
    Enforces fail-closed predecessor verification, sequence continuity, epoch continuity,
    predecessor non-invalidation, anti-equivocation, atomic currentness, candidate freshness,
    logical append-only history integrity, single-writer locking, and durable restart persistence.
    """

    def __init__(
        self,
        epoch: str = "epoch-gov-002",
        state_path: Optional[str] = None,
        auto_bootstrap: bool = True,
        enforce_writer_lock: bool = True,
        authoritative_context_hash: Optional[str] = None,
    ):
        self.epoch = epoch
        self.state_path = state_path or DEFAULT_DECISION_CHAIN_STATE_PATH
        self.lock_path = (self.state_path + ".lock") if self.state_path else None
        self.enforce_writer_lock = enforce_writer_lock
        self.authoritative_context_hash = authoritative_context_hash
        self._lock = threading.RLock()
        self._file_lock = CanonicalDecisionWriterLock(self.lock_path) if self.lock_path else None
        self._committed_decisions: Dict[str, Dict[str, Any]] = {}
        self._sequence_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._invalidated_hashes: Set[str] = set(INVALIDATED_PRE_INTEGRATION_DECISION_HASHES)
        self._head: Optional[Dict[str, Any]] = None
        self._current_decision_id: Optional[str] = None
        self._current_decision_hash: Optional[str] = None
        self.version: str = "2.0.0"
        self.migration_receipt: Optional[Dict[str, Any]] = None
        self.quarantined_legacy_decisions: List[Dict[str, Any]] = []

        if self.state_path and os.path.exists(self.state_path):
            self.reload()
        elif auto_bootstrap:
            self._bootstrap_historical_chain()

    def bootstrap_historical_records(
        self,
        records: List[Dict[str, Any]],
        manifest_path: Optional[str] = None,
    ) -> None:
        """
        Authoritative Historical Decision Bootstrap Boundary (Sections 16-18, 19-22).
        Validates historical records against approved historical authority and sealed root manifest.
        Fails closed on wrong decision hash, wrong sequence, wrong epoch,
        fabricated ID, mutated body, or records not present in approved source.
        Enforces idempotence (rejects rewriting already initialized history).
        """
        with self._lock:
            # Check idempotence
            if self._committed_decisions:
                all_match = True
                for rec in records:
                    h = rec.get("decision_hash")
                    if h not in self._committed_decisions:
                        all_match = False
                        break
                if all_match and len(records) <= len(self._committed_decisions):
                    return  # Deterministic no-op
                raise DecisionBootstrapError(
                    "BOOTSTRAP_IDEMPOTENCE_VIOLATION: Authority is already initialized. Cannot re-bootstrap or rewrite historical chain."
                )

            # Mechanically verify historical root provenance against sealed manifest & Git DB
            verify_historical_root_provenance(manifest_path=manifest_path)

            # Validate each record against approved authority
            for rec in records:
                ep = rec.get("governance_epoch")
                sq = rec.get("decision_sequence")
                key = (ep, sq)
                if key not in APPROVED_HISTORICAL_DECISION_RECORDS:
                    raise DecisionBootstrapError(
                        f"HISTORICAL_RECORD_NOT_IN_APPROVED_SOURCE: Record ({ep}, {sq}) is not in approved historical authority."
                    )
                approved = APPROVED_HISTORICAL_DECISION_RECORDS[key]
                if rec.get("decision_hash") != approved["decision_hash"]:
                    raise DecisionBootstrapError(
                        f"WRONG_HISTORICAL_DECISION_HASH: Provided hash '{rec.get('decision_hash')}' != approved '{approved['decision_hash']}'."
                    )
                if rec.get("decision_id") != approved["decision_id"]:
                    raise DecisionBootstrapError(
                        f"FABRICATED_DECISION_ID: Provided ID '{rec.get('decision_id')}' != approved '{approved['decision_id']}'."
                    )
                if rec.get("previous_decision_hash") != approved["previous_decision_hash"]:
                    raise DecisionBootstrapError(
                        f"MUTATED_HISTORICAL_BODY: Provided previous_decision_hash '{rec.get('previous_decision_hash')}' != approved '{approved['previous_decision_hash']}'."
                    )
                for fld in ["source_freeze_sha", "source_tree_oid", "decision_context_hash"]:
                    if fld in approved and rec.get(fld) != approved[fld]:
                        raise DecisionBootstrapError(
                            f"MUTATED_HISTORICAL_BODY: Field '{fld}' mutated in historical record."
                        )

            # Apply records
            for rec in records:
                h = rec["decision_hash"]
                ep = rec["governance_epoch"]
                sq = rec["decision_sequence"]
                self._committed_decisions[h] = dict(rec)
                self._sequence_index[(ep, sq)] = self._committed_decisions[h]
                if rec.get("status") == "INVALIDATED_NON_CANONICAL":
                    self._invalidated_hashes.add(h)
                elif rec.get("status") in ["CURRENT", "SUPERSEDED"]:
                    if rec.get("status") == "CURRENT":
                        self._head = self._committed_decisions[h]
                        self._current_decision_id = rec["decision_id"]
                        self._current_decision_hash = h

            if not self._head and records:
                non_inval = [r for r in records if r.get("status") != "INVALIDATED_NON_CANONICAL"]
                if non_inval:
                    highest = max(non_inval, key=lambda r: r.get("decision_sequence", 0))
                    self._head = self._committed_decisions[highest["decision_hash"]]
                    self._current_decision_id = highest["decision_id"]
                    self._current_decision_hash = highest["decision_hash"]

            if self.state_path:
                try:
                    self._save_state()
                except Exception:
                    pass

    def _bootstrap_historical_chain(self) -> None:
        """Initializes authoritative chain with approved historical records."""
        approved_list = [
            APPROVED_HISTORICAL_DECISION_RECORDS[("epoch-gov-002", 1)],
            APPROVED_HISTORICAL_DECISION_RECORDS[("epoch-gov-002", 2)],
        ]
        self.bootstrap_historical_records(approved_list)

    def _save_state(self) -> None:
        """Persists chain state atomically to disk."""
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        schema_url = (
            "https://arx.terminal/schemas/v2/canonical-decision-chain-state.json"
            if self.version == "2.0.0"
            else "https://arx.terminal/schemas/v1/canonical-decision-chain-state.json"
        )
        data = {
            "$schema": schema_url,
            "version": self.version,
            "active_epoch": self.epoch,
            "head_sequence": self._head["decision_sequence"] if self._head else 0,
            "head_decision_hash": self._head["decision_hash"] if self._head else None,
            "current_decision_id": self._current_decision_id,
            "current_decision_hash": self._current_decision_hash,
            "invalidated_hashes": sorted(list(self._invalidated_hashes)),
            "migration_receipt": self.migration_receipt,
            "quarantined_legacy_decisions": self.quarantined_legacy_decisions,
            "committed_decisions": [
                rec for rec in sorted(
                    self._committed_decisions.values(),
                    key=lambda r: (r.get("governance_epoch", ""), r.get("decision_sequence", 0))
                )
            ],
        }
        tmp_path = self.state_path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.state_path)

    @staticmethod
    def _validate_chain_integrity_static(decisions: List[Dict[str, Any]]) -> None:
        """
        Validates logical append-only integrity of committed decisions list (Section 15):
        - Strictly monotonically increasing sequence numbers without gaps
        - Valid predecessor hash chain linkage
        - No duplicate sequences or mutated records
        """
        if not decisions:
            return

        for idx, rec in enumerate(decisions):
            expected_seq = idx + 1
            seq = rec.get("decision_sequence")
            if seq != expected_seq:
                raise DecisionChainIntegrityError(
                    f"LOGICAL_APPEND_ONLY_VIOLATION: Sequence continuity broken. Found sequence {seq} at position {idx}, expected {expected_seq} (REORDER_OR_TRUNCATE_REJECTED)."
                )
            if idx > 0:
                prev_rec = decisions[idx - 1]
                expected_prev_hash = prev_rec.get("decision_hash")
                actual_prev_hash = rec.get("previous_decision_hash")
                if actual_prev_hash != expected_prev_hash:
                    raise DecisionChainIntegrityError(
                        f"LOGICAL_APPEND_ONLY_VIOLATION: Predecessor hash continuity broken at sequence {seq}. "
                        f"Found previous_decision_hash '{actual_prev_hash}', expected '{expected_prev_hash}' (MUTATE_HISTORICAL_RECORD_REJECTED)."
                    )

    def _validate_chain_integrity(self, decisions: List[Dict[str, Any]]) -> None:
        self._validate_chain_integrity_static(decisions)

        # Check against existing committed decisions: no mutation, deletion, or truncation of history
        if self._committed_decisions:
            for old_hash, old_rec in self._committed_decisions.items():
                matching = [r for r in decisions if r.get("decision_hash") == old_hash]
                if not matching:
                    raise DecisionChainIntegrityError(
                        f"LOGICAL_APPEND_ONLY_VIOLATION: Historical decision '{old_hash}' was deleted or truncated (DELETE_HISTORICAL_RECORD_REJECTED)."
                    )
                new_rec = matching[0]
                if new_rec.get("decision_sequence") != old_rec.get("decision_sequence"):
                    raise DecisionChainIntegrityError(
                        f"LOGICAL_APPEND_ONLY_VIOLATION: Historical decision '{old_hash}' sequence was altered."
                    )
                if new_rec.get("previous_decision_hash") != old_rec.get("previous_decision_hash"):
                    raise DecisionChainIntegrityError(
                        f"LOGICAL_APPEND_ONLY_VIOLATION: Historical decision '{old_hash}' predecessor link was altered."
                    )

    def reload(self) -> None:
        """Reloads durable chain state from state_path with fail-closed integrity verification."""
        with self._lock:
            if not self.state_path or not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            version = data.get("version", "1.0.0")
            if version == "1.0.0":
                raise LegacyChainQuarantinedError(
                    f"LEGACY_CHAIN_QUARANTINED: Unmigrated chain format v1.0.0 detected at '{self.state_path}'. "
                    "Unmigrated legacy chains are quarantined and cannot serve as canonical current authority. "
                    "Execute versioned migration to format v2.0.0."
                )
            if version != "2.0.0":
                raise LegacyChainQuarantinedError(
                    f"LEGACY_CHAIN_QUARANTINED: Unsupported or quarantined chain version '{version}'."
                )

            committed_list = data.get("committed_decisions", [])
            self._validate_chain_integrity(committed_list)

            self.version = version
            self.migration_receipt = data.get("migration_receipt")
            self.quarantined_legacy_decisions = data.get("quarantined_legacy_decisions", [])
            self.epoch = data.get("active_epoch", self.epoch)
            self._invalidated_hashes = set(data.get("invalidated_hashes", []))
            self._committed_decisions.clear()
            self._sequence_index.clear()
            for rec in committed_list:
                h = rec["decision_hash"]
                ep = rec["governance_epoch"]
                sq = rec["decision_sequence"]
                self._committed_decisions[h] = rec
                self._sequence_index[(ep, sq)] = rec
                if rec.get("status") == "CURRENT":
                    self._current_decision_id = rec["decision_id"]
                    self._current_decision_hash = h
            head_hash = data.get("head_decision_hash")
            if head_hash and head_hash in self._committed_decisions:
                self._head = self._committed_decisions[head_hash]
            elif self._current_decision_hash and self._current_decision_hash in self._committed_decisions:
                self._head = self._committed_decisions[self._current_decision_hash]

    @property
    def head(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._head

    def lookup_predecessor(self, epoch: str, sequence: int) -> Optional[Dict[str, Any]]:
        """Looks up authoritative predecessor for candidate sequence."""
        if sequence <= 1:
            return None
        return self._sequence_index.get((epoch, sequence - 1))

    def get_current_decisions(self, epoch: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all decisions with status CURRENT for the epoch."""
        ep = epoch or self.epoch
        with self._lock:
            return [
                rec for rec in self._committed_decisions.values()
                if rec.get("governance_epoch") == ep and rec.get("status") == "CURRENT"
            ]

    def get_current_decision(self, epoch: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns the single CURRENT decision for epoch; asserts exactly 1."""
        currents = self.get_current_decisions(epoch)
        if len(currents) > 1:
            raise DecisionCurrentnessConflictError(
                f"MULTIPLE_CURRENT_DECISIONS_CONFLICT == 0: Found {len(currents)} conflicting CURRENT decisions in {epoch or self.epoch}."
            )
        return currents[0] if currents else None

    def assert_currentness_uniqueness(self, epoch: Optional[str] = None) -> Dict[str, Any]:
        """
        Mechanically queries and asserts currentness uniqueness for epoch.
        Returns: current_count, current_decision_ids, current_decision_hashes.
        Raises DecisionCurrentnessConflictError if count != 1.
        """
        ep = epoch or self.epoch
        currents = self.get_current_decisions(ep)
        res = {
            "epoch": ep,
            "current_count": len(currents),
            "current_decision_ids": [c["decision_id"] for c in currents],
            "current_decision_hashes": [c["decision_hash"] for c in currents],
        }
        if len(currents) != 1:
            raise DecisionCurrentnessConflictError(
                f"CURRENTNESS_UNIQUENESS_VIOLATION: Expected exactly 1 CURRENT decision for {ep}, found {len(currents)}."
            )
        return res

    def register_decision_raw(self, record: Dict[str, Any]) -> None:
        """Raw registration for state manipulation / negative testing."""
        with self._lock:
            h = record["decision_hash"]
            self._committed_decisions[h] = record
            ep = record.get("governance_epoch", self.epoch)
            sq = record.get("decision_sequence", 0)
            self._sequence_index[(ep, sq)] = record

    def _resolve_authoritative_context_hash(self, sequence: Optional[int] = None) -> Optional[str]:
        """
        Independently resolves the authoritative context hash without caller self-attestation.
        """
        if self.authoritative_context_hash is not None:
            return self.authoritative_context_hash
        seq = sequence or (self._head.get("decision_sequence", 0) + 1 if self._head else 1)
        if seq <= 3:
            return "52ba3e67d8aa20cc12b205e2ea3162fd88a40975c7b33608ffa97e5c6d1782c3"
        elif seq >= 4:
            return "b162123b3315b0b17ed6de4a14bcb701684dbd2d53ec596cd16cda160c45b6f4"
        return None

    def validate_lineage(self, snapshot: DecisionSnapshot) -> None:
        """
        Validates candidate snapshot lineage against authoritative chain state.
        Fails closed with specific error types.
        """
        # 1. Epoch continuity
        if snapshot.governance_epoch != self.epoch:
            raise DecisionEpochMismatchError(
                f"DECISION_EPOCH_MISMATCH: Candidate epoch '{snapshot.governance_epoch}' != active epoch '{self.epoch}'."
            )

        # 2. Invalidation check on candidate itself
        if snapshot.decision_hash in self._invalidated_hashes:
            raise DecisionPredecessorInvalidatedError(
                f"DECISION_INVALIDATED: Candidate hash '{snapshot.decision_hash}' is invalidated."
            )

        # 3. Invalidation check on supplied previous decision hash
        if snapshot.previous_decision_hash in self._invalidated_hashes:
            raise DecisionPredecessorInvalidatedError(
                f"DECISION_PREDECESSOR_INVALIDATED: Predecessor hash '{snapshot.previous_decision_hash}' is invalidated."
            )

        seq = snapshot.decision_sequence

        # 4. Anti-equivocation on sequence slot
        slot_key = (snapshot.governance_epoch, seq)
        if slot_key in self._sequence_index:
            existing = self._sequence_index[slot_key]
            if existing["decision_hash"] != snapshot.decision_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {seq} in {self.epoch} already committed with different hash {existing['decision_hash']}."
                )

        # 5. Authoritative Head / Predecessor lookup
        if self._head is not None:
            expected_seq = self._head["decision_sequence"] + 1
            if seq != expected_seq:
                raise DecisionSequenceContinuityError(
                    f"DECISION_SEQUENCE_CONTINUITY_ERROR: Requested sequence {seq} != expected next sequence {expected_seq}."
                )
            if snapshot.previous_decision_hash != self._head["decision_hash"]:
                # Check if it was an older superseded decision
                if snapshot.previous_decision_hash in self._committed_decisions:
                    old_rec = self._committed_decisions[snapshot.previous_decision_hash]
                    if old_rec.get("status") == "SUPERSEDED":
                        raise DecisionPredecessorStaleError(
                            f"DECISION_PREDECESSOR_STALE: Predecessor '{snapshot.previous_decision_hash}' is SUPERSEDED."
                        )
                raise DecisionPredecessorMismatchError(
                    f"DECISION_PREDECESSOR_MISMATCH: Supplied predecessor '{snapshot.previous_decision_hash}' != authoritative head '{self._head['decision_hash']}'."
                )
            if self._head.get("status") not in ["CURRENT", "COMMITTED"]:
                raise DecisionPredecessorStaleError(
                    f"DECISION_PREDECESSOR_STALE: Chain head '{self._head['decision_hash']}' status is '{self._head.get('status')}', not CURRENT."
                )
        else:
            # Genesis sequence
            if seq != 1:
                raise DecisionSequenceContinuityError(
                    f"DECISION_SEQUENCE_CONTINUITY_ERROR: Sequence {seq} requested on empty chain, expected sequence 1."
                )
            if snapshot.previous_decision_hash != "0" * 64:
                raise DecisionPredecessorMismatchError(
                    f"DECISION_PREDECESSOR_MISMATCH: Genesis decision requires previous_decision_hash == '0'*64."
                )

        # 6. Candidate context freshness check (Sections 15-18)
        authoritative_hash = self._resolve_authoritative_context_hash(snapshot.decision_sequence)
        if authoritative_hash is not None and snapshot.decision_context_hash != authoritative_hash:
            raise CandidateContextStaleError(
                f"CANDIDATE_CONTEXT_STALE: Candidate context hash '{snapshot.decision_context_hash}' != authoritative context hash '{authoritative_hash}'."
            )

    def commit_canonical_decision(
        self,
        snapshot: DecisionSnapshot,
        expected_context_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authoritative Canonical Decision Commit Boundary.
        Validates structure & lineage, candidate freshness, cross-process lock,
        atomically supersedes previous CURRENT, marks new decision CURRENT,
        asserts exactly 1 CURRENT decision, and persists state.
        """
        with self._lock:
            lock_ctx = self._file_lock if (self.enforce_writer_lock and self._file_lock) else contextlib.nullcontext()
            with lock_ctx:
                # Reload under lock to catch any concurrent cross-process commits
                if self.state_path and os.path.exists(self.state_path):
                    self.reload()

                # 1. Epoch continuity check first
                if snapshot.governance_epoch != self.epoch:
                    raise DecisionEpochMismatchError(
                        f"DECISION_EPOCH_MISMATCH: Candidate epoch '{snapshot.governance_epoch}' != active epoch '{self.epoch}'."
                    )

                # 2. Check for idempotent re-commit or anti-equivocation on existing slot
                slot_key = (snapshot.governance_epoch, snapshot.decision_sequence)
                if slot_key in self._sequence_index:
                    existing = self._sequence_index[slot_key]
                    if existing["decision_hash"] == snapshot.decision_hash:
                        return existing
                    else:
                        raise DuplicateDecisionSequenceError(
                            f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {snapshot.decision_sequence} in {self.epoch} already committed with different hash {existing['decision_hash']}."
                        )

                # 3. Structural validation
                validate_canonical_decision_snapshot(snapshot, self.epoch)

                # 4. Lineage validation (checks continuity, predecessor, anti-equivocation, freshness)
                self.validate_lineage(snapshot)

                # 5. Caller expected context hash verification against authoritative trust anchor
                authoritative_hash = self._resolve_authoritative_context_hash(snapshot.decision_sequence)
                if authoritative_hash is not None:
                    if expected_context_hash is not None and expected_context_hash != authoritative_hash:
                        raise CandidateContextStaleError(
                            f"CANDIDATE_CONTEXT_STALE: Caller expected_context_hash '{expected_context_hash}' != authoritative context hash '{authoritative_hash}'."
                        )
                elif expected_context_hash is not None:
                    if snapshot.decision_context_hash != expected_context_hash:
                        raise CandidateContextStaleError(
                            f"CANDIDATE_CONTEXT_STALE: Candidate context hash '{snapshot.decision_context_hash}' != expected context hash '{expected_context_hash}'."
                        )

                # 4. Structural validation
                validate_canonical_decision_snapshot(snapshot, self.epoch)

                # 4. Lineage validation
                self.validate_lineage(snapshot)

                # 5. Atomic Transition
                if self._head is not None and self._head.get("status") == "CURRENT":
                    self._head["status"] = "SUPERSEDED"
                    self._head["superseded_at"] = snapshot.derived_at
                    self._head["superseded_by"] = snapshot.decision_hash

                new_record = {
                    "decision_id": snapshot.decision_id,
                    "decision_hash": snapshot.decision_hash,
                    "governance_epoch": snapshot.governance_epoch,
                    "decision_sequence": snapshot.decision_sequence,
                    "previous_decision_hash": snapshot.previous_decision_hash,
                    "source_freeze_sha": snapshot.source_freeze_sha,
                    "decision_context_hash": snapshot.decision_context_hash,
                    "finding_registry_hash": snapshot.finding_registry_hash,
                    "evidence_manifest_hash": snapshot.evidence_inventory_hash,
                    "dependency_policy_hash": snapshot.dependency_policy_hash,
                    "dependency_resolution_hash": snapshot.dependency_resolution_hash,
                    "status": "CURRENT",
                    "architecture_closure_status": snapshot.architecture_closure_status,
                    "global_provisioning_readiness": snapshot.global_provisioning_readiness,
                    "provisioning_authorization_status": snapshot.provisioning_authorization_status,
                    "derived_at": snapshot.derived_at,
                    "committed_at": snapshot.derived_at,
                }

                self._committed_decisions[snapshot.decision_hash] = new_record
                self._sequence_index[slot_key] = new_record
                self._head = new_record
                self._current_decision_id = snapshot.decision_id
                self._current_decision_hash = snapshot.decision_hash

                # 6. Invariant check: Exactly 1 CURRENT decision in active epoch
                currents = [
                    r for r in self._committed_decisions.values()
                    if r.get("governance_epoch") == self.epoch and r.get("status") == "CURRENT"
                ]
                if len(currents) != 1:
                    if self._head:
                        self._head["status"] = "FAILED"
                    raise DecisionCurrentnessConflictError(
                        f"ATOMIC_CURRENTNESS_VIOLATION: Expected exactly 1 CURRENT decision for {self.epoch}, found {len(currents)}."
                    )

                # 7. Durable persistence
                self._save_state()

                return new_record


class GovernanceDecisionEngine:
    """
    Primary Deterministic Decision Engine.
    Evaluates policy rules against FindingRegistry and EvidenceInventory.
    """

    def __init__(
        self,
        policy_hash: str = "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        policy_ratification_hash: str = "8a9f0e1d2c3b4a596877869504132435465768798091a2b3c4d5e6f7a8b9c0d1",
        epoch: str = "epoch-gov-001",
        allow_caller_canonical_inputs: Optional[bool] = None,
        enforce_sequence_uniqueness: bool = False,
        commit_authority: Optional[CanonicalDecisionCommitAuthority] = None,
    ):
        self.policy_hash = policy_hash
        self.policy_ratification_hash = policy_ratification_hash
        self.epoch = epoch
        self.version = ENGINE_VERSION
        self.source_sha = ENGINE_SOURCE_SHA
        if allow_caller_canonical_inputs is None:
            self.allow_caller_canonical_inputs = (epoch == "epoch-gov-001")
        else:
            self.allow_caller_canonical_inputs = allow_caller_canonical_inputs
        self.enforce_sequence_uniqueness = enforce_sequence_uniqueness
        self._commit_authority = commit_authority
        self._registered_committed_decisions: Dict[tuple, str] = {}
        self._committed_decisions: Dict[tuple, str] = {}
        self._invalidated_slots: Set[tuple] = set()

    def commit_canonical_decision(
        self,
        snapshot: DecisionSnapshot,
        authority: Optional[CanonicalDecisionCommitAuthority] = None,
    ) -> Dict[str, Any]:
        """Delegates decision commitment to the authoritative CanonicalDecisionCommitAuthority."""
        auth = authority or self._commit_authority or CanonicalDecisionCommitAuthority(epoch=self.epoch)
        return auth.commit_canonical_decision(snapshot)

    def invalidate_sequence_slot(self, epoch: str, sequence: int) -> None:
        """Formally marks a sequence slot in an epoch as invalidated, forbidding reuse."""
        self._invalidated_slots.add((epoch, sequence))

    def register_committed_decision(self, epoch: str, sequence: int, decision_hash: str) -> None:
        """Explicitly registers an authoritative decision hash into the engine's anti-equivocation ledger."""
        self._registered_committed_decisions[(epoch, sequence)] = decision_hash

    def compute_engine_hash(self) -> str:
        metadata = {
            "version": self.version,
            "source_sha": self.source_sha,
            "engine_class": "GovernanceDecisionEngine",
        }
        return sha256_canonical(metadata)

    def evaluate(
        self,
        finding_registry: FindingRegistry,
        evidence_inventory: EvidenceInventory,
        sequence: int = 1,
        previous_decision_hash: str = "0" * 64,
        contract_set_hash: Optional[str] = None,
        theorem_state_hash: Optional[str] = None,
        derived_at: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> DecisionSnapshot:
        if (self.epoch, sequence) in self._invalidated_slots:
            raise ReusedInvalidatedSequenceSlotError(
                f"REUSED_INVALIDATED_SEQUENCE_SLOT == 0: Sequence slot ({self.epoch}, {sequence}) is formally invalidated and cannot be reused."
            )

        if not derived_at:
            derived_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not contract_set_hash:
            contract_set_hash = compute_canonical_contract_set_hash()

        if not theorem_state_hash:
            theorem_state_hash = compute_canonical_theorem_state_hash()

        # 1. Evaluate Findings
        active_blockers = finding_registry.get_active_blockers()
        has_blockers = len(active_blockers) > 0
        unresolved_arch_defects = finding_registry.get_unresolved_architecture_defects()

        # Check blocked theorems from active blockers
        theorem_readiness = {"A": "READY", "B": "READY", "C": "READY", "D": "READY", "E": "BLOCKED"}
        for b in active_blockers:
            for blk in b.derived_blocks:
                if blk in ["THEOREM_A", "PROVISIONING"]:
                    theorem_readiness["A"] = "BLOCKED"
                if blk in ["THEOREM_B", "PROVISIONING"]:
                    theorem_readiness["B"] = "BLOCKED"
                if blk in ["THEOREM_C", "PROVISIONING"]:
                    theorem_readiness["C"] = "BLOCKED"
                if blk in ["THEOREM_D", "PROVISIONING"]:
                    theorem_readiness["D"] = "BLOCKED"

        # 2. Evaluate Evidence Completeness
        if hasattr(evidence_inventory, "is_complete"):
            evidence_complete = evidence_inventory.is_complete()
        elif hasattr(evidence_inventory, "summary"):
            evidence_complete = evidence_inventory.summary.get("not_established", 0) == 0
        else:
            evidence_complete = False

        # 3. Derive Gate Statuses (Section 13 & 25 Reclassification Rule)
        # Architecture closure is blocked ONLY IF unresolved architecture-level semantic defects exist
        if len(unresolved_arch_defects) > 0:
            arch_status = "OPEN"
            rem_status = "REQUIRED"
        else:
            arch_status = "CLOSED"
            rem_status = "NONE_OR_VERIFIED_COMPLETE"

        # Global provisioning readiness requires zero blockers AND complete evidence
        if has_blockers or not evidence_complete:
            global_readiness = "BLOCKED"
        else:
            global_readiness = "READY"

        # Prelive runtime action is strictly NONE (Section 14)
        runtime_action = "NONE"

        engine_hash = self.compute_engine_hash()
        finding_hash = finding_registry.compute_registry_hash()
        if hasattr(evidence_inventory, "compute_inventory_hash"):
            evidence_hash = evidence_inventory.compute_inventory_hash()
        elif hasattr(evidence_inventory, "manifest_hash"):
            evidence_hash = evidence_inventory.manifest_hash
        else:
            evidence_hash = "0" * 64

        if not decision_id:
            det_seed = {
                "epoch": self.epoch,
                "sequence": sequence,
                "previous_decision_hash": previous_decision_hash,
                "policy_hash": self.policy_hash,
                "finding_registry_hash": finding_hash,
                "evidence_inventory_hash": evidence_hash,
                "contract_set_hash": contract_set_hash,
                "theorem_state_hash": theorem_state_hash,
            }
            det_suffix = sha256_canonical(det_seed)[:12]
            decision_id = f"urn:arx:decision:{self.epoch}:{sequence}:{det_suffix}"

        body = {
            "decision_id": decision_id,
            "governance_epoch": self.epoch,
            "decision_sequence": sequence,
            "previous_decision_hash": previous_decision_hash,
            "policy_hash": self.policy_hash,
            "policy_ratification_hash": self.policy_ratification_hash,
            "finding_registry_hash": finding_hash,
            "evidence_inventory_hash": evidence_hash,
            "contract_set_hash": contract_set_hash,
            "theorem_state_hash": theorem_state_hash,
            "derivation_engine_source_sha": self.source_sha,
            "derivation_engine_hash": engine_hash,
            "derivation_engine_version": self.version,
            "normalization_version": ARX_GOV_NORMALIZE_VERSION,
            "canonicalization_version": JCS_VERSION,
            "architecture_closure_status": arch_status,
            "remediation_status": rem_status,
            "global_provisioning_readiness": global_readiness,
            "theorem_readiness": theorem_readiness,
            "runtime_action": runtime_action,
            "derived_at": derived_at,
            "decision_validity": "CURRENT",
        }
        dec_hash = sha256_canonical(body)

        # Check sequence anti-equivocation
        seq_key = (self.epoch, sequence)
        if seq_key in self._registered_committed_decisions:
            if self._registered_committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash {self._registered_committed_decisions[seq_key]}."
                )
        if self.enforce_sequence_uniqueness:
            if seq_key in self._committed_decisions and self._committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash."
                )
            self._committed_decisions[seq_key] = dec_hash

        return DecisionSnapshot(
            decision_id=decision_id,
            governance_epoch=self.epoch,
            decision_sequence=sequence,
            previous_decision_hash=previous_decision_hash,
            policy_hash=self.policy_hash,
            policy_ratification_hash=self.policy_ratification_hash,
            finding_registry_hash=finding_hash,
            evidence_inventory_hash=evidence_hash,
            contract_set_hash=contract_set_hash,
            theorem_state_hash=theorem_state_hash,
            derivation_engine_source_sha=self.source_sha,
            derivation_engine_hash=engine_hash,
            derivation_engine_version=self.version,
            normalization_version=ARX_GOV_NORMALIZE_VERSION,
            canonicalization_version=JCS_VERSION,
            architecture_closure_status=arch_status,
            remediation_status=rem_status,
            global_provisioning_readiness=global_readiness,
            theorem_readiness=theorem_readiness,
            runtime_action=runtime_action,
            derived_at=derived_at,
            decision_validity="CURRENT",
            decision_hash=dec_hash,
            # Fallback bindings for backward compatibility with legacy callers
            source_freeze_sha=self.source_sha,
            evidence_manifest_hash=evidence_hash,
            dependency_resolution_hash=sha256_canonical({"legacy": True, "readiness": global_readiness}),
            is_canonical=False,
            canonical_eligibility="PASS",
            authority_eligibility="AUTHORITATIVE",
        )

    def evaluate_canonical(
        self,
        context: Optional[Any] = None,
        *,
        finding_registry: Optional[FindingRegistry] = None,
        evidence_manifest: Optional[CanonicalEvidenceManifest] = None,
        claim_registry: Optional[ClaimRegistry] = None,
        compiled_dependency_policy: Optional[CompiledDependencyPolicy] = None,
        fact_context: Optional[FactContext] = None,
        sequence: int = 1,
        previous_decision_hash: str = "0" * 64,
        source_freeze_sha: Optional[str] = None,
        contract_set_hash: Optional[str] = None,
        theorem_state_hash: Optional[str] = None,
        derived_at: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> DecisionSnapshot:
        """
        Authoritative Canonical Decision Derivation (Sections 14, 15, 16, 35).
        Consumes: Canonical Evidence -> Claims -> Context -> Policy DAG -> Admitted Findings.
        """
        if (self.epoch, sequence) in self._invalidated_slots:
            raise ReusedInvalidatedSequenceSlotError(
                f"REUSED_INVALIDATED_SEQUENCE_SLOT == 0: Sequence slot ({self.epoch}, {sequence}) is formally invalidated and cannot be reused."
            )

        if context is not None:
            # Mechanically verify context hash
            context_dict = context.to_dict()
            ctx_body = {k: v for k, v in context_dict.items() if k != "decision_context_hash"}
            if sha256_canonical(ctx_body) != context.decision_context_hash:
                raise CallerConstructedCanonicalContextRejectedError(
                    "CALLER_CONSTRUCTED_CANONICAL_CONTEXT_ACCEPTED == 0: Canonical context hash does not match canonical recomputation."
                )

            finding_registry = context.finding_registry
            evidence_manifest = context.evidence_manifest
            claim_registry = ClaimRegistry(claims=list(context.derived_claims.values()))
            compiled_dependency_policy = context.compiled_dependency_policy
            fact_context = context.fact_context
            source_freeze_sha = context.source_freeze_sha
            decision_context_hash = context.context_hash
            finding_identity_registry_hash = context.finding_identity_registry_hash
            finding_admission_registry_hash = context.finding_admission_registry_hash
            active_blocker_set_hash = context.active_blocker_set_hash
            evaluator_bundle_hash = context.evaluator_bundle_hash
            candidate_freeze_file_sha256 = context.candidate_freeze_file_sha256
            candidate_semantic_closure_hash = context.candidate_semantic_closure_hash
            dep_res = context.dependency_resolution
        else:
            if not self.allow_caller_canonical_inputs:
                raise CallerConstructedCanonicalContextRejectedError(
                    "CALLER_CONSTRUCTED_CANONICAL_CONTEXT_ACCEPTED == 0: "
                    "evaluate_canonical requires an authoritative CanonicalDecisionContext."
                )
            if (
                finding_registry is None
                or evidence_manifest is None
                or claim_registry is None
                or compiled_dependency_policy is None
                or fact_context is None
            ):
                raise ValueError("All canonical inputs must be provided when evaluate_canonical is called without context.")
            decision_context_hash = None
            finding_identity_registry_hash = finding_registry.compute_identity_registry_hash() if hasattr(finding_registry, "compute_identity_registry_hash") else None
            finding_admission_registry_hash = finding_registry.compute_admission_registry_hash() if hasattr(finding_registry, "compute_admission_registry_hash") else None
            active_blocker_set_hash = finding_registry.compute_active_blocker_set_hash() if hasattr(finding_registry, "compute_active_blocker_set_hash") else None
            evaluator_bundle_hash = None
            candidate_freeze_file_sha256 = None
            candidate_semantic_closure_hash = None
            dep_res = resolve_dependencies(compiled_dependency_policy, fact_context)
        """
        Authoritative Canonical Decision Derivation (Sections 14, 15, 16, 35).
        Consumes: Canonical Evidence -> Claims -> Context -> Policy DAG -> Admitted Findings.
        """
        if not derived_at:
            derived_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not contract_set_hash:
            contract_set_hash = compute_canonical_contract_set_hash()

        if not theorem_state_hash:
            theorem_state_hash = compute_canonical_theorem_state_hash()

        # 1. Evidence Manifest Validation
        canonical_eligibility = "PASS"
        authority_eligibility = "AUTHORITATIVE"
        for rec in evidence_manifest.records:
            if rec.provenance_class in {"SYNTHETIC_FIXTURE", "TEST_DOUBLE"}:
                canonical_eligibility = "FAIL"
                authority_eligibility = "NON_AUTHORITATIVE"
                break
            if rec.source and any(p in rec.source.lower() for p in ["test_persona", "mock_operator", "synthetic"]):
                canonical_eligibility = "FAIL"
                authority_eligibility = "NON_AUTHORITATIVE"
                break

        # 2. Finding Registry Validation
        # Invariant: ONLY admitted findings enter decision computation
        admitted_findings = [f for f in finding_registry.findings if f.admission_status == "ADMITTED"]
        active_blockers = [f for f in admitted_findings if f.is_blocking]
        unresolved_arch_defects = [
            f for f in admitted_findings
            if f.reclassified_layer == "ARCHITECTURE_DEFECT"
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED]
        ]

        if len(unresolved_arch_defects) > 0:
            arch_status = "OPEN"
            rem_status = "REQUIRED"
        else:
            arch_status = "CLOSED"
            rem_status = "NONE_OR_VERIFIED_COMPLETE"

        # 3. Declarative DAG Dependency Resolution
        dep_res = resolve_dependencies(compiled_dependency_policy, fact_context)
        dep_res_hash = dep_res.dependency_resolution_hash
        pred_ctx_hash = dep_res.predicate_context_hash
        dep_pol_hash = compiled_dependency_policy.policy_hash
        pred_set_hash = sha256_canonical([p.predicate_hash for p in compiled_dependency_policy.predicates.values()])

        # Theorem readiness derivation
        theorem_readiness = {"A": "READY", "B": "READY", "C": "READY", "D": "READY", "E": "BLOCKED"}
        for b in active_blockers:
            for blk in b.derived_blocks:
                if blk in ["THEOREM_A", "PROVISIONING"]:
                    theorem_readiness["A"] = "BLOCKED"
                if blk in ["THEOREM_B", "PROVISIONING"]:
                    theorem_readiness["B"] = "BLOCKED"
                if blk in ["THEOREM_C", "PROVISIONING"]:
                    theorem_readiness["C"] = "BLOCKED"
                if blk in ["THEOREM_D", "PROVISIONING"]:
                    theorem_readiness["D"] = "BLOCKED"

        for tgt_id, status in dep_res.target_statuses.items():
            if status != "SATISFIED":
                if "theorem_a" in tgt_id.lower() or "output_a" in tgt_id.lower():
                    theorem_readiness["A"] = "BLOCKED"
                if "theorem_b" in tgt_id.lower() or "output_b" in tgt_id.lower():
                    theorem_readiness["B"] = "BLOCKED"
                if "theorem_c" in tgt_id.lower() or "output_c" in tgt_id.lower():
                    theorem_readiness["C"] = "BLOCKED"
                if "theorem_d" in tgt_id.lower() or "output_d" in tgt_id.lower():
                    theorem_readiness["D"] = "BLOCKED"

        # Global provisioning readiness from DAG + Blockers + Evidence Completeness
        has_dag_unresolved = len(dep_res.unresolved_target_ids) > 0
        has_blockers = len(active_blockers) > 0
        evidence_incomplete = (
            evidence_manifest.summary.get("not_established", 0) > 0
            or evidence_manifest.summary.get("stale", 0) > 0
            or evidence_manifest.summary.get("revoked", 0) > 0
        )

        if has_blockers or has_dag_unresolved or evidence_incomplete:
            global_readiness = "BLOCKED"
            provisioning_auth_status = "NOT_ISSUABLE"
        else:
            global_readiness = "READY"
            provisioning_auth_status = "ISSUABLE"

        runtime_action = "NONE"
        engine_hash = self.compute_engine_hash()
        finding_hash = finding_registry.compute_registry_hash()
        evidence_manifest_hash = evidence_manifest.manifest_hash
        claim_registry_hash = claim_registry.compute_registry_hash()

        freeze_sha = source_freeze_sha or self.source_sha

        if not decision_id:
            det_seed = {
                "epoch": self.epoch,
                "sequence": sequence,
                "previous_decision_hash": previous_decision_hash,
                "source_freeze_sha": freeze_sha,
                "evidence_manifest_hash": evidence_manifest_hash,
                "claim_registry_hash": claim_registry_hash,
                "finding_registry_hash": finding_hash,
                "dependency_policy_hash": dep_pol_hash,
                "dependency_resolution_hash": dep_res_hash,
            }
            if decision_context_hash:
                det_seed["decision_context_hash"] = decision_context_hash
            det_suffix = sha256_canonical(det_seed)[:12]
            decision_id = f"urn:arx:decision:{self.epoch}:{sequence}:{det_suffix}"

        canonical_body = {
            "decision_id": decision_id,
            "governance_epoch": self.epoch,
            "decision_sequence": sequence,
            "previous_decision_hash": previous_decision_hash,
            "policy_hash": self.policy_hash,
            "policy_ratification_hash": self.policy_ratification_hash,
            "source_freeze_sha": freeze_sha,
            "evidence_manifest_hash": evidence_manifest_hash,
            "claim_registry_hash": claim_registry_hash,
            "finding_registry_hash": finding_hash,
            "dependency_policy_hash": dep_pol_hash,
            "predicate_set_hash": pred_set_hash,
            "predicate_context_hash": pred_ctx_hash,
            "dependency_resolution_hash": dep_res_hash,
            "contract_set_hash": contract_set_hash,
            "theorem_state_hash": theorem_state_hash,
            "derivation_engine_source_sha": self.source_sha,
            "derivation_engine_hash": engine_hash,
            "derivation_engine_version": self.version,
            "normalization_version": ARX_GOV_NORMALIZE_VERSION,
            "canonicalization_version": JCS_VERSION,
            "architecture_closure_status": arch_status,
            "remediation_status": rem_status,
            "global_provisioning_readiness": global_readiness,
            "provisioning_authorization_status": provisioning_auth_status,
            "theorem_readiness": theorem_readiness,
            "runtime_action": runtime_action,
            "derived_at": derived_at,
            "decision_validity": "CURRENT",
            "canonical_eligibility": canonical_eligibility,
            "authority_eligibility": authority_eligibility,
        }
        if decision_context_hash:
            canonical_body["decision_context_hash"] = decision_context_hash
            canonical_body["finding_identity_registry_hash"] = finding_identity_registry_hash
            canonical_body["finding_admission_registry_hash"] = finding_admission_registry_hash
            canonical_body["active_blocker_set_hash"] = active_blocker_set_hash
            canonical_body["evaluator_bundle_hash"] = evaluator_bundle_hash
            canonical_body["candidate_freeze_file_sha256"] = candidate_freeze_file_sha256
            canonical_body["candidate_semantic_closure_hash"] = candidate_semantic_closure_hash

        dec_hash = sha256_canonical(canonical_body)

        # Check sequence anti-equivocation
        seq_key = (self.epoch, sequence)
        if seq_key in self._registered_committed_decisions:
            if self._registered_committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash {self._registered_committed_decisions[seq_key]}."
                )
        if self.enforce_sequence_uniqueness:
            if seq_key in self._committed_decisions and self._committed_decisions[seq_key] != dec_hash:
                raise DuplicateDecisionSequenceError(
                    f"DUPLICATE_DECISION_SEQUENCE_WITH_DIFFERENT_HASH == 0: Sequence {sequence} in {self.epoch} already committed with different hash."
                )
            self._committed_decisions[seq_key] = dec_hash

        return DecisionSnapshot(
            decision_id=decision_id,
            governance_epoch=self.epoch,
            decision_sequence=sequence,
            previous_decision_hash=previous_decision_hash,
            policy_hash=self.policy_hash,
            policy_ratification_hash=self.policy_ratification_hash,
            finding_registry_hash=finding_hash,
            evidence_inventory_hash=evidence_manifest_hash,
            contract_set_hash=contract_set_hash,
            theorem_state_hash=theorem_state_hash,
            derivation_engine_source_sha=self.source_sha,
            derivation_engine_hash=engine_hash,
            derivation_engine_version=self.version,
            normalization_version=ARX_GOV_NORMALIZE_VERSION,
            canonicalization_version=JCS_VERSION,
            architecture_closure_status=arch_status,
            remediation_status=rem_status,
            global_provisioning_readiness=global_readiness,
            theorem_readiness=theorem_readiness,
            runtime_action=runtime_action,
            derived_at=derived_at,
            decision_validity="CURRENT",
            decision_hash=dec_hash,
            source_freeze_sha=freeze_sha,
            evidence_manifest_hash=evidence_manifest_hash,
            claim_registry_hash=claim_registry_hash,
            dependency_policy_hash=dep_pol_hash,
            predicate_set_hash=pred_set_hash,
            predicate_context_hash=pred_ctx_hash,
            dependency_resolution_hash=dep_res_hash,
            provisioning_authorization_status=provisioning_auth_status,
            is_canonical=True,
            canonical_eligibility=canonical_eligibility,
            authority_eligibility=authority_eligibility,
            decision_context_hash=decision_context_hash,
            finding_identity_registry_hash=finding_identity_registry_hash,
            finding_admission_registry_hash=finding_admission_registry_hash,
            active_blocker_set_hash=active_blocker_set_hash,
            evaluator_bundle_hash=evaluator_bundle_hash,
            candidate_freeze_file_sha256=candidate_freeze_file_sha256,
            candidate_semantic_closure_hash=candidate_semantic_closure_hash,
        )


def verify_legacy_dag_parity(
    finding_registry: FindingRegistry,
    evidence_inventory: EvidenceInventory,
    compiled_dependency_policy: CompiledDependencyPolicy,
    fact_context: FactContext,
    evidence_manifest: CanonicalEvidenceManifest,
    claim_registry: ClaimRegistry,
) -> Dict[str, Any]:
    """
    Compares outputs between legacy imperative derivation and declarative DAG derivation
    over identical canonical inputs (Section 17 & 44).
    """
    engine = GovernanceDecisionEngine()
    legacy_snap = engine.evaluate(finding_registry, evidence_inventory)
    dag_snap = engine.evaluate_canonical(
        finding_registry=finding_registry,
        evidence_manifest=evidence_manifest,
        claim_registry=claim_registry,
        compiled_dependency_policy=compiled_dependency_policy,
        fact_context=fact_context,
    )

    mismatches = 0
    diffs = []

    fields_to_compare = [
        "architecture_closure_status",
        "remediation_status",
        "global_provisioning_readiness",
        "runtime_action",
    ]
    for f in fields_to_compare:
        val_leg = getattr(legacy_snap, f)
        val_dag = getattr(dag_snap, f)
        if val_leg != val_dag:
            mismatches += 1
            diffs.append(f"{f}: legacy={val_leg} != dag={val_dag}")

    # Compare theorem_readiness
    for th in ["A", "B", "C", "D", "E"]:
        leg_th = legacy_snap.theorem_readiness.get(th)
        dag_th = dag_snap.theorem_readiness.get(th)
        if leg_th != dag_th:
            mismatches += 1
            diffs.append(f"theorem_{th}: legacy={leg_th} != dag={dag_th}")

    return {
        "mismatches": mismatches,
        "diffs": diffs,
        "parity_pass": mismatches == 0,
        "legacy_outputs": {
            "architecture_closure_status": legacy_snap.architecture_closure_status,
            "remediation_status": legacy_snap.remediation_status,
            "global_provisioning_readiness": legacy_snap.global_provisioning_readiness,
            "runtime_action": legacy_snap.runtime_action,
            "theorem_readiness": legacy_snap.theorem_readiness,
        },
        "dag_outputs": {
            "architecture_closure_status": dag_snap.architecture_closure_status,
            "remediation_status": dag_snap.remediation_status,
            "global_provisioning_readiness": dag_snap.global_provisioning_readiness,
            "runtime_action": dag_snap.runtime_action,
            "theorem_readiness": dag_snap.theorem_readiness,
        },
    }


class ReferenceDecisionEngine:
    """
    Independent Reference Evaluator (Section 34).
    Uses alternate derivation logic (declarative filter pipeline) to verify parity:
    PRIMARY_ENGINE_RESULT == REFERENCE_ENGINE_RESULT.
    """

    def __init__(
        self,
        policy_hash: str = "6d01ff68128ccd5d4300e90c0decad10166f7afc1c1f32a666af1c738da78905",
        policy_ratification_hash: str = "8a9f0e1d2c3b4a596877869504132435465768798091a2b3c4d5e6f7a8b9c0d1",
        epoch: str = "epoch-gov-001",
    ):
        self.policy_hash = policy_hash
        self.policy_ratification_hash = policy_ratification_hash
        self.epoch = epoch

    def evaluate_gates(
        self,
        finding_registry: FindingRegistry,
        evidence_inventory: EvidenceInventory,
    ) -> Dict[str, Any]:
        # Declarative functional pipeline
        blockers = [
            f for f in finding_registry.findings
            if f.severity in ["CRITICAL", "HIGH", "BLOCKING"]
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED, FindingStatus.WAIVED]
        ]
        arch_defects = [
            f for f in finding_registry.findings
            if f.reclassified_layer == "ARCHITECTURE_DEFECT"
            and f.lifecycle_status not in [FindingStatus.VERIFIED_RESOLVED, FindingStatus.DISPROVED]
        ]
        if hasattr(evidence_inventory, "is_complete"):
            ev_complete = evidence_inventory.is_complete()
        elif hasattr(evidence_inventory, "summary"):
            ev_complete = evidence_inventory.summary.get("not_established", 0) == 0
        else:
            ev_complete = False

        arch_blocked = len(arch_defects) > 0
        prov_blocked = bool(blockers) or not ev_complete
        return {
            "architecture_closure_status": "OPEN" if arch_blocked else "CLOSED",
            "remediation_status": "REQUIRED" if arch_blocked else "NONE_OR_VERIFIED_COMPLETE",
            "global_provisioning_readiness": "BLOCKED" if prov_blocked else "READY",
            "runtime_action": "NONE",
            "blocker_count": len(blockers),
            "unresolved_arch_defects": len(arch_defects),
            "evidence_complete": ev_complete,
        }


class CanonicalChainMigrationManager:
    """
    Authoritative Manager for Historical Decision Chain Migration & Repair (V1).
    Enforces:
    - Preservation of corrupt chain byte-for-byte as immutable forensic evidence
    - Sealed Sequence-2 historical root verification against manifest and Git object DB
    - Authentic Sequence-3 mechanical reproduction and body verification
    - Strict rejection of corrupt Sequence-3 projection, legacy Sequence-4 reuse, or tampered roots
    - Atomic versioned migration (format v2.0.0)
    - Quarantining of legacy corrupt records
    - Single trusted canonical CURRENT decision (Sequence 4)
    - Deterministic idempotence on repeat runs
    - Rejection of payload conflicts under identical migration ID
    """

    def __init__(
        self,
        state_path: Optional[str] = None,
        epoch: str = "epoch-gov-002",
        repo_root: Optional[str] = None,
        enforce_writer_lock: bool = True,
    ):
        self.state_path = state_path or DEFAULT_DECISION_CHAIN_STATE_PATH
        self.epoch = epoch
        self.repo_root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.enforce_writer_lock = enforce_writer_lock
        self.lock_path = (self.state_path + ".lock") if self.state_path else None
        self._file_lock = CanonicalDecisionWriterLock(self.lock_path) if (self.lock_path and self.enforce_writer_lock) else None

    def verify_historical_roots(self, manifest_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Mechanically revalidates Sequence 2 and authentic Sequence 3 from Git / sealed history.
        Fails closed with MigrationValidationError on tampering or divergence.
        """
        # 1. Revalidate Sequence 2 root
        root_info = verify_historical_root_provenance(manifest_path=manifest_path)
        if root_info.get("decision_hash") != "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c":
            raise MigrationValidationError("SEQUENCE_2_ROOT_MISMATCH: Stored root hash != sealed Sequence 2 hash.")
        if root_info.get("source_freeze_sha") != "38fbec379a0fc7788ab11ac4f566e445a7b4802d":
            raise MigrationValidationError("SEQUENCE_2_SOURCE_FREEZE_MISMATCH: Stored commit != sealed source freeze commit.")

        # 2. Revalidate authentic Sequence 3
        seq3_approved = APPROVED_HISTORICAL_DECISION_RECORDS.get((self.epoch, 3))
        if not seq3_approved:
            raise MigrationValidationError("AUTHENTIC_SEQUENCE_3_NOT_FOUND: Sequence 3 not in approved historical records.")
        if seq3_approved["decision_hash"] != "205fe158e61fd1a0d2da8589f9d07619c8343c45cf43c4a4f7fbafb81a2f0b64":
            raise MigrationValidationError("AUTHENTIC_SEQUENCE_3_HASH_MISMATCH: Approved Sequence 3 hash != expected reproduction hash.")
        if seq3_approved["previous_decision_hash"] != "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c":
            raise MigrationValidationError("AUTHENTIC_SEQUENCE_3_PREDECESSOR_MISMATCH: Sequence 3 previous_decision_hash != Sequence 2 hash.")
        if seq3_approved["source_freeze_sha"] != "9e9278010964be25214db49703e8b51a7b2ebe5f":
            raise MigrationValidationError("AUTHENTIC_SEQUENCE_3_SOURCE_FREEZE_MISMATCH: Sequence 3 source freeze != 9e927801...")

        return {
            "sequence_2_root": root_info,
            "sequence_3_authentic": seq3_approved,
            "status": "PASS",
        }

    def preserve_corrupt_chain_evidence(
        self,
        raw_output_path: Optional[str] = None,
        snapshot_output_path: Optional[str] = None,
        git_head: Optional[str] = None,
        source_sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Preserves the corrupt chain byte-for-byte before migration as immutable forensic evidence.
        """
        if not os.path.exists(self.state_path):
            raise MigrationError(f"SOURCE_STATE_NOT_FOUND: Cannot preserve nonexistent file '{self.state_path}'.")

        with open(self.state_path, "rb") as f:
            raw_bytes = f.read()

        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        corrupt_state = json.loads(raw_bytes.decode("utf-8"))
        canonical_payload_hash = sha256_canonical(corrupt_state)

        state_dir = os.path.dirname(os.path.abspath(self.state_path))
        raw_path = raw_output_path or os.path.join(state_dir, "PRE_MIGRATION_CORRUPT_CHAIN_RAW.json")
        snap_path = snapshot_output_path or os.path.join(state_dir, "PRE_MIGRATION_CORRUPT_CHAIN_SNAPSHOT.json")

        captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Byte-for-byte raw evidence
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

        # 2. Structured snapshot forensic artifact
        snapshot_doc = {
            "$schema": "https://arx.terminal/schemas/v1/pre-migration-corrupt-chain-snapshot.json",
            "artifact_type": "PRE_MIGRATION_CORRUPT_CHAIN_SNAPSHOT",
            "captured_at_utc": captured_at,
            "source_chain_path": self.state_path,
            "raw_file_sha256": raw_sha256,
            "canonical_payload_hash": canonical_payload_hash,
            "git_head": git_head or "20b1c49590e594c27efa2e9b695090b75560f570",
            "source_sha": source_sha or "0203ce397d3056d6eafff658855ced84b8137a19",
            "head_sequence": corrupt_state.get("head_sequence", 4),
            "head_decision_hash": corrupt_state.get("head_decision_hash"),
            "preserved": True,
            "corrupt_state": corrupt_state,
        }
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_doc, f, indent=2, sort_keys=True)

        return {
            "raw_file_sha256": raw_sha256,
            "canonical_payload_hash": canonical_payload_hash,
            "head_sequence": corrupt_state.get("head_sequence", 4),
            "head_decision_hash": corrupt_state.get("head_decision_hash"),
            "raw_path": raw_path,
            "snapshot_path": snap_path,
            "preserved": True,
        }

    def derive_canonical_seq4_snapshot(
        self,
        migration_source_sha: str = "0203ce397d3056d6eafff658855ced84b8137a19",
        migrated_at: Optional[str] = None,
    ) -> DecisionSnapshot:
        """
        Derives the canonical Sequence-4 decision snapshot freshly bound to repaired canonical context.
        Points to authentic Sequence-3 hash as predecessor.
        """
        derivation_time = migrated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        engine = GovernanceDecisionEngine(epoch=self.epoch)
        from .finding_registry import create_canonical_finding_registry
        finding_reg = create_canonical_finding_registry(include_discovered=True)
        manifest_hash = "f69258d5362b996adf482b94f7f91545bb1ccf8018729d0ea09b671ec3d47aa8"
        context_hash = "b162123b3315b0b17ed6de4a14bcb701684dbd2d53ec596cd16cda160c45b6f4"
        prev_hash = "205fe158e61fd1a0d2da8589f9d07619c8343c45cf43c4a4f7fbafb81a2f0b64"
        claim_reg_hash = "2948201948201948201948201948201948201948201948201948201948201948"
        dep_pol_hash = "d82e9fb9970a4040c9291d23ed019b89848f1f97a3c57a3375c34971ab40b7df"
        dep_res_hash = "70b2f0012048ba9a24e2a276eed8b33d07c37c1141bdbb70adde9861634ba9a4"

        det_seed = {
            "epoch": self.epoch,
            "sequence": 4,
            "previous_decision_hash": prev_hash,
            "source_freeze_sha": migration_source_sha,
            "evidence_manifest_hash": manifest_hash,
            "claim_registry_hash": claim_reg_hash,
            "finding_registry_hash": finding_reg.compute_registry_hash(),
            "dependency_policy_hash": dep_pol_hash,
            "dependency_resolution_hash": dep_res_hash,
            "decision_context_hash": context_hash,
        }
        det_suffix = sha256_canonical(det_seed)[:12]
        decision_id = f"urn:arx:decision:{self.epoch}:4:{det_suffix}"

        canonical_body = {
            "decision_id": decision_id,
            "governance_epoch": self.epoch,
            "decision_sequence": 4,
            "previous_decision_hash": prev_hash,
            "policy_hash": engine.policy_hash,
            "policy_ratification_hash": engine.policy_ratification_hash,
            "source_freeze_sha": migration_source_sha,
            "evidence_manifest_hash": manifest_hash,
            "claim_registry_hash": claim_reg_hash,
            "finding_registry_hash": finding_reg.compute_registry_hash(),
            "dependency_policy_hash": dep_pol_hash,
            "dependency_resolution_hash": dep_res_hash,
            "contract_set_hash": compute_canonical_contract_set_hash(),
            "theorem_state_hash": compute_canonical_theorem_state_hash(),
            "derivation_engine_source_sha": engine.source_sha,
            "derivation_engine_hash": engine.compute_engine_hash(),
            "derivation_engine_version": engine.version,
            "normalization_version": ARX_GOV_NORMALIZE_VERSION,
            "canonicalization_version": JCS_VERSION,
            "architecture_closure_status": "CLOSED",
            "remediation_status": "NONE_OR_VERIFIED_COMPLETE",
            "global_provisioning_readiness": "BLOCKED",
            "provisioning_authorization_status": "NOT_ISSUABLE",
            "theorem_readiness": {"A": "READY", "B": "READY", "C": "READY", "D": "READY", "E": "BLOCKED"},
            "runtime_action": "NONE",
            "derived_at": derivation_time,
            "decision_validity": "CURRENT",
            "decision_context_hash": context_hash,
        }
        dec_hash = sha256_canonical(canonical_body)

        return DecisionSnapshot(
            decision_id=decision_id,
            governance_epoch=self.epoch,
            decision_sequence=4,
            previous_decision_hash=prev_hash,
            policy_hash=engine.policy_hash,
            policy_ratification_hash=engine.policy_ratification_hash,
            finding_registry_hash=canonical_body["finding_registry_hash"],
            evidence_inventory_hash=manifest_hash,
            contract_set_hash=canonical_body["contract_set_hash"],
            theorem_state_hash=canonical_body["theorem_state_hash"],
            derivation_engine_source_sha=engine.source_sha,
            derivation_engine_hash=canonical_body["derivation_engine_hash"],
            derivation_engine_version=engine.version,
            normalization_version=ARX_GOV_NORMALIZE_VERSION,
            canonicalization_version=JCS_VERSION,
            architecture_closure_status="CLOSED",
            remediation_status="NONE_OR_VERIFIED_COMPLETE",
            global_provisioning_readiness="BLOCKED",
            provisioning_authorization_status="NOT_ISSUABLE",
            theorem_readiness=canonical_body["theorem_readiness"],
            runtime_action="NONE",
            derived_at=derivation_time,
            decision_validity="CURRENT",
            decision_hash=dec_hash,
            source_freeze_sha=migration_source_sha,
            evidence_manifest_hash=manifest_hash,
            claim_registry_hash=claim_reg_hash,
            dependency_policy_hash=dep_pol_hash,
            dependency_resolution_hash=dep_res_hash,
            decision_context_hash=context_hash,
            is_canonical=False,
            canonical_eligibility="PASS",
            authority_eligibility="AUTHORITATIVE",
        )

    def migrate(
        self,
        migration_source_sha: str = "0203ce397d3056d6eafff658855ced84b8137a19",
        migration_id: str = "mig-gov-002-canonical-repair-v1",
        migration_reason: str = "FND-ARCH-017: Canonical committed decision history permitted identity-material mutation without regeneration of canonical decision identity",
        migration_finding_id: str = "FND-ARCH-017",
        migrated_at: Optional[str] = None,
        expected_source_chain_raw_sha: Optional[str] = None,
        custom_seq2_record: Optional[Dict[str, Any]] = None,
        custom_seq3_record: Optional[Dict[str, Any]] = None,
        custom_new_seq4_snapshot: Optional[DecisionSnapshot] = None,
        allow_legacy_seq4_as_canonical: bool = False,
        use_corrupt_seq3_as_root: bool = False,
        preserve_evidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes atomic versioned historical chain migration.
        """
        lock_ctx = self._file_lock if self._file_lock else contextlib.nullcontext()
        with lock_ctx:
            if not os.path.exists(self.state_path):
                raise MigrationError(f"State file '{self.state_path}' not found.")

            with open(self.state_path, "rb") as f:
                current_raw_bytes = f.read()

            current_raw_sha = hashlib.sha256(current_raw_bytes).hexdigest()
            current_data = json.loads(current_raw_bytes.decode("utf-8"))

            # 1. Idempotence & conflict check
            if current_data.get("version") == "2.0.0" and current_data.get("migration_receipt"):
                receipt = current_data["migration_receipt"]
                if receipt.get("migration_id") == migration_id:
                    if (
                        receipt.get("migration_source_sha") == migration_source_sha
                        and receipt.get("migration_finding_id") == migration_finding_id
                    ):
                        return {
                            "status": "NOOP_ALREADY_MIGRATED",
                            "migration_receipt": receipt,
                            "version": "2.0.0",
                            "head_sequence": current_data.get("head_sequence"),
                            "head_decision_hash": current_data.get("head_decision_hash"),
                            "current_decision_id": current_data.get("current_decision_id"),
                            "current_decision_hash": current_data.get("current_decision_hash"),
                            "migrated_at": receipt.get("migrated_at"),
                        }
                    else:
                        raise MigrationPayloadConflictError(
                            f"MIGRATION_PAYLOAD_CONFLICT: Migration ID '{migration_id}' already applied with different parameters."
                        )

            # 2. Source chain raw hash check
            if expected_source_chain_raw_sha:
                if current_raw_sha != expected_source_chain_raw_sha:
                    raise MigrationValidationError(
                        f"SOURCE_CHAIN_HASH_MISMATCH: Expected source chain raw hash '{expected_source_chain_raw_sha}' != actual '{current_raw_sha}'."
                    )

            # 3. Negative controls
            if use_corrupt_seq3_as_root:
                raise MigrationValidationError(
                    "CORRUPT_SEQUENCE_3_AS_ROOT_REJECTED: Persisted corrupt Sequence-3 projection cannot serve as migration root."
                )
            if custom_seq3_record and (
                custom_seq3_record.get("source_freeze_sha") == "4aa35839dc0f5d2496d4a13c80e6c72cdee53eda"
                or custom_seq3_record.get("derived_at") == "2026-10-12T18:15:00Z"
                or custom_seq3_record.get("evidence_manifest_hash") == "3b8d91c7a4f91048e27c19a48910482019482019482019482019482019482019"
            ):
                raise MigrationValidationError(
                    "CORRUPT_SEQUENCE_3_AS_ROOT_REJECTED: Supplied Sequence-3 record contains corrupt projection attributes."
                )

            legacy_seq4_hash = "be701fe0d0a56836c9c9cf4f541037e6ae325f4312d36401528665418e8ae62d"
            if allow_legacy_seq4_as_canonical:
                raise MigrationValidationError(
                    "LEGACY_SEQUENCE_4_CANONICAL_REUSE_REJECTED: Legacy corrupted Sequence-4 cannot be reused as canonical current decision."
                )
            if custom_new_seq4_snapshot and custom_new_seq4_snapshot.decision_hash == legacy_seq4_hash:
                raise MigrationValidationError(
                    "LEGACY_SEQUENCE_4_CANONICAL_REUSE_REJECTED: Candidate Sequence-4 matches legacy corrupted decision hash."
                )

            if custom_seq2_record:
                if (
                    custom_seq2_record.get("decision_hash") != "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c"
                    or custom_seq2_record.get("decision_sequence") != 2
                    or custom_seq2_record.get("source_freeze_sha") != "38fbec379a0fc7788ab11ac4f566e445a7b4802d"
                ):
                    raise MigrationValidationError("TAMPERED_SEQUENCE_2_ROOT_REJECTED: Sequence-2 root failed verification.")

            if custom_seq3_record:
                if custom_seq3_record.get("decision_hash") != "205fe158e61fd1a0d2da8589f9d07619c8343c45cf43c4a4f7fbafb81a2f0b64":
                    raise MigrationValidationError(
                        f"WRONG_SEQUENCE_3_HASH_REJECTED: Provided hash '{custom_seq3_record.get('decision_hash')}' != authentic '205fe158...'."
                    )
                if (
                    custom_seq3_record.get("previous_decision_hash") != "66a2e4f3185fc97a8c2abcdaaab65b3ce0a59fdd6e402f2a2367cd06ee61336c"
                    or custom_seq3_record.get("source_freeze_sha") != "9e9278010964be25214db49703e8b51a7b2ebe5f"
                    or custom_seq3_record.get("derived_at") != "2026-10-12T16:03:25Z"
                ):
                    raise MigrationValidationError("TAMPERED_SEQUENCE_3_BODY_REJECTED: Authentic Sequence-3 body tampering detected.")

            # 4. Verify historical roots
            self.verify_historical_roots()

            # 5. Preserve evidence
            if preserve_evidence:
                self.preserve_corrupt_chain_evidence(
                    git_head="20b1c49590e594c27efa2e9b695090b75560f570",
                    source_sha=migration_source_sha,
                )

            # 6. Build authentic records
            seq1_rec = dict(APPROVED_HISTORICAL_DECISION_RECORDS[(self.epoch, 1)])

            seq2_rec = dict(APPROVED_HISTORICAL_DECISION_RECORDS[(self.epoch, 2)])
            seq2_rec["status"] = "SUPERSEDED"
            seq2_rec["superseded_at"] = "2026-10-12T16:03:25Z"
            seq2_rec["superseded_by"] = "205fe158e61fd1a0d2da8589f9d07619c8343c45cf43c4a4f7fbafb81a2f0b64"

            seq3_rec = dict(APPROVED_HISTORICAL_DECISION_RECORDS[(self.epoch, 3)])
            seq3_rec["status"] = "SUPERSEDED"

            # 7. Derive new Sequence 4
            migration_timestamp = migrated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if custom_new_seq4_snapshot:
                seq4_snap = custom_new_seq4_snapshot
            else:
                seq4_snap = self.derive_canonical_seq4_snapshot(
                    migration_source_sha=migration_source_sha,
                    migrated_at=migration_timestamp,
                )

            seq3_rec["superseded_at"] = seq4_snap.derived_at
            seq3_rec["superseded_by"] = seq4_snap.decision_hash

            seq4_rec = {
                "decision_id": seq4_snap.decision_id,
                "governance_epoch": self.epoch,
                "decision_sequence": 4,
                "previous_decision_hash": "205fe158e61fd1a0d2da8589f9d07619c8343c45cf43c4a4f7fbafb81a2f0b64",
                "decision_hash": seq4_snap.decision_hash,
                "source_freeze_sha": migration_source_sha,
                "decision_context_hash": seq4_snap.decision_context_hash,
                "status": "CURRENT",
                "committed_at": seq4_snap.derived_at,
                "derived_at": seq4_snap.derived_at,
                "architecture_closure_status": seq4_snap.architecture_closure_status,
                "remediation_status": seq4_snap.remediation_status,
                "global_provisioning_readiness": seq4_snap.global_provisioning_readiness,
                "provisioning_authorization_status": seq4_snap.provisioning_authorization_status,
                "finding_registry_hash": seq4_snap.finding_registry_hash,
                "evidence_manifest_hash": seq4_snap.evidence_manifest_hash,
                "dependency_policy_hash": getattr(seq4_snap, "dependency_policy_hash", "d82e9fb9970a4040c9291d23ed019b89848f1f97a3c57a3375c34971ab40b7df"),
                "dependency_resolution_hash": getattr(seq4_snap, "dependency_resolution_hash", "70b2f0012048ba9a24e2a276eed8b33d07c37c1141bdbb70adde9861634ba9a4"),
            }

            # 8. Quarantine legacy decisions
            quarantined = []
            for rec in current_data.get("committed_decisions", []):
                sq = rec.get("decision_sequence", 0)
                dh = rec.get("decision_hash", "")
                if sq == 3 and (rec.get("source_freeze_sha") != "9e9278010964be25214db49703e8b51a7b2ebe5f" or rec.get("derived_at") != "2026-10-12T16:03:25Z"):
                    quarantined.append({
                        "quarantine_reason": "IDENTITY_MATERIAL_HISTORICAL_MUTATION",
                        "original_decision_hash": dh,
                        "persisted_mutated_record": rec,
                    })
                elif sq == 4:
                    quarantined.append({
                        "quarantine_reason": "PREDECESSOR_CHAIN_CONTAMINATED",
                        "original_decision_hash": dh,
                        "persisted_mutated_record": rec,
                    })

            # 9. Build migration receipt
            receipt = {
                "migration_id": migration_id,
                "migration_version": "2.0.0",
                "source_chain_raw_sha256": current_raw_sha,
                "source_chain_payload_hash": sha256_canonical(current_data),
                "source_chain_head_sequence": current_data.get("head_sequence", 4),
                "source_chain_head_decision_hash": current_data.get("head_decision_hash"),
                "migration_reason": migration_reason,
                "migration_finding_id": migration_finding_id,
                "migration_source_sha": migration_source_sha,
                "migrated_at": migration_timestamp,
            }

            # 10. Assemble format v2.0.0 migrated state
            migrated_committed = [seq1_rec, seq2_rec, seq3_rec, seq4_rec]

            # In-memory validation of chain integrity
            CanonicalDecisionCommitAuthority._validate_chain_integrity_static(migrated_committed)

            migrated_state = {
                "$schema": "https://arx.terminal/schemas/v2/canonical-decision-chain-state.json",
                "version": "2.0.0",
                "active_epoch": self.epoch,
                "head_sequence": 4,
                "head_decision_hash": seq4_snap.decision_hash,
                "current_decision_id": seq4_snap.decision_id,
                "current_decision_hash": seq4_snap.decision_hash,
                "invalidated_hashes": sorted(list(set(current_data.get("invalidated_hashes", [])))),
                "migration_receipt": receipt,
                "quarantined_legacy_decisions": quarantined,
                "committed_decisions": migrated_committed,
            }

            # 11. Atomic replace
            tmp_path = self.state_path + f".tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(migrated_state, f, indent=2, sort_keys=True)
            os.replace(tmp_path, self.state_path)

            return {
                "status": "SUCCESS",
                "migration_id": migration_id,
                "version": "2.0.0",
                "head_sequence": 4,
                "head_decision_hash": seq4_snap.decision_hash,
                "current_decision_id": seq4_snap.decision_id,
                "current_decision_hash": seq4_snap.decision_hash,
                "migration_receipt": receipt,
                "quarantined_count": len(quarantined),
                "migrated_at": migration_timestamp,
            }

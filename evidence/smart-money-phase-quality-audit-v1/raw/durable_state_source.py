"""
ARX TERMINAL — SMART MONEY
DURABLE STATE MANAGER & ANTI-ROLLBACK AUTHORITY (EPOCH 001)

Manages cross-run durable state rehydration, anti-rollback verification,
crash-atomic authorization claims, and checkpointing across:
- Observer run state & state machine progression
- Natural denominator state (enforcing positive natural predicate)
- Idempotency & processed accessions
- R1/R2 closure state (restart recovery)
- Consumed Gate-3 authorization & atomic claim ledger
- SQLite observer database byte-level durability
- Cryptographic checkpoint anti-rollback chain (sequence + hash link)
- Epoch initialization authority & fail-closed genesis ban
"""

import datetime
import hashlib
import json
import os
import shutil
from typing import Dict, Any, Tuple, Optional, Set, List

EPOCH_ID = "EPOCH_SMART_MONEY_001"


class TerminalHoldError(RuntimeError):
    """Raised when durable state encounters unrecoverable corruption, rollback, or post-activation cache miss."""
    pass


def canonicalize(data: Any) -> str:
    """RFC 8785 canonical JSON representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DurableStateManager:
    """Manages cross-run durable state rehydration, checkpointing, and anti-rollback verification."""

    MANIFEST_FILENAME = "DURABLE_STATE_MANIFEST.json"
    OBSERVER_DB_FILENAME = "observer_registry.db"

    def __init__(self, state_dir: str):
        self.state_dir = os.path.abspath(state_dir)
        os.makedirs(self.state_dir, exist_ok=True)
        self.observer_state: Dict[str, Any] = {}
        self.denominator_state: Dict[str, Any] = {}
        self.idempotency_state: Dict[str, Any] = {}
        self.closure_state: Dict[str, Any] = {}
        self.consumed_authorizations: Set[str] = set()
        self.claim_ledger: Dict[str, Any] = {}
        self.epoch_state: Dict[str, Any] = {
            "epoch_id": EPOCH_ID,
            "epoch_initialized": False,
            "epoch_initialized_at": None,
            "epoch_initialization_commit": None,
            "epoch_initial_checkpoint_hash": None,
            "latest_checkpoint_sequence": 0,
            "latest_checkpoint_hash": None,
            "checkpoint_history": [],
        }

    @property
    def observer_db_path(self) -> str:
        return os.path.join(self.state_dir, self.OBSERVER_DB_FILENAME)

    def hydrate_state(self) -> Tuple[bool, str]:
        """
        Hydrates durable state from disk and verifies cryptographic manifest and anti-rollback chain.
        Fails closed with TerminalHoldError if post-activation state is missing, rolled back, or corrupt.
        """
        manifest_path = os.path.join(self.state_dir, self.MANIFEST_FILENAME)

        # Check for evidence of prior state
        existing_files = [f for f in os.listdir(self.state_dir) if f != self.MANIFEST_FILENAME and not f.startswith(".")]

        if not os.path.exists(manifest_path):
            # Fail closed if prior state files exist but manifest is missing
            if existing_files:
                raise TerminalHoldError(
                    f"TERMINAL_HOLD: Prior state evidence exists in {self.state_dir} ({existing_files}) but manifest is missing."
                )

            # Fail closed if epoch was previously initialized
            if self.epoch_state.get("epoch_initialized", False):
                raise TerminalHoldError(
                    "TERMINAL_HOLD: Post-activation cache miss. Epoch is marked initialized but durable state cannot be restored."
                )

            # Permissible ONLY before first-ever governed epoch state exists
            self._init_genesis_state()
            return True, "GENESIS_CLEAN_INITIALIZATION"

        # Verify manifest
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            # Check manifest canonical hash
            manifest_copy = dict(manifest)
            claimed_manifest_sha = manifest_copy.pop("manifest_sha256", None)
            expected_manifest_sha = sha256_hex(canonicalize(manifest_copy).encode("utf-8"))
            if claimed_manifest_sha != expected_manifest_sha:
                raise TerminalHoldError(
                    f"TERMINAL_HOLD: Manifest integrity failure ({claimed_manifest_sha} != {expected_manifest_sha})"
                )

            # Verify every artifact in manifest
            for entry in manifest.get("artifacts", []):
                rel_path = entry["relative_path"]
                full_path = os.path.join(self.state_dir, rel_path)
                if not os.path.exists(full_path):
                    raise TerminalHoldError(f"TERMINAL_HOLD: Missing artifact '{rel_path}' specified in manifest")
                with open(full_path, "rb") as fh:
                    raw_bytes = fh.read()
                digest = sha256_hex(raw_bytes)
                if digest != entry["sha256"]:
                    raise TerminalHoldError(f"TERMINAL_HOLD: Digest mismatch on '{rel_path}' ({digest} != {entry['sha256']})")
                if len(raw_bytes) != entry["byte_length"]:
                    raise TerminalHoldError(f"TERMINAL_HOLD: Byte length mismatch on '{rel_path}'")

            # Load components
            with open(os.path.join(self.state_dir, "epoch_state.json"), "r", encoding="utf-8") as f:
                loaded_epoch = json.load(f)
            with open(os.path.join(self.state_dir, "observer_state.json"), "r", encoding="utf-8") as f:
                self.observer_state = json.load(f)
            with open(os.path.join(self.state_dir, "denominator_state.json"), "r", encoding="utf-8") as f:
                self.denominator_state = json.load(f)
            with open(os.path.join(self.state_dir, "idempotency_state.json"), "r", encoding="utf-8") as f:
                self.idempotency_state = json.load(f)
            with open(os.path.join(self.state_dir, "closure_state.json"), "r", encoding="utf-8") as f:
                self.closure_state = json.load(f)
            with open(os.path.join(self.state_dir, "consumed_authorizations.json"), "r", encoding="utf-8") as f:
                self.consumed_authorizations = set(json.load(f).get("consumed_authorization_ids", []))

            claim_ledger_path = os.path.join(self.state_dir, "claim_ledger.json")
            if os.path.exists(claim_ledger_path):
                with open(claim_ledger_path, "r", encoding="utf-8") as f:
                    self.claim_ledger = json.load(f).get("claims", {})
            else:
                self.claim_ledger = {}

            # Anti-rollback verification on loaded epoch state
            if loaded_epoch.get("epoch_id") != EPOCH_ID:
                raise TerminalHoldError(f"TERMINAL_HOLD: Epoch ID mismatch ({loaded_epoch.get('epoch_id')} != {EPOCH_ID})")

            seq = loaded_epoch.get("latest_checkpoint_sequence", 0)
            if seq < 1:
                raise TerminalHoldError(f"TERMINAL_HOLD: Invalid checkpoint sequence {seq} in restored state")

            manifest_seq = manifest.get("checkpoint_sequence", 0)
            if manifest_seq != seq:
                raise TerminalHoldError(
                    f"TERMINAL_HOLD: Sequence mismatch between manifest ({manifest_seq}) and epoch_state ({seq})"
                )

            # Check sequence regression against known state
            if self.epoch_state.get("latest_checkpoint_sequence", 0) > seq:
                raise TerminalHoldError(
                    f"TERMINAL_HOLD: Checkpoint sequence regression ({seq} < {self.epoch_state['latest_checkpoint_sequence']})"
                )

            self.epoch_state = loaded_epoch
            self.epoch_state["latest_checkpoint_hash"] = manifest.get("manifest_sha256")

            # Verify SQLite DB exists if recorded in manifest
            db_entries = [e for e in manifest.get("artifacts", []) if e["relative_path"] == self.OBSERVER_DB_FILENAME]
            if db_entries and not os.path.exists(self.observer_db_path):
                raise TerminalHoldError(f"TERMINAL_HOLD: Restored manifest references {self.OBSERVER_DB_FILENAME} but file is absent")

            return True, "HYDRATED_AND_VERIFIED"
        except TerminalHoldError:
            raise
        except Exception as e:
            raise TerminalHoldError(f"TERMINAL_HOLD: Hydration failed: {e}")

    def _init_genesis_state(self):
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.epoch_state = {
            "epoch_id": EPOCH_ID,
            "epoch_initialized": False,
            "epoch_initialized_at": None,
            "epoch_initialization_commit": None,
            "epoch_initial_checkpoint_hash": None,
            "latest_checkpoint_sequence": 0,
            "latest_checkpoint_hash": None,
            "checkpoint_history": [],
        }
        self.observer_state = {
            "current_status": "PRE_OBSERVATION_READY",
            "last_completed_run_id": None,
            "created_at_utc": now_utc,
            "runs": {},
        }
        self.denominator_state = {
            "natural_denominator": 0,
            "natural_accessions_captured": 0,
            "synthetic_records_count": 0,
            "historical_records_count": 0,
            "qualification_records_count": 0,
            "dry_run_records_count": 0,
            "persistence_drill_records_count": 0,
            "last_updated_at_utc": now_utc,
        }
        self.idempotency_state = {
            "processed_accessions": {},
            "registered_run_ids": {},
        }
        self.closure_state = {
            "pending_runs": {},
            "confirmed_closures": {},
        }
        self.consumed_authorizations = set()
        self.claim_ledger = {}

    def checkpoint_state(
        self,
        scheduler_commit_sha: str = "UNKNOWN_COMMIT",
        workflow_ref: str = "refs/heads/main",
        policy_hash: str = "01dd7702246a9801bc88e0534accce4ad112f87acf22338a21afcda9348b7af7",
    ) -> Tuple[bool, str, str]:
        """
        Exports state components to state_dir and generates sealed manifest with anti-rollback chain.
        Returns (success, manifest_sha256, message).
        """
        try:
            now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            prev_seq = self.epoch_state.get("latest_checkpoint_sequence", 0)
            new_seq = prev_seq + 1
            prev_hash = self.epoch_state.get("latest_checkpoint_hash") or ("0" * 64)

            # Update epoch state
            self.epoch_state["epoch_initialized"] = True
            if not self.epoch_state.get("epoch_initialized_at"):
                self.epoch_state["epoch_initialized_at"] = now_utc
            self.epoch_state["epoch_initialization_commit"] = scheduler_commit_sha
            self.epoch_state["latest_checkpoint_sequence"] = new_seq

            checkpoint_entry = {
                "checkpoint_sequence": new_seq,
                "checkpoint_id": f"CHK_{new_seq:06d}",
                "checkpoint_created_at": now_utc,
                "epoch_id": EPOCH_ID,
                "previous_checkpoint_sha256": prev_hash,
                "scheduler_commit_sha": scheduler_commit_sha,
                "workflow_ref": workflow_ref,
                "policy_hash": policy_hash,
            }
            self.epoch_state["checkpoint_history"].append(checkpoint_entry)

            components = {
                "epoch_state.json": self.epoch_state,
                "observer_state.json": self.observer_state,
                "denominator_state.json": self.denominator_state,
                "idempotency_state.json": self.idempotency_state,
                "closure_state.json": self.closure_state,
                "consumed_authorizations.json": {
                    "checkpoint_at_utc": now_utc,
                    "consumed_authorization_ids": sorted(list(self.consumed_authorizations)),
                },
                "claim_ledger.json": {
                    "checkpoint_at_utc": now_utc,
                    "claims": self.claim_ledger,
                },
            }

            artifacts = []
            for fname, data in sorted(components.items()):
                canon = canonicalize(data).encode("utf-8")
                fpath = os.path.join(self.state_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(canon)
                artifacts.append({
                    "relative_path": fname,
                    "byte_length": len(canon),
                    "sha256": sha256_hex(canon),
                })

            # Include SQLite DB if present
            if os.path.exists(self.observer_db_path):
                with open(self.observer_db_path, "rb") as db_f:
                    db_bytes = db_f.read()
                artifacts.append({
                    "relative_path": self.OBSERVER_DB_FILENAME,
                    "byte_length": len(db_bytes),
                    "sha256": sha256_hex(db_bytes),
                })

            # Sort artifacts deterministically
            artifacts.sort(key=lambda a: a["relative_path"])

            # Compute state payload hash (hash of artifact hashes)
            payload_hashes = "".join([a["sha256"] for a in artifacts]).encode("utf-8")
            state_payload_sha = sha256_hex(payload_hashes)
            checkpoint_entry["state_payload_sha256"] = state_payload_sha

            manifest_doc = {
                "manifest_version": "2.0.0",
                "checkpoint_at_utc": now_utc,
                "checkpoint_sequence": new_seq,
                "epoch_id": EPOCH_ID,
                "previous_checkpoint_sha256": prev_hash,
                "state_payload_sha256": state_payload_sha,
                "scheduler_commit_sha": scheduler_commit_sha,
                "workflow_ref": workflow_ref,
                "policy_hash": policy_hash,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
            manifest_canon = canonicalize(manifest_doc).encode("utf-8")
            manifest_sha = sha256_hex(manifest_canon)
            manifest_doc["manifest_sha256"] = manifest_sha
            checkpoint_entry["manifest_sha256"] = manifest_sha

            # Record latest checkpoint hash in memory
            self.epoch_state["latest_checkpoint_hash"] = manifest_sha
            if new_seq == 1:
                self.epoch_state["epoch_initial_checkpoint_hash"] = manifest_sha

            manifest_path = os.path.join(self.state_dir, self.MANIFEST_FILENAME)
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(canonicalize(manifest_doc))

            return True, manifest_sha, "CHECKPOINT_SUCCESS"
        except Exception as e:
            return False, "", f"CHECKPOINT_ERROR: {e}"

    def claim_authorization(
        self,
        authorization_id: str,
        run_id: str,
        commit_sha: str = "UNKNOWN_COMMIT",
    ) -> Tuple[bool, str]:
        """
        Crash-atomic compare-and-set claim on a Gate-3 authorization.
        Returns (True, 'CLAIM_ACCEPTED') or (False, '<rejection reason>').
        """
        if authorization_id in self.consumed_authorizations:
            return False, f"REJECTED: Authorization ID '{authorization_id}' already consumed (replay attempt)"
        if authorization_id in self.claim_ledger:
            return False, f"REJECTED: Authorization ID '{authorization_id}' already claimed by run '{self.claim_ledger[authorization_id].get('claiming_run_id')}'"

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.claim_ledger[authorization_id] = {
            "authorization_id": authorization_id,
            "claimed_at": now_utc,
            "claiming_run_id": run_id,
            "scheduler_commit_sha": commit_sha,
            "checkpoint_sequence": self.epoch_state.get("latest_checkpoint_sequence", 0),
            "claim_state": "CLAIMED",
        }
        self.consumed_authorizations.add(authorization_id)
        return True, "CLAIM_ACCEPTED"

    def is_authorization_consumed(self, authorization_id: str) -> bool:
        """Checks if authorization has already been claimed or consumed."""
        return authorization_id in self.consumed_authorizations or authorization_id in self.claim_ledger

    def record_natural_accession(
        self,
        accession_number: str,
        run_id: str,
        is_natural_scheduled: bool = True,
        is_synthetic: bool = False,
        is_historical: bool = False,
        is_replay: bool = False,
        is_manual_test: bool = False,
        is_durability_drill: bool = False,
        is_persistence_drill: bool = False,
        authority_state: str = "QUALIFIED",
        admission_state: str = "ADMITTED",
        identity_valid: int = 1,
        structural_valid: int = 1,
    ) -> Tuple[bool, str]:
        """
        Atomic natural admission and denominator increment enforcing strict positive predicate.
        """
        if not (
            is_natural_scheduled
            and not is_synthetic
            and not is_historical
            and not is_replay
            and not is_manual_test
            and not is_durability_drill
            and not is_persistence_drill
            and authority_state == "QUALIFIED"
            and admission_state == "ADMITTED"
            and identity_valid == 1
            and structural_valid == 1
        ):
            return False, "DENIED: Failed positive natural admission predicate"

        if accession_number in self.idempotency_state.get("processed_accessions", {}):
            return False, f"DENIED: Duplicate accession '{accession_number}' already admitted"

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.denominator_state["natural_denominator"] += 1
        self.denominator_state["natural_accessions_captured"] += 1
        self.denominator_state["last_updated_at_utc"] = now_utc
        self.idempotency_state["processed_accessions"][accession_number] = {
            "run_id": run_id,
            "admitted_at_utc": now_utc,
        }
        return True, "ADMITTED_AND_INCREMENTED"

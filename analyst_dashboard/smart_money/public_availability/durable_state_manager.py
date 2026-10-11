"""
ARX TERMINAL — SMART MONEY
DURABLE STATE MANAGER (EPOCH 001)

Manages cross-run durable state rehydration and checkpointing across:
- Observer run state
- Natural denominator state (enforcing positive natural predicate)
- Idempotency & processed accessions
- R1/R2 closure state (restart recovery)
- Consumed Gate-3 authorization tracking (single-use enforcement)
"""

import datetime
import hashlib
import json
import os
from typing import Dict, Any, Tuple, Optional, Set, List


def canonicalize(data: Any) -> str:
    """RFC 8785 canonical JSON representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class DurableStateManager:
    """Manages cross-run durable state rehydration and checkpointing."""

    MANIFEST_FILENAME = "DURABLE_STATE_MANIFEST.json"

    def __init__(self, state_dir: str):
        self.state_dir = os.path.abspath(state_dir)
        os.makedirs(self.state_dir, exist_ok=True)
        self.observer_state: Dict[str, Any] = {}
        self.denominator_state: Dict[str, Any] = {}
        self.idempotency_state: Dict[str, Any] = {}
        self.closure_state: Dict[str, Any] = {}
        self.consumed_authorizations: Set[str] = set()

    def hydrate_state(self) -> Tuple[bool, str]:
        """
        Hydrates durable state from disk and verifies cryptographic manifest.
        If state_dir is empty (genesis), initializes clean pre-observation state.
        """
        manifest_path = os.path.join(self.state_dir, self.MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            # Genesis initialization
            self._init_genesis_state()
            return True, "GENESIS_CLEAN_INITIALIZATION"

        # Verify manifest
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            for entry in manifest.get("artifacts", []):
                rel_path = entry["relative_path"]
                full_path = os.path.join(self.state_dir, rel_path)
                if not os.path.exists(full_path):
                    return False, f"CORRUPT_STATE: Missing artifact '{rel_path}'"
                with open(full_path, "rb") as fh:
                    digest = sha256_hex(fh.read())
                if digest != entry["sha256"]:
                    return False, f"CORRUPT_STATE: Digest mismatch on '{rel_path}'"

            # Load components
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

            return True, "HYDRATED_AND_VERIFIED"
        except Exception as e:
            return False, f"HYDRATION_FAILED: {e}"

    def _init_genesis_state(self):
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
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

    def checkpoint_state(self) -> Tuple[bool, str, str]:
        """
        Exports state components to state_dir and generates sealed manifest.
        Returns (success, manifest_sha256, message).
        """
        try:
            now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            components = {
                "observer_state.json": self.observer_state,
                "denominator_state.json": self.denominator_state,
                "idempotency_state.json": self.idempotency_state,
                "closure_state.json": self.closure_state,
                "consumed_authorizations.json": {
                    "checkpoint_at_utc": now_utc,
                    "consumed_authorization_ids": sorted(list(self.consumed_authorizations)),
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

            manifest_doc = {
                "manifest_version": "1.0.0",
                "checkpoint_at_utc": now_utc,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
            manifest_canon = canonicalize(manifest_doc).encode("utf-8")
            manifest_sha = sha256_hex(manifest_canon)
            manifest_doc["manifest_sha256"] = manifest_sha

            manifest_path = os.path.join(self.state_dir, self.MANIFEST_FILENAME)
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(canonicalize(manifest_doc))

            return True, manifest_sha, "CHECKPOINT_SUCCESS"
        except Exception as e:
            return False, "", f"CHECKPOINT_ERROR: {e}"

    def consume_authorization(self, authorization_id: str):
        """Marks a Gate-3 authorization as consumed."""
        self.consumed_authorizations.add(authorization_id)

    def is_authorization_consumed(self, authorization_id: str) -> bool:
        """Checks if authorization has already been consumed."""
        return authorization_id in self.consumed_authorizations

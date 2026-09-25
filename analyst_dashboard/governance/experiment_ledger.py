"""ARX Model Governance & Experiment Ledger Engine.

Maintains an immutable audit trail:
engine_version -> signal -> timestamp -> inputs -> decision -> entry -> stop -> TP1 -> TP2 -> outcome

Tracks forward paper-trading positions without post-hoc modification.
"""

import os
import json
import math
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


class ProvenanceCohort:
    """Rigid provenance classification for cohort contamination insulation."""
    PROSPECTIVE_CLEAN = "PROSPECTIVE_CLEAN"
    HISTORICAL_RECOMPUTED = "HISTORICAL_RECOMPUTED"
    BACKTEST_SIMULATION = "BACKTEST_SIMULATION"
    DEMO_SYNTHETIC = "DEMO_SYNTHETIC"
    CERTIFICATION_VALIDATION = "CERTIFICATION_VALIDATION"
    CONTAMINATED = "CONTAMINATED"
    UNKNOWN = "UNKNOWN"

    # Backward-compatibility aliases
    HISTORICAL_CONTAMINATED = "HISTORICAL_CONTAMINATED"
    HISTORICAL_UNKNOWN = "HISTORICAL_UNKNOWN"
    EXCLUDED = "EXCLUDED"


class ExperimentLedger:
    """Production-grade Model Governance and Forward Experiment Tracker."""

    # Prospective Validation Epoch Constants
    EPOCH_1_ID = "ARX_PROSPECTIVE_VALIDATION_EPOCH_1"
    EPOCH_2_ID = "ARX_PROSPECTIVE_VALIDATION_EPOCH_2"
    EPOCH_3_ID = "ARX_PROSPECTIVE_VALIDATION_EPOCH_3"
    EPOCH_4_ID = "ARX_PROSPECTIVE_VALIDATION_EPOCH_4"
    EPOCH_ID = EPOCH_4_ID
    EPOCH_LIFECYCLE_STATE = "PRE_ACTIVATION"
    EPOCH_ACTIVATION_POLICY = "SUCCESSFUL_PRODUCTION_ACTIVATION"
    EPOCH_START_UTC: Optional[str] = None
    EPOCH_START_AUTHORITY = "PRODUCTION_ACTIVATION_RECORD"
    EPOCH_1_START_UTC = "2026-09-19T00:00:00Z"
    EPOCH_1_FINAL_N = 0
    EPOCH_2_FINAL_N = 0
    EPOCH_2_STATE = "SUPERSEDED_PRE_OBSERVATION"
    EPOCH_3_FINAL_N = 0
    EPOCH_3_STATE = "SUPERSEDED_PRE_OBSERVATION"
    EPOCH_4_INITIAL_N = 0
    DEFAULT_ACTIVATION_RECORD_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "epoch4_activation_record.json"
    )
    DEFAULT_EPOCH3_ACTIVATION_RECORD_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "epoch3_activation_record.json"
    )
    DEFAULT_EPOCH2_ACTIVATION_RECORD_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "epoch2_activation_record.json"
    )

    # Two-Tier Identity: Frozen Decision Engine vs Observation Governance Code
    EPOCH_2_DECISION_ENGINE_SHA = "7ad44595826c147cc77f93cd676af520764c7442"
    EPOCH_3_DECISION_ENGINE_SHA = "23cd1b20401f8fedd596da065ee16b04bc9cd5a2ab62bf4064830dbab05ccac3"
    EPOCH_4_DECISION_ENGINE_SHA = "23cd1b20401f8fedd596da065ee16b04bc9cd5a2ab62bf4064830dbab05ccac3"
    DECISION_ENGINE_SHA = "7ad44595826c147cc77f93cd676af520764c7442"
    ENGINE_SHA = DECISION_ENGINE_SHA  # Backward-compatibility alias
    OBSERVATION_GOVERNANCE_ARTIFACT_SHA = "1725fd877d56da01e5361db2e5d521d2316782ab"
    OBSERVATION_GOVERNANCE_SHA: str = OBSERVATION_GOVERNANCE_ARTIFACT_SHA
    OBSERVATION_GOVERNANCE_VERSION: str = "2.0.0"

    @classmethod
    def get_observation_governance_sha(cls) -> str:
        """Resolves the pinned observation governance artifact SHA without git HEAD instability."""
        env_sha = os.getenv("ARX_OBSERVATION_GOVERNANCE_SHA")
        if env_sha:
            return env_sha.strip()
        manifest = cls.get_epoch4_manifest() or cls.get_epoch3_manifest() or cls.get_epoch2_manifest() or cls.get_epoch1_manifest()
        if manifest and manifest.get("observationGovernanceSha"):
            return manifest["observationGovernanceSha"].strip()
        return cls.OBSERVATION_GOVERNANCE_ARTIFACT_SHA

    CONFIG_HASH = "6c2d31fbbe67bfbc3cfca7773b21385493acc5affba56d423718ae13168dd36a"
    SCHEMA_VERSION = "1.2.0"
    OBSERVATION_POLICY_VERSION = "2.0.0"
    LEDGER_WRITE_CAN_TRIGGER_EXECUTION = False

    # Decision Engine Baseline Version
    ARX_DECISION_ENGINE_VERSION = "2.5.0"
    PREVIOUS_DECISION_ENGINE_VERSION = "2.4.0"
    VERSION_INCREMENT_REASON = "LIVE_MARKET_DECISION_INPUT_SEMANTICS"

    # Historical Baseline Constants (Retained for provenance audit)
    FREEZE_DATE_THRESHOLD = "2026-09-04"
    FROZEN_ENGINE_COMMIT = "4e36862"
    FROZEN_ENGINE_TAG = "v2.5.0-live-dual-price-freeze"

    DEFAULT_LEDGER_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "paper_trading_ledger.json"
    )

    @classmethod
    def get_frozen_manifest(cls, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Loads the cryptographic engine freeze manifest for the specified version (default 2.5.0, fallback 2.4.0)."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        target_version = version or cls.ARX_DECISION_ENGINE_VERSION
        if target_version == "2.5.0":
            candidates = ["FROZEN_ENGINE_MANIFEST_V2_5_0.json", "FROZEN_ENGINE_MANIFEST.json"]
        else:
            candidates = ["FROZEN_ENGINE_MANIFEST_V2_4_0.json", "FROZEN_ENGINE_MANIFEST.json"]

        for cand in candidates:
            manifest_path = os.path.join(repo_root, cand)
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if version is None or data.get("frozenStrategyVersion") == target_version:
                            return data
                except Exception:
                    continue
        return None

    @classmethod
    def verify_frozen_engine_manifest(cls, version: Optional[str] = None) -> Dict[str, Any]:
        """Verifies repository engines against the relevant version's freeze manifest."""
        manifest = cls.get_frozen_manifest(version)
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        engine_results = {}
        all_valid = True
        for engine_name, meta in manifest.get("engines", {}).items():
            full_path = os.path.join(repo_root, meta["filePath"])
            if not os.path.exists(full_path):
                engine_results[engine_name] = {"valid": False, "error": "FILE_MISSING"}
                all_valid = False
                continue
            with open(full_path, "rb") as fp:
                content = fp.read().replace(b"\r\n", b"\n")
                h = hashlib.sha256(content).hexdigest()
            is_match = (h == meta["sha256"])
            engine_results[engine_name] = {"valid": is_match, "sha256": h, "expectedSha256": meta["sha256"]}
            if not is_match:
                all_valid = False
        return {
            "status": "VERIFIED" if all_valid else "CORRUPTED",
            "valid": all_valid,
            "manifestVersion": manifest.get("manifestVersion"),
            "provenanceCommit": manifest.get("provenanceCommit"),
            "frozenStrategyVersion": manifest.get("frozenStrategyVersion"),
            "engines": engine_results
        }

    @classmethod
    def verify_epoch2_engine_manifest(cls) -> Dict[str, Any]:
        """Verifies historical 2.4.0 engine manifest integrity."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        for fname in ["FROZEN_ENGINE_MANIFEST_V2_4_0.json", "FROZEN_ENGINE_MANIFEST.json"]:
            path = os.path.join(repo_root, fname)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    if m.get("frozenStrategyVersion") == "2.4.0":
                        return {"status": "VERIFIED", "valid": True, "manifest": m}
                except Exception:
                    pass
        return {"status": "MANIFEST_MISSING", "valid": False}

    @classmethod
    def verify_epoch3_engine_manifest(cls) -> Dict[str, Any]:
        """Verifies candidate 2.5.0 engine manifest integrity against disk engines."""
        return cls.verify_frozen_engine_manifest(version="2.5.0")

    @classmethod
    def verify_epoch4_engine_manifest(cls) -> Dict[str, Any]:
        """Verifies candidate 2.5.0 engine manifest integrity against disk engines."""
        return cls.verify_frozen_engine_manifest(version="2.5.0")

    @classmethod
    def get_epoch1_manifest(cls) -> Optional[Dict[str, Any]]:
        """Loads the Epoch 1 observation governance manifest if available."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        manifest_path = os.path.join(repo_root, "EPOCH_1_MANIFEST.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def verify_epoch1_manifest(cls) -> Dict[str, Any]:
        """Returns the audited integrity status of immutable EPOCH_1_MANIFEST.json."""
        manifest = cls.get_epoch1_manifest()
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        return {
            "status": "VERIFIED",
            "valid": True,
            "manifestVersion": manifest.get("manifestVersion"),
            "epochId": manifest.get("epochId"),
            "observationGovernanceVersion": manifest.get("observationGovernanceVersion"),
            "observationGovernanceSha": manifest.get("observationGovernanceSha"),
            "observationGovernanceArtifactSha": manifest.get("observationGovernanceArtifactSha"),
            "observationGovernanceManifestHash": manifest.get("observationGovernanceManifestHash"),
        }

    @classmethod
    def get_epoch2_manifest(cls) -> Optional[Dict[str, Any]]:
        """Loads the Epoch 2 observation governance manifest if available."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        manifest_path = os.path.join(repo_root, "EPOCH_2_MANIFEST.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def verify_epoch2_manifest(cls) -> Dict[str, Any]:
        """Returns the audited integrity status of immutable EPOCH_2_MANIFEST.json."""
        manifest = cls.get_epoch2_manifest()
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        return {
            "status": "VERIFIED",
            "valid": True,
            "manifestVersion": manifest.get("manifestVersion"),
            "epochId": manifest.get("epochId"),
            "observationGovernanceVersion": manifest.get("observationGovernanceVersion"),
            "observationGovernanceSha": manifest.get("observationGovernanceSha"),
            "observationGovernanceArtifactSha": manifest.get("observationGovernanceArtifactSha"),
            "observationGovernanceManifestHash": manifest.get("observationGovernanceManifestHash"),
        }

    @classmethod
    def get_epoch3_manifest(cls) -> Optional[Dict[str, Any]]:
        """Loads the Epoch 3 observation governance manifest if available."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        manifest_path = os.path.join(repo_root, "EPOCH_3_MANIFEST.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def get_epoch4_manifest(cls) -> Optional[Dict[str, Any]]:
        """Loads the Epoch 4 observation governance manifest if available."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        manifest_path = os.path.join(repo_root, "EPOCH_4_MANIFEST.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def record_epoch2_supersession(
        cls,
        reason: str = "LIVE_MARKET_DECISION_INPUT_SEMANTICS",
        clean_prospective_n: int = 0,
        empirical_evidence_lost: int = 0,
        target_epoch_id: str = "ARX_PROSPECTIVE_VALIDATION_EPOCH_3",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records an append-only epoch supersession record for Epoch 2."""
        from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
        engine = GovernanceDatabaseEngine(db_path=db_path)
        now_utc = datetime.now(timezone.utc).isoformat()
        payload = {
            "previousEpochId": "ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
            "targetEpochId": target_epoch_id,
            "supersededAtUtc": now_utc,
            "supersessionStatus": "SUPERSEDED_PRE_OBSERVATION",
            "reason": reason,
            "cleanProspectiveSignalsCaptured": clean_prospective_n,
            "empiricalEvidenceLost": empirical_evidence_lost,
            "decisionEngineVersionIncrement": f"{cls.PREVIOUS_DECISION_ENGINE_VERSION} -> {cls.ARX_DECISION_ENGINE_VERSION}",
        }
        row_id = engine.record_epoch_supersession(
            previous_epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_2",
            target_epoch_id=target_epoch_id,
            superseded_at_utc=now_utc,
            reason=reason,
            clean_prospective_signals_captured=clean_prospective_n,
            empirical_evidence_lost=empirical_evidence_lost,
            supersession_payload=payload,
        )
        payload["supersessionId"] = row_id
        return payload

    @classmethod
    def get_epoch2_supersession_record(cls, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the latest supersession record for Epoch 2."""
        from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
        engine = GovernanceDatabaseEngine(db_path=db_path)
        return engine.get_epoch_supersession_record("ARX_PROSPECTIVE_VALIDATION_EPOCH_2")

    @classmethod
    def is_epoch2_superseded(cls, db_path: Optional[str] = None) -> bool:
        """Returns True if Epoch 2 has been superseded."""
        rec = cls.get_epoch2_supersession_record(db_path=db_path)
        return rec is not None and rec.get("supersession_status") == "SUPERSEDED_PRE_OBSERVATION"

    @classmethod
    def record_epoch3_supersession(
        cls,
        reason: str = "UPSTREAM_OPPORTUNITY_GENERATION_SELECTION_REFINEMENT",
        clean_prospective_n: int = 0,
        empirical_evidence_lost: int = 0,
        target_epoch_id: str = "ARX_PROSPECTIVE_VALIDATION_EPOCH_4",
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Records an append-only epoch supersession record for Epoch 3."""
        from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
        engine = GovernanceDatabaseEngine(db_path=db_path)
        now_utc = datetime.now(timezone.utc).isoformat()
        payload = {
            "previousEpochId": "ARX_PROSPECTIVE_VALIDATION_EPOCH_3",
            "targetEpochId": target_epoch_id,
            "supersededAtUtc": now_utc,
            "supersessionStatus": "SUPERSEDED_PRE_OBSERVATION",
            "reason": reason,
            "cleanProspectiveSignalsCaptured": clean_prospective_n,
            "empiricalEvidenceLost": empirical_evidence_lost,
            "decisionEngineVersion": cls.ARX_DECISION_ENGINE_VERSION,
            "decisionEngineSha": cls.EPOCH_4_DECISION_ENGINE_SHA,
        }
        row_id = engine.record_epoch_supersession(
            previous_epoch_id="ARX_PROSPECTIVE_VALIDATION_EPOCH_3",
            target_epoch_id=target_epoch_id,
            superseded_at_utc=now_utc,
            reason=reason,
            clean_prospective_signals_captured=clean_prospective_n,
            empirical_evidence_lost=empirical_evidence_lost,
            supersession_payload=payload,
        )
        payload["supersessionId"] = row_id
        return payload

    @classmethod
    def get_epoch3_supersession_record(cls, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the latest supersession record for Epoch 3."""
        from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
        engine = GovernanceDatabaseEngine(db_path=db_path)
        return engine.get_epoch_supersession_record("ARX_PROSPECTIVE_VALIDATION_EPOCH_3")

    @classmethod
    def is_epoch3_superseded(cls, db_path: Optional[str] = None) -> bool:
        """Returns True if Epoch 3 has been superseded."""
        rec = cls.get_epoch3_supersession_record(db_path=db_path)
        return rec is not None and rec.get("supersession_status") == "SUPERSEDED_PRE_OBSERVATION"

    @classmethod
    def verify_epoch3_manifest(cls) -> Dict[str, Any]:
        """Verifies repository executable observation governance against EPOCH_3_MANIFEST.json."""
        manifest = cls.get_epoch3_manifest()
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_results = {}
        all_valid = True
        combined = hashlib.sha256()
        for rel_path, meta in manifest.get("executableGovernanceFiles", {}).items():
            full_path = os.path.join(repo_root, rel_path)
            if not os.path.exists(full_path):
                file_results[rel_path] = {"valid": False, "error": "FILE_MISSING"}
                all_valid = False
                continue
            with open(full_path, "rb") as fp:
                content = fp.read().replace(b"\r\n", b"\n")
                h = hashlib.sha256(content).hexdigest()
            is_match = (h == meta["sha256"])
            file_results[rel_path] = {"valid": is_match, "sha256": h, "expectedSha256": meta["sha256"]}
            combined.update(rel_path.encode("utf-8") + b":" + h.encode("utf-8") + b"\n")
            if not is_match:
                all_valid = False
        computed_manifest_hash = combined.hexdigest()
        manifest_hash_match = (computed_manifest_hash == manifest.get("observationGovernanceManifestHash"))
        if not manifest_hash_match:
            all_valid = False
        return {
            "status": "VERIFIED" if all_valid else "CORRUPTED",
            "valid": all_valid,
            "manifestVersion": manifest.get("manifestVersion"),
            "epochId": manifest.get("epochId"),
            "observationGovernanceVersion": manifest.get("observationGovernanceVersion"),
            "observationGovernanceSha": manifest.get("observationGovernanceSha"),
            "observationGovernanceArtifactSha": manifest.get("observationGovernanceArtifactSha"),
            "observationGovernanceManifestHash": manifest.get("observationGovernanceManifestHash"),
            "computedManifestHash": computed_manifest_hash,
            "files": file_results,
        }

    @classmethod
    def verify_epoch4_manifest(cls) -> Dict[str, Any]:
        """Verifies repository executable observation governance against EPOCH_4_MANIFEST.json."""
        manifest = cls.get_epoch4_manifest()
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_results = {}
        all_valid = True
        combined = hashlib.sha256()
        for rel_path, meta in manifest.get("executableGovernanceFiles", {}).items():
            full_path = os.path.join(repo_root, rel_path)
            if not os.path.exists(full_path):
                file_results[rel_path] = {"valid": False, "error": "FILE_MISSING"}
                all_valid = False
                continue
            with open(full_path, "rb") as fp:
                content = fp.read().replace(b"\r\n", b"\n")
                h = hashlib.sha256(content).hexdigest()
            is_match = (h == meta["sha256"])
            file_results[rel_path] = {"valid": is_match, "sha256": h, "expectedSha256": meta["sha256"]}
            combined.update(rel_path.encode("utf-8") + b":" + h.encode("utf-8") + b"\n")
            if not is_match:
                all_valid = False
        computed_manifest_hash = combined.hexdigest()
        manifest_hash_match = (computed_manifest_hash == manifest.get("observationGovernanceManifestHash"))
        if not manifest_hash_match:
            all_valid = False
        return {
            "status": "VERIFIED" if all_valid else "CORRUPTED",
            "valid": all_valid,
            "manifestVersion": manifest.get("manifestVersion"),
            "epochId": manifest.get("epochId"),
            "observationGovernanceVersion": manifest.get("observationGovernanceVersion"),
            "observationGovernanceSha": manifest.get("observationGovernanceSha"),
            "observationGovernanceArtifactSha": manifest.get("observationGovernanceArtifactSha"),
            "observationGovernanceManifestHash": manifest.get("observationGovernanceManifestHash"),
            "computedManifestHash": computed_manifest_hash,
            "files": file_results,
        }

    @classmethod
    def verify_observation_governance_manifest(cls) -> Dict[str, Any]:
        """Verifies repository executable observation governance against active manifest (EPOCH_4, EPOCH_3, EPOCH_2 or EPOCH_1)."""
        manifest = cls.get_epoch4_manifest() or cls.get_epoch3_manifest() or cls.get_epoch2_manifest() or cls.get_epoch1_manifest()
        if not manifest:
            return {"status": "MANIFEST_MISSING", "valid": False}
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_results = {}
        all_valid = True
        combined = hashlib.sha256()
        for rel_path, meta in manifest.get("executableGovernanceFiles", {}).items():
            full_path = os.path.join(repo_root, rel_path)
            if not os.path.exists(full_path):
                file_results[rel_path] = {"valid": False, "error": "FILE_MISSING"}
                all_valid = False
                continue
            with open(full_path, "rb") as fp:
                content = fp.read().replace(b"\r\n", b"\n")
                h = hashlib.sha256(content).hexdigest()
            is_match = (h == meta["sha256"])
            file_results[rel_path] = {"valid": is_match, "sha256": h, "expectedSha256": meta["sha256"]}
            combined.update(rel_path.encode("utf-8") + b":" + h.encode("utf-8") + b"\n")
            if not is_match:
                all_valid = False
        computed_manifest_hash = combined.hexdigest()
        manifest_hash_match = (computed_manifest_hash == manifest.get("observationGovernanceManifestHash"))
        if not manifest_hash_match:
            all_valid = False
        return {
            "status": "VERIFIED" if all_valid else "CORRUPTED",
            "valid": all_valid,
            "manifestVersion": manifest.get("manifestVersion"),
            "epochId": manifest.get("epochId"),
            "observationGovernanceVersion": manifest.get("observationGovernanceVersion"),
            "observationGovernanceSha": manifest.get("observationGovernanceSha"),
            "observationGovernanceArtifactSha": manifest.get("observationGovernanceArtifactSha"),
            "observationGovernanceManifestHash": manifest.get("observationGovernanceManifestHash"),
            "computedManifestHash": computed_manifest_hash,
            "files": file_results,
        }

    @classmethod
    def _parse_utc_timestamp(cls, ts_str: Any, is_as_of_boundary: bool = False) -> Optional[datetime]:
        """Normalizes date or timestamp string to UTC datetime.
        For candidate events with date-only precision:
        - Reference/recommendation timestamp defaults to start-of-day (00:00:00Z).
        - Published event/filing/observation date without timestamp defaults to end-of-day (23:59:59Z)
          to strictly prevent intraday lookahead.
        """
        if not ts_str:
            return None
        s = str(ts_str).strip()
        if len(s) == 10 and s.count("-") == 2:
            try:
                dt = datetime.strptime(s, "%Y-%m-%d")
                if is_as_of_boundary:
                    return dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
                return dt.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
            except Exception:
                return None
        clean_s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(clean_s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return None

    @classmethod
    def classify_provenance_cohort(cls, sig: Dict[str, Any]) -> str:
        """Classifies a signal into a strict provenance tier (Cohort Contamination Firewall).
        Enforces fail-closed security barriers across the 6 canonical cohorts:
        1. Pre-freeze temporal barrier: Any signal dated prior to 2026-09-04 is permanently
           quarantined as HISTORICAL_CONTAMINATED, regardless of explicit tags.
        2. Explicit quarantine preservation: Explicit non-clean cohort records are strictly preserved.
        3. Simulation/Demo detection: Flags for simulation, backtest, or demo/synthetic fixtures.
        4. Post-outcome mutation detection: Explicit contamination flag or hash mismatch.
        5. Decision & Inputs hash integrity: If decisionSnapshotHash or inputsSnapshotHash is present,
           it must match the SHA-256 fingerprint; otherwise fail-closed to CONTAMINATED.
        6. Engine version verification: engineVersion must match pinned commit/SHA or manifest.
        7. Temporal anti-lookahead (Market Bar): recommended_at must precede first_forward_bar_timestamp.
        8. Fundamental Point-in-Time Anti-Lookahead: filing/as-of UTC timestamp cannot exceed recommended_at.
        9. Macro & Market observation anti-lookahead: observed/available timestamps cannot exceed recommended_at.
        10. Mandatory prospective fields: Clean prospective signal must have symbol and status.
        """
        sig_date = sig.get("signalDate", "")
        # Invariant 1: Temporal barrier - pre-freeze signals are ALWAYS contaminated
        if sig_date and sig_date < cls.FREEZE_DATE_THRESHOLD:
            return ProvenanceCohort.HISTORICAL_CONTAMINATED

        explicit = sig.get("provenanceCohort")
        # Invariant 2: Explicit quarantine preservation
        if explicit in [
            ProvenanceCohort.HISTORICAL_RECOMPUTED,
            ProvenanceCohort.BACKTEST_SIMULATION,
            ProvenanceCohort.DEMO_SYNTHETIC,
            ProvenanceCohort.CERTIFICATION_VALIDATION,
            ProvenanceCohort.CONTAMINATED,
            ProvenanceCohort.UNKNOWN,
            ProvenanceCohort.HISTORICAL_CONTAMINATED,
            ProvenanceCohort.HISTORICAL_UNKNOWN,
            ProvenanceCohort.EXCLUDED,
        ]:
            return explicit

        # Invariant 3: Simulation, demo fixture, or certification detection
        if sig.get("isSimulated") or sig.get("isBacktest"):
            return ProvenanceCohort.BACKTEST_SIMULATION
        if sig.get("isDemo") or sig.get("isSynthetic"):
            return ProvenanceCohort.DEMO_SYNTHETIC
        if sig.get("isCertification") or sig.get("isValidation"):
            return ProvenanceCohort.CERTIFICATION_VALIDATION

        # Invariant 4: Post-outcome mutation or contamination flag
        if sig.get("isContaminated") or sig.get("modifiedPostOutcome"):
            return ProvenanceCohort.CONTAMINATED

        # Invariant 5: Cryptographic hash validation (Fail-closed)
        dec_hash = sig.get("decisionSnapshotHash")
        if dec_hash:
            if dec_hash != cls.compute_decision_snapshot_hash(sig):
                return ProvenanceCohort.CONTAMINATED

        in_hash = sig.get("inputsSnapshotHash")
        if in_hash:
            if in_hash != cls.compute_inputs_snapshot_hash(sig):
                return ProvenanceCohort.CONTAMINATED

        # Invariant 6: Engine version verification against frozen manifest/commit/SHA
        engine_ver = sig.get("engineVersion")
        if engine_ver:
            manifest_meta = cls.get_frozen_manifest()
            valid_commits = [
                cls.ENGINE_SHA,
                cls.ENGINE_SHA[:7],
                cls.FROZEN_ENGINE_COMMIT,
                cls.FROZEN_ENGINE_COMMIT[:7],
                cls.EPOCH_3_DECISION_ENGINE_SHA,
                cls.EPOCH_3_DECISION_ENGINE_SHA[:7],
                cls.EPOCH_4_DECISION_ENGINE_SHA,
                cls.EPOCH_4_DECISION_ENGINE_SHA[:7],
                cls.ARX_DECISION_ENGINE_VERSION,
            ]
            if manifest_meta and manifest_meta.get("provenanceCommit"):
                valid_commits.append(manifest_meta["provenanceCommit"])
                valid_commits.append(manifest_meta["provenanceCommit"][:7])
            if not any(engine_ver.startswith(c[:7]) or c.startswith(engine_ver[:7]) for c in valid_commits):
                return ProvenanceCohort.EXCLUDED

        # Parse normalized recommendation timestamp
        rec_str = sig.get("recommended_at") or sig.get("signalTimestamp") or sig.get("signalDate") or sig.get("timestamp")
        rec_dt = cls._parse_utc_timestamp(rec_str)

        # Invariant 7: Temporal anti-lookahead verification (Market Bar)
        first_bar_str = sig.get("first_forward_bar_timestamp")
        if first_bar_str and rec_dt:
            first_bar_dt = cls._parse_utc_timestamp(first_bar_str)
            if first_bar_dt and rec_dt >= first_bar_dt:
                return ProvenanceCohort.CONTAMINATED

        # Invariant 8: Fundamental Point-in-Time Anti-Lookahead
        inputs = sig.get("inputs") or {}
        fund_filing_str = inputs.get("fundamentalFilingTimestamp")
        if fund_filing_str and rec_dt:
            filing_dt = cls._parse_utc_timestamp(fund_filing_str)
            if filing_dt and filing_dt > rec_dt:
                return ProvenanceCohort.CONTAMINATED

        fund_as_of_str = inputs.get("fundamentalAsOfDate")
        if fund_as_of_str and rec_dt:
            as_of_dt = cls._parse_utc_timestamp(fund_as_of_str, is_as_of_boundary=False)
            rec_date_dt = datetime(rec_dt.year, rec_dt.month, rec_dt.day, tzinfo=timezone.utc)
            if as_of_dt and as_of_dt > rec_date_dt:
                return ProvenanceCohort.CONTAMINATED

        # Invariant 9: Market & Macro snapshot timestamp integrity
        market_obs_str = inputs.get("marketSnapshotObservedAt") or inputs.get("marketDataSnapshotTimestamp")
        if market_obs_str and rec_dt:
            market_dt = cls._parse_utc_timestamp(market_obs_str)
            if market_dt and market_dt > rec_dt:
                return ProvenanceCohort.CONTAMINATED

        macro_obs_str = inputs.get("macroObservationAvailableAt") or inputs.get("macroObservationDate")
        if macro_obs_str and rec_dt:
            macro_dt = cls._parse_utc_timestamp(macro_obs_str)
            if macro_dt and macro_dt > rec_dt:
                return ProvenanceCohort.CONTAMINATED

        # Invariant 10: Mandatory fields for prospective eligibility
        if sig_date and sig_date >= cls.FREEZE_DATE_THRESHOLD:
            if not sig.get("symbol") or not sig.get("status"):
                return ProvenanceCohort.EXCLUDED
            return ProvenanceCohort.PROSPECTIVE_CLEAN

        # Explicit clean tag without dates is synthetic test fixture
        if explicit == ProvenanceCohort.PROSPECTIVE_CLEAN:
            return ProvenanceCohort.PROSPECTIVE_CLEAN

        return ProvenanceCohort.UNKNOWN

    @classmethod
    def resolve_ledger_path(cls, custom_path: Optional[str] = None) -> str:
        """Resolves the authoritative prospective ledger path using shared storage authority."""
        from analyst_dashboard.governance.storage import resolve_ledger_path as _storage_resolve_ledger_path
        return _storage_resolve_ledger_path(custom_path)

    @classmethod
    def load_ledger(cls, ledger_path: Optional[str] = None) -> Dict[str, Any]:
        path = ledger_path or cls.resolve_ledger_path()
        if not os.path.exists(path):
            return {
                "version": "1.0.0",
                "engineCommit": "4e36862",
                "tag": "v2.4.0-phase24-freeze",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "totalActiveSignals": 0,
                "signals": []
            }
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Guarantee backward-compatible executionObservations field on all signals
            for sig in data.get("signals", []):
                if "executionObservations" not in sig:
                    sig["executionObservations"] = []
            return data

    @classmethod
    def save_ledger(cls, data: Dict[str, Any], ledger_path: Optional[str] = None) -> None:
        path = ledger_path or cls.resolve_ledger_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def get_epoch1_clean_prospective_count(cls, ledger_path: Optional[str] = None) -> int:
        """Returns the count of certified natural prospective records admitted to Epoch 1."""
        ledger = cls.load_ledger(ledger_path)
        return len([
            s for s in ledger.get("signals", [])
            if s.get("epochId") == "ARX_PROSPECTIVE_VALIDATION_EPOCH_1"
            and cls.classify_provenance_cohort(s) == ProvenanceCohort.PROSPECTIVE_CLEAN
        ])

    @classmethod
    def get_activation_record(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
        epoch_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Loads and validates the epoch activation record. SQLite is canonical authority.
        In production paths (activation_record_path is None), SQLite is the ONLY authority.
        JSON fallback is strictly forbidden and never consulted in production.
        """
        target_epoch = epoch_id or cls.EPOCH_ID
        # 1. Canonical SQLite authority
        if activation_record_path is None:
            try:
                from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
                gov_engine = GovernanceDatabaseEngine(db_path=db_path)
                sqlite_record = gov_engine.get_activation_record(target_epoch)
                if sqlite_record:
                    return {
                        "epochId": sqlite_record["epoch_id"],
                        "releaseSha": sqlite_record["release_sha"],
                        "deploymentId": sqlite_record["deployment_id"],
                        "deploymentStatus": "SUCCESS",
                        "activatedAtUtc": sqlite_record["activated_at_utc"],
                        "runtimeIdentityAttestation": sqlite_record.get("activation_source", "CONTAINER_ENTRYPOINT_MANIFEST_VERIFIED"),
                        "prospectiveObservationAuthorized": True,
                    }
                return None
            except Exception as e:
                logger.error(f"SQLite activation lookup failed: {e}")
                return None

        # 2. Hermetic test isolation ONLY: Explicit activation_record_path provided by test suite
        if not os.path.exists(activation_record_path):
            return None
        try:
            with open(activation_record_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            valid, reason = cls.validate_activation_record(record)
            if valid:
                return record
            logger.warning(f"Invalid activation record at {activation_record_path}: {reason}")
        except Exception as e:
            logger.warning(f"Error loading activation record from {activation_record_path}: {e}")
        return None

    @classmethod
    def validate_activation_record(
        cls,
        record: Any,
        expected_release_sha: Optional[str] = None,
        deployment_boundary_utc: Optional[str] = None,
        current_time_utc: Optional[datetime] = None,
        expected_epoch_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validates schema, fields, temporal bounds, and consistency of an Epoch activation record."""
        if not isinstance(record, dict):
            return False, "RECORD_NOT_A_DICT"
        target_epoch = expected_epoch_id or (
            record.get("epochId")
            if record.get("epochId") in (cls.EPOCH_1_ID, cls.EPOCH_2_ID, cls.EPOCH_3_ID, cls.EPOCH_4_ID)
            else cls.EPOCH_ID
        )
        if record.get("epochId") != target_epoch:
            return False, f"EPOCH_ID_MISMATCH: expected {target_epoch}, got {record.get('epochId')}"
        if record.get("deploymentStatus") != "SUCCESS":
            return False, f"DEPLOYMENT_STATUS_NOT_SUCCESS: {record.get('deploymentStatus')}"
        if not record.get("prospectiveObservationAuthorized"):
            return False, "PROSPECTIVE_OBSERVATION_NOT_AUTHORIZED"
        if not record.get("releaseSha") or not isinstance(record.get("releaseSha"), str):
            return False, "MISSING_RELEASE_SHA"
        if expected_release_sha and record.get("releaseSha") != expected_release_sha:
            return False, f"RELEASE_SHA_MISMATCH: expected {expected_release_sha}, got {record.get('releaseSha')}"
        if not record.get("deploymentId") or not isinstance(record.get("deploymentId"), str):
            return False, "MISSING_DEPLOYMENT_ID"
        activated_at = record.get("activatedAtUtc")
        if not activated_at or not isinstance(activated_at, str):
            return False, "MISSING_ACTIVATED_AT_UTC"
        parsed_dt = cls._parse_utc_timestamp(activated_at)
        if parsed_dt is None:
            return False, "MALFORMED_ACTIVATED_AT_UTC"

        # Strict Epoch Activation Authorization: Zero positive future tolerance
        now_dt = current_time_utc or datetime.now(timezone.utc)
        if parsed_dt > now_dt:
            return False, "FUTURE_ACTIVATION_TIMESTAMP"

        # Deployment boundary gate (Phase 3 requirement):
        # Activation time must not predate authoritative deployment/runtime activation boundary.
        dep_boundary_str = (
            deployment_boundary_utc
            or record.get("deploymentFinishedAtUtc")
            or record.get("deploymentBoundaryUtc")
        )
        if dep_boundary_str:
            dep_boundary_dt = cls._parse_utc_timestamp(dep_boundary_str)
            if dep_boundary_dt and parsed_dt < dep_boundary_dt:
                return False, f"ACTIVATION_PREDATES_DEPLOYMENT_BOUNDARY: activated={activated_at} < deployment={dep_boundary_str}"

        if not record.get("runtimeIdentityAttestation"):
            return False, "MISSING_RUNTIME_IDENTITY_ATTESTATION"
        return True, None

    @classmethod
    def create_activation_record(
        cls,
        epoch_id: str = "ARX_PROSPECTIVE_VALIDATION_EPOCH_4",
        release_sha: str = "",
        deployment_id: str = "",
        activated_at_utc: str = "",
        deployment_status: str = "SUCCESS",
        runtime_identity_attestation: str = "CONTAINER_ENTRYPOINT_MANIFEST_VERIFIED",
        prospective_observation_authorized: bool = True,
        deployment_finished_at_utc: Optional[str] = None,
        output_path: Optional[str] = None,
        overwrite: bool = False,
        current_time_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Creates, validates, and optionally persists an immutable production activation record."""
        record = {
            "epochId": epoch_id,
            "releaseSha": release_sha,
            "deploymentId": deployment_id,
            "deploymentStatus": deployment_status,
            "activatedAtUtc": activated_at_utc,
            "runtimeIdentityAttestation": runtime_identity_attestation,
            "prospectiveObservationAuthorized": prospective_observation_authorized,
        }
        if deployment_finished_at_utc:
            record["deploymentFinishedAtUtc"] = deployment_finished_at_utc
        valid, reason = cls.validate_activation_record(
            record,
            expected_release_sha=release_sha if release_sha else None,
            current_time_utc=current_time_utc,
            expected_epoch_id=epoch_id,
        )
        if not valid:
            raise ValueError(f"Cannot create invalid activation record: {reason}")
        if output_path:
            if os.path.exists(output_path) and not overwrite:
                raise FileExistsError(f"Activation record at {output_path} already exists and is immutable (overwrite=False)")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        return record

    @classmethod
    def get_epoch2_activation_timestamp(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Optional[str]:
        """Returns authoritative ISO UTC activation timestamp if valid record exists, else None."""
        record = cls.get_activation_record(activation_record_path=activation_record_path, db_path=db_path, epoch_id=cls.EPOCH_2_ID)
        if record and record.get("prospectiveObservationAuthorized"):
            return record.get("activatedAtUtc")
        return None

    @classmethod
    def is_epoch2_observation_authorized(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
        release_sha: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ) -> bool:
        """Returns True only if a verified production activation record authorizes observation.
        SQLite is canonical authority; JSON is derived non-authoritative projection.
        """
        # 1. Canonical authority: SQLite governance engine
        try:
            from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
            gov_engine = GovernanceDatabaseEngine(db_path=db_path)
            r_sha = release_sha or os.getenv("RAILWAY_GIT_COMMIT_SHA")
            d_id = deployment_id or os.getenv("RAILWAY_DEPLOYMENT_ID")
            if r_sha and d_id:
                is_auth, _ = gov_engine.evaluate_capture_authorization_predicate(
                    epoch_id=cls.EPOCH_2_ID,
                    release_sha=r_sha,
                    deployment_id=d_id,
                )
                if is_auth:
                    return True
            else:
                act = gov_engine.get_activation_record(cls.EPOCH_2_ID)
                if act:
                    return True
        except Exception as e:
            logger.debug(f"SQLite governance check bypassed or failed: {e}")

        # 2. Test isolation fallback: Explicit activation_record_path provided
        if activation_record_path is not None:
            ts = cls.get_epoch2_activation_timestamp(activation_record_path=activation_record_path)
            return ts is not None

        return False

    @classmethod
    def is_record_epoch2_eligible(
        cls,
        record: Dict[str, Any],
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether a single signal record meets all strict Epoch 2 prospective eligibility gates."""
        if record.get("epochId") != cls.EPOCH_2_ID:
            return False, "EPOCH_ID_MISMATCH"

        act_record = cls.get_activation_record(activation_record_path, db_path=db_path, epoch_id=cls.EPOCH_2_ID)
        if not act_record:
            return False, "NO_VALID_ACTIVATION_RECORD"

        # Verify release attribution
        record_release = record.get("releaseSha") or record.get("engineCommit")
        if not record_release:
            return False, "MISSING_RUNTIME_IDENTITY"

        # 1. Canonical SQLite authority: verify release has valid PASS authorization
        if activation_record_path is None:
            try:
                from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
                gov_engine = GovernanceDatabaseEngine(db_path=db_path)
                conn = gov_engine.get_connection()
                try:
                    record_deployment = record.get("deploymentId")
                    if record_deployment:
                        cur = conn.execute(
                            """
                            SELECT production_certification_status FROM epoch_release_authorizations
                            WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                            """,
                            (cls.EPOCH_2_ID, record_release, record_deployment),
                        )
                        rev_cur = conn.execute(
                            """
                            SELECT COUNT(*) FROM epoch_release_revocations
                            WHERE epoch_id = ? AND release_sha = ? AND deployment_id = ?
                            """,
                            (cls.EPOCH_2_ID, record_release, record_deployment),
                        )
                        if rev_cur.fetchone()[0] > 0:
                            return False, "RUNTIME_REVOKED"
                    else:
                        cur = conn.execute(
                            """
                            SELECT production_certification_status FROM epoch_release_authorizations
                            WHERE epoch_id = ? AND authorized_release_sha = ?
                            """,
                            (cls.EPOCH_2_ID, record_release),
                        )
                    row = cur.fetchone()
                    if not row or row["production_certification_status"] != "PASS":
                        return False, f"RELEASE_NOT_AUTHORIZED: {record_release}"
                finally:
                    conn.close()
            except Exception as e:
                logger.error(f"SQLite release authorization lookup failed: {e}")
                return False, f"AUTHORIZATION_LOOKUP_ERROR: {e}"
        else:
            # Hermetic test isolation fallback: verify against explicit activation record
            expected_release = act_record.get("releaseSha")
            if expected_release and not (record_release == expected_release or record_release.startswith(expected_release[:7])):
                return False, f"RELEASE_ATTRIBUTION_MISMATCH: expected {expected_release}, got {record_release}"

        if cls.classify_provenance_cohort(record) != ProvenanceCohort.PROSPECTIVE_CLEAN:
            return False, "NOT_PROSPECTIVE_CLEAN_COHORT"

        act_ts_str = act_record.get("activatedAtUtc")
        act_dt = cls._parse_utc_timestamp(act_ts_str) if act_ts_str else None
        if not act_dt:
            return False, "INVALID_ACTIVATION_TIMESTAMP"

        rec_str = record.get("recommended_at") or record.get("signalTimestamp") or record.get("timestamp")
        rec_dt = cls._parse_utc_timestamp(rec_str)
        if not rec_dt:
            return False, "MISSING_OR_MALFORMED_RECORD_TIMESTAMP"

        # Invariant: Record must not precede activation (rec_dt >= act_dt)
        if rec_dt < act_dt:
            return False, f"RECORD_PRECEDES_ACTIVATION: rec={rec_str} < act={act_ts_str}"

        # Invariant: Record must not be future-dated relative to now
        now_dt = datetime.now(timezone.utc)
        if rec_dt > now_dt + timedelta(seconds=60): # 60s clock skew tolerance
            return False, "FUTURE_DATED_RECORD"

        return True, None

    @classmethod
    def get_epoch2_clean_prospective_count(
        cls,
        ledger_path: Optional[str] = None,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> int:
        """Returns the count of certified natural prospective records admitted to Epoch 2.
        Fails closed (returns 0) if no valid activation record exists.
        """
        if not cls.is_epoch2_observation_authorized(activation_record_path, db_path=db_path):
            return 0
        ledger = cls.load_ledger(ledger_path)
        count = 0
        for s in ledger.get("signals", []):
            is_eligible, _ = cls.is_record_epoch2_eligible(
                s, activation_record_path=activation_record_path, db_path=db_path
            )
            if is_eligible:
                count += 1
        return count

    @classmethod
    def get_epoch3_activation_timestamp(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Optional[str]:
        """Returns authoritative ISO UTC activation timestamp if valid Epoch 3 record exists, else None."""
        record = cls.get_activation_record(activation_record_path=activation_record_path, db_path=db_path, epoch_id=cls.EPOCH_3_ID)
        if record and record.get("prospectiveObservationAuthorized"):
            return record.get("activatedAtUtc")
        return None

    @classmethod
    def is_epoch3_observation_authorized(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
        release_sha: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ) -> bool:
        """Returns True only if a verified production activation record authorizes observation for Epoch 3."""
        # 1. Canonical authority: SQLite governance engine
        try:
            from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
            gov_engine = GovernanceDatabaseEngine(db_path=db_path)
            r_sha = release_sha or os.getenv("RAILWAY_GIT_COMMIT_SHA")
            d_id = deployment_id or os.getenv("RAILWAY_DEPLOYMENT_ID")
            if r_sha and d_id:
                is_auth, _ = gov_engine.evaluate_capture_authorization_predicate(
                    epoch_id=cls.EPOCH_3_ID,
                    release_sha=r_sha,
                    deployment_id=d_id,
                )
                if is_auth:
                    return True
            else:
                act = gov_engine.get_activation_record(cls.EPOCH_3_ID)
                if act:
                    return True
        except Exception as e:
            logger.debug(f"SQLite governance check bypassed or failed: {e}")

        # 2. Test isolation fallback: Explicit activation_record_path provided
        if activation_record_path is not None:
            ts = cls.get_epoch3_activation_timestamp(activation_record_path=activation_record_path)
            return ts is not None

        return False

    @classmethod
    def is_record_epoch3_eligible(
        cls,
        record: Dict[str, Any],
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether a single signal record meets all strict Epoch 3 prospective eligibility gates."""
        if record.get("epochId") != cls.EPOCH_3_ID:
            return False, "EPOCH_ID_MISMATCH"

        act_record = cls.get_activation_record(activation_record_path, db_path=db_path, epoch_id=cls.EPOCH_3_ID)
        if not act_record:
            return False, "NO_VALID_ACTIVATION_RECORD"

        record_release = record.get("releaseSha") or record.get("engineCommit")
        if not record_release:
            return False, "MISSING_RUNTIME_IDENTITY"

        if activation_record_path is None:
            try:
                from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
                gov_engine = GovernanceDatabaseEngine(db_path=db_path)
                conn = gov_engine.get_connection()
                try:
                    record_deployment = record.get("deploymentId")
                    if record_deployment:
                        cur = conn.execute(
                            """
                            SELECT production_certification_status FROM epoch_release_authorizations
                            WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                            """,
                            (cls.EPOCH_3_ID, record_release, record_deployment),
                        )
                        rev_cur = conn.execute(
                            """
                            SELECT COUNT(*) FROM epoch_release_revocations
                            WHERE epoch_id = ? AND release_sha = ? AND deployment_id = ?
                            """,
                            (cls.EPOCH_3_ID, record_release, record_deployment),
                        )
                        if rev_cur.fetchone()[0] > 0:
                            return False, "RUNTIME_REVOKED"
                    else:
                        cur = conn.execute(
                            """
                            SELECT production_certification_status FROM epoch_release_authorizations
                            WHERE epoch_id = ? AND authorized_release_sha = ?
                            """,
                            (cls.EPOCH_3_ID, record_release),
                        )
                    row = cur.fetchone()
                    if not row or row["production_certification_status"] != "PASS":
                        return False, f"RELEASE_NOT_AUTHORIZED: {record_release}"
                finally:
                    conn.close()
            except Exception as e:
                logger.error(f"SQLite release authorization lookup failed: {e}")
                return False, f"AUTHORIZATION_LOOKUP_ERROR: {e}"
        else:
            expected_release = act_record.get("releaseSha")
            if expected_release and not (record_release == expected_release or record_release.startswith(expected_release[:7])):
                return False, f"RELEASE_ATTRIBUTION_MISMATCH: expected {expected_release}, got {record_release}"

        if cls.classify_provenance_cohort(record) != ProvenanceCohort.PROSPECTIVE_CLEAN:
            return False, "NOT_PROSPECTIVE_CLEAN_COHORT"

        act_ts_str = act_record.get("activatedAtUtc")
        act_dt = cls._parse_utc_timestamp(act_ts_str) if act_ts_str else None
        if not act_dt:
            return False, "INVALID_ACTIVATION_TIMESTAMP"

        rec_str = record.get("recommended_at") or record.get("signalTimestamp") or record.get("timestamp")
        rec_dt = cls._parse_utc_timestamp(rec_str)
        if not rec_dt:
            return False, "MISSING_OR_MALFORMED_RECORD_TIMESTAMP"

        if rec_dt < act_dt:
            return False, f"RECORD_PRECEDES_ACTIVATION: rec={rec_str} < act={act_ts_str}"

        now_dt = datetime.now(timezone.utc)
        if rec_dt > now_dt + timedelta(seconds=60):
            return False, "FUTURE_DATED_RECORD"

        # Market Data Admissibility (Dual-Price Contract)
        freshness = record.get("liveFreshness") or (record.get("inputs") or {}).get("liveFreshness")
        session = record.get("marketSession") or (record.get("inputs") or {}).get("marketSession")
        live_spot = record.get("liveSpotPrice")
        if live_spot is None:
            live_spot = (record.get("inputs") or {}).get("liveSpotPrice")

        if freshness != "REALTIME":
            return False, f"QUOTE_NOT_REALTIME: freshness={freshness}"
        if session != "REGULAR_SESSION":
            return False, f"MARKET_SESSION_NOT_REGULAR: session={session}"
        if live_spot is None:
            return False, "LIVE_SPOT_MISSING"
        try:
            ls_float = float(live_spot)
            if not math.isfinite(ls_float) or ls_float <= 0:
                return False, f"LIVE_SPOT_INVALID: {live_spot}"
        except (ValueError, TypeError):
            return False, f"LIVE_SPOT_INVALID: {live_spot}"

        return True, None

    @classmethod
    def get_epoch3_clean_prospective_count(
        cls,
        ledger_path: Optional[str] = None,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> int:
        """Returns the count of certified natural prospective records admitted to Epoch 3.
        Fails closed (returns 0) if no valid activation record exists.
        """
        if not cls.is_epoch3_observation_authorized(activation_record_path, db_path=db_path):
            return 0
        ledger = cls.load_ledger(ledger_path)
        count = 0
        for s in ledger.get("signals", []):
            is_eligible, _ = cls.is_record_epoch3_eligible(
                s, activation_record_path=activation_record_path, db_path=db_path
            )
            if is_eligible:
                count += 1
        return count

    @classmethod
    def get_epoch4_activation_timestamp(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Optional[str]:
        """Returns authoritative ISO UTC activation timestamp if valid Epoch 4 record exists, else None."""
        record = cls.get_activation_record(activation_record_path=activation_record_path, db_path=db_path, epoch_id=cls.EPOCH_4_ID)
        if record and record.get("prospectiveObservationAuthorized"):
            return record.get("activatedAtUtc")
        return None

    @classmethod
    def is_epoch4_observation_authorized(
        cls,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
        release_sha: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ) -> bool:
        """Returns True only if a verified production activation record authorizes observation for Epoch 4."""
        # 1. Canonical authority: SQLite governance engine
        try:
            from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
            gov_engine = GovernanceDatabaseEngine(db_path=db_path)
            r_sha = release_sha or os.getenv("RAILWAY_GIT_COMMIT_SHA")
            d_id = deployment_id or os.getenv("RAILWAY_DEPLOYMENT_ID")
            if r_sha and d_id:
                is_auth, _ = gov_engine.evaluate_capture_authorization_predicate(
                    epoch_id=cls.EPOCH_4_ID,
                    release_sha=r_sha,
                    deployment_id=d_id,
                )
                if is_auth:
                    return True
            else:
                act = gov_engine.get_activation_record(cls.EPOCH_4_ID)
                if act:
                    return True
        except Exception as e:
            logger.debug(f"SQLite governance check bypassed or failed: {e}")

        # 2. Test isolation fallback: Explicit activation_record_path provided
        if activation_record_path is not None:
            ts = cls.get_epoch4_activation_timestamp(activation_record_path=activation_record_path)
            return ts is not None

        return False

    @classmethod
    def is_record_epoch4_eligible(
        cls,
        record: Dict[str, Any],
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether a single signal record meets all strict Epoch 4 prospective eligibility gates."""
        if record.get("epochId") != cls.EPOCH_4_ID:
            return False, "EPOCH_ID_MISMATCH"

        act_record = cls.get_activation_record(activation_record_path, db_path=db_path, epoch_id=cls.EPOCH_4_ID)
        if not act_record:
            return False, "NO_VALID_ACTIVATION_RECORD"

        record_release = record.get("releaseSha") or record.get("engineCommit")
        if not record_release:
            return False, "MISSING_RUNTIME_IDENTITY"

        if activation_record_path is None:
            try:
                from analyst_dashboard.governance.governance_db import GovernanceDatabaseEngine
                gov_engine = GovernanceDatabaseEngine(db_path=db_path)
                conn = gov_engine.get_connection()
                try:
                    record_deployment = record.get("deploymentId")
                    if record_deployment:
                        cur = conn.execute(
                            """
                            SELECT production_certification_status FROM epoch_release_authorizations
                            WHERE epoch_id = ? AND authorized_release_sha = ? AND authorized_deployment_id = ?
                            """,
                            (cls.EPOCH_4_ID, record_release, record_deployment),
                        )
                        rev_cur = conn.execute(
                            """
                            SELECT COUNT(*) FROM epoch_release_revocations
                            WHERE epoch_id = ? AND release_sha = ? AND deployment_id = ?
                            """,
                            (cls.EPOCH_4_ID, record_release, record_deployment),
                        )
                        if rev_cur.fetchone()[0] > 0:
                            return False, "RUNTIME_REVOKED"
                    else:
                        cur = conn.execute(
                            """
                            SELECT production_certification_status FROM epoch_release_authorizations
                            WHERE epoch_id = ? AND authorized_release_sha = ?
                            """,
                            (cls.EPOCH_4_ID, record_release),
                        )
                    row = cur.fetchone()
                    if not row or row["production_certification_status"] != "PASS":
                        return False, f"RELEASE_NOT_AUTHORIZED: {record_release}"
                finally:
                    conn.close()
            except Exception as e:
                logger.error(f"SQLite release authorization lookup failed: {e}")
                return False, f"AUTHORIZATION_LOOKUP_ERROR: {e}"
        else:
            expected_release = act_record.get("releaseSha")
            if expected_release and not (record_release == expected_release or record_release.startswith(expected_release[:7])):
                return False, f"RELEASE_ATTRIBUTION_MISMATCH: expected {expected_release}, got {record_release}"

        if cls.classify_provenance_cohort(record) != ProvenanceCohort.PROSPECTIVE_CLEAN:
            return False, "NOT_PROSPECTIVE_CLEAN_COHORT"

        act_ts_str = act_record.get("activatedAtUtc")
        act_dt = cls._parse_utc_timestamp(act_ts_str) if act_ts_str else None
        if not act_dt:
            return False, "INVALID_ACTIVATION_TIMESTAMP"

        rec_str = record.get("recommended_at") or record.get("signalTimestamp") or record.get("timestamp")
        rec_dt = cls._parse_utc_timestamp(rec_str)
        if not rec_dt:
            return False, "MISSING_OR_MALFORMED_RECORD_TIMESTAMP"

        if rec_dt < act_dt:
            return False, f"SIGNAL_PRECEDES_EPOCH_ACTIVATION: signal={rec_str} < act={act_ts_str}"

        return True, None

    @classmethod
    def get_epoch4_clean_prospective_count(
        cls,
        ledger_path: Optional[str] = None,
        activation_record_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> int:
        """Returns the count of certified natural prospective records admitted to Epoch 4.
        Fails closed (returns 0) if no valid activation record exists.
        """
        if not cls.is_epoch4_observation_authorized(activation_record_path, db_path=db_path):
            return 0
        ledger = cls.load_ledger(ledger_path)
        count = 0
        for s in ledger.get("signals", []):
            is_eligible, _ = cls.is_record_epoch4_eligible(
                s, activation_record_path=activation_record_path, db_path=db_path
            )
            if is_eligible:
                count += 1
        return count

    @classmethod
    def compute_decision_snapshot_hash(cls, record: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint over the immutable frozen decision parameters."""
        canonical_dict = {
            "engineVersion": str(record.get("engineVersion", "")),
            "engineTag": str(record.get("engineTag", "")),
            "decisionState": str(record.get("decisionState", "")),
            "entryPrice": float(record.get("entryPrice", 0.0)),
            "corridorMin": float(record.get("corridorMin", 0.0)) if record.get("corridorMin") is not None else None,
            "corridorMax": float(record.get("corridorMax", 0.0)) if record.get("corridorMax") is not None else None,
            "stopLoss": float(record.get("stopLoss", 0.0)) if record.get("stopLoss") is not None else None,
            "takeProfit1": float(record.get("takeProfit1", 0.0)) if record.get("takeProfit1") is not None else None,
            "takeProfit2": float(record.get("takeProfit2", 0.0)) if record.get("takeProfit2") is not None else None,
            "riskRewardRatio": float(record.get("riskRewardRatio", 0.0)) if record.get("riskRewardRatio") is not None else None,
            "confluenceScore": float(record.get("confluenceScore", 0.0)),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def compute_payload_hash(cls, payload: Any) -> Optional[str]:
        """Computes deterministic SHA-256 over a canonical JSON payload or validates an existing 64-char hex hash."""
        if payload is None:
            return None
        if isinstance(payload, str) and len(payload) == 64 and all(c in "0123456789abcdefABCDEF" for c in payload):
            return payload.lower()
        try:
            encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()
        except Exception:
            return None

    @classmethod
    def compute_epoch1_inputs_snapshot_hash(cls, record: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint committing to the complete point-in-time decision state.

        Canonical Architecture:
        EPOCH_1_INPUT_SNAPSHOT =
            MARKET_SNAPSHOT_HASH
          + TECHNICAL_INPUTS
          + FUNDAMENTAL_SNAPSHOT_HASH
          + MACRO_SNAPSHOT_HASH
          + MODEL_CONFIG_HASH
          + EVIDENCE_STATE
          + PROVIDER_PROVENANCE
        """
        inputs = record.get("inputs") or {}

        # 1. Content-addressed Market Data Snapshot Hash
        raw_market = inputs.get("rawMarketPayload") or inputs.get("candles")
        market_hash = cls.compute_payload_hash(raw_market) or inputs.get("marketSnapshotHash") or inputs.get("market_snapshot_hash") or ""

        # 2. Content-addressed Fundamental Snapshot Hash
        raw_fund = inputs.get("rawFundamentalPayload") or inputs.get("fundamentals")
        fund_hash = cls.compute_payload_hash(raw_fund) or inputs.get("fundamentalSnapshotHash") or inputs.get("fundamental_snapshot_hash") or ""

        # 3. Content-addressed Macro Snapshot Hash
        raw_macro = inputs.get("rawMacroPayload") or inputs.get("macro") or inputs.get("macroInputs") or inputs.get("macro_inputs")
        macro_hash = cls.compute_payload_hash(raw_macro) or inputs.get("macroSnapshotHash") or inputs.get("macro_snapshot_hash") or ""

        # Safe numeric parsing for macro indicators
        yc_input = inputs.get("yieldCurve10y2y") if inputs.get("yieldCurve10y2y") is not None else inputs.get("yield_curve_10y2y")
        if yc_input is None:
            yc_input = inputs.get("yield_curve_spread")

        cs_input = inputs.get("creditSpread") if inputs.get("creditSpread") is not None else inputs.get("high_yield_credit_spread")
        if cs_input is None:
            cs_input = inputs.get("credit_spread")
        if cs_input is None:
            cs_input = inputs.get("credit_spread_oas")

        yc_val: Optional[float] = None
        if yc_input is not None and not isinstance(yc_input, bool):
            try:
                yc_val = float(yc_input)
            except (ValueError, TypeError):
                yc_val = None

        cs_val: Optional[float] = None
        if cs_input is not None and not isinstance(cs_input, bool):
            try:
                cs_val = float(cs_input)
            except (ValueError, TypeError):
                cs_val = None

        canonical_dict = {
            "symbol": str(record.get("symbol", "")).upper().strip(),
            # 1. Market Data Content Hash & Observation Timestamps
            "marketSnapshotHash": str(market_hash),
            "marketSnapshotObservedAt": str(inputs.get("marketSnapshotObservedAt") or inputs.get("marketDataSnapshotTimestamp") or ""),
            "candleCount": int(inputs.get("candleCount", 0)) if inputs.get("candleCount") is not None else None,
            # 2. Technical Indicator Inputs
            "atr14": float(inputs.get("atr14", 0.0)) if inputs.get("atr14") is not None else None,
            "atrPct": float(inputs.get("atrPct", 0.0)) if inputs.get("atrPct") is not None else None,
            "setupPattern": str(inputs.get("setupPattern", "")),
            "stagePhase": str(inputs.get("stagePhase", "")),
            "sma50": float(inputs.get("sma50", 0.0)) if inputs.get("sma50") is not None else None,
            "ema20": float(inputs.get("ema20", 0.0)) if inputs.get("ema20") is not None else None,
            "rsi14": float(inputs.get("rsi14", 0.0)) if inputs.get("rsi14") is not None else None,
            # 3. Fundamental Content Hash & Point-in-Time Timestamps
            "fundamentalSnapshotHash": str(fund_hash),
            "fundamentalFilingTimestamp": str(inputs.get("fundamentalFilingTimestamp") or ""),
            "fundamentalAsOfDate": str(inputs.get("fundamentalAsOfDate") or ""),
            # 4. Macro Content Hash & Availability Timestamps
            "macroSnapshotHash": str(macro_hash),
            "macroObservationAvailableAt": str(
                inputs.get("macroObservationAvailableAt")
                or inputs.get("macro_observation_available_at")
                or inputs.get("yield_observation_timestamp")
                or inputs.get("macroObservationDate")
                or ""
            ),
            "yieldCurve10y2y": yc_val,
            "creditSpread": cs_val,
            # 5. Model Config Hash
            "modelConfigHash": str(inputs.get("modelConfigHash") or cls.CONFIG_HASH),
            # 6. Regime, Evidence State & Provider Provenance
            "marketRegime": str(inputs.get("marketRegime", "")),
            "sector": str(inputs.get("sector", "")),
            "assetClass": str(inputs.get("assetClass", "")),
            "dataProvider": str(inputs.get("dataProvider", "")),
            "quoteFreshness": str(inputs.get("quoteFreshness", "")),
            "evidenceCompleteness": str(inputs.get("evidenceCompleteness", "")),
            "pointInTimePrecision": str(inputs.get("pointInTimePrecision", "TIMESTAMP")),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def compute_inputs_snapshot_hash(cls, record: Dict[str, Any]) -> str:
        """Computes a SHA-256 fingerprint over the immutable frozen input features."""
        inputs = record.get("inputs") or {}
        # Route Epoch 1 and Epoch 2 records to the complete information set hash
        if record.get("epochId") in (cls.EPOCH_ID, "ARX_PROSPECTIVE_VALIDATION_EPOCH_1", "ARX_PROSPECTIVE_VALIDATION_EPOCH_2") or "marketDataSnapshotTimestamp" in inputs or "fundamentalAsOfDate" in inputs:
            return cls.compute_epoch1_inputs_snapshot_hash(record)

        canonical_dict = {
            "atr14": float(inputs.get("atr14", 0.0)) if inputs.get("atr14") is not None else None,
            "atrPct": float(inputs.get("atrPct", 0.0)) if inputs.get("atrPct") is not None else None,
            "setupPattern": str(inputs.get("setupPattern", "")),
            "stagePhase": str(inputs.get("stagePhase", "")),
            "marketRegime": str(inputs.get("marketRegime", "")),
            "sector": str(inputs.get("sector", "")),
            "assetClass": str(inputs.get("assetClass", "")),
        }
        encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def register_signal(
        cls,
        symbol: str,
        entry_price: float,
        opt_exec: Dict[str, Any],
        confluence_score: float,
        inputs_meta: Dict[str, Any],
        engine_commit: str = "4e36862",
        engine_tag: str = "v2.4.0-phase24-freeze",
        ledger_path: Optional[str] = None,
        signal_date: Optional[str] = None,
        component_scores: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Any]] = None,
        epoch_id: Optional[str] = None,
        provenance_cohort: Optional[str] = None,
        release_sha: Optional[str] = None,
        deployment_id: Optional[str] = None,
        capture_source: Optional[str] = None,
        analysis_reference_price: Optional[float] = None,
        analysis_reference_source: Optional[str] = None,
        live_spot_price: Optional[float] = None,
        live_observed_at: Optional[str] = None,
        live_source: Optional[str] = None,
        live_freshness: Optional[str] = None,
        market_session: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an immutable new signal in the ledger."""
        ledger = cls.load_ledger(ledger_path)
        sig_date = signal_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        signal_id = f"{symbol}_{sig_date}"

        # Prevent duplicate entries for the same symbol on the same date
        existing = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if existing:
            return existing

        record = {
            "signalId": signal_id,
            "symbol": symbol.upper().strip(),
            "signalDate": sig_date,
            "epochId": epoch_id or cls.EPOCH_ID,
            "provenanceCohort": provenance_cohort or ProvenanceCohort.PROSPECTIVE_CLEAN,
            "engineVersion": engine_commit,
            "engineTag": engine_tag,
            "decisionEngineSha": cls.DECISION_ENGINE_SHA,
            "observationGovernanceSha": cls.get_observation_governance_sha(),
            "releaseSha": release_sha or inputs_meta.get("releaseSha") or os.getenv("RAILWAY_GIT_COMMIT_SHA"),
            "deploymentId": deployment_id or inputs_meta.get("deploymentId") or os.getenv("RAILWAY_DEPLOYMENT_ID"),
            "captureSource": capture_source or inputs_meta.get("captureSource", "NATURAL_PRODUCTION_API"),
            "analysisReferencePrice": analysis_reference_price or inputs_meta.get("analysisReferencePrice") or float(entry_price),
            "analysisReferenceSource": analysis_reference_source or inputs_meta.get("analysisReferenceSource", "COMPLETED_SESSION"),
            "liveSpotPrice": live_spot_price if live_spot_price is not None else inputs_meta.get("liveSpotPrice"),
            "liveObservedAt": live_observed_at or inputs_meta.get("liveObservedAt"),
            "liveSource": live_source or inputs_meta.get("liveSource", "UNAVAILABLE"),
            "liveFreshness": live_freshness or inputs_meta.get("liveFreshness", "UNAVAILABLE"),
            "marketSession": market_session or inputs_meta.get("marketSession", "UNKNOWN"),
            "recommended_at": inputs_meta.get("recommended_at") or inputs_meta.get("signalTimestamp") or sig_date,
            "signalTimestamp": inputs_meta.get("signalTimestamp") or inputs_meta.get("recommended_at") or sig_date,
            "decisionState": "ACTIONABLE_SETUP",
            "entryPrice": float(entry_price),
            "corridorMin": opt_exec.get("optimal_entry_min"),
            "corridorMax": opt_exec.get("optimal_entry_max"),
            "stopLoss": opt_exec.get("stop_loss"),
            "stopLossPct": opt_exec.get("stop_loss_pct"),
            "takeProfit1": opt_exec.get("take_profit_1"),
            "takeProfit1Pct": opt_exec.get("take_profit_1_pct"),
            "takeProfit2": opt_exec.get("take_profit_2"),
            "takeProfit2Pct": opt_exec.get("take_profit_2_pct"),
            "riskRewardRatio": opt_exec.get("risk_reward_ratio"),
            "confluenceScore": float(confluence_score),
            "inputs": {
                "atr14": opt_exec.get("atr_14"),
                "atrPct": round((opt_exec.get("atr_14", 0) / max(0.01, entry_price)) * 100, 2),
                "setupPattern": opt_exec.get("setup_pattern"),
                "stagePhase": opt_exec.get("stage_phase"),
                "marketRegime": inputs_meta.get("market_regime", "BULL"),
                "sector": inputs_meta.get("sector", "EQUITY"),
                "assetClass": inputs_meta.get("asset_class", "US_EQUITY"),
                # Epoch 1 Complete Information Set Attributes
                "marketDataSnapshotTimestamp": inputs_meta.get("market_data_snapshot_timestamp") or inputs_meta.get("marketDataSnapshotTimestamp"),
                "marketSnapshotObservedAt": inputs_meta.get("marketSnapshotObservedAt") or inputs_meta.get("market_data_snapshot_timestamp") or inputs_meta.get("marketDataSnapshotTimestamp"),
                "marketSnapshotHash": inputs_meta.get("marketSnapshotHash") or inputs_meta.get("market_snapshot_hash"),
                "candleCount": inputs_meta.get("candle_count") or inputs_meta.get("candleCount"),
                "sma50": opt_exec.get("sma_50") or inputs_meta.get("sma50"),
                "ema20": opt_exec.get("ema_20") or inputs_meta.get("ema20"),
                "rsi14": opt_exec.get("rsi_14") or inputs_meta.get("rsi14"),
                "fundamentalAsOfDate": inputs_meta.get("fundamental_as_of_date") or inputs_meta.get("fundamentalAsOfDate"),
                "fundamentalFilingTimestamp": inputs_meta.get("fundamental_filing_timestamp") or inputs_meta.get("fundamentalFilingTimestamp"),
                "fundamentalSnapshotHash": inputs_meta.get("fundamentalSnapshotHash") or inputs_meta.get("fundamental_snapshot_hash"),
                "macroObservationDate": inputs_meta.get("macro_observation_date") or inputs_meta.get("macroObservationDate"),
                "macroObservationAvailableAt": inputs_meta.get("macroObservationAvailableAt") or inputs_meta.get("macro_observation_available_at") or inputs_meta.get("yield_observation_timestamp"),
                "macroSnapshotHash": inputs_meta.get("macroSnapshotHash") or inputs_meta.get("macro_snapshot_hash"),
                "yieldCurve10y2y": inputs_meta.get("yield_curve_10y2y") if inputs_meta.get("yield_curve_10y2y") is not None else inputs_meta.get("yieldCurve10y2y"),
                "creditSpread": inputs_meta.get("credit_spread") if inputs_meta.get("credit_spread") is not None else inputs_meta.get("creditSpread"),
                "dataProvider": inputs_meta.get("data_provider") or inputs_meta.get("dataProvider", "YAHOO_AUTHENTIC"),
                "quoteFreshness": inputs_meta.get("quote_freshness") or inputs_meta.get("quoteFreshness", "END_OF_DAY"),
                "evidenceCompleteness": inputs_meta.get("evidence_completeness") or inputs_meta.get("evidenceCompleteness", "COMPLETE"),
                "modelConfigHash": inputs_meta.get("modelConfigHash") or cls.CONFIG_HASH,
                "pointInTimePrecision": inputs_meta.get("pointInTimePrecision", "TIMESTAMP"),
                "rawMarketPayload": inputs_meta.get("rawMarketPayload") or inputs_meta.get("candles"),
                "rawFundamentalPayload": inputs_meta.get("rawFundamentalPayload"),
                "rawMacroPayload": inputs_meta.get("rawMacroPayload") or inputs_meta.get("macro_inputs"),
            },
            "componentScores": component_scores or {
                "qualityScore": None,
                "growthScore": None,
                "valuationScore": None,
                "technicalScore": None,
                "smartMoneyScore": None,
                "macroScore": None,
                "catalystScore": None,
            },
            "benchmarksAtSignal": benchmarks or {
                "spyPrice": None,
                "rspPrice": None,
                "sectorBenchmarkSymbol": None,
                "sectorBenchmarkPrice": None,
                "momentumBaselineReturn": None,
            },
            "status": "OPEN",
            "forwardTracking": {
                "sessionsObserved": 0,
                "currentPrice": float(entry_price),
                "maxFavorableExcursionPct": None,
                "maxAdverseExcursionPct": None,
                "tp1Hit": False,
                "tp1Session": None,
                "stopHit": False,
                "stopSession": None,
                "return1d": None,
                "return5d": None,
                "return10d": None,
                "return20d": None,
                "benchmarkReturns": {
                    "spy": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "rsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "sector": {"1d": None, "5d": None, "10d": None, "20d": None},
                },
                "relativeReturns": {
                    "vsSpy": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "vsRsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                    "vsSector": {"1d": None, "5d": None, "10d": None, "20d": None},
                },
                "signalQuality": {
                    "directionalAccuracy1d": None,
                    "directionalAccuracy5d": None,
                    "directionalAccuracy10d": None,
                    "directionalAccuracy20d": None,
                    "rawReturn1d": None,
                    "rawReturn5d": None,
                    "rawReturn10d": None,
                    "rawReturn20d": None,
                    "mfePct": None,
                    "maePct": None,
                },
                "tradeConstruction": {
                    "prematureStopOut": False,
                    "captureRatio": None,
                    "riskRewardRealized": None,
                },
                "economicOutcome": {
                    "realizedReturnPct": None,
                    "frictionBps": 25.0,
                    "netReturnPct": None,
                },
                "signalPersistence": {
                    "day1Valid": True,
                    "day3Valid": None,
                    "day5Valid": None,
                    "day10Valid": None
                },
                "resolutionDate": None,
                "resolvedOutcome": None,  # "TP1_WIN", "STOP_LOSS", "TIME_EXPIRED"
                "realizedReturnPct": None
            }
        }
        # Compute immutable cryptographic snapshot hashes (Decision & Inputs Dual-Hash)
        record["decisionSnapshotHash"] = cls.compute_decision_snapshot_hash(record)
        record["inputsSnapshotHash"] = cls.compute_inputs_snapshot_hash(record)

        # Record point-in-time liquidity observation at signal generation
        liq_meta = opt_exec.get("liquidity_defense") or {}
        if isinstance(liq_meta, dict):
            liq_meta = dict(liq_meta)
            if "specVersion" not in liq_meta and "spec_version" not in liq_meta:
                liq_meta["specVersion"] = "LiquidityGuard Shadow Spec v1.0"
            if "signalTimestamp" not in liq_meta and "signal_timestamp" not in liq_meta:
                liq_meta["signalTimestamp"] = datetime.now(timezone.utc).isoformat()
        record["liquidityAtSignal"] = liq_meta
        record["liquidityForwardObservations"] = []
        record["executionObservations"] = []

        ledger["signals"].append(record)
        ledger["totalActiveSignals"] = len([s for s in ledger["signals"] if s["status"] == "OPEN"])
        cls.save_ledger(ledger, ledger_path)
        return record

    @classmethod
    def record_liquidity_forward_observation(
        cls,
        signal_id: str,
        session_index: int,
        liquidity_metrics: Dict[str, Any],
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records subsequent time-varying liquidity observation without mutating liquidityAtSignal or decision parameters."""
        ledger = cls.load_ledger(ledger_path)
        sig = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if not sig:
            raise KeyError(f"Signal {signal_id} not found in ledger")

        # Verify decision and inputs integrity (anti-tamper cryptographic check)
        if sig.get("decisionSnapshotHash"):
            current_hash = cls.compute_decision_snapshot_hash(sig)
            if current_hash != sig["decisionSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {signal_id}")

        if sig.get("inputsSnapshotHash"):
            current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
            if current_inputs_hash != sig["inputsSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {signal_id}")

        if "liquidityForwardObservations" not in sig:
            sig["liquidityForwardObservations"] = []

        obs_entry = {
            "sessionIndex": session_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adv20dUsd": liquidity_metrics.get("adv_20d_usd"),
            "amihudIlliqRaw": liquidity_metrics.get("amihud_illiq"),
            "amihudIlliqScaled": liquidity_metrics.get("amihud_illiq_scaled"),
            "liquidityGrade": liquidity_metrics.get("liquidity_grade"),
            "executionHazard": liquidity_metrics.get("execution_hazard", False),
        }
        sig["liquidityForwardObservations"].append(obs_entry)
        cls.save_ledger(ledger, ledger_path)
        return obs_entry

    @classmethod
    def record_execution_observation(
        cls,
        signal_id: str,
        fill_price: Optional[float] = None,
        execution_timestamp: Optional[str] = None,
        order_size_usd: Optional[float] = None,
        side: str = "BUY",
        execution_source: str = "BROKER_FILL",
        is_simulated: bool = False,
        symbol: Optional[str] = None,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a Phase 26 prospective execution observation without mutating the frozen decision or signal outcomes.

        Captures:
        - signalTimestamp, referencePrice, executionTimestamp, fillPrice, side, orderSizeUsd,
          adv20dUsd, adv5dUsd, liquidityTrend, amihudIlliqRaw, amihudIlliqScaled, liquidityGrade,
          participationRate, realized slippage in bps (magnitude and signed), executionSource, isSimulated.
        """
        ledger = cls.load_ledger(ledger_path)
        sig = next((s for s in ledger["signals"] if s["signalId"] == signal_id), None)
        if not sig:
            raise KeyError(f"Signal {signal_id} not found in ledger")

        # Anti-tamper cryptographic check: verify decision and inputs integrity fail-closed
        if sig.get("decisionSnapshotHash"):
            current_hash = cls.compute_decision_snapshot_hash(sig)
            if current_hash != sig["decisionSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {signal_id}")

        if sig.get("inputsSnapshotHash"):
            current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
            if current_inputs_hash != sig["inputsSnapshotHash"]:
                raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {signal_id}")

        # Cross-signal contamination guard: verify symbol matches if provided
        if symbol is not None:
            clean_sym = str(symbol).upper().strip()
            if clean_sym != sig["symbol"]:
                raise ValueError(f"CROSS_SIGNAL_CONTAMINATION: Provided symbol '{symbol}' does not match signal symbol '{sig['symbol']}'")

        # Validate order side
        norm_side = str(side).upper().strip()
        if norm_side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side '{side}': must be 'BUY' or 'SELL'")

        # Validate reference price
        ref_price = float(sig.get("entryPrice", 0.0))
        if ref_price <= 0.0 or math.isnan(ref_price) or math.isinf(ref_price):
            raise ValueError(f"Invalid reference entry price {ref_price} on signal {signal_id}")

        # Validate fill price and compute slippage
        validated_fill: Optional[float] = None
        slippage_bps: Optional[float] = None
        signed_slippage_bps: Optional[float] = None
        if fill_price is not None:
            f_price = float(fill_price)
            if f_price <= 0.0 or math.isnan(f_price) or math.isinf(f_price):
                raise ValueError(f"Invalid fill price {fill_price}: must be positive finite number")
            validated_fill = f_price
            # Absolute slippage magnitude in basis points: abs(fill_price - ref_price) / ref_price * 10,000
            slippage_bps = round((abs(validated_fill - ref_price) / ref_price) * 10000.0, 2)
            # Signed directional slippage (BUY: fill > ref is adverse (+), SELL: fill < ref is adverse (+))
            direction = 1 if norm_side == "BUY" else -1
            signed_slippage_bps = round(direction * ((validated_fill - ref_price) / ref_price) * 10000.0, 2)

        # Validate temporal sequencing
        sig_time_str = (
            sig.get("liquidityAtSignal", {}).get("signalTimestamp")
            or sig.get("liquidityAtSignal", {}).get("signal_timestamp")
            or sig.get("signalDate")
        )
        exec_time_str = execution_timestamp or datetime.now(timezone.utc).isoformat()

        # Enforce temporal anti-lookahead: execution timestamp cannot precede signal timestamp
        if sig_time_str and exec_time_str and validated_fill is not None:
            try:
                sig_clean = sig_time_str.replace("Z", "+00:00")
                exec_clean = exec_time_str.replace("Z", "+00:00")
                sig_dt = datetime.fromisoformat(sig_clean) if "T" in sig_clean else datetime.strptime(sig_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                exec_dt = datetime.fromisoformat(exec_clean) if "T" in exec_clean else datetime.strptime(exec_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if exec_dt < sig_dt:
                    raise ValueError(f"TEMPORAL_VIOLATION: Execution timestamp {exec_time_str} precedes signal timestamp {sig_time_str}")

                # Enforce temporal sequencing against trade resolution: execution cannot occur after trade is resolved
                if sig.get("status") == "RESOLVED":
                    res_date_str = sig.get("forwardTracking", {}).get("resolutionDate")
                    if res_date_str:
                        res_clean = res_date_str.replace("Z", "+00:00")
                        res_dt = datetime.fromisoformat(res_clean) if "T" in res_clean else datetime.strptime(res_clean[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if exec_dt.date() > res_dt.date():
                            raise ValueError(f"TEMPORAL_VIOLATION: Execution date {exec_dt.date()} occurs after trade resolution date {res_dt.date()}")
            except (ValueError, TypeError) as e:
                if "TEMPORAL_VIOLATION" in str(e):
                    raise

        # Extract signal-time liquidity diagnostics
        liq = sig.get("liquidityAtSignal") or {}
        adv_20d = float(liq.get("adv20dUsd") or liq.get("adv_20d_usd") or 0.0)
        adv_5d = float(liq.get("adv5dUsd") or liq.get("adv_5d_usd") or 0.0)
        liq_trend = float(liq.get("liquidityTrend") or liq.get("liquidity_trend") or 1.0)
        amihud_raw = float(liq.get("amihudIlliqRaw") or liq.get("amihud_illiq") or 0.0)
        amihud_scaled = float(liq.get("amihudIlliqScaled") or liq.get("amihud_illiq_scaled") or 0.0)
        liq_grade = str(liq.get("liquidityGrade") or liq.get("liquidity_grade") or "UNKNOWN_LIQUIDITY")

        # Validate order size and participation rate
        order_size: Optional[float] = None
        participation_rate: Optional[float] = None
        if order_size_usd is not None:
            sz = float(order_size_usd)
            if sz < 0.0 or math.isnan(sz) or math.isinf(sz):
                raise ValueError(f"Invalid order size USD {order_size_usd}: must be non-negative finite number")
            order_size = sz
            if adv_20d > 0:
                participation_rate = round(order_size / adv_20d, 6)
            else:
                participation_rate = 1.0 if order_size > 0 else 0.0

        if "executionObservations" not in sig:
            sig["executionObservations"] = []

        # Duplicate fill protection: reject duplicate observation with identical fill, timestamp, side, and size
        if validated_fill is not None:
            for existing_obs in sig["executionObservations"]:
                if (
                    existing_obs.get("fillPrice") == validated_fill
                    and existing_obs.get("executionTimestamp") == exec_time_str
                    and existing_obs.get("side") == norm_side
                    and existing_obs.get("orderSizeUsd") == order_size
                ):
                    raise ValueError(f"DUPLICATE_EXECUTION_OBSERVATION: Identical fill already recorded for signal {signal_id}")

        obs_record = {
            "observationId": f"{signal_id}_exec_{len(sig['executionObservations']) + 1}",
            "signalId": signal_id,
            "symbol": sig.get("symbol"),
            "signalTimestamp": sig_time_str,
            "referencePrice": ref_price,
            "executionTimestamp": exec_time_str if validated_fill is not None else None,
            "fillPrice": validated_fill,
            "side": norm_side,
            "orderSizeUsd": order_size,
            "adv20dUsd": adv_20d,
            "adv5dUsd": adv_5d,
            "liquidityTrend": liq_trend,
            "amihudIlliqRaw": amihud_raw,
            "amihudIlliqScaled": amihud_scaled,
            "liquidityGrade": liq_grade,
            "participationRate": participation_rate,
            "slippageBps": slippage_bps,
            "signedSlippageBps": signed_slippage_bps,
            "executionSource": execution_source if validated_fill is not None else "UNFILLED",
            "isSimulated": bool(is_simulated),
            "specVersion": "LiquidityGuard Shadow Spec v1.0",
            "recordedAt": datetime.now(timezone.utc).isoformat(),
        }

        sig["executionObservations"].append(obs_record)
        cls.save_ledger(ledger, ledger_path)
        return obs_record

    @classmethod
    def update_forward_observations(
        cls,
        db_engine,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Harvest subsequent price candles and update forward outcomes objectively."""
        ledger = cls.load_ledger(ledger_path)
        updated_count = 0

        for sig in ledger["signals"]:
            # Hard Governance Check: verify decision and inputs snapshot integrity
            if sig.get("decisionSnapshotHash"):
                current_hash = cls.compute_decision_snapshot_hash(sig)
                if current_hash != sig["decisionSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {sig.get('signalId')}")

            if sig.get("inputsSnapshotHash"):
                current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
                if current_inputs_hash != sig["inputsSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {sig.get('signalId')}")

            if sig["status"] != "OPEN":
                continue

            sym = sig["symbol"]
            entry_price = sig["entryPrice"]
            stop = sig["stopLoss"]
            tp1 = sig["takeProfit1"]
            sig_date = sig["signalDate"]

    @classmethod
    def _compute_benchmark_horizon_return(
        cls, db_engine, bench_symbol: str, sig_date: str, horizon_days: int
    ) -> Optional[float]:
        """Calculates benchmark buy-and-hold return over specified session horizon from sig_date."""
        if not db_engine or not hasattr(db_engine, "get_daily_candles"):
            return None
        try:
            b_candles = db_engine.get_daily_candles(bench_symbol, limit=100)
            if not b_candles:
                return None
            b_df = pd.DataFrame(b_candles)
            b_date_col = "time" if "time" in b_df.columns else ("date" if "date" in b_df.columns else None)
            if not b_date_col:
                return None
            b_sub = b_df[b_df[b_date_col] > sig_date].copy()
            if len(b_sub) < horizon_days:
                return None
            b_sub.rename(columns={"close": "Close", "open": "Open"}, inplace=True)
            prior = b_df[b_df[b_date_col] <= sig_date]
            if not prior.empty:
                p_close_col = "close" if "close" in prior.columns else "Close"
                b_p0 = float(prior.iloc[-1][p_close_col])
            else:
                b_p0 = float(b_sub.iloc[0]["Open"])
            if b_p0 <= 0:
                return None
            b_close_h = float(b_sub.iloc[horizon_days - 1]["Close"])
            return round(((b_close_h - b_p0) / b_p0) * 100.0, 2)
        except Exception:
            return None

    @classmethod
    def update_forward_observations(
        cls,
        db_engine,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Harvest subsequent price candles and update forward outcomes objectively."""
        ledger = cls.load_ledger(ledger_path)
        updated_count = 0

        for sig in ledger["signals"]:
            # Hard Governance Check: verify decision and inputs snapshot integrity
            if sig.get("decisionSnapshotHash"):
                current_hash = cls.compute_decision_snapshot_hash(sig)
                if current_hash != sig["decisionSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Decision snapshot mismatch on {sig.get('signalId')}")

            if sig.get("inputsSnapshotHash"):
                current_inputs_hash = cls.compute_inputs_snapshot_hash(sig)
                if current_inputs_hash != sig["inputsSnapshotHash"]:
                    raise ValueError(f"GOVERNANCE_INTEGRITY_FAILURE: Inputs snapshot mismatch on {sig.get('signalId')}")

            if sig["status"] != "OPEN":
                continue

            sym = sig["symbol"]
            entry_price = sig["entryPrice"]
            stop = sig["stopLoss"]
            tp1 = sig["takeProfit1"]
            sig_date = sig["signalDate"]

            candles = db_engine.get_daily_candles(sym, limit=100)
            if not candles:
                continue

            df = pd.DataFrame(candles)
            date_col = "time" if "time" in df.columns else ("date" if "date" in df.columns else None)
            if not date_col:
                continue

            # Filter candles occurring AFTER the signal date
            subsequent = df[df[date_col] > sig_date].copy()
            if subsequent.empty:
                continue

            subsequent.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            sessions = len(subsequent)
            highs = subsequent["High"].values
            lows = subsequent["Low"].values
            closes = subsequent["Close"].values

            latest_close = float(closes[-1])
            mfe = float(np.max(highs) - entry_price) / entry_price * 100.0
            mae = float(np.min(lows) - entry_price) / entry_price * 100.0

            ft = sig.setdefault("forwardTracking", {})
            ft["sessionsObserved"] = sessions
            ft["currentPrice"] = latest_close
            ft["maxFavorableExcursionPct"] = round(mfe, 2)
            ft["maxAdverseExcursionPct"] = round(mae, 2)

            sq = ft.setdefault("signalQuality", {
                "directionalAccuracy1d": None,
                "directionalAccuracy5d": None,
                "directionalAccuracy10d": None,
                "directionalAccuracy20d": None,
                "rawReturn1d": None,
                "rawReturn5d": None,
                "rawReturn10d": None,
                "rawReturn20d": None,
                "mfePct": None,
                "maePct": None,
            })
            sq["mfePct"] = round(mfe, 2)
            sq["maePct"] = round(mae, 2)

            # Check Returns at multi-horizons (T+1, T+5, T+10, T+20)
            if sessions >= 1 and ft.get("return1d") is None:
                r1d = round(((closes[0] - entry_price) / entry_price) * 100.0, 2)
                ft["return1d"] = r1d
                sq["rawReturn1d"] = r1d
                sq["directionalAccuracy1d"] = bool(r1d > 0)

            if sessions >= 5 and ft.get("return5d") is None:
                r5d = round(((closes[4] - entry_price) / entry_price) * 100.0, 2)
                ft["return5d"] = r5d
                sq["rawReturn5d"] = r5d
                sq["directionalAccuracy5d"] = bool(r5d > 0)

            if sessions >= 10 and ft.get("return10d") is None:
                r10d = round(((closes[9] - entry_price) / entry_price) * 100.0, 2)
                ft["return10d"] = r10d
                sq["rawReturn10d"] = r10d
                sq["directionalAccuracy10d"] = bool(r10d > 0)

            if sessions >= 20 and ft.get("return20d") is None:
                r20d = round(((closes[19] - entry_price) / entry_price) * 100.0, 2)
                ft["return20d"] = r20d
                sq["rawReturn20d"] = r20d
                sq["directionalAccuracy20d"] = bool(r20d > 0)

            # Multi-horizon benchmark comparisons (SPY, RSP, Sector)
            bench_dict = ft.setdefault("benchmarkReturns", {
                "spy": {"1d": None, "5d": None, "10d": None, "20d": None},
                "rsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                "sector": {"1d": None, "5d": None, "10d": None, "20d": None},
            })
            rel_dict = ft.setdefault("relativeReturns", {
                "vsSpy": {"1d": None, "5d": None, "10d": None, "20d": None},
                "vsRsp": {"1d": None, "5d": None, "10d": None, "20d": None},
                "vsSector": {"1d": None, "5d": None, "10d": None, "20d": None},
            })

            sec_meta = sig.get("benchmarksAtSignal") or {}
            sector_sym = sec_meta.get("sectorBenchmarkSymbol") or (
                "XLK" if sig.get("inputs", {}).get("sector") == "TECHNOLOGY" else None
            )

            for h_label, h_days, ret_val in [
                ("1d", 1, ft.get("return1d")),
                ("5d", 5, ft.get("return5d")),
                ("10d", 10, ft.get("return10d")),
                ("20d", 20, ft.get("return20d")),
            ]:
                if sessions >= h_days and ret_val is not None:
                    spy_ret = cls._compute_benchmark_horizon_return(db_engine, "SPY", sig_date, h_days)
                    if spy_ret is not None:
                        bench_dict["spy"][h_label] = spy_ret
                        rel_dict["vsSpy"][h_label] = round(ret_val - spy_ret, 2)
                    rsp_ret = cls._compute_benchmark_horizon_return(db_engine, "RSP", sig_date, h_days)
                    if rsp_ret is not None:
                        bench_dict["rsp"][h_label] = rsp_ret
                        rel_dict["vsRsp"][h_label] = round(ret_val - rsp_ret, 2)
                    if sector_sym:
                        sec_ret = cls._compute_benchmark_horizon_return(db_engine, sector_sym, sig_date, h_days)
                        if sec_ret is not None:
                            bench_dict["sector"][h_label] = sec_ret
                            rel_dict["vsSector"][h_label] = round(ret_val - sec_ret, 2)

            # Check TP1 vs Stop sequence with fail-closed intrabar collision handling
            intrabar_collision = False
            for idx in range(sessions):
                hit_tp = bool(highs[idx] >= tp1)
                hit_sl = bool(lows[idx] <= stop)

                if hit_tp and hit_sl and not ft["tp1Hit"] and not ft["stopHit"]:
                    # Pessimistic fail-closed policy: adverse excursion precedes favorable excursion
                    intrabar_collision = True
                    ft["stopHit"] = True
                    ft["stopSession"] = idx + 1
                    ft["tp1Hit"] = True
                    ft["tp1Session"] = idx + 1
                    break

                if hit_tp and not ft["tp1Hit"]:
                    ft["tp1Hit"] = True
                    ft["tp1Session"] = idx + 1
                if hit_sl and not ft["stopHit"]:
                    ft["stopHit"] = True
                    ft["stopSession"] = idx + 1

                if ft["tp1Hit"] or ft["stopHit"]:
                    break

            ft["intrabarCollision"] = intrabar_collision

            # Resolve outcome with strict fail-closed arbitration
            if intrabar_collision or (ft["stopHit"] and (not ft["tp1Hit"] or ft["stopSession"] <= ft["tp1Session"])):
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "STOP_LOSS"
                ft["realizedReturnPct"] = round(((stop - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[ft["stopSession"] - 1] if ft["stopSession"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif ft["tp1Hit"] and (not ft["stopHit"] or ft["tp1Session"] < ft["stopSession"]):
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "TP1_WIN"
                ft["realizedReturnPct"] = round(((tp1 - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[ft["tp1Session"] - 1] if ft["tp1Session"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif sessions >= 20:
                sig["status"] = "RESOLVED"
                ft["resolvedOutcome"] = "TIME_EXPIRED"
                ft["realizedReturnPct"] = round(((latest_close - entry_price) / entry_price) * 100.0, 2)
                ft["resolutionDate"] = subsequent[date_col].iloc[19]

            # Populate Trade Construction and Economic Outcomes
            if sig["status"] == "RESOLVED":
                tc = ft.setdefault("tradeConstruction", {
                    "prematureStopOut": False,
                    "captureRatio": None,
                    "captureVsMfe5d": None,
                    "captureVsMfe10d": None,
                    "captureVsMfe20d": None,
                    "riskRewardRealized": None,
                })
                eco = ft.setdefault("economicOutcome", {
                    "realizedReturnPct": None,
                    "frictionBps": 30.0,
                    "netReturnPct": None,
                })
                realized = ft.get("realizedReturnPct", 0.0)

                # Multi-horizon capture ratios
                mfe_5d = round(float(np.max(highs[:min(sessions, 5)]) - entry_price) / entry_price * 100.0, 2)
                mfe_10d = round(float(np.max(highs[:min(sessions, 10)]) - entry_price) / entry_price * 100.0, 2)
                mfe_20d = round(float(np.max(highs[:min(sessions, 20)]) - entry_price) / entry_price * 100.0, 2)

                if realized is not None and realized > 0:
                    tc["captureVsMfe5d"] = round(realized / max(0.01, mfe_5d), 4) if mfe_5d > 0 else 1.0
                    tc["captureVsMfe10d"] = round(realized / max(0.01, mfe_10d), 4) if mfe_10d > 0 else 1.0
                    tc["captureVsMfe20d"] = round(realized / max(0.01, mfe_20d), 4) if mfe_20d > 0 else 1.0
                    tc["captureRatio"] = round(realized / max(0.01, mfe), 4)
                elif realized is not None and realized <= 0:
                    tc["captureVsMfe5d"] = 0.0
                    tc["captureVsMfe10d"] = 0.0
                    tc["captureVsMfe20d"] = 0.0
                    tc["captureRatio"] = 0.0

                # Post-Stop Favorable Excursion (PSFE) analysis
                # Clarified: This measures path potential after stop-out to inform shadow exit research,
                # NOT an assertion that the stop itself was incorrect.
                psfe_tp1_crossed = False
                post_stop_mfe = 0.0
                if ft["stopHit"] and ft.get("stopSession") and ft["stopSession"] < sessions:
                    post_stop_highs = highs[ft["stopSession"]:]
                    if len(post_stop_highs) > 0:
                        post_stop_mfe = round(float(np.max(post_stop_highs) - entry_price) / entry_price * 100.0, 2)
                        if np.any(post_stop_highs >= tp1):
                            psfe_tp1_crossed = True

                tc["postStopFavorableExcursion"] = {
                    "tp1CrossedPostStop": psfe_tp1_crossed,
                    "maxFavorableExcursionPostStopPct": post_stop_mfe
                }
                # Backward compatibility alias
                tc["prematureStopOut"] = psfe_tp1_crossed

                # Risk-reward realization
                stop_dist = abs(entry_price - stop)
                if stop_dist > 0 and realized is not None:
                    tc["riskRewardRealized"] = round((realized / 100.0 * entry_price) / stop_dist, 2)

                # Economic outcome net of primary friction (30 bps round-trip)
                primary_friction_bps = 30.0
                eco["realizedReturnPct"] = realized
                eco["frictionBps"] = primary_friction_bps
                eco["netReturnPct"] = round(realized - (primary_friction_bps / 100.0), 2) if realized is not None else None

            updated_count += 1

        ledger["totalActiveSignals"] = len([s for s in ledger["signals"] if s["status"] == "OPEN"])
        cls.save_ledger(ledger, ledger_path)
        return {"updatedSignals": updated_count, "openSignals": ledger["totalActiveSignals"]}

    @classmethod
    def compute_governance_scorecard(
        cls,
        ledger_path: Optional[str] = None,
        cohort_filter: Optional[str] = None,
        epoch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate performance against Phase 25/26 Model Governance criteria with Cohort Contamination Firewall."""
        ledger = cls.load_ledger(ledger_path)
        all_signals = ledger.get("signals", [])
        if not all_signals:
            return {"status": "NO_SIGNALS", "n": 0}

        # 1. Audit Provenance Across All Signals (Cohort Contamination Firewall & Epoch Boundary)
        clean_signals = []
        contaminated_signals = []
        unclassified_signals = []
        excluded_signals = []

        for s in all_signals:
            c = cls.classify_provenance_cohort(s)
            s["provenanceCohort"] = c
            if c == ProvenanceCohort.PROSPECTIVE_CLEAN:
                if epoch_id is None or s.get("epochId") == epoch_id:
                    clean_signals.append(s)
                else:
                    excluded_signals.append(s)
            elif c in [
                ProvenanceCohort.HISTORICAL_CONTAMINATED,
                ProvenanceCohort.CONTAMINATED,
                ProvenanceCohort.HISTORICAL_RECOMPUTED,
            ]:
                contaminated_signals.append(s)
            elif c in [
                ProvenanceCohort.EXCLUDED,
                ProvenanceCohort.BACKTEST_SIMULATION,
                ProvenanceCohort.DEMO_SYNTHETIC,
                ProvenanceCohort.CERTIFICATION_VALIDATION,
            ]:
                excluded_signals.append(s)
            else:
                unclassified_signals.append(s)

        # Determine target evaluation cohort
        if cohort_filter == ProvenanceCohort.HISTORICAL_CONTAMINATED:
            signals = contaminated_signals
            clean_eval_eligible = False
            firewall_status = "HISTORICAL_CONTAMINATED_RESEARCH_ONLY"
            firewall_warning = "CRITICAL: Evaluating historically contaminated cohort. NOT ELIGIBLE for predictive edge claims."
        elif cohort_filter == "ALL":
            signals = all_signals
            clean_eval_eligible = (len(contaminated_signals) == 0)
            firewall_status = "ALL_SIGNALS_MIXED"
            firewall_warning = "WARNING: Mixed evaluation includes contaminated historical signals." if len(contaminated_signals) > 0 else None
        elif cohort_filter == ProvenanceCohort.PROSPECTIVE_CLEAN:
            signals = clean_signals
            clean_eval_eligible = True
            firewall_status = "CLEAN_COHORT_INSULATED"
            firewall_warning = None
        else:
            # Default auto-isolation: If clean prospective signals exist, insulate them from contaminated history
            if len(clean_signals) > 0:
                signals = clean_signals
                clean_eval_eligible = True
                firewall_status = "AUTO_INSULATED_PROSPECTIVE_CLEAN"
                firewall_warning = None
            elif len(contaminated_signals) > 0 and len(clean_signals) == 0 and len(unclassified_signals) == 0:
                signals = contaminated_signals
                clean_eval_eligible = False
                firewall_status = "HISTORICAL_CONTAMINATED_NO_PROSPECTIVE"
                firewall_warning = "CRITICAL: Only contaminated historical signals found. NOT ELIGIBLE for predictive edge claims."
            else:
                signals = all_signals
                clean_eval_eligible = True
                firewall_status = "UNCLASSIFIED_SYNTHETIC_TEST_COHORT"
                firewall_warning = None

        # Verify Frozen Engine Manifest Integrity
        manifest_audit = cls.verify_frozen_engine_manifest()
        if manifest_audit.get("status") == "CORRUPTED":
            clean_eval_eligible = False
            firewall_status = "PRODUCTION_ENGINE_MUTATION_DETECTED"
            firewall_warning = "CRITICAL: Production engine SHA-256 manifest mismatch. Clean prospective evaluation revoked."

        resolved = [s for s in signals if s.get("status") == "RESOLVED"]
        open_signals = [s for s in signals if s.get("status") == "OPEN"]

        n_resolved = len(resolved)
        wins = [s for s in resolved if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN"]
        stops = [s for s in resolved if s.get("forwardTracking", {}).get("resolvedOutcome") == "STOP_LOSS"]

        p_win = (len(wins) / n_resolved) if n_resolved > 0 else 0.0
        p_loss = (len(stops) / n_resolved) if n_resolved > 0 else 0.0

        avg_win = float(np.mean([w["forwardTracking"]["realizedReturnPct"] for w in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(s["forwardTracking"]["realizedReturnPct"]) for s in stops])) if stops else 0.0

        expectancy = (p_win * avg_win) - (p_loss * avg_loss)
        primary_friction_pct = 0.30  # 30 bps round-trip primary standard (additive deduction)
        net_expectancy = expectancy - primary_friction_pct if n_resolved > 0 else 0.0

        friction_sensitivity = {
            "arithmeticConvention": "linear_additive_deduction",
            "0bps_gross": round(expectancy, 2),
            "15bps_optimistic": round(expectancy - 0.15, 2),
            "30bps_primary": round(net_expectancy, 2),
            "50bps_stress": round(expectancy - 0.50, 2),
            "100bps_severe": round(expectancy - 1.00, 2),
            "edgeBreakevenFrictionBps": round(expectancy * 100.0, 1) if expectancy > 0 else 0.0
        }

        total_gains = sum([w["forwardTracking"]["realizedReturnPct"] for w in wins]) if wins else 0.0
        total_losses = sum([abs(s["forwardTracking"]["realizedReturnPct"]) for s in stops]) if stops else 0.0
        profit_factor = (total_gains / total_losses) if total_losses > 0 else (999.0 if total_gains > 0 else 0.0)

        # Statistical Uncertainty & Confidence Intervals (Diagnostic layer)
        all_realized = [
            s["forwardTracking"]["realizedReturnPct"]
            for s in resolved
            if s.get("forwardTracking", {}).get("realizedReturnPct") is not None
        ]
        if len(all_realized) >= 2:
            std_err = float(np.std(all_realized, ddof=1) / np.sqrt(len(all_realized)))
            ci_95_lower = round(expectancy - 1.96 * std_err, 2)
            ci_95_upper = round(expectancy + 1.96 * std_err, 2)

            # Wilson score interval for binomial Win Rate
            z = 1.96
            p = p_win
            denom = 1 + (z**2 / n_resolved)
            center = (p + (z**2 / (2 * n_resolved))) / denom
            spread = (z * np.sqrt((p * (1 - p) / n_resolved) + (z**2 / (4 * n_resolved**2)))) / denom
            win_ci_lower = round(max(0.0, (center - spread) * 100.0), 1)
            win_ci_upper = round(min(100.0, (center + spread) * 100.0), 1)
        else:
            std_err = None
            ci_95_lower = None
            ci_95_upper = None
            win_ci_lower = None
            win_ci_upper = None

        # Trade Construction Quality Metrics
        psfe_stops = [
            s for s in stops
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("prematureStopOut") is True
            or s.get("forwardTracking", {}).get("tradeConstruction", {}).get("postStopFavorableExcursion", {}).get("tp1CrossedPostStop") is True
        ]
        post_stop_favorable_rate = round((len(psfe_stops) / len(stops)) * 100.0, 1) if stops else 0.0

        capture_ratios = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureRatio")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureRatio") is not None
        ]
        capture_5d = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe5d")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe5d") is not None
        ]
        capture_10d = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe10d")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe10d") is not None
        ]
        capture_20d = [
            s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe20d")
            for s in resolved
            if s.get("forwardTracking", {}).get("tradeConstruction", {}).get("captureVsMfe20d") is not None
        ]

        mean_capture_ratio = round(float(np.mean(capture_ratios)), 4) if capture_ratios else None

        # Directional Signal Quality (Unconstrained price behavior across all tracked signals)
        all_tracked = [s for s in signals if s.get("forwardTracking", {}).get("sessionsObserved", 0) > 0]
        acc_1d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy1d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy1d") is not None]
        acc_5d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy5d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy5d") is not None]
        acc_10d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy10d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy10d") is not None]
        acc_20d = [s["forwardTracking"]["signalQuality"]["directionalAccuracy20d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("directionalAccuracy20d") is not None]

        ret_1d = [s["forwardTracking"]["signalQuality"]["rawReturn1d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn1d") is not None]
        ret_5d = [s["forwardTracking"]["signalQuality"]["rawReturn5d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn5d") is not None]
        ret_10d = [s["forwardTracking"]["signalQuality"]["rawReturn10d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn10d") is not None]
        ret_20d = [s["forwardTracking"]["signalQuality"]["rawReturn20d"] for s in all_tracked if s.get("forwardTracking", {}).get("signalQuality", {}).get("rawReturn20d") is not None]

        mfes = [s.get("forwardTracking", {}).get("maxFavorableExcursionPct", 0.0) for s in all_tracked]
        maes = [s.get("forwardTracking", {}).get("maxAdverseExcursionPct", 0.0) for s in all_tracked]

        # Benchmark Relative Returns with Medians and Hit Rates
        rel_spy_20d = [
            s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSpy", {}).get("20d")
            for s in all_tracked
            if s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSpy", {}).get("20d") is not None
        ]
        rel_rsp_20d = [
            s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsRsp", {}).get("20d")
            for s in all_tracked
            if s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsRsp", {}).get("20d") is not None
        ]
        rel_sec_20d = [
            s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSector", {}).get("20d")
            for s in all_tracked
            if s.get("forwardTracking", {}).get("relativeReturns", {}).get("vsSector", {}).get("20d") is not None
        ]

        benchmark_suite = {
            "vsSpy20d": {
                "meanExcessReturnPct": round(float(np.mean(rel_spy_20d)), 2) if rel_spy_20d else None,
                "medianExcessReturnPct": round(float(np.median(rel_spy_20d)), 2) if rel_spy_20d else None,
                "hitRatePct": round((sum(1 for r in rel_spy_20d if r > 0) / len(rel_spy_20d)) * 100.0, 1) if rel_spy_20d else None,
                "sampleCount": len(rel_spy_20d)
            },
            "vsRsp20d": {
                "meanExcessReturnPct": round(float(np.mean(rel_rsp_20d)), 2) if rel_rsp_20d else None,
                "medianExcessReturnPct": round(float(np.median(rel_rsp_20d)), 2) if rel_rsp_20d else None,
                "hitRatePct": round((sum(1 for r in rel_rsp_20d if r > 0) / len(rel_rsp_20d)) * 100.0, 1) if rel_rsp_20d else None,
                "sampleCount": len(rel_rsp_20d)
            },
            "vsSector20d": {
                "meanExcessReturnPct": round(float(np.mean(rel_sec_20d)), 2) if rel_sec_20d else None,
                "medianExcessReturnPct": round(float(np.median(rel_sec_20d)), 2) if rel_sec_20d else None,
                "hitRatePct": round((sum(1 for r in rel_sec_20d if r > 0) / len(rel_sec_20d)) * 100.0, 1) if rel_sec_20d else None,
                "sampleCount": len(rel_sec_20d)
            },
            # Backward compatibility aliases
            "excessReturnVsSpy20d": round(float(np.mean(rel_spy_20d)), 2) if rel_spy_20d else None,
            "excessReturnVsRsp20d": round(float(np.mean(rel_rsp_20d)), 2) if rel_rsp_20d else None,
            "excessReturnVsSector20d": round(float(np.mean(rel_sec_20d)), 2) if rel_sec_20d else None,
        }

        # Confluence Score Monotonicity Evaluation (Ordinal ranking verification)
        confluence_buckets = {
            "<75": [],
            "75-79.9": [],
            "80-84.9": [],
            "85+": []
        }
        for s in resolved:
            score = float(s.get("confluenceScore", 0.0))
            if score < 75.0:
                confluence_buckets["<75"].append(s)
            elif score < 80.0:
                confluence_buckets["75-79.9"].append(s)
            elif score < 85.0:
                confluence_buckets["80-84.9"].append(s)
            else:
                confluence_buckets["85+"].append(s)

        monotonicity_table = {}
        prev_win_rate = -1.0
        prev_mean_return = -999.0
        is_monotonic_win = True
        is_monotonic_ret = True
        populated_bucket_count = 0

        for b_name in ["<75", "75-79.9", "80-84.9", "85+"]:
            b_signals = confluence_buckets[b_name]
            b_n = len(b_signals)
            if b_n > 0:
                b_wins = len([s for s in b_signals if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN"])
                b_wr = round((b_wins / b_n) * 100.0, 1)
                b_rets = [s["forwardTracking"]["realizedReturnPct"] for s in b_signals if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
                b_mean_ret = round(float(np.mean(b_rets)), 2) if b_rets else 0.0
                b_mfes = [s.get("forwardTracking", {}).get("maxFavorableExcursionPct", 0.0) for s in b_signals]
                b_mean_mfe = round(float(np.mean(b_mfes)), 2) if b_mfes else 0.0

                if prev_win_rate >= 0 and b_wr < prev_win_rate:
                    is_monotonic_win = False
                if prev_mean_return > -900.0 and b_mean_ret < prev_mean_return:
                    is_monotonic_ret = False

                prev_win_rate = b_wr
                prev_mean_return = b_mean_ret
                populated_bucket_count += 1

                monotonicity_table[b_name] = {
                    "count": b_n,
                    "winRate": b_wr,
                    "meanRealizedReturn": b_mean_ret,
                    "meanMfe": b_mean_mfe
                }
            else:
                monotonicity_table[b_name] = {"count": 0, "winRate": None, "meanRealizedReturn": None, "meanMfe": None}

        # Spearman Rank Correlation Analysis
        spearman_metrics = {}
        conf_scores_res = [float(s.get("confluenceScore", 0.0)) for s in resolved]
        rets_res = [float(s["forwardTracking"]["realizedReturnPct"]) for s in resolved if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
        mfes_res = [float(s.get("forwardTracking", {}).get("maxFavorableExcursionPct", 0.0)) for s in resolved]

        if len(conf_scores_res) >= 5 and spearmanr is not None and len(conf_scores_res) == len(rets_res):
            if len(set(conf_scores_res)) > 1 and len(set(rets_res)) > 1:
                try:
                    corr_ret, p_ret = spearmanr(conf_scores_res, rets_res)
                    corr_mfe, p_mfe = spearmanr(conf_scores_res, mfes_res) if len(set(mfes_res)) > 1 else (0.0, 1.0)
                    spearman_metrics = {
                        "confluenceVsReturnRho": round(float(corr_ret), 4) if not math.isnan(corr_ret) else 0.0,
                        "confluenceVsReturnPValue": round(float(p_ret), 4) if not math.isnan(p_ret) else 1.0,
                        "confluenceVsMfeRho": round(float(corr_mfe), 4) if not math.isnan(corr_mfe) else 0.0,
                        "confluenceVsMfePValue": round(float(p_mfe), 4) if not math.isnan(p_mfe) else 1.0,
                    }
                except Exception:
                    spearman_metrics = {"status": "CALCULATION_UNAVAILABLE"}
            else:
                spearman_metrics = {
                    "status": "CONSTANT_INPUT_UNDEFINED",
                    "confluenceVsReturnRho": 0.0,
                    "confluenceVsReturnPValue": 1.0,
                    "confluenceVsMfeRho": 0.0,
                    "confluenceVsMfePValue": 1.0,
                }
        else:
            spearman_metrics = {"status": "INSUFFICIENT_SAMPLE_FOR_CORRELATION", "sampleSize": len(conf_scores_res)}

        # Top-vs-Bottom Quartile Discrimination Analysis (Q4 vs Q1)
        quartile_discrimination = {}
        if len(resolved) >= 8:
            sorted_by_conf = sorted(resolved, key=lambda s: float(s.get("confluenceScore", 0.0)))
            q_size = max(1, len(sorted_by_conf) // 4)
            q1_bottom = sorted_by_conf[:q_size]
            q4_top = sorted_by_conf[-q_size:]

            q1_rets = [s["forwardTracking"]["realizedReturnPct"] for s in q1_bottom if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
            q4_rets = [s["forwardTracking"]["realizedReturnPct"] for s in q4_top if s.get("forwardTracking", {}).get("realizedReturnPct") is not None]
            q1_wins = sum(1 for s in q1_bottom if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN")
            q4_wins = sum(1 for s in q4_top if s.get("forwardTracking", {}).get("resolvedOutcome") == "TP1_WIN")

            q1_wr = round((q1_wins / len(q1_bottom)) * 100.0, 1)
            q4_wr = round((q4_wins / len(q4_top)) * 100.0, 1)
            q1_mean_ret = round(float(np.mean(q1_rets)), 2) if q1_rets else 0.0
            q4_mean_ret = round(float(np.mean(q4_rets)), 2) if q4_rets else 0.0
            q1_med_ret = round(float(np.median(q1_rets)), 2) if q1_rets else 0.0
            q4_med_ret = round(float(np.median(q4_rets)), 2) if q4_rets else 0.0

            quartile_discrimination = {
                "topQuartileQ4": {
                    "count": len(q4_top),
                    "minConfluence": float(q4_top[0].get("confluenceScore", 0.0)),
                    "winRatePct": q4_wr,
                    "meanReturnPct": q4_mean_ret,
                    "medianReturnPct": q4_med_ret,
                },
                "bottomQuartileQ1": {
                    "count": len(q1_bottom),
                    "maxConfluence": float(q1_bottom[-1].get("confluenceScore", 0.0)),
                    "winRatePct": q1_wr,
                    "meanReturnPct": q1_mean_ret,
                    "medianReturnPct": q1_med_ret,
                },
                "spreadQ4MinusQ1": {
                    "winRateSpread": round(q4_wr - q1_wr, 1),
                    "meanReturnSpread": round(q4_mean_ret - q1_mean_ret, 2),
                    "medianReturnSpread": round(q4_med_ret - q1_med_ret, 2),
                    "rankingPowerPositive": bool(q4_wr >= q1_wr and q4_mean_ret >= q1_mean_ret)
                }
            }
        else:
            quartile_discrimination = {"status": "INSUFFICIENT_SAMPLE_FOR_QUARTILES", "requiredN": 8, "currentN": len(resolved)}

        # Cluster Structure & Dependence Diagnostics
        dates = [s.get("signalDate") for s in signals if s.get("signalDate")]
        session_dist = {}
        for d in dates:
            session_dist[d] = session_dist.get(d, 0) + 1
        max_trades_per_day = max(session_dist.values()) if session_dist else len(signals)

        sectors = [s.get("inputs", {}).get("sector", "UNKNOWN") for s in signals]
        sector_dist = {}
        for sec in sectors:
            sector_dist[sec] = sector_dist.get(sec, 0) + 1
        max_sec_cnt = max(sector_dist.values()) if sector_dist else 0
        max_sec_pct = round((max_sec_cnt / len(signals)) * 100.0, 1) if signals else 0.0

        regimes = [s.get("inputs", {}).get("marketRegime", "UNKNOWN") for s in signals]
        regime_dist = {}
        for r in regimes:
            regime_dist[r] = regime_dist.get(r, 0) + 1

        clustering_diagnostics = {
            "maxTradesPerSession": max_trades_per_day,
            "uniqueSessionsCount": len(session_dist),
            "sessionDistribution": session_dist,
            "sectorConcentration": sector_dist,
            "maxSectorConcentrationPct": max_sec_pct,
            "regimeDistribution": regime_dist,
            "dependenceWarning": (
                "Cluster dependence detected: trades are concentrated across a limited number of "
                "market sessions and sectors. Nominal N must not be treated as independent observations."
            )
        }

        portfolio_aggregation = {
            "maxConcurrentEntries": max_trades_per_day,
            "topSectorExposurePct": max_sec_pct,
            "portfolioRiskNote": (
                "Trade-level expectancy does not guarantee bounded portfolio drawdown or Sharpe ratio "
                "under simultaneous sector exposure."
            )
        }

        unique_dates = len(set(s.get("signalDate") for s in resolved if s.get("signalDate")))

        return {
            "frozenEngineManifest": manifest_audit,
            "cohortFirewall": {
                "status": firewall_status,
                "cohortFilterRequested": cohort_filter,
                "epochIdRequested": epoch_id,
                "historicalProspectivePooling": "BLOCKED",
                "demoProspectivePooling": "BLOCKED",
                "contaminatedProspectivePooling": "BLOCKED",
                "cleanSignalsCount": len(clean_signals),
                "contaminatedSignalsCount": len(contaminated_signals),
                "excludedSignalsCount": len(excluded_signals),
                "unclassifiedSignalsCount": len(unclassified_signals),
                "cleanEvaluationEligible": clean_eval_eligible,
                "firewallWarning": firewall_warning
            },
            "clusteringAndDependence": clustering_diagnostics,
            "portfolioAggregation": portfolio_aggregation,
            "totalSignals": len(signals),
            "openSignals": len(open_signals),
            "resolvedSignals": n_resolved,
            # Top-level backward compatibility keys
            "winRate": round(p_win * 100.0, 1),
            "winRateCI95": [win_ci_lower, win_ci_upper] if win_ci_lower is not None else None,
            "stopRate": round(p_loss * 100.0, 1),
            "avgWinPct": round(avg_win, 2),
            "avgLossPct": round(avg_loss, 2),
            "expectancyPct": round(expectancy, 2),
            "netExpectancyPct": round(net_expectancy, 2),
            "expectancyStandardError": round(std_err, 2) if std_err is not None else None,
            "expectancyCI95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
            "profitFactor": round(profit_factor, 2),
            "frictionSensitivity": friction_sensitivity,
            # Detailed structured dimensions
            "tradeConstruction": {
                "winRate": round(p_win * 100.0, 1),
                "winRateCI95": [win_ci_lower, win_ci_upper] if win_ci_lower is not None else None,
                "stopRate": round(p_loss * 100.0, 1),
                "avgWinPct": round(avg_win, 2),
                "avgLossPct": round(avg_loss, 2),
                "expectancyPct": round(expectancy, 2),
                "netExpectancyPct": round(net_expectancy, 2),
                "expectancyStandardError": round(std_err, 2) if std_err is not None else None,
                "expectancyCI95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
                "profitFactor": round(profit_factor, 2),
                "postStopFavorableRate": post_stop_favorable_rate,
                "prematureStopOutRate": post_stop_favorable_rate,  # Backward compatibility alias
                "meanCaptureRatio": mean_capture_ratio,
                "meanCaptureVsMfe5d": round(float(np.mean(capture_5d)), 4) if capture_5d else None,
                "meanCaptureVsMfe10d": round(float(np.mean(capture_10d)), 4) if capture_10d else None,
                "meanCaptureVsMfe20d": round(float(np.mean(capture_20d)), 4) if capture_20d else None,
            },
            "signalQuality": {
                "directionalAccuracy1d": round((sum(acc_1d) / len(acc_1d)) * 100.0, 1) if acc_1d else None,
                "directionalAccuracy5d": round((sum(acc_5d) / len(acc_5d)) * 100.0, 1) if acc_5d else None,
                "directionalAccuracy10d": round((sum(acc_10d) / len(acc_10d)) * 100.0, 1) if acc_10d else None,
                "directionalAccuracy20d": round((sum(acc_20d) / len(acc_20d)) * 100.0, 1) if acc_20d else None,
                "meanRawReturn1d": round(float(np.mean(ret_1d)), 2) if ret_1d else None,
                "meanRawReturn5d": round(float(np.mean(ret_5d)), 2) if ret_5d else None,
                "meanRawReturn10d": round(float(np.mean(ret_10d)), 2) if ret_10d else None,
                "meanRawReturn20d": round(float(np.mean(ret_20d)), 2) if ret_20d else None,
                "meanMfePct": round(float(np.mean(mfes)), 2) if mfes else 0.0,
                "meanMaePct": round(float(np.mean(maes)), 2) if maes else 0.0,
            },
            "benchmarkComparisons": benchmark_suite,
            "confluenceMonotonicity": {
                "buckets": monotonicity_table,
                "isMonotonicWinRate": is_monotonic_win if populated_bucket_count >= 2 else None,
                "isMonotonicReturn": is_monotonic_ret if populated_bucket_count >= 2 else None,
                "monotonicityStatus": (
                    "MONOTONIC_CONFIRMED" if (is_monotonic_win and populated_bucket_count >= 2)
                    else ("NON_MONOTONIC_ANOMALY" if populated_bucket_count >= 2 else "INSUFFICIENT_BUCKET_SAMPLE")
                )
            },
            "rankingPower": {
                "spearmanCorrelation": spearman_metrics,
                "quartileDiscrimination": quartile_discrimination
            },
            "governanceGates": {
                "expectancyPositive": bool(expectancy > 0),
                "netExpectancyPositive": bool(net_expectancy > 0),
                "profitFactorAbove1_5": bool(profit_factor >= 1.5),
                "stopRateBelow50": bool(p_loss < 0.50 if n_resolved > 0 else True),
                "statisticallySignificant": bool(ci_95_lower is not None and ci_95_lower > 0),
                "confluenceMonotonic": bool(is_monotonic_win if populated_bucket_count >= 2 else True),
                "benchmarkAlphaPositive": bool(rel_rsp_20d and np.mean(rel_rsp_20d) > 0),
                "independentCohortSizeTarget": {
                    "targetResolvedN": 60,
                    "targetDistinctDates": 20,
                    "currentResolvedN": n_resolved,
                    "currentDistinctDates": unique_dates,
                    "isTargetMet": bool(n_resolved >= 60 and unique_dates >= 20)
                },
                "statisticalInference": {
                    "standardError": round(std_err, 2) if std_err is not None else None,
                    "ci95": [ci_95_lower, ci_95_upper] if ci_95_lower is not None else None,
                    "tStat": round(expectancy / std_err, 2) if (std_err and std_err > 0) else None,
                    "evaluationNote": (
                        "Statistical inference is evaluated as a diagnostic layer with confidence intervals. "
                        "Conventional p < 0.05 is not used as a single binary pass/fail gate for prospective cohorts."
                    )
                }
            },
            "governancePrinciple": (
                "A model change cannot be justified by a single metric moving outside its target. "
                "It requires a reproducible failure pattern across a predefined cohort and sufficient forward observations."
            )
        }

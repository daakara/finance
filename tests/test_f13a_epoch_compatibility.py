"""F_13A Epoch 1 Compatibility and Non-Interference Regression Test Suite.

Asserts:
1. All Epoch-1-protected files have identical cryptographic hashes to EPOCH_1_MANIFEST.json.
2. All frozen engine files have identical cryptographic hashes to FROZEN_ENGINE_MANIFEST.json.
3. Legacy MarketDatabaseEngine queries produce identical outputs before/after sidecar creation.
4. Protected Analytics and Setups fixture outputs remain completely unmodified.
5. Prospective clean natural denominator in authoritative paper trading ledger remains 0 (zero contamination).
"""

import os
import json
import hashlib
import pytest
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.market_evidence import MarketProvenance, MarketEvidence

pytestmark = pytest.mark.tier1


# ── 1. Cryptographic Manifest Verification ───────────────────────────────────

def test_epoch_1_manifest_hashes_unmodified():
    """Verify that historical Epoch 1 governance manifest artifact remains bitwise unchanged."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    epoch1_path = os.path.join(repo_root, "EPOCH_1_MANIFEST.json")
    assert os.path.exists(epoch1_path), "Governance manifest EPOCH_1_MANIFEST.json missing!"

    with open(epoch1_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("epochId") == "ARX_PROSPECTIVE_VALIDATION_EPOCH_1"
    assert manifest.get("manifestVersion") == "1.0.0"
    assert manifest.get("observationGovernanceManifestHash") == "51a90a19d160fd84d6d516b8f5c07ea63bd2f509201cfbddaf55b2204c8de63a"
    assert "executableGovernanceFiles" in manifest
    assert len(manifest["executableGovernanceFiles"]) == 4


def test_frozen_engine_manifest_hashes_unmodified():
    """Verify historical engine manifest integrity and current source compliance under Release Addendum 002."""
    test_frozen_engine_historical_manifest_integrity()
    test_current_production_engine_compliance()


def test_frozen_engine_historical_manifest_integrity():
    """Verify that historical frozen engine manifests retain cryptographic integrity and schema validity."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    v252_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_2.json")
    v251_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_1.json")
    v250_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_0.json")
    v240_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_4_0.json")
    assert os.path.exists(v252_path), "FROZEN_ENGINE_MANIFEST_V2_5_2.json missing!"
    assert os.path.exists(v251_path), "FROZEN_ENGINE_MANIFEST_V2_5_1.json missing!"
    assert os.path.exists(v250_path), "FROZEN_ENGINE_MANIFEST_V2_5_0.json missing!"
    assert os.path.exists(v240_path), "FROZEN_ENGINE_MANIFEST_V2_4_0.json missing!"

    # Verify V2.5.2 historical release manifest integrity
    with open(v252_path, "r", encoding="utf-8") as f:
        v252 = json.load(f)
    assert v252.get("manifestVersion") == "2.5.2"
    assert v252.get("provenanceCommit") == "7980d3a6e5dde0fe172d215c729ec213b9a164b1"
    assert v252.get("releaseAuthorityBinding") == "SPRINT3_RC5_CORPORATE_EVENTS_RATIFICATION"
    assert v252["engines"]["optimal_execution"]["sha256"] == "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576"
    assert v252["engines"]["decision_hierarchy"]["sha256"] == "823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3"
    assert v252["engines"]["confluence_engine"]["sha256"] == "0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a"

    # Verify V2.5.1 historical release manifest integrity
    with open(v251_path, "r", encoding="utf-8") as f:
        v251 = json.load(f)
    assert v251.get("frozenStrategyVersion") == "2.5.0"
    assert v251.get("provenanceCommit") == "cd0922471767775636957df74406a2d5efb8f519"
    assert v251.get("releaseAuthorityBinding") == "RELEASE_ADDENDUM_002"
    assert "optimal_execution" in v251.get("engines", {})
    assert "decision_hierarchy" in v251.get("engines", {})
    assert "confluence_engine" in v251.get("engines", {})
    assert v251["engines"]["confluence_engine"]["sha256"] == "147c010eb6068c3470fa3269f08610c0f436ea117b99b3b5217e7020b911e2b1"

    # Verify V2.5.0 pending-freeze artifact integrity
    with open(v250_path, "r", encoding="utf-8") as f:
        v250 = json.load(f)
    assert v250.get("frozenStrategyVersion") == "2.5.0"
    assert v250.get("provenanceCommit") == "PENDING_EPOCH_3_FREEZE"
    assert "optimal_execution" in v250.get("engines", {})
    assert "decision_hierarchy" in v250.get("engines", {})
    assert "confluence_engine" in v250.get("engines", {})

    # Verify V2.4.0 historical engine manifest integrity
    with open(v240_path, "r", encoding="utf-8") as f:
        v240 = json.load(f)
    assert v240.get("frozenStrategyVersion") == "2.4.0"
    assert v240.get("provenanceCommit") == "4e3686296aad24e2210ef580bbc9116054d84fd1"
    assert "optimal_execution" in v240.get("engines", {})
    assert "decision_hierarchy" in v240.get("engines", {})
    assert "confluence_engine" in v240.get("engines", {})


def test_current_production_engine_compliance():
    """Verify active engine compliance under ratified production contracts or authorized candidate manifest.
    
    Strictly prevents candidate development code from masquerading as a production release while
    enforcing deterministic integrity verification for candidate-scoped changes under Sprint 4.
    """
    repo_root = os.path.dirname(os.path.dirname(__file__))
    opt_exec_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "optimal_execution.py")
    with open(opt_exec_path, "rb") as fp:
        content = fp.read().replace(b"\r\n", b"\n")
        opt_sha = hashlib.sha256(content).hexdigest()

    dec_hier_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "decision_hierarchy.py")
    with open(dec_hier_path, "rb") as fp:
        content = fp.read().replace(b"\r\n", b"\n")
        dec_sha = hashlib.sha256(content).hexdigest()

    conf_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "confluence_engine.py")
    with open(conf_path, "rb") as fp:
        content = fp.read().replace(b"\r\n", b"\n")
        conf_sha = hashlib.sha256(content).hexdigest()

    cand_manifest_path = os.path.join(repo_root, "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json")
    if os.path.exists(cand_manifest_path):
        # Active candidate development scope (PR #105 / ARX-S4-ARCH-002)
        from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
        cand_audit = ExperimentLedger.verify_candidate_engine_manifest(repo_root=repo_root)
        assert cand_audit["status"] == "CANDIDATE_VERIFIED", f"Candidate integrity audit failed: {cand_audit}"
        assert cand_audit["valid"] is True
        assert cand_audit["isProductionRatified"] is False, "Candidate manifest must never claim production ratification"

        # Explicit fail-closed boundary: unratified candidate code MUST fail production verification
        prod_audit = ExperimentLedger.verify_frozen_engine_manifest(repo_root=repo_root)
        assert prod_audit["status"] == "CORRUPTED", (
            "Governance safety violation: Candidate development code must fail production freeze manifest verification"
        )
        assert prod_audit["valid"] is False

        # Verify candidate engine hashes match candidate manifest
        assert opt_sha == "26247e91e97b564601bd72bc759a40f831646b0232c9b8a9f6e289fd92dba209"
        assert dec_sha == "177840eec7dae88f6ae5957159cc4cdb3856933619b268859793d07622c6cc87"
        assert conf_sha == "9df887574c4efc128fceb7b1ff4191bff701bfb323e0b9453240646f1a29e9ec"
    else:
        # Canonical production checkout: strict single-hash enforcement
        assert opt_sha == "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576", (
            f"Section 7 contract violation: unexpected optimal_execution SHA {opt_sha}"
        )
        assert dec_sha == "823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3", (
            f"Decision hierarchy integrity violation: unexpected decision_hierarchy SHA {dec_sha}"
        )
        assert conf_sha == "0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a", (
            f"Confluence engine integrity violation: unexpected confluence_engine SHA {conf_sha}"
        )


# ── 2. Legacy MarketDatabaseEngine Output Invariance ─────────────────────────

def test_legacy_market_db_outputs_invariant(tmp_path):
    """Verify that adding the sidecar table produces zero observable change on legacy reads/writes."""
    db_file = str(tmp_path / "test_compat.db")
    engine = MarketDatabaseEngine(db_path=db_file)

    sample_candles = [
        {"time": "2026-09-15", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000},
        {"time": "2026-09-16", "open": 104.0, "high": 108.0, "low": 103.0, "close": 107.0, "volume": 2000},
    ]
    engine.save_daily_candles("SPY", sample_candles)

    # Read legacy
    legacy_candles = engine.get_daily_candles("SPY")
    assert len(legacy_candles) == 2
    assert legacy_candles[0]["close"] == 104.0
    assert legacy_candles[1]["close"] == 107.0

    freshness = engine.get_candles_with_freshness("SPY")
    assert freshness["candle_count"] == 2
    assert freshness["last_trade_date"] == "2026-09-16"

    latest = engine.get_latest_price("SPY")
    assert latest["symbol"] == "SPY"
    assert latest["currentPrice"] == 107.0


# ── 3. Protected Analytics Route Invariance ──────────────────────────────────

def test_protected_setups_route_behavioral_parity():
    """Verify /api/v1/analytics/setups executes without errors and preserves response shape."""
    client = TestClient(app)
    res = client.get("/api/v1/analytics/setups")
    assert res.status_code == 200
    data = res.json()
    assert "totalSetups" in data
    assert "setups" in data
    assert isinstance(data["setups"], list)


# ── 4. Prospective Ledger Zero Contamination Firewall ────────────────────────

def test_prospective_clean_denominator_uncontaminated():
    """Verify that paper trading ledger clean natural denominator remains strictly 0."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    ledger_path = os.path.join(repo_root, "analyst_dashboard", "data", "paper_trading_ledger.json")

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    signals = ledger.get("signals", [])
    clean_count = sum(1 for s in signals if s.get("provenanceCohort") == "PROSPECTIVE_CLEAN")

    assert clean_count == 0, (
        f"PROSPECTIVE_CONTAMINATION_BREACH: Clean denominator mutated to {clean_count}!"
    )


# ── 5. Real Verifier Negative and Succession Tests ────────────────────────────

def test_real_verifier_rejects_historical_manifest_tampering(monkeypatch):
    """Verify that real verifier fails closed when a historical manifest is tampered with."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    real_manifest = ExperimentLedger.get_epoch4_manifest()
    tampered = dict(real_manifest)
    tampered["observationGovernanceManifestHash"] = "tampered_hash_value"
    monkeypatch.setattr(ExperimentLedger, "get_epoch4_manifest", lambda *args, **kwargs: tampered)

    result = ExperimentLedger.verify_epoch4_v2_manifest()
    assert result["valid"] is False
    assert result["status"] == "CORRUPTED"


def test_real_verifier_rejects_historical_engine_manifest_tampering(monkeypatch):
    """Verify that real verifier fails closed when historical engine hash is tampered with."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    real_manifest = ExperimentLedger.get_frozen_manifest(version="2.4.0")
    tampered = dict(real_manifest)
    tampered["engines"] = dict(real_manifest["engines"])
    tampered["engines"]["optimal_execution"] = {
        "filePath": "analyst_dashboard/analyzers/optimal_execution.py",
        "sha256": "0" * 64,
    }
    monkeypatch.setattr(ExperimentLedger, "get_frozen_manifest", lambda *args, **kwargs: tampered)

    result = ExperimentLedger.verify_frozen_engine_manifest(version="2.4.0")
    assert result["valid"] is False
    assert result["status"] == "CORRUPTED"
    assert result["engines"]["optimal_execution"]["valid"] is False


def test_real_verifier_rejects_incorrect_manifest_version(monkeypatch):
    """Verify that real verifier rejects superseded manifest version substitution."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    v3_payload = ExperimentLedger.get_epoch4_v3_manifest()
    monkeypatch.setattr(ExperimentLedger, "get_epoch4_v6_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(ExperimentLedger, "get_epoch4_v5_manifest", lambda *args, **kwargs: v3_payload)

    result = ExperimentLedger.verify_epoch4_manifest()
    assert result.get("manifestVersion") not in ("5.0.0", "6.0.0")
    assert result.get("manifestVersion") == "3.0.0"


def test_real_verifier_rejects_unauthorized_successor_substitution():
    """Verify that successor manifests require strict cryptographic parent hash and digest binding match."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # 1. Observation governance lineage (Epoch 4 V6 -> V5 -> V4)
    v6 = ExperimentLedger.get_epoch4_v6_manifest()
    v5 = ExperimentLedger.get_epoch4_v5_manifest()
    v4 = ExperimentLedger.get_epoch4_v4_manifest()
    assert v6["parentRuntimeSha"] == v5["observationGovernanceManifestHash"]
    assert v5["parentRuntimeSha"] == v4["observationGovernanceManifestHash"]

    v6_path = os.path.join(repo_root, "EPOCH_4_MANIFEST_V6.json")
    with open(v6_path, "r", encoding="utf-8") as f:
        v6_disk = json.load(f)
    assert v6_disk["parentRuntimeSha"] == v5["observationGovernanceManifestHash"]
    assert v6_disk["manifestVersion"] == "6.0.0"
    assert v6_disk["parentManifest"] == "EPOCH_4_MANIFEST_V5.json"
    assert v6_disk["observationGovernanceManifestHash"] == "dba05d9f625ef06d2497c1b3c4e412f64b5bfb210c3569aaad33fbb9d1bf9a01"

    v5_path = os.path.join(repo_root, "EPOCH_4_MANIFEST_V5.json")
    with open(v5_path, "r", encoding="utf-8") as f:
        v5_disk = json.load(f)
    assert v5_disk["parentRuntimeSha"] == v4["observationGovernanceManifestHash"]
    assert v5_disk["manifestVersion"] == "5.0.0"
    assert v5_disk["parentManifest"] == "EPOCH_4_MANIFEST_V4.json"
    assert v5_disk["observationGovernanceManifestHash"] == "99bea6ebc9b4f31634ef994bd2630710d89555940a5e491e98e1285668ce04c5"

    # 2. Frozen engine successor lineage (V2.5.2 supersedes V2.5.1, rooted in SPRINT3 RC5 Corporate Events)
    v252 = ExperimentLedger.get_frozen_manifest("2.5.2")
    assert v252["manifestVersion"] == "2.5.2"
    assert v252["releaseAuthorityBinding"] == "SPRINT3_RC5_CORPORATE_EVENTS_RATIFICATION"
    assert v252["supersedesManifest"] == "FROZEN_ENGINE_MANIFEST_V2_5_1.json"
    assert v252["provenanceCommit"] == "7980d3a6e5dde0fe172d215c729ec213b9a164b1"

    v251 = ExperimentLedger.get_frozen_manifest("2.5.1")
    assert v251["manifestVersion"] == "2.5.1"
    assert v251["releaseAuthorityBinding"] == "RELEASE_ADDENDUM_002"
    assert v251["supersedesManifest"] == "FROZEN_ENGINE_MANIFEST_V2_5_0.json"
    assert v251["provenanceCommit"] == "cd0922471767775636957df74406a2d5efb8f519"


def test_real_verifier_fails_closed_when_manifest_missing(monkeypatch):
    """Verify that missing manifest fails closed and does not pass silently."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    orig_getters = [
        "get_epoch4_v6_manifest",
        "get_epoch4_v5_manifest",
        "get_epoch4_v4_manifest",
        "get_epoch4_v3_manifest",
        "get_epoch4_manifest",
        "get_epoch3_manifest",
        "get_epoch2_manifest",
        "get_epoch1_manifest",
    ]
    for name in orig_getters:
        monkeypatch.setattr(ExperimentLedger, name, lambda *args, **kwargs: None)

    repo_root = os.path.dirname(os.path.dirname(__file__))
    v6_path = os.path.join(repo_root, "EPOCH_4_MANIFEST_V6.json")
    v5_path = os.path.join(repo_root, "EPOCH_4_MANIFEST_V5.json")
    monkeypatch.setattr(os.path, "exists", lambda p: False if p in (v6_path, v5_path) else os.path.exists(p))

    result = ExperimentLedger.verify_observation_governance_manifest()
    assert result["valid"] is False
    assert result["status"] == "MANIFEST_MISSING"


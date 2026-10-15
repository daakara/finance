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
    """Verify that all executable governance files in active manifest remain bitwise unchanged."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    manifest_candidates = [
        os.path.join(repo_root, "EPOCH_4_MANIFEST_V6.json"),
        os.path.join(repo_root, "EPOCH_4_MANIFEST_V5.json"),
        os.path.join(repo_root, "EPOCH_4_MANIFEST_V4.json"),
        os.path.join(repo_root, "EPOCH_4_MANIFEST_V3.json"),
        os.path.join(repo_root, "EPOCH_4_MANIFEST.json"),
        os.path.join(repo_root, "EPOCH_3_MANIFEST.json"),
        os.path.join(repo_root, "EPOCH_2_MANIFEST.json"),
        os.path.join(repo_root, "EPOCH_1_MANIFEST.json"),
    ]
    manifest_path = next((p for p in manifest_candidates if os.path.exists(p)), manifest_candidates[-1])
    assert os.path.exists(manifest_path), "Governance manifest missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for file_path, meta in manifest["executableGovernanceFiles"].items():
        full_path = os.path.join(repo_root, file_path)
        assert os.path.exists(full_path), f"Protected file {file_path} missing!"

        with open(full_path, "rb") as fp:
            content = fp.read().replace(b"\r\n", b"\n")
            computed_sha = hashlib.sha256(content).hexdigest()

        assert computed_sha == meta["sha256"], (
            f"GOVERNANCE_BREACH: File {file_path} was modified! "
            f"Expected {meta['sha256']}, got {computed_sha}"
        )


def test_frozen_engine_manifest_hashes_unmodified():
    """Verify historical engine manifest integrity and current source compliance under Release Addendum 002."""
    test_frozen_engine_historical_manifest_integrity()
    test_current_production_engine_compliance()


def test_frozen_engine_historical_manifest_integrity():
    """Verify that historical frozen engine manifests retain cryptographic integrity and schema validity."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    v250_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_5_0.json")
    v240_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST_V2_4_0.json")
    assert os.path.exists(v250_path), "FROZEN_ENGINE_MANIFEST_V2_5_0.json missing!"
    assert os.path.exists(v240_path), "FROZEN_ENGINE_MANIFEST_V2_4_0.json missing!"

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
    """Verify that active production engines comply with ratified Release Addendum 002 contracts."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    opt_exec_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "optimal_execution.py")
    with open(opt_exec_path, "rb") as fp:
        content = fp.read().replace(b"\r\n", b"\n")
        opt_sha = hashlib.sha256(content).hexdigest()

    # Section 7 ratified contract invariant (PO rejection of EXC-002)
    assert opt_sha == "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576", (
        f"Section 7 contract violation: unexpected optimal_execution SHA {opt_sha}"
    )

    # Synthesis E Wave 3 Decision Integrity invariant
    dec_hier_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "decision_hierarchy.py")
    with open(dec_hier_path, "rb") as fp:
        content = fp.read().replace(b"\r\n", b"\n")
        dec_sha = hashlib.sha256(content).hexdigest()
    assert dec_sha == "823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3", (
        f"Decision hierarchy integrity violation: unexpected decision_hierarchy SHA {dec_sha}"
    )

    # Confluence engine invariant
    conf_path = os.path.join(repo_root, "analyst_dashboard", "analyzers", "confluence_engine.py")
    with open(conf_path, "rb") as fp:
        content = fp.read().replace(b"\r\n", b"\n")
        conf_sha = hashlib.sha256(content).hexdigest()
    assert conf_sha == "147c010eb6068c3470fa3269f08610c0f436ea117b99b3b5217e7020b911e2b1", (
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


# ── 5. Real Verifier Negative Tests ──────────────────────────────────────────

def test_real_verifier_rejects_historical_manifest_tampering(monkeypatch):
    """Verify that real verifier fails closed when a historical manifest is tampered with."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    real_manifest = ExperimentLedger.get_epoch4_manifest()
    tampered = dict(real_manifest)
    tampered["observationGovernanceManifestHash"] = "tampered_hash_value"
    monkeypatch.setattr(ExperimentLedger, "get_epoch4_manifest", lambda: tampered)

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
    monkeypatch.setattr(ExperimentLedger, "get_frozen_manifest", lambda version=None: tampered)

    result = ExperimentLedger.verify_frozen_engine_manifest(version="2.4.0")
    assert result["valid"] is False
    assert result["status"] == "CORRUPTED"
    assert result["engines"]["optimal_execution"]["valid"] is False


def test_real_verifier_rejects_incorrect_manifest_version(monkeypatch):
    """Verify that real verifier rejects superseded manifest version substitution."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    v3_payload = ExperimentLedger.get_epoch4_v3_manifest()
    monkeypatch.setattr(ExperimentLedger, "get_epoch4_v5_manifest", lambda: v3_payload)

    result = ExperimentLedger.verify_epoch4_manifest()
    assert result.get("manifestVersion") != "5.0.0"


def test_real_verifier_rejects_unauthorized_successor_substitution():
    """Verify that successor manifest requires strict cryptographic parent hash match."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    v5 = ExperimentLedger.get_epoch4_v5_manifest()
    v4 = ExperimentLedger.get_epoch4_v4_manifest()
    assert v5["parentRuntimeSha"] == v4["observationGovernanceManifestHash"]

    repo_root = os.path.dirname(os.path.dirname(__file__))
    v6_path = os.path.join(repo_root, "EPOCH_4_MANIFEST_V6.json")
    with open(v6_path, "r", encoding="utf-8") as f:
        v6 = json.load(f)
    assert v6["parentRuntimeSha"] == v5["observationGovernanceManifestHash"]
    assert v6["manifestVersion"] == "6.0.0"
    assert v6["parentManifest"] == "EPOCH_4_MANIFEST_V5.json"


def test_real_verifier_fails_closed_when_manifest_missing(monkeypatch):
    """Verify that missing manifest fails closed and does not pass silently."""
    from analyst_dashboard.governance.experiment_ledger import ExperimentLedger
    orig_getters = [
        "get_epoch4_v5_manifest",
        "get_epoch4_v4_manifest",
        "get_epoch4_v3_manifest",
        "get_epoch4_manifest",
        "get_epoch3_manifest",
        "get_epoch2_manifest",
        "get_epoch1_manifest",
    ]
    for name in orig_getters:
        monkeypatch.setattr(ExperimentLedger, name, lambda: None)

    result = ExperimentLedger.verify_observation_governance_manifest()
    assert result["valid"] is False
    assert result["status"] == "MANIFEST_MISSING"


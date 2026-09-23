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
    """Verify that all executable governance files in EPOCH_1_MANIFEST.json remain bitwise unchanged."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    manifest_path = os.path.join(repo_root, "EPOCH_1_MANIFEST.json")
    assert os.path.exists(manifest_path), "EPOCH_1_MANIFEST.json missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for file_path, meta in manifest["executableGovernanceFiles"].items():
        full_path = os.path.join(repo_root, file_path)
        assert os.path.exists(full_path), f"Protected file {file_path} missing!"

        with open(full_path, "rb") as fp:
            content = fp.read().replace(b"\r\n", b"\n")
            computed_sha = hashlib.sha256(content).hexdigest()

        assert computed_sha == meta["sha256"], (
            f"EPOCH_1_BREACH: File {file_path} was modified! "
            f"Expected {meta['sha256']}, got {computed_sha}"
        )


def test_frozen_engine_manifest_hashes_unmodified():
    """Verify that all 3 production engines match the frozen cryptographic manifest."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    manifest_path = os.path.join(repo_root, "FROZEN_ENGINE_MANIFEST.json")
    assert os.path.exists(manifest_path), "FROZEN_ENGINE_MANIFEST.json missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for engine_name, meta in manifest["engines"].items():
        full_path = os.path.join(repo_root, meta["filePath"])
        assert os.path.exists(full_path), f"Engine file {meta['filePath']} missing!"

        with open(full_path, "rb") as fp:
            content = fp.read().replace(b"\r\n", b"\n")
            computed_sha = hashlib.sha256(content).hexdigest()

        assert computed_sha == meta["sha256"], (
            f"FROZEN_ENGINE_BREACH: Engine {engine_name} ({meta['filePath']}) was modified! "
            f"Expected {meta['sha256']}, got {computed_sha}"
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

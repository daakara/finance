"""ARX-FINANCIAL-AUTHORITY-039: ratified production manifest attacks.

All fixtures run against isolated copies; no production engine or frozen
historical manifest is modified. Existing NEG-26 and NEG-27 retain their
original identity-attack meanings.
"""
import json
import shutil
from pathlib import Path

import pytest

from scripts.release import rq10_config_integrity as rq10

ROOT = Path(__file__).resolve().parents[1]
GOVERNED = [
    "FROZEN_ENGINE_MANIFEST_V2_5_3.json",
    "docs/governance/ARX_RC5_ENGINE_MANIFEST_V253_SUCCESSION_RECORD.md",
    "analyst_dashboard/governance/experiment_ledger.py",
    *[path for path, _ in rq10.EXPECTED_V253_DIGESTS.values()],
]


@pytest.fixture
def isolated_manifest(monkeypatch, tmp_path):
    for rel in GOVERNED:
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    monkeypatch.setattr(rq10, "PROJECT_ROOT", str(tmp_path))
    return tmp_path


def update_manifest(tmp_path, callback):
    path = tmp_path / rq10.V253_MANIFEST
    obj = json.loads(path.read_text(encoding="utf-8"))
    callback(obj)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def test_v253_production_manifest_passes_with_all_three_hashes(isolated_manifest):
    result = rq10.verify_engine_manifest("release/production")
    assert result["manifest_version"] == "2.5.3"
    assert set(result["engine_hashes"]) == set(rq10.EXPECTED_V253_DIGESTS)


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(isProductionRatified=False),
    lambda m: m.update(authorityScope="CANDIDATE_DEVELOPMENT_ONLY"),
    lambda m: m.update(provenanceCommit="0" * 40),
    lambda m: m["engines"]["confluence_engine"].update(sha256="0" * 64),
    lambda m: m["engines"]["optimal_execution"].update(sha256="0" * 64),
    lambda m: m["engines"]["decision_hierarchy"].update(sha256="0" * 64),
])
def test_production_rejects_unratified_or_tampered_manifest(isolated_manifest, mutation):
    update_manifest(isolated_manifest, mutation)
    with pytest.raises(AssertionError):
        rq10.verify_engine_manifest("release/production")


def test_missing_ratification_document_fails_closed(isolated_manifest):
    (isolated_manifest / rq10.V253_ADR).unlink()
    with pytest.raises(AssertionError, match="decision record"):
        rq10.verify_engine_manifest("release/production")


def test_tampered_source_hash_fails_closed(isolated_manifest):
    source = isolated_manifest / rq10.EXPECTED_V253_DIGESTS["optimal_execution"][0]
    source.write_bytes(source.read_bytes() + b"\n# tampered source\n")
    with pytest.raises(AssertionError, match="Engine digest mismatch"):
        rq10.verify_engine_manifest("release/production")


def test_missing_ledger_binding_fails_closed(isolated_manifest):
    ledger = isolated_manifest / "analyst_dashboard/governance/experiment_ledger.py"
    ledger.write_text(
        ledger.read_text(encoding="utf-8").replace(
            '"2.5.3": {', '"9.9.9": {', 1
        ), encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="ledger|Ledger|registration"):
        rq10.verify_engine_manifest("release/production")


def test_forged_boolean_cannot_bypass_digest_pin(isolated_manifest):
    update_manifest(isolated_manifest, lambda m: m.update(isProductionRatified=True, frozenTimestamp="2099-01-01T00:00:00Z"))
    with pytest.raises(AssertionError, match="digest mismatch"):
        rq10.verify_engine_manifest("release/production")


def test_missing_manifest_fails_closed(isolated_manifest):
    (isolated_manifest / rq10.V253_MANIFEST).unlink()
    with pytest.raises(AssertionError, match="manifest missing"):
        rq10.verify_engine_manifest("release/production")


def test_unauthorized_target_fails_closed(isolated_manifest):
    with pytest.raises(AssertionError, match="Unrecognized"):
        rq10.verify_engine_manifest("feature/rogue")

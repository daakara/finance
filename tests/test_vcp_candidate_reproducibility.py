"""ARX Terminal — Candidate 004 vs Candidate 005 Reproducibility & Dual Execution Suite.

Proves:
1. Default instantiation of VCPClassifier selects Candidate 004.
2. Candidate 004 produces 100% bit-exact outputs matching frozen baseline records on HVA-00 development cases.
3. Candidate 004 and Candidate 005 can execute side-by-side against identical inputs without state leakage.
4. Numeric contract defaults preserve Candidate 004 monotonic tightening rules.
"""

import json
from pathlib import Path
import sqlite3
import pytest

from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.numeric_contract import VCPNumericContract
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar


def test_01_default_classifier_is_candidate_004():
    """Default classifier instantiation must be Candidate 004 to preserve production behavior."""
    clf = VCPClassifier()
    assert clf.algorithm_version == VCPClassifier.VERSION_CANDIDATE_004
    assert clf.algorithm_version == "ARX_VCP_CLASSIFIER_V1_CANDIDATE_004"


def test_02_numeric_contract_defaults_preserve_candidate_004():
    """VCPNumericContract.verify_progressive_tightening default arguments must enforce strict monotonic decrease."""
    nc = VCPNumericContract()
    # Shallow depths without 5% reduction: 0.0400 to 0.0399 (decrease of 0.25%, not 5%)
    # Under Candidate 004 (relative_reduction=0.0): PASS
    assert nc.verify_progressive_tightening([0.0400, 0.0399]) is True
    # Under Candidate 005 (relative_reduction=0.05): FAIL (0.0400 * 0.95 = 0.0380, 0.0399 > 0.0380)
    assert nc.verify_progressive_tightening([0.0400, 0.0399], relative_reduction=0.05) is False


def test_03_candidate_004_bit_exact_reproducibility_on_frozen_cases():
    """Proves Candidate 004 yields bit-exact outputs matching vcp_hva00_frozen_algorithm_results.json."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = worktree_root / "data" / "research" / "vcp_hva00_case_manifest.json"
    frozen_path = worktree_root / "data" / "research" / "vcp_hva00_frozen_algorithm_results.json"

    if not (db_path.exists() and manifest_path.exists() and frozen_path.exists()):
        pytest.skip("Research data files not present in worktree; skipping frozen replay.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(frozen_path, "r", encoding="utf-8") as f:
        frozen = json.load(f)

    frozen_records = {r["case_id"]: r for r in frozen["algorithm_records"]}
    pos_cases = [c for c in manifest["cases"] if c.get("stratum") == "CLEAR_POSITIVE_VCP"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    compiler = VCPTemporalCaseCompiler()
    clf_c004 = VCPClassifier()  # default Candidate 004

    discrepancies = []

    for defn in pos_cases:
        cid = defn["case_id"]
        sym = defn["symbol"]
        cutoff = defn["evaluation_cutoff_date"]
        cutoff_iso = f"{cutoff}T21:00:00Z"

        rows = cursor.execute(
            """
            SELECT trade_date, open, high, low, close, volume
            FROM asset_ohlcv_daily
            WHERE symbol = ? AND trade_date <= ?
            ORDER BY trade_date ASC
            """,
            (sym, cutoff),
        ).fetchall()

        bars = [
            DailyOHLCVBar(
                symbol=sym,
                bar_date=r[0],
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=int(r[5]),
                valid_time=f"{r[0]}T21:00:00Z",
                known_at=f"{r[0]}T22:00:00Z",
            )
            for r in rows
        ]

        pkg = compiler.compile_case(
            case_id=cid,
            security_id=f"SEC-{sym}",
            symbol=sym,
            evaluation_as_of=cutoff_iso,
            raw_bars=bars,
            reference_data={"rs_rank": 85.0},
            corporate_actions=[],
        )

        assessment = clf_c004.classify_case(pkg)
        obs = clf_c004.extract_observations(pkg)
        frz = frozen_records.get(cid, {})

        qual_match = (assessment.vcp_qualified == frz.get("vcp_qualified"))
        class_match = (assessment.vcp_classification == frz.get("vcp_classification"))
        depths_match = (obs.contraction_depths == frz.get("contraction_depths"))
        tight_match = (obs.is_progressive_tightening == frz.get("is_progressive_tightening"))
        vol_match = (obs.final_wave_volume_ratio == frz.get("volume_dry_up_ratio"))

        if not (qual_match and class_match and depths_match and tight_match and vol_match):
            discrepancies.append((cid, {
                "qual": (assessment.vcp_qualified, frz.get("vcp_qualified")),
                "class": (assessment.vcp_classification, frz.get("vcp_classification")),
                "depths": (obs.contraction_depths, frz.get("contraction_depths")),
                "tight": (obs.is_progressive_tightening, frz.get("is_progressive_tightening")),
                "vol": (obs.final_wave_volume_ratio, frz.get("volume_dry_up_ratio")),
            }))

    conn.close()
    assert len(discrepancies) == 0, f"Discrepancies found against frozen baseline: {discrepancies}"


def test_04_dual_candidate_isolated_side_by_side_execution():
    """Instantiating both Candidate 004 and Candidate 005 executes cleanly without shared state contamination."""
    worktree_root = Path(__file__).resolve().parent.parent
    db_path = worktree_root / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
    manifest_path = worktree_root / "data" / "research" / "vcp_hva00_case_manifest.json"

    if not (db_path.exists() and manifest_path.exists()):
        pytest.skip("Database or manifest not present in worktree.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    case_defn = manifest["cases"][0]
    sym = case_defn["symbol"]
    cutoff = case_defn["evaluation_cutoff_date"]
    cutoff_iso = f"{cutoff}T21:00:00Z"

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    rows = cursor.execute(
        """
        SELECT trade_date, open, high, low, close, volume
        FROM asset_ohlcv_daily
        WHERE symbol = ? AND trade_date <= ?
        ORDER BY trade_date ASC
        """,
        (sym, cutoff),
    ).fetchall()
    conn.close()

    bars = [
        DailyOHLCVBar(
            symbol=sym,
            bar_date=r[0],
            open=float(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            volume=int(r[5]),
            valid_time=f"{r[0]}T21:00:00Z",
            known_at=f"{r[0]}T22:00:00Z",
        )
        for r in rows
    ]

    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id=case_defn["case_id"],
        security_id=f"SEC-{sym}",
        symbol=sym,
        evaluation_as_of=cutoff_iso,
        raw_bars=bars,
        reference_data={"rs_rank": 85.0},
        corporate_actions=[],
    )

    clf_004 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_004)
    clf_005 = VCPClassifier(algorithm_version=VCPClassifier.VERSION_CANDIDATE_005)

    obs_004 = clf_004.extract_observations(pkg)
    obs_005 = clf_005.extract_observations(pkg)

    # Candidate 004 does not populate macro metadata
    assert obs_004.metadata == {}
    # Candidate 005 populates macro metadata
    assert "algorithm_version" in obs_005.metadata
    assert obs_005.metadata["algorithm_version"] == VCPClassifier.VERSION_CANDIDATE_005
    assert "anchor_selected" in obs_005.metadata

    # Running them again does not alter version identities
    assert clf_004.algorithm_version == VCPClassifier.VERSION_CANDIDATE_004
    assert clf_005.algorithm_version == VCPClassifier.VERSION_CANDIDATE_005

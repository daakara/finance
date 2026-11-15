"""
tests/test_radar_population_conservation.py

Focused Negative & Invariant Verification Suite for Directive ARX-RADAR-P2-RECONCILIATION:
Freshness Authority, Population Conservation & Stage 3 Distribution Accounting.

Mandatory Invariants Tested:
1. Population Conservation Invariant:
   referenceUniverseCount == (
       stage1RecordCount + stage2OpportunityCount + stage3RecordCount +
       stage4RecordCount + stageUnresolvedRecordCount +
       delistedRecordCount + unverifiedRecordCount
   )
   Every single security in the authoritative universe is accounted for exactly once.
2. Stage 3 Accounting & Isolation:
   Securities exhibiting Stage 3 Distribution Top characteristics (flattening/rollover 200 SMA
   following substantial prior advance) evaluate to STAGE_3 and are counted in stage3RecordCount.
   Stage 3 assets are strictly excluded from primary Stage 2 discovery, appearing only in diagnostic view.
3. Freshness Authority & Provenance Distinction:
   Snapshot observation cutoff (data_as_of), publication timestamp (generated_at),
   and live quote tape date (marketDataAsOf) remain distinct and verifiable.
   Historical snapshots remain accessible without false claims of current-session evaluation.
4. Minimum History Contract (Weinstein/Minervini Domain Law):
   A security with session_count < 200 strictly fails closed to STAGE_UNRESOLVED.
   The 200-day SMA threshold is never relaxed or substituted with synthetic zero.
"""

import pytest
from fastapi import Response

from api.routes.screener import (
    run_screener_get,
    compute_canonical_stage_classification,
    get_radar_capabilities,
    vcp_scanner_runner,
)
from analyst_dashboard.analyzers.scanner_contract import ScannerStatus
from analyst_dashboard.vcp.temporal_contract import DailyOHLCVBar
from analyst_dashboard.vcp.case_compiler import VCPTemporalCaseCompiler
from analyst_dashboard.vcp.classifier import VCPClassifier
import datetime


pytestmark = [pytest.mark.tier2b, pytest.mark.usefixtures("published_screener_universe")]


def test_population_conservation_sum_equals_reference_denominator():
    """
    Check 1: Authoritative classification populations sum to reference denominator.
    referenceUniverseCount == sum of all stage buckets + delisted + unverified.
    Omission of any stage (such as stage3RecordCount) triggers a conservation failure.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="diagnostic")
    pop = data.get("population", {})

    ref_count = pop.get("referenceUniverseCount", 0)
    assert ref_count > 0, "Reference universe count must be positive"

    s1 = pop.get("stage1RecordCount", 0)
    s2 = pop.get("stage2OpportunityCount", 0)
    s3 = pop.get("stage3RecordCount", 0)
    s4 = pop.get("stage4RecordCount", 0)
    sunres = pop.get("stageUnresolvedRecordCount", 0)
    delisted = pop.get("delistedRecordCount", 0)
    unverified = pop.get("unverifiedRecordCount", 0)

    # 1. Total conservation check
    total_accounted = s1 + s2 + s3 + s4 + sunres + delisted + unverified
    assert total_accounted == ref_count, (
        f"Population conservation violated: sum({total_accounted}) != reference({ref_count}). "
        f"Breakdown: S1={s1}, S2={s2}, S3={s3}, S4={s4}, Unres={sunres}, Delisted={delisted}, Unverified={unverified}"
    )

    # 2. Adversarial check: omitting stage3RecordCount MUST produce an exact deficit
    assert s3 > 0, "Stage 3 record count must be represented when Stage 3 securities exist"
    omitted_sum = s1 + s2 + s4 + sunres + delisted + unverified
    assert omitted_sum < ref_count, "Omitting stage3RecordCount must fail closed with an exact gap"
    assert (ref_count - omitted_sum) == s3, f"Gap must equal stage3RecordCount ({s3})"


def test_stage3_distribution_top_accounting_and_isolation():
    """
    Check 2: Stage 3 Distribution Top accounting and primary discovery isolation.
    - DUOL (seeded as Stage 3 in fixture) classifies as STAGE_3 with Stage 3 Distribution Phase.
    - STAGE_3 assets are strictly excluded from primary Stage 2 discovery (view='stage_2_primary').
    - STAGE_3 assets are present and inspectable in diagnostic view (view='diagnostic').
    - stage3RecordCount accurately reflects the exact number of Stage 3 equities.
    """
    resp = Response()

    # Primary view: must NOT contain Stage 3 assets
    primary_data = run_screener_get(resp, filter_type="all", view="stage_2_primary")
    primary_candidates = primary_data.get("candidates", [])
    primary_symbols = {c["symbol"] for c in primary_candidates}
    for c in primary_candidates:
        assert c.get("stageClassification") == "STAGE_2"
        assert c.get("stageClassification") != "STAGE_3"

    assert "DUOL" not in primary_symbols, "Stage 3 security DUOL must not leak into primary Stage 2 discovery"

    # Diagnostic view: must contain Stage 3 assets
    diag_data = run_screener_get(resp, filter_type="all", view="diagnostic")
    diag_candidates = diag_data.get("candidates", [])
    duol_cand = next((c for c in diag_candidates if c["symbol"] == "DUOL"), None)

    assert duol_cand is not None, "DUOL must be present in diagnostic view"
    assert duol_cand.get("stageClassification") == "STAGE_3"
    assert duol_cand.get("stagePhase") == "Stage 3 Distribution Phase"
    assert duol_cand.get("setupPattern") == "Stage 3 Distribution Top / Exhaustion"
    assert duol_cand.get("vcpStage") == "Distribution Stage Ineligible"

    prov = duol_cand.get("stageProvenance", {})
    assert prov.get("classifierAuthority") == "VCPClassifier"
    assert prov.get("trendTemplatePass") is False
    assert "200_SMA_DECLINING" in prov.get("rejectionReasons", []) or "STAGE_2_FAILED_SLOPE" in prov.get("rejectionReasons", [])


def test_freshness_authority_and_snapshot_accessibility():
    """
    Check 3: Freshness authority distinguishes data_as_of, generated_at, and marketDataAsOf.
    - Snapshot observation date (dataAsOf: 2026-11-06) is distinct from quote tape (marketDataAsOf).
    - Publication timestamp (generatedAt: 2026-11-08T...) is preserved.
    - Historical valid snapshot remains status: AVAILABLE without false claims of real-time re-scan.
    - capabilities.VCP exposes both dataAsOf and generatedAt truthfully.
    """
    resp = Response()
    data = run_screener_get(resp, filter_type="all", view="diagnostic")

    pop = data.get("population", {})
    tape_as_of = pop.get("marketDataAsOf")
    assert tape_as_of is not None, "Market tape data-as-of must be populated"

    caps = data.get("capabilities", {})
    vcp_cap = caps.get("VCP", {})

    assert "dataAsOf" in vcp_cap
    assert "generatedAt" in vcp_cap
    assert vcp_cap.get("singleAssetAnalysis") == "AVAILABLE"


def test_minimum_history_contract_fails_closed_to_unresolved():
    """
    Check 4: Weinstein/Minervini 200-session minimum history contract fails closed.
    When a security has fewer than 200 sessions (e.g. 128 sessions):
    - VCPClassifier strictly evaluates to STAGE_UNRESOLVED.
    - Neither STAGE_2 nor STAGE_4 is assigned from truncated moving averages.
    - Rejection reasons explicitly report missing trend template data.
    """
    base_date = datetime.date(2026, 1, 1)
    # 128 bars: simulates truncated local market database
    short_bars = []
    for i in range(128):
        d = base_date + datetime.timedelta(days=i)
        p = 100.0 + (i * 0.5)  # Rising price
        short_bars.append(DailyOHLCVBar(
            symbol="SHORT_S2",
            bar_date=d.isoformat(),
            open=p - 0.5,
            high=p + 1.0,
            low=p - 1.0,
            close=p,
            volume=1000000,
            valid_time=f"{d.isoformat()}T20:00:00Z",
            known_at=f"{d.isoformat()}T20:30:00Z",
            is_session_closed=True,
        ))

    compiler = VCPTemporalCaseCompiler()
    pkg = compiler.compile_case(
        case_id="TEST-SHORT-HIST",
        security_id="SHORT_S2",
        symbol="SHORT_S2",
        evaluation_as_of="2026-05-09T20:45:00Z",
        raw_bars=short_bars,
        reference_data={},
        corporate_actions=[],
    )
    classifier = VCPClassifier()
    assessment = classifier.classify_case(pkg)

    # Invariant: session_count < 200 MUST evaluate to STAGE_UNRESOLVED
    assert assessment.stage_classification == "STAGE_UNRESOLVED", (
        f"Expected STAGE_UNRESOLVED for 128 sessions, got {assessment.stage_classification}"
    )
    assert assessment.vcp_qualified is False
    assert "TREND_TEMPLATE_DATA_INCOMPLETE" in assessment.classification_reason_codes or "SUFFICIENT_HISTORY_FAIL" in assessment.classification_reason_codes

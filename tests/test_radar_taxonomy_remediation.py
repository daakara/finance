"""
Contract and Adversarial Regression Tests for Radar Taxonomy Authority & Evidence Remediation.
Enforces canonical taxonomy (VALUE_GARP, VCP, SMART_MONEY), root-level capabilities,
zero-semantics invariants, removal of fabricated archetype fallbacks, and immunity
to substring-based category inference.
"""

import pytest
from fastapi import Response
from api.routes.screener import (
    run_screener_get,
    CANONICAL_RADAR_CATEGORIES,
    RADAR_CAPABILITY_CONTRACT,
    derive_canonical_radar_categories,
    classify_candidate_category_evidence,
)


pytestmark = pytest.mark.tier2b


def test_canonical_vocabulary_constants():
    """Requirement 1: Establish one canonical category vocabulary."""
    expected = {"VALUE_GARP", "VCP", "SMART_MONEY"}
    assert set(CANONICAL_RADAR_CATEGORIES) == expected
    assert "VALUE" not in CANONICAL_RADAR_CATEGORIES


def test_root_capability_contract():
    """Requirement 3: Explicit root capability metadata distinguishing scope and status."""
    caps = RADAR_CAPABILITY_CONTRACT
    assert caps["VALUE_GARP"]["status"] == "AVAILABLE"
    assert caps["VALUE_GARP"]["universeScreening"] == "AVAILABLE"
    assert caps["VALUE_GARP"]["singleAssetAnalysis"] == "AVAILABLE"

    assert caps["VCP"]["status"] == "PIPELINE_PENDING"
    assert caps["VCP"]["universeScreening"] == "PIPELINE_PENDING"
    assert caps["VCP"]["singleAssetAnalysis"] == "AVAILABLE"

    assert caps["SMART_MONEY"]["status"] == "PIPELINE_PENDING"
    assert caps["SMART_MONEY"]["universeScreening"] == "PIPELINE_PENDING"
    assert caps["SMART_MONEY"]["singleAssetAnalysis"] == "AVAILABLE"


def test_screener_payload_exposes_canonical_categories_and_capabilities():
    """Requirement 1, 2, 3: Endpoint emits typed categories, categoryEvidence, and capabilities."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    assert "capabilities" in data
    assert data["capabilities"] == RADAR_CAPABILITY_CONTRACT

    candidates = data.get("candidates", [])
    assert len(candidates) > 0

    for c in candidates:
        assert "categories" in c
        assert isinstance(c["categories"], list)
        for cat in c["categories"]:
            assert cat in CANONICAL_RADAR_CATEGORIES, f"Candidate has invalid category: {cat}"

        assert "categoryEvidence" in c
        assert isinstance(c["categoryEvidence"], dict)
        for cat, ev in c["categoryEvidence"].items():
            assert cat in CANONICAL_RADAR_CATEGORIES
            assert ev in ["CRITERIA_MATCHED", "CRITERIA_UNMET", "UNASSESSED"]


def test_fallback_archetype_fabrication_removed():
    """Requirement 5: Fallback archetypes must NOT synthesize an investment strategy."""
    # Test with an uncataloged custom ticker
    resp = Response()
    data = run_screener_get(resp, filter_type="all", custom_tickers="ZZZZNONEXISTENT")
    candidates = data.get("candidates", [])
    assert len(candidates) == 1
    c = candidates[0]
    assert c["symbol"] == "ZZZZNONEXISTENT"
    # expertArchetype must NOT be fabricated as "Peter Lynch GARP Compounder" or "High-Beta Momentum Leader"
    assert c["expertArchetype"] != "Peter Lynch GARP Compounder"
    assert c["expertArchetype"] != "High-Beta Momentum Leader"
    assert c["expertArchetype"] != "Minervini Stage 2 VCP"
    assert c["categories"] == []
    assert c["categoryEvidence"] == {}


def test_adversarial_substring_cannot_generate_categories():
    """Requirement 9 Adversarial: Strings alone must NOT generate category membership."""
    # 1. "Smart Growth Compounder" must not become SMART_MONEY
    cats_1 = derive_canonical_radar_categories("Smart Growth Compounder", None, None, None, "AAPL")
    assert "SMART_MONEY" not in cats_1

    # 2. "Flow Traders" must not become SMART_MONEY
    cats_2 = derive_canonical_radar_categories("Flow Traders", None, None, None, "XYZ")
    assert "SMART_MONEY" not in cats_2

    # 3. "Minervini Stage 2 VCP" text alone must not become VCP
    cats_3 = derive_canonical_radar_categories("Minervini Stage 2 VCP", None, None, None, "ABC")
    assert "VCP" not in cats_3

    # 4. "VCP Candidate" must not become VCP
    cats_4 = derive_canonical_radar_categories("VCP Candidate", None, None, None, "ABC")
    assert "VCP" not in cats_4

    # 5. "GARP Compounder" text alone with unverified metrics must not become VALUE_GARP
    cats_5 = derive_canonical_radar_categories("Peter Lynch-like", None, None, None, "ZZZ")
    assert "VALUE_GARP" not in cats_5

    # 6. None / Empty must return empty categories
    cats_6 = derive_canonical_radar_categories(None, None, None, None, "NONE")
    assert cats_6 == []


def test_evidence_scoping_no_manufactured_unperformed_evidence():
    """Requirement 2 & 6: Do not manufacture candidate evidence for unperformed evaluations."""
    ev = classify_candidate_category_evidence(["VALUE_GARP"])
    assert ev["VALUE_GARP"] == "CRITERIA_MATCHED"
    # VCP and SMART_MONEY were not evaluated for universe screening, so they MUST NOT appear
    assert "VCP" not in ev
    assert "SMART_MONEY" not in ev


def test_value_garp_candidates_render_from_typed_authority():
    """Requirement 9: VALUE_GARP candidates continue to render from typed authority."""
    resp = Response()
    data = run_screener_get(resp, filter_type="lynch")
    candidates = data.get("candidates", [])
    assert len(candidates) > 0
    for c in candidates:
        assert "VALUE_GARP" in c["categories"]
        assert c["categoryEvidence"]["VALUE_GARP"] == "CRITERIA_MATCHED"

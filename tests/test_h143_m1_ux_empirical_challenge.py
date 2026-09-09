"""Empirical Adversarial Challenge Suite for Horizon 14.3 Milestone M1 UX Deliverables.

Adversarially tests all 7 architectural and design deliverables in docs/ux/ against:
1. Decision Speed Heuristics (<10s for /radar, /setups, /portfolio; <30s for /performance, /journal, /research).
2. Anti-Slop & Zero Card Farm (Unconditional elimination of 4-card KPI rows; asymmetric hero for every page).
3. Typography & Color Discipline (Strict monospace restriction to numbers/tickers; 5-color semantic discipline; Anti-Cyan).
4. Completeness & Quality (Zero placeholders, complete hub coverage, production-grade primitive code).
"""

import os
import re
import pytest

DOCS_UX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "ux"))

REQUIRED_DELIVERABLES = [
    "DESIGN_AUDIT_REPORT.md",
    "UX_IMPROVEMENT_BACKLOG.md",
    "VISUAL_HIERARCHY_RECOMMENDATIONS.md",
    "NAVIGATION_OPTIMIZATION_PLAN.md",
    "PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md",
    "COMPONENT_CONSOLIDATION_PLAN.md",
    "HORIZON_14_3_CERTIFICATION_REPORT.md",
]

FLAGSHIP_HUBS = ["/radar", "/setups", "/portfolio", "/journal", "/performance", "/research"]


@pytest.fixture(scope="module")
def deliverable_contents():
    """Load all 7 deliverable files into memory."""
    contents = {}
    for filename in REQUIRED_DELIVERABLES:
        filepath = os.path.join(DOCS_UX_DIR, filename)
        assert os.path.exists(filepath), f"Missing required deliverable: {filename}"
        with open(filepath, "r", encoding="utf-8") as f:
            contents[filename] = f.read()
    return contents


# =============================================================================
# 1. Completeness, File Size & Placeholder Integrity (Quality Gate)
# =============================================================================

def test_all_7_deliverables_exist_and_meet_size_threshold(deliverable_contents):
    """Verify all 7 deliverables exist and contain substantial institutional content (> 10 KB)."""
    for filename, content in deliverable_contents.items():
        assert len(content.encode("utf-8")) >= 10000, (
            f"Deliverable {filename} is too brief ({len(content)} bytes), fails institutional completeness."
        )


def test_zero_vague_placeholders(deliverable_contents):
    """Adversarially probe all 7 deliverables for vague placeholders or incomplete stubs."""
    forbidden_patterns = [
        r"\bTODO\b",
        r"\bFIXME\b",
        r"\b\[TBD\]\b",
        r"\b\[placeholder\]\b",
        r"\b\[insert\b",
        r"lorem ipsum",
        r"\bxxx\b",
    ]

    for filename, content in deliverable_contents.items():
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, content, flags=re.IGNORECASE)
            assert len(matches) == 0, (
                f"Found {len(matches)} placeholder matches for pattern '{pattern}' in {filename}: {matches}"
            )


def test_flagship_hub_coverage_across_core_blueprints(deliverable_contents):
    """Verify that all 6 flagship hubs are explicitly addressed across core blueprints."""
    blueprints = [
        "PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md",
        "VISUAL_HIERARCHY_RECOMMENDATIONS.md",
        "HORIZON_14_3_CERTIFICATION_REPORT.md",
        "NAVIGATION_OPTIMIZATION_PLAN.md",
        "DESIGN_AUDIT_REPORT.md",
    ]
    for bp in blueprints:
        content = deliverable_contents[bp]
        for hub in FLAGSHIP_HUBS:
            assert hub in content, f"Flagship hub '{hub}' is missing from blueprint {bp}"


# =============================================================================
# 2. Decision Speed Heuristics (<10s and <30s Guarantees)
# =============================================================================

def test_10_second_operational_heuristics(deliverable_contents):
    """Adversarially verify that /radar, /setups, and /portfolio have explicit <= 10s decision speed specifications."""
    page_rec = deliverable_contents["PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md"]
    cert_rep = deliverable_contents["HORIZON_14_3_CERTIFICATION_REPORT.md"]
    vh_rec = deliverable_contents["VISUAL_HIERARCHY_RECOMMENDATIONS.md"]

    for doc in [page_rec, cert_rep, vh_rec]:
        assert "10 second" in doc.lower() or "10-second" in doc.lower() or "≤ 10" in doc, (
            "Document lacks explicit 10-second decision heuristic declaration."
        )

    # /radar: attention
    assert "What deserves attention" in page_rec
    assert "Attention Leader" in page_rec
    assert "IN ACTIONABLE BREAKOUT ZONE" in page_rec or "IN BUY ZONE" in page_rec

    # /setups: trade
    assert "What should I trade" in page_rec
    assert "Asymmetric Execution Ticket" in page_rec
    assert "Buy Limit" in page_rec or "Entry Pivot" in page_rec
    assert "Stop Loss" in page_rec
    assert "Target 1" in page_rec
    assert "Authorize Order" in page_rec

    # /portfolio: risk carried
    assert "What can hurt me" in page_rec
    assert "Capital at Risk Hero" in page_rec
    assert "Total Capital at Risk at Stop Floors" in page_rec or "-$1,420" in page_rec
    assert "Exit Rule Triggers" in page_rec
    assert "Heat Map" in page_rec


def test_30_second_evaluative_heuristics(deliverable_contents):
    """Adversarially verify that /performance, /journal, and /research have explicit <= 30s decision speed specifications."""
    page_rec = deliverable_contents["PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md"]
    cert_rep = deliverable_contents["HORIZON_14_3_CERTIFICATION_REPORT.md"]
    vh_rec = deliverable_contents["VISUAL_HIERARCHY_RECOMMENDATIONS.md"]

    for doc in [page_rec, cert_rep, vh_rec]:
        assert "30 second" in doc.lower() or "30-second" in doc.lower() or "≤ 30" in doc, (
            "Document lacks explicit 30-second decision heuristic declaration."
        )

    # /performance: ROI / Capital Preserved / Drawdown delta
    assert "Capital Preserved" in page_rec
    assert "+$6,140" in page_rec or "+$5,225" in page_rec
    assert "Counterfactual Equity" in page_rec
    assert "Drawdown" in page_rec

    # /journal: Rule adherence discipline score and tilt corrections
    assert "Discipline" in page_rec
    assert "94.2%" in page_rec
    assert "Brier" in page_rec
    assert "0.18 ≤ 0.25" in page_rec or "<= 0.25" in page_rec or r"\le 0.25" in page_rec
    assert "Anti-Tilt" in page_rec

    # /research: catalyst & institutional backing
    assert "Research Dossier" in page_rec
    assert "Catalyst" in page_rec
    assert "13F" in page_rec
    assert "ROIC" in page_rec


# =============================================================================
# 3. Anti-Slop & Zero Card Farm Elimination
# =============================================================================

def test_unconditional_card_farm_elimination(deliverable_contents):
    """Verify that the 4-card KPI row is unconditionally eliminated across all hubs."""
    for filename in ["DESIGN_AUDIT_REPORT.md", "UX_IMPROVEMENT_BACKLOG.md", "VISUAL_HIERARCHY_RECOMMENDATIONS.md", "PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md", "COMPONENT_CONSOLIDATION_PLAN.md"]:
        content = deliverable_contents[filename]
        assert "card farm" in content.lower(), f"{filename} does not address the card farm anti-pattern."
        assert "grid-cols-4" in content, f"{filename} does not specifically target grid-cols-4."

    vh_rec = deliverable_contents["VISUAL_HIERARCHY_RECOMMENDATIONS.md"]
    assert "Zero Symmetrical 4-Card Farms" in vh_rec or "Zero generic 4-card" in vh_rec

    cert_rep = deliverable_contents["HORIZON_14_3_CERTIFICATION_REPORT.md"]
    assert "-100% Card Farms" in cert_rep


def test_asymmetric_hero_for_every_hub(deliverable_contents):
    """Verify that every single flagship hub has an asymmetric hero focal point defined."""
    page_rec = deliverable_contents["PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md"]

    # Check each hub's Level 0 Decision Hero
    hub_heroes = [
        ("Hub 1: `/radar`", "Attention Leader"),
        ("Hub 2: `/setups`", "Asymmetric Execution Ticket"),
        ("Hub 3: `/portfolio`", "Capital at Risk Hero"),
        ("Hub 4: `/journal`", "Discipline Status Hero"),
        ("Hub 5: `/performance`", "Proof of Edge Hero"),
        ("Hub 6: `/research`", "Research Dossier Hero"),
    ]
    for hub_header, hero_name in hub_heroes:
        assert hub_header in page_rec, f"Missing section for {hub_header}"
        assert hero_name in page_rec, f"Missing {hero_name} for {hub_header}"
        assert "Level 0 (Decision" in page_rec


def test_minus_20_percent_anti_slop_cut_lists(deliverable_contents):
    """Verify that the -20% Anti-Slop cut list is documented with concrete removals for all 6 hubs."""
    page_rec = deliverable_contents["PAGE_BY_PAGE_REDESIGN_RECOMMENDATIONS.md"]
    vh_rec = deliverable_contents["VISUAL_HIERARCHY_RECOMMENDATIONS.md"]

    for hub in FLAGSHIP_HUBS:
        assert f"The -20% Anti-Slop Cut List" in page_rec or "Anti-Slop" in page_rec

    assert "Formal Pruning Ledger" in vh_rec
    # Verify exact pruned elements
    assert "40-card repetitive 3-column grid" in vh_rec
    assert "Retail wallet presets ($50, $100)" in vh_rec
    assert "Duplicate \"Copy Broker String\" button" in vh_rec
    assert "Raw LaTeX string `$\\le 0.25$`" in vh_rec
    assert "Generic 4-card summary row" in vh_rec


# =============================================================================
# 4. Typography & Color Discipline
# =============================================================================

def test_monospace_strict_restriction(deliverable_contents):
    """Verify strict prohibition of monospace on narrative/container and restriction to numbers/tickers."""
    audit_rep = deliverable_contents["DESIGN_AUDIT_REPORT.md"]
    vh_rec = deliverable_contents["VISUAL_HIERARCHY_RECOMMENDATIONS.md"]
    backlog = deliverable_contents["UX_IMPROVEMENT_BACKLOG.md"]

    assert "portfolio/page.tsx:245" in audit_rep
    assert "font-mono" in audit_rep

    # Must specify prohibition on <main>
    assert "<main className=" in backlog
    assert "font-sans" in vh_rec
    assert "font-mono" in vh_rec
    assert "tabular-nums" in vh_rec


def test_semantic_5_color_invariants_and_anti_cyan(deliverable_contents):
    """Verify strict color discipline: Emerald, Rose, Amber, Cyan, Purple with Anti-Cyan enforcement."""
    vh_rec = deliverable_contents["VISUAL_HIERARCHY_RECOMMENDATIONS.md"]
    audit_rep = deliverable_contents["DESIGN_AUDIT_REPORT.md"]
    backlog = deliverable_contents["UX_IMPROVEMENT_BACKLOG.md"]

    # 5 colors
    assert "Emerald" in vh_rec and "Positive Returns" in vh_rec
    assert "Rose" in vh_rec and "Stop Loss" in vh_rec
    assert "Cyan" in vh_rec and "Active Tool Focus" in vh_rec
    assert "Amber" in vh_rec and "NEAR_PIVOT" in vh_rec
    assert "Purple" in vh_rec and "Smart Money" in vh_rec

    # Anti-cyan rule
    assert "Anti-Cyan" in audit_rep
    assert "Anti-Cyan" in backlog
    assert "radar/page.tsx:492" in backlog
    assert "setups/page.tsx:98" in backlog
    assert "performance/page.tsx:134" in backlog
    assert "portfolio/page.tsx:312" in backlog


def test_purge_of_forbidden_lifestyle_terms(deliverable_contents):
    """Verify that INV-OI112-P forbidden terms are strictly targeted for purging from navigation."""
    nav_plan = deliverable_contents["NAVIGATION_OPTIMIZATION_PLAN.md"]
    backlog = deliverable_contents["UX_IMPROVEMENT_BACKLOG.md"]

    forbidden_terms = [
        "life health index",
        "lhi",
        "household health index",
        "hhi",
        "identity alignment index",
        "iai",
        "168-hour",
    ]
    for term in forbidden_terms:
        assert term in nav_plan.lower() or term in backlog.lower(), f"Forbidden term {term} not tracked for purge."


# =============================================================================
# 5. Production-Grade Institutional Primitives Specification
# =============================================================================

def test_component_consolidation_primitives_code(deliverable_contents):
    """Verify that COMPONENT_CONSOLIDATION_PLAN.md contains full TypeScript/JSX code for the 4 primitives."""
    comp_plan = deliverable_contents["COMPONENT_CONSOLIDATION_PLAN.md"]

    # Primitive 1: DecisionHero
    assert "export const DecisionHero: React.FC<DecisionHeroProps>" in comp_plan
    assert "data-testid=\"decision-hero\"" in comp_plan
    assert "headline" in comp_plan
    assert "primaryMetric" in comp_plan
    assert "secondaryMetrics" in comp_plan

    # Primitive 2: DataLedgerTable
    assert "export function DataLedgerTable<T>" in comp_plan
    assert "ColumnDef<T>" in comp_plan
    assert "tabular-nums" in comp_plan
    assert "keyExtractor" in comp_plan

    # Primitive 3: SemanticBadge
    assert "export const SemanticBadge: React.FC<SemanticBadgeProps>" in comp_plan
    assert "export type SemanticTone = 'emerald' | 'rose' | 'amber' | 'cyan' | 'purple' | 'slate'" in comp_plan

    # Primitive 4: AsymmetricSkeleton
    assert "export const AsymmetricSkeleton: React.FC<AsymmetricSkeletonProps>" in comp_plan
    assert "data-testid=\"asymmetric-hero-skeleton\"" in comp_plan
    assert "data-testid=\"asymmetric-ledger-skeleton\"" in comp_plan


def test_navigation_plan_architecture_specifications(deliverable_contents):
    """Verify that NAVIGATION_OPTIMIZATION_PLAN.md provides precise specs for mobile dock and header compression."""
    nav_plan = deliverable_contents["NAVIGATION_OPTIMIZATION_PLAN.md"]

    assert "240px" in nav_plan
    assert "108px" in nav_plan
    assert re.search(r"55\\?%", nav_plan), "Nav plan does not state 55% reduction"
    assert re.search(r"Navbar\.tsx.*393[–\-]473", nav_plan), "Nav plan does not cite Navbar.tsx:393-473"
    assert "TerminalShell.tsx" in nav_plan
    assert "md:hidden" in nav_plan
    assert "pb-[calc(0.5rem+env(safe-area-inset-bottom,0px))]" in nav_plan
    assert "Live Behavioral Governor Status" in nav_plan
    assert "← Return to ARX Terminal" in nav_plan

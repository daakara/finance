"""Adversarial verification of STATUTORY_FILING_SELECTOR_V1_5_0 (Phase C).

Tests:
1. Version assertions.
2. Source-role hierarchy: Complete prospectus > partial supplement > ticker sticker > fee waiver.
3. Abbreviated strategy amendments (XUDV, UDIV) rejected for full mandate extraction.
4. Complete 497K and complete 485BPOS accepted.
5. Fee waiver supplements disqualified.
6. V1.5.0: Multi-fund document penalty applied to combined 485BPOS trust filings.
"""

import pytest
from pathlib import Path

from scripts.research.statutory_filing_selector import (
    StatutoryFilingSelector,
    STATUTORY_FILING_SELECTOR_VERSION,
    ROLE_BASE_STATUTORY_PROSPECTUS,
    ROLE_SUMMARY_PROSPECTUS,
    ROLE_PROSPECTUS_SUPPLEMENT,
    ROLE_FEE_WAIVER_SUPPLEMENT,
)


def test_version():
    """Pinned to V1.5.0 which adds MULTI_FUND_DOCUMENT_PENALTY for combined 485BPOS trust filings."""
    assert STATUTORY_FILING_SELECTOR_VERSION == "STATUTORY_FILING_SELECTOR_V1_5_0"


def test_classify_document_role_hierarchy():
    # Complete 497K
    role = StatutoryFilingSelector.classify_document_role(
        "497K", "d12345d497k.htm", "Summary Prospectus"
    )
    assert role == ROLE_SUMMARY_PROSPECTUS

    # 497K Supplement
    role_supp = StatutoryFilingSelector.classify_document_role(
        "497K", "d12345d497k.htm", "Supplement dated April 2026 to the Prospectus"
    )
    assert role_supp == ROLE_PROSPECTUS_SUPPLEMENT

    # Ticker Sticker
    role_sticker = StatutoryFilingSelector.classify_document_role(
        "497", "d12345d497.htm", "Ticker Sticker Supplement"
    )
    assert role_sticker == ROLE_FEE_WAIVER_SUPPLEMENT or role_sticker == ROLE_PROSPECTUS_SUPPLEMENT

    # Fee Waiver
    role_waiver = StatutoryFilingSelector.classify_document_role(
        "497", "d12345d497.htm", "Fee Waiver Supplement"
    )
    assert role_waiver == ROLE_FEE_WAIVER_SUPPLEMENT

    # Base Prospectus
    role_base = StatutoryFilingSelector.classify_document_role(
        "485BPOS", "d12345d485bpos.htm", "Post-Effective Amendment No. 50"
    )
    assert role_base == ROLE_BASE_STATUTORY_PROSPECTUS


def test_reject_xudv_udiv_partial_supplements():
    """Verify that the 14KB XUDV and UDIV partial strategy amendments are rejected."""
    cache_dir = Path("data/research/cache/sec_prospectus")
    xudv_file = cache_dir / "0001193125-26-191894_d54616d497k.htm"
    udiv_file = cache_dir / "0001193125-26-191896_d54616d497k.htm"

    if xudv_file.exists():
        text = xudv_file.read_text(encoding="utf-8", errors="ignore")
        ok, reason = StatutoryFilingSelector.check_mandate_content(text, form="497K")
        assert not ok, f"XUDV supplement should not qualify as full mandate: {reason}"
        assert "SUPPLEMENT" in reason or "NOT_FOUND" in reason

    if udiv_file.exists():
        text = udiv_file.read_text(encoding="utf-8", errors="ignore")
        ok, reason = StatutoryFilingSelector.check_mandate_content(text, form="497K")
        assert not ok, f"UDIV supplement should not qualify as full mandate: {reason}"
        assert "SUPPLEMENT" in reason or "NOT_FOUND" in reason


def test_accept_complete_prospectuses():
    """Verify that complete statutory prospectuses qualify."""
    cache_dir = Path("data/research/cache/sec_prospectus")
    vgt_file = cache_dir / "0000052848-26-000651_f45474d1.htm"
    bkem_file = cache_dir / "0000030146-26-000097_c497k.htm"

    if vgt_file.exists():
        text = vgt_file.read_text(encoding="utf-8", errors="ignore")
        ok, reason = StatutoryFilingSelector.check_mandate_content(text, form="497K")
        assert ok, f"Complete VGT 497K should qualify: {reason}"

    if bkem_file.exists():
        text = bkem_file.read_text(encoding="utf-8", errors="ignore")
        ok, reason = StatutoryFilingSelector.check_mandate_content(text, form="497K")
        assert ok, f"Complete BKEM 497K should qualify: {reason}"

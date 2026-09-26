"""Comprehensive Unit, Integration, Adversarial, and Temporal Tests for StatutoryFilingSelector.

Covers:
1. Role classification (BASE_STATUTORY_PROSPECTUS, SUMMARY_PROSPECTUS, FEE_WAIVER_SUPPLEMENT, SAI_PART_B).
2. Strict temporal boundary gating (POST_BOUNDARY_FILING_SELECTED = 0).
3. Candidate prioritization & multi-accession search.
4. Target series presence check & rejection of compensation/SAI-only occurrences.
5. Mandate content check (Item 4, StrategyNarrativeTextBlock, RiskReturnHeading).
6. Adversarial cases:
   - Latest filing belongs to another series (rejection and multi-accession search continuation)
   - 497 fee waiver supplement rejection
   - SAI Part B containing series in trustee table rejection
   - Post-boundary filing with better match strictly rejected
   - Multi-fund omnibus selection
7. Cache key determinism & audit trail provenance.
8. End-to-end integration with DocumentIndex and SeriesProspectusMapper.
"""

import pytest
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

from scripts.research.statutory_filing_selector import (
    StatutoryFilingSelector,
    FilingCandidate,
    FilingSelectionResult,
    STATUTORY_FILING_SELECTOR_VERSION,
    SNAPSHOT_BOUNDARY,
    ROLE_BASE_STATUTORY_PROSPECTUS,
    ROLE_SUMMARY_PROSPECTUS,
    ROLE_PROSPECTUS_SUPPLEMENT,
    ROLE_FEE_WAIVER_SUPPLEMENT,
    ROLE_SAI_PART_B,
    OUTCOME_SELECTED_STATUTORY_PROSPECTUS,
    OUTCOME_SELECTED_SUMMARY_PROSPECTUS,
    OUTCOME_NO_PREBOUNDARY_CANDIDATE,
    OUTCOME_TARGET_ABSENT_FROM_ALL,
    OUTCOME_SOURCE_CACHE_MISS,
)
from scripts.research.series_prospectus_mapper import SeriesMetadata, SeriesProspectusMapper
from scripts.research.document_index_engine import DocumentIndex, DocumentIdentity


class TestStatutoryFilingSelectorUnit:
    """Unit tests for role classification, presence check, and cache keys."""

    def test_role_classification_497k_summary_prospectus(self):
        role = StatutoryFilingSelector.classify_document_role(
            form="497K",
            primary_document="ampliusaggressiveallocatio.htm",
            primary_doc_description="497K"
        )
        assert role == ROLE_SUMMARY_PROSPECTUS

    def test_role_classification_base_485bpos(self):
        role = StatutoryFilingSelector.classify_document_role(
            form="485BPOS",
            primary_document="ck0001592900-20260924.htm",
            primary_doc_description="485BPOS"
        )
        assert role == ROLE_BASE_STATUTORY_PROSPECTUS

    def test_role_classification_fee_waiver_supplement(self):
        role = StatutoryFilingSelector.classify_document_role(
            form="497",
            primary_document="glx-20260916.htm",
            primary_doc_description="497 - IRVH RE. LIQUIDATION"
        )
        assert role == ROLE_FEE_WAIVER_SUPPLEMENT

        role2 = StatutoryFilingSelector.classify_document_role(
            form="497",
            primary_document="aaaa-feewaiversticker102025.htm",
            primary_doc_description="497"
        )
        assert role2 == ROLE_FEE_WAIVER_SUPPLEMENT

    def test_role_classification_sai_part_b(self):
        role = StatutoryFilingSelector.classify_document_role(
            form="497",
            primary_document="octcombinedsai.htm",
            primary_doc_description="497E CONSOLIDATED SAI"
        )
        assert role == ROLE_SAI_PART_B

        role2 = StatutoryFilingSelector.classify_document_role(
            form="485BPOS",
            primary_document="strivestxtbuxx-sai.htm",
            primary_doc_description="STATEMENT OF ADDITIONAL INFORMATION"
        )
        assert role2 == ROLE_SAI_PART_B

    def test_check_target_presence_valid(self):
        target = SeriesMetadata(
            symbol="TEST",
            cik="1234567",
            series_id="S000011111",
            class_id="C000022222",
            legal_name="Alpha Beta Quant ETF"
        )
        html_doc = """
        <html><body>
          <h2>Alpha Beta Quant ETF</h2>
          <p>Series ID: S000011111 Class ID: C000022222</p>
          <h3>Principal Investment Strategies</h3>
          <p>Invests in quant equities.</p>
        </body></html>
        """
        present, reason = StatutoryFilingSelector.check_target_presence(target, html_doc)
        assert present is True
        assert reason == "TARGET_PRESENT"

    def test_check_target_presence_absent(self):
        target = SeriesMetadata(
            symbol="MISS",
            cik="1234567",
            series_id="S000099999",
            class_id="C000099999",
            legal_name="Missing Target Fund ETF"
        )
        html_doc = "<html><body><h2>Other Fund ETF</h2></body></html>"
        present, reason = StatutoryFilingSelector.check_target_presence(target, html_doc)
        assert present is False
        assert reason == "TARGET_NOT_FOUND_IN_TEXT"

    def test_check_target_presence_sai_only_rejected(self):
        target = SeriesMetadata(
            symbol="TRUSTEE",
            cik="1234567",
            series_id="S000088888",
            class_id="C000088888",
            legal_name="Trustee Mention Only ETF"
        )
        html_doc = ("A" * 6000) + """
        <h2>Statement of Additional Information</h2>
        <table>
          <tr><td>Trustee Compensation Table</td><td>Trustee Mention Only ETF (S000088888)</td></tr>
        </table>
        """
        present, reason = StatutoryFilingSelector.check_target_presence(target, html_doc)
        assert present is False
        assert reason == "TARGET_ONLY_IN_SAI_SECTION"

    def test_check_mandate_content_detection(self):
        doc1 = "<p>Principal Investment Strategies: The fund invests 80%...</p>"
        assert StatutoryFilingSelector.check_mandate_content(doc1)[0] is True

        doc2 = '<ix:nonNumeric name="oef:StrategyNarrativeTextBlock">Strategy</ix:nonNumeric>'
        assert StatutoryFilingSelector.check_mandate_content(doc2)[0] is True

        doc3 = "<html><body>Financial statements and balance sheet.</body></html>"
        assert StatutoryFilingSelector.check_mandate_content(doc3)[0] is False

    def test_cache_key_determinism_and_isolation(self):
        target1 = SeriesMetadata(symbol="A", cik="1", series_id="S1", class_id="C1", legal_name="Fund A")
        target2 = SeriesMetadata(symbol="B", cik="1", series_id="S2", class_id="C2", legal_name="Fund B")

        k1 = StatutoryFilingSelector.compute_cache_key(target1, "1")
        k1_repeat = StatutoryFilingSelector.compute_cache_key(target1, "1")
        k2 = StatutoryFilingSelector.compute_cache_key(target2, "1")

        assert k1 == k1_repeat
        assert k1 != k2
        assert len(k1) == 64


class TestStatutoryFilingSelectorAdversarialAndTemporal:
    """Adversarial and boundary tests enforcing fail-closed invariant guarantees."""

    def test_post_boundary_filings_strictly_rejected(self, tmp_path):
        """Temporal test: filings with filingDate > 2026-09-24 must NEVER be selected."""
        target = SeriesMetadata(
            symbol="FUTUR",
            cik="1234567",
            series_id="S000077777",
            class_id="C000077777",
            legal_name="Future Innovation ETF"
        )
        sub_json = {
            "filings": {
                "recent": {
                    "form": ["485BPOS", "485BPOS"],
                    "filingDate": ["2026-09-28", "2026-09-25"],  # Both strictly post-boundary
                    "accessionNumber": ["0001234567-26-000999", "0001234567-26-000998"],
                    "primaryDocument": ["future_doc2.htm", "future_doc1.htm"],
                    "primaryDocDescription": ["485BPOS", "485BPOS"]
                }
            }
        }
        res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_NO_PREBOUNDARY_CANDIDATE
        assert res.selected_accession == "NONE"
        assert len(res.rejected_candidates) == 2
        for r in res.rejected_candidates:
            assert "POST_BOUNDARY_FILING" in r["rejection_reason"]

    def test_latest_filing_bias_rejection_and_multi_accession_search(self, tmp_path):
        """Adversarial test: Newest filing belongs to Series B; Selector must bypass it and find Series A in older accession."""
        target_a = SeriesMetadata(
            symbol="TFNDA",
            cik="9999999",
            series_id="S000012345",
            class_id="C000012345",
            legal_name="Alpha Momentum ETF"
        )

        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True, exist_ok=True)

        # Accession 2 (newest, 2026-09-23): contains ONLY Series B
        doc2_text = """
        <html><body>
          <h2>Beta Dividend ETF</h2>
          <p>Series ID: S000099999 Class ID: C000099999</p>
          <h3>Principal Investment Strategies</h3>
          <p>Invests in dividend payers.</p>
        </body></html>
        """
        (prospectus_dir / "0009999999-26-000002_newest_series_b.htm").write_text(doc2_text, encoding="utf-8")

        # Accession 1 (older, 2026-06-15): contains Series A!
        doc1_text = """
        <html><body>
          <h2>Alpha Momentum ETF</h2>
          <p>Series ID: S000012345 Class ID: C000012345</p>
          <h3>Principal Investment Strategies</h3>
          <p>Invests in momentum equities.</p>
        </body></html>
        """
        (prospectus_dir / "0009999999-26-000001_older_series_a.htm").write_text(doc1_text, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["485BPOS", "485BPOS"],
                    "filingDate": ["2026-09-23", "2026-06-15"],
                    "accessionNumber": ["0009999999-26-000002", "0009999999-26-000001"],
                    "primaryDocument": ["newest_series_b.htm", "older_series_a.htm"],
                    "primaryDocDescription": ["485BPOS", "485BPOS"]
                }
            }
        }

        res = StatutoryFilingSelector.select_statutory_filing(target_a, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_SELECTED_STATUTORY_PROSPECTUS
        assert res.selected_accession == "0009999999-26-000001"
        assert res.document_filename == "older_series_a.htm"

    def test_fee_waiver_supplement_rejected_over_base_prospectus(self, tmp_path):
        """Adversarial test: Form 497 fee waiver supplement must NOT be chosen over base 485BPOS."""
        target = SeriesMetadata(
            symbol="TAREQ",
            cik="8888888",
            series_id="S000055555",
            class_id="C000055555",
            legal_name="Target Quantitative Equity ETF"
        )

        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True, exist_ok=True)

        # 497 fee waiver (newest)
        doc_waiver = """<html><body>Sticker supplement: fee waiver extended.</body></html>"""
        (prospectus_dir / "0008888888-26-000020_glx-waiver.htm").write_text(doc_waiver, encoding="utf-8")

        # 485BPOS base (older)
        doc_base = """
        <html><body>
          <h2>Target Quantitative Equity ETF</h2>
          <p>Series ID: S000055555 Class ID: C000055555</p>
          <h3>Principal Investment Strategies</h3>
          <p>Target quant strategies.</p>
        </body></html>
        """
        (prospectus_dir / "0008888888-26-000010_base_prospectus.htm").write_text(doc_base, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["497", "485BPOS"],
                    "filingDate": ["2026-09-18", "2026-03-31"],
                    "accessionNumber": ["0008888888-26-000020", "0008888888-26-000010"],
                    "primaryDocument": ["glx-waiver.htm", "base_prospectus.htm"],
                    "primaryDocDescription": ["497 - FEE WAIVER STICKER", "485BPOS BASE"]
                }
            }
        }

        res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_SELECTED_STATUTORY_PROSPECTUS
        assert res.selected_accession == "0008888888-26-000010"
        assert res.document_filename == "base_prospectus.htm"


class TestStatutoryFilingSelectorIntegrationWithResolver:
    """Integration test verifying end-to-end pipeline with frozen DocumentIndex and SeriesProspectusMapper."""

    def test_ten_previously_validated_targets_zero_regressions(self):
        """Section 27: Verify all 10 correct-document targets continue to map with 0 regressions."""
        cache_dir = Path("data/research/cache")
        targets_10 = [
            ("GQQQ", "1592900", "S000088111", "C000254131", "Astoria US Quality Growth Kings ETF", "EA Series TRUST"),
            ("ROE",  "1592900", "S000081203", "C000244015", "Astoria US Equal Weight Quality Kings ETF", "EA Series TRUST"),
            ("AGGA", "1592900", "S000091819", "C000259601", "EA Astoria Beacon Dynamic Core US Fixed Income ETF", "EA Series TRUST"),
            ("CGCP", "1870117", "S000074251", "C000231860", "Capital Group Core Plus Income ETF", "Capital Group Fixed Income ETF Trust"),
            ("CGMS", "1870117", "S000077688", "C000238176", "Capital Group Short Duration Municipal Income ETF", "Capital Group Fixed Income ETF Trust"),
            ("CGSD", "1870117", "S000074252", "C000231861", "Capital Group Short Duration Income ETF", "Capital Group Fixed Income ETF Trust"),
            ("CGHY", "1870117", "S000080123", "C000241982", "Capital Group High Yield Bond ETF", "Capital Group Fixed Income ETF Trust"),
            ("CGGG", "2034928", "S000092695", "C000260959", "Capital Group U.S. Large Growth ETF", "Capital Group Active ETF Trust"),
            ("CGMM", "2034928", "S000088874", "C000255476", "Capital Group U.S. Small and Mid Cap ETF", "Capital Group Active ETF Trust"),
            ("CGVV", "2034928", "S000092696", "C000260960", "Capital Group U.S. Large Value ETF", "Capital Group Active ETF Trust"),
        ]

        regressions = 0
        for sym, cik, sid, cid, name, trust in targets_10:
            target = SeriesMetadata(symbol=sym, cik=cik, series_id=sid, class_id=cid, legal_name=name, trust_name=trust)
            sub_path = cache_dir / "sec_submissions" / f"CIK{int(cik):010d}.json"
            assert sub_path.exists(), f"Missing CIK submission JSON for CIK {cik}"
            with open(sub_path, "r", encoding="utf-8") as f:
                sub_json = json.load(f)

            sel_res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, cache_dir)
            assert sel_res.selection_outcome == OUTCOME_SELECTED_STATUTORY_PROSPECTUS

            doc_path = cache_dir / "sec_prospectus" / f"{sel_res.selected_accession}_{sel_res.document_filename}"
            assert doc_path.exists(), f"Selected document {doc_path} not found"
            with open(doc_path, "rb") as f:
                raw_bytes = f.read()

            ident = DocumentIdentity(
                cik=cik,
                accession=sel_res.selected_accession,
                form=sel_res.selected_form,
                filing_date=sel_res.filing_date,
                document_filename=sel_res.document_filename
            )
            doc_index = DocumentIndex(ident, raw_bytes, known_series_metadata=[{"legal_name": name}])
            map_res = SeriesProspectusMapper.map_series(target, doc_index)

            assert map_res.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
            assert len(map_res.extracted_strategy_text) >= 50
            assert map_res.leakage_status == SeriesProspectusMapper.LEAKAGE_CHECKED_CLEAN
            assert map_res.cross_series_text_leakage == 0

        assert regressions == 0


class TestStatutoryFilingSelectorRemediationV1_1_0:
    """Targeted regression and invariant tests for STATUTORY_FILING_SELECTOR_V1_1_0 remediation.

    Covers:
    1. Issuer-name-only false match rejection (e.g. Goldman, Sachs, SPDR tokens alone score 0).
    2. Large multi-fund registrant handling (ensuring base prospectuses are properly prioritized).
    3. Termination of backward walking (candidate bounding preventing runaway crawls).
    """

    def test_issuer_name_only_false_match_rejection(self, tmp_path):
        """Tokens belonging exclusively to COMMON_ISSUER_AND_GENERIC_TOKENS must NOT produce a metadata match."""
        target = SeriesMetadata(
            symbol="GEM",
            cik="1479026",
            series_id="S000050854",
            class_id="C000160276",
            legal_name="Goldman Sachs ActiveBeta Emerging Markets Equity ETF"
        )

        cand = FilingCandidate(
            accession="0001193125-24-123456",
            form="497K",
            filing_date="2024-05-01",
            primary_document="goldman_sachs_other_fund.htm",
            primary_doc_description="Goldman Sachs Physical Gold ETF 497K",
            document_role=ROLE_SUMMARY_PROSPECTUS,
            is_preboundary=True
        )

        match = StatutoryFilingSelector.match_target_metadata(
            target, cand.primary_document, cand.primary_doc_description
        )
        # Even though "Goldman" and "Sachs" are in legal_name and primary_doc_description,
        # they are common issuer tokens and must NOT produce a false match.
        assert match is False

    def test_large_multi_fund_registrant_base_prospectus_precedence(self, tmp_path):
        """In multi-fund trusts, a base Form 485BPOS must be prioritized over newer 497Ks of other series."""
        target = SeriesMetadata(
            symbol="GEM",
            cik="1479026",
            series_id="S000050854",
            class_id="C000160276",
            legal_name="Goldman Sachs ActiveBeta Emerging Markets Equity ETF"
        )

        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True, exist_ok=True)

        # Newer Form 497K for an unrelated Goldman fund
        unrelated_497k = """
        <html><body>
          <h2>Goldman Sachs Semiconductor Innovators ETF</h2>
          <p>Series S000099999 Class C000099999</p>
          <h3>Principal Investment Strategies</h3>
          <p>Invests in semiconductor companies.</p>
        </body></html>
        """
        (prospectus_dir / "0001193125-26-000099_goldman_semi.htm").write_text(unrelated_497k, encoding="utf-8")

        # Older Form 485BPOS covering GEM
        gem_base = """
        <html><body>
          <h2>Goldman Sachs ActiveBeta Emerging Markets Equity ETF</h2>
          <p>Series S000050854 Class C000160276</p>
          <h3>Principal Investment Strategies</h3>
          <p>The Fund seeks long-term capital appreciation by tracking ActiveBeta index.</p>
        </body></html>
        """
        (prospectus_dir / "0001193125-25-334307_d819171d485bpos.htm").write_text(gem_base, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["497K", "485BPOS"],
                    "filingDate": ["2026-09-20", "2025-12-29"],
                    "accessionNumber": ["0001193125-26-000099", "0001193125-25-334307"],
                    "primaryDocument": ["goldman_semi.htm", "d819171d485bpos.htm"],
                    "primaryDocDescription": ["Goldman Sachs Semiconductor Innovators ETF", "485BPOS BASE PROSPECTUS"]
                }
            }
        }

        res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_SELECTED_STATUTORY_PROSPECTUS
        assert res.selected_accession == "0001193125-25-334307"
        assert res.document_filename == "d819171d485bpos.htm"

    def test_termination_of_backward_walking_without_cache_miss_leak(self, tmp_path):
        """When candidates do not match target metadata and are uncached, they must NOT leak into cache_miss_candidate."""
        target = SeriesMetadata(
            symbol="NONEXIST",
            cik="1479026",
            series_id="S000000001",
            class_id="C000000001",
            legal_name="Nonexistent Hypothetical Trust ETF"
        )

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["497K", "497K", "485BPOS"],
                    "filingDate": ["2026-09-20", "2026-08-15", "2026-01-10"],
                    "accessionNumber": ["0001193125-26-000001", "0001193125-26-000002", "0001193125-26-000003"],
                    "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm"],
                    "primaryDocDescription": ["Goldman Sachs Asset Allocation", "Goldman Sachs Bond Fund", "Goldman Sachs Base"]
                }
            }
        }

        res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
        # Because none of doc1/doc2/doc3 match "Nonexistent Hypothetical Trust ETF",
        # they must not be reported as SOURCE_CACHE_MISS, but TARGET_ABSENT_FROM_ALL.
        assert res.selection_outcome == OUTCOME_TARGET_ABSENT_FROM_ALL

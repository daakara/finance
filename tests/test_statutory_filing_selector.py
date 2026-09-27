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
            assert sel_res.selection_outcome in {OUTCOME_SELECTED_STATUTORY_PROSPECTUS, OUTCOME_SELECTED_SUMMARY_PROSPECTUS}

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


class TestStatutoryFilingSelectorV120Unit:
    """Unit tests for V1.2.0 improvements (Section 26)."""

    def test_historical_sec_submission_loading_and_deduplication(self, tmp_path):
        submissions_dir = tmp_path / "sec_submissions"
        submissions_dir.mkdir(parents=True)

        recent_data = {
            "filings": {
                "recent": {
                    "form": ["497K", "485BPOS"],
                    "filingDate": ["2026-05-01", "2025-10-15"],
                    "accessionNumber": ["0001-26-0001", "0001-25-0002"],
                    "primaryDocument": ["doc1.htm", "doc2.htm"],
                    "primaryDocDescription": ["Doc 1", "Doc 2"]
                },
                "files": [
                    {"name": "CIK0000000001-submissions-001.json", "filingCount": 2}
                ]
            }
        }
        hist_data = {
            "form": ["485BPOS", "497"],
            "filingDate": ["2025-10-15", "2024-03-01"],
            "accessionNumber": ["0001-25-0002", "0001-24-0003"],  # 0001-25-0002 is duplicated
            "primaryDocument": ["doc2.htm", "doc3.htm"],
            "primaryDocDescription": ["Doc 2 Duplicate", "Doc 3 Historical"]
        }
        (submissions_dir / "CIK0000000001-submissions-001.json").write_text(json.dumps(hist_data), encoding="utf-8")

        records, history_sha = StatutoryFilingSelector.load_normalized_submission_history(
            cik="0000000001",
            submission_json=recent_data,
            submissions_dir=submissions_dir
        )

        assert len(records) == 3  # 0001-26-0001, 0001-25-0002, 0001-24-0003
        accs = [r.accession for r in records]
        assert accs == ["0001-26-0001", "0001-25-0002", "0001-24-0003"]
        assert len(history_sha) == 64

    def test_ticker_presence_context_aware(self):
        target = SeriesMetadata(
            symbol="ARMH",
            cik="1499655",
            series_id="S000089658",
            class_id="C000256275",
            legal_name="Arm Holdings PLC ADRhedged"
        )
        text_with_parens = "<html><body>Arm Holdings PLC ADRhedged&#8482; (ARMH) Summary Prospectus</body></html>"
        present, reason = StatutoryFilingSelector.check_target_presence(target, text_with_parens)
        assert present is True

        text_with_label = "<html><body>Ticker: ARMH Exchange: Cboe</body></html>"
        present2, reason2 = StatutoryFilingSelector.check_target_presence(target, text_with_label)
        assert present2 is True

    def test_short_ticker_collision_safety(self):
        # Target with short ticker "IT"
        target_it = SeriesMetadata(
            symbol="IT",
            cik="1234567",
            series_id="S000012345",
            class_id="C000012345",
            legal_name="Gartner Tech ETF"
        )
        # Prose containing ordinary English word "it" or "it is"
        prose_text = "<html><body>It is important to evaluate credit and interest rate risk for all assets.</body></html>"
        present, reason = StatutoryFilingSelector.check_target_presence(target_it, prose_text)
        assert present is False

        # Target with short ticker "AI"
        target_ai = SeriesMetadata(
            symbol="AI",
            cik="1234567",
            series_id="S000054321",
            class_id="C000054321",
            legal_name="C3 AI ETF"
        )
        prose_ai = "<html><body>The fund uses artificial intelligence (ai) techniques in portfolio management.</body></html>"
        # Unless formatted as (AI) or Ticker: AI, must not collide
        present_ai, _ = StatutoryFilingSelector.check_target_presence(target_ai, prose_ai)
        assert present_ai is False

    def test_concatenated_ticker_support(self):
        target_armh = SeriesMetadata(
            symbol="ARMH",
            cik="1499655",
            series_id="S000089658",
            class_id="C000256275",
            legal_name="Arm Holdings PLC ADRhedged"
        )
        target_asmh = SeriesMetadata(
            symbol="ASMH",
            cik="1499655",
            series_id="S000089657",
            class_id="C000256274",
            legal_name="ASML Holding NV ADRhedged"
        )

        match_armh = StatutoryFilingSelector.match_target_metadata(
            target_armh, "precidian-armhasmhandsthhs.htm", "497"
        )
        assert match_armh is True

        match_asmh = StatutoryFilingSelector.match_target_metadata(
            target_asmh, "precidian-armhasmhandsthhs.htm", "497"
        )
        assert match_asmh is True

    def test_monthly_buffer_differentiation(self):
        target_apr = SeriesMetadata(
            symbol="APRP",
            cik="1992104",
            series_id="S000084000",
            class_id="C000248000",
            legal_name="PGIM S&P 500 Buffer 12 ETF - April"
        )
        # Candidate description mentions May buffer, NOT April
        cand_may = "PGIM S&P 500 Buffer 12 ETF - May Annual Update"
        match = StatutoryFilingSelector.match_target_metadata(target_apr, "f44136d1.htm", cand_may)
        assert match is False

        # Candidate description does not mention any month, only generic strategy
        cand_generic = "PGIM S&P 500 Buffer 12 ETF Base Document"
        match2 = StatutoryFilingSelector.match_target_metadata(target_apr, "f44136d1.htm", cand_generic)
        assert match2 is False

    def test_etf_vs_mutual_fund_share_class(self):
        target_etf = SeriesMetadata(
            symbol="VBK",
            cik="36405",
            series_id="S000000001",
            class_id="C000000001",
            legal_name="Vanguard Small-Cap Growth ETF"
        )
        # Candidate description is Admiral Shares (mutual fund share class)
        desc_admiral = "VANGUARD SMALL CAP INDEX FUND SUMMARY PROSPECTUS ADMIRAL SHARES"
        match = StatutoryFilingSelector.match_target_metadata(target_etf, "f12170d1.htm", desc_admiral)
        assert match is False

    def test_html_entity_and_punctuation_normalization(self):
        target = SeriesMetadata(
            symbol="AIQ",
            cik="1432353",
            series_id="S000061326",
            class_id="C000198548",
            legal_name="Global X Artificial Intelligence & Technology ETF"
        )
        html_doc = """
        <html><body>
          <title>Global X</title>
          <div>Ticker&#58; AIQ NASDAQ&#58; AIQ</div>
          <h1>Global X Artificial Intelligence &#38; Technology ETF</h1>
          <h3>Principal Investment Strategies</h3>
          <p>The Fund invests in artificial intelligence companies.</p>
        </body></html>
        """
        present, reason = StatutoryFilingSelector.check_target_presence(target, html_doc)
        assert present is True

        mandate, m_reason = StatutoryFilingSelector.check_mandate_content(html_doc)
        assert mandate is True


class TestStatutoryFilingSelectorV120Adversarial:
    """Adversarial tests for fail-closed edge cases (Section 27)."""

    def test_two_unrelated_funds_sharing_buffer_tokens(self, tmp_path):
        """Two funds from same issuer sharing 'S&P 500 Buffer' must not cross-match."""
        target_aug = SeriesMetadata(
            symbol="AUGP",
            cik="1992104",
            series_id="S000084008",
            class_id="C000248008",
            legal_name="PGIM S&P 500 Buffer 12 ETF - August"
        )
        # Cached document belongs to January buffer fund
        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True)
        jan_doc = """
        <html><body>
          <h1>PGIM S&P 500 Buffer 12 ETF - January (JANP)</h1>
          <h3>Principal Investment Strategies</h3>
          <p>Seeks to provide buffer protection for January cycle.</p>
        </body></html>
        """
        (prospectus_dir / "0001193125-26-000001_jan_buffer.htm").write_text(jan_doc, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["497K"],
                    "filingDate": ["2026-01-15"],
                    "accessionNumber": ["0001193125-26-000001"],
                    "primaryDocument": ["jan_buffer.htm"],
                    "primaryDocDescription": ["PGIM S&P 500 Buffer 12 ETF - January"]
                }
            }
        }
        res = StatutoryFilingSelector.select_statutory_filing(target_aug, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_TARGET_ABSENT_FROM_ALL

    def test_correct_target_only_in_historical_submissions_file(self, tmp_path):
        """Target whose only filing is in a historical submission JSON must be successfully selected."""
        submissions_dir = tmp_path / "sec_submissions"
        prospectus_dir = tmp_path / "sec_prospectus"
        submissions_dir.mkdir(parents=True)
        prospectus_dir.mkdir(parents=True)

        target = SeriesMetadata(
            symbol="HISTETF",
            cik="9999999",
            series_id="S000099999",
            class_id="C000099999",
            legal_name="Historical Pioneer Growth ETF"
        )

        hist_doc = """
        <html><body>
          <h1>Historical Pioneer Growth ETF (HISTETF)</h1>
          <p>Series S000099999 Class C000099999</p>
          <h3>Principal Investment Strategies</h3>
          <p>Seeks long term capital growth.</p>
        </body></html>
        """
        (prospectus_dir / "0000999999-23-000001_hist_growth.htm").write_text(hist_doc, encoding="utf-8")

        recent_json = {
            "filings": {
                "recent": {
                    "form": ["497K"],
                    "filingDate": ["2026-08-01"],
                    "accessionNumber": ["0000999999-26-000001"],
                    "primaryDocument": ["recent_other.htm"],
                    "primaryDocDescription": ["Other Modern ETF"]
                },
                "files": [
                    {"name": "CIK0009999999-submissions-001.json", "filingCount": 1}
                ]
            }
        }
        hist_json = {
            "form": ["497K"],
            "filingDate": ["2023-05-15"],
            "accessionNumber": ["0000999999-23-000001"],
            "primaryDocument": ["hist_growth.htm"],
            "primaryDocDescription": ["Historical Pioneer Growth ETF Summary Prospectus"]
        }
        (submissions_dir / "CIK0009999999-submissions-001.json").write_text(json.dumps(hist_json), encoding="utf-8")

        res = StatutoryFilingSelector.select_statutory_filing(target, recent_json, tmp_path)
        assert res.selection_outcome == OUTCOME_SELECTED_SUMMARY_PROSPECTUS
        assert res.selected_accession == "0000999999-23-000001"
        assert res.document_filename == "hist_growth.htm"


class TestStatutoryFilingSelectorRemediationV1_3_0:
    """Targeted regression and invariant tests for STATUTORY_FILING_SELECTOR_V1_3_0 remediation.

    Covers:
    1. Part C / Ancillary Boundary Rejection (TARGET_ONLY_IN_PART_C_OR_ANCILLARY).
    2. Short Form 497 Fee Waiver / Sticker Rejection (SUPPLEMENT_LACKS_SUBSTANTIVE_STRATEGY).
    3. Authoritative Series Directory Priority Scoring (+2000 points).
    4. Ancient Filing Penalty (< 2010 filings penalized -1000 when modern filings exist).
    5. Candidate Selection Trace Logging in FilingSelectionResult.
    """

    def test_part_c_only_occurrence_rejected(self, tmp_path):
        """When target series only appears in Part C / Item 28 exhibits, candidate must be rejected."""
        target = SeriesMetadata(
            symbol="ECOW",
            cik="0001616668",
            series_id="S000064827",
            class_id="C000210214",
            legal_name="Pacer Emerging Markets Cash Cows 100 ETF"
        )

        doc_with_part_c = """
        <html><body>
          <h1>Pacer Trendpilot US Large Cap ETF</h1>
          <p>Series S000012345 Class C000012345</p>
          <h3>Principal Investment Strategies</h3>
          <p>Invests in large cap equity indices with trend following rules.</p>
          <div style="height: 6000px;">... large body ...</div>
          <h2>PART C - OTHER INFORMATION</h2>
          <h3>Item 28. Exhibits</h3>
          <p>Opinion and Consent of Counsel for Pacer Emerging Markets Cash Cows 100 ETF (S000064827) is incorporated herein by reference.</p>
        </body></html>
        """
        is_pres, reason = StatutoryFilingSelector.check_target_presence(target, doc_with_part_c, form="485BPOS")
        assert is_pres is False
        assert reason == "TARGET_ONLY_IN_PART_C_OR_ANCILLARY"

    def test_short_497_supplement_without_strategy_rejected(self):
        """Form 497 supplement under 30k chars lacking substantive strategy sections must be rejected."""
        target = SeriesMetadata(
            symbol="TEST",
            cik="0001234567",
            series_id="S000011111",
            class_id="C000022222",
            legal_name="Test Alpha Growth ETF"
        )
        fee_waiver_text = """
        <html><body>
          <h2>Test Alpha Growth ETF (TEST)</h2>
          <p>Series S000011111 Class C000022222</p>
          <h3>Notice of Fee Waiver Extension</h3>
          <p>Effective October 1, 2025, the adviser has agreed to waive 2 bps of management fees through September 30, 2026.</p>
        </body></html>
        """
        is_pres, reason = StatutoryFilingSelector.check_target_presence(target, fee_waiver_text, form="497")
        assert is_pres is False
        assert reason == "SUPPLEMENT_LACKS_SUBSTANTIVE_STRATEGY"

    def test_authoritative_series_directory_priority(self, tmp_path):
        """Authoritative series directory match receives +2000 points and is selected first."""
        target = SeriesMetadata(
            symbol="FAV",
            cik="0001234567",
            series_id="S000077777",
            class_id="C000088888",
            legal_name="Favorite Authoritative ETF"
        )
        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True)

        # Candidate 1: Generic base prospectus
        cand1_doc = """
        <html><body>
          <h1>Favorite Authoritative ETF (FAV)</h1>
          <p>Series S000077777 Class C000088888</p>
          <h3>Principal Investment Strategies</h3>
          <p>Strategy in base prospectus.</p>
        </body></html>
        """
        (prospectus_dir / "0001234567-25-000001_base.htm").write_text(cand1_doc, encoding="utf-8")

        # Candidate 2: Authoritative 497K
        cand2_doc = """
        <html><body>
          <h1>Favorite Authoritative ETF (FAV)</h1>
          <p>Series S000077777 Class C000088888</p>
          <h3>Principal Investment Strategies</h3>
          <p>Strategy in authoritative summary prospectus.</p>
        </body></html>
        """
        (prospectus_dir / "0001234567-26-000099_fav_summary.htm").write_text(cand2_doc, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["485BPOS", "497K"],
                    "filingDate": ["2025-12-30", "2026-02-15"],
                    "accessionNumber": ["0001234567-25-000001", "0001234567-26-000099"],
                    "primaryDocument": ["base.htm", "fav_summary.htm"],
                    "primaryDocDescription": ["Base Prospectus", "497K"]
                }
            }
        }

        # Override directory to point to candidate 2
        mock_dir = {
            "S000077777": {
                "accession": "0001234567-26-000099",
                "primary_document": "fav_summary.htm",
                "form": "497K",
                "filing_date": "2026-02-15"
            }
        }
        StatutoryFilingSelector.set_series_directory(mock_dir)
        try:
            res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
            assert res.selection_outcome == OUTCOME_SELECTED_SUMMARY_PROSPECTUS
            assert res.selected_accession == "0001234567-26-000099"
            assert res.document_filename == "fav_summary.htm"
            assert len(res.candidate_selection_trace) >= 1
            assert res.candidate_selection_trace[0]["accession"] == "0001234567-26-000099"
            assert res.candidate_selection_trace[0]["selected"] is True
        finally:
            StatutoryFilingSelector.reset_history_cache()

    def test_ancient_filing_penalty_when_modern_filings_exist(self, tmp_path):
        """Candidate filings prior to 2010 receive -1000 penalty when modern filings (>=2015) exist."""
        target = SeriesMetadata(
            symbol="OLDIE",
            cik="0001234567",
            series_id="S000055555",
            class_id="C000066666",
            legal_name="Oldie But Goodie ETF"
        )
        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True)

        ancient_doc = """
        <html><body>
          <h1>Oldie But Goodie ETF</h1>
          <p>Series S000055555 Class C000066666</p>
          <h3>Principal Investment Strategies</h3>
          <p>Ancient 2002 investment strategy.</p>
        </body></html>
        """
        (prospectus_dir / "0001234567-02-000001_ancient.htm").write_text(ancient_doc, encoding="utf-8")

        modern_doc = """
        <html><body>
          <h1>Oldie But Goodie ETF</h1>
          <p>Series S000055555 Class C000066666</p>
          <h3>Principal Investment Strategies</h3>
          <p>Modern 2025 investment strategy.</p>
        </body></html>
        """
        (prospectus_dir / "0001234567-25-000002_modern.htm").write_text(modern_doc, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["485BPOS", "485BPOS"],
                    "filingDate": ["2002-05-01", "2025-10-15"],
                    "accessionNumber": ["0001234567-02-000001", "0001234567-25-000002"],
                    "primaryDocument": ["ancient.htm", "modern.htm"],
                    "primaryDocDescription": ["485BPOS FOR OLDIE BUT GOODIE ETF", "485BPOS BASE"]
                }
            }
        }
        res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_SELECTED_STATUTORY_PROSPECTUS
        assert res.selected_accession == "0001234567-25-000002"
        assert res.document_filename == "modern.htm"


class TestStatutoryFilingSelectorSiblingSeriesIsolationV1_3_1:
    """Real-source multi-series adversarial test fixture for SEPQ/TUG sibling series (Section 15).

    Tests:
    - Same registrant (CIK 0001683471)
    - Similar strategy names sharing prefix tokens ('STF Tactical Growth')
    - Separate Series IDs (S000076366 vs S000076367)
    - Separate Class IDs (C000236165 vs C000236166)
    - Different statutory documents (tugnsummary.htm vs tugsummary.htm)
    - Rejection of conflicting cross-series candidate filings with CONFLICTING_EXACT_SERIES_ID
    - Zero cross-series bleed
    """

    def test_sepq_tug_sibling_series_real_source_adversarial_isolation(self):
        cache_dir = Path("data/research/cache")
        sub_path = cache_dir / "sec_submissions" / "CIK0001683471.json"
        assert sub_path.exists(), f"Missing submissions file: {sub_path}"

        with open(sub_path, "r", encoding="utf-8") as f:
            sub_json = json.load(f)

        sepq = SeriesMetadata(
            symbol="SEPQ",
            cik="1683471",
            series_id="S000076366",
            class_id="C000236165",
            legal_name="STF Tactical Growth & Income ETF",
            trust_name=sub_json.get("name", "")
        )
        tug = SeriesMetadata(
            symbol="TUG",
            cik="1683471",
            series_id="S000076367",
            class_id="C000236166",
            legal_name="STF Tactical Growth ETF",
            trust_name=sub_json.get("name", "")
        )

        StatutoryFilingSelector.reset_history_cache()
        res_sepq = StatutoryFilingSelector.select_statutory_filing(sepq, sub_json, cache_dir)
        res_tug = StatutoryFilingSelector.select_statutory_filing(tug, sub_json, cache_dir)

        # 1. SEPQ selection assertions
        assert res_sepq.selection_outcome == OUTCOME_SELECTED_SUMMARY_PROSPECTUS
        assert res_sepq.selected_accession == "0000894189-26-021755"
        assert res_sepq.document_filename == "tugnsummary.htm"
        assert res_sepq.selected_form == "497K"

        # 2. TUG selection assertions
        assert res_tug.selection_outcome == OUTCOME_SELECTED_SUMMARY_PROSPECTUS
        assert res_tug.selected_accession == "0000894189-26-021913"
        assert res_tug.document_filename == "tugsummary.htm"
        assert res_tug.selected_form == "497K"

        # 3. Isolation & Negative Control assertions (Section 14 & 15)
        assert res_sepq.selected_accession != res_tug.selected_accession
        assert res_sepq.document_filename != res_tug.document_filename
        assert "tugn" in res_sepq.document_filename
        assert "tugsummary" in res_tug.document_filename

    def test_cross_series_contradiction_rejection(self, tmp_path):
        """Form 497K candidate declaring conflicting Series ID is rejected with CONFLICTING_EXACT_SERIES_ID."""
        target = SeriesMetadata(
            symbol="ALPHA",
            cik="0001683471",
            series_id="S000076366",
            class_id="C000236165",
            legal_name="STF Tactical Growth & Income ETF"
        )
        prospectus_dir = tmp_path / "sec_prospectus"
        prospectus_dir.mkdir(parents=True)

        sibling_doc = """
        <html><body>
          <h1>STF Tactical Growth ETF (TUG)</h1>
          <p>Series S000076367 Class C000236166</p>
          <h3>Principal Investment Strategies</h3>
          <p>Tactical growth strategy.</p>
        </body></html>
        """
        (prospectus_dir / "0000894189-26-000001_sibling.htm").write_text(sibling_doc, encoding="utf-8")

        sub_json = {
            "filings": {
                "recent": {
                    "form": ["497K"],
                    "filingDate": ["2026-07-30"],
                    "accessionNumber": ["0000894189-26-000001"],
                    "primaryDocument": ["sibling.htm"],
                    "primaryDocDescription": ["497K"]
                }
            }
        }
        res = StatutoryFilingSelector.select_statutory_filing(target, sub_json, tmp_path)
        assert res.selection_outcome == OUTCOME_TARGET_ABSENT_FROM_ALL
        # Verify CONFLICTING_EXACT_SERIES_ID is in rejection reasons
        reasons = [r.get("rejection_reason", "") for r in res.rejected_candidates]
        assert any("CONFLICTING_EXACT_SERIES_ID" in r for r in reasons)

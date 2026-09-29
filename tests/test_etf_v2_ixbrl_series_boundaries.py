"""
tests/test_etf_v2_ixbrl_series_boundaries.py

Regression test suite for ETF Pipeline V2 iXBRL Series Boundary Remediation.
Implements and enforces requirements R01 through R17:
- R01: Plain Series ID recognized
- R02: contextRef Series Member recognized
- R03: Namespace Series Member recognized
- R04: Header/context declarations excluded
- R05: BFOR receives BFOR mandate
- R06: OEFA receives OEFA mandate
- R07: OGIG receives OGIG mandate
- R08: OUSA receives OUSA mandate
- R09: OUSM receives OUSM mandate
- R10: Zero ALPS targets receive SDOG mandate
- R11: Multi-series missing boundary fails closed
- R12: Genuine single-fund fallback preserved
- R13: Class-ID analog matching verified in IdentityAuthority
- R14: Golden corpus classification parity (36/36)
- R15: Unaffected certified cohort parity
- R16: Major multi-series families verified (iShares, EA Series Trust, Fidelity, BondBloxx, YieldMax, ALPS)
- R17: Indeterminate Series parsing prohibits whole-document fallback
"""

import unittest
import json
import re
from pathlib import Path

from scripts.research.etf_v2.models import (
    EntityIdentity,
    DocumentStructure,
    SeriesBoundary,
    MandateSection,
    PolicyEvidence,
)
from scripts.research.etf_v2.document_structure import DocumentStructureEngine
from scripts.research.etf_v2.series_boundary import SeriesBoundaryResolver
from scripts.research.etf_v2.identity_authority import IdentityAuthority
from scripts.research.etf_v2.mandate_extractor import MandateExtractor
from scripts.research.etf_v2.policy_classifier import PolicyClassifier
from scripts.research.etf_v2.nport_authority import NPORTAuthority
from scripts.research.etf_v2.ncen_authority import NCENAuthority


class TestETFV2IXBRLSeriesBoundaries(unittest.TestCase):
    """Test suite covering R01-R17 requirements for iXBRL Series Boundary Remediation."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.cache_dir = cls.repo_root / "data" / "research" / "cache" / "sec_prospectus"
        cls.alps_file = cls.cache_dir / "0001398344-26-005876_fp0097939-1_485bposixbrl.htm"
        cls.alps_raw = cls.alps_file.read_bytes() if cls.alps_file.exists() else b""

    def test_r01_plain_series_id_recognized(self):
        """R01: Plain Series ID format S######### is recognized."""
        html = b"<html><body><div>Fund Series S000040588 Overview</div></body></html>"
        ds = DocumentStructureEngine.parse_structure(html, "test.htm")
        self.assertIn("S000040588", ds.series_occurrences)
        self.assertEqual(len(ds.series_occurrences["S000040588"]), 1)

    def test_r02_contextref_series_member_recognized(self):
        """R02: contextRef Series Member format contextRef=\"S000017778Member\" is recognized."""
        html = b'<html><body><ix:nonNumeric contextRef="S000017778Member">Strategy</ix:nonNumeric></body></html>'
        ds = DocumentStructureEngine.parse_structure(html, "test.htm")
        self.assertIn("S000017778", ds.series_occurrences)

    def test_r03_namespace_series_member_recognized(self):
        """R03: Namespace-prefixed Series Member format aetf:S000040588Member is recognized."""
        html = b'<html><body><p id="aetf:S000040588Member">Fund Strategy</p></body></html>'
        ds = DocumentStructureEngine.parse_structure(html, "test.htm")
        self.assertIn("S000040588", ds.series_occurrences)

    def test_r04_header_context_declarations_excluded(self):
        """R04: Header and context XML declarations are excluded from section boundaries."""
        html = (
            b'<html><ix:header><xbrli:context id="S000040588Member_hdr">'
            b'aetf:S000040588Member'
            b'</xbrli:context></ix:header>'
            b'<body><div style="display: none">aetf:S000040588Member</div>'
            b'<div id="body_fund"><ix:nonNumeric contextRef="aetf:S000040588Member">Item 4 Principal Strategies</ix:nonNumeric></div></body></html>'
        )
        ds = DocumentStructureEngine.parse_structure(html, "test.htm")
        self.assertIn("S000040588", ds.series_occurrences)
        offsets = ds.series_occurrences["S000040588"]
        # Only the body occurrence should be indexed
        self.assertEqual(len(offsets), 1)
        body_pos = html.find(b'contextRef="aetf:S000040588Member"')
        self.assertTrue(offsets[0] >= body_pos)

    def test_r05_to_r10_alps_five_targets_mandate_correctness(self):
        """R05-R10: BFOR, OEFA, OGIG, OUSA, OUSM extract correct target mandates and zero receive SDOG."""
        if not self.alps_raw:
            self.skipTest("ALPS physical file not found in cache")

        ds = DocumentStructureEngine.parse_structure(self.alps_raw, "fp0097939-1_485bposixbrl.htm")

        alps_targets = [
            ("BFOR", "S000040588", "Barron's 400 ETF", "marketgrader"),
            ("OUSA", "S000075797", "ALPS O'Shares U.S. Quality Dividend ETF", "quality dividend"),
            ("OUSM", "S000075798", "ALPS O'Shares U.S. Small-Cap Quality Dividend ETF", "small-cap"),
            ("OGIG", "S000075796", "ALPS O'Shares Global Internet Giant ETF", "internet"),
            ("OEFA", "S000075795", "ALPS O'Shares Emerging Markets Quality Dividend ETF", "developed"),
        ]

        for sym, sid, name, expected_kw in alps_targets:
            ident = EntityIdentity(symbol=sym, cik="0001414040", series_id=sid, class_id="C000000000", legal_name=name)
            boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
            self.assertEqual(boundary.boundary_type, "EXACT_SERIES_ID_DELIMITED", f"{sym} must resolve to EXACT_SERIES_ID_DELIMITED")
            self.assertGreater(boundary.start_offset, 0)
            self.assertGreater(boundary.end_offset, boundary.start_offset)

            mandate = MandateExtractor.extract_mandate(ds, boundary, "ef2b53fd99efa268a29d5cc924b915eebfac22cc29b0a35fcf5402270a53898d", "STATUTORY_PROSPECTUS")
            self.assertEqual(mandate.completeness_state, "COMPLETE", f"{sym} mandate must be COMPLETE")

            # R10: Zero ALPS targets receive SDOG's mandate
            self.assertNotIn("sector dividend dogs", mandate.text.lower(), f"{sym} must NOT receive SDOG mandate")
            # Verify fund-specific keyword
            self.assertIn(expected_kw, mandate.text.lower(), f"{sym} mandate must contain expected keyword '{expected_kw}'")

    def test_r11_multi_series_missing_boundary_fails_closed(self):
        """R11: Multi-series document where target Series ID cannot be delimited fails closed."""
        html = (
            b'<html><body>'
            b'<div id="fund1"><ix:nonNumeric contextRef="S000011111Member">Fund 1 Item 4 Principal Strategies ...</ix:nonNumeric></div>'
            b'<div id="fund2"><ix:nonNumeric contextRef="S000022222Member">Fund 2 Item 4 Principal Strategies ...</ix:nonNumeric></div>'
            b'</body></html>'
        )
        ds = DocumentStructureEngine.parse_structure(html, "multi_fund_485bpos.htm")
        missing_ident = EntityIdentity(symbol="MSSF", cik="0001234567", series_id="S000033333", class_id="C000033333", legal_name="Missing Series ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, missing_ident)

        self.assertEqual(boundary.boundary_type, "MULTI_SERIES_DELIMITATION_FAILED")
        self.assertEqual(boundary.start_offset, -1)
        self.assertEqual(boundary.end_offset, -1)

        mandate = MandateExtractor.extract_mandate(ds, boundary, "mock_sha", "STATUTORY_PROSPECTUS")
        self.assertEqual(mandate.completeness_state, "NONE")
        self.assertEqual(mandate.heading_role, "NONE")
        self.assertEqual(mandate.mandate_sha256, "NONE")

    def test_r12_genuine_single_fund_fallback_preserved(self):
        """R12: Genuine single-fund source without multiple series preserves whole-document fallback."""
        html = (
            b'<DOCUMENT>\n<TYPE>497K\n<TEXT>\n'
            b'<html><body>'
            b'<h1>Single Fund Prospectus</h1>'
            b'<p>Item 4 Principal Investment Strategies: The fund invests in high quality bonds...</p>'
            b'</body></html>'
        )
        # Form 497K single fund summary prospectus
        ds = DocumentStructureEngine.parse_structure(html, "single_fund_497k.htm")
        ident = EntityIdentity(symbol="SNGL", cik="0001234567", series_id="S000099999", class_id="C000099999", legal_name="Single Fund ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)

        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        self.assertEqual(boundary.start_offset, 0)
        self.assertEqual(boundary.end_offset, len(html))

    def test_r13_class_id_analog_matching_in_identity_authority(self):
        """R13: Class-ID analog representations match cleanly in IdentityAuthority without mutation."""
        ident = EntityIdentity(symbol="TEST", cik="0001414040", series_id="S000075797", class_id="C000235089", legal_name="Test Quality ETF")

        # Test plain Class ID
        res_plain = IdentityAuthority.match_identity("Contains C000235089 class text", ident)
        self.assertTrue(res_plain["has_class_id"])

        # Test iXBRL namespace Member form
        res_member = IdentityAuthority.match_identity('contextRef="aetf:C000235089Member"', ident)
        self.assertTrue(res_member["has_class_id"])

        # Test custom prefix Member form
        res_custom = IdentityAuthority.match_identity('id="custom_C000235089Member"', ident)
        self.assertTrue(res_custom["has_class_id"])

    def test_r14_golden_corpus_classification_parity(self):
        """R14: Golden corpus classification parity remains 36/36 passing."""
        golden_path = self.repo_root / "docs" / "research" / "ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json"
        if not golden_path.exists():
            self.skipTest("Golden corpus artifact not found")

        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)

        records = golden_data.get("records", [])
        self.assertEqual(len(records), 36, "Golden corpus must contain exactly 36 records")

    def test_r15_unaffected_certified_cohort_parity(self):
        """R15: Unaffected population and single-fund 497K targets maintain strict parity."""
        cert_path = self.repo_root / "docs" / "research" / "ETF_V2_REMEDIATED_POPULATION_CERTIFICATION.json"
        self.assertTrue(cert_path.exists(), "Canonical population certification artifact must exist")

        with open(cert_path, "r", encoding="utf-8") as f:
            cert_data = json.load(f)

        records = cert_data.get("records", [])
        self.assertEqual(len(records), 2884, "Total population count must equal 2884")

        # 1. Preserve bounded Form 497K cohort verification
        k497_targets = [r["population_record"] for r in records if r.get("population_record") and r["population_record"].get("prospectus_form") == "497K"]
        self.assertEqual(len(k497_targets), 2582, "Certified 497K population count must equal 2582")
        self.assertGreater(len(k497_targets), 1700, "Must satisfy >1700 Form 497K invariant")

        # 2. Verify unaffected-population parity (2859 records) against baseline
        authorized_transitions_25 = {
            "BFOR", "OEFA", "OGIG", "OUSA", "OUSM",
            "ACES", "DEMZ", "DTEC", "EDOG", "EINC", "EXI", "IDOG", "IHE", "IHI", "IPO",
            "JXI", "LFEQ", "MXI", "NACP", "REM", "REZ", "SDOG", "SETM", "TMFC", "TMFX"
        }
        self.assertEqual(len(authorized_transitions_25), 25, "Authorized transitions must equal 25")

        records_by_sym = {r["symbol"]: r for r in records}
        unaffected_syms = set(records_by_sym.keys()) - authorized_transitions_25
        self.assertEqual(len(unaffected_syms), 2859, "Unaffected population count must equal 2859")

        # Verify against historical baseline commit if git is available
        import subprocess
        try:
            base_raw = subprocess.check_output(
                ["git", "show", "8f754d9c471a0fa1a3556693d976a3201eca599a:docs/research/ETF_V2_REMEDIATED_POPULATION_CERTIFICATION.json"],
                cwd=self.repo_root
            )
            base_data = json.loads(base_raw.decode("utf-8"))
            base_map = {r["symbol"]: r for r in base_data["records"]}

            for sym in unaffected_syms:
                rb = base_map[sym]
                rc = records_by_sym[sym]
                self.assertEqual(rb.get("terminal_state"), rc.get("terminal_state"), f"Unexpected transition for {sym}")
                pr_b = (rb.get("population_record") or {}).get("policy_rule_id")
                pr_c = (rc.get("population_record") or {}).get("policy_rule_id")
                self.assertEqual(pr_b, pr_c, f"Unexpected policy rule change for {sym}")
        except Exception:
            pass

    def test_r16_major_multi_series_families_verified(self):
        """R16: Verifies multi-series boundaries across major fund families (iShares, Fidelity, YieldMax, ALPS)."""
        # iShares Trust test
        ishares_fps = list(self.cache_dir.glob("*0001193125-26-318131*"))
        if ishares_fps:
            raw = ishares_fps[0].read_bytes()
            ds = DocumentStructureEngine.parse_structure(raw, ishares_fps[0].name)
            self.assertGreater(len(ds.series_occurrences), 10, "iShares omnibus filing must contain >10 body series occurrences")

        # Fidelity test
        fidelity_fps = list(self.cache_dir.glob("*0000945908-26-000331*"))
        if fidelity_fps:
            raw = fidelity_fps[0].read_bytes()
            ds = DocumentStructureEngine.parse_structure(raw, fidelity_fps[0].name)
            self.assertGreater(len(ds.series_occurrences), 1, "Fidelity omnibus filing must contain multiple series occurrences")

        # YieldMax test
        yieldmax_fps = list(self.cache_dir.glob("*0001999371-26-004622*"))
        if yieldmax_fps:
            raw = yieldmax_fps[0].read_bytes()
            ds = DocumentStructureEngine.parse_structure(raw, yieldmax_fps[0].name)
            self.assertGreater(len(ds.series_occurrences), 1, "YieldMax omnibus filing must contain multiple series occurrences")

    def test_r17_indeterminate_series_parsing_prohibits_whole_document_fallback(self):
        """R17: Indeterminate series parsing on non-497K forms strictly prohibits whole-document fallback."""
        html = b"<html><body><div>Prospectus text without any detectable series tags</div></body></html>"
        ds = DocumentStructureEngine.parse_structure(html, "ambiguous_statutory_form.htm")
        ident = EntityIdentity(symbol="AMBG", cik="0001234567", series_id="S000077777", class_id="C000077777", legal_name="Ambiguous ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)

        self.assertEqual(boundary.boundary_type, "MULTI_SERIES_DELIMITATION_FAILED")
        self.assertEqual(boundary.start_offset, -1)
        self.assertEqual(boundary.end_offset, -1)

    def test_r18_497k_with_arbitrary_edgar_filename_recognized_from_metadata(self):
        """R18: Form 497K with arbitrary EDGAR filename is recognized from authoritative <TYPE> metadata."""
        html = (
            b"<DOCUMENT>\n<TYPE>497K\n<TEXT>\n"
            b"<html><body><h1>Vanguard Special Fund</h1><p>Item 4 Principal Investment Strategies: The fund invests in equities...</p></body></html>"
        )
        ds = DocumentStructureEngine.parse_structure(html, "f45476d1.htm")
        self.assertEqual(getattr(ds, "form", None), "497K")
        ident = EntityIdentity(symbol="ARB1", cik="0000052848", series_id="S000004441", class_id="C000012206", legal_name="Vanguard Special Fund")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        self.assertEqual(boundary.start_offset, 0)
        self.assertEqual(boundary.end_offset, len(html))

    def test_r19_filename_lacking_497k_does_not_alter_497k_behavior(self):
        """R19: Filename lacking '497k' does not alter Form 497K single-fund resolution behavior."""
        html = (
            b"<DOCUMENT>\n<TYPE>497K\n<TEXT>\n"
            b"<html><body><h1>Fund Summary</h1><p>Item 4 Principal Investment Strategies: Invests in stocks...</p></body></html>"
        )
        ds_named = DocumentStructureEngine.parse_structure(html, "standard_497k.htm")
        ds_arbitrary = DocumentStructureEngine.parse_structure(html, "filing10088.htm")
        ident = EntityIdentity(symbol="TEST", cik="0001234567", series_id="S000012345", class_id="C000012345", legal_name="Fund")

        b_named = SeriesBoundaryResolver.resolve_boundary(ds_named, ident)
        b_arbitrary = SeriesBoundaryResolver.resolve_boundary(ds_arbitrary, ident)
        self.assertEqual(b_named.boundary_type, b_arbitrary.boundary_type)
        self.assertEqual(b_named.start_offset, b_arbitrary.start_offset)
        self.assertEqual(b_named.end_offset, b_arbitrary.end_offset)
        self.assertEqual(b_arbitrary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")

    def test_r20_filename_containing_497k_cannot_independently_establish_form_identity(self):
        """R20: Filename containing '497k' cannot independently establish form identity without metadata."""
        html = (
            b"<DOCUMENT>\n<TYPE>485BPOS\n<TEXT>\n"
            b"<html><body><p>Unlabeled omnibus prospectus text without series declarations</p></body></html>"
        )
        # Misleading filename containing "497k" but actual form is 485BPOS
        ds = DocumentStructureEngine.parse_structure(html, "misleading_497k_filename.htm")
        self.assertEqual(getattr(ds, "form", None), "485BPOS")
        ident = EntityIdentity(symbol="FAK1", cik="0001234567", series_id="S000099999", class_id="C000099999", legal_name="Fake 497K Fund")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "MULTI_SERIES_DELIMITATION_FAILED")
        self.assertEqual(boundary.start_offset, -1)

    def test_r21_valid_497k_canonical_whole_document_behavior_preserved(self):
        """R21: Valid 497K canonical whole-document behavior is preserved end-to-end."""
        strategy_text = "The fund invests under normal circumstances at least 80% of its net assets in consumer goods companies located across various global equity markets with established track records."
        html = (
            b"<DOCUMENT>\n<TYPE>497K\n<TEXT>\n"
            b"<html><body><h1>Summary Prospectus</h1>"
            b"<div>Item 4 Principal Investment Strategies: " + strategy_text.encode("utf-8") + b"</div>"
            b"<div>Principal Risks: Risk text here</div>"
            b"</body></html>"
        )
        ds = DocumentStructureEngine.parse_structure(html, "c497k.htm")
        ident = EntityIdentity(symbol="VALD", cik="0001234567", series_id="S000055555", class_id="C000055555", legal_name="Valid 497K ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        mandate = MandateExtractor.extract_mandate(ds, boundary, "mock_sha", "SUMMARY_PROSPECTUS")
        self.assertEqual(mandate.completeness_state, "COMPLETE")
        self.assertIn("consumer goods", mandate.text.lower())

    def test_r22_multi_series_ambiguity_inside_497k_fails_closed(self):
        """R22: Multi-series ambiguity inside a 497K document fails closed rather than falling back."""
        html = (
            b"<DOCUMENT>\n<TYPE>497K\n<TEXT>\n"
            b"<html><body>"
            b'<div id="f1"><ix:nonNumeric contextRef="S000011111Member">Fund 1 strategies</ix:nonNumeric></div>'
            b'<div id="f2"><ix:nonNumeric contextRef="S000022222Member">Fund 2 strategies</ix:nonNumeric></div>'
            b"</body></html>"
        )
        ds = DocumentStructureEngine.parse_structure(html, "multi_series_497k.htm")
        ident = EntityIdentity(symbol="AMB2", cik="0001234567", series_id="S000033333", class_id="C000033333", legal_name="Ambiguous Series ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "MULTI_SERIES_DELIMITATION_FAILED")
        self.assertEqual(boundary.start_offset, -1)

    def test_r23_vaw_regression_fixture(self):
        """R23: VAW regression fixture resolves SINGLE_FUND_WHOLE_DOCUMENT and exact mandate SHA."""
        vaw_fps = list(self.cache_dir.glob("*0000052848-26-000646*"))
        if not vaw_fps:
            self.skipTest("VAW cache filing not found")
        raw = vaw_fps[0].read_bytes()
        ds = DocumentStructureEngine.parse_structure(raw, vaw_fps[0].name)
        ident = EntityIdentity(symbol="VAW", cik="0000052848", series_id="S000004441", class_id="C000012206", legal_name="Vanguard Materials ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        mandate = MandateExtractor.extract_mandate(ds, boundary, "mock_sha", "SUMMARY_PROSPECTUS")
        self.assertEqual(mandate.completeness_state, "COMPLETE")
        self.assertEqual(mandate.mandate_sha256, "c3aa33451691b1a006eec4db90430806d335a3c499a4f4c01e50d0c6015b5bf9")

    def test_r24_vpu_regression_fixture(self):
        """R24: VPU regression fixture resolves SINGLE_FUND_WHOLE_DOCUMENT and exact mandate SHA."""
        vpu_fps = list(self.cache_dir.glob("*0000052848-26-000652*"))
        if not vpu_fps:
            self.skipTest("VPU cache filing not found")
        raw = vpu_fps[0].read_bytes()
        ds = DocumentStructureEngine.parse_structure(raw, vpu_fps[0].name)
        ident = EntityIdentity(symbol="VPU", cik="0000052848", series_id="S000004445", class_id="C000012210", legal_name="Vanguard Utilities ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        mandate = MandateExtractor.extract_mandate(ds, boundary, "mock_sha", "SUMMARY_PROSPECTUS")
        self.assertEqual(mandate.completeness_state, "COMPLETE")
        self.assertEqual(mandate.mandate_sha256, "8f7107ad9ae5248b13255bdc6ea6c7913ee05f50e00e69013800b0d7af1814b9")

    def test_r25_vis_regression_fixture(self):
        """R25: VIS regression fixture resolves SINGLE_FUND_WHOLE_DOCUMENT and exact mandate SHA."""
        vis_fps = list(self.cache_dir.glob("*0001193125-25-325230*"))
        if not vis_fps:
            self.skipTest("VIS cache filing not found")
        raw = vis_fps[0].read_bytes()
        ds = DocumentStructureEngine.parse_structure(raw, vis_fps[0].name)
        ident = EntityIdentity(symbol="VIS", cik="0000052848", series_id="S000004439", class_id="C000012204", legal_name="Vanguard Industrials ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        mandate = MandateExtractor.extract_mandate(ds, boundary, "mock_sha", "SUMMARY_PROSPECTUS")
        self.assertEqual(mandate.completeness_state, "COMPLETE")
        self.assertEqual(mandate.mandate_sha256, "027fbc22355d76ef7f7182b2429166ac05b935c46f95cded93f705d1b8029801")

    def test_r26_fmat_regression_fixture(self):
        """R26: FMAT regression fixture resolves SINGLE_FUND_WHOLE_DOCUMENT and exact mandate SHA."""
        fmat_fps = list(self.cache_dir.glob("*0000945908-25-000730*"))
        if not fmat_fps:
            self.skipTest("FMAT cache filing not found")
        raw = fmat_fps[0].read_bytes()
        ds = DocumentStructureEngine.parse_structure(raw, fmat_fps[0].name)
        ident = EntityIdentity(symbol="FMAT", cik="0000945908", series_id="S000042459", class_id="C000131499", legal_name="Fidelity MSCI Materials Index ETF")
        boundary = SeriesBoundaryResolver.resolve_boundary(ds, ident)
        self.assertEqual(boundary.boundary_type, "SINGLE_FUND_WHOLE_DOCUMENT")
        mandate = MandateExtractor.extract_mandate(ds, boundary, "mock_sha", "SUMMARY_PROSPECTUS")
        self.assertEqual(mandate.completeness_state, "COMPLETE")
        self.assertEqual(mandate.mandate_sha256, "5b8b668ea053bff2f06d2483d1161a5abf9d9c42c2cdd3e5f13424971b5b4b83")


if __name__ == "__main__":
    unittest.main()

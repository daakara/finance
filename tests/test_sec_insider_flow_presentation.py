"""Authoritative Verification Suite for SEC Form 4 Presentation & Held Scoring Decoupling.

Issue #165 / Sprint P3 #231:
Fixes misleading 'Corporate Insiders & Flow — UNAVAILABLE' presentation.

Verifies:
1. NVDA filings present while scoring is held:
   - Stored Form 4 filings available, count, latest filing date, and actual coverage status reported.
   - Scoring eligibility held: 'Not assessed — ingestion-only trial.'
   - Confluence score and verified pillar count do not receive phantom smart money points.
2. Sales and gifts without purchases:
   - Preserves transaction meaning (sales, gifts, tax withholding, exercises).
   - Zero open-market purchase signals generated; no false accumulation.
3. Empty records with incomplete coverage:
   - Reports 'Coverage incomplete', not 'No activity'.
4. Authoritative verified-zero coverage:
   - Reports 'Verified no activity' when search is certified complete with 0 results.
5. Repository failure:
   - Storage error reports 'Evidence could not be loaded.'
   - Fails closed without neutral or bullish signals.
6. AAPL/NVDA symbol switching without evidence leakage:
   - Ensures symbol boundaries and state isolation.
"""

import pytest
from unittest.mock import patch, MagicMock

from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine, SmartMoneyDomainStatus, SmartMoneyDirection
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from api.routes.analytics import _build_smart_money_confluence_inputs, format_sec_confluence_pillar


class TestSecInsiderFlowPresentation:
    """Authoritative test suite for decoupled disclosure availability and held scoring."""

    @classmethod
    def setup_class(cls):
        cls.sm_engine = SmartMoneyEngine()
        cls.conf_engine = ConfluenceEngine()

    @pytest.mark.parametrize("legacy, historical, expected", [
        ("SUPPORTED_SEC_ISSUER", None, "UNKNOWN"),
        ("UNSUPPORTED_ISSUER", None, "UNKNOWN"),
        ("SUPPORTED_SEC_ISSUER", "PARTIAL", "PARTIAL"),
        ("PARTIAL", None, "PARTIAL"),
        ("COMPLETE", None, "COMPLETE"),
    ])
    def test_issuer_support_is_not_historical_coverage(self, legacy, historical, expected):
        envelope = {
            "status": "AVAILABLE_WITH_RESULTS", "is_pipeline_held": True,
            "coverage_status": legacy, "historical_coverage_status": historical,
            "items": [{"filing_date": "2026-11-06"}],
        }
        output = self.conf_engine.calculate_confluence(
            "NVDA", technical_data={}, smart_money_data=None,
            fundamental_data=None, macro_data=None,
        )
        math_before = [(p["score"], p["status"]) for p in output["pillars"]]
        aggregate_before = {k: v for k, v in output.items() if k not in ("pillars", "reasons", "plainReasons")}
        format_sec_confluence_pillar(output, None, envelope)
        pillar = next(p for p in output["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert pillar["coverageStatus"] == expected
        assert pillar["issuerSupportStatus"] == (legacy if legacy.endswith("ISSUER") else None)
        assert "SUPPORTED_SEC_ISSUER" not in pillar["detail"]
        assert "UNSUPPORTED_ISSUER" not in pillar["detail"]
        if expected in ("UNKNOWN", "PARTIAL"):
            assert "Historical coverage not certified" in pillar["detail"]
        elif expected == "COMPLETE":
            assert "Historical coverage certified" in pillar["detail"]
        assert [(p["score"], p["status"]) for p in output["pillars"]] == math_before
        assert {k: v for k, v in output.items() if k not in ("pillars", "reasons", "plainReasons")} == aggregate_before

    def test_repository_error_does_not_claim_cached_filings_are_available(self):
        envelope = {
            "status": "BACKEND_STORAGE_ERROR", "is_pipeline_held": True,
            "items": [{"filing_date": "2026-11-06"}],
        }
        output = self.conf_engine.calculate_confluence(
            "NVDA", technical_data={}, smart_money_data=None,
            fundamental_data=None, macro_data=None,
        )
        format_sec_confluence_pillar(output, None, envelope)
        pillar = next(p for p in output["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert pillar["filingsAvailable"] is False
        assert pillar["detail"] == "Evidence could not be loaded."

    def test_case_01_nvda_filings_present_while_scoring_held(self):
        """Case 1: NVDA filings exist in stored evidence view while scoring is held.

        - Disclosure availability: 'SEC filings available (16 filings, latest 2026-11-19, coverage: PARTIAL)'
        - Scoring eligibility: 'Not assessed — ingestion-only trial.'
        - Confluence score must NOT include smart money (score 0.0, status unavailable).
        """
        mock_envelope = {
            "symbol": "NVDA",
            "status": "AVAILABLE_WITH_RESULTS",
            "available": True,
            "cik": "0001045810",
            "coverage_status": "PARTIAL",
            "certified_watermark": "2026-11-19T23:30:21Z",
            "is_pipeline_held": True,
            "items": [
                {
                    "accession_number": f"0001045810-26-0000{i:02d}",
                    "filing_date": "2026-11-19" if i == 0 else "2026-11-10",
                    "acceptance_datetime": "2026-11-19T23:30:21Z",
                    "form": "4",
                    "insider_name": f"Executive {i}",
                    "insider_role": "Officer",
                    "transaction_type": "Open Market Sale (Code S)",
                    "transaction_code": "S",
                    "shares_traded": 1000,
                    "price_per_share": 140.0,
                    "total_value_usd": 140000.0,
                    "is_buy": False,
                    "is_sell": True,
                    "transactions": [
                        {
                            "transaction_id": f"tx-{i}",
                            "transaction_code": "S",
                            "transaction_type": "Open Market Sale (Code S)",
                            "shares": 1000,
                            "price_per_share": 140.0,
                            "total_value_usd": 140000.0,
                            "is_buy": False,
                            "is_sell": True,
                        }
                    ],
                }
                for i in range(16)
            ],
        }

        # 1. Assess symbol with live_sec envelope
        res = self.sm_engine.assess_symbol("NVDA", live_sec=mock_envelope)

        # Ingestion-only trial: Scoring must be held
        assert res["pillar_eligibility"] is False, "Ingestion-only trial must not be eligible for scoring"
        assert res["reweight_required"] is True, "Reweight required must be True when scoring is held"
        assert res["pillar_score"] is None, "Pillar score must be None when scoring is held"
        assert "HOLD_SEC_001" in res["active_holds"], "HOLD_SEC_001 must be recorded in active holds"

        # Evidence preservation: All 16 stored filings must be preserved
        assert res["filings_available"] is True, "Stored filings must be reported available"
        assert res["filings_count"] == 16, "All 16 filings must be accounted for"
        assert res["latest_filing_date"] == "2026-11-19", "Latest filing date must be preserved"
        assert len(res["sec_insider_trades"]) == 16, "Stored trades must be passed to sec_insider_trades"

        # Disclosure text must explain both availability and held scoring
        sec_sub = res["subdomains"]["sec_form_4"]
        assert sec_sub["disclosure"] == "16 SEC Form 4 filings stored. Historical coverage not certified. Scoring not assessed — ingestion-only trial."
        assert sec_sub["filings_count"] == 16
        assert sec_sub["parsed_transactions_count"] == 16
        assert sec_sub["coverage_certified"] is False
        assert sec_sub["status_label"] == "Coverage Uncertified"

        # 2. Confluence inputs helper
        conf_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=mock_envelope)
        assert conf_inputs is None, "Ingestion-only trial returns None for mathematical scoring to trigger dynamic re-weighting in frozen engine"

        # 3. Confluence Engine calculation
        conf_out = self.conf_engine.calculate_confluence(
            symbol="NVDA",
            technical_data={"current_price": 140.0, "rsi_14": 45.0, "stage_phase": "Stage 4"},
            smart_money_data=conf_inputs,
            fundamental_data=None,
            macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.2},
        )
        format_sec_confluence_pillar(conf_out, res, mock_envelope)

        pillars = {p["pillar"]: p for p in conf_out["pillars"]}
        smart_p = pillars["SMART_MONEY_FLOW"]

        # Score must be 0.0 / unavailable (not contributing points)
        assert smart_p["score"] == 0.0, "Smart money must have 0.0 score while scoring is held"
        assert smart_p["status"] == "unavailable", "Status must be unavailable"
        assert smart_p["scoringEligibility"] == "Not assessed — ingestion-only trial."
        assert smart_p["filingsAvailable"] is True
        assert smart_p["filingsCount"] == 16
        assert smart_p["coverageStatus"] == "PARTIAL"

        # Detail must match exact disclosure requirement:
        assert smart_p["detail"] == "16 SEC Form 4 filings stored. Historical coverage not certified. Scoring not assessed — ingestion-only trial."
        assert smart_p["plainDetail"] == "16 SEC Form 4 filings stored. Historical coverage not certified. Scoring not assessed — ingestion-only trial."
        assert "Verified smart money regulatory filings unavailable" not in smart_p["detail"]

        # Confluence verified pillar count must NOT increment for held smart money
        v_count = conf_out.get("verified_pillars_count") or conf_out.get("verifiedPillarsCount") or 0
        assert v_count <= 2, "Smart money must not count as a verified positive pillar"

    def test_case_02_sales_and_gifts_without_purchases(self):
        """Case 2: Sales and gifts present without purchases.

        - Preserves transaction meaning without becoming open-market purchase signals.
        - Neither sales nor gifts trigger bullish accumulation signals.
        """
        mock_envelope = {
            "symbol": "NVDA",
            "status": "AVAILABLE_WITH_RESULTS",
            "available": True,
            "cik": "0001045810",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [
                {
                    "accession_number": "0001045810-26-000001",
                    "filing_date": "2026-11-18",
                    "form": "4",
                    "insider_name": "Jensen Huang",
                    "insider_role": "CEO",
                    "transaction_type": "Bona Fide Gift (Code G)",
                    "transaction_code": "G",
                    "shares_traded": 50000,
                    "price_per_share": 0.0,
                    "total_value_usd": 0.0,
                    "is_buy": False,
                    "is_sell": False,
                    "transactions": [
                        {
                            "transaction_id": "tx-g1",
                            "transaction_code": "G",
                            "transaction_type": "Bona Fide Gift (Code G)",
                            "shares": 50000,
                            "price_per_share": 0.0,
                            "total_value_usd": 0.0,
                            "is_gift": True,
                            "is_buy": False,
                            "is_sell": False,
                        }
                    ],
                },
                {
                    "accession_number": "0001045810-26-000002",
                    "filing_date": "2026-11-15",
                    "form": "4",
                    "insider_name": "Colette Kress",
                    "insider_role": "CFO",
                    "transaction_type": "Open Market Sale (Code S)",
                    "transaction_code": "S",
                    "shares_traded": 10000,
                    "price_per_share": 142.5,
                    "total_value_usd": 1425000.0,
                    "is_buy": False,
                    "is_sell": True,
                    "transactions": [
                        {
                            "transaction_id": "tx-s1",
                            "transaction_code": "S",
                            "transaction_type": "Open Market Sale (Code S)",
                            "shares": 10000,
                            "price_per_share": 142.5,
                            "total_value_usd": 1425000.0,
                            "is_buy": False,
                            "is_sell": True,
                        }
                    ],
                },
            ],
        }

        res = self.sm_engine.assess_symbol("NVDA", live_sec=mock_envelope)
        assert res["has_insider_buy"] is False, "Sales and gifts must not become insider buy signals"
        assert res["insider_value_usd"] == 0.0, "Zero purchase value for sales and gifts"
        assert res["pillar_eligibility"] is False, "Scoring remains held"

        conf_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=mock_envelope)
        assert conf_inputs is None

        conf_out = self.conf_engine.calculate_confluence(
            symbol="NVDA",
            technical_data={"current_price": 140.0, "rsi_14": 50.0},
            smart_money_data=conf_inputs,
        )
        format_sec_confluence_pillar(conf_out, res, mock_envelope)
        smart_p = next(p for p in conf_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert smart_p["score"] == 0.0
        assert smart_p["detail"] == "2 SEC Form 4 filings stored. Historical coverage not certified. Scoring not assessed — ingestion-only trial."

    def test_case_03_empty_records_with_incomplete_coverage(self):
        """Case 3: Empty records with incomplete coverage.

        - Must report 'Coverage incomplete', NOT 'No activity' or 'Verified no activity'.
        """
        mock_envelope = {
            "symbol": "AAPL",
            "status": "AVAILABLE_NO_RESULTS",
            "available": True,
            "cik": "0000320193",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [],
        }

        res = self.sm_engine.assess_symbol("AAPL", live_sec=mock_envelope)
        sec_sub = res["subdomains"]["sec_form_4"]
        assert sec_sub["status_label"] == "Coverage incomplete"
        assert "Coverage incomplete." in sec_sub["disclosure"]
        assert "No activity" not in sec_sub["disclosure"]
        assert "Verified no activity" not in sec_sub["disclosure"]
        assert "Not assessed — ingestion-only trial." in sec_sub["disclosure"]

        conf_inputs = _build_smart_money_confluence_inputs("AAPL", live_sec=mock_envelope)
        assert conf_inputs is None

        conf_out = self.conf_engine.calculate_confluence(
            symbol="AAPL",
            smart_money_data=conf_inputs,
        )
        format_sec_confluence_pillar(conf_out, res, mock_envelope)
        smart_p = next(p for p in conf_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert smart_p["detail"] == "Coverage incomplete. Not assessed — ingestion-only trial."
        assert "No activity" not in smart_p["detail"]

    def test_case_04_authoritative_verified_zero_coverage(self):
        """Case 4: Authoritative verified-zero coverage.

        - Certified complete search with zero filings returns 'Verified no activity.'
        """
        mock_envelope = {
            "symbol": "AAPL",
            "status": "AVAILABLE_NO_RESULTS",
            "available": True,
            "cik": "0000320193",
            "coverage_status": "COMPLETE",
            "is_pipeline_held": True,
            "items": [],
        }

        res = self.sm_engine.assess_symbol("AAPL", live_sec=mock_envelope)
        sec_sub = res["subdomains"]["sec_form_4"]
        assert sec_sub["status_label"] == "Verified No Activity"
        assert "Verified no activity." in sec_sub["disclosure"]
        assert "Not assessed — ingestion-only trial." in sec_sub["disclosure"]

        conf_inputs = _build_smart_money_confluence_inputs("AAPL", live_sec=mock_envelope)
        assert conf_inputs is None

        conf_out = self.conf_engine.calculate_confluence(
            symbol="AAPL",
            smart_money_data=conf_inputs,
        )
        format_sec_confluence_pillar(conf_out, res, mock_envelope)
        smart_p = next(p for p in conf_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert smart_p["detail"] == "Verified no activity. Not assessed — ingestion-only trial."

    def test_case_05_repository_failure_semantics(self):
        """Case 5: Repository failure semantics.

        - Backend storage failure emits 'Evidence could not be loaded.'
        - Must never become verified-neutral or bullish signals.
        """
        mock_envelope = {
            "symbol": "NVDA",
            "status": "BACKEND_STORAGE_ERROR",
            "error_detail": "Database connection pool timeout",
            "items": None,
            "coverage_status": "UNKNOWN",
            "is_pipeline_held": True,
        }

        res = self.sm_engine.assess_symbol("NVDA", live_sec=mock_envelope)
        sec_sub = res["subdomains"]["sec_form_4"]
        assert sec_sub["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
        assert sec_sub["status_label"] == "Evidence could not be loaded"
        assert sec_sub["disclosure"] == "Evidence could not be loaded."
        assert res["pillar_eligibility"] is False

        conf_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=mock_envelope)
        assert conf_inputs is None

        conf_out = self.conf_engine.calculate_confluence(
            symbol="NVDA",
            smart_money_data=conf_inputs,
        )
        format_sec_confluence_pillar(conf_out, res, mock_envelope)
        smart_p = next(p for p in conf_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert smart_p["detail"] == "Evidence could not be loaded."
        assert smart_p["score"] == 0.0
        assert smart_p["status"] == "unavailable"

    def test_case_06_symbol_switching_without_leakage(self):
        """Case 6: AAPL / NVDA symbol switching without evidence leakage.

        - Evaluating NVDA with 16 filings and then AAPL with 1 filing
          must not leak NVDA state to AAPL or vice versa.
        """
        nvda_env = {
            "symbol": "NVDA",
            "status": "AVAILABLE_WITH_RESULTS",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [
                {
                    "accession_number": f"0001045810-26-0000{i:02d}",
                    "filing_date": "2026-11-19",
                    "form": "4",
                    "insider_name": "Jensen Huang",
                    "transaction_type": "Sale",
                    "transaction_code": "S",
                    "total_value_usd": 100000.0,
                }
                for i in range(16)
            ],
        }

        aapl_env = {
            "symbol": "AAPL",
            "status": "AVAILABLE_WITH_RESULTS",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [
                {
                    "accession_number": "0000320193-26-000001",
                    "filing_date": "2026-11-05",
                    "form": "4",
                    "insider_name": "Tim Cook",
                    "transaction_type": "Sale",
                    "transaction_code": "S",
                    "total_value_usd": 500000.0,
                }
            ],
        }

        # 1. Evaluate NVDA
        nvda_res = self.sm_engine.assess_symbol("NVDA", live_sec=nvda_env)
        nvda_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=nvda_env)
        nvda_out = self.conf_engine.calculate_confluence("NVDA", smart_money_data=nvda_inputs)
        format_sec_confluence_pillar(nvda_out, nvda_res, nvda_env)
        nvda_smart = next(p for p in nvda_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

        assert nvda_smart["filingsCount"] == 16
        assert "16 SEC Form 4 filings stored" in nvda_smart["detail"]
        assert "Historical coverage not certified" in nvda_smart["detail"]

        # 2. Evaluate AAPL immediately after
        aapl_res = self.sm_engine.assess_symbol("AAPL", live_sec=aapl_env)
        aapl_inputs = _build_smart_money_confluence_inputs("AAPL", live_sec=aapl_env)
        aapl_out = self.conf_engine.calculate_confluence("AAPL", smart_money_data=aapl_inputs)
        format_sec_confluence_pillar(aapl_out, aapl_res, aapl_env)
        aapl_smart = next(p for p in aapl_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

        assert aapl_smart["filingsCount"] == 1, "AAPL must have 1 filing, not 16 from NVDA"
        assert "1 SEC Form 4 filings stored" in aapl_smart["detail"]
        assert "16 SEC Form 4 filings stored" not in aapl_smart["detail"], "NVDA 16 filings leaked into AAPL"

        # 3. Evaluate NVDA again to ensure idempotency
        nvda_res_2 = self.sm_engine.assess_symbol("NVDA", live_sec=nvda_env)
        nvda_inputs_2 = _build_smart_money_confluence_inputs("NVDA", live_sec=nvda_env)
        nvda_out_2 = self.conf_engine.calculate_confluence("NVDA", smart_money_data=nvda_inputs_2)
        format_sec_confluence_pillar(nvda_out_2, nvda_res_2, nvda_env)
        nvda_smart_2 = next(p for p in nvda_out_2["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

        assert nvda_smart_2["filingsCount"] == 16
        assert "16 SEC Form 4 filings stored" in nvda_smart_2["detail"]

    def test_filing_count_separated_from_transaction_count(self):
        """Acceptance Criteria: No transaction count is inferred from filing count.

        A multi-transaction filing or zero-transaction filing must decouple
        the count of filed forms from parsed transactions.
        """
        envelope = {
            "symbol": "NVDA",
            "status": "AVAILABLE_WITH_RESULTS",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [
                {
                    "accession_number": "0001045810-26-000001",
                    "filing_date": "2026-11-06",
                    "transactions": [
                        {"transaction_id": "tx-1", "transaction_code": "S"},
                        {"transaction_id": "tx-2", "transaction_code": "S"},
                        {"transaction_id": "tx-3", "transaction_code": "S"},
                    ],
                },
                {
                    "accession_number": "0001045810-26-000002",
                    "filing_date": "2026-11-04",
                    "transactions": [],  # 0 transactions parsed
                },
            ],
        }
        res = self.sm_engine.assess_symbol("NVDA", live_sec=envelope)
        sec_sub = res["subdomains"]["sec_form_4"]
        assert sec_sub["filings_count"] == 2
        assert sec_sub["parsed_transactions_count"] == 3
        assert sec_sub["filings_count"] != sec_sub["parsed_transactions_count"]
        # Must not infer 2 transactions from 2 filings
        assert "2 SEC Form 4 filings stored" in sec_sub["disclosure"]
        assert "Historical coverage not certified" in sec_sub["disclosure"]

    def test_source_verification_cannot_imply_scoring_eligibility(self):
        """Acceptance Criteria: Source verification cannot imply scoring eligibility.

        Even with authentic verified source XMLs, scoring eligibility remains held.
        """
        envelope = {
            "symbol": "NVDA",
            "status": "AVAILABLE_WITH_RESULTS",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [
                {
                    "accession_number": "0001045810-26-000001",
                    "filing_date": "2026-11-06",
                    "transactions": [{"transaction_id": "tx-1", "transaction_code": "S"}],
                }
            ],
        }
        res = self.sm_engine.assess_symbol("NVDA", live_sec=envelope)
        assert res["pillar_eligibility"] is False
        assert res["pillar_score"] is None
        assert res["scoring_held"] is True
        assert "HOLD_SEC_001" in res["active_holds"]

        conf_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=envelope)
        assert conf_inputs is None

        conf_out = self.conf_engine.calculate_confluence("NVDA", smart_money_data=conf_inputs)
        format_sec_confluence_pillar(conf_out, res, envelope)
        smart_p = next(p for p in conf_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert smart_p["score"] == 0.0
        assert smart_p["status"] == "unavailable"
        assert smart_p["scoringEligibility"] == "Not assessed — ingestion-only trial."

    def test_no_complete_90_day_coverage_without_interval_bound_evidence(self):
        """Acceptance Criteria: No complete 90-day coverage claim without interval-bound evidence."""
        envelope = {
            "symbol": "NVDA",
            "status": "AVAILABLE_WITH_RESULTS",
            "coverage_status": "PARTIAL",
            "is_pipeline_held": True,
            "items": [{"filing_date": "2026-11-06"}],
        }
        res = self.sm_engine.assess_symbol("NVDA", live_sec=envelope)
        sec_sub = res["subdomains"]["sec_form_4"]
        assert sec_sub["coverage_certified"] is False
        assert sec_sub["status_label"] == "Coverage Uncertified"
        assert "Historical coverage not certified" in sec_sub["disclosure"]

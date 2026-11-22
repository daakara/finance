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
from api.routes.analytics import _build_smart_money_confluence_inputs


class TestSecInsiderFlowPresentation:
    """Authoritative test suite for decoupled disclosure availability and held scoring."""

    @classmethod
    def setup_class(cls):
        cls.sm_engine = SmartMoneyEngine()
        cls.conf_engine = ConfluenceEngine()

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
        assert "SEC filings available" in sec_sub["disclosure"]
        assert "16 filings" in sec_sub["disclosure"]
        assert "latest 2026-11-19" in sec_sub["disclosure"]
        assert "coverage: PARTIAL" in sec_sub["disclosure"]
        assert "Not assessed — ingestion-only trial." in sec_sub["disclosure"]

        # 2. Confluence inputs helper
        conf_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=mock_envelope)
        assert conf_inputs is not None, "Confluence inputs must provide evidence even when scoring is held"
        assert conf_inputs["scoring_held"] is True
        assert conf_inputs["filings_count"] == 16
        assert conf_inputs["coverage_status"] == "PARTIAL"
        assert conf_inputs["has_insider_buy"] is False
        assert conf_inputs["insider_buy_count_90d"] == 0

        # 3. Confluence Engine calculation
        conf_out = self.conf_engine.calculate_confluence(
            symbol="NVDA",
            technical_data={"current_price": 140.0, "rsi_14": 45.0, "stage_phase": "Stage 4"},
            smart_money_data=conf_inputs,
            fundamental_data=None,
            macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.2},
        )

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
        assert "SEC filings available (16 filings, latest 2026-11-19, coverage: PARTIAL)." in smart_p["detail"]
        assert "Not assessed — ingestion-only trial." in smart_p["detail"]
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
        assert conf_inputs["has_insider_buy"] is False
        assert conf_inputs["insider_buy_count_90d"] == 0

        conf_out = self.conf_engine.calculate_confluence(
            symbol="NVDA",
            technical_data={"current_price": 140.0, "rsi_14": 50.0},
            smart_money_data=conf_inputs,
        )
        smart_p = next(p for p in conf_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
        assert smart_p["score"] == 0.0
        assert "SEC filings available (2 filings, latest 2026-11-18, coverage: PARTIAL)." in smart_p["detail"]

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
        assert conf_inputs["is_incomplete_coverage"] is True
        assert conf_inputs["is_verified_empty"] is False

        conf_out = self.conf_engine.calculate_confluence(
            symbol="AAPL",
            smart_money_data=conf_inputs,
        )
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
        assert conf_inputs["is_verified_empty"] is True
        assert conf_inputs["is_incomplete_coverage"] is False

        conf_out = self.conf_engine.calculate_confluence(
            symbol="AAPL",
            smart_money_data=conf_inputs,
        )
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
        assert conf_inputs["is_repo_error"] is True

        conf_out = self.conf_engine.calculate_confluence(
            symbol="NVDA",
            smart_money_data=conf_inputs,
        )
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
        nvda_inputs = _build_smart_money_confluence_inputs("NVDA", live_sec=nvda_env)
        nvda_out = self.conf_engine.calculate_confluence("NVDA", smart_money_data=nvda_inputs)
        nvda_smart = next(p for p in nvda_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

        assert nvda_smart["filingsCount"] == 16
        assert "16 filings" in nvda_smart["detail"]
        assert "2026-11-19" in nvda_smart["detail"]

        # 2. Evaluate AAPL immediately after
        aapl_inputs = _build_smart_money_confluence_inputs("AAPL", live_sec=aapl_env)
        aapl_out = self.conf_engine.calculate_confluence("AAPL", smart_money_data=aapl_inputs)
        aapl_smart = next(p for p in aapl_out["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

        assert aapl_smart["filingsCount"] == 1, "AAPL must have 1 filing, not 16 from NVDA"
        assert "1 filings" in aapl_smart["detail"]
        assert "2026-11-05" in aapl_smart["detail"]
        assert "16 filings" not in aapl_smart["detail"], "NVDA 16 filings leaked into AAPL"

        # 3. Evaluate NVDA again to ensure idempotency
        nvda_inputs_2 = _build_smart_money_confluence_inputs("NVDA", live_sec=nvda_env)
        nvda_out_2 = self.conf_engine.calculate_confluence("NVDA", smart_money_data=nvda_inputs_2)
        nvda_smart_2 = next(p for p in nvda_out_2["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

        assert nvda_smart_2["filingsCount"] == 16
        assert "16 filings" in nvda_smart_2["detail"]

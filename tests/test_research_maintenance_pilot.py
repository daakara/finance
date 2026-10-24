"""
tests/test_research_maintenance_pilot.py

Comprehensive Test Suite for ARX Research-Maintenance Pilot under ARX Closure Model v2
and ARX-DECISION-RESEARCH-PROCESS-V1.

19 Acceptance Test Controls (Section F):
1. Valid baseline and draft calculations match independent formulas within tolerances.
2. Changed scenario assumptions affect modeled output.
3. Missing tax, reinvestment, or scenario assumptions suppress valuation.
4. Bear or bull baseline mismatch prevents verified comparison.
5. Solver failure or non-finite result prevents verified status.
6. Suppressed baseline remains suppressed.
7. Standalone quarterly margin < 13.0% invalidates the thesis while TTM margin is >= 13.0%.
8. Standalone quarterly margin >= 13.0% preserves the thesis while TTM margin is < 13.0%.
9. Missing quarterly revenue or EBIT leaves the invalidation condition UNRESOLVED.
10. Deteriorating inputs produce accurate negative narrative text.
11. Reference price above, below, and equal to base-case modeled value.
12. Net debt position describes debt without cash-cushion language.
13. Missing short-term investments that was included in the baseline bridge marks the bridge UNRESOLVED and suppresses valuation.
14. Explicit zero for short-term investments is handled correctly.
15. Missing cash or debt is displayed as UNRESOLVED, never $0.0M.
16. Actual cache refresh reveals a newer eligible filing when one exists.
17. Receipt hashing against known bytes produces matching SHA-256.
18. Missing receipt produces UNAVAILABLE without claiming verification.
19. Accepted public files remain bit-for-bit unchanged.

Additional Operational & Accounting Controls:
20. Contradictory debt total and components fail closed.
21. Source-backed multi-period TTM derivation.
22. Actual quarterly CFO and capex derivation.
23. Interrupted processing followed by successful retry.
24. Replay/live separation (replay does not consume live event).
25. Packet generation failure without completed state.
"""

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from analyst_dashboard.research.maintenance.contracts import (
    EventProcessingState,
    EventRegister,
    FactStatus,
    InterpretationType,
    MaintenanceEvent,
    ProposedDisposition,
    ReconciledFinancialChange,
    ReviewPacket,
    SourceReceipt,
)
from analyst_dashboard.research.maintenance.event_poller import SecEventPoller
from analyst_dashboard.research.maintenance.fact_reconciler import FactReconciler
from analyst_dashboard.research.maintenance.decision_assessor import (
    DecisionAssessor,
    run_fcff_model,
    solve_reverse_dcf,
)
from analyst_dashboard.research.maintenance.packet_generator import PacketGenerator
from analyst_dashboard.research.maintenance.service import ResearchMaintenanceService

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestResearchMaintenancePilot(unittest.TestCase):
    """Test suite for research maintenance lifecycle, accounting controls, and failure modes."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.register_path = Path(self.test_dir) / "event_register.json"
        self.packets_dir = Path(self.test_dir) / "packets"
        self.cache_dir = Path(self.test_dir) / "raw_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Baseline fixture for FIX (Comfort Systems USA) matching accepted public payload
        self.baseline_dossier = {
            "dossier_id": "OPP-FIX-20261016",
            "publication_metadata": {
                "filing_availability_cutoff_utc": "2026-10-16T20:00:00Z",
            },
            "identity": {
                "symbol": "FIX",
                "company_name": "Comfort Systems USA, Inc.",
                "cik": "0001035983",
            },
            "financial_profile": {
                "revenue": 11228024000.0,
                "revenue_display": "$11.23B",
                "revenue_period": "TTM (Ended 2026-06-30)",
                "operating_income": 1849300000.0,
                "operating_margin_pct": 16.47,
                "cash_and_equivalents": 1854794000.0,
                "short_term_investments": None,
                "restricted_cash": None,
                "total_financial_debt": 54064000.0,
                "debt_display": "$54.1M",
                "net_debt_bridge": -1800730000.0,
                "selected_share_basis": {
                    "value": 35254000.0,
                    "basis_type": "GAAP_DILUTED_WEIGHTED_AVERAGE",
                    "measurement_period": "2026-06-30",
                },
            },
            "central_assessment": {
                "disposition": "WATCH_CONDITIONS",
                "presentation_summary": "Comfort Systems USA presents an exceptional balance sheet ($1.80B net cash)...",
                "price_watch_condition": "Priced attractive for entry if share price pulls back to base-case modeled value range ($1,050 - $1,100).",
                "main_uncertainty": "Cyclicality of hyperscale cloud capex budgets and customer concentration in top-tier tech clients.",
            },
            "monitoring": {
                "price_watch_condition": "Priced attractive for entry if share price pulls back to base-case modeled value range ($1,050 - $1,100).",
                "observable_invalidation_trigger": "Backlog book-to-bill falling below 0.95x or quarterly operating margin dropping below 13.0%.",
            },
            "valuation": {
                "valuation_status": "COMPUTED_VERIFIED",
                "reference_price": 1439.07,
                "reference_date": "2026-10-16",
                "solved_5y_cagr_pct": 30.07,
                "reverse_dcf_parameters": {
                    "target_price": 1439.07,
                    "solved_revenue_cagr_pct": 30.07,
                    "operating_margin_pct": 15.0,
                    "wacc_pct": 9.5,
                    "terminal_growth_pct": 2.5,
                    "tax_rate_pct": 21.0,
                    "reinvestment_rate_pct": 15.0,
                    "base_revenue": 11228024000.0,
                    "current_shares": 35254000.0,
                    "net_debt_bridge": -1800730000.0,
                },
                "scenarios": {
                    "conservative": {
                        "revenue_cagr_pct": 12.0,
                        "operating_margin_pct": 12.5,
                        "wacc_pct": 10.5,
                        "terminal_growth_pct": 2.0,
                        "tax_rate_pct": 21.0,
                        "reinvestment_rate_pct": 15.0,
                        "per_share_value": 533.46,
                    },
                    "base": {
                        "revenue_cagr_pct": 22.0,
                        "operating_margin_pct": 15.0,
                        "wacc_pct": 9.5,
                        "terminal_growth_pct": 2.5,
                        "tax_rate_pct": 21.0,
                        "reinvestment_rate_pct": 15.0,
                        "per_share_value": 1081.99,
                    },
                    "optimistic": {
                        "revenue_cagr_pct": 30.0,
                        "operating_margin_pct": 16.5,
                        "wacc_pct": 8.5,
                        "terminal_growth_pct": 3.0,
                        "tax_rate_pct": 21.0,
                        "reinvestment_rate_pct": 15.0,
                        "per_share_value": 1996.38,
                    },
                },
            },
        }

        # Mock submissions file
        self.mock_submissions = {
            "name": "COMFORT SYSTEMS USA INC",
            "cik": "0001035983",
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K", "10-Q", "10-K"],
                    "filingDate": ["2026-10-22", "2026-10-22", "2026-07-23", "2026-02-19"],
                    "reportDate": ["2026-09-30", "2026-10-22", "2026-06-30", "2025-12-31"],
                    "acceptanceDateTime": [
                        "2026-10-23T00:12:01.000Z",
                        "2026-10-22T21:05:00.000Z",
                        "2026-07-23T20:15:00.000Z",
                        "2026-02-19T21:00:00.000Z",
                    ],
                    "accessionNumber": [
                        "0001104659-26-118159",
                        "0001104659-26-118156",
                        "0001104659-26-086258",
                        "0001104659-26-017530",
                    ],
                    "primaryDocument": [
                        "fix-20260930x10q.htm",
                        "fix-20261022x8k.htm",
                        "fix-20260630x10q.htm",
                        "fix-20251231x10k.htm",
                    ],
                }
            }
        }

        # Mock XBRL company facts with multi-period component history
        self.mock_company_facts = {
            "cik": 1035983,
            "entityName": "COMFORT SYSTEMS USA INC",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"start": "2025-01-01", "end": "2025-12-31", "val": 9101641000.0, "accn": "0001104659-26-017530", "form": "10-K"},
                                {"start": "2025-01-01", "end": "2025-09-30", "val": 6455574000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                                {"start": "2026-01-01", "end": "2026-09-30", "val": 9897290000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                                {"start": "2026-07-01", "end": "2026-09-30", "val": 3766302000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                            ]
                        }
                    },
                    "OperatingIncomeLoss": {
                        "units": {
                            "USD": [
                                {"start": "2025-01-01", "end": "2025-12-31", "val": 1314589000.0, "accn": "0001104659-26-017530", "form": "10-K"},
                                {"start": "2025-01-01", "end": "2025-09-30", "val": 887845000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                                {"start": "2026-01-01", "end": "2026-09-30", "val": 1668254000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                                {"start": "2026-07-01", "end": "2026-09-30", "val": 624570000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {"end": "2026-09-30", "val": 2158027000.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                            ]
                        }
                    },
                    "LongTermDebtCurrent": {
                        "units": {
                            "USD": [
                                {"end": "2026-09-30", "val": 5710000.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                            ]
                        }
                    },
                    "LongTermDebtNoncurrent": {
                        "units": {
                            "USD": [
                                {"end": "2026-09-30", "val": 43845000.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                            ]
                        }
                    },
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                {"end": "2026-09-30", "val": 49555000.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                            ]
                        }
                    },
                    "OperatingLeaseLiabilityNoncurrent": {
                        "units": {
                            "USD": [
                                {"end": "2026-09-30", "val": 324878000.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                            ]
                        }
                    },
                    "WeightedAverageNumberOfDilutedSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"start": "2026-07-01", "end": "2026-09-30", "val": 35152000.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {"start": "2026-01-01", "end": "2026-06-30", "val": 1528254000.0, "accn": "0001104659-26-086258", "form": "10-Q"},
                                {"start": "2026-01-01", "end": "2026-09-30", "val": 2538380000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {"start": "2026-01-01", "end": "2026-06-30", "val": 288837000.0, "accn": "0001104659-26-086258", "form": "10-Q"},
                                {"start": "2026-01-01", "end": "2026-09-30", "val": 499437000.0, "accn": "0001104659-26-118159", "form": "10-Q"},
                            ]
                        }
                    },
                }
            }
        }

        # Write cache files to allow offline test execution
        sub_cache = self.cache_dir / "CIK0001035983_submissions.json"
        with open(sub_cache, "w", encoding="utf-8") as f:
            json.dump(self.mock_submissions, f)

        facts_cache = self.cache_dir / "CIK0001035983_companyfacts.json"
        with open(facts_cache, "w", encoding="utf-8") as f:
            json.dump(self.mock_company_facts, f)

        # Mock snapshot file
        self.mock_snapshot_path = Path(self.test_dir) / "opportunityDossierSnapshot.json"
        with open(self.mock_snapshot_path, "w", encoding="utf-8") as f:
            json.dump({"dossiers": {"FIX": self.baseline_dossier}}, f)

        self.service = ResearchMaintenanceService(
            register_path=self.register_path,
            packets_dir=self.packets_dir,
            cache_dir=self.cache_dir,
            baseline_snapshot_path=self.mock_snapshot_path,
            offline_mode=True,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # SECTION F ACCEPTANCE CONTROLS (1 - 19)
    # =========================================================================

    def test_01_valid_baseline_and_draft_calculations(self):
        """1. Valid baseline and draft calculations match independent formulas within tolerances."""
        packet, event, warnings = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
        self.assertIsNotNone(event)
        self.assertIsNotNone(packet)

        val = packet.valuation_comparison
        self.assertEqual(val["valuation_machinery_status"], "COMPUTED_VERIFIED")

        # Baseline recomputed CAGR matches accepted within 0.05
        base_cagr = val["solved_5y_cagr"]["baseline_cagr_pct"]
        self.assertAlmostEqual(base_cagr, 30.07, delta=0.05)

        # Baseline bear, base, bull scenarios match within $0.50
        sc = val["scenarios"]
        self.assertAlmostEqual(sc["conservative"]["baseline_per_share"], 533.46, delta=0.50)
        self.assertAlmostEqual(sc["base"]["baseline_per_share"], 1081.99, delta=0.50)
        self.assertAlmostEqual(sc["optimistic"]["baseline_per_share"], 1996.38, delta=0.50)

        # Draft calculations match independent values
        draft_cagr = val["solved_5y_cagr"]["draft_cagr_pct"]
        self.assertAlmostEqual(draft_cagr, 26.76, delta=0.10)
        self.assertAlmostEqual(sc["conservative"]["draft_per_share"], 600.44, delta=1.0)
        self.assertAlmostEqual(sc["base"]["draft_per_share"], 1215.00, delta=1.0)
        self.assertAlmostEqual(sc["optimistic"]["draft_per_share"], 2239.47, delta=1.0)

    def test_02_changed_scenario_assumptions_affect_outputs(self):
        """2. Changed scenario assumptions affect modeled output."""
        modified_baseline = copy.deepcopy(self.baseline_dossier)
        # Increase base scenario operating margin from 15.0% to 18.0%
        modified_baseline["valuation"]["scenarios"]["base"]["operating_margin_pct"] = 18.0
        # Recompute expected baseline per_share for 18.0% margin:
        # Base rev $11.228B, shares 35.254M, net debt -$1.800B, growth 22%, margin 18%, wacc 9.5%, g 2.5%, tax 21%, reinvest 15%
        res = run_fcff_model(11228024000.0, 35254000.0, -1800730000.0, 22.0, 18.0, 9.5, 2.5, 21.0, 15.0)
        modified_baseline["valuation"]["scenarios"]["base"]["per_share_value"] = round(res["per_share"], 2)

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=modified_baseline["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        assessor = DecisionAssessor()
        _, val_comp, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=modified_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )

        self.assertEqual(val_comp["valuation_machinery_status"], "COMPUTED_VERIFIED")
        # With 18% margin, draft per_share is substantially higher than standard 15% margin ($1,215.00)
        new_draft_base = val_comp["scenarios"]["base"]["draft_per_share"]
        self.assertGreater(new_draft_base, 1400.0)

    def test_03_missing_tax_reinvestment_or_scenario_assumptions_suppress(self):
        """3. Missing tax, reinvestment, or scenario assumptions suppress valuation."""
        assessor = DecisionAssessor()
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        # Case A: Missing tax rate
        corrupted_a = copy.deepcopy(self.baseline_dossier)
        del corrupted_a["valuation"]["reverse_dcf_parameters"]["tax_rate_pct"]
        _, val_a, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=corrupted_a, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertTrue(val_a["valuation_machinery_status"].startswith("SUPPRESSED_"))

        # Case B: Missing reinvestment rate
        corrupted_b = copy.deepcopy(self.baseline_dossier)
        del corrupted_b["valuation"]["reverse_dcf_parameters"]["reinvestment_rate_pct"]
        _, val_b, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=corrupted_b, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertTrue(val_b["valuation_machinery_status"].startswith("SUPPRESSED_"))

        # Case C: Missing scenario (e.g. optimistic scenario removed)
        corrupted_c = copy.deepcopy(self.baseline_dossier)
        del corrupted_c["valuation"]["scenarios"]["optimistic"]
        _, val_c, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=corrupted_c, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertTrue(val_c["valuation_machinery_status"].startswith("SUPPRESSED_"))

    def test_04_bear_or_bull_baseline_mismatch_prevents_verified_comparison(self):
        """4. Bear or bull baseline mismatch prevents verified comparison."""
        assessor = DecisionAssessor()
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        # Bear mismatch: conservative per_share_value altered from 533.46 to 750.00
        bear_mismatch = copy.deepcopy(self.baseline_dossier)
        bear_mismatch["valuation"]["scenarios"]["conservative"]["per_share_value"] = 750.00
        _, val_bear, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=bear_mismatch, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertEqual(val_bear["valuation_machinery_status"], "SUPPRESSED_BASELINE_UNRECONCILED")

        # Bull mismatch: optimistic per_share_value altered from 1996.38 to 2500.00
        bull_mismatch = copy.deepcopy(self.baseline_dossier)
        bull_mismatch["valuation"]["scenarios"]["optimistic"]["per_share_value"] = 2500.00
        _, val_bull, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=bull_mismatch, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertEqual(val_bull["valuation_machinery_status"], "SUPPRESSED_BASELINE_UNRECONCILED")

    def test_05_solver_failure_or_non_finite_prevents_verified_status(self):
        """5. Solver failure or non-finite result prevents verified status."""
        assessor = DecisionAssessor()
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        # Solver failure: Target price astronomically high so reverse DCF solver cannot converge
        unsolvable_baseline = copy.deepcopy(self.baseline_dossier)
        unsolvable_baseline["valuation"]["reference_price"] = 10000000.0  # $10M per share
        _, val_fail, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=unsolvable_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertEqual(val_fail["valuation_machinery_status"], "SUPPRESSED_SOLVER_FAILURE")

        # Non-finite scenario: WACC <= terminal growth leads to zero/negative denominator
        non_finite_baseline = copy.deepcopy(self.baseline_dossier)
        non_finite_baseline["valuation"]["scenarios"]["base"]["wacc_pct"] = 2.0
        non_finite_baseline["valuation"]["scenarios"]["base"]["terminal_growth_pct"] = 2.5
        _, val_inf, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=non_finite_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=list(exceptions), undisclosed_info=[],
        )
        self.assertEqual(val_inf["valuation_machinery_status"], "SUPPRESSED_SOLVER_FAILURE")

    def test_06_suppressed_baseline_remains_suppressed(self):
        """6. Suppressed baseline remains suppressed and propagates UNRESOLVED disposition."""
        suppressed_baseline = copy.deepcopy(self.baseline_dossier)
        suppressed_baseline["valuation"]["valuation_status"] = "SUPPRESSED_INPUT_UNRESOLVED"

        assessor = DecisionAssessor()
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=suppressed_baseline["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        _, val_comp, disposition, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=suppressed_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        self.assertEqual(val_comp["valuation_machinery_status"], "SUPPRESSED_BASELINE_UNVERIFIED")
        self.assertEqual(disposition, ProposedDisposition.UNRESOLVED)

    def test_07_quarterly_margin_below_13pct_invalidates_while_ttm_above(self):
        """7. Standalone quarterly margin < 13.0% invalidates the thesis while TTM margin is >= 13.0%."""
        q_low_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        q_low_profile["quarterly_revenue"] = 3766302000.0
        q_low_profile["quarterly_operating_income"] = 376630200.0  # 10.0% quarterly margin (< 13.0%)
        q_low_profile["quarterly_operating_margin_pct"] = 10.0
        q_low_profile["revenue"] = 12543357000.0
        q_low_profile["operating_income"] = 2000000000.0  # TTM margin = 15.94% (>= 13.0%)
        q_low_profile["operating_margin_pct"] = 15.94

        assessor = DecisionAssessor()
        effects, _, disposition, rationale, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=q_low_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        inval_effect = next(e for e in effects if e.conclusion_key == "invalidation_condition")
        self.assertTrue("BREACHED" in inval_effect.draft_state or "TRIGGERED" in inval_effect.draft_state)
        self.assertIn("10.00%", inval_effect.draft_state)
        self.assertEqual(disposition, ProposedDisposition.INVALIDATE)

        # Decisive reason must challenge thesis and require owner review
        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        self.assertNotIn("Core decisive thesis is maintained", decisive_effect.draft_state)
        self.assertIn("INVALIDATION WARNING", decisive_effect.draft_state)
        self.assertTrue(decisive_effect.owner_judgment_required)
        self.assertNotIn("No thesis invalidation occurred", rationale)

    def test_08_quarterly_margin_above_13pct_safe_while_ttm_below(self):
        """8. Standalone quarterly margin >= 13.0% preserves the thesis while TTM margin is < 13.0%."""
        q_safe_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        q_safe_profile["quarterly_revenue"] = 3766302000.0
        q_safe_profile["quarterly_operating_income"] = 624570000.0  # 16.58% quarterly margin (>= 13.0%)
        q_safe_profile["quarterly_operating_margin_pct"] = 16.58
        q_safe_profile["revenue"] = 12543357000.0
        q_safe_profile["operating_income"] = 1400000000.0  # TTM margin = 11.16% (< 13.0%)
        q_safe_profile["operating_margin_pct"] = 11.16

        assessor = DecisionAssessor()
        effects, _, disposition, rationale, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=q_safe_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        inval_effect = next(e for e in effects if e.conclusion_key == "invalidation_condition")
        self.assertNotIn("BREACHED", inval_effect.draft_state)
        self.assertIn("16.58%", inval_effect.draft_state)
        self.assertIn("did not breach", inval_effect.draft_state)
        self.assertNotEqual(disposition, ProposedDisposition.INVALIDATE)

        # Decisive reason must state conditional maintenance with owner judgment
        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        self.assertNotIn("No thesis invalidation occurred", decisive_effect.draft_state)
        self.assertIn("conditionally maintained pending Product Owner review", decisive_effect.draft_state)
        self.assertTrue(decisive_effect.owner_judgment_required)
        self.assertNotIn("No thesis invalidation occurred", rationale)

    def test_09_missing_quarterly_margin_produces_unresolved_condition(self):
        """9. Missing quarterly revenue or EBIT leaves the invalidation condition UNRESOLVED."""
        missing_q_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        missing_q_profile["quarterly_revenue"] = None
        missing_q_profile["quarterly_operating_income"] = None
        missing_q_profile["quarterly_operating_margin_pct"] = None

        assessor = DecisionAssessor()
        effects, _, disposition, rationale, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=missing_q_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        inval_effect = next(e for e in effects if e.conclusion_key == "invalidation_condition")
        self.assertIn("UNRESOLVED", inval_effect.draft_state)
        self.assertEqual(inval_effect.interpretation_type, InterpretationType.UNRESOLVED_EVIDENCE)
        self.assertEqual(disposition, ProposedDisposition.UNRESOLVED)

        # Decisive reason must also remain unresolved and require owner review
        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        self.assertNotIn("Core decisive thesis is maintained", decisive_effect.draft_state)
        self.assertIn("UNRESOLVED", decisive_effect.draft_state)
        self.assertTrue(decisive_effect.owner_judgment_required)
        self.assertNotIn("No thesis invalidation occurred", rationale)

    def test_10_deteriorating_inputs_produce_accurate_negative_text(self):
        """10. Deteriorating inputs produce accurate negative narrative text."""
        deteriorated_facts = copy.deepcopy(self.mock_company_facts)
        # Revenue and operating income decline sharply
        deteriorated_facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][2]["val"] = 5000000000.0
        deteriorated_facts["facts"]["us-gaap"]["OperatingIncomeLoss"]["units"]["USD"][2]["val"] = 500000000.0

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=deteriorated_facts,
        )

        assessor = DecisionAssessor()
        effects, val_comp, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )

        # Modeled base value must fall
        self.assertLess(val_comp["scenarios"]["base"]["draft_per_share"], val_comp["scenarios"]["base"]["baseline_per_share"])
        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        self.assertEqual(decisive_effect.materiality, "INCREMENTAL_NEGATIVE")
        self.assertIn("contracted", decisive_effect.draft_state.lower())

    def test_11_reference_price_below_above_and_equal_base_value(self):
        """11. Reference price above, below, and equal to base-case modeled value."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        assessor = DecisionAssessor()

        # Case A: Reference price ($1,439.07) > draft base value ($1,215.00) -> Premium / above
        effects_a, _, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        decisive_a = next(e for e in effects_a if e.conclusion_key == "decisive_reason")
        self.assertIn("remains above base intrinsic value", decisive_a.draft_state)

        # Case B: Reference price ($1,000.00) < draft base value ($1,215.00) -> Discount / below
        discount_baseline = copy.deepcopy(self.baseline_dossier)
        discount_baseline["valuation"]["reference_price"] = 1000.00
        discount_baseline["valuation"]["reverse_dcf_parameters"]["target_price"] = 1000.00
        recomp_cagr, _ = solve_reverse_dcf(1000.00, 11228024000.0, 35254000.0, -1800730000.0, 15.0, 9.5, 2.5, 21.0, 15.0)
        discount_baseline["valuation"]["solved_5y_cagr_pct"] = round(recomp_cagr, 2)
        discount_baseline["valuation"]["reverse_dcf_parameters"]["solved_revenue_cagr_pct"] = round(recomp_cagr, 2)

        effects_b, _, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=discount_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        decisive_b = next(e for e in effects_b if e.conclusion_key == "decisive_reason")
        self.assertIn("is below base intrinsic value", decisive_b.draft_state)

        # Case C: Reference price ($1,215.00) == draft base value ($1,215.00) -> Equal
        equal_baseline = copy.deepcopy(self.baseline_dossier)
        equal_baseline["valuation"]["reference_price"] = 1215.00
        equal_baseline["valuation"]["reverse_dcf_parameters"]["target_price"] = 1215.00
        recomp_cagr_eq, _ = solve_reverse_dcf(1215.00, 11228024000.0, 35254000.0, -1800730000.0, 15.0, 9.5, 2.5, 21.0, 15.0)
        equal_baseline["valuation"]["solved_5y_cagr_pct"] = round(recomp_cagr_eq, 2)
        equal_baseline["valuation"]["reverse_dcf_parameters"]["solved_revenue_cagr_pct"] = round(recomp_cagr_eq, 2)

        effects_c, _, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=equal_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        decisive_c = next(e for e in effects_c if e.conclusion_key == "decisive_reason")
        self.assertIn("equals base intrinsic value", decisive_c.draft_state)

    def test_12_net_debt_position_without_cash_cushion_language(self):
        """12. Net debt position describes debt without cash-cushion language."""
        net_debt_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        net_debt_profile["cash_and_equivalents"] = 200000000.0
        net_debt_profile["total_financial_debt"] = 1000000000.0
        net_debt_profile["net_debt_bridge"] = 800000000.0  # +$800M Net Debt

        assessor = DecisionAssessor()
        effects, _, _, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=net_debt_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        self.assertIn("net debt of $800.0M", decisive_effect.draft_state)
        self.assertNotIn("cash cushion", decisive_effect.draft_state.lower())

    def test_13_missing_included_investments_suppresses_bridge(self):
        """13. Missing short-term investments that was included in the baseline bridge marks the bridge UNRESOLVED and suppresses valuation."""
        baseline_with_sti = copy.deepcopy(self.baseline_dossier)
        baseline_with_sti["financial_profile"]["short_term_investments"] = 150000000.0

        # Facts omit ShortTermInvestments concept
        facts_without_sti = copy.deepcopy(self.mock_company_facts)

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=baseline_with_sti["financial_profile"],
            facts_data=facts_without_sti,
        )

        self.assertIsNone(draft_profile.get("net_debt_bridge"))
        self.assertTrue(any("Short-term investments was an included component" in e for e in exceptions))

        assessor = DecisionAssessor()
        _, val_comp, disposition, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=baseline_with_sti, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        self.assertEqual(val_comp["valuation_machinery_status"], "SUPPRESSED_INPUT_UNRESOLVED")
        self.assertEqual(disposition, ProposedDisposition.UNRESOLVED)

    def test_14_explicit_zero_investment_handled_correctly(self):
        """14. Explicit zero for short-term investments is handled correctly."""
        baseline_with_sti = copy.deepcopy(self.baseline_dossier)
        baseline_with_sti["financial_profile"]["short_term_investments"] = 150000000.0

        facts_with_zero_sti = copy.deepcopy(self.mock_company_facts)
        facts_with_zero_sti["facts"]["us-gaap"]["ShortTermInvestments"] = {
            "units": {
                "USD": [
                    {"end": "2026-09-30", "val": 0.0, "accn": "0001104659-26-118159", "form": "10-Q"}
                ]
            }
        }

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=baseline_with_sti["financial_profile"],
            facts_data=facts_with_zero_sti,
        )

        # Explicit zero is recognized, bridge succeeds: $49.555M debt - ($2158.027M cash + $0.0M sti) = -$2108.472M
        self.assertIsNotNone(draft_profile.get("net_debt_bridge"))
        self.assertAlmostEqual(draft_profile["net_debt_bridge"], 49555000.0 - 2158027000.0, delta=1.0)

    def test_15_missing_cash_or_debt_displayed_as_unresolved_never_zero(self):
        """15. Missing cash or debt is displayed as UNRESOLVED, never $0.0M."""
        missing_facts = copy.deepcopy(self.mock_company_facts)
        del missing_facts["facts"]["us-gaap"]["CashAndCashEquivalentsAtCarryingValue"]
        del missing_facts["facts"]["us-gaap"]["LongTermDebt"]
        del missing_facts["facts"]["us-gaap"]["LongTermDebtCurrent"]
        del missing_facts["facts"]["us-gaap"]["LongTermDebtNoncurrent"]

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=missing_facts,
        )

        cash_c = next(c for c in changes if c.metric_key == "cash_and_equivalents")
        debt_c = next(c for c in changes if c.metric_key == "total_financial_debt")

        self.assertEqual(cash_c.draft_display, "UNRESOLVED")
        self.assertNotEqual(cash_c.draft_display, "$0.0M")
        self.assertEqual(debt_c.draft_display, "UNRESOLVED")
        self.assertNotEqual(debt_c.draft_display, "$0.0M")

        # Test in rendered packet markdown
        gen = PacketGenerator(output_dir=self.packets_dir)
        packet = ReviewPacket(
            packet_id="TEST-PACKET-MISSING",
            symbol="FIX",
            cik="0001035983",
            company_name="Comfort Systems USA, Inc.",
            accepted_baseline_id="OPP-FIX-20261016",
            accepted_baseline_cutoff="2026-10-16",
            draft_successor_id="OPP-FIX-20260930-DRAFT",
            created_at_utc="2026-10-23T00:00:00Z",
            filing_metadata={"form": "10-Q", "filing_date": "2026-10-22", "reporting_period": "2026-09-30"},
            proposed_disposition=ProposedDisposition.UNRESOLVED,
            disposition_rationale="Suppressed due to missing balance sheet inputs.",
            material_financial_changes=changes,
            assessment_effects=[],
            valuation_comparison={"valuation_machinery_status": "SUPPRESSED_INPUT_UNRESOLVED"},
            accounting_exceptions=exceptions,
            undisclosed_information=[],
            questions_requiring_owner_judgment=[],
            stopping_rule="Cease investigation.",
            is_replay=False,
        )
        md = gen.render_markdown(packet)
        self.assertIn("| **Cash & Cash Equivalents** | $1854.8M | UNRESOLVED |", md)
        self.assertIn("| **Total Financial Debt** | $54.1M | UNRESOLVED |", md)
        self.assertNotIn("| **Cash & Cash Equivalents** | $1854.8M | $0.0M |", md)
        self.assertNotIn("| **Total Financial Debt** | $54.1M | $0.0M |", md)

    @patch("urllib.request.urlopen")
    def test_16_actual_cache_refresh_reveals_newer_eligible_filing(self, mock_urlopen):
        """16. Actual cache refresh reveals a newer eligible filing via HTTP request and retains exact response bytes."""
        reg = EventRegister(self.register_path)
        poller = SecEventPoller(register=reg, cache_dir=self.cache_dir, offline_mode=False, cache_ttl_sec=3600)

        # 1. Start with an existing cached submissions, index, and primary doc response
        sub_cache = self.cache_dir / "CIK0001035983_submissions.json"
        initial_bytes = json.dumps(self.mock_submissions).encode("utf-8")
        sub_cache.write_bytes(initial_bytes)
        sub_cache.with_name(sub_cache.name + ".meta.json").write_text(json.dumps({
            "url": "https://data.sec.gov/submissions/CIK0001035983.json",
            "remote_retrieval_time_utc": "2026-10-23T00:00:00Z",
            "byte_count": len(initial_bytes),
            "sha256": hashlib.sha256(initial_bytes).hexdigest(),
            "content_type": "application/json",
            "byte_representation": "decoded_http_response_bytes",
        }))

        idx_cache = self.cache_dir / "CIK0001035983_000110465926118159_index.htm"
        idx_bytes = b"<html>initial index</html>"
        idx_cache.write_bytes(idx_bytes)
        idx_cache.with_name(idx_cache.name + ".meta.json").write_text(json.dumps({
            "url": "https://www.sec.gov/index.htm", "remote_retrieval_time_utc": "2026-10-23T00:00:00Z",
            "byte_count": len(idx_bytes), "sha256": hashlib.sha256(idx_bytes).hexdigest(),
            "content_type": "text/html", "byte_representation": "decoded_http_response_bytes"
        }))

        doc_cache = self.cache_dir / "CIK0001035983_000110465926118159_fix-20260930x10q.htm"
        doc_bytes = b"<html>initial doc</html>"
        doc_cache.write_bytes(doc_bytes)
        doc_cache.with_name(doc_cache.name + ".meta.json").write_text(json.dumps({
            "url": "https://www.sec.gov/doc.htm", "remote_retrieval_time_utc": "2026-10-23T00:00:00Z",
            "byte_count": len(doc_bytes), "sha256": hashlib.sha256(doc_bytes).hexdigest(),
            "content_type": "text/html", "byte_representation": "decoded_http_response_bytes"
        }))

        # Initial discovery from cache (HTTP request must NOT occur)
        event1, _ = poller.discover_filing_event("FIX", "0001035983", "Comfort Systems USA, Inc.", cutoff_date="2026-07-01")
        self.assertIsNotNone(event1)
        self.assertEqual(event1.accession_number, "0001104659-26-118159")
        self.assertFalse(mock_urlopen.called)
        self.assertEqual(poller.last_submissions_receipt["retrieval_mode"], "CACHED")

        # 2. Provide a mocked newer HTTP response containing newly eligible filing
        refreshed_submissions = copy.deepcopy(self.mock_submissions)
        refreshed_submissions["filings"]["recent"]["form"].insert(0, "10-Q")
        refreshed_submissions["filings"]["recent"]["filingDate"].insert(0, "2026-11-20")
        refreshed_submissions["filings"]["recent"]["reportDate"].insert(0, "2026-11-15")
        refreshed_submissions["filings"]["recent"]["acceptanceDateTime"].insert(0, "2026-11-20T21:00:00.000Z")
        refreshed_submissions["filings"]["recent"]["accessionNumber"].insert(0, "0001104659-26-999999")
        refreshed_submissions["filings"]["recent"]["primaryDocument"].insert(0, "fix-20261115x10q.htm")

        newer_response_bytes = json.dumps(refreshed_submissions, indent=3).encode("utf-8")

        def mock_urlopen_fn(req, timeout=15):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            resp = MagicMock()
            resp.info.return_value.get.return_value = None
            resp.__enter__.return_value = resp
            if "submissions" in url:
                resp.read.return_value = newer_response_bytes
            elif "index" in url:
                resp.read.return_value = b"<html>newer index</html>"
            else:
                resp.read.return_value = b"<html>newer doc</html>"
            return resp

        mock_urlopen.side_effect = mock_urlopen_fn

        # 3. Request explicit refresh (force_refresh=True)
        event2, _ = poller.discover_filing_event(
            "FIX", "0001035983", "Comfort Systems USA, Inc.",
            cutoff_date="2026-07-01", force_refresh=True
        )

        # 4. Assert that an HTTP request occurred
        self.assertTrue(mock_urlopen.called)

        # 5. Assert that newly eligible filing was discovered
        self.assertIsNotNone(event2)
        self.assertEqual(event2.accession_number, "0001104659-26-999999")

        # 6. Assert that retained artifact on disk contains the exact response bytes used
        retained_disk_bytes = sub_cache.read_bytes()
        self.assertEqual(retained_disk_bytes, newer_response_bytes)
        self.assertEqual(len(retained_disk_bytes), len(newer_response_bytes))
        self.assertEqual(hashlib.sha256(retained_disk_bytes).hexdigest(), hashlib.sha256(newer_response_bytes).hexdigest())

        # 7. Assert correct LIVE provenance on receipt
        receipt = poller.last_submissions_receipt
        self.assertEqual(receipt["retrieval_mode"], "LIVE")
        self.assertEqual(receipt["byte_count"], len(newer_response_bytes))
        self.assertEqual(receipt["sha256"], hashlib.sha256(newer_response_bytes).hexdigest())

    @patch("urllib.request.urlopen")
    def test_17_receipt_hashing_against_known_bytes(self, mock_urlopen):
        """17. Receipt hashing against known bytes exercises fetch, disk retention, byte equality, and cached reads."""
        reg = EventRegister(self.register_path)
        poller = SecEventPoller(register=reg, cache_dir=self.cache_dir, offline_mode=False)
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=False)

        # 1. Return known JSON bytes with distinctive whitespace
        distinctive_bytes = b'{\n  "distinctive_header":   "test_whitespace_bytes_4981",\n  "count":   42\n}\n'
        mock_resp = MagicMock()
        mock_resp.read.return_value = distinctive_bytes
        mock_resp.info.return_value.get.return_value = None
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        # 2. Allow implementation to fetch, retain, and create receipt for submissions
        data = poller.fetch_submissions("0001035983", force_refresh=True)
        self.assertEqual(data["distinctive_header"], "test_whitespace_bytes_4981")
        receipt = poller.last_submissions_receipt
        self.assertIsNotNone(receipt)

        # 3. Independently read the retained file from disk
        retained_file = self.cache_dir / "CIK0001035983_submissions.json"
        self.assertTrue(retained_file.exists())
        disk_bytes = retained_file.read_bytes()

        # 4. Assert exact byte equality, byte count, and SHA-256 equality
        self.assertEqual(disk_bytes, distinctive_bytes)
        self.assertEqual(len(disk_bytes), len(distinctive_bytes))
        self.assertEqual(len(disk_bytes), receipt["byte_count"])
        expected_sha = hashlib.sha256(distinctive_bytes).hexdigest()
        self.assertEqual(hashlib.sha256(disk_bytes).hexdigest(), expected_sha)
        self.assertEqual(receipt["sha256"], expected_sha)
        self.assertEqual(receipt["retrieval_mode"], "LIVE")
        self.assertEqual(receipt["byte_representation"], "decoded_http_response_bytes")

        # 5. Exercise a subsequent cached read and verify identity and provenance
        mock_urlopen.reset_mock()
        cached_data = poller.fetch_submissions("0001035983", force_refresh=False)
        self.assertEqual(cached_data["distinctive_header"], "test_whitespace_bytes_4981")
        self.assertFalse(mock_urlopen.called)
        cached_receipt = poller.last_submissions_receipt
        self.assertEqual(cached_receipt["retrieval_mode"], "CACHED")
        self.assertEqual(cached_receipt["sha256"], expected_sha)
        self.assertEqual(cached_receipt["byte_count"], len(distinctive_bytes))
        self.assertIsNotNone(cached_receipt["remote_retrieval_time_utc"])
        self.assertIsNotNone(cached_receipt["cache_read_time_utc"])

        # 6. Test company facts retention and receipt matching
        facts_distinctive_bytes = b'{\n  "facts":   {\n    "us-gaap":   {}\n  }\n}\n'
        mock_facts_resp = MagicMock()
        mock_facts_resp.read.return_value = facts_distinctive_bytes
        mock_facts_resp.info.return_value.get.return_value = None
        mock_facts_resp.__enter__.return_value = mock_facts_resp
        mock_urlopen.return_value = mock_facts_resp

        facts_data = reconciler.fetch_company_facts("0001035983", force_refresh=True)
        facts_receipt = reconciler.last_facts_receipt
        retained_facts_file = self.cache_dir / "CIK0001035983_companyfacts.json"
        self.assertTrue(retained_facts_file.exists())
        self.assertEqual(retained_facts_file.read_bytes(), facts_distinctive_bytes)
        self.assertEqual(facts_receipt["byte_count"], len(facts_distinctive_bytes))
        self.assertEqual(facts_receipt["sha256"], hashlib.sha256(facts_distinctive_bytes).hexdigest())

        # 7. Failure case for tampered / inconsistent retained evidence
        retained_file.write_bytes(b'{"tampered": true}')
        tampered_data = poller.fetch_submissions("0001035983", force_refresh=False)
        tampered_receipt = poller.last_submissions_receipt
        self.assertIn("mismatch", tampered_receipt["limitation"])
        self.assertNotEqual(tampered_receipt["sha256"], expected_sha)

    def test_18_missing_receipt_produces_unavailable_without_verification_claim(self):
        """18. Missing receipt produces UNAVAILABLE without claiming verification."""
        gen = PacketGenerator(output_dir=self.packets_dir)
        packet = ReviewPacket(
            packet_id="TEST-PACKET-NO-RECEIPT",
            symbol="FIX",
            cik="0001035983",
            company_name="Comfort Systems USA, Inc.",
            accepted_baseline_id="OPP-FIX-20261016",
            accepted_baseline_cutoff="2026-10-16",
            draft_successor_id="OPP-FIX-20260930-DRAFT",
            created_at_utc="2026-10-23T00:00:00Z",
            filing_metadata={
                "form": "10-Q",
                "filing_date": "2026-10-22",
                "reporting_period": "2026-09-30",
                "index_sha256": None,
                "sha256": None,
            },
            proposed_disposition=ProposedDisposition.MAINTAIN,
            disposition_rationale="Maintenance rationale.",
            material_financial_changes=[],
            assessment_effects=[],
            valuation_comparison={"valuation_machinery_status": "COMPUTED_VERIFIED"},
            accounting_exceptions=[],
            undisclosed_information=[],
            questions_requiring_owner_judgment=[],
            stopping_rule="Cease investigation.",
            is_replay=False,
        )
        md = gen.render_markdown(packet)
        self.assertIn("UNAVAILABLE (not retrieved or unhashed in bounded session)", md)
        self.assertNotIn("Verified via EDGAR index", md)

    def test_19_accepted_public_files_remain_unchanged(self):
        """19. Accepted public files (snapshot and manifest) remain bit-for-bit unchanged."""
        prod_snapshot = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
        prod_manifest = REPO_ROOT / "frontend/lib/publishedWatchManifest.json"

        with open(prod_snapshot, "rb") as f:
            hash_snap_before = hashlib.sha256(f.read()).hexdigest()
        with open(prod_manifest, "rb") as f:
            hash_man_before = hashlib.sha256(f.read()).hexdigest()

        # Run pipeline
        self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")

        with open(prod_snapshot, "rb") as f:
            hash_snap_after = hashlib.sha256(f.read()).hexdigest()
        with open(prod_manifest, "rb") as f:
            hash_man_after = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(hash_snap_before, hash_snap_after)
        self.assertEqual(hash_man_before, hash_man_after)

    # =========================================================================
    # ADDITIONAL OPERATIONAL & ACCOUNTING CONTROLS (20 - 25)
    # =========================================================================

    def test_20_contradictory_debt_total_and_components(self):
        """20. Contradictory debt total and components: Conflicts create validation exception and suppress output."""
        tampered_facts = copy.deepcopy(self.mock_company_facts)
        # Components sum to $49.555M, but reported total is altered to $999M
        tampered_facts["facts"]["us-gaap"]["LongTermDebt"]["units"]["USD"][0]["val"] = 999000000.0

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=tampered_facts,
        )

        debt_change = next(c for c in changes if c.metric_key == "total_financial_debt")
        self.assertEqual(debt_change.status, FactStatus.VALIDATION_CONFLICT)
        self.assertIsNone(debt_change.draft_value)
        self.assertIsNone(draft_profile.get("total_financial_debt"))
        self.assertIsNone(draft_profile.get("net_debt_bridge"))
        self.assertTrue(any("Validation Conflict: Reported LongTermDebt" in e for e in exceptions))

    def test_21_source_backed_ttm_derivation(self):
        """21. Source-backed TTM revenue and operating income derivation preserves component provenance."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        _, changes, _, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        rev_change = next(c for c in changes if c.metric_key == "revenue")
        # $9,101,641,000 (FY25) - $6,455,574,000 (9M25) + $9,897,290,000 (9M26) = $12,543,357,000
        self.assertEqual(rev_change.draft_value, 12543357000.0)
        self.assertEqual(rev_change.status, FactStatus.DERIVED)
        self.assertTrue(len(rev_change.source_accessions) >= 2)
        self.assertIn("0001104659-26-017530", rev_change.source_accessions)
        self.assertIn("0001104659-26-118159", rev_change.source_accessions)

        ebit_change = next(c for c in changes if c.metric_key == "operating_income")
        # $1,314,589,000 (FY25) - $887,845,000 (9M25) + $1,668,254,000 (9M26) = $2,094,998,000
        self.assertEqual(ebit_change.draft_value, 2094998000.0)
        self.assertEqual(ebit_change.status, FactStatus.DERIVED)
        self.assertTrue(len(ebit_change.source_accessions) >= 2)

    def test_22_actual_quarterly_cfo_and_capex_derivation(self):
        """22. Standalone quarterly CFO and capex derivation from YTD components."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, _, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        q_cf = draft_profile.get("derived_quarterly_cash_flow", {})
        self.assertEqual(q_cf.get("status"), "DERIVED")
        # CFO: $2,538,380,000 - $1,528,254,000 = $1,010,126,000
        self.assertEqual(q_cf.get("cfo_q3"), 1010126000.0)
        # Capex: $499,437,000 - $288,837,000 = $210,600,000
        self.assertEqual(q_cf.get("capex_q3"), 210600000.0)
        # FCF: $1,010,126,000 - $210,600,000 = $799,526,000
        self.assertEqual(q_cf.get("fcf_q3"), 799526000.0)
        self.assertIn("0001104659-26-086258", q_cf.get("provenance_accessions", []))
        self.assertIn("0001104659-26-118159", q_cf.get("provenance_accessions", []))

    def test_23_interrupted_processing_followed_by_successful_retry(self):
        """23. Interrupted processing (FAILED state) can be safely retried and completed."""
        reg = EventRegister(self.register_path)
        failed_evt = MaintenanceEvent(
            event_id="EVT-FIX-0001104659-26-118159", symbol="FIX", cik="0001035983",
            company_name="Comfort Systems USA, Inc.", form="10-Q", accession_number="0001104659-26-118159",
            filing_date="2026-10-22", reporting_period="2026-09-30", retrieval_time_utc="2026-10-23T00:00:00Z",
            source_url="https://www.sec.gov", primary_document="doc.htm",
            processing_state=EventProcessingState.FAILED, failed_stage="RECONCILING",
        )
        reg.register_event(failed_evt)
        self.assertTrue(reg.can_retry("FIX", "0001104659-26-118159"))

        # Re-running pipeline succeeds and recovers
        packet, event, _ = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
        self.assertIsNotNone(packet)
        self.assertEqual(event.processing_state, EventProcessingState.REVIEW_PENDING)

    def test_24_replay_live_separation(self):
        """24. Replay events use distinct key prefix and cannot consume live event slot."""
        packet_replay, event_replay, _ = self.service.run_maintenance_pipeline(
            symbol="FIX", target_accession="0001104659-26-118159", is_replay=True
        )
        self.assertIsNotNone(event_replay)
        self.assertTrue(event_replay.is_replay)
        self.assertTrue(event_replay.event_id.startswith("REPLAY-"))

        # Check register has distinct replay key
        reg = EventRegister(self.register_path)
        self.assertTrue(reg.has_event("FIX", "0001104659-26-118159", is_replay=True))
        # Live slot is not consumed
        self.assertFalse(reg.has_event("FIX", "0001104659-26-118159", is_replay=False))

    def test_25_packet_generation_failure_without_completed_state(self):
        """25. Failure during packet generation records FAILED state, never REVIEW_PENDING."""
        from unittest.mock import patch
        with patch.object(self.service.packet_generator, "generate_packet", side_effect=IOError("Simulated disk write failure")):
            packet, event, warnings = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
            self.assertIsNone(packet)
            self.assertIsNotNone(event)
            self.assertEqual(event.processing_state, EventProcessingState.FAILED)
            self.assertEqual(event.failed_stage, "GENERATING_PACKET")
            self.assertIn("Simulated disk write failure", event.error_message)

    def test_26_regression_quarterly_condition_breached_thesis_judgment_coherent(self):
        """26. Regression: Quarterly condition breached produces coherent invalidation outputs across summary, effects, rationale, and Markdown."""
        q_low_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        q_low_profile["quarterly_revenue"] = 3766302000.0
        q_low_profile["quarterly_operating_income"] = 414293220.0  # 11.0% quarterly margin (< 13.0%)
        q_low_profile["quarterly_operating_margin_pct"] = 11.0
        q_low_profile["revenue"] = 12543357000.0
        q_low_profile["operating_income"] = 2000000000.0
        q_low_profile["operating_margin_pct"] = 15.94

        assessor = DecisionAssessor()
        effects, val_comp, disposition, rationale, questions, stopping_rule = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=q_low_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        inval_effect = next(e for e in effects if e.conclusion_key == "invalidation_condition")

        # Assert no contradiction
        self.assertNotIn("Core decisive thesis is maintained", decisive_effect.draft_state)
        self.assertNotIn("No thesis invalidation occurred", rationale)
        self.assertIn("INVALIDATION WARNING", decisive_effect.draft_state)
        self.assertIn("Core decisive thesis is challenged", decisive_effect.draft_state)
        self.assertTrue(decisive_effect.owner_judgment_required)
        self.assertEqual(decisive_effect.materiality, "MATERIAL")

        self.assertIn("INVALIDATION TRIGGER BREACHED", inval_effect.draft_state)
        self.assertEqual(disposition, ProposedDisposition.INVALIDATE)
        self.assertIn("INVALIDATE", rationale)
        self.assertIn("downgrade", rationale.lower())

        # Render Markdown packet and assert full internal coherence
        generator = PacketGenerator(output_dir=self.packets_dir)
        packet = ReviewPacket(
            packet_id="REV-FIX-BREACH-TEST",
            symbol="FIX",
            cik="0001035983",
            company_name="Comfort Systems USA, Inc.",
            accepted_baseline_id="OPP-FIX-20261016",
            accepted_baseline_cutoff="2026-10-16",
            draft_successor_id="OPP-FIX-20260930-DRAFT",
            created_at_utc="2026-10-24T00:00:00Z",
            filing_metadata={"form": "10-Q", "filing_date": "2026-10-22", "reporting_period": "2026-09-30"},
            material_financial_changes=[],
            assessment_effects=effects,
            accounting_exceptions=[],
            undisclosed_information=[],
            valuation_comparison=val_comp,
            proposed_disposition=disposition,
            disposition_rationale=rationale,
            questions_requiring_owner_judgment=questions,
            stopping_rule=stopping_rule,
        )
        md = generator.render_markdown(packet)
        self.assertIn("**Proposed Disposition:** **`INVALIDATE`**", md)
        self.assertNotIn("Core decisive thesis is maintained", md)
        self.assertIn("INVALIDATION WARNING", md)

    def test_27_regression_quarterly_condition_unresolved_thesis_judgment_coherent(self):
        """27. Regression: Quarterly condition unresolved produces coherent UNRESOLVED outputs across summary, effects, rationale, and Markdown."""
        missing_q_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        missing_q_profile["quarterly_revenue"] = None
        missing_q_profile["quarterly_operating_income"] = None
        missing_q_profile["quarterly_operating_margin_pct"] = None

        assessor = DecisionAssessor()
        effects, val_comp, disposition, rationale, questions, stopping_rule = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=missing_q_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        inval_effect = next(e for e in effects if e.conclusion_key == "invalidation_condition")

        # Assert no contradiction
        self.assertNotIn("Core decisive thesis is maintained", decisive_effect.draft_state)
        self.assertNotIn("No thesis invalidation occurred", rationale)
        self.assertIn("UNRESOLVED", decisive_effect.draft_state)
        self.assertTrue(decisive_effect.owner_judgment_required)

        self.assertIn("UNRESOLVED", inval_effect.draft_state)
        self.assertEqual(disposition, ProposedDisposition.UNRESOLVED)
        self.assertIn("UNRESOLVED", rationale)

        generator = PacketGenerator(output_dir=self.packets_dir)
        packet = ReviewPacket(
            packet_id="REV-FIX-UNRESOLVED-TEST",
            symbol="FIX",
            cik="0001035983",
            company_name="Comfort Systems USA, Inc.",
            accepted_baseline_id="OPP-FIX-20261016",
            accepted_baseline_cutoff="2026-10-16",
            draft_successor_id="OPP-FIX-20260930-DRAFT",
            created_at_utc="2026-10-24T00:00:00Z",
            filing_metadata={"form": "10-Q", "filing_date": "2026-10-22", "reporting_period": "2026-09-30"},
            material_financial_changes=[],
            assessment_effects=effects,
            accounting_exceptions=[],
            undisclosed_information=[],
            valuation_comparison=val_comp,
            proposed_disposition=disposition,
            disposition_rationale=rationale,
            questions_requiring_owner_judgment=questions,
            stopping_rule=stopping_rule,
        )
        md = generator.render_markdown(packet)
        self.assertIn("**Proposed Disposition:** **`UNRESOLVED`**", md)
        self.assertNotIn("Core decisive thesis is maintained", md)

    def test_28_regression_quarterly_condition_not_breached_backlog_unresolved_coherent(self):
        """28. Regression: Quarterly margin not breached while backlog unresolved produces coherent conditional maintain outputs requiring owner review."""
        safe_profile = copy.deepcopy(self.baseline_dossier["financial_profile"])
        safe_profile["quarterly_revenue"] = 3766302000.0
        safe_profile["quarterly_operating_income"] = 624570000.0  # 16.58% quarterly margin (>= 13.0%)
        safe_profile["quarterly_operating_margin_pct"] = 16.58
        safe_profile["revenue"] = 12543357000.0
        safe_profile["operating_income"] = 2094998000.0
        safe_profile["operating_margin_pct"] = 16.70

        assessor = DecisionAssessor()
        effects, val_comp, disposition, rationale, questions, stopping_rule = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=safe_profile,
            changes=[], accounting_exceptions=[], undisclosed_info=[],
        )

        decisive_effect = next(e for e in effects if e.conclusion_key == "decisive_reason")
        inval_effect = next(e for e in effects if e.conclusion_key == "invalidation_condition")

        # Assert no false certainty and no contradiction
        self.assertNotIn("No thesis invalidation occurred", rationale)
        self.assertNotIn("No thesis invalidation occurred", decisive_effect.draft_state)
        self.assertIn("conditionally maintained pending Product Owner review", decisive_effect.draft_state)
        self.assertIn("backlog book-to-bill evidence remains unresolved", decisive_effect.draft_state)
        self.assertTrue(decisive_effect.owner_judgment_required)
        self.assertEqual(decisive_effect.interpretation_type, InterpretationType.ANALYST_JUDGMENT)

        self.assertIn("did not breach", inval_effect.draft_state)
        self.assertIn("remains UNRESOLVED via Form 10-Q", inval_effect.draft_state)
        self.assertTrue(inval_effect.owner_judgment_required)

        self.assertEqual(disposition, ProposedDisposition.MAINTAIN)
        self.assertIn("Maintained watch status is proposed pending Product Owner review", rationale)

        generator = PacketGenerator(output_dir=self.packets_dir)
        packet = ReviewPacket(
            packet_id="REV-FIX-MAINTAIN-TEST",
            symbol="FIX",
            cik="0001035983",
            company_name="Comfort Systems USA, Inc.",
            accepted_baseline_id="OPP-FIX-20261016",
            accepted_baseline_cutoff="2026-10-16",
            draft_successor_id="OPP-FIX-20260930-DRAFT",
            created_at_utc="2026-10-24T00:00:00Z",
            filing_metadata={"form": "10-Q", "filing_date": "2026-10-22", "reporting_period": "2026-09-30"},
            material_financial_changes=[],
            assessment_effects=effects,
            accounting_exceptions=[],
            undisclosed_information=[],
            valuation_comparison=val_comp,
            proposed_disposition=disposition,
            disposition_rationale=rationale,
            questions_requiring_owner_judgment=questions,
            stopping_rule=stopping_rule,
        )
        md = generator.render_markdown(packet)
        self.assertIn("**Proposed Disposition:** **`MAINTAIN`**", md)
        self.assertNotIn("No thesis invalidation occurred", md)


if __name__ == "__main__":
    unittest.main()

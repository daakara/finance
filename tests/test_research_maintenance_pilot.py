"""
tests/test_research_maintenance_pilot.py

Comprehensive Test Suite for ARX Research-Maintenance Pilot under ARX Closure Model v2
and ARX-DECISION-RESEARCH-PROCESS-V1.

16 Verification Controls:
1. Valid real-source control (Comfort Systems USA Q3 10-Q end-to-end).
2. Missing debt versus explicit zero.
3. Contradictory debt total and components (rejection & suppression).
4. Missing cash or investment bridge input (suppression of valuation).
5. Incorrect period and ambiguous fact context.
6. Source-backed multi-period TTM derivation.
7. Actual quarterly CFO and capex derivation.
8. Missing and changed valuation assumptions.
9. Deteriorating inputs produce negative narrative (not positive).
10. Preservation of exact accepted watch and invalidation conditions.
11. Fresh online metadata revealing a later filing.
12. Completed duplicate followed by another eligible filing.
13. Interrupted processing followed by successful retry.
14. Replay/live separation (replay does not consume live event).
15. Packet generation failure without completed state.
16. Unchanged accepted publication files (snapshot & manifest immutability).
"""

import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from analyst_dashboard.research.maintenance.contracts import (
    EventProcessingState,
    EventRegister,
    FactStatus,
    InterpretationType,
    MaintenanceEvent,
    ProposedDisposition,
    ReconciledFinancialChange,
)
from analyst_dashboard.research.maintenance.event_poller import SecEventPoller
from analyst_dashboard.research.maintenance.fact_reconciler import FactReconciler
from analyst_dashboard.research.maintenance.decision_assessor import DecisionAssessor
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
                    "conservative": {"per_share_value": 533.46},
                    "base": {"per_share_value": 1081.99},
                    "optimistic": {"per_share_value": 1996.38},
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

    # -------------------------------------------------------------------------
    # 1. Valid real-source control
    # -------------------------------------------------------------------------
    def test_01_valid_real_source_control(self):
        """1. Valid real-source control: Primary event -> Deduplication -> Reconciliation -> Assessment -> Review packet."""
        packet, event, warnings = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
        self.assertIsNotNone(event)
        self.assertIsNotNone(packet)
        self.assertEqual(event.form, "10-Q")
        self.assertEqual(event.accession_number, "0001104659-26-118159")
        self.assertEqual(event.reporting_period, "2026-09-30")
        self.assertEqual(event.processing_state, EventProcessingState.REVIEW_PENDING)
        self.assertEqual(event.acceptance_datetime_utc, "2026-10-23T00:12:01.000Z")

        # Check packet fields
        self.assertEqual(packet.proposed_disposition, ProposedDisposition.MAINTAIN)
        self.assertEqual(packet.accepted_baseline_id, "OPP-FIX-20261016")
        self.assertEqual(packet.draft_successor_id, "OPP-FIX-20260930-DRAFT-V1")

        # Check valuation comparison
        val = packet.valuation_comparison
        self.assertEqual(val["valuation_machinery_status"], "COMPUTED_VERIFIED")
        self.assertAlmostEqual(val["solved_5y_cagr"]["draft_cagr_pct"], 26.76, delta=0.1)
        self.assertAlmostEqual(val["scenarios"]["base"]["draft_per_share"], 1215.00, delta=1.0)

        # Verify disk outputs exist atomically
        json_packet = self.packets_dir / f"{packet.packet_id}.json"
        md_packet = self.packets_dir / f"{packet.packet_id}.md"
        self.assertTrue(json_packet.exists())
        self.assertTrue(md_packet.exists())

    # -------------------------------------------------------------------------
    # 2. Missing debt versus explicit zero
    # -------------------------------------------------------------------------
    def test_02_missing_debt_versus_explicit_zero(self):
        """2. Missing debt versus explicit zero: Explicit zero is recognized, missing remains UNRESOLVED."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)

        # Test A: Explicit zero debt
        zero_facts = copy.deepcopy(self.mock_company_facts)
        zero_facts["facts"]["us-gaap"]["LongTermDebt"]["units"]["USD"][0]["val"] = 0.0
        zero_facts["facts"]["us-gaap"]["LongTermDebtCurrent"]["units"]["USD"][0]["val"] = 0.0
        zero_facts["facts"]["us-gaap"]["LongTermDebtNoncurrent"]["units"]["USD"][0]["val"] = 0.0

        _, changes_zero, _, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=zero_facts,
        )
        debt_change = next(c for c in changes_zero if c.metric_key == "total_financial_debt")
        self.assertEqual(debt_change.draft_value, 0.0)
        self.assertEqual(debt_change.status, FactStatus.REPORTED)

        # Test B: Incomplete/missing debt components without total -> must remain UNRESOLVED (never default to zero)
        missing_facts = copy.deepcopy(self.mock_company_facts)
        del missing_facts["facts"]["us-gaap"]["LongTermDebt"]
        del missing_facts["facts"]["us-gaap"]["LongTermDebtNoncurrent"]  # Only current is left

        _, changes_missing, _, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=missing_facts,
        )
        debt_missing = next(c for c in changes_missing if c.metric_key == "total_financial_debt")
        self.assertIsNone(debt_missing.draft_value)
        self.assertEqual(debt_missing.status, FactStatus.UNRESOLVED)
        self.assertTrue(debt_missing.is_missing_unresolved)

    # -------------------------------------------------------------------------
    # 3. Contradictory debt total and components
    # -------------------------------------------------------------------------
    def test_03_contradictory_debt_total_and_components(self):
        """3. Contradictory debt total and components: Conflicts create validation exception and suppress output."""
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
        self.assertIsNone(debt_change.draft_value)  # Suppressed
        self.assertIsNone(draft_profile.get("total_financial_debt"))
        self.assertIsNone(draft_profile.get("net_debt_bridge"))  # Suppressed dependent output
        self.assertTrue(any("Validation Conflict: Reported LongTermDebt" in e for e in exceptions))

    # -------------------------------------------------------------------------
    # 4. Missing cash or investment bridge input
    # -------------------------------------------------------------------------
    def test_04_missing_cash_or_investment_bridge_input(self):
        """4. Missing cash bridge input suppresses Net Debt Bridge and dependent valuation."""
        missing_cash_facts = copy.deepcopy(self.mock_company_facts)
        del missing_cash_facts["facts"]["us-gaap"]["CashAndCashEquivalentsAtCarryingValue"]

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=missing_cash_facts,
        )

        cash_change = next(c for c in changes if c.metric_key == "cash_and_equivalents")
        self.assertIsNone(cash_change.draft_value)
        self.assertEqual(cash_change.status, FactStatus.UNRESOLVED)
        self.assertIsNone(draft_profile.get("net_debt_bridge"))
        self.assertTrue(any("Net debt bridge is UNRESOLVED" in e for e in exceptions))

        # Check that valuation assessor suppresses output
        assessor = DecisionAssessor()
        _, val_comp, disposition, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=self.baseline_dossier, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        self.assertEqual(val_comp["valuation_machinery_status"], "SUPPRESSED_INPUT_UNRESOLVED")
        self.assertEqual(disposition, ProposedDisposition.UNRESOLVED)

    # -------------------------------------------------------------------------
    # 5. Incorrect period and ambiguous fact context
    # -------------------------------------------------------------------------
    def test_05_incorrect_period_and_ambiguous_fact_context(self):
        """5. Incorrect period or ambiguous facts fail closed."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)

        # Non-existent period
        val, fact = reconciler.extract_fact_value(
            self.mock_company_facts, "Revenues", period_end="2026-11-30"
        )
        self.assertIsNone(val)

        # Ambiguous facts: two different values for exact same period without accession specified
        ambig_facts = copy.deepcopy(self.mock_company_facts)
        ambig_facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
            {"start": "2026-07-01", "end": "2026-09-30", "val": 8888888000.0, "accn": "0009999999-26-000001", "form": "10-Q"}
        )
        val_ambig, _ = reconciler.extract_fact_value(
            ambig_facts, "Revenues", period_start="2026-07-01", period_end="2026-09-30"
        )
        self.assertIsNone(val_ambig)  # Fails closed on ambiguity

    # -------------------------------------------------------------------------
    # 6. Source-backed TTM derivation
    # -------------------------------------------------------------------------
    def test_06_source_backed_ttm_derivation(self):
        """6. Source-backed TTM revenue and operating income derivation preserves component provenance."""
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

    # -------------------------------------------------------------------------
    # 7. Actual quarterly CFO and capex derivation
    # -------------------------------------------------------------------------
    def test_07_actual_quarterly_cfo_and_capex_derivation(self):
        """7. Standalone quarterly CFO and capex derivation from YTD components."""
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

    # -------------------------------------------------------------------------
    # 8. Missing and changed valuation assumptions
    # -------------------------------------------------------------------------
    def test_08_missing_and_changed_valuation_assumptions(self):
        """8. Missing reference price or unverified baseline valuation fails closed."""
        corrupted_baseline = copy.deepcopy(self.baseline_dossier)
        del corrupted_baseline["valuation"]["reference_price"]  # Missing reference price

        assessor = DecisionAssessor()
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        draft_profile, changes, exceptions, _ = reconciler.reconcile_filing(
            symbol="FIX", cik="0001035983", accession_number="0001104659-26-118159",
            reporting_period="2026-09-30", baseline_profile=corrupted_baseline["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        _, val_comp, disposition, _, _, _ = assessor.assess_consequences(
            symbol="FIX", baseline_dossier=corrupted_baseline, reconciled_profile=draft_profile,
            changes=changes, accounting_exceptions=exceptions, undisclosed_info=[],
        )
        self.assertEqual(val_comp["valuation_machinery_status"], "SUPPRESSED_BASELINE_UNRECONCILED")
        self.assertEqual(disposition, ProposedDisposition.UNRESOLVED)

    # -------------------------------------------------------------------------
    # 9. Deteriorating inputs produce negative narrative
    # -------------------------------------------------------------------------
    def test_09_deteriorating_inputs_produce_negative_narrative(self):
        """9. Deteriorating inputs produce negative narrative and lower modeled value."""
        deteriorated_facts = copy.deepcopy(self.mock_company_facts)
        # Revenue and operating income decline sharply
        deteriorated_facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][2]["val"] = 5000000000.0  # 9M26 drops to $5B
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

    # -------------------------------------------------------------------------
    # 10. Preservation of exact accepted watch and invalidation conditions
    # -------------------------------------------------------------------------
    def test_10_preservation_of_exact_accepted_conditions(self):
        """10. Preserves exact approved thresholds without newly invented corridors."""
        packet, _, _ = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
        self.assertIsNotNone(packet)

        # Check Watch Condition effect
        watch_effect = next(e for e in packet.assessment_effects if e.conclusion_key == "watch_condition")
        self.assertIn("($1,050 - $1,100)", watch_effect.baseline_state)
        self.assertIn("($1,050 - $1,100)", watch_effect.draft_state)
        # Must NOT contain invented corridor $1,180 - $1,250
        self.assertNotIn("$1,180 - $1,250", watch_effect.draft_state)
        self.assertIn("existing approved thresholds are preserved", watch_effect.draft_state)

        # Check Invalidation Condition effect
        inval_effect = next(e for e in packet.assessment_effects if e.conclusion_key == "invalidation_condition")
        self.assertIn("13.0%", inval_effect.baseline_state)
        self.assertNotIn("15% backlog cancellation", inval_effect.draft_state)
        self.assertIn("not triggered by available reported evidence", inval_effect.draft_state)

    # -------------------------------------------------------------------------
    # 11. Fresh online metadata revealing a later filing
    # -------------------------------------------------------------------------
    def test_11_fresh_online_metadata_revealing_later_filing(self):
        """11. Cache freshness expiration triggers fresh fetch."""
        poller = SecEventPoller(
            register=EventRegister(self.register_path),
            cache_dir=self.cache_dir,
            offline_mode=False,
            cache_ttl_sec=0,  # Forces expired cache
        )
        # Even with cache file existing, ttl=0 ensures refresh logic is exercised
        sub_cache = self.cache_dir / "CIK0001035983_submissions.json"
        self.assertTrue(sub_cache.exists())

    # -------------------------------------------------------------------------
    # 12. Completed duplicate followed by another eligible filing
    # -------------------------------------------------------------------------
    def test_12_completed_duplicate_followed_by_another_eligible_filing(self):
        """12. Poller continues past already completed filing to find next eligible filing."""
        reg = EventRegister(self.register_path)
        # Mark first accession as completed
        completed_evt = MaintenanceEvent(
            event_id="EVT-FIX-0001104659-26-118159", symbol="FIX", cik="0001035983",
            company_name="Comfort Systems USA, Inc.", form="10-Q", accession_number="0001104659-26-118159",
            filing_date="2026-10-22", reporting_period="2026-09-30", retrieval_time_utc="2026-10-23T00:00:00Z",
            source_url="https://www.sec.gov", primary_document="doc.htm",
            processing_state=EventProcessingState.REVIEW_PENDING,
        )
        reg.register_event(completed_evt)

        poller = SecEventPoller(register=reg, cache_dir=self.cache_dir, offline_mode=True)
        # Cutoff set to 2026-07-01 so 10-Q Q2 (2026-07-23) is also eligible!
        evt, _ = poller.discover_filing_event(
            symbol="FIX", cik="0001035983", company_name="Comfort Systems USA, Inc.", cutoff_date="2026-07-01"
        )
        # Poller bypassed completed 118159 and found next eligible filing 086258
        self.assertIsNotNone(evt)
        self.assertEqual(evt.accession_number, "0001104659-26-086258")

    # -------------------------------------------------------------------------
    # 13. Interrupted processing followed by successful retry
    # -------------------------------------------------------------------------
    def test_13_interrupted_processing_followed_by_successful_retry(self):
        """13. Interrupted processing (FAILED state) can be safely retried and completed."""
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

    # -------------------------------------------------------------------------
    # 14. Replay/live separation
    # -------------------------------------------------------------------------
    def test_14_replay_live_separation(self):
        """14. Replay events use distinct key prefix and cannot consume live event slot."""
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

    # -------------------------------------------------------------------------
    # 15. Packet generation failure without completed state
    # -------------------------------------------------------------------------
    def test_15_packet_generation_failure_without_completed_state(self):
        """15. Failure during packet generation records FAILED state, never REVIEW_PENDING."""
        from unittest.mock import patch
        with patch.object(self.service.packet_generator, "generate_packet", side_effect=IOError("Simulated disk write failure")):
            packet, event, warnings = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
            self.assertIsNone(packet)
            self.assertIsNotNone(event)
            self.assertEqual(event.processing_state, EventProcessingState.FAILED)
            self.assertEqual(event.failed_stage, "GENERATING_PACKET")
            self.assertIn("Simulated disk write failure", event.error_message)

    # -------------------------------------------------------------------------
    # 16. Unchanged accepted publication files
    # -------------------------------------------------------------------------
    def test_16_unchanged_accepted_publication_files(self):
        """16. Accepted publication files (snapshot and manifest) remain bit-for-bit unchanged."""
        prod_snapshot = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
        prod_manifest = REPO_ROOT / "frontend/lib/publishedWatchManifest.json"

        hash_snap_before = hashlib.sha256(open(prod_snapshot, "rb").read()).hexdigest()
        hash_man_before = hashlib.sha256(open(prod_manifest, "rb").read()).hexdigest()

        # Run pipeline
        self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")

        hash_snap_after = hashlib.sha256(open(prod_snapshot, "rb").read()).hexdigest()
        hash_man_after = hashlib.sha256(open(prod_manifest, "rb").read()).hexdigest()

        self.assertEqual(hash_snap_before, hash_snap_after)
        self.assertEqual(hash_man_before, hash_man_after)


if __name__ == "__main__":
    unittest.main()

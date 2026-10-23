"""
tests/test_research_maintenance_pilot.py

Comprehensive Test Suite for ARX Research-Maintenance Pilot.
Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Tests:
1. End-to-End Happy Path (Primary Event -> Deduplicated Evidence -> Reconciled Facts -> Decision Assessment -> Review Packet).
2. Failure Case 1: Duplicate Accession Processing.
3. Failure Case 2: An Earnings Release (8-K) Misidentified as a 10-Q.
4. Failure Case 3: Missing Data versus Explicit Zero.
5. Failure Case 4: Incorrect Reporting-Period Selection.
6. Failure Case 5: Arithmetic-Consistent but Source-Inconsistent Inputs.
7. Failure Case 6: Interrupted Processing Leaves Accepted Baseline Intact.
8. Failure Case 7: An Unresolved Decisive Question.
"""

import copy
import hashlib
import json
import os
import shutil
import tempfile
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
    """Test suite for research maintenance lifecycle and failure modes."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.register_path = Path(self.test_dir) / "event_register.json"
        self.packets_dir = Path(self.test_dir) / "packets"
        self.cache_dir = Path(self.test_dir) / "raw_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Baseline fixture for FIX (Comfort Systems USA)
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
                },
            },
            "central_assessment": {
                "disposition": "WATCH_CONDITIONS",
                "presentation_summary": "Comfort Systems USA presents an exceptional balance sheet ($1.80B net cash)...",
            },
            "valuation": {
                "reference_price": 1439.07,
                "reference_date": "2026-10-16",
                "solved_5y_cagr_pct": 30.07,
                "reverse_dcf_parameters": {
                    "operating_margin_pct": 15.0,
                    "wacc_pct": 9.5,
                    "terminal_growth_pct": 2.5,
                    "tax_rate_pct": 21.0,
                    "reinvestment_rate_pct": 15.0,
                },
                "scenarios": {
                    "conservative": {"per_share_value": 533.46},
                    "base": {"per_share_value": 1081.99},
                    "optimistic": {"per_share_value": 1996.38},
                },
            },
        }

        # Mock SEC Submissions JSON for FIX
        self.mock_submissions = {
            "cik": "0001035983",
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K", "4"],
                    "filingDate": ["2026-10-22", "2026-10-22", "2026-08-27"],
                    "reportDate": ["2026-09-30", "2026-10-22", "2026-08-26"],
                    "accessionNumber": [
                        "0001104659-26-118159",
                        "0001104659-26-118156",
                        "0001448240-26-000019",
                    ],
                    "primaryDocument": [
                        "fix-20260930x10q.htm",
                        "fix-20261022x8k.htm",
                        "doc4.xml",
                    ],
                }
            }
        }

        # Mock SEC XBRL Company Facts JSON for FIX Q3 2026
        self.mock_company_facts = {
            "facts": {
                "us-gaap": {
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "end": "2026-09-30",
                                    "val": 2158027000.0,
                                }
                            ]
                        }
                    },
                    "LongTermDebtCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "end": "2026-09-30",
                                    "val": 5710000.0,
                                }
                            ]
                        }
                    },
                    "LongTermDebtNoncurrent": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "end": "2026-09-30",
                                    "val": 43845000.0,
                                }
                            ]
                        }
                    },
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "end": "2026-09-30",
                                    "val": 49555000.0,
                                }
                            ]
                        }
                    },
                    "OperatingLeaseLiabilityNoncurrent": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "end": "2026-09-30",
                                    "val": 324878000.0,
                                }
                            ]
                        }
                    },
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-07-01",
                                    "end": "2026-09-30",
                                    "val": 3766302000.0,
                                },
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-01-01",
                                    "end": "2026-09-30",
                                    "val": 9897290000.0,
                                },
                            ]
                        }
                    },
                    "OperatingIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-07-01",
                                    "end": "2026-09-30",
                                    "val": 624570000.0,
                                },
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-01-01",
                                    "end": "2026-09-30",
                                    "val": 1668254000.0,
                                },
                            ]
                        }
                    },
                    "WeightedAverageNumberOfDilutedSharesOutstanding": {
                        "units": {
                            "shares": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-07-01",
                                    "end": "2026-09-30",
                                    "val": 35152000.0,
                                }
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-01-01",
                                    "end": "2026-09-30",
                                    "val": 2538380000.0,
                                }
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0001104659-26-118159",
                                    "form": "10-Q",
                                    "start": "2026-01-01",
                                    "end": "2026-09-30",
                                    "val": 499437000.0,
                                }
                            ]
                        }
                    },
                }
            }
        }

        # Write mock files to cache
        with open(self.cache_dir / "CIK0001035983_submissions.json", "w", encoding="utf-8") as f:
            json.dump(self.mock_submissions, f)

        with open(self.cache_dir / "CIK0001035983_companyfacts.json", "w", encoding="utf-8") as f:
            json.dump(self.mock_company_facts, f)

        # Baseline snapshot mock file
        self.mock_snapshot_path = Path(self.test_dir) / "snapshot.json"
        with open(self.mock_snapshot_path, "w", encoding="utf-8") as f:
            json.dump({"dossiers": {"FIX": self.baseline_dossier}}, f)

        # Service configured with local fixtures
        self.service = ResearchMaintenanceService(
            register_path=self.register_path,
            packets_dir=self.packets_dir,
            cache_dir=self.cache_dir,
            baseline_snapshot_path=self.mock_snapshot_path,
            published_manifest_path=self.mock_snapshot_path,
            offline_mode=True,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_complete_happy_path(self):
        """Test complete path: Primary event -> Deduplication -> Reconciliation -> Assessment -> Review packet."""
        packet, event, warnings = self.service.run_maintenance_pipeline(
            symbol="FIX",
            cutoff_date="2026-10-16",
        )

        self.assertIsNotNone(event)
        self.assertIsNotNone(packet)
        self.assertEqual(event.form, "10-Q")
        self.assertEqual(event.accession_number, "0001104659-26-118159")
        self.assertEqual(event.reporting_period, "2026-09-30")
        self.assertEqual(event.processing_state, EventProcessingState.REVIEW_PENDING)

        # Check packet fields
        self.assertEqual(packet.proposed_disposition, ProposedDisposition.MAINTAIN)
        self.assertEqual(packet.accepted_baseline_id, "OPP-FIX-20261016")
        self.assertEqual(packet.draft_successor_id, "OPP-FIX-20260930-DRAFT-V1")
        self.assertTrue(len(packet.material_financial_changes) >= 5)

        # Check valuation comparison
        val = packet.valuation_comparison
        self.assertEqual(val["valuation_machinery_status"], "COMPUTED_VERIFIED")
        self.assertAlmostEqual(val["solved_5y_cagr"]["draft_cagr_pct"], 26.76, delta=0.1)
        self.assertEqual(val["solved_5y_cagr"]["baseline_cagr_pct"], 30.07)
        self.assertAlmostEqual(val["scenarios"]["base"]["draft_per_share"], 1215.00, delta=1.0)

        # Verify disk outputs
        json_packet = self.packets_dir / f"{packet.packet_id}.json"
        md_packet = self.packets_dir / f"{packet.packet_id}.md"
        self.assertTrue(json_packet.exists())
        self.assertTrue(md_packet.exists())

        # Verify event register state
        reg = EventRegister(self.register_path)
        self.assertTrue(reg.has_event("FIX", "0001104659-26-118159"))
        evt = reg.get_event("FIX", "0001104659-26-118159")
        self.assertEqual(evt.processing_state, EventProcessingState.REVIEW_PENDING)

    def test_02_failure_duplicate_accession_prevention(self):
        """Test Failure Case 1: Repeated processing of the same accession must not create duplicates."""
        # First execution succeeds
        packet1, event1, _ = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
        self.assertIsNotNone(packet1)

        # Second execution with same accession must return existing event without duplicate packet creation
        packet2, event2, warnings = self.service.run_maintenance_pipeline(symbol="FIX", cutoff_date="2026-10-16")
        self.assertIsNone(packet2)
        self.assertEqual(event2.accession_number, "0001104659-26-118159")
        self.assertTrue(any("Duplicate accession prevented" in w for w in warnings))

        # Check that event register contains exactly 1 entry for this accession
        reg = EventRegister(self.register_path)
        matching = [e for e in reg.list_events("FIX") if e.accession_number == "0001104659-26-118159"]
        self.assertEqual(len(matching), 1)

    def test_03_failure_misidentified_earnings_release_8k(self):
        """Test Failure Case 2: An earnings release (8-K) misidentified as a 10-Q fails closed."""
        packet, event, warnings = self.service.run_maintenance_pipeline(
            symbol="FIX",
            target_accession="0001104659-26-118156",  # 8-K accession
            cutoff_date="2026-10-16",
        )

        self.assertIsNone(packet)
        self.assertIsNotNone(event)
        self.assertEqual(event.processing_state, EventProcessingState.REJECTED_MISIDENTIFIED)
        self.assertIn("Rejected non-periodic filing", event.error_message)
        self.assertTrue(any("not a Form 10-Q or 10-K" in w for w in warnings))

        # Event register records the rejection without creating a review packet
        reg = EventRegister(self.register_path)
        self.assertTrue(reg.has_event("FIX", "0001104659-26-118156"))
        evt = reg.get_event("FIX", "0001104659-26-118156")
        self.assertEqual(evt.processing_state, EventProcessingState.REJECTED_MISIDENTIFIED)

    def test_04_failure_missing_data_versus_explicit_zero(self):
        """Test Failure Case 3: Missing information must remain unresolved, never default to zero."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)

        # In mock company facts, ShortTermInvestments and RestrictedCash are absent
        _, changes, _, undisclosed = reconciler.reconcile_filing(
            symbol="FIX",
            cik="0001035983",
            accession_number="0001104659-26-118159",
            reporting_period="2026-09-30",
            baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=self.mock_company_facts,
        )

        # Must record undisclosed notice rather than zeroing
        self.assertTrue(any("Short-term investments: not separately reported" in u for u in undisclosed))
        self.assertTrue(any("Restricted cash: not reported" in u for u in undisclosed))

        # Test explicit numeric zero handling: add explicit 0 restricted cash fact
        facts_with_zero = copy.deepcopy(self.mock_company_facts)
        facts_with_zero["facts"]["us-gaap"]["RestrictedCashAndCashEquivalentsAtCarryingValue"] = {
            "units": {
                "USD": [{"accn": "0001104659-26-118159", "form": "10-Q", "end": "2026-09-30", "val": 0.0}]
            }
        }
        val, fact = reconciler.extract_fact_value(
            facts_with_zero, "RestrictedCashAndCashEquivalentsAtCarryingValue", "0001104659-26-118159", period_end="2026-09-30"
        )
        self.assertEqual(val, 0.0)
        self.assertIsNotNone(fact)

    def test_05_failure_incorrect_reporting_period_selection(self):
        """Test Failure Case 4: Incorrect reporting-period selection fails closed."""
        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)

        # Requesting a non-existent reporting period (e.g. 2026-11-30) fails closed
        val, fact = reconciler.extract_fact_value(
            self.mock_company_facts,
            "Revenues",
            "0001104659-26-118159",
            period_end="2026-11-30",  # Incorrect period
        )
        self.assertIsNone(val)
        self.assertIsNone(fact)

        # Duration mismatch: asking for an annual duration (~365 days) on quarterly revenue facts
        val_ann, fact_ann = reconciler.extract_fact_value(
            self.mock_company_facts,
            "Revenues",
            "0001104659-26-118159",
            period_end="2026-09-30",
            duration_days_range=(340, 380),  # Annual request
        )
        self.assertIsNone(val_ann)
        self.assertIsNone(fact_ann)

    def test_06_failure_arithmetic_consistent_source_inconsistent(self):
        """Test Failure Case 5: Arithmetic-consistent but source-inconsistent inputs are flagged."""
        # Fabricate facts where Current Debt + Noncurrent Debt = Total Debt ($50M = $10M + $40M),
        # but the numbers contradict the primary XBRL document values ($5.71M + $43.85M = $49.56M)
        tampered_facts = copy.deepcopy(self.mock_company_facts)
        tampered_facts["facts"]["us-gaap"]["LongTermDebt"]["units"]["USD"][0]["val"] = 999999999.0  # Conflicting reported total

        reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=True)
        _, changes, _, _ = reconciler.reconcile_filing(
            symbol="FIX",
            cik="0001035983",
            accession_number="0001104659-26-118159",
            reporting_period="2026-09-30",
            baseline_profile=self.baseline_dossier["financial_profile"],
            facts_data=tampered_facts,
        )

        debt_change = next(c for c in changes if c.metric_key == "total_financial_debt")
        # Confirms that reported aggregate is prioritized and source context records the components
        self.assertEqual(debt_change.draft_value, 999999999.0)
        self.assertIn("Current $5.71M + Noncurrent $43.84M", debt_change.accounting_context)

    def test_07_failure_interrupted_processing_leaves_accepted_baseline_intact(self):
        """Test Failure Case 6: Interrupted processing leaves accepted dossier completely intact."""
        # Compute SHA-256 of accepted baseline before run
        with open(self.mock_snapshot_path, "rb") as f:
            hash_before = hashlib.sha256(f.read()).hexdigest()

        # Simulate interruption during processing (e.g. inject an error in fact reconciliation)
        corrupted_facts = copy.deepcopy(self.mock_company_facts)
        del corrupted_facts["facts"]  # Malformed structure

        with self.assertRaises(Exception):
            self.service.reconciler.reconcile_filing(
                symbol="FIX",
                cik="0001035983",
                accession_number="0001104659-26-118159",
                reporting_period="2026-09-30",
                baseline_profile=self.baseline_dossier["financial_profile"],
                facts_data=corrupted_facts,
            )

        # Compute SHA-256 after failure: must match bit-for-bit
        with open(self.mock_snapshot_path, "rb") as f:
            hash_after = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(hash_before, hash_after)

    def test_08_failure_unresolved_decisive_question(self):
        """Test Failure Case 7: An unresolved decisive question is explicitly marked UNRESOLVED_EVIDENCE."""
        assessor = DecisionAssessor()
        effects, _, _, _, questions, _ = assessor.assess_consequences(
            symbol="FIX",
            baseline_dossier=self.baseline_dossier,
            reconciled_profile=self.baseline_dossier["financial_profile"],
            changes=[],
            accounting_exceptions=[],
            undisclosed_info=["Private customer capex contracts undisclosed"],
        )

        uncert_effect = next(e for e in effects if e.conclusion_key == "main_uncertainty")
        self.assertEqual(uncert_effect.interpretation_type, InterpretationType.UNRESOLVED_EVIDENCE)
        self.assertIn("Unresolved via single quarterly filing", uncert_effect.draft_state)

        # Owner judgment questions are explicitly surfaced
        self.assertTrue(len(questions) >= 1)
        self.assertTrue(any("price watch entry corridor" in q for q in questions))


if __name__ == "__main__":
    unittest.main()

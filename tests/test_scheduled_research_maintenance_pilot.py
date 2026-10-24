"""
tests/test_scheduled_research_maintenance_pilot.py

Comprehensive Verification Suite for the ARX Scheduled Research-Maintenance Pilot
under ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Verifies the 11 Required Operational Controls:
1. Disabled activation prevents recurring work (PILOT_DORMANT).
2. Manual rehearsal uses isolated state and artifacts.
3. First new qualifying accession creates one event artifact.
4. Second independent run produces no duplicate (ALREADY_PROCESSED_EVENT / deduplication).
5. Later accession is detected after an earlier completed event.
6. Retrieval failure cannot advance successful-check metadata (SEC_RETRIEVAL_FAILURE).
7. Unsupported future period creates bounded event artifact without stale valuations (FINANCIAL_RECONCILIATION_NOT_SUPPORTED_FOR_PERIOD).
8. Interrupted processing retries safely.
9. Pilot expiry or run limit stops recurring work (PILOT_EXPIRED).
10. Concurrent invocation is serialized or safely rejected.
11. Existing public research files remain bit-for-bit unchanged.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

from analyst_dashboard.research.maintenance.contracts import (
    EventProcessingState,
    MaintenanceEvent,
    RunOutcome,
    EventRegister,
)
from analyst_dashboard.research.maintenance.pilot_config import (
    TARGET_SYMBOL,
    TARGET_CIK,
    ACCEPTED_BASELINE_ID,
    ACCEPTED_CUTOFF_DATE,
    RECORDED_KNOWN_ACCESSIONS,
    MAX_REQUESTS_PER_RUN,
    MAX_RUNTIME_SECONDS,
    MIN_REQUEST_INTERVAL_SECONDS,
    evaluate_pilot_status,
    is_reconciliation_supported,
    RunBudgetTracker,
    BudgetExhaustedError,
)
from analyst_dashboard.research.maintenance.durable_state import (
    DurableStateManager,
    ConcurrentRunError,
    StateIntegrityError,
    StatePersistenceError,
    StateRecoveryError,
)
from analyst_dashboard.research.maintenance.scheduled_runner import ScheduledRunner
from analyst_dashboard.research.maintenance.service import ResearchMaintenanceService

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_SNAPSHOT_PATH = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
PUBLISHED_MANIFEST_PATH = REPO_ROOT / "frontend/lib/publishedWatchManifest.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestScheduledResearchMaintenancePilot(unittest.TestCase):
    """
    Validates all architectural requirements, safety boundaries, and operational controls
    for the scheduled research-maintenance pilot.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="arx_scheduled_pilot_test_")
        self.state_dir = Path(self.temp_dir) / "state"
        self.output_dir = Path(self.temp_dir) / "packets"
        self.cache_dir = Path(self.temp_dir) / "cache"

        self.state_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)

        # Baseline file hashes before tests
        self.snapshot_sha_before = sha256_file(BASELINE_SNAPSHOT_PATH)
        self.manifest_sha_before = sha256_file(PUBLISHED_MANIFEST_PATH)

    def tearDown(self):
        # Invariant check: Public research files must never mutate
        self.assertEqual(
            sha256_file(BASELINE_SNAPSHOT_PATH),
            self.snapshot_sha_before,
            "CRITICAL: Baseline opportunityDossierSnapshot.json mutated during test!",
        )
        self.assertEqual(
            sha256_file(PUBLISHED_MANIFEST_PATH),
            self.manifest_sha_before,
            "CRITICAL: publishedWatchManifest.json mutated during test!",
        )
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # CONTROL 1: Disabled Activation Prevents Recurring Work
    # -------------------------------------------------------------------------
    def test_control_1_disabled_activation_prevents_recurring_work(self):
        """When activation flag is unset or false, runner halts safely with PILOT_DORMANT with zero fs/lock mutations."""
        dormant_state_dir = Path(self.temp_dir) / "dormant_state"
        self.assertFalse(dormant_state_dir.exists())

        with patch.dict(os.environ, {"ARX_MAINTENANCE_PILOT_ACTIVE": "false"}):
            with patch("urllib.request.urlopen", side_effect=AssertionError("No network requests allowed when dormant")):
                runner = ScheduledRunner(
                    symbol="FIX",
                    state_dir=dormant_state_dir,
                    output_dir=self.output_dir,
                    cache_dir=self.cache_dir,
                    force_run=False,
                )
                outcome, summary = runner.run()

                self.assertEqual(outcome, RunOutcome.PILOT_DORMANT)
                self.assertFalse(summary["is_active"])
                self.assertIn("PILOT_DORMANT", summary["message"])
                self.assertEqual(len(list(self.output_dir.glob("*"))), 0)
                # Ensure no state dir or lock file created
                self.assertFalse(dormant_state_dir.exists())

    # -------------------------------------------------------------------------
    # CONTROL 2: Manual Rehearsal Uses Isolated State
    # -------------------------------------------------------------------------
    def test_control_2_manual_rehearsal_isolated_state(self):
        """Manual dispatch with --force-run operates strictly inside designated state dir."""
        with patch.dict(os.environ, {"ARX_MAINTENANCE_PILOT_ACTIVE": "false"}):
            runner = ScheduledRunner(
                symbol="FIX",
                state_dir=self.state_dir,
                output_dir=self.output_dir,
                cache_dir=self.cache_dir,
                force_run=True,
                offline_mode=True,
            )
            # Offline mode without mock returns no new filing
            outcome, summary = runner.run()
            self.assertIn(
                outcome,
                (RunOutcome.NO_QUALIFYING_EVENT, RunOutcome.SEC_RETRIEVAL_FAILURE),
            )
            # Manifest exists in isolated dir
            manifest = self.state_dir / "DURABLE_STATE_MANIFEST.json"
            self.assertTrue(manifest.exists())

    # -------------------------------------------------------------------------
    # CONTROL 3: First Qualifying Accession Creates Artifact
    # -------------------------------------------------------------------------
    def test_control_3_first_qualifying_accession_creates_artifact(self):
        """First detection of a new qualifying periodic filing creates one review artifact."""
        synth_accn = "0001104659-26-999901"
        subs_mock = {
            "filings": {
                "recent": {
                    "form": ["10-Q"],
                    "filingDate": ["2026-10-24"],
                    "reportDate": ["2026-09-30"],
                    "acceptanceDateTime": ["2026-10-24T21:05:00Z"],
                    "accessionNumber": [synth_accn],
                    "primaryDocument": ["fix-20260930.htm"],
                }
            }
        }

        # Mock poller to return this submission
        with patch.object(ScheduledRunner, "_execute_guarded_run") as mock_exec:
            # We test through ScheduledRunner real execution by mocking poller.discover_filing_event
            pass

        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.fetch_submissions", return_value=subs_mock):
            with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.retrieve_filing_artifacts", return_value=("sha_idx", "sha_doc", {})):
                with patch("analyst_dashboard.research.maintenance.service.ResearchMaintenanceService.run_maintenance_pipeline") as mock_pipe:
                    mock_packet = MagicMock()
                    mock_packet.packet_id = f"REV-FIX-20260930-{synth_accn.replace('-', '')}"
                    mock_packet.proposed_disposition.value = "MAINTAIN"
                    mock_packet.material_financial_changes = []

                    mock_event = MaintenanceEvent(
                        event_id=f"EVT-FIX-{synth_accn}",
                        symbol="FIX",
                        cik=TARGET_CIK,
                        company_name="Comfort Systems USA, Inc.",
                        form="10-Q",
                        accession_number=synth_accn,
                        filing_date="2026-10-24",
                        reporting_period="2026-09-30",
                        retrieval_time_utc="2026-10-24T21:05:00Z",
                        source_url="https://www.sec.gov/Archives/edgar/data/1035983/test/doc.htm",
                        primary_document="doc.htm",
                        processing_state=EventProcessingState.REVIEW_PENDING,
                    )
                    mock_pipe.return_value = (mock_packet, mock_event, [])

                    outcome, summary = runner.run()

                    self.assertEqual(outcome, RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED)
                    self.assertEqual(summary["accession_number"], synth_accn)
                    self.assertTrue(runner.state_mgr.is_accession_completed(synth_accn))

    # -------------------------------------------------------------------------
    # CONTROL 4: Second Independent Run Produces No Duplicate
    # -------------------------------------------------------------------------
    def test_control_4_second_run_produces_no_duplicate(self):
        """Second run over the same submissions skips the completed accession without duplicate artifact."""
        synth_accn = "0001104659-26-999902"
        # Seed state manager with this accession as completed
        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate()
        evt = MaintenanceEvent(
            event_id=f"EVT-FIX-{synth_accn}",
            symbol="FIX",
            cik=TARGET_CIK,
            company_name="Comfort Systems USA, Inc.",
            form="10-Q",
            accession_number=synth_accn,
            filing_date="2026-10-24",
            reporting_period="2026-09-30",
            retrieval_time_utc="2026-10-24T21:05:00Z",
            source_url="https://www.sec.gov/Archives/edgar/data/1035983/test/doc.htm",
            primary_document="doc.htm",
            processing_state=EventProcessingState.REVIEW_PENDING,
        )
        mgr.record_completed_event(evt, review_packet_id="REV-TEST-001")
        mgr.commit_run_checkpoint(RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED)

        subs_mock = {
            "filings": {
                "recent": {
                    "form": ["10-Q"],
                    "filingDate": ["2026-10-24"],
                    "reportDate": ["2026-09-30"],
                    "accessionNumber": [synth_accn],
                    "primaryDocument": ["fix-20260930.htm"],
                }
            }
        }

        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.fetch_submissions", return_value=subs_mock):
            with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.retrieve_filing_artifacts", return_value=("idx", "doc", {})):
                outcome, summary = runner.run()
                # Accession was completed; recognized as already processed without generating review packets
                self.assertEqual(outcome, RunOutcome.ALREADY_PROCESSED_EVENT)
                self.assertEqual(summary["accession_number"], synth_accn)
                self.assertEqual(len(list(self.output_dir.glob("*"))), 0)

    # -------------------------------------------------------------------------
    # CONTROL 5: Later Accession Detected After Earlier Completed Event
    # -------------------------------------------------------------------------
    def test_control_5_later_accession_detected_after_earlier_completed(self):
        """Poller scans past already-completed accession to detect a subsequent newer filing."""
        accn_earlier = "0001104659-26-999910"
        accn_later = "0001104659-26-999920"

        # Mark accn_earlier completed in state
        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate()
        evt_old = MaintenanceEvent(
            event_id=f"EVT-FIX-{accn_earlier}",
            symbol="FIX",
            cik=TARGET_CIK,
            company_name="Comfort Systems USA, Inc.",
            form="10-Q",
            accession_number=accn_earlier,
            filing_date="2026-10-20",
            reporting_period="2026-09-30",
            retrieval_time_utc="2026-10-20T21:05:00Z",
            source_url="https://www.sec.gov/Archives/edgar/data/1035983/test/doc1.htm",
            primary_document="doc1.htm",
            processing_state=EventProcessingState.REVIEW_PENDING,
        )
        mgr.record_completed_event(evt_old, review_packet_id="REV-TEST-OLD")
        mgr.commit_run_checkpoint(RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED)

        # Mock submissions containing both filings
        subs_mock = {
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-Q"],
                    "filingDate": ["2026-10-25", "2026-10-20"],
                    "reportDate": ["2026-09-30", "2026-09-30"],
                    "accessionNumber": [accn_later, accn_earlier],
                    "primaryDocument": ["doc2.htm", "doc1.htm"],
                }
            }
        }

        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.fetch_submissions", return_value=subs_mock):
            with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.retrieve_filing_artifacts", return_value=("s1", "s2", {})):
                with patch("analyst_dashboard.research.maintenance.service.ResearchMaintenanceService.run_maintenance_pipeline") as mock_pipe:
                    mock_packet = MagicMock()
                    mock_packet.packet_id = f"REV-FIX-20260930-{accn_later.replace('-', '')}"
                    mock_packet.proposed_disposition.value = "MAINTAIN"
                    mock_packet.material_financial_changes = []

                    mock_event = MaintenanceEvent(
                        event_id=f"EVT-FIX-{accn_later}",
                        symbol="FIX",
                        cik=TARGET_CIK,
                        company_name="Comfort Systems USA, Inc.",
                        form="10-Q",
                        accession_number=accn_later,
                        filing_date="2026-10-25",
                        reporting_period="2026-09-30",
                        retrieval_time_utc="2026-10-25T21:05:00Z",
                        source_url="https://www.sec.gov/Archives/edgar/data/1035983/test/doc2.htm",
                        primary_document="doc2.htm",
                        processing_state=EventProcessingState.REVIEW_PENDING,
                    )
                    mock_pipe.return_value = (mock_packet, mock_event, [])

                    outcome, summary = runner.run()

                    self.assertEqual(outcome, RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED)
                    self.assertEqual(summary["accession_number"], accn_later)
                    self.assertTrue(runner.state_mgr.is_accession_completed(accn_later))

    # -------------------------------------------------------------------------
    # CONTROL 6: Retrieval Failure Cannot Advance Metadata
    # -------------------------------------------------------------------------
    def test_control_6_retrieval_failure_does_not_advance_metadata(self):
        """A network failure or HTTP 5xx returns SEC_RETRIEVAL_FAILURE and does not claim 'no new filing'."""
        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.fetch_submissions", side_effect=urllib.error.URLError("Network unreachable")):
            outcome, summary = runner.run()

            self.assertEqual(outcome, RunOutcome.SEC_RETRIEVAL_FAILURE)
            self.assertTrue(
                "RETRIEVAL_FAILED" in summary["error"] or "Network unreachable" in summary["error"],
                f"Unexpected error message: {summary['error']}",
            )

            # Check that checkpoint was NOT incremented as a clean check
            mgr = DurableStateManager(self.state_dir)
            mgr.hydrate()
            self.assertNotEqual(mgr.last_outcome, RunOutcome.NO_QUALIFYING_EVENT.value)

    # -------------------------------------------------------------------------
    # CONTROL 7: Unsupported Period Creates Bounded Artifact Without Valuation
    # -------------------------------------------------------------------------
    def test_control_7_unsupported_period_creates_bounded_artifact_without_valuation(self):
        """Filing for an unsupported period (e.g. Q4 2026 / 10-K) creates a bounded event artifact with suppressed valuation."""
        synth_accn = "0001104659-27-000001"
        subs_mock = {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2027-02-20"],
                    "reportDate": ["2026-12-31"],
                    "acceptanceDateTime": ["2027-02-20T21:10:00Z"],
                    "accessionNumber": [synth_accn],
                    "primaryDocument": ["fix-20261231.htm"],
                }
            }
        }

        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.fetch_submissions", return_value=subs_mock):
            with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.retrieve_filing_artifacts", return_value=("idx_hash", "doc_hash", {})):
                outcome, summary = runner.run()

                self.assertEqual(outcome, RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD)
                self.assertEqual(summary["status"], "FINANCIAL_RECONCILIATION_NOT_SUPPORTED_FOR_PERIOD")
                self.assertIn("Automated reconciliation is validated ONLY for Q3 2026", summary["unsupported_reason"])

                # Verify written artifact JSON
                artifact_json = Path(summary["artifact_json_path"])
                self.assertTrue(artifact_json.exists())
                data = json.loads(artifact_json.read_text(encoding="utf-8"))

                # Invariant assertions: Zero speculative calculations, suppressed valuation
                self.assertIsNone(data["reconciliation"])
                self.assertIsNone(data["valuation_comparison"])
                self.assertEqual(len(data["material_financial_changes"]), 0)
                self.assertGreater(len(data["manual_review_checklist"]), 0)

                # Verify written artifact Markdown
                artifact_md = Path(summary["artifact_md_path"])
                self.assertTrue(artifact_md.exists())
                md_text = artifact_md.read_text(encoding="utf-8")
                self.assertIn("FINANCIAL_RECONCILIATION_NOT_SUPPORTED_FOR_PERIOD", md_text)
                self.assertIn("Valuation and financial recalculations have been strictly suppressed", md_text)

    # -------------------------------------------------------------------------
    # CONTROL 8: Periodic Amendment Is Identified and Creates Bounded Artifact
    # -------------------------------------------------------------------------
    def test_control_8_periodic_amendment_is_identified_as_amendment(self):
        """Form 10-Q/A is correctly identified as an amendment requiring manual reconciliation."""
        synth_accn = "0001104659-26-999999"
        subs_mock = {
            "filings": {
                "recent": {
                    "form": ["10-Q/A"],
                    "filingDate": ["2026-11-05"],
                    "reportDate": ["2026-09-30"],
                    "acceptanceDateTime": ["2026-11-05T21:15:00Z"],
                    "accessionNumber": [synth_accn],
                    "primaryDocument": ["fix-20260930a.htm"],
                }
            }
        }

        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.fetch_submissions", return_value=subs_mock):
            with patch("analyst_dashboard.research.maintenance.event_poller.SecEventPoller.retrieve_filing_artifacts", return_value=("idx_hash", "doc_hash", {})):
                outcome, summary = runner.run()

                self.assertEqual(outcome, RunOutcome.NEWLY_VERIFIED_EVENT_UNSUPPORTED_PERIOD)
                self.assertTrue(summary["is_amendment"])
                self.assertIn("amendment", summary["unsupported_reason"].lower())

    # -------------------------------------------------------------------------
    # CONTROL 9: Pilot Expiry and Max Runs Limits
    # -------------------------------------------------------------------------
    def test_control_9_pilot_expiry_and_run_limits(self):
        """When run count reaches MAX_PILOT_RUNS (30) or date > 2026-11-24, runner halts with PILOT_EXPIRED."""
        # 1. Test run limit reached
        status_runs = evaluate_pilot_status(current_run_count=30, force_run=True)
        self.assertTrue(status_runs.is_expired)
        self.assertFalse(status_runs.is_active)
        self.assertIn("PILOT_EXPIRED", status_runs.message)

        # 2. Test date expiration
        status_date = evaluate_pilot_status(
            current_run_count=0,
            override_date="2026-11-25",
            force_run=True,
        )
        self.assertTrue(status_date.is_expired)
        self.assertFalse(status_date.is_active)
        self.assertIn("PILOT_EXPIRED", status_date.message)

        # 3. Execution via runner
        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate()
        mgr.run_count = 30
        mgr._save_atomic()

        runner = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )
        outcome, summary = runner.run()
        self.assertEqual(outcome, RunOutcome.PILOT_EXPIRED)

    # -------------------------------------------------------------------------
    # CONTROL 10: Concurrent Invocation Serialized or Rejected
    # -------------------------------------------------------------------------
    def test_control_10_concurrent_invocation_rejected(self):
        """Active lock file causes concurrent runner to fail closed without corrupting state."""
        runner1 = ScheduledRunner(
            symbol="FIX",
            state_dir=self.state_dir,
            output_dir=self.output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
        )

        # Acquire lock in runner 1 context
        with runner1.state_mgr.acquire_lock():
            runner2 = ScheduledRunner(
                symbol="FIX",
                state_dir=self.state_dir,
                output_dir=self.output_dir,
                cache_dir=self.cache_dir,
                force_run=True,
            )
            # Runner 2 attempts to run while runner 1 holds lock
            outcome, summary = runner2.run()
            self.assertEqual(outcome, RunOutcome.STATE_PERSISTENCE_FAILURE)
            self.assertIn("CONCURRENT_RUN_REJECTED", summary["error"])

    # -------------------------------------------------------------------------
    # CONTROL 11: Known Baseline Accessions Pre-Populated
    # -------------------------------------------------------------------------
    def test_control_11_known_baseline_accessions_pre_populated(self):
        """Genesis state seeds RECORDED_KNOWN_ACCESSIONS so PR #74 worked example is never re-announced."""
        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate()

        for acc in RECORDED_KNOWN_ACCESSIONS:
            self.assertTrue(
                mgr.is_accession_completed(acc),
                f"Expected known baseline accession {acc} to be completed at genesis",
            )

    # -------------------------------------------------------------------------
    # CONTROL 12: Boundary Violation Check
    # -------------------------------------------------------------------------
    def test_control_12_non_fix_symbol_rejected(self):
        """Attempting to run pilot against any ticker other than FIX raises ValueError immediately."""
        with self.assertRaises(ValueError) as ctx:
            ScheduledRunner(symbol="DUOL", state_dir=self.state_dir)
        self.assertIn("BOUNDARY_VIOLATION", str(ctx.exception))


    # -------------------------------------------------------------------------
    # CONTROL 13: Cross-Run Recovery and Fail-Closed on Missing State
    # -------------------------------------------------------------------------
    def test_control_13_cross_run_recovery_and_fail_closed_on_missing_state(self):
        """State is recovered across runs; missing state after prior runs fails closed with StateRecoveryError."""
        backup_dir = Path(self.temp_dir) / "backup_state"
        mgr1 = DurableStateManager(self.state_dir, backup_dir=backup_dir)
        mgr1.hydrate_or_recover()
        # Record an event
        evt = MaintenanceEvent(
            event_id="EVT-FIX-test-accn",
            symbol="FIX",
            cik=TARGET_CIK,
            company_name="Comfort Systems USA, Inc.",
            form="10-Q",
            accession_number="0001104659-26-888888",
            filing_date="2026-10-24",
            reporting_period="2026-09-30",
            retrieval_time_utc="2026-10-24T21:05:00Z",
            source_url="https://www.sec.gov/test",
            primary_document="doc.htm",
            processing_state=EventProcessingState.REVIEW_PENDING,
        )
        mgr1.record_completed_event(evt, review_packet_id="REV-TEST-888")
        mgr1.commit_run_checkpoint(RunOutcome.NEWLY_VERIFIED_EVENT_SUPPORTED)
        self.assertEqual(mgr1.run_count, 1)
        self.assertEqual(mgr1.checkpoint_sequence, 1)

        # Wipe local state_dir (simulating ephemeral GitHub Actions runner)
        shutil.rmtree(self.state_dir)
        self.assertFalse(self.state_dir.exists())

        # Runner B recovers from backup_dir
        mgr2 = DurableStateManager(self.state_dir, backup_dir=backup_dir)
        ok, msg = mgr2.hydrate_or_recover()
        self.assertTrue(ok)
        self.assertEqual(mgr2.run_count, 1)
        self.assertEqual(mgr2.checkpoint_sequence, 1)
        self.assertTrue(mgr2.is_accession_completed("0001104659-26-888888"))

        # Wipe local state_dir AND remove backup manifest, but leave marker that prior runs existed
        shutil.rmtree(self.state_dir)
        (backup_dir / "DURABLE_STATE_MANIFEST.json").unlink()
        self.assertTrue((backup_dir / ".has_prior_runs").exists())

        mgr3 = DurableStateManager(self.state_dir, backup_dir=backup_dir)
        with self.assertRaises(StateRecoveryError) as ctx:
            mgr3.hydrate_or_recover(allow_remote_artifact=False)
        self.assertIn("STATE_RECOVERY_FAILURE", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 14: Corrupted State Rejected with StateIntegrityError
    # -------------------------------------------------------------------------
    def test_control_14_corrupted_state_rejected_with_integrity_error(self):
        """Tampered or corrupted manifest file fails checksum verification with StateIntegrityError."""
        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate_or_recover()
        mgr.commit_run_checkpoint(RunOutcome.NO_QUALIFYING_EVENT)

        # Tamper with the manifest content on disk
        manifest_path = self.state_dir / "DURABLE_STATE_MANIFEST.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["run_count"] = 999  # Tampered run count without updating SHA-256
        manifest_path.write_text(json.dumps(data), encoding="utf-8")

        mgr_tampered = DurableStateManager(self.state_dir)
        with self.assertRaises(StateIntegrityError) as ctx:
            mgr_tampered.hydrate_or_recover()
        self.assertIn("Manifest SHA-256 mismatch", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 15: Multiple Events Chronological Ordering and Retryability
    # -------------------------------------------------------------------------
    def test_control_15_multiple_events_chronological_and_retryability(self):
        """Poller orders events chronologically; failed events remain retryable."""
        accn1 = "0001104659-26-900001"
        accn2 = "0001104659-26-900002"

        subs_mock = {
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-Q"],
                    "filingDate": ["2026-10-24", "2026-10-25"],
                    "reportDate": ["2026-09-30", "2026-09-30"],
                    "acceptanceDateTime": ["2026-10-24T21:00:00Z", "2026-10-25T21:00:00Z"],
                    "accessionNumber": [accn1, accn2],
                    "primaryDocument": ["doc1.htm", "doc2.htm"],
                }
            }
        }

        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate_or_recover()

        service = ResearchMaintenanceService(
            packets_dir=self.output_dir,
            cache_dir=self.cache_dir,
            completed_accessions=mgr.completed_accessions,
        )

        with patch.object(service.poller, "fetch_submissions", return_value=subs_mock):
            with patch.object(service.poller, "retrieve_filing_artifacts", return_value=("h1", "h2", {})):
                # First run should discover the EARLIEST uncompleted filing (accn1, filingDate 2026-10-24)
                event, _ = service.poller.discover_filing_event(
                    symbol="FIX",
                    cik=TARGET_CIK,
                    company_name="Comfort Systems USA, Inc.",
                    cutoff_date=ACCEPTED_CUTOFF_DATE,
                )
                self.assertIsNotNone(event)
                self.assertEqual(event.accession_number, accn1)

                # Record failure on accn1: it remains retryable (not completed)
                mgr.record_failed_event(event, stage="PIPELINE", error_msg="Simulated failure")
                self.assertFalse(mgr.is_accession_completed(accn1))

                # Mark accn1 completed now: subsequent scan must discover accn2
                mgr.record_completed_event(event, review_packet_id="REV-001")
                self.assertTrue(mgr.is_accession_completed(accn1))

                # Next run should find accn2
                service.poller.completed_accessions = mgr.completed_accessions
                event2, _ = service.poller.discover_filing_event(
                    symbol="FIX",
                    cik=TARGET_CIK,
                    company_name="Comfort Systems USA, Inc.",
                    cutoff_date=ACCEPTED_CUTOFF_DATE,
                )
                self.assertIsNotNone(event2)
                self.assertEqual(event2.accession_number, accn2)

    # -------------------------------------------------------------------------
    # CONTROL 16: Budget Request Limit Enforced
    # -------------------------------------------------------------------------
    def test_control_16_budget_request_limit_enforced(self):
        """Active RunBudgetTracker strictly blocks 6th request attempt with BudgetExhaustedError."""
        tracker = RunBudgetTracker(max_requests=5, max_runtime_seconds=120, min_request_interval_seconds=0.001)
        for i in range(5):
            tracker.acquire_request_slot()
        self.assertEqual(tracker.request_count, 5)

        # 6th attempt must raise BudgetExhaustedError and never start
        with self.assertRaises(BudgetExhaustedError) as ctx:
            tracker.acquire_request_slot()
        self.assertIn("REQUEST_BUDGET_EXHAUSTED", str(ctx.exception))
        self.assertEqual(tracker.request_count, 5)

    # -------------------------------------------------------------------------
    # CONTROL 17: Budget Runtime Deadline Enforced
    # -------------------------------------------------------------------------
    def test_control_17_budget_runtime_deadline_enforced(self):
        """Active RunBudgetTracker raises BudgetExhaustedError when monotonic runtime deadline expires."""
        tracker = RunBudgetTracker(max_requests=5, max_runtime_seconds=1.0, min_request_interval_seconds=0.001)
        # Advance deadline artificially by setting deadline in the past
        tracker.deadline = time.monotonic() - 1.0

        with self.assertRaises(BudgetExhaustedError) as ctx:
            tracker.acquire_request_slot()
        self.assertIn("RUNTIME_BUDGET_EXHAUSTED", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 18: Atomic Lock Creation Prevents Race
    # -------------------------------------------------------------------------
    def test_control_18_atomic_lock_creation_prevents_race(self):
        """Mutual exclusion lock uses atomic OS-level file creation to prevent concurrent runs."""
        mgr1 = DurableStateManager(self.state_dir)
        with mgr1.acquire_lock():
            # Concurrently attempt with mgr2
            mgr2 = DurableStateManager(self.state_dir)
            with self.assertRaises(ConcurrentRunError) as ctx:
                with mgr2.acquire_lock(timeout_seconds=60.0):
                    pass
            self.assertIn("CONCURRENT_RUN_REJECTED", str(ctx.exception))

        # After mgr1 exits, mgr2 can acquire lock
        with mgr2.acquire_lock():
            self.assertTrue(mgr2.lock_path.exists())


if __name__ == "__main__":
    unittest.main()

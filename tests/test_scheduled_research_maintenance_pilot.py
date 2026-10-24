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
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
import urllib.error
import zipfile

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
    OPERATIONAL_STATE_DIR,
    OPERATIONAL_OUTPUT_DIR,
    OPERATIONAL_CACHE_DIR,
    REHEARSAL_STATE_DIR,
    REHEARSAL_OUTPUT_DIR,
    REHEARSAL_CACHE_DIR,
    validate_rehearsal_paths,
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
    validate_github_artifact_list_response,
    validate_github_workflow_runs_response,
    validate_github_run_jobs_response,
    validate_github_workflow_run_response,
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
                allow_bootstrap=True,
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
            allow_bootstrap=True,
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
            allow_bootstrap=True,
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
            allow_bootstrap=True,
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
            allow_bootstrap=True,
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
            allow_bootstrap=True,
        )

        # Acquire lock in runner 1 context
        with runner1.state_mgr.acquire_lock():
            runner2 = ScheduledRunner(
                symbol="FIX",
                state_dir=self.state_dir,
                output_dir=self.output_dir,
                cache_dir=self.cache_dir,
                force_run=True,
                allow_bootstrap=True,
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
        mgr1.hydrate_or_recover(allow_bootstrap=True)
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

        # Runner B recovers from backup_dir (without bootstrap needed, since backup manifest exists)
        mgr2 = DurableStateManager(self.state_dir, backup_dir=backup_dir)
        ok, msg = mgr2.hydrate_or_recover(allow_bootstrap=False)
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
        mgr.hydrate_or_recover(allow_bootstrap=True)
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
        mgr.hydrate_or_recover(allow_bootstrap=True)

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

    # -------------------------------------------------------------------------
    # CONTROL 19: Recovery from Mocked GitHub Artifact ZIP
    # -------------------------------------------------------------------------
    def test_control_19_recovery_from_mocked_github_artifact_zip(self):
        """Runner recovers state from remote GitHub Actions artifact ZIP and validates schema."""
        from analyst_dashboard.research.maintenance.durable_state import deterministic_json, sha256_hex

        # Build in-memory artifact ZIP containing a valid manifest
        manifest_data = {
            "manifest_version": "ARX-MAINTENANCE-STATE-V1",
            "symbol": "FIX",
            "cik": "0001035983",
            "checkpoint_sequence": 5,
            "prev_checkpoint_hash": "a" * 64,
            "created_at_utc": "2026-10-24T12:00:00Z",
            "last_checked_utc": "2026-10-24T18:00:00Z",
            "last_outcome": "NO_QUALIFYING_EVENT",
            "run_count": 5,
            "completed_accessions": sorted(list(RECORDED_KNOWN_ACCESSIONS | {"0001104659-26-999999"})),
            "events": {},
        }
        manifest_copy = dict(manifest_data)
        manifest_copy.pop("manifest_sha256", None)
        manifest_data["manifest_sha256"] = sha256_hex(deterministic_json(manifest_copy).encode("utf-8"))

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("durable_state/research_maintenance/DURABLE_STATE_MANIFEST.json", json.dumps(manifest_data, indent=2))
        zip_bytes = zip_buf.getvalue()

        # Mock GitHub API responses:
        # 1. artifacts list: returns artifact info with archive_download_url
        # 2. archive download: returns zip_bytes
        artifacts_resp = {
            "total_count": 1,
            "artifacts": [
                {
                    "id": 88888,
                    "name": "research-maintenance-fix-state",
                    "archive_download_url": "https://api.github.com/repos/daakara/finance/actions/artifacts/88888/zip",
                    "workflow_run": {
                        "id": 55555,
                    },
                }
            ],
        }

        run_resp = {
            "id": 55555,
            "path": ".github/workflows/scheduled_research_maintenance.yml",
            "repository": {
                "full_name": "daakara/finance",
            },
        }

        class MockHTTPResponse:
            def __init__(self, data_bytes):
                self._data = data_bytes

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def mock_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "artifacts/88888/zip" in url:
                return MockHTTPResponse(zip_bytes)
            elif "actions/artifacts" in url:
                return MockHTTPResponse(json.dumps(artifacts_resp).encode("utf-8"))
            elif "actions/runs/55555" in url:
                return MockHTTPResponse(json.dumps(run_resp).encode("utf-8"))
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token", "GITHUB_REPOSITORY": "daakara/finance"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen):
                mgr = DurableStateManager(self.state_dir)
                ok, msg = mgr.hydrate_or_recover(allow_bootstrap=False)

                self.assertTrue(ok)
                self.assertEqual(mgr.checkpoint_sequence, 5)
                self.assertEqual(mgr.run_count, 5)
                self.assertTrue(mgr.is_accession_completed("0001104659-26-999999"))
                self.assertTrue((self.state_dir / "DURABLE_STATE_MANIFEST.json").exists())

    # -------------------------------------------------------------------------
    # CONTROL 20: Recovery Fails Closed When History Unavailable or Unauthorized
    # -------------------------------------------------------------------------
    def test_control_20_recovery_fails_closed_when_history_unavailable_or_unauthorized(self):
        """HTTP error or unauthorized history response raises StateRecoveryError and never defaults to genesis."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("https://api.github.com", 401, "Unauthorized", {}, None)):
                mgr = DurableStateManager(self.state_dir)
                # Even if allow_bootstrap=True, unavailable history fails closed!
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=True)
                self.assertIn("STATE_RECOVERY_FAILURE", str(ctx.exception))
                self.assertFalse((self.state_dir / "DURABLE_STATE_MANIFEST.json").exists())

    # -------------------------------------------------------------------------
    # CONTROL 21: Dormant Runs Do Not Count as Prior Operational Execution
    # -------------------------------------------------------------------------
    def test_control_21_dormant_runs_do_not_count_as_prior_operational_execution(self):
        """A successful dormant workflow run (no state artifact) is not counted as prior operational execution."""
        class MockHTTPResponse:
            def __init__(self, data_bytes):
                self._data = data_bytes

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # Sub-case A: Successful dormant run (stateful steps skipped) -> bootstrap permitted
        def mock_urlopen_dormant(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts?name=" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs?status=success" in url:
                return MockHTTPResponse(b'{"total_count": 1, "workflow_runs": [{"id": 1111}]}')
            elif "runs/1111/artifacts" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs/1111/jobs" in url:
                jobs_data = {
                    "total_count": 1,
                    "jobs": [
                        {
                            "id": 11111,
                            "name": "Detect Periodic Filing & Prepare Review Packet (FIX)",
                            "status": "completed",
                            "steps": [
                                {"name": "1. Checkout Repository", "status": "completed", "conclusion": "success"},
                                {"name": "4. Restore Durable State Cache", "status": "completed", "conclusion": "skipped"},
                                {"name": "6. Save Durable State Cache", "status": "completed", "conclusion": "skipped"},
                                {"name": "7. Upload State and Review Artifacts", "status": "completed", "conclusion": "skipped"},
                            ],
                        }
                    ],
                }
                return MockHTTPResponse(json.dumps(jobs_data).encode("utf-8"))
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_dormant):
                mgr = DurableStateManager(self.state_dir)
                ok, msg = mgr.hydrate_or_recover(allow_bootstrap=True)
                self.assertTrue(ok)
                self.assertEqual(msg, "GENESIS_INITIALIZED")

        # Sub-case B: Successful operational run found in run artifacts -> bootstrap blocked without prior state
        state_dir_2 = Path(self.temp_dir) / "state_sub_b"
        def mock_urlopen_operational(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts?name=" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs?status=success" in url:
                return MockHTTPResponse(b'{"total_count": 1, "workflow_runs": [{"id": 2222}]}')
            elif "runs/2222/artifacts" in url:
                return MockHTTPResponse(b'{"total_count": 1, "artifacts": [{"id": 777, "name": "research-maintenance-fix-state", "archive_download_url": "https://api.github.com/repos/daakara/finance/actions/artifacts/777/zip"}]}')
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_operational):
                mgr2 = DurableStateManager(state_dir_2)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr2.hydrate_or_recover(allow_bootstrap=True)
                self.assertIn("Evidence of prior operational runs exists", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 22: Rehearsal Namespace Isolation Leaves Operational State Untouched
    # -------------------------------------------------------------------------
    def test_control_22_rehearsal_namespace_isolation_leaves_operational_state_untouched(self):
        """Rehearsal run operates strictly in rehearsal namespace and leaves operational manifest byte-identical."""
        op_state_dir = Path(self.temp_dir) / "op_state"
        reh_state_dir = Path(self.temp_dir) / "reh_state"
        reh_output_dir = Path(self.temp_dir) / "reh_output"

        # Initialize operational state
        op_mgr = DurableStateManager(op_state_dir)
        op_mgr.hydrate_or_recover(allow_bootstrap=True)
        op_mgr.commit_run_checkpoint(RunOutcome.NO_QUALIFYING_EVENT)
        op_manifest_path = op_state_dir / "DURABLE_STATE_MANIFEST.json"
        op_manifest_bytes_before = op_manifest_path.read_bytes()
        op_sha_before = hashlib.sha256(op_manifest_bytes_before).hexdigest()
        self.assertEqual(op_mgr.run_count, 1)

        # Execute rehearsal run
        reh_runner = ScheduledRunner(
            symbol="FIX",
            state_dir=reh_state_dir,
            output_dir=reh_output_dir,
            cache_dir=self.cache_dir,
            force_run=True,
            is_rehearsal=True,
            allow_bootstrap=True,
            offline_mode=True,
        )
        self.assertEqual(reh_runner.artifact_name, "research-maintenance-fix-rehearsal-state")

        outcome, summary = reh_runner.run()
        self.assertIn(outcome, (RunOutcome.NO_QUALIFYING_EVENT, RunOutcome.SEC_RETRIEVAL_FAILURE))

        # Operational manifest must be 100% bit-for-bit unchanged
        op_manifest_bytes_after = op_manifest_path.read_bytes()
        op_sha_after = hashlib.sha256(op_manifest_bytes_after).hexdigest()
        self.assertEqual(op_sha_before, op_sha_after, "Operational manifest was mutated by rehearsal!")

        # Rehearsal manifest exists in separate rehearsal directory
        self.assertTrue((reh_state_dir / "DURABLE_STATE_MANIFEST.json").exists())

    # -------------------------------------------------------------------------
    # CONTROL 23: Expired Pilot Halts at Step 0 Before Lock or Mutation
    # -------------------------------------------------------------------------
    def test_control_23_expired_pilot_halts_at_step_0_before_lock_or_mutation(self):
        """Expired pilot halts at step 0 without creating state directories, acquiring locks, or network calls."""
        uncreated_dir = Path(self.temp_dir) / "uncreated_state_dir"
        self.assertFalse(uncreated_dir.exists())

        with patch("analyst_dashboard.research.maintenance.pilot_config.datetime") as mock_dt:
            # Set time after PILOT_END_DATE (2026-11-24)
            fake_now = datetime(2026, 11, 26, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = fake_now
            mock_dt.timezone = timezone

            with patch("urllib.request.urlopen", side_effect=AssertionError("Network called when expired")):
                runner = ScheduledRunner(
                    symbol="FIX",
                    state_dir=uncreated_dir,
                    output_dir=self.output_dir,
                    cache_dir=self.cache_dir,
                    force_run=True,  # Even force_run must not bypass lifecycle expiration!
                )
                outcome, summary = runner.run()

                self.assertEqual(outcome, RunOutcome.PILOT_EXPIRED)
                self.assertTrue(summary["is_expired"])
                self.assertFalse(uncreated_dir.exists(), "State directory was created by expired pilot!")

    # -------------------------------------------------------------------------
    # CONTROL 24: Dynamic Timeout Never Inflated Above Remaining Deadline
    # -------------------------------------------------------------------------
    def test_control_24_dynamic_timeout_never_inflated_above_remaining_deadline(self):
        """Dynamic timeout is strictly bounded by remaining runtime (never inflated with max(0.5, ...))."""
        # Create tracker with mock clock
        clock_time = [1000.0]
        def mock_clock():
            return clock_time[0]

        tracker = RunBudgetTracker(
            max_requests=5,
            max_runtime_sec=10.0,
            min_interval_sec=0.0,
            clock_fn=mock_clock,
        )
        # Advance clock to leave only 0.3s remaining
        clock_time[0] = 1009.7
        self.assertAlmostEqual(tracker.remaining_runtime(), 0.3, places=2)

        timeout = tracker.acquire_request_slot(default_timeout_sec=15.0)
        self.assertAlmostEqual(timeout, 0.3, places=2)
        self.assertLessEqual(timeout, 0.3, "Timeout was inflated above remaining runtime!")

        # Advance clock to leave < 0.1s remaining (e.g. 0.05s)
        clock_time[0] = 1009.96
        with self.assertRaises(BudgetExhaustedError) as ctx:
            tracker.acquire_request_slot()
        self.assertIn("Insufficient runtime", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 25: Recovery Requests Separated from SEC Budget
    # -------------------------------------------------------------------------
    def test_control_25_recovery_requests_separated_from_sec_budget(self):
        """Recovery HTTP requests are tracked separately and do not consume from the 5 SEC request ceiling."""
        tracker = RunBudgetTracker(max_requests=5, max_runtime_sec=120.0, min_interval_sec=0.0)

        # 3 recovery requests
        for _ in range(3):
            timeout = tracker.acquire_recovery_request_slot(default_timeout_sec=10.0)
            self.assertGreater(timeout, 0.0)

        self.assertEqual(tracker.recovery_request_count, 3)
        self.assertEqual(tracker.sec_request_count, 0)

        # 5 SEC requests permitted
        for _ in range(5):
            tracker.acquire_sec_request_slot()

        self.assertEqual(tracker.sec_request_count, 5)
        self.assertEqual(tracker.total_requests, 8)

        # 6th SEC request strictly rejected
        with self.assertRaises(BudgetExhaustedError) as ctx:
            tracker.acquire_sec_request_slot()
        self.assertIn("REQUEST_BUDGET_EXHAUSTED", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 26: Boundary Deadline Checks and Event Retryability
    # -------------------------------------------------------------------------
    def test_control_26_boundary_deadline_checks_and_event_retryability(self):
        """Consequential boundaries check deadline, and budget exhaustion leaves event retryable."""
        tracker = RunBudgetTracker(max_requests=5, max_runtime_sec=1.0)
        tracker.deadline = time.monotonic() - 1.0  # Expired

        with self.assertRaises(BudgetExhaustedError) as ctx:
            tracker.check_deadline("pre_reconciliation")
        self.assertIn("pre_reconciliation", str(ctx.exception))

        # Event failure due to budget exhaustion leaves event uncompleted (retryable)
        mgr = DurableStateManager(self.state_dir)
        mgr.hydrate_or_recover(allow_bootstrap=True)
        evt = MaintenanceEvent(
            event_id="EVT-FIX-test-retry",
            symbol="FIX",
            cik=TARGET_CIK,
            company_name="Comfort Systems USA, Inc.",
            form="10-Q",
            accession_number="0001104659-26-777777",
            filing_date="2026-10-24",
            reporting_period="2026-09-30",
            retrieval_time_utc="2026-10-24T21:05:00Z",
            source_url="https://www.sec.gov/test",
            primary_document="doc.htm",
            processing_state=EventProcessingState.REVIEW_PENDING,
        )
        mgr.record_failed_event(evt, stage="BUDGET_EXHAUSTED", error_msg="Timeout exceeded")
        # Assert accession is NOT completed
        self.assertFalse(mgr.is_accession_completed("0001104659-26-777777"))

    # -------------------------------------------------------------------------
    # CONTROL 27: Malformed Responses Fail Without Genesis
    # -------------------------------------------------------------------------
    def test_control_27_recovery_malformed_responses_fail_without_genesis(self):
        """Malformed, empty dict, missing fields, or incorrect types fail without genesis."""
        class MockHTTPResponse:
            def __init__(self, data_bytes):
                self._data = data_bytes

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # Case A: {} returned for artifacts
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", return_value=MockHTTPResponse(b"{}")):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=True)
                self.assertIn("missing required field 'total_count'", str(ctx.exception))
                self.assertFalse((self.state_dir / "DURABLE_STATE_MANIFEST.json").exists())

        # Case B: total_count is boolean or string or negative
        for bad_tc in (b'{"total_count": true, "artifacts": []}', b'{"total_count": "zero", "artifacts": []}', b'{"total_count": -1, "artifacts": []}'):
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
                with patch("urllib.request.urlopen", return_value=MockHTTPResponse(bad_tc)):
                    mgr = DurableStateManager(self.state_dir)
                    with self.assertRaises(StateRecoveryError) as ctx:
                        mgr.hydrate_or_recover(allow_bootstrap=True)
                    self.assertIn("invalid 'total_count'", str(ctx.exception))
                    self.assertFalse((self.state_dir / "DURABLE_STATE_MANIFEST.json").exists())

        # Case C: total_count < len(artifacts) (inconsistent)
        inconsistent = b'{"total_count": 0, "artifacts": [{"id": 1, "name": "foo", "archive_download_url": "http://x"}]}'
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", return_value=MockHTTPResponse(inconsistent)):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=True)
                self.assertIn("Inconsistent", str(ctx.exception))
                self.assertFalse((self.state_dir / "DURABLE_STATE_MANIFEST.json").exists())

    # -------------------------------------------------------------------------
    # CONTROL 28: Artifact Provenance & Workflow Association Verification
    # -------------------------------------------------------------------------
    def test_control_28_recovery_rejects_artifact_from_wrong_workflow_or_missing_provenance(self):
        """Artifacts must originate from daakara/finance and scheduled_research_maintenance.yml."""
        class MockHTTPResponse:
            def __init__(self, data_bytes):
                self._data = data_bytes

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # Sub-case A: Artifact missing workflow_run provenance
        bad_art_resp = {
            "total_count": 1,
            "artifacts": [
                {
                    "id": 99991,
                    "name": "research-maintenance-fix-state",
                    "archive_download_url": "https://api.github.com/zip",
                }
            ],
        }
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token", "GITHUB_REPOSITORY": "daakara/finance"}):
            with patch("urllib.request.urlopen", return_value=MockHTTPResponse(json.dumps(bad_art_resp).encode("utf-8"))):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=False)
                self.assertIn("missing 'workflow_run' provenance", str(ctx.exception))

        # Sub-case B: Artifact from wrong workflow (e.g. .github/workflows/deploy.yml)
        art_with_run = {
            "total_count": 1,
            "artifacts": [
                {
                    "id": 99992,
                    "name": "research-maintenance-fix-state",
                    "archive_download_url": "https://api.github.com/zip",
                    "workflow_run": {"id": 123456},
                }
            ],
        }
        wrong_wf_run = {
            "id": 123456,
            "path": ".github/workflows/deploy.yml",
            "repository": {"full_name": "daakara/finance"},
        }
        def mock_urlopen_wrong_wf(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts" in url:
                return MockHTTPResponse(json.dumps(art_with_run).encode("utf-8"))
            elif "actions/runs/123456" in url:
                return MockHTTPResponse(json.dumps(wrong_wf_run).encode("utf-8"))
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token", "GITHUB_REPOSITORY": "daakara/finance"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_wrong_wf):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=False)
                self.assertIn("unauthorized workflow", str(ctx.exception))

        # Sub-case C: Artifact from wrong repository
        wrong_repo_run = {
            "id": 123456,
            "path": ".github/workflows/scheduled_research_maintenance.yml",
            "repository": {"full_name": "evil-org/finance"},
        }
        def mock_urlopen_wrong_repo(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts" in url:
                return MockHTTPResponse(json.dumps(art_with_run).encode("utf-8"))
            elif "actions/runs/123456" in url:
                return MockHTTPResponse(json.dumps(wrong_repo_run).encode("utf-8"))
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token", "GITHUB_REPOSITORY": "daakara/finance"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_wrong_repo):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=False)
                self.assertIn("originates from repository 'evil-org/finance'", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 29: Missing Historical Artifacts Do Not Automatically Establish Dormancy
    # -------------------------------------------------------------------------
    def test_control_29_missing_historical_artifacts_do_not_automatically_establish_dormancy(self):
        """Absence of state artifacts does not prove dormancy if execution evidence shows state steps executed."""
        class MockHTTPResponse:
            def __init__(self, data_bytes):
                self._data = data_bytes

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # Run 3333 had 0 artifacts (e.g. expired or deleted), but its jobs show "Upload State and Review Artifacts" succeeded!
        def mock_urlopen_expired_operational(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts?name=" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs?status=success" in url:
                return MockHTTPResponse(b'{"total_count": 1, "workflow_runs": [{"id": 3333}]}')
            elif "runs/3333/artifacts" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs/3333/jobs" in url:
                jobs_data = {
                    "total_count": 1,
                    "jobs": [
                        {
                            "id": 33331,
                            "name": "Detect Periodic Filing & Prepare Review Packet (FIX)",
                            "status": "completed",
                            "steps": [
                                {"name": "1. Checkout Repository", "status": "completed", "conclusion": "success"},
                                {"name": "4. Restore Durable State Cache", "status": "completed", "conclusion": "success"},
                                {"name": "6. Save Durable State Cache", "status": "completed", "conclusion": "success"},
                                {"name": "7. Upload State and Review Artifacts", "status": "completed", "conclusion": "success"},
                            ],
                        }
                    ],
                }
                return MockHTTPResponse(json.dumps(jobs_data).encode("utf-8"))
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_expired_operational):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=True)
                # Fails closed because prior operational run executed, never establishing false dormancy or genesis!
                self.assertIn("Evidence of prior operational runs exists", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 30: Truncated History Cannot Establish Initialization
    # -------------------------------------------------------------------------
    def test_control_30_incomplete_or_truncated_history_cannot_establish_initialization(self):
        """Truncated page of workflow runs or missing jobs cannot establish first initialization."""
        class MockHTTPResponse:
            def __init__(self, data_bytes):
                self._data = data_bytes

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        # total_count=10 but workflow_runs has only 2 items (truncated)
        def mock_urlopen_truncated(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts?name=" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs?status=success" in url:
                return MockHTTPResponse(b'{"total_count": 10, "workflow_runs": [{"id": 1}, {"id": 2}]}')
            raise ValueError(f"Unexpected url: {url}")

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_truncated):
                mgr = DurableStateManager(self.state_dir)
                with self.assertRaises(StateRecoveryError) as ctx:
                    mgr.hydrate_or_recover(allow_bootstrap=True)
                self.assertIn("Remote workflow run history is truncated", str(ctx.exception))

        # Case B: Genuinely empty history (total_count=0) with explicit bootstrap permits genuine initialization
        def mock_urlopen_empty(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "actions/artifacts?name=" in url:
                return MockHTTPResponse(b'{"total_count": 0, "artifacts": []}')
            elif "runs?status=success" in url:
                return MockHTTPResponse(b'{"total_count": 0, "workflow_runs": []}')
            raise ValueError(f"Unexpected url: {url}")

        state_dir_empty = Path(self.temp_dir) / "state_empty"
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "mock-token"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_empty):
                mgr_empty = DurableStateManager(state_dir_empty)
                ok, msg = mgr_empty.hydrate_or_recover(allow_bootstrap=True)
                self.assertTrue(ok)
                self.assertEqual(msg, "GENESIS_INITIALIZED")
                self.assertTrue((state_dir_empty / "DURABLE_STATE_MANIFEST.json").exists())

    # -------------------------------------------------------------------------
    # CONTROL 31: Rehearsal Rejects Explicit Operational Paths & Contradictions
    # -------------------------------------------------------------------------
    def test_control_31_rehearsal_rejects_explicit_operational_paths_and_contradictory_settings(self):
        """Rehearsal invocation rejects operational paths and contradictory force_run settings."""
        # 1. Contradictory settings: force_run=True, is_rehearsal=False
        with self.assertRaises(ValueError) as ctx:
            ScheduledRunner(
                symbol="FIX",
                force_run=True,
                is_rehearsal=False,
            )
        self.assertIn("Contradictory settings", str(ctx.exception))

        # 2. Explicit operational state_dir passed to rehearsal
        with self.assertRaises(ValueError) as ctx:
            ScheduledRunner(
                symbol="FIX",
                state_dir=REPO_ROOT / OPERATIONAL_STATE_DIR,
                is_rehearsal=True,
            )
        self.assertIn("matches protected operational state dir", str(ctx.exception))

        # 3. Explicit operational output_dir passed to rehearsal
        with self.assertRaises(ValueError) as ctx:
            ScheduledRunner(
                symbol="FIX",
                output_dir=REPO_ROOT / OPERATIONAL_OUTPUT_DIR,
                is_rehearsal=True,
            )
        self.assertIn("matches protected operational output dir", str(ctx.exception))

        # 4. Explicit operational cache_dir passed to rehearsal
        with self.assertRaises(ValueError) as ctx:
            ScheduledRunner(
                symbol="FIX",
                cache_dir=REPO_ROOT / OPERATIONAL_CACHE_DIR,
                is_rehearsal=True,
            )
        self.assertIn("matches protected operational cache dir", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 32: Rehearsal Rejects Parent/Child Overlaps and Aliases
    # -------------------------------------------------------------------------
    def test_control_32_rehearsal_rejects_parent_child_overlaps_and_aliases(self):
        """Rehearsal paths cannot contain, be contained in, or alias protected operational roots."""
        # 1. Parent overlap (state_dir contains operational state root)
        with self.assertRaises(ValueError) as ctx:
            validate_rehearsal_paths(
                state_dir=REPO_ROOT / "durable_state",
                output_dir=self.output_dir,
                cache_dir=self.cache_dir,
                repo_root=REPO_ROOT,
            )
        self.assertIn("contains protected operational state dir", str(ctx.exception))

        # 2. Child overlap (output_dir is inside operational output root)
        with self.assertRaises(ValueError) as ctx:
            validate_rehearsal_paths(
                state_dir=self.state_dir,
                output_dir=REPO_ROOT / OPERATIONAL_OUTPUT_DIR / "nested_subdir",
                cache_dir=self.cache_dir,
                repo_root=REPO_ROOT,
            )
        self.assertIn("is inside protected operational output dir", str(ctx.exception))

        # 3. Parent overlap: cache_dir = repo_root
        with self.assertRaises(ValueError) as ctx:
            validate_rehearsal_paths(
                state_dir=self.state_dir,
                output_dir=self.output_dir,
                cache_dir=REPO_ROOT,
                repo_root=REPO_ROOT,
            )
        self.assertIn("contains protected", str(ctx.exception))

        # 4. Symlink alias pointing to operational directory
        symlink_dir = Path(self.temp_dir) / "symlink_to_op_state"
        try:
            os.symlink(REPO_ROOT / OPERATIONAL_STATE_DIR, symlink_dir, target_is_directory=True)
            has_symlink = True
        except (OSError, NotImplementedError):
            has_symlink = False

        if has_symlink:
            with self.assertRaises(ValueError) as ctx:
                validate_rehearsal_paths(
                    state_dir=symlink_dir,
                    output_dir=self.output_dir,
                    cache_dir=self.cache_dir,
                    repo_root=REPO_ROOT,
                )
            self.assertIn("matches protected operational state dir", str(ctx.exception))

    # -------------------------------------------------------------------------
    # CONTROL 33: Safe Custom Rehearsal Paths and Manifest Immutability
    # -------------------------------------------------------------------------
    def test_control_33_rehearsal_safe_custom_paths_and_manifest_immutability(self):
        """Safe custom paths work, while failed or valid rehearsals leave operational manifest untouched."""
        op_state_dir = Path(self.temp_dir) / "op_state_c33"
        op_mgr = DurableStateManager(op_state_dir)
        op_mgr.hydrate_or_recover(allow_bootstrap=True)
        op_mgr.commit_run_checkpoint(RunOutcome.NO_QUALIFYING_EVENT)
        op_mgr.commit_run_checkpoint(RunOutcome.NO_QUALIFYING_EVENT)
        self.assertEqual(op_mgr.run_count, 2)

        op_manifest_path = op_state_dir / "DURABLE_STATE_MANIFEST.json"
        op_bytes_before = op_manifest_path.read_bytes()
        op_sha_before = hashlib.sha256(op_bytes_before).hexdigest()

        # Custom safe temporary rehearsal paths
        custom_reh_state = Path(self.temp_dir) / "custom_rehearsal_state"
        custom_reh_output = Path(self.temp_dir) / "custom_rehearsal_output"
        custom_reh_cache = Path(self.temp_dir) / "custom_rehearsal_cache"

        reh_runner = ScheduledRunner(
            symbol="FIX",
            state_dir=custom_reh_state,
            output_dir=custom_reh_output,
            cache_dir=custom_reh_cache,
            force_run=True,
            is_rehearsal=True,
            allow_bootstrap=True,
            offline_mode=True,
        )
        self.assertEqual(reh_runner.artifact_name, "research-maintenance-fix-rehearsal-state")

        outcome, summary = reh_runner.run()
        self.assertIn(outcome, (RunOutcome.NO_QUALIFYING_EVENT, RunOutcome.SEC_RETRIEVAL_FAILURE))
        self.assertTrue((custom_reh_state / "DURABLE_STATE_MANIFEST.json").exists())

        # Operational manifest is 100% bit-for-bit unchanged
        op_bytes_after = op_manifest_path.read_bytes()
        op_sha_after = hashlib.sha256(op_bytes_after).hexdigest()
        self.assertEqual(op_sha_before, op_sha_after, "Operational manifest mutated by rehearsal run!")

        # Attempt an unsafe rehearsal that gets rejected, verifying no mutation occurs
        with self.assertRaises(ValueError):
            ScheduledRunner(
                symbol="FIX",
                state_dir=REPO_ROOT / OPERATIONAL_STATE_DIR,
                is_rehearsal=True,
            )

        op_bytes_after_rejected = op_manifest_path.read_bytes()
        self.assertEqual(op_sha_before, hashlib.sha256(op_bytes_after_rejected).hexdigest())


if __name__ == "__main__":
    unittest.main()

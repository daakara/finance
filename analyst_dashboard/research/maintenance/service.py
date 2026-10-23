"""
analyst_dashboard/research/maintenance/service.py

Maintenance Pipeline Service for ARX Research Dossiers.
Orchestrates:
Primary-source event
→ deduplicated evidence record
→ reconciled draft financial changes
→ assessment of affected research conclusions
→ reviewable update packet

Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from .contracts import (
    EventProcessingState,
    EventRegister,
    MaintenanceEvent,
    ProposedDisposition,
    ReviewPacket,
)
from .event_poller import SecEventPoller
from .fact_reconciler import FactReconciler
from .decision_assessor import DecisionAssessor
from .packet_generator import PacketGenerator

logger = logging.getLogger(__name__)

# Default file authorities
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_REGISTER_PATH = REPO_ROOT / "docs/research/maintenance/event_register.json"
DEFAULT_PACKETS_DIR = REPO_ROOT / "docs/research/maintenance/packets"
DEFAULT_BASELINE_SNAPSHOT = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
DEFAULT_PUBLISHED_MANIFEST = REPO_ROOT / "frontend/lib/publishedWatchManifest.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "data/research/sourced_fundamentals/raw"

PUBLISHED_SYMBOLS_CIK = {
    "FIX": {"name": "Comfort Systems USA, Inc.", "cik": "0001035983"},
    "DUOL": {"name": "Duolingo, Inc.", "cik": "0001562088"},
    "ULTA": {"name": "Ulta Beauty, Inc.", "cik": "0001403568"},
    "MEDP": {"name": "Medpace Holdings, Inc.", "cik": "0001668397"},
    "ISRG": {"name": "Intuitive Surgical, Inc.", "cik": "0001035267"},
    "DECK": {"name": "Deckers Outdoor Corp.", "cik": "0000910521"},
}


class ResearchMaintenanceService:
    """
    Orchestrates the bounded research-maintenance pilot for ARX Terminal.
    """

    def __init__(
        self,
        register_path: Optional[Union[str, Path]] = None,
        packets_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        baseline_snapshot_path: Optional[Union[str, Path]] = None,
        published_manifest_path: Optional[Union[str, Path]] = None,
        offline_mode: bool = False,
    ):
        self.register_path = Path(register_path) if register_path else DEFAULT_REGISTER_PATH
        self.packets_dir = Path(packets_dir) if packets_dir else DEFAULT_PACKETS_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.baseline_snapshot_path = Path(baseline_snapshot_path) if baseline_snapshot_path else DEFAULT_BASELINE_SNAPSHOT
        self.published_manifest_path = Path(published_manifest_path) if published_manifest_path else DEFAULT_PUBLISHED_MANIFEST
        self.offline_mode = offline_mode

        self.register = EventRegister(self.register_path)
        self.poller = SecEventPoller(self.register, cache_dir=self.cache_dir, offline_mode=self.offline_mode)
        self.reconciler = FactReconciler(cache_dir=self.cache_dir, offline_mode=self.offline_mode)
        self.assessor = DecisionAssessor()
        self.packet_generator = PacketGenerator(output_dir=self.packets_dir)

    def load_accepted_baseline_dossier(self, symbol: str) -> Dict[str, Any]:
        """Loads accepted baseline dossier from snapshot or published manifest."""
        clean_sym = symbol.strip().upper()

        # 1. Check baseline snapshot (FIX, DUOL, IREN, CORZ, MU)
        if self.baseline_snapshot_path.exists():
            with open(self.baseline_snapshot_path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            if clean_sym in snap.get("dossiers", {}):
                return snap["dossiers"][clean_sym]

        # 2. Check published watch manifest (ULTA, MEDP, ISRG, DECK)
        if self.published_manifest_path.exists():
            with open(self.published_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if clean_sym in manifest.get("accepted_additions", {}):
                return manifest["accepted_additions"][clean_sym]

        raise KeyError(f"Symbol {clean_sym} not found in accepted baseline research.")

    def run_maintenance_pipeline(
        self,
        symbol: str,
        target_accession: Optional[str] = None,
        cutoff_date: str = "2026-10-16",
        is_replay: bool = False,
        facts_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ReviewPacket], Optional[MaintenanceEvent], List[str]]:
        """
        Executes complete maintenance path for one published company:
        Primary-source event -> Deduplicated evidence record -> Reconciled facts -> Assessment -> Review packet.
        """
        clean_sym = symbol.strip().upper()
        if clean_sym not in PUBLISHED_SYMBOLS_CIK:
            return None, None, [f"Symbol '{clean_sym}' is not among the six published Quality Watch assets."]

        meta_info = PUBLISHED_SYMBOLS_CIK[clean_sym]
        cik = meta_info["cik"]
        company_name = meta_info["name"]

        # Step 1: Load accepted baseline dossier
        baseline_dossier = self.load_accepted_baseline_dossier(clean_sym)
        baseline_profile = baseline_dossier.get("financial_profile", {})

        # Step 2: Poll / Discover filing event
        event, poll_warnings = self.poller.discover_filing_event(
            symbol=clean_sym,
            cik=cik,
            company_name=company_name,
            cutoff_date=cutoff_date,
            target_accession=target_accession,
            is_replay=is_replay,
        )

        if event is None:
            return None, None, poll_warnings

        # Check if rejected non-periodic form (e.g. 8-K)
        if event.processing_state == EventProcessingState.REJECTED_MISIDENTIFIED:
            self.register.register_event(event, fail_if_duplicate=False)
            return None, event, poll_warnings

        # Step 3: Register event (enforcing duplicate accession prevention)
        if self.register.has_event(clean_sym, event.accession_number):
            existing = self.register.get_event(clean_sym, event.accession_number)
            msg = f"Duplicate accession prevented: Filing {event.accession_number} for {clean_sym} is already registered."
            logger.info(msg)
            return None, existing, [msg]

        self.register.register_event(event, fail_if_duplicate=True)

        # Step 4: Reconcile financial facts
        reconciled_profile, changes, exceptions, undisclosed = self.reconciler.reconcile_filing(
            symbol=clean_sym,
            cik=cik,
            accession_number=event.accession_number,
            reporting_period=event.reporting_period,
            baseline_profile=baseline_profile,
            facts_data=facts_data,
        )

        self.register.update_event_state(
            symbol=clean_sym,
            accession_number=event.accession_number,
            state=EventProcessingState.RECONCILED,
            reconciliation_summary={"changes_count": len(changes), "reporting_period": event.reporting_period},
        )

        # Step 5: Assess research decision consequences
        effects, val_comp, disposition, rationale, questions, stopping_rule = self.assessor.assess_consequences(
            symbol=clean_sym,
            baseline_dossier=baseline_dossier,
            reconciled_profile=reconciled_profile,
            changes=changes,
            accounting_exceptions=exceptions,
            undisclosed_info=undisclosed,
        )

        self.register.update_event_state(
            symbol=clean_sym,
            accession_number=event.accession_number,
            state=EventProcessingState.ASSESSED,
        )

        # Step 6: Generate Review Packet
        packet, json_path, md_path = self.packet_generator.generate_packet(
            event=event,
            baseline_dossier=baseline_dossier,
            reconciled_changes=changes,
            assessment_effects=effects,
            accounting_exceptions=exceptions,
            undisclosed_info=undisclosed,
            valuation_comparison=val_comp,
            proposed_disposition=disposition,
            disposition_rationale=rationale,
            questions=questions,
            stopping_rule=stopping_rule,
        )

        # Update event record with packet pointers
        self.register.update_event_state(
            symbol=clean_sym,
            accession_number=event.accession_number,
            state=EventProcessingState.REVIEW_PENDING,
            successor_draft_id=packet.draft_successor_id,
            review_packet_id=packet.packet_id,
        )

        return packet, event, poll_warnings

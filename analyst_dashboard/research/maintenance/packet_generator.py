"""
analyst_dashboard/research/maintenance/packet_generator.py

Review Packet Generator for the ARX Research-Maintenance Pilot.
Produces both structured JSON and GitHub-flavored Markdown review packets.
Adheres strictly to ARX-DECISION-RESEARCH-PROCESS-V1 and PILOT_MAINTENANCE_POLICY.md.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from .contracts import (
    MaintenanceEvent,
    ProposedDisposition,
    ReconciledFinancialChange,
    DecisionAssessmentEffect,
    ReviewPacket,
)


class PacketGenerator:
    """
    Renders structured review packets and human-readable Markdown summaries.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_packet(
        self,
        event: MaintenanceEvent,
        baseline_dossier: Dict[str, Any],
        reconciled_changes: List[ReconciledFinancialChange],
        assessment_effects: List[DecisionAssessmentEffect],
        accounting_exceptions: List[str],
        undisclosed_info: List[str],
        valuation_comparison: Dict[str, Any],
        proposed_disposition: ProposedDisposition,
        disposition_rationale: str,
        questions: List[str],
        stopping_rule: str,
    ) -> Tuple[ReviewPacket, Path, Path]:
        """
        Builds ReviewPacket object, saves JSON and Markdown, and returns (packet, json_path, md_path).
        """
        now_utc = datetime.utcnow().isoformat() + "Z"
        packet_id = f"REV-{event.symbol}-{event.reporting_period.replace('-', '')}-{event.accession_number.replace('-', '')}"
        successor_draft_id = f"OPP-{event.symbol}-{event.reporting_period.replace('-', '')}-DRAFT-V1"
        baseline_id = baseline_dossier.get("dossier_id") or f"OPP-{event.symbol}-20261016"
        baseline_cutoff = baseline_dossier.get("publication_metadata", {}).get("filing_availability_cutoff_utc") or "2026-10-16T20:00:00Z"

        filing_meta = {
            "form": event.form,
            "accession_number": event.accession_number,
            "filing_date": event.filing_date,
            "reporting_period": event.reporting_period,
            "source_url": event.source_url,
            "primary_document": event.primary_document,
            "retrieval_time_utc": event.retrieval_time_utc,
            "is_replay": event.is_replay,
        }

        packet = ReviewPacket(
            packet_id=packet_id,
            symbol=event.symbol,
            company_name=event.company_name,
            cik=event.cik,
            accepted_baseline_id=baseline_id,
            accepted_baseline_cutoff=baseline_cutoff,
            draft_successor_id=successor_draft_id,
            filing_metadata=filing_meta,
            material_financial_changes=reconciled_changes,
            assessment_effects=assessment_effects,
            accounting_exceptions=accounting_exceptions,
            undisclosed_information=undisclosed_info,
            valuation_comparison=valuation_comparison,
            proposed_disposition=proposed_disposition,
            disposition_rationale=disposition_rationale,
            questions_requiring_owner_judgment=questions,
            stopping_rule=stopping_rule,
            created_at_utc=now_utc,
            is_replay=event.is_replay,
        )

        # File paths
        out_dir = self.output_dir or Path("docs/research/maintenance/packets")
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{packet_id}.json"
        md_path = out_dir / f"{packet_id}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(packet.to_dict(), f, indent=2)

        md_content = self.render_markdown(packet)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return packet, json_path, md_path

    def render_markdown(self, packet: ReviewPacket) -> str:
        """Renders GitHub-flavored Markdown for the review packet."""
        meta = packet.filing_metadata
        val = packet.valuation_comparison

        replay_badge = " *(EXPLICIT REPLAY)*" if packet.is_replay else ""

        lines = [
            f"# ARX Terminal — Research Maintenance Review Packet: {packet.symbol}{replay_badge}",
            f"## Review Packet ID: `{packet.packet_id}`",
            "",
            f"**Governing Authority:** ARX Closure Model v2 / ARX-DECISION-RESEARCH-PROCESS-V1  ",
            f"**Company:** {packet.company_name} (`{packet.symbol}` | CIK `{packet.cik}`)  ",
            f"**Accepted Baseline:** `{packet.accepted_baseline_id}` (Cutoff: `{packet.accepted_baseline_cutoff}`)  ",
            f"**Draft Successor:** `{packet.draft_successor_id}`  ",
            f"**Trigger Filing:** Form {meta.get('form')} (Filed: `{meta.get('filing_date')}` | Period: `{meta.get('reporting_period')}`)  ",
            f"**SEC EDGAR Accession:** [`{meta.get('accession_number')}`]({meta.get('source_url')})  ",
            f"**Packet Generated:** `{packet.created_at_utc}`  ",
            f"**Proposed Disposition:** **`{packet.proposed_disposition.value}`**  ",
            "",
            "---",
            "",
            "### 1. Executive Summary & Review Decision",
            "",
            f"**Proposed Action:** `{packet.proposed_disposition.value}`  ",
            f"**Disposition Rationale:** {packet.disposition_rationale}  ",
            "",
            "> [!IMPORTANT]",
            "> **Publication Invariant:** This review packet creates a staged draft successor record. "
            "> The accepted production dossier remains intact and immutable until formal independent research review and acceptance.",
            "",
            "---",
            "",
            "### 2. Primary-Source Event Metadata",
            "",
            "| Field | Recorded Filing Value | Evidence Context |",
            "| :--- | :--- | :--- |",
            f"| **Issuer** | {packet.company_name} ({packet.symbol}) | CIK: {packet.cik} |",
            f"| **Filing Form** | Form {meta.get('form')} | Periodic unaudited financial report |",
            f"| **Filing Date** | {meta.get('filing_date')} | Primary-source publication on SEC EDGAR |",
            f"| **Reporting Period** | Ended {meta.get('reporting_period')} | Q3 fiscal quarter close |",
            f"| **Accession Number** | {meta.get('accession_number')} | Unique SEC EDGAR archive accession |",
            f"| **Primary Document** | [{meta.get('primary_document')}]({meta.get('source_url')}) | SEC EDGAR primary HTML filing |",
            f"| **Ingestion Time** | {meta.get('retrieval_time_utc')} | Retrieved by automated maintenance poller |",
            f"| **Processing State** | `REVIEW_PENDING` | Staged for research lead evaluation |",
            "",
            "---",
            "",
            "### 3. Reconciled Financial Changes",
            "",
            "| Metric | Accepted Baseline | Draft Successor | Absolute Delta | Delta % | Accounting Context & Source |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for c in packet.material_financial_changes:
            d_val = f"{c.delta_value:+.2f}" if c.delta_value is not None else "—"
            if c.units == "USD" and c.delta_value is not None:
                d_val = f"+${c.delta_value / 1e6:.1f}M" if c.delta_value >= 0 else f"-${abs(c.delta_value) / 1e6:.1f}M"
            d_pct = f"{c.delta_pct:+.2f}%" if c.delta_pct is not None else "—"
            lines.append(
                f"| **{c.label}** | {c.baseline_display} | {c.draft_display} | {d_val} | {d_pct} | {c.accounting_context} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "### 4. Decision Consequence Assessment",
            "",
            "| Decision Dimension | Baseline State | Draft State | Epistemological Class | Materiality |",
            "| :--- | :--- | :--- | :---: | :---: |",
        ])

        for e in packet.assessment_effects:
            lines.append(
                f"| **{e.conclusion_label}** | {e.baseline_state} | {e.draft_state} | `{e.interpretation_type.value}` | `{e.materiality}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "### 5. Conditional Valuation Differences",
            "",
            f"Valuation model machinery status: **`{val.get('valuation_machinery_status')}`**  ",
            f"Reference Market Price: **${val.get('reference_price', 0.0):.2f}** (Observed {val.get('reference_date', 'N/A')}; held constant).  ",
            "",
        ])

        if val.get("valuation_machinery_status") == "COMPUTED_VERIFIED":
            cagr = val.get("solved_5y_cagr", {})
            sc = val.get("scenarios", {})
            lines.extend([
                "#### A. Solved Reverse-DCF 5-Year Revenue CAGR",
                f"- **Baseline Implied CAGR:** {cagr.get('baseline_cagr_pct', 0.0):.2f}%",
                f"- **Draft Successor Implied CAGR:** {cagr.get('draft_cagr_pct', 0.0):.2f}%",
                f"- **Delta:** **{cagr.get('delta_cagr_bps', 0):+.0f} bps** ({cagr.get('interpretation')})",
                "",
                "#### B. Conditional Scenario Differences",
                "| Scenario | Baseline Per Share | Draft Per Share | Delta ($) | Delta (%) |",
                "| :--- | :---: | :---: | :---: | :---: |",
                f"| **Conservative** (12% CAGR, 12.5% Margin, 10.5% WACC) | ${sc.get('conservative', {}).get('baseline_per_share', 0.0):.2f} | ${sc.get('conservative', {}).get('draft_per_share', 0.0):.2f} | ${sc.get('conservative', {}).get('delta_dollars', 0.0):+.2f} | {sc.get('conservative', {}).get('delta_pct', 0.0):+.2f}% |",
                f"| **Base** (22% CAGR, 15.0% Margin, 9.5% WACC) | ${sc.get('base', {}).get('baseline_per_share', 0.0):.2f} | ${sc.get('base', {}).get('draft_per_share', 0.0):.2f} | ${sc.get('base', {}).get('delta_dollars', 0.0):+.2f} | {sc.get('base', {}).get('delta_pct', 0.0):+.2f}% |",
                f"| **Optimistic** (30% CAGR, 16.5% Margin, 8.5% WACC) | ${sc.get('optimistic', {}).get('baseline_per_share', 0.0):.2f} | ${sc.get('optimistic', {}).get('draft_per_share', 0.0):.2f} | ${sc.get('optimistic', {}).get('delta_dollars', 0.0):+.2f} | {sc.get('optimistic', {}).get('delta_pct', 0.0):+.2f}% |",
            ])
        else:
            lines.append(f"Valuation unresolved: {val.get('reason')}")

        lines.extend([
            "",
            "---",
            "",
            "### 6. Accounting Controls & Exceptions",
            "",
        ])
        if packet.accounting_exceptions:
            for exc in packet.accounting_exceptions:
                lines.append(f"- **Accounting Control:** {exc}")
        else:
            lines.append("- Zero accounting exceptions recorded.")

        lines.extend([
            "",
            "### 7. Undisclosed Information & Unresolved Inquiries",
            "",
        ])
        if packet.undisclosed_information:
            for und in packet.undisclosed_information:
                lines.append(f"- `[UNRESOLVED]` {und}")
        else:
            lines.append("- Zero undisclosed items.")

        lines.extend([
            "",
            "---",
            "",
            "### 8. Questions Requiring Research Lead Judgment",
            "",
        ])
        for idx, q in enumerate(packet.questions_requiring_owner_judgment, start=1):
            lines.append(f"{idx}. {q}")

        lines.extend([
            "",
            "---",
            "",
            "### 9. Bounded Stopping Rule",
            "",
            f"{packet.stopping_rule}",
            "",
        ])

        return "\n".join(lines)

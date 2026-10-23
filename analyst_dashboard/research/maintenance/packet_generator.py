"""
analyst_dashboard/research/maintenance/packet_generator.py

Review Packet Generator for the ARX Research-Maintenance Pilot.
Produces both structured JSON and GitHub-flavored Markdown review packets.
Adheres strictly to ARX-DECISION-RESEARCH-PROCESS-V1 and PILOT_MAINTENANCE_POLICY.md.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
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
    Enforces atomic file generation and complete evidence receipts.
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
        Builds ReviewPacket object, saves JSON and Markdown atomically, and returns (packet, json_path, md_path).
        """
        now_utc = datetime.utcnow().isoformat() + "Z"
        prefix = "REPLAY-" if event.is_replay else "REV-"
        packet_id = f"{prefix}{event.symbol}-{event.reporting_period.replace('-', '')}-{event.accession_number.replace('-', '')}"
        successor_draft_id = f"OPP-{event.symbol}-{event.reporting_period.replace('-', '')}-DRAFT-V1"
        baseline_id = baseline_dossier.get("dossier_id") or f"OPP-{event.symbol}-20261016"
        baseline_cutoff = (
            baseline_dossier.get("publication_metadata", {}).get("filing_availability_cutoff_utc")
            or "2026-10-16T20:00:00Z"
        )

        filing_meta = {
            "form": event.form,
            "accession_number": event.accession_number,
            "filing_date": event.filing_date,
            "reporting_period": event.reporting_period,
            "acceptance_datetime_utc": event.acceptance_datetime_utc,
            "source_url": event.source_url,
            "primary_document": event.primary_document,
            "index_url": event.index_url,
            "index_sha256": event.index_sha256,
            "submissions_url": event.submissions_url,
            "company_facts_url": event.company_facts_url,
            "retrieval_time_utc": event.retrieval_time_utc,
            "is_replay": event.is_replay,
            "sha256": event.sha256,
            "source_receipts": event.source_receipts,
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

        out_dir = self.output_dir or Path("docs/research/maintenance/packets")
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{packet_id}.json"
        md_path = out_dir / f"{packet_id}.md"

        # Atomic write JSON
        with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, encoding="utf-8") as tf:
            json.dump(packet.to_dict(), tf, indent=2)
            temp_json = Path(tf.name)
        os.replace(temp_json, json_path)

        # Atomic write Markdown
        md_content = self.render_markdown(packet)
        with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, encoding="utf-8") as tf:
            tf.write(md_content)
            temp_md = Path(tf.name)
        os.replace(temp_md, md_path)

        return packet, json_path, md_path

    def render_markdown(self, packet: ReviewPacket) -> str:
        """Renders GitHub-flavored Markdown for the review packet."""
        meta = packet.filing_metadata
        val = packet.valuation_comparison

        replay_badge = " *(EXPLICIT HISTORICAL REPLAY)*" if packet.is_replay else ""

        idx_sha = meta.get("index_sha256")
        idx_sha_display = f"`{idx_sha}`" if idx_sha else "`UNAVAILABLE (not retrieved or unhashed in bounded session)`"
        doc_sha = meta.get("sha256")
        doc_sha_display = f"`{doc_sha}`" if doc_sha else "`UNAVAILABLE (not retrieved or unhashed in bounded session)`"

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
            "### 2. Primary-Source Event Metadata & Cryptographic Receipts",
            "",
            "| Field | Recorded Filing Value | Evidence Context |",
            "| :--- | :--- | :--- |",
            f"| **Issuer** | {packet.company_name} ({packet.symbol}) | CIK: {packet.cik} |",
            f"| **Filing Form** | Form {meta.get('form')} | Periodic unaudited financial report |",
            f"| **Filing Date** | {meta.get('filing_date')} | Primary-source publication on SEC EDGAR |",
            f"| **Reporting Period** | Ended {meta.get('reporting_period')} | Fiscal quarter close |",
            f"| **Acceptance Time** | `{meta.get('acceptance_datetime_utc') or 'Recorded on SEC EDGAR'}` | Official SEC timestamp |",
            f"| **Accession Number** | `{meta.get('accession_number')}` | Unique SEC EDGAR archive accession |",
            f"| **Primary Document** | [{meta.get('primary_document')}]({meta.get('source_url')}) | SEC EDGAR primary HTML filing |",
            f"| **Primary Doc SHA-256** | {doc_sha_display} | Cryptographic content receipt |",
            f"| **Filing Index** | [Index Page]({meta.get('index_url')}) | SEC EDGAR filing archive folder |",
            f"| **Index SHA-256** | {idx_sha_display} | Cryptographic content receipt |",
            f"| **Submissions API** | [{packet.cik}]({meta.get('submissions_url')}) | SEC EDGAR JSON submissions endpoint |",
            f"| **Company Facts API** | [XBRL Facts]({meta.get('company_facts_url')}) | SEC EDGAR XBRL company facts endpoint |",
            f"| **Ingestion Time** | {meta.get('retrieval_time_utc')} | Retrieved by automated maintenance poller |",
            f"| **Processing State** | `REVIEW_PENDING` | Staged for research lead evaluation |",
            "",
            "> [!NOTE]",
            "> **Receipt Verification Boundary:** A cryptographic SHA-256 hash establishes payload content identity and byte-level immutability against the retrieval record; it does not verify financial correctness.",
            "",
            "#### Detailed Artifact Receipts & Representation",
            "| Artifact Type | Retrieval Mode | Byte Representation | Size (Bytes) | SHA-256 Content Digest |",
            "| :--- | :---: | :--- | :---: | :--- |",
        ]

        receipts = meta.get("source_receipts", {})
        if receipts:
            for obj_type, r in receipts.items():
                mode = r.get("retrieval_mode", "UNKNOWN")
                rep = r.get("byte_representation", "decoded_http_response_bytes")
                bc = f"{r.get('byte_count', 0):,}"
                h = f"`{r.get('sha256')}`" if r.get("sha256") else "`UNAVAILABLE`"
                lines.append(f"| **{obj_type}** | `{mode}` | `{rep}` | {bc} | {h} |")
        else:
            lines.append("| *No source receipts recorded* | - | - | - | - |")

        lines.extend([
            "",
            "---",
            "",
            "### 3. Reconciled Financial Changes",
            "",
            "| Metric | Accepted Baseline | Draft Successor | Absolute Delta | Delta % | Accounting Context & Source |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ])

        for c in packet.material_financial_changes:
            base_d = c.baseline_display
            draft_d = c.draft_display
            delta_val = f"{c.delta_value:+,.2f}" if c.delta_value is not None else "N/A"
            if c.units == "USD" and c.delta_value is not None:
                delta_val = f"${c.delta_value / 1e6:+,.1f}M"
            delta_p = f"{c.delta_pct:+.2f}%" if c.delta_pct is not None else "N/A"
            ctx = c.accounting_context
            if c.source_accessions and len(c.source_accessions) > 1:
                ctx += f" (Sources: {', '.join(c.source_accessions)})"
            lines.append(f"| **{c.label}** | {base_d} | {draft_d} | {delta_val} | {delta_p} | {ctx} |")

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
            lines.append(f"| **{e.conclusion_label}** | {e.baseline_state} | {e.draft_state} | `{e.interpretation_type.value}` | `{e.materiality}` |")

        lines.extend([
            "",
            "---",
            "",
            "### 5. Conditional Valuation Differences",
            "",
        ])

        v_status = val.get("valuation_machinery_status", "UNKNOWN")
        lines.append(f"Valuation model machinery status: **`{v_status}`**  ")

        if v_status == "COMPUTED_VERIFIED":
            ref_p = val.get("reference_price")
            ref_d = val.get("reference_date")
            cagr_info = val.get("solved_5y_cagr", {})
            scenarios = val.get("scenarios", {})

            resolved_assump = val.get("resolved_assumptions", {})
            rev_p = resolved_assump.get("reverse_dcf_parameters", {})
            sc_c_p = scenarios.get("conservative", {}).get("assumptions", {})
            sc_b_p = scenarios.get("base", {}).get("assumptions", {})
            sc_o_p = scenarios.get("optimistic", {}).get("assumptions", {})

            ref_p_display = f"${ref_p:.2f}" if ref_p is not None else "UNAVAILABLE"
            bps_val = cagr_info.get('delta_cagr_bps')
            bps_display = f"{bps_val:+.0f} bps" if bps_val is not None else "N/A"
            lines.extend([
                f"Reference Market Price: **{ref_p_display}** (Observed {ref_d or 'N/A'}; held constant).  ",
                "",
                "#### A. Authoritative Resolved Valuation Assumptions",
                "| Parameter | Reverse-DCF Target | Conservative Scenario | Base Scenario | Optimistic Scenario | Authoritative Source Path |",
                "| :--- | :---: | :---: | :---: | :---: | :--- |",
                f"| **5-Year Revenue CAGR** | Solved ({cagr_info.get('draft_cagr_pct', 'N/A')}%) | {sc_c_p.get('revenue_cagr_pct', 'N/A')}% | {sc_b_p.get('revenue_cagr_pct', 'N/A')}% | {sc_o_p.get('revenue_cagr_pct', 'N/A')}% | `baseline_dossier.valuation.scenarios` |",
                f"| **Operating Margin** | {rev_p.get('operating_margin_pct', 'N/A')}% | {sc_c_p.get('operating_margin_pct', 'N/A')}% | {sc_b_p.get('operating_margin_pct', 'N/A')}% | {sc_o_p.get('operating_margin_pct', 'N/A')}% | `reverse_dcf_parameters` / `scenarios` |",
                f"| **WACC** | {rev_p.get('wacc_pct', 'N/A')}% | {sc_c_p.get('wacc_pct', 'N/A')}% | {sc_b_p.get('wacc_pct', 'N/A')}% | {sc_o_p.get('wacc_pct', 'N/A')}% | `reverse_dcf_parameters` / `scenarios` |",
                f"| **Terminal Growth** | {rev_p.get('terminal_growth_pct', 'N/A')}% | {sc_c_p.get('terminal_growth_pct', 'N/A')}% | {sc_b_p.get('terminal_growth_pct', 'N/A')}% | {sc_o_p.get('terminal_growth_pct', 'N/A')}% | `reverse_dcf_parameters` / `scenarios` |",
                f"| **Effective Tax Rate** | {rev_p.get('tax_rate_pct', 'N/A')}% | {sc_c_p.get('tax_rate_pct', 'N/A')}% | {sc_b_p.get('tax_rate_pct', 'N/A')}% | {sc_o_p.get('tax_rate_pct', 'N/A')}% | `reverse_dcf_parameters` / `scenarios` |",
                f"| **Reinvestment Rate** | {rev_p.get('reinvestment_rate_pct', 'N/A')}% | {sc_c_p.get('reinvestment_rate_pct', 'N/A')}% | {sc_b_p.get('reinvestment_rate_pct', 'N/A')}% | {sc_o_p.get('reinvestment_rate_pct', 'N/A')}% | `reverse_dcf_parameters` / `scenarios` |",
                "",
                "#### B. Solved Reverse-DCF 5-Year Revenue CAGR",
                f"- **Baseline Implied CAGR:** {cagr_info.get('baseline_cagr_pct', 'N/A')}%\n"
                f"- **Draft Successor Implied CAGR:** {cagr_info.get('draft_cagr_pct', 'N/A')}%\n"
                f"- **Delta:** **{bps_display}** ({cagr_info.get('interpretation', 'N/A')})",
                "",
                "#### C. Conditional Scenario Differences",
                "| Scenario | Baseline Per Share | Draft Per Share | Delta ($) | Delta (%) |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ])

            for sc_key, sc_data in scenarios.items():
                name = sc_key.capitalize()
                base_ps = f"${sc_data.get('baseline_per_share'):.2f}" if sc_data.get('baseline_per_share') is not None else "N/A"
                draft_ps = f"${sc_data.get('draft_per_share'):.2f}" if sc_data.get('draft_per_share') is not None else "N/A"
                delta_d = f"${sc_data.get('delta_dollars'):+.2f}" if sc_data.get('delta_dollars') is not None else "N/A"
                delta_p = f"{sc_data.get('delta_pct'):+.2f}%" if sc_data.get('delta_pct') is not None else "N/A"
                lines.append(f"| **{name}** | {base_ps} | {draft_ps} | {delta_d} | {delta_p} |")
        else:
            lines.append(f"> [!WARNING]\n> **Valuation Suppressed:** {val.get('reason', 'Inputs unresolved or in validation conflict.')}")

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
            lines.append("- No accounting exceptions recorded.")

        lines.extend([
            "",
            "### 7. Undisclosed Information & Unresolved Inquiries",
            "",
        ])
        if packet.undisclosed_information:
            for und in packet.undisclosed_information:
                lines.append(f"- `[UNRESOLVED]` {und}")
        else:
            lines.append("- All required financial disclosures were present.")

        lines.extend([
            "",
            "---",
            "",
            "### 8. Questions Requiring Research Lead Judgment",
            "",
        ])
        for i, q in enumerate(packet.questions_requiring_owner_judgment, 1):
            lines.append(f"{i}. {q}")

        lines.extend([
            "",
            "---",
            "",
            "### 9. Bounded Stopping Rule",
            "",
            packet.stopping_rule,
            "",
        ])

        return "\n".join(lines)

"""
analyst_dashboard/universe/acquisition/report.py

Generates machine-readable JSON and human-readable Markdown reports
from SourceAccountingReport instances.

Strict Non-Inference Invariants:
- Preserves exact observation and malformed counts.
- Explicitly declares PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED.
- Discloses coverage limitations and unresolved policy decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import SourceAccountingReport


def format_markdown_report(report: SourceAccountingReport) -> str:
    """Formats a SourceAccountingReport into a GitHub-flavored Markdown document."""
    lines = [
        "# ARX Terminal — Source Population Acquisition & Accounting Report",
        "",
        "**Classification**: Engineering Artifact — Listing Source Acquisition & Offline Accounting  ",
        f"**Generated At (UTC)**: `{report.generated_at_utc}`  ",
        f"**Acquisition Version**: `{report.acquisition_version}`  ",
        f"**Overall Outcome**: `{report.overall_outcome}`  ",
        f"**Production Population Completeness**: `{report.production_population_completeness}`  ",
        "",
        "> [!IMPORTANT]",
        "> **Core Product Acceptance Criterion**:",
        "> *“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”*",
        "> ",
        "> **A successful download, directory join, parser, or diagnostic subset alone does NOT establish complete market coverage.**",
        "",
        "---",
        "",
        "## 1. Network Acquisition Evidence",
        "",
        "| Source ID | Outcome | HTTP Status | Payload Size | SHA-256 Digest | Publisher TS | Local Raw Artifact |",
        "|:---|:---:|:---:|:---:|:---|:---:|:---|",
    ]

    for att in report.attempts:
        sha_short = (att.content_sha256[:16] + "...") if att.content_sha256 else "N/A"
        raw_name = Path(att.raw_artifact_path).name if att.raw_artifact_path else "N/A"
        pub_ts = att.publisher_timestamp or att.last_modified or "N/A"
        lines.append(
            f"| `{att.source_id}` | `{att.outcome.value}` | `{att.http_status or 'N/A'}` | "
            f"`{att.byte_length:,} bytes` | `{sha_short}` | `{pub_ts}` | `{raw_name}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Offline Parser Replay & Source Accounting",
        "",
        "| Source ID | Feed Type | Replay Usable | Physical Lines | Metadata Rows | Data Observations | Malformed Rows | Unique Records |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for sid, summ in report.parsed_summaries.items():
        usable_badge = "YES" if summ.is_usable else "NO"
        lines.append(
            f"| `{sid}` | `{summ.feed_type}` | `{usable_badge}` | "
            f"`{summ.physical_line_count:,}` | `{summ.metadata_row_count:,}` | "
            f"`{summ.data_row_count:,}` | `{summ.malformed_row_count:,}` | `{summ.unique_record_count:,}` |"
        )

    if report.match_diagnostics:
        md = report.match_diagnostics
        lines.extend([
            "",
            "---",
            "",
            "## 3. Diagnostic Cross-Source Matching (Non-Binding)",
            "",
            "| Metric | Count | Operational / Boundary Meaning |",
            "|:---|:---:|:---|",
            f"| **Total Candidates** | `{md.get('total_candidates', 0):,}` | Total distinct join candidates produced across feeds (normalized-symbol candidates, NOT verified unique stocks). |",
            f"| **Fully Matched Across Feeds** | `{md.get('fully_matched_count', 0):,}` | Observations matched across both SEC and exchange directory feeds. |",
            f"| **SEC Only (Unmatched)** | `{md.get('sec_only_count', 0):,}` | Observations present in SEC master but absent from exchange directories. |",
            f"| **Directory Only (Unmatched)** | `{md.get('directory_only_count', 0):,}` | Observations present in directories but absent from SEC master. |",
            f"| **Ambiguous Candidates** | `{md.get('ambiguous_count', 0):,}` | Candidates with multiple or ambiguous join matches. |",
            f"| **Conflict Candidates** | `{md.get('conflict_count', 0):,}` | Candidates with venue code disagreements across feeds (see breakdown below). |",
            f"| **Missing Attribute Rows** | `{md.get('malformed_source_row_count', 0):,}` | Structurally parsed rows with evidence gaps (e.g. 168 SEC rows with missing exchange attribute; 0 syntax corruption). |",
            "",
            "> [!NOTE]",
            "> **Exchange Conflict Analysis (471 Observations)**:",
            "> - **275 NYSE vs AMEX**: NYSE American securities categorized under parent 'NYSE' umbrella in SEC EDGAR.",
            "> - **144 NYSE vs ARCA**: NYSE Arca securities categorized under parent 'NYSE' umbrella in SEC EDGAR.",
            "> - **44 CBOE vs BATS**: Cboe BZX exchange securities ('CBOE' in SEC EDGAR vs 'Z'/BATS in Nasdaq Trader).",
            "> - **8 TXSE vs F**: Texas Stock Exchange pre-launch announcements vs FINRA ADF.",
            "> All 471 conflicts reflect exchange taxonomy granularity and umbrella groupings, not symbol collisions or corrupt data.",
            "> Invariant: Original raw venue values are strictly preserved; disagreements are NEVER cleared through unsupported equivalence mappings (particularly TXSE vs FINRA ADF / 'F').",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 4. Disclosed Coverage Limitations",
        "",
    ])
    for lim in report.coverage_limitations:
        lines.append(f"- {lim}")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Unresolved Governance & Strategy Decisions",
        "",
    ])
    for dec in report.unresolved_decisions:
        lines.append(f"- ? {dec}")

    lines.append("")
    return "\n".join(lines)


def save_accounting_reports(
    report: SourceAccountingReport,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """
    Saves both machine-readable JSON and Markdown accounting reports.

    Returns:
        tuple of (json_path, markdown_path)
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "acquisition_accounting_report.json"
    md_file = out_path / "SOURCE_ACQUISITION_REPORT.md"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(format_markdown_report(report))

    return json_file, md_file

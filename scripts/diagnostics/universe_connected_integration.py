#!/usr/bin/env python3
"""
scripts/diagnostics/universe_connected_integration.py

Sprint 2 Package 3 (S2-04): Connected Offline Integration CLI.
Explicitly executes the offline pipeline connecting:
Archived Acquisition Artifacts
→ Offline Parsers
→ SecurityEvidenceResolver
→ UniverseInputValidator
→ Guarded SourceSecurity Conversion
→ DeterministicUniverseBuilder

Usage:
  # Synthetic Fixture Run (Demonstrates successful integration & ADR ineligibility):
  python scripts/diagnostics/universe_connected_integration.py \
    --fixture \
    --sec-path tests/fixtures/connected_integration/synthetic_sec_company_tickers_exchange.json \
    --nasdaq-listed-path tests/fixtures/connected_integration/synthetic_nasdaq_listed_directory.txt \
    --other-listed-path tests/fixtures/connected_integration/synthetic_other_listed_directory.txt \
    --openfigi-response-path tests/fixtures/connected_integration/synthetic_openfigi_response.json \
    --openfigi-request-path tests/fixtures/connected_integration/synthetic_openfigi_requests.json \
    --identity-mapping-path tests/fixtures/connected_integration/synthetic_identity_mapping.json \
    --evidence-overlay-path tests/fixtures/connected_integration/synthetic_evidence_overlay.json \
    --output-dir output/diagnostics/fixture_run

  # Real Archived Artifact Run (Truthfully reports admission blockers):
  python scripts/diagnostics/universe_connected_integration.py \
    --archive-dir C:/Users/akara/Documents/Projects/finance-sprint-2-listing-acquisition/data/raw_feeds/2026-10-16 \
    --output-dir output/diagnostics/archived_run
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analyst_dashboard.universe.connected_integration import (
    CANONICAL_2026_10_16_HASHES,
    ConnectedIntegrationResult,
    IntegrationConfig,
    format_markdown_report,
    run_connected_integration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARX Terminal Sprint 2 Package 3: Connected Offline Universe Integration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Directory containing standard archived feeds (sec_company_tickers_exchange.json, nasdaq_listed_directory.txt, other_listed_directory.txt)",
    )
    parser.add_argument("--sec-path", type=str, default=None, help="Explicit path to SEC company tickers exchange feed")
    parser.add_argument("--nasdaq-listed-path", type=str, default=None, help="Explicit path to Nasdaq Listed directory")
    parser.add_argument("--other-listed-path", type=str, default=None, help="Explicit path to Other Listed directory")
    parser.add_argument("--openfigi-response-path", type=str, default=None, help="Optional saved OpenFIGI response JSON")
    parser.add_argument("--openfigi-request-path", type=str, default=None, help="Optional saved OpenFIGI request JSON")
    parser.add_argument("--identity-mapping-path", type=str, default=None, help="Optional caller security_id mapping JSON")
    parser.add_argument("--evidence-overlay-path", type=str, default=None, help="Optional caller field evidence overlay JSON")
    parser.add_argument("--as-of", type=str, default="2026-10-16", help="Target session as-of date (YYYY-MM-DD)")
    parser.add_argument("--evaluation-cutoff", type=str, default=None, help="Optional ISO 8601 evaluation cutoff timestamp")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write diagnostic JSON and Markdown artifacts")
    parser.add_argument("--fixture", action="store_true", default=False, help="Explicitly declare run as synthetic fixture evaluation")
    parser.add_argument(
        "--diagnostic-inspect-incomplete-sources",
        action="store_true",
        default=False,
        help="Diagnostic inspection only: permits downstream diagnostic accounting to inspect records despite parse errors. Does NOT admit malformed rows, treat missing files as present, or enable builder evaluation on invalid batches.",
    )
    parser.add_argument(
        "--allow-incomplete-source-files",
        action="store_true",
        default=False,
        help="[DEPRECATED] Alias for --diagnostic-inspect-incomplete-sources. Emits deprecation warning.",
    )
    parser.add_argument("--require-venue-match", action="store_true", default=False, help="Require exact venue concordance across sources")

    args = parser.parse_args()

    # Handle deprecation notice for --allow-incomplete-source-files
    if args.allow_incomplete_source_files:
        print(
            "DEPRECATION WARNING: --allow-incomplete-source-files is deprecated and will be removed in a future release. "
            "Use --diagnostic-inspect-incomplete-sources for inspection-only diagnostics.",
            file=sys.stderr,
        )

    # Resolve paths from --fixture or --archive-dir if provided
    sec_p = args.sec_path
    nasdaq_p = args.nasdaq_listed_path
    other_p = args.other_listed_path

    if args.fixture:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fixtures_dir = os.path.join(repo_root, "tests", "fixtures", "connected_integration")
        if not sec_p:
            sec_p = os.path.join(fixtures_dir, "synthetic_sec_company_tickers_exchange.json")
        if not nasdaq_p:
            nasdaq_p = os.path.join(fixtures_dir, "synthetic_nasdaq_listed_directory.txt")
        if not other_p:
            other_p = os.path.join(fixtures_dir, "synthetic_other_listed_directory.txt")
        if not args.openfigi_response_path:
            args.openfigi_response_path = os.path.join(fixtures_dir, "synthetic_openfigi_response.json")
        if not args.openfigi_request_path:
            args.openfigi_request_path = os.path.join(fixtures_dir, "synthetic_openfigi_requests.json")
        if not args.identity_mapping_path:
            args.identity_mapping_path = os.path.join(fixtures_dir, "synthetic_identity_mapping.json")
        if not args.evidence_overlay_path:
            args.evidence_overlay_path = os.path.join(fixtures_dir, "synthetic_evidence_overlay.json")

    if args.archive_dir:
        if not sec_p:
            sec_p = os.path.join(args.archive_dir, "sec_company_tickers_exchange.json")
        if not nasdaq_p:
            nasdaq_p = os.path.join(args.archive_dir, "nasdaq_listed_directory.txt")
        if not other_p:
            other_p = os.path.join(args.archive_dir, "other_listed_directory.txt")

    if not sec_p and not nasdaq_p and not other_p:
        print("ERROR: No input sources specified. Provide --archive-dir, --fixture, or explicit source paths.", file=sys.stderr)
        return 1

    # For archived real sources, use canonical hashes by default
    expected_hashes = None
    if not args.fixture:
        expected_hashes = CANONICAL_2026_10_16_HASHES

    allow_incomplete = args.diagnostic_inspect_incomplete_sources or args.allow_incomplete_source_files

    config = IntegrationConfig(
        sec_path=sec_p,
        nasdaq_listed_path=nasdaq_p,
        other_listed_path=other_p,
        openfigi_response_path=args.openfigi_response_path,
        openfigi_request_path=args.openfigi_request_path,
        identity_mapping_path=args.identity_mapping_path,
        evidence_overlay_path=args.evidence_overlay_path,
        expected_hashes=expected_hashes,
        as_of=args.as_of,
        evaluation_cutoff=args.evaluation_cutoff,
        output_dir=args.output_dir,
        require_venue_match=args.require_venue_match,
        allow_incomplete_source_files=allow_incomplete,
        diagnostic_inspect_incomplete_sources=allow_incomplete,
        is_synthetic_fixture=args.fixture,
    )

    result = run_connected_integration(config)

    # Print summary to stdout
    print(format_markdown_report(result))
    print(f"\n[OK] Diagnostic artifacts written to: {args.output_dir}")
    print(f"  - Summary Markdown: {result.summary_markdown_file}")
    print(f"  - Summary JSON:     {result.summary_json_file}")
    print(f"  - Diagnostics JSON: {result.record_diagnostics_file}")

    # Explicit Exit Code Policy:
    # 0 = EXIT_SUCCESS_ADMITTED: Pipeline executed and batch admitted/evaluated (e.g. synthetic fixture run)
    # 2 = EXIT_BLOCKED_FAIL_CLOSED: Diagnostic run completed truthfully; admission blocked fail-closed
    # 1 = EXIT_EXECUTION_ERROR: Execution error or unhandled failure
    if result.overall_outcome == "SUCCESS_BUILDER_EVALUATED":
        return 0
    elif result.overall_outcome in (
        "BLOCKED_VALIDATION_FAILED",
        "ADMITTED_BUILDER_BLOCKED",
        "FAILED_SOURCE_INTEGRITY",
    ):
        return 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

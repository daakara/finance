"""
scripts/package_corrected_review.py

Packages the complete, verified review bundle into:
sourced_fundamentals_integration_corrected_review_package.zip

Enforces strict fail-closed packaging: fails with exit code 1 if ANY required file
is missing or inaccessible. Computes and prints SHA-256 hashes for all contents,
the cumulative patch, and the final archive.
"""

import hashlib
import zipfile
import sys
from pathlib import Path

REQUIRED_FILES = [
    # Cumulative Patch
    "sourced_fundamentals_integration_pilot.patch",

    # Governance & Acceptance Documents
    "docs/research/ACCEPTANCE_RESULTS.md",
    "docs/research/SOURCED_FUNDAMENTALS_INTEGRATION_REVIEW.md",
    "docs/research/UNDERVALUED_GROWTH_RESEARCH_UPDATE.md",
    "docs/research/screenshots/BROWSER_VERIFICATION_REPORT.md",

    # Unobscured Element Screenshots
    "docs/research/screenshots/unobscured_card_fix_desktop.png",
    "docs/research/screenshots/unobscured_card_fix_mobile.png",
    "docs/research/screenshots/unobscured_card_duol_desktop.png",
    "docs/research/screenshots/unobscured_card_duol_mobile.png",
    "docs/research/screenshots/unobscured_card_ulta_desktop.png",
    "docs/research/screenshots/unobscured_card_ulta_mobile.png",
    "docs/research/screenshots/unobscured_card_aapl_desktop.png",
    "docs/research/screenshots/unobscured_card_aapl_mobile.png",

    # Unobscured Viewport Screenshots
    "docs/research/screenshots/unobscured_viewport_fix_desktop.png",
    "docs/research/screenshots/unobscured_viewport_fix_mobile.png",
    "docs/research/screenshots/unobscured_viewport_duol_desktop.png",
    "docs/research/screenshots/unobscured_viewport_duol_mobile.png",
    "docs/research/screenshots/unobscured_viewport_ulta_desktop.png",
    "docs/research/screenshots/unobscured_viewport_ulta_mobile.png",
    "docs/research/screenshots/unobscured_viewport_aapl_desktop.png",
    "docs/research/screenshots/unobscured_viewport_aapl_mobile.png",

    # Backend Service & Data Layer
    "analyst_dashboard/research/sourced_fundamentals/service.py",
    "analyst_dashboard/research/sourced_fundamentals/sourced_fundamentals_ledger.json",
    "api/routes/fundamentals.py",

    # Frontend UI Components & Static Snapshot
    "frontend/components/SourcedFundamentalsCard.tsx",
    "frontend/lib/sourcedFundamentals.ts",
    "frontend/lib/sourcedFundamentalsSnapshot.json",

    # Verification Scripts & Tests
    "scripts/capture_unobscured_screenshots.py",
    "scripts/export_frontend_fundamentals.py",
    "scripts/generate_research_tables.py",
    "tests/test_fundamentals_api.py",
]

ARCHIVE_NAME = "sourced_fundamentals_integration_corrected_review_package.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Validating existence of {len(REQUIRED_FILES)} required review deliverables...")
    missing = []
    file_manifest = []

    for rel_str in REQUIRED_FILES:
        p = Path(rel_str)
        if not p.exists():
            missing.append(rel_str)
            print(f"  [MISSING] {rel_str}")
        else:
            file_hash = sha256_file(p)
            size = p.stat().st_size
            file_manifest.append((rel_str, size, file_hash))
            print(f"  [EXISTS] {rel_str} ({size} bytes, sha256: {file_hash[:12]}...)")

    if missing:
        print(f"\nFATAL ERROR: {len(missing)} required files are missing. Halting packaging.")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"\nAll {len(REQUIRED_FILES)} required files exist and verified.")
    print(f"Creating review archive '{ARCHIVE_NAME}'...")

    archive_path = Path(ARCHIVE_NAME)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel_str, size, file_hash in file_manifest:
            zf.write(rel_str, arcname=rel_str)

    archive_size = archive_path.stat().st_size
    archive_hash = sha256_file(archive_path)

    print(f"\n[SUCCESS] Successfully created '{ARCHIVE_NAME}'")
    print(f"Archive Size: {archive_size} bytes")
    print(f"Archive SHA-256: {archive_hash}")

    patch_path = Path("sourced_fundamentals_integration_pilot.patch")
    print(f"\nPatch File: {patch_path.name}")
    print(f"Patch Size: {patch_path.stat().st_size} bytes")
    print(f"Patch SHA-256: {sha256_file(patch_path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

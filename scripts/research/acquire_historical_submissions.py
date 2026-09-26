"""ARX Terminal — Acquire Historical SEC Submission Files for the 120 CIKs (Section 5 & 31).

Discovers all submission files referenced under filings.files in CIK*.json,
downloads them from https://data.sec.gov/submissions/{file_name} with rate-limiting,
and saves them to data/research/cache/sec_submissions/{file_name}.
"""

import sys
import json
import time
from pathlib import Path
import requests

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SUBMISSIONS_DIR = Path("data/research/cache/sec_submissions")
HEADERS = {"User-Agent": "ArxTerminal/1.0 (research@arxterminal.org)"}


def main():
    print("=" * 80)
    print("ARX TERMINAL — ACQUIRE HISTORICAL SUBMISSION FILES (SECTION 5 & 31)")
    print("=" * 80)

    # 1. Enumerate all required historical files
    main_ciks = [p for p in SUBMISSIONS_DIR.glob("CIK*.json") if "-" not in p.name]
    print(f"Loaded {len(main_ciks)} main CIK submission files.")

    files_to_download = []
    ciks_with_files = set()

    for p in main_ciks:
        try:
            sub = json.load(open(p, encoding="utf-8"))
            h_files = sub.get("filings", {}).get("files", [])
            if h_files:
                ciks_with_files.add(p.stem)
                for f in h_files:
                    fname = f.get("name")
                    if fname:
                        local_path = SUBMISSIONS_DIR / fname
                        if not local_path.exists():
                            files_to_download.append((p.stem, fname, local_path))
        except Exception as e:
            print(f"Error reading {p}: {e}")

    print(f"CIKs with historical files: {len(ciks_with_files)}")
    print(f"Total historical files to acquire: {len(files_to_download)}")

    success_count = 0
    failure_count = 0

    for idx, (cik, fname, local_path) in enumerate(files_to_download):
        url = f"https://data.sec.gov/submissions/{fname}"
        time.sleep(0.12)
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                # verify it's valid json
                json_data = r.json()
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f)
                success_count += 1
                if (idx + 1) % 20 == 0 or idx == len(files_to_download) - 1:
                    print(f"[{idx+1}/{len(files_to_download)}] Acquired {fname} ({len(r.content):,} bytes)")
            else:
                failure_count += 1
                print(f"[{idx+1}/{len(files_to_download)}] HTTP_{r.status_code} for {fname}")
        except Exception as e:
            failure_count += 1
            print(f"[{idx+1}/{len(files_to_download)}] Error for {fname}: {e}")

    print("\nACQUISITION RESULTS:")
    print(f"  CIKS_WITH_HISTORICAL_FILES = {len(ciks_with_files)}")
    print(f"  ACQUIRED_HISTORICAL_FILES  = {success_count}")
    print(f"  FAILED_HISTORICAL_FILES    = {failure_count}")


if __name__ == "__main__":
    main()

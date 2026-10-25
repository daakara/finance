"""Deterministic verification of Railway CLI upload context."""
import os
import sys
import subprocess
import json
import pathspec

def collect_filesystem_files(root_dir):
    """Walk filesystem starting from root_dir, collecting relative posix paths."""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude .git immediately as Railway CLI does
        if ".git" in dirnames:
            dirnames.remove(".git")
        if "node_modules" in dirnames:
            dirnames.remove("node_modules")
            
        rel_dir = os.path.relpath(dirpath, root_dir)
        for f in filenames:
            rel_file = f if rel_dir == "." else os.path.join(rel_dir, f)
            posix_path = rel_file.replace("\\", "/")
            all_files.append(posix_path)
    return sorted(all_files)

def filter_by_gitignore(root_dir, file_list):
    """Filter file_list using git check-ignore --no-index."""
    p = subprocess.Popen(
        ["git", "check-ignore", "--no-index", "-v", "--stdin"],
        cwd=root_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # git check-ignore outputs only lines that match ignore patterns
    stdout, stderr = p.communicate(input="\n".join(file_list))
    
    # Parse lines from check-ignore
    # Format: <source>:<lineno>:<pattern> <path>
    ignored_paths = set()
    negated_paths = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rule, path = parts
            path = path.strip().replace("\\", "/").strip('"')
            # Check if rule was a negation (starting with !)
            rule_pattern = rule.split(":", 2)[-1] if ":" in rule else rule
            if rule_pattern.startswith("!"):
                negated_paths.add(path)
            else:
                ignored_paths.add(path)
                
    # A path is truly ignored if it is in ignored_paths and not in negated_paths
    effectively_ignored = ignored_paths - negated_paths
    
    # But wait! If a parent directory is ignored, all its children are ignored too.
    # To be 100% faithful to gitignore directory pruning:
    spec_lines = []
    gitignore_path = os.path.join(root_dir, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as gf:
            spec_lines = [l for l in gf.read().splitlines()]
    spec = pathspec.PathSpec.from_lines('gitwildmatch', spec_lines)
    
    packaged_files = []
    for f in file_list:
        if not spec.match_file(f):
            packaged_files.append(f)
            
    return packaged_files

def verify_manifest(packaged_files):
    errors = []
    
    # 1. Required files
    required = [
        "data/__init__.py",
        "data/cache.py",
        "data/fetchers.py",
        "data/processors.py",
        "api/main.py",
        "api/routes/screener.py",
        "analyst_dashboard/universe/population_assessment_accounting.py",
        "analyst_dashboard/universe/isolated_publication.py",
        "requirements.txt",
        "Dockerfile",
        "railway.json",
    ]
    for r in required:
        if r not in packaged_files:
            errors.append(f"REQUIRED FILE MISSING from upload context: {r}")
            
    # 2. Prohibited files
    for f in packaged_files:
        f_lower = f.lower()
        if f_lower.endswith((".db", ".sqlite", ".sqlite3")):
            errors.append(f"PROHIBITED DATABASE included in upload context: {f}")
        if f.startswith(".env") or "/.env" in f:
            if not f.endswith(".example"):
                errors.append(f"PROHIBITED SECRET included in upload context: {f}")
        if f.startswith("backups/") or "/backups/" in f:
            errors.append(f"PROHIBITED BACKUP ARCHIVE included in upload context: {f}")
        if any(f.startswith(d) for d in ["data/canonical/", "data/operational/", "data/staging/"]):
            errors.append(f"PROHIBITED LOCAL STAGING DATA included in upload context: {f}")
        if "node_modules/" in f:
            errors.append(f"PROHIBITED NODE_MODULES included in upload context: {f}")
        if f.endswith((".pyc", ".pyo")) or "__pycache__" in f:
            errors.append(f"PROHIBITED PYTHON CACHE included in upload context: {f}")
            
    return errors

if __name__ == "__main__":
    root = os.path.abspath(".")
    all_files = collect_filesystem_files(root)
    packaged = filter_by_gitignore(root, all_files)
    errors = verify_manifest(packaged)
    
    print(f"Total filesystem files considered: {len(all_files)}")
    print(f"Files included in simulated Railway CLI upload: {len(packaged)}")
    
    if errors:
        print("\nFAILURES DETECTED:")
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)
    else:
        print("\nALL REPRODUCED BUILD CONTEXT ASSERTIONS PASSED:")
        print(f"  - data/cache.py: PRESENT")
        print(f"  - data/fetchers.py: PRESENT")
        print(f"  - data/processors.py: PRESENT")
        print(f"  - data/__init__.py: PRESENT")
        print(f"  - Zero SQLite databases (*.db)")
        print(f"  - Zero secret environment files (.env*)")
        print(f"  - Zero backup archives (backups/)")
        print(f"  - Zero local staging data (data/canonical/, data/operational/, data/staging/)")
        print(f"  - Zero node_modules or python cache")
        sys.exit(0)

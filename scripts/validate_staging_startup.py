import os
import sys
sys.path.insert(0, os.path.abspath("."))

import shutil
import tempfile
import subprocess
import time
import urllib.request
import json

from scripts.verify_upload_context import collect_filesystem_files, filter_by_gitignore

print("=== 1. Building Isolated Deployment Context in Temporary Directory ===")
root = os.path.abspath(".")
all_files = collect_filesystem_files(root)
packaged_files = filter_by_gitignore(root, all_files)

with tempfile.TemporaryDirectory() as staging_dir:
    print(f"Staging directory: {staging_dir}")
    print(f"Copying {len(packaged_files)} packaged files to simulate container filesystem (/app)...")
    for f in packaged_files:
        src = os.path.join(root, f.replace("/", os.sep))
        dst = os.path.join(staging_dir, f.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        
    print("Files copied successfully.")
    
    # Verify data/cache.py exists in staging dir
    cache_py = os.path.join(staging_dir, "data", "cache.py")
    if not os.path.exists(cache_py):
        print(f"FATAL: {cache_py} does not exist in staging context!")
        sys.exit(1)
    else:
        print(f"VERIFIED: {cache_py} is present in staging context.")
        
    print("\n=== 2. Testing Python Import Hierarchy in Staging Context ===")
    test_import_code = """
import sys
print("Python version:", sys.version)
from api.main import app
print("FastAPI app loaded successfully:", app.title)
from analysis.etf import ETFAnalyzer
print("ETFAnalyzer imported successfully from analysis.etf")
from data.cache import cache_result
print("cache_result imported successfully from data.cache")
from data.fetchers import safe_yfinance_fetch
print("safe_yfinance_fetch imported successfully from data.fetchers")
from data.processors import DataProcessor
print("DataProcessor imported successfully from data.processors")
"""
    p_imp = subprocess.run([sys.executable, "-c", test_import_code], cwd=staging_dir, capture_output=True, text=True)
    print("STDOUT:\n" + p_imp.stdout)
    if p_imp.returncode != 0:
        print("STDERR:\n" + p_imp.stderr)
        print("FATAL: Import failed in staging context!")
        sys.exit(1)
        
    print("\n=== 3. Launching Production Startup Command in Staging Context ===")
    port = 8999
    env = os.environ.copy()
    env["ENVIRONMENT"] = "development"
    env["PORT"] = str(port)
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=staging_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"Waiting for Uvicorn server on port {port}...")
    server_ready = False
    for attempt in range(15):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    server_ready = True
                    body = r.read().decode('utf-8')
                    print(f"Server is READY! /health returned: {body}")
                    break
        except Exception:
            pass
            
    if not server_ready:
        print("FATAL: Server failed to start within 15 seconds!")
        server_proc.terminate()
        stdout, stderr = server_proc.communicate()
        print("SERVER STDOUT:\n", stdout)
        print("SERVER STDERR:\n", stderr)
        sys.exit(1)
        
    print("\n=== 4. Testing Endpoints on Isolated Staging Server ===")
    endpoints = [
        "/health",
        "/api/v1/governance/epoch-2/status",
        "/api/v1/screener/vcp/coordination",
    ]
    for ep in endpoints:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{ep}", timeout=5) as r:
                content = r.read().decode('utf-8')
                print(f"  {ep:35} -> HTTP {r.status} (Length: {len(content)})")
        except Exception as e:
            print(f"  {ep:35} -> ERROR: {e}")
            
    server_proc.terminate()
    server_proc.wait(timeout=5)
    print("\nStaging server terminated cleanly. Smoke validation COMPLETE.")

import sys
import os
import subprocess
import base64
import socket

# Force IPv4
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.check_pr167_ci import get_token

def main():
    token = get_token()
    b64_creds = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("utf-8")
    auth_header = f"Authorization: Basic {b64_creds}"

    branch = "feature/radar-p2-004-hydration-readiness"
    cmd = [
        "git",
        "-c", "credential.helper=",
        "-c", f"http.extraHeader={auth_header}",
        "push",
        "-u",
        "origin",
        f"{branch}:refs/heads/{branch}"
    ]
    print(f"Pushing {branch} to origin...")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    print("STDOUT:", proc.stdout)
    print("STDERR:", proc.stderr)
    return proc.returncode

if __name__ == "__main__":
    sys.exit(main())

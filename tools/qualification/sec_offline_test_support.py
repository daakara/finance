"""Pytest plugin: loopback-only egress and real disposable PostgreSQL over TCP.

Use with PYTHONPATH=tools/qualification pytest -p sec_offline_test_support.
This changes test infrastructure only; application persistence is not mocked.
"""
import socket
import subprocess
from pathlib import Path

import pgserver
from pgserver._commands import POSTGRES_BIN_PATH

_servers = []
_originals = {}


class LoopbackPostgres:
    def __init__(self, pgdata):
        self.pgdata = Path(pgdata)
        self.pgdata.mkdir(parents=True, exist_ok=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        if not (self.pgdata / "PG_VERSION").exists():
            subprocess.run([str(POSTGRES_BIN_PATH / "initdb"), "-D", str(self.pgdata),
                            "-U", "postgres", "--auth=trust", "--no-locale"],
                           check=True, capture_output=True, text=True, timeout=30)
        self.started = False
        result = subprocess.run([
            str(POSTGRES_BIN_PATH / "pg_ctl"), "-D", str(self.pgdata),
            "-l", str(self.pgdata / "offline-postgres.log"), "-w", "start", "-o",
            f"-h 127.0.0.1 -p {self.port} -k '' -c fsync=on",
        ], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr + (self.pgdata / "offline-postgres.log").read_text())
        self.started = True
        _servers.append(self)

    def get_uri(self, database=None):
        return f"postgresql://postgres@127.0.0.1:{self.port}/{database or 'postgres'}"

    def cleanup(self):
        if self.started:
            subprocess.run([str(POSTGRES_BIN_PATH / "pg_ctl"), "-D", str(self.pgdata),
                            "-m", "immediate", "-w", "stop"],
                           check=True, capture_output=True, text=True, timeout=30)
            self.started = False


def pytest_configure(config):
    _originals.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex,
                      getaddrinfo=socket.getaddrinfo, get_server=pgserver.get_server)

    def check(address):
        if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("OFFLINE_REPLAY_EXTERNAL_EGRESS_REFUSED")

    def connect(sock, address):
        check(address)
        return _originals["connect"](sock, address)

    def connect_ex(sock, address):
        check(address)
        return _originals["connect_ex"](sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("OFFLINE_REPLAY_EXTERNAL_DNS_REFUSED")
        return _originals["getaddrinfo"](host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    pgserver.get_server = lambda pgdata, cleanup_mode="stop": LoopbackPostgres(pgdata)


def pytest_unconfigure(config):
    try:
        for server in reversed(_servers):
            server.cleanup()
    finally:
        if _originals:
            socket.socket.connect = _originals["connect"]
            socket.socket.connect_ex = _originals["connect_ex"]
            socket.getaddrinfo = _originals["getaddrinfo"]
            pgserver.get_server = _originals["get_server"]

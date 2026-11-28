"""Block sockets and database connections throughout offline paper-pilot tests."""
import socket
import sqlite3
from unittest.mock import patch

_patches = []


def _blocked(*args, **kwargs):
    raise AssertionError("Offline paper-pilot suite forbids sockets and database connections")


def pytest_sessionstart(session):
    for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                   "sqlite3.connect"):
        p = patch(target, _blocked)
        p.start()
        _patches.append(p)


def pytest_sessionfinish(session, exitstatus):
    for p in reversed(_patches):
        p.stop()
    _patches.clear()

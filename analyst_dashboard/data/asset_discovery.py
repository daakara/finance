"""Read-only US asset-directory discovery, independent of analytical history.

On-demand refresh only. No orders, scans, DB writes, scoring or SEC requests.
Credentials stay on WEB; reference-directory membership grants no eligibility.
"""
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from urllib.request import Request, HTTPRedirectHandler, build_opener


class DiscoveryUnavailable(ValueError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_catalog():
    key, secret = os.getenv('ALPACA_API_KEY_ID'), os.getenv('ALPACA_API_SECRET_KEY')
    if not key or not secret:
        raise DiscoveryUnavailable('Asset directory not configured')
    # Same reference directory used by the Security Master; no order endpoint.
    request = Request('https://paper-api.alpaca.markets/v2/assets?status=active&asset_class=us_equity',
                      headers={'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret,
                               'Accept': 'application/json'})
    try:
        with build_opener(_NoRedirect()).open(request, timeout=5) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise DiscoveryUnavailable('Asset directory response exceeds bound')
            return json.loads(raw)
    except Exception as exc:
        raise DiscoveryUnavailable('Asset directory temporarily unavailable') from exc


class AssetDirectoryDiscovery:
    def __init__(self, loader=None, clock=None):
        self._loader = loader or _fetch_catalog
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._assets = None
        self._expires = 0
        self._observed = None
        self._retry_after = 0

    def search(self, query):
        query = query.strip()
        if len(query) < 2 or len(query) > 80 or any(ord(c) < 32 or ord(c) == 127 for c in query):
            raise ValueError('Query must contain 2–80 printable characters')
        if not self._lock.acquire(timeout=1):
            raise DiscoveryUnavailable('Asset directory refresh in progress')
        try:
            if self._clock() < self._retry_after:
                raise DiscoveryUnavailable('Asset directory temporarily unavailable')
            if self._assets is None or self._clock() >= self._expires:
                payload = self._loader()
                if not isinstance(payload, list) or not payload or len(payload) > 100_000:
                    raise DiscoveryUnavailable('Invalid asset directory response')
                assets, seen = [], set()
                for row in payload:
                    if not isinstance(row, dict):
                        continue
                    symbol, name = row.get('symbol'), row.get('name')
                    if (row.get('status') != 'active' or row.get('class') != 'us_equity'
                            or not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9.\-_]{0,15}', symbol)
                            or symbol in seen or not isinstance(name, str) or not name.strip()):
                        continue
                    seen.add(symbol)
                    assets.append({'symbol': symbol, 'name': name[:160], 'type': 'US_EQUITY',
                                   'exchange': str(row.get('exchange') or 'Not supplied')[:60],
                                   'source': 'ALPACA_ASSET_DIRECTORY', 'identityVerified': False})
                # Do not turn a malformed provider payload into a certified empty directory.
                if payload and not assets:
                    raise DiscoveryUnavailable('No usable reference assets in directory response')
                self._assets = tuple(assets)
                self._expires = self._clock() + 300
                self._observed = datetime.now(timezone.utc).isoformat()
            q = query.casefold()
            matches = [a for a in self._assets if q in a['symbol'].casefold() or q in a['name'].casefold()]
            def rank(asset):
                symbol, name = asset['symbol'].casefold(), asset['name'].casefold()
                return (0 if symbol == q else 1 if name.startswith(q) else 2 if symbol.startswith(q) else 3, symbol)
            return {'status': 'AVAILABLE', 'results': sorted(matches, key=rank)[:8],
                    'source': 'ALPACA_ASSET_DIRECTORY', 'sourceObservedAt': self._observed,
                    'scope': 'ACTIVE_US_EQUITIES', 'identityVerified': False}
        except DiscoveryUnavailable:
            if self._clock() >= self._retry_after:
                self._retry_after = self._clock() + 30
            raise
        except Exception as exc:
            self._retry_after = self._clock() + 30
            raise DiscoveryUnavailable('Asset directory temporarily unavailable') from exc
        finally:
            self._lock.release()


asset_directory_discovery = AssetDirectoryDiscovery()

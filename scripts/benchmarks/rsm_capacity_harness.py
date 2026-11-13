"""
ARX-ARCHITECTURE-RSM-001 — Isolated Throughput & Capacity Qualification Harness.

Tests ARX API routes under controlled concurrency levels (1, 5, 10, 25, 50) in an isolated,
in-process test environment with representative data and mock provider fixtures.
Never invokes live external market data or production databases.
"""

import sys
import os
import time
import math
import statistics
import concurrent.futures
from typing import Dict, Any, List, Tuple

# Set test environment auth token before importing app
os.environ["ARX_API_KEY"] = "test-rsm-benchmark-key-xyz"
TEST_HEADERS = {"X-API-Key": "test-rsm-benchmark-key-xyz"}

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

BENCHMARK_ROUTES = [
    ("Health Check", "/health", "GET"),
    ("Tactical Setups (Warm)", "/api/v1/analytics/setups?user_role=LONG_TERM", "GET"),
    ("Tactical Setups (Day Trader)", "/api/v1/analytics/setups?user_role=DAY_TRADER", "GET"),
    ("Symbol Setups (LNTH)", "/api/v1/analytics/setups/LNTH", "GET"),
    ("Portfolio Summary", "/api/v1/portfolio/summary", "GET"),
    ("Smart Money Status", "/api/v1/smart-money/status", "GET"),
]

CONCURRENCY_LEVELS = [1, 5, 10, 25, 50]
REQUESTS_PER_CONCURRENCY = 50


def run_single_request(route: str, method: str) -> Tuple[float, int]:
    t0 = time.perf_counter()
    try:
        if method == "GET":
            resp = client.get(route, headers=TEST_HEADERS)
        else:
            resp = client.post(route, headers=TEST_HEADERS)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, resp.status_code
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, 500


def benchmark_concurrency(route: str, method: str, concurrency: int, total_requests: int) -> Dict[str, Any]:
    latencies = []
    status_codes = []

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_single_request, route, method) for _ in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            lat, code = f.result()
            latencies.append(lat)
            status_codes.append(code)
    total_time = time.perf_counter() - t_start

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else 0.0
    p95 = latencies[int(n * 0.95)] if n else 0.0
    p99 = latencies[int(n * 0.99)] if n else 0.0
    rps = n / total_time if total_time > 0 else 0.0
    error_count = sum(1 for c in status_codes if c >= 500)
    error_rate = (error_count / n) * 100.0 if n else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": n,
        "total_time_s": round(total_time, 2),
        "rps": round(rps, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "error_count": error_count,
        "error_rate_pct": round(error_rate, 2),
    }


def run_full_suite() -> Dict[str, Any]:
    print("=================================================================")
    print(" ARX-ARCHITECTURE-RSM-001 — Capacity & Throughput Benchmark")
    print("=================================================================")
    
    # Pre-warm tactical setups cache
    print("Pre-warming route caches...")
    for name, route, method in BENCHMARK_ROUTES:
        client.get(route, headers=TEST_HEADERS) if method == "GET" else client.post(route, headers=TEST_HEADERS)
    print("Pre-warming complete.\n")

    results_by_route = {}
    for name, route, method in BENCHMARK_ROUTES:
        print(f"Testing route: {name} ({route})")
        route_results = []
        for c in CONCURRENCY_LEVELS:
            res = benchmark_concurrency(route, method, concurrency=c, total_requests=REQUESTS_PER_CONCURRENCY)
            route_results.append(res)
            print(f"  Concurrency {c:2d}: RPS={res['rps']:6.1f} | P50={res['p50_ms']:6.1f}ms | P95={res['p95_ms']:6.1f}ms | P99={res['p99_ms']:6.1f}ms | 5xx={res['error_rate_pct']:4.1f}%")
        results_by_route[name] = route_results
        print()

    return results_by_route


if __name__ == "__main__":
    run_full_suite()

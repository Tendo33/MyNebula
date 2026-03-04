"""Benchmark v2 graph APIs.

Usage:
    uv run python scripts/perf/benchmark_graph_api.py --runs 30
    uv run python scripts/perf/benchmark_graph_api.py --runs 30 --mode mock
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class MeasureResult:
    name: str
    durations_ms: list[float]

    @property
    def p50(self) -> float:
        return statistics.median(self.durations_ms) if self.durations_ms else 0.0

    @property
    def p95(self) -> float:
        if not self.durations_ms:
            return 0.0
        sorted_values = sorted(self.durations_ms)
        index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.95) - 1))
        return sorted_values[index]


def _measure(client: httpx.Client, path: str, runs: int) -> MeasureResult:
    durations: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        response = client.get(path)
        response.raise_for_status()
        durations.append((time.perf_counter() - started) * 1000)
    return MeasureResult(name=path, durations_ms=durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--mode", choices=["api", "mock"], default="api")
    args = parser.parse_args()

    if args.mode == "mock":
        warm_tti = 1.22
        cold_tti = 2.68
        edges_p95 = 186.4
        large_graph_fps = 51.3
        print(f"warm_tti_seconds={warm_tti:.2f}")
        print(f"cold_tti_seconds={cold_tti:.2f}")
        print(f"edges_p95_ms={edges_p95:.2f}")
        print(f"large_graph_fps={large_graph_fps:.2f}")
        return

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        graph = _measure(
            client, "/api/v2/graph?version=active&include_edges=false", args.runs
        )
        edges = _measure(
            client,
            "/api/v2/graph/edges?version=active&cursor=0&limit=1200",
            args.runs,
        )

    print(f"{graph.name}: p50={graph.p50:.2f}ms p95={graph.p95:.2f}ms")
    print(f"{edges.name}: p50={edges.p50:.2f}ms p95={edges.p95:.2f}ms")


if __name__ == "__main__":
    main()

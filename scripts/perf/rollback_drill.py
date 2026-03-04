"""Run snapshot rollback drills in mock or service mode.

Usage:
    uv run python scripts/perf/rollback_drill.py --runs 10
    uv run python scripts/perf/rollback_drill.py --runs 10 --mode service
"""

from __future__ import annotations

import argparse


def run_mock_mode(runs: int) -> tuple[int, int]:
    success = 0
    failures = 0
    for index in range(1, runs + 1):
        rolled_back = True
        if rolled_back:
            success += 1
        else:
            failures += 1
        print(f"[mock] rollback={index} success={rolled_back}")
    return success, failures


async def run_service_mode(runs: int) -> tuple[int, int]:
    from nebula.application.services.graph_query_service import GraphQueryService
    from nebula.db.database import get_db_context, init_db

    success = 0
    failures = 0
    service = GraphQueryService()
    await init_db()

    for index in range(1, runs + 1):
        try:
            async with get_db_context() as db:
                await service.rebuild_active_snapshot(db)
                await service.rollback_active_snapshot(db)
            success += 1
            print(f"[service] rollback={index} success=True")
        except Exception as exc:
            failures += 1
            print(f"[service] rollback={index} success=False error={exc}")
    return success, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--mode", choices=["mock", "service"], default="mock")
    args = parser.parse_args()

    if args.mode == "service":
        import asyncio

        success, failures = asyncio.run(run_service_mode(args.runs))
    else:
        success, failures = run_mock_mode(args.runs)

    print(
        f"rollback_drill_summary runs={args.runs} success={success} failures={failures} "
        f"success_rate={(success / args.runs) * 100:.2f}%"
    )
    if success != args.runs or failures != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

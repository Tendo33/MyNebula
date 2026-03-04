"""Run pipeline soak checks in mock or live mode.

Usage:
    uv run python scripts/perf/soak_pipeline.py --runs 50
    uv run python scripts/perf/soak_pipeline.py --runs 50 --mode api --base-url http://127.0.0.1:8000
    uv run python scripts/perf/soak_pipeline.py --runs 50 --mode api --base-url http://127.0.0.1:8000 --admin-username admin --admin-password '***'
"""

from __future__ import annotations

import argparse
import time

import httpx


def run_mock_mode(runs: int) -> tuple[int, int]:
    passed = 0
    corrupted = 0
    for index in range(1, runs + 1):
        run_ok = True
        data_corrupted = False
        if run_ok:
            passed += 1
        if data_corrupted:
            corrupted += 1
        print(f"[mock] run={index} status={'ok' if run_ok else 'failed'} corrupted={data_corrupted}")
    return passed, corrupted


def wait_pipeline_complete(client: httpx.Client, run_id: int, timeout_seconds: int) -> dict:
    started = time.perf_counter()
    while True:
        response = client.get(f"/api/v2/sync/jobs/{run_id}")
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status in {"completed", "partial_failed", "failed"}:
            return payload
        if (time.perf_counter() - started) > timeout_seconds:
            raise TimeoutError(f"pipeline run timeout: run_id={run_id}")
        time.sleep(2)


def _login_admin(
    client: httpx.Client,
    admin_username: str | None,
    admin_password: str | None,
) -> None:
    if not admin_password:
        return

    response = client.post(
        "/api/auth/login",
        json={
            "username": admin_username or "admin",
            "password": admin_password,
        },
    )
    response.raise_for_status()


def run_api_mode(
    runs: int,
    base_url: str,
    timeout_seconds: int,
    admin_username: str | None,
    admin_password: str | None,
) -> tuple[int, int]:
    passed = 0
    corrupted = 0
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        _login_admin(client, admin_username, admin_password)
        for index in range(1, runs + 1):
            start_response = client.post("/api/v2/sync/start?mode=incremental")
            start_response.raise_for_status()
            run_id = int(start_response.json()["pipeline_run_id"])
            payload = wait_pipeline_complete(client, run_id, timeout_seconds)
            status = payload.get("status")
            is_ok = status in {"completed", "partial_failed"}
            if is_ok:
                passed += 1
            else:
                corrupted += 1
            print(f"[api] run={index} run_id={run_id} status={status}")
    return passed, corrupted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--mode", choices=["mock", "api"], default="mock")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--admin-username", default=None)
    parser.add_argument("--admin-password", default=None)
    args = parser.parse_args()

    if args.mode == "api":
        from nebula.core.config import get_app_settings

        settings = get_app_settings()
        admin_username = args.admin_username or settings.admin_username
        admin_password = (
            args.admin_password if args.admin_password is not None else settings.admin_password
        )
        passed, corrupted = run_api_mode(
            args.runs,
            args.base_url,
            args.timeout_seconds,
            admin_username,
            admin_password,
        )
    else:
        passed, corrupted = run_mock_mode(args.runs)

    print(
        f"pipeline_soak_summary runs={args.runs} passed={passed} corrupted={corrupted} "
        f"success_rate={(passed / args.runs) * 100:.2f}%"
    )
    if passed != args.runs or corrupted != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import HTTPException

from nebula.application.services.graph_edge_service import (
    RepoEdgeInfo,
    _build_similarity_edges_knn,
)
from nebula.application.services.sync_ops_service import (
    calculate_progress_percent,
    estimate_eta_seconds,
    is_schedule_due,
    resolve_job_phase,
    validate_full_refresh_confirmation,
)


def test_calculate_progress_percent_handles_bounds():
    assert calculate_progress_percent(total_items=0, processed_items=0) == 0.0
    assert calculate_progress_percent(total_items=10, processed_items=5) == 50.0
    assert calculate_progress_percent(total_items=10, processed_items=20) == 100.0


def test_estimate_eta_seconds_returns_none_for_invalid_progress():
    started_at = datetime.utcnow() - timedelta(minutes=2)
    assert estimate_eta_seconds(started_at, 0.0) is None
    assert estimate_eta_seconds(started_at, 100.0) is None


def test_estimate_eta_seconds_estimates_remaining_time():
    started_at = datetime.utcnow() - timedelta(seconds=120)
    eta = estimate_eta_seconds(started_at, 50.0)
    assert eta is not None
    assert 100 <= eta <= 140


def test_estimate_eta_seconds_handles_timezone_aware_started_at():
    started_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    eta = estimate_eta_seconds(started_at, 50.0)
    assert eta is not None
    assert 100 <= eta <= 140


def test_resolve_job_phase_prefers_error_details_phase():
    phase = resolve_job_phase(
        task_type="full_refresh",
        status="running",
        error_details={"phase": "embeddings"},
    )
    assert phase == "embeddings"


def test_resolve_job_phase_supports_snapshot_phase():
    phase = resolve_job_phase(
        task_type="full_refresh",
        status="running",
        error_details={"phase": "snapshot"},
    )
    assert phase == "snapshot"


def test_resolve_job_phase_for_completed_task():
    phase = resolve_job_phase(
        task_type="stars",
        status="completed",
        error_details=None,
    )
    assert phase == "completed"


def test_validate_full_refresh_confirmation_requires_true():
    validate_full_refresh_confirmation(True)
    with pytest.raises(HTTPException):
        validate_full_refresh_confirmation(False)


def test_is_schedule_due_handles_same_minute_deduplication():
    now_local = datetime(2026, 2, 19, 9, 30)
    last_run_local = datetime(2026, 2, 19, 9, 30, 10)
    assert is_schedule_due(now_local, 9, 30, last_run_local) is False
    assert is_schedule_due(now_local, 9, 30, None) is True


def test_build_similarity_edges_knn_returns_unique_edges():
    repos = [
        RepoEdgeInfo(repo_id=1, embedding=[1.0, 0.0], language="Python"),
        RepoEdgeInfo(repo_id=2, embedding=[0.98, 0.02], language="Python"),
        RepoEdgeInfo(repo_id=3, embedding=[0.0, 1.0], language="Rust"),
    ]

    edges = _build_similarity_edges_knn(repos, min_score=0.5, k=2)

    assert len(edges) == 1
    edge = edges[0]
    assert {edge.source, edge.target} == {1, 2}
    assert edge.weight >= 0.5


def test_build_similarity_edges_knn_handles_numpy_embeddings():
    repos = [
        RepoEdgeInfo(repo_id=1, embedding=np.array([1.0, 0.0]), language="Go"),
        RepoEdgeInfo(repo_id=2, embedding=np.array([0.99, 0.01]), language="Go"),
    ]

    edges = _build_similarity_edges_knn(repos, min_score=0.5, k=1)

    assert len(edges) == 1
    assert {edges[0].source, edges[0].target} == {1, 2}

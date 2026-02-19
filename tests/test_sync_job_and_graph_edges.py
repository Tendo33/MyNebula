from datetime import datetime, timedelta

import pytest

from nebula.api.graph import _build_similarity_edges_knn
from nebula.api.sync import (
    _calculate_progress_percent,
    _estimate_eta_seconds,
    _is_schedule_due,
    _resolve_job_phase,
    _validate_full_refresh_confirmation,
)


def test_calculate_progress_percent_handles_bounds():
    assert _calculate_progress_percent(total_items=0, processed_items=0) == 0.0
    assert _calculate_progress_percent(total_items=10, processed_items=5) == 50.0
    assert _calculate_progress_percent(total_items=10, processed_items=20) == 100.0


def test_estimate_eta_seconds_returns_none_for_invalid_progress():
    started_at = datetime.utcnow() - timedelta(minutes=2)
    assert _estimate_eta_seconds(started_at, 0.0) is None
    assert _estimate_eta_seconds(started_at, 100.0) is None


def test_estimate_eta_seconds_estimates_remaining_time():
    started_at = datetime.utcnow() - timedelta(seconds=120)
    eta = _estimate_eta_seconds(started_at, 50.0)
    assert eta is not None
    assert 100 <= eta <= 140


def test_resolve_job_phase_prefers_error_details_phase():
    phase = _resolve_job_phase(
        task_type="full_refresh",
        status="running",
        error_details={"phase": "embeddings"},
    )
    assert phase == "embeddings"


def test_resolve_job_phase_for_completed_task():
    phase = _resolve_job_phase(
        task_type="stars",
        status="completed",
        error_details=None,
    )
    assert phase == "completed"


def test_validate_full_refresh_confirmation_requires_true():
    _validate_full_refresh_confirmation(True)
    with pytest.raises(Exception):
        _validate_full_refresh_confirmation(False)


def test_is_schedule_due_handles_same_minute_deduplication():
    now_local = datetime(2026, 2, 19, 9, 30)
    last_run_local = datetime(2026, 2, 19, 9, 30, 10)
    assert _is_schedule_due(now_local, 9, 30, last_run_local) is False
    assert _is_schedule_due(now_local, 9, 30, None) is True


def test_build_similarity_edges_knn_returns_unique_edges():
    repo_ids = [1, 2, 3]
    embeddings = [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.0, 1.0],
    ]

    edges = _build_similarity_edges_knn(
        repo_ids=repo_ids,
        embeddings=embeddings,
        min_similarity=0.8,
        k=2,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert {edge.source, edge.target} == {1, 2}
    assert edge.weight >= 0.8

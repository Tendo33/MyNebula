from datetime import datetime, timezone

from nebula.application.services.graph_snapshot_service import GraphSnapshotBuilderService
from nebula.domain import PipelinePhase, PipelineStatus, SnapshotStatus
from nebula.schemas.graph import GraphData


def test_build_version_contains_counts():
    service = GraphSnapshotBuilderService()
    now = datetime(2026, 3, 3, 12, 30, tzinfo=timezone.utc)
    graph_data = GraphData(
        nodes=[],
        edges=[],
        clusters=[],
        star_lists=[],
        total_nodes=12,
        total_edges=34,
        total_clusters=2,
        total_star_lists=1,
    )

    version = service._build_version(graph_data, now)

    assert version.startswith("snapshot-20260303123000-")
    assert "n12" in version
    assert "e34" in version


def test_domain_status_values_are_stable():
    assert SnapshotStatus.ready.value == "ready"
    assert PipelineStatus.completed.value == "completed"
    assert PipelinePhase.snapshot.value == "snapshot"

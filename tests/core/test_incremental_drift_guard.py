from nebula.application.services.sync_execution_service import (
    should_force_full_recluster,
)


def test_force_full_recluster_when_new_ratio_exceeds_threshold():
    assert should_force_full_recluster(total_repos=100, new_repos=25)


def test_force_full_recluster_when_centroid_drift_exceeds_threshold():
    assert should_force_full_recluster(
        total_repos=100,
        new_repos=5,
        centroid_drift=0.5,
    )


def test_keep_incremental_when_under_thresholds():
    assert not should_force_full_recluster(
        total_repos=100,
        new_repos=5,
        centroid_drift=0.1,
    )

from nebula.api.sync import _should_force_full_recluster


def test_force_full_recluster_when_new_ratio_exceeds_threshold():
    assert _should_force_full_recluster(total_repos=100, new_repos=25)


def test_force_full_recluster_when_centroid_drift_exceeds_threshold():
    assert _should_force_full_recluster(
        total_repos=100,
        new_repos=5,
        centroid_drift=0.5,
    )


def test_keep_incremental_when_under_thresholds():
    assert not _should_force_full_recluster(
        total_repos=100,
        new_repos=5,
        centroid_drift=0.1,
    )

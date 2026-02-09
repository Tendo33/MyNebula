from types import SimpleNamespace

from nebula.api.sync import _resolve_cluster_assignments


def test_resolve_cluster_assignments_maps_temp_ids_without_mutating_repos():
    repos = [
        SimpleNamespace(id=1, cluster_id=10),
        SimpleNamespace(id=2, cluster_id=20),
    ]

    provisional_assignments = {
        1: -1,
        2: 20,
    }
    temp_cluster_to_db_id = {
        -1: 100,
    }

    resolved = _resolve_cluster_assignments(
        repos,
        provisional_assignments,
        temp_cluster_to_db_id,
    )

    assert resolved == {1: 100, 2: 20}
    assert repos[0].cluster_id == 10
    assert repos[1].cluster_id == 20


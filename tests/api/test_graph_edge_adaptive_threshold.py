from nebula.application.services.graph_edge_service import (
    RepoEdgeInfo,
    _estimate_adaptive_relevance_threshold,
)


def _make_repos(embeddings: list[list[float]]) -> list[RepoEdgeInfo]:
    return [RepoEdgeInfo(repo_id=i, embedding=e) for i, e in enumerate(embeddings)]


def test_adaptive_threshold_stays_in_bounds_sparse_and_dense():
    dense_repos = _make_repos([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]])
    sparse_repos = _make_repos([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    dense_threshold = _estimate_adaptive_relevance_threshold(
        dense_repos, target_degree=2
    )
    sparse_threshold = _estimate_adaptive_relevance_threshold(
        sparse_repos, target_degree=2
    )

    assert 0.35 <= dense_threshold <= 0.85
    assert 0.35 <= sparse_threshold <= 0.85
    assert sparse_threshold <= dense_threshold

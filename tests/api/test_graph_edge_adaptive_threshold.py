from nebula.api.graph import _estimate_adaptive_similarity_threshold


def test_adaptive_threshold_stays_in_bounds_sparse_and_dense():
    dense_embeddings = [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03]]
    sparse_embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]

    dense_threshold = _estimate_adaptive_similarity_threshold(dense_embeddings, target_degree=2)
    sparse_threshold = _estimate_adaptive_similarity_threshold(sparse_embeddings, target_degree=2)

    assert 0.5 <= dense_threshold <= 0.95
    assert 0.5 <= sparse_threshold <= 0.95
    assert sparse_threshold <= dense_threshold

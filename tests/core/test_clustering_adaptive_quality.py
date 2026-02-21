import numpy as np

from nebula.core.clustering import ClusteringService, mark_cluster_outliers


def test_mark_cluster_outliers_marks_extreme_points_with_noise_cap():
    labels = np.array([0] * 20)
    features = np.array([[0.0, 0.0]] * 18 + [[8.0, 8.0], [9.0, 9.0]], dtype=np.float32)

    updated = mark_cluster_outliers(labels, features, max_noise_ratio=0.15)
    noise_count = int((updated == -1).sum())

    assert noise_count >= 1
    assert noise_count <= 3


def test_clustering_service_produces_multiple_clusters_on_separable_data():
    embeddings = (
        [[1.0, 0.0, 0.0] for _ in range(12)]
        + [[0.0, 1.0, 0.0] for _ in range(12)]
        + [[0.0, 0.0, 1.0] for _ in range(12)]
    )

    service = ClusteringService(min_clusters=3, target_min_clusters=3, target_max_clusters=8)
    result = service.fit_transform(embeddings=embeddings, resolve_overlap=False)

    assert result.n_clusters >= 3

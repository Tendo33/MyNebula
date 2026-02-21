from scripts.evals.eval_clusters import evaluate_clusters
from scripts.evals.eval_related import evaluate_related


def test_eval_harness_outputs_required_keys():
    related = evaluate_related(
        goldset_rows=[{"anchor_repo_id": 1, "positive_ids": [2], "negative_ids": [3]}],
        predictions={1: [2, 4]},
    )
    clusters = evaluate_clusters(
        [
            {"repo_id": 1, "predicted_cluster": 0, "true_label": "rag"},
            {"repo_id": 2, "predicted_cluster": 0, "true_label": "rag"},
        ]
    )

    assert "p_at_5" in related
    assert "ndcg_at_10" in related
    assert "coverage" in related
    assert "cluster_purity" in clusters

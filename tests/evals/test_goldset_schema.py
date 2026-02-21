import json

from scripts.evals.eval_clusters import load_jsonl as load_cluster_jsonl
from scripts.evals.eval_related import load_jsonl as load_related_jsonl


def test_related_goldset_schema(tmp_path):
    path = tmp_path / "related_goldset.jsonl"
    path.write_text(
        json.dumps(
            {
                "anchor_repo_id": 1,
                "positive_ids": [2, 3],
                "negative_ids": [4],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_related_jsonl(path)
    assert rows

    for row in rows:
        assert isinstance(row["anchor_repo_id"], int)
        assert isinstance(row["positive_ids"], list)
        assert isinstance(row["negative_ids"], list)


def test_cluster_goldset_schema(tmp_path):
    path = tmp_path / "cluster_goldset.jsonl"
    path.write_text(
        json.dumps(
            {
                "repo_id": 1,
                "predicted_cluster": 0,
                "true_label": "rag",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_cluster_jsonl(path)
    assert rows

    for row in rows:
        assert isinstance(row["repo_id"], int)
        assert isinstance(row["predicted_cluster"], int)
        assert isinstance(row["true_label"], str)

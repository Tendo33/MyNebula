"""Run offline quality checks and enforce thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLDS = {
    "p_at_5": 0.80,
    "coverage": 0.95,
    "cluster_purity": 0.75,
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _evaluate_related(goldset_rows: list[dict], predictions: dict[int, list[int]]) -> dict:
    if not goldset_rows:
        return {"p_at_5": 0.0, "ndcg_at_10": 0.0, "coverage": 0.0}

    def p_at_5(predicted: list[int], positives: set[int]) -> float:
        topk = predicted[:5]
        if not topk:
            return 0.0
        hit = sum(1 for rid in topk if rid in positives)
        return hit / 5.0

    p_scores = []
    covered = 0
    for row in goldset_rows:
        anchor = int(row["anchor_repo_id"])
        positives = {int(x) for x in row.get("positive_ids", [])}
        predicted = predictions.get(anchor, [])
        if predicted:
            covered += 1
        p_scores.append(p_at_5(predicted, positives))

    n = len(goldset_rows)
    return {
        "p_at_5": sum(p_scores) / n,
        "ndcg_at_10": 0.0,
        "coverage": covered / float(n),
    }


def _evaluate_clusters(rows: list[dict]) -> dict:
    if not rows:
        return {"cluster_purity": 0.0}

    by_cluster: dict[int, dict[str, int]] = {}
    clustered_count = 0
    for row in rows:
        cluster = row.get("predicted_cluster")
        label = str(row.get("true_label", "")).strip()
        if cluster is None or int(cluster) == -1:
            continue
        clustered_count += 1
        bucket = by_cluster.setdefault(int(cluster), {})
        bucket[label] = bucket.get(label, 0) + 1

    if clustered_count == 0:
        return {"cluster_purity": 0.0}

    dominant = 0
    for labels in by_cluster.values():
        dominant += max(labels.values())

    return {"cluster_purity": dominant / float(clustered_count)}


def run_checks() -> dict:
    related_rows = _load_jsonl(Path("data/eval/related_goldset.jsonl"))
    related_predictions_path = Path("data/eval/related_predictions.json")
    if related_predictions_path.exists():
        predictions_raw = json.loads(related_predictions_path.read_text(encoding="utf-8"))
        predictions = {int(k): [int(v) for v in values] for k, values in predictions_raw.items()}
    else:
        predictions = {}

    related_metrics = _evaluate_related(related_rows, predictions)
    cluster_rows = _load_jsonl(Path("data/eval/cluster_goldset.jsonl"))
    cluster_metrics = _evaluate_clusters(cluster_rows)

    result = {
        **related_metrics,
        **cluster_metrics,
    }
    return result


def main() -> None:
    metrics = run_checks()
    failed: list[str] = []

    if metrics.get("p_at_5", 0.0) < THRESHOLDS["p_at_5"]:
        failed.append("p_at_5")
    if metrics.get("coverage", 0.0) < THRESHOLDS["coverage"]:
        failed.append("coverage")
    if metrics.get("cluster_purity", 0.0) < THRESHOLDS["cluster_purity"]:
        failed.append("cluster_purity")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    report_path = Path("data/eval/quality_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if failed:
        print(f"Quality gate failed: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

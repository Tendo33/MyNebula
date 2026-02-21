"""Offline evaluation for clustering quality."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def evaluate_clusters(goldset_rows: list[dict]) -> dict:
    """Compute purity from labeled sample rows.

    Expected row shape:
    {
      "repo_id": 123,
      "predicted_cluster": 4,
      "true_label": "rag"
    }
    """
    if not goldset_rows:
        return {
            "cluster_purity": 0.0,
            "cluster_coverage": 0.0,
            "samples": 0,
        }

    grouped: dict[int, list[str]] = defaultdict(list)
    clustered_samples = 0
    for row in goldset_rows:
        cluster_id = row.get("predicted_cluster")
        true_label = str(row.get("true_label", "")).strip()
        if cluster_id is None or cluster_id == -1:
            continue
        clustered_samples += 1
        grouped[int(cluster_id)].append(true_label)

    if not grouped:
        return {
            "cluster_purity": 0.0,
            "cluster_coverage": 0.0,
            "samples": len(goldset_rows),
        }

    weighted_majority = 0
    total = 0
    for labels in grouped.values():
        counter = Counter(labels)
        majority = counter.most_common(1)[0][1]
        weighted_majority += majority
        total += len(labels)

    purity = (weighted_majority / float(total)) if total > 0 else 0.0
    coverage = clustered_samples / float(len(goldset_rows))
    return {
        "cluster_purity": purity,
        "cluster_coverage": coverage,
        "samples": len(goldset_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clustering quality")
    parser.add_argument(
        "--goldset",
        type=Path,
        default=Path("data/eval/cluster_goldset.jsonl"),
    )
    args = parser.parse_args()

    rows = load_jsonl(args.goldset)
    result = evaluate_clusters(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

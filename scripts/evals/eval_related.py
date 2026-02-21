"""Offline evaluation for related-repository quality."""

from __future__ import annotations

import argparse
import json
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


def precision_at_k(predicted: list[int], positives: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = predicted[:k]
    if not topk:
        return 0.0
    hit = sum(1 for rid in topk if rid in positives)
    return hit / float(k)


def ndcg_at_k(predicted: list[int], positives: set[int], k: int) -> float:
    import math

    topk = predicted[:k]
    if not topk:
        return 0.0

    dcg = 0.0
    for i, rid in enumerate(topk, start=1):
        rel = 1.0 if rid in positives else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)

    ideal_hits = min(len(positives), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_related(
    goldset_rows: list[dict],
    predictions: dict[int, list[int]],
    k: int = 5,
) -> dict:
    if not goldset_rows:
        return {
            "p_at_5": 0.0,
            "ndcg_at_10": 0.0,
            "coverage": 0.0,
            "anchors": 0,
        }

    p_scores: list[float] = []
    ndcg_scores: list[float] = []
    covered = 0

    for row in goldset_rows:
        anchor = int(row["anchor_repo_id"])
        positives = {int(x) for x in row.get("positive_ids", [])}
        predicted = [int(x) for x in predictions.get(anchor, [])]
        if predicted:
            covered += 1

        p_scores.append(precision_at_k(predicted, positives, k=k))
        ndcg_scores.append(ndcg_at_k(predicted, positives, k=10))

    n = len(goldset_rows)
    return {
        "p_at_5": sum(p_scores) / n,
        "ndcg_at_10": sum(ndcg_scores) / n,
        "coverage": covered / float(n),
        "anchors": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate related repository quality")
    parser.add_argument(
        "--goldset",
        type=Path,
        default=Path("data/eval/related_goldset.jsonl"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/eval/related_predictions.json"),
    )
    args = parser.parse_args()

    goldset_rows = load_jsonl(args.goldset)
    predictions_raw = {}
    if args.predictions.exists():
        predictions_raw = json.loads(args.predictions.read_text(encoding="utf-8"))

    predictions: dict[int, list[int]] = {
        int(k): [int(v) for v in values]
        for k, values in predictions_raw.items()
    }

    result = evaluate_related(goldset_rows, predictions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

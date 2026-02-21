"""Interactive helper to append rows to evaluation goldsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_int_list(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def append_related_row(path: Path) -> None:
    anchor = int(input("anchor_repo_id: ").strip())
    positives = _parse_int_list(input("positive_ids (comma-separated): "))
    negatives = _parse_int_list(input("negative_ids (comma-separated): "))
    row = {
        "anchor_repo_id": anchor,
        "positive_ids": positives,
        "negative_ids": negatives,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended related row to {path}")


def append_cluster_row(path: Path) -> None:
    repo_id = int(input("repo_id: ").strip())
    predicted_cluster = int(input("predicted_cluster (-1 for noise): ").strip())
    true_label = input("true_label: ").strip()
    row = {
        "repo_id": repo_id,
        "predicted_cluster": predicted_cluster,
        "true_label": true_label,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended cluster row to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation goldsets")
    parser.add_argument(
        "--type",
        choices=["related", "cluster"],
        required=True,
        help="goldset type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override output path",
    )
    args = parser.parse_args()

    if args.type == "related":
        path = args.output or Path("data/eval/related_goldset.jsonl")
        append_related_row(path)
    else:
        path = args.output or Path("data/eval/cluster_goldset.jsonl")
        append_cluster_row(path)


if __name__ == "__main__":
    main()

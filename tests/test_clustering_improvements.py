import numpy as np

from nebula.core.clustering import (
    build_cluster_naming_inputs,
    deduplicate_cluster_entries,
    merge_similar_clusters,
    normalize_topic_lists,
    sanitize_cluster_name,
)


def test_normalize_topic_lists_maps_agent_memory_synonyms():
    topics = [
        ["agent-memory", "mem0", "RAG-Memory"],
        ["long-term-memory", "Agent_Memory"],
    ]

    normalized = normalize_topic_lists(topics)

    assert normalized[0][0] == "agent-memory"
    assert normalized[0].count("agent-memory") == 1
    assert normalized[1] == ["agent-memory"]


def test_merge_similar_clusters_merges_close_centroids():
    labels = np.array([0, 0, 1, 1, 2, 2])
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.05, 0.0],
            [0.99, 0.02, 0.0],
            [0.97, 0.04, 0.0],
            [0.0, 1.0, 0.0],
            [0.02, 0.99, 0.0],
        ],
        dtype=np.float32,
    )

    merged = merge_similar_clusters(
        labels=labels,
        embeddings=embeddings,
        similarity_threshold=0.95,
    )

    # Clusters 0 and 1 should merge, cluster 2 should stay separate
    assert len(set(merged.tolist())) == 2


def test_deduplicate_cluster_entries_generates_unique_names():
    entries = [
        {
            "cluster_id": 1,
            "name": "AI 开发工具集",
            "keywords": ["agent"],
            "repo_count": 12,
        },
        {
            "cluster_id": 2,
            "name": "AI 开发工具",
            "keywords": ["training"],
            "repo_count": 8,
        },
    ]

    deduped = deduplicate_cluster_entries(entries)

    assert deduped[0]["name"] == "AI 开发工具"
    assert deduped[1]["name"] != "AI 开发工具"
    assert deduped[1]["name"].startswith("AI 开发工具")


def test_sanitize_cluster_name_normalizes_spacing_and_suffix():
    assert sanitize_cluster_name("  AI   开发 工具集 ") == "AI 开发 工具"


def test_build_cluster_naming_inputs_sorts_by_stars_and_normalizes_topics():
    class Repo:
        def __init__(self, full_name: str, stars: int, topics: list[str]):
            self.full_name = full_name
            self.stargazers_count = stars
            self.description = ""
            self.topics = topics
            self.language = "Python"

    repos = [
        Repo("z/second", 20, ["distributed-training", "deepspeed"]),
        Repo("a/first", 200, ["agent-memory", "mem0"]),
    ]

    repo_names, _, topics, _ = build_cluster_naming_inputs(repos)

    assert repo_names == ["a/first", "z/second"]
    assert topics[0] == ["agent-memory"]

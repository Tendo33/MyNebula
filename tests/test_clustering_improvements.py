import numpy as np

from nebula.core.clustering import (
    ClusteringService,
    build_cluster_naming_inputs,
    deduplicate_cluster_entries,
    generate_incremental_coords,
    merge_similar_clusters,
    normalize_topic_lists,
    pick_incremental_cluster,
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


def test_pick_incremental_cluster_prefers_highest_similarity():
    cluster_embeddings = {
        10: [1.0, 0.0, 0.0],
        11: [0.0, 1.0, 0.0],
    }

    cluster_id, similarity = pick_incremental_cluster(
        embedding=[0.9, 0.1, 0.0],
        cluster_embeddings=cluster_embeddings,
        min_similarity=0.6,
    )

    assert cluster_id == 10
    assert similarity > 0.9


def test_pick_incremental_cluster_returns_none_when_similarity_low():
    cluster_embeddings = {
        10: [1.0, 0.0, 0.0],
    }

    cluster_id, similarity = pick_incremental_cluster(
        embedding=[0.0, 1.0, 0.0],
        cluster_embeddings=cluster_embeddings,
        min_similarity=0.3,
    )

    assert cluster_id is None
    assert similarity == 0.0


def test_generate_incremental_coords_is_deterministic():
    center = [0.3, -0.2, 1.1]

    first = generate_incremental_coords(center, seed=42, radius=0.1)
    second = generate_incremental_coords(center, seed=42, radius=0.1)

    assert first == second
    assert first != center


def test_clustering_projection_does_not_import_umap(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "umap":
            raise AssertionError("UMAP should not be imported for layout projection")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    service = ClusteringService(min_cluster_size=2, min_samples=1)
    result = service.fit_transform(
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        resolve_overlap=False,
    )

    assert len(result.coords_3d) == 2
    assert len(result.coords_3d[0]) == 3


def test_clustering_projection_is_deterministic_for_same_input():
    service = ClusteringService(min_cluster_size=2, min_samples=1)
    embeddings = [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0]]

    first = service.fit_transform(embeddings=embeddings, resolve_overlap=False)
    second = service.fit_transform(embeddings=embeddings, resolve_overlap=False)

    assert first.coords_3d == second.coords_3d


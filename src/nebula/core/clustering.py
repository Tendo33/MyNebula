"""Semantic clustering service with embedding-space clustering.

This module clusters repositories in normalized embedding space for semantic
fidelity, while using UMAP solely for 3D visualization coordinates.

Key features:
- Embedding-space HDBSCAN clustering for robust semantic grouping
- Soft post-merge of highly similar clusters to reduce fragmentation
- Noise point assignment to ensure all nodes are attached to a cluster
"""

import math
import re
from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from nebula.utils import get_logger

logger = get_logger(__name__)


def resolve_collisions(
    coords: np.ndarray,
    node_sizes: list[float] | None = None,
    min_distance: float = 0.35,
    iterations: int = 100,
    repulsion_strength: float = 0.5,
) -> np.ndarray:
    """Resolve node collisions in 3D space using force-directed repulsion.

    This algorithm iteratively pushes overlapping nodes apart while trying
    to preserve the overall cluster structure from UMAP.

    Args:
        coords: Nx3 array of 3D coordinates
        node_sizes: Optional list of node sizes (used for distance calculation)
        min_distance: Minimum distance between node centers
        iterations: Number of iterations for collision resolution
        repulsion_strength: How strongly nodes repel each other (0.5 = fix half the overlap per step)

    Returns:
        Adjusted Nx3 coordinates with resolved collisions
    """
    if len(coords) < 2:
        return coords

    result = coords.copy()
    n = len(result)

    # Default sizes if not provided
    if node_sizes is None:
        node_sizes = [1.0] * n

    rng = np.random.RandomState(42)

    for iteration in range(iterations):
        collision_count = 0
        displacements = np.zeros_like(result)

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate distance between nodes
                diff = result[i] - result[j]
                dist = np.linalg.norm(diff)

                # Calculate minimum allowed distance based on node sizes
                size_i = node_sizes[i] if i < len(node_sizes) else 1.0
                size_j = node_sizes[j] if j < len(node_sizes) else 1.0
                required_dist = min_distance * (size_i + size_j) / 2

                # If nodes are too close, push them apart
                if dist < required_dist and dist > 1e-6:
                    # Calculate repulsion direction
                    direction = diff / dist
                    overlap = required_dist - dist

                    # Apply displacement proportional to overlap
                    displacement = direction * overlap * repulsion_strength
                    displacements[i] += displacement
                    displacements[j] -= displacement
                    collision_count += 1
                elif dist <= 1e-6:
                    # Nodes are at same position, add deterministic random displacement
                    random_dir = rng.randn(3)
                    norm = np.linalg.norm(random_dir)
                    if norm > 0:
                        random_dir /= norm
                    else:
                        random_dir = np.array([1.0, 0.0, 0.0])
                    displacements[i] += random_dir * min_distance * 0.5
                    displacements[j] -= random_dir * min_distance * 0.5
                    collision_count += 1

        # Apply displacements
        result += displacements

        # Early exit if no collisions were found
        if collision_count == 0:
            logger.debug(f"Collision resolution converged at iteration {iteration}")
            break

        if iteration % 20 == 0 and iteration > 0:
            logger.debug(
                f"Collision resolution iteration {iteration}: {collision_count} collisions remaining"
            )

    return result


def assign_noise_to_nearest_cluster(
    coords: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Assign noise points (-1 labels) to the nearest cluster.

    Uses K-Nearest Neighbors to find the closest clustered point for each
    noise point and assigns it to that cluster.

    Args:
        coords: Nx3 array of 3D coordinates
        labels: Array of cluster labels (-1 indicates noise)
        embeddings: Optional original embeddings for distance calculation.
                   If provided, uses embedding space; otherwise uses 3D coords.

    Returns:
        Updated labels with no noise points (-1)
    """
    labels = labels.copy()
    noise_mask = labels == -1

    if not noise_mask.any():
        logger.debug("No noise points to assign")
        return labels

    noise_count = noise_mask.sum()
    clustered_mask = ~noise_mask

    if not clustered_mask.any():
        # All points are noise - assign all to cluster 0
        logger.warning("All points are noise, assigning all to cluster 0")
        return np.zeros_like(labels)

    # Use embeddings if available, otherwise use 3D coordinates
    if embeddings is not None:
        reference_data = embeddings[clustered_mask]
        query_data = embeddings[noise_mask]
        metric = "cosine"
    else:
        reference_data = coords[clustered_mask]
        query_data = coords[noise_mask]
        metric = "euclidean"

    # Find nearest clustered neighbor for each noise point
    nn = NearestNeighbors(n_neighbors=1, metric=metric)
    nn.fit(reference_data)
    _, indices = nn.kneighbors(query_data)

    # Get the labels of the nearest clustered points
    clustered_labels = labels[clustered_mask]
    labels[noise_mask] = clustered_labels[indices.flatten()]

    logger.info(f"Assigned {noise_count} noise points to nearest clusters")
    return labels


def estimate_cluster_count(n_samples: int, min_clusters: int = 3) -> int:
    """Estimate a reasonable number of clusters based on sample count.

    Uses a heuristic based on sqrt(n) which works well for most datasets.

    Args:
        n_samples: Number of data points
        min_clusters: Minimum number of clusters to return

    Returns:
        Estimated number of clusters (guaranteed <= n_samples)
    """
    # sqrt(n) rule of thumb, with minimum and maximum bounds
    estimated = int(math.sqrt(n_samples))
    # Ensure we don't request more clusters than samples
    max_clusters = min(n_samples, 20)
    result = max(min_clusters, min(estimated, max_clusters))
    # Final safety check: never more clusters than samples
    return min(result, n_samples)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for cosine-like distance computations."""
    if embeddings.size == 0:
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def relabel_clusters(labels: np.ndarray) -> np.ndarray:
    """Relabel cluster IDs to contiguous integers while preserving noise as -1."""
    unique_labels = sorted(
        {int(label) for label in labels.tolist() if int(label) != -1}
    )
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    relabeled = np.array([mapping.get(int(label), -1) for label in labels], dtype=int)
    return relabeled


def merge_similar_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.9,
    target_max_clusters: int | None = None,
    min_similarity_for_forced_merge: float = 0.45,
) -> np.ndarray:
    """Merge clusters with very similar centroids.

    This is used as a post-processing step to reduce over-fragmentation and to
    enforce a soft target maximum cluster count without forcing arbitrary splits.
    """
    merged_labels = labels.copy()
    if len(merged_labels) == 0:
        return merged_labels

    max_iterations = max(50, len(set(merged_labels.tolist())) * 4)

    for _ in range(max_iterations):
        cluster_ids = sorted(
            {
                int(cluster_id)
                for cluster_id in merged_labels.tolist()
                if cluster_id != -1
            }
        )

        if len(cluster_ids) <= 1:
            break

        centers: dict[int, np.ndarray] = {}
        sizes: dict[int, int] = {}

        for cluster_id in cluster_ids:
            mask = merged_labels == cluster_id
            sizes[cluster_id] = int(mask.sum())
            center = embeddings[mask].mean(axis=0)
            center_norm = np.linalg.norm(center)
            centers[cluster_id] = center / center_norm if center_norm > 0 else center

        best_pair: tuple[int, int] | None = None
        best_similarity = -1.0

        for i, cluster_i in enumerate(cluster_ids):
            center_i = centers[cluster_i]
            for cluster_j in cluster_ids[i + 1 :]:
                similarity = float(np.dot(center_i, centers[cluster_j]))
                if not np.isfinite(similarity):
                    continue
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (cluster_i, cluster_j)

        if best_pair is None:
            break

        need_force_reduce = (
            target_max_clusters is not None and len(cluster_ids) > target_max_clusters
        )
        can_merge_by_similarity = best_similarity >= similarity_threshold
        can_merge_by_target = (
            need_force_reduce and best_similarity >= min_similarity_for_forced_merge
        )

        if not can_merge_by_similarity and not can_merge_by_target:
            break

        cluster_a, cluster_b = best_pair
        keep_cluster, drop_cluster = (
            (cluster_a, cluster_b)
            if sizes[cluster_a] >= sizes[cluster_b]
            else (cluster_b, cluster_a)
        )
        merged_labels[merged_labels == drop_cluster] = keep_cluster

    return relabel_clusters(merged_labels)


def fallback_hierarchical_clustering(
    feature_vectors: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Perform hierarchical clustering as fallback when HDBSCAN fails.

    Args:
        feature_vectors: NxD feature vectors used for clustering
        n_clusters: Target number of clusters

    Returns:
        Array of cluster labels (0 to n_clusters-1)
    """
    n_samples = len(feature_vectors)

    # Safety check: n_clusters cannot exceed n_samples
    effective_n_clusters = min(n_clusters, n_samples)
    if effective_n_clusters < n_clusters:
        logger.warning(
            f"Requested {n_clusters} clusters but only {n_samples} samples, "
            f"using {effective_n_clusters} clusters"
        )

    # Edge case: only 1 sample
    if n_samples == 1:
        return np.array([0])

    logger.info(
        f"Using fallback hierarchical clustering with {effective_n_clusters} clusters"
    )

    clustering = AgglomerativeClustering(
        n_clusters=effective_n_clusters,
        metric="euclidean",
        linkage="ward",
    )
    labels = clustering.fit_predict(feature_vectors)

    return labels


class ClusterResult(BaseModel):
    """Result of clustering operation."""

    labels: list[int]  # Cluster label for each point (-1 = noise)
    n_clusters: int
    coords_3d: list[list[float]]  # 3D coordinates for visualization
    cluster_centers: dict[int, list[float]]  # Cluster ID -> center coordinates


class ClusteringService:
    """Service for semantic clustering with embedding-space clustering.

    This service follows a two-stage strategy:
    1. Cluster on normalized high-dimensional embeddings (semantic fidelity)
    2. Project to 3D with UMAP for visualization layout

    Compared with clustering on UMAP coordinates, this keeps related repositories
    together more reliably for diverse users and mixed technical domains.

    Usage:
        service = ClusteringService()
        result = service.fit_transform(embeddings)
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 5,
        min_samples: int = 1,
        cluster_selection_method: str = "eom",
        min_clusters: int = 3,
        target_min_clusters: int | None = None,
        target_max_clusters: int | None = None,
        assign_all_points: bool = True,
    ):
        """Initialize clustering service.

        Args:
            n_neighbors: UMAP parameter for local neighborhood size
            min_dist: UMAP parameter for minimum distance between points
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN min samples
            cluster_selection_method: HDBSCAN method
            min_clusters: Minimum expected number of clusters (fallback heuristic)
            target_min_clusters: Optional lower bound hint for fallback only
            target_max_clusters: Soft target upper bound via semantic merge
            assign_all_points: Whether to assign noise points to nearest clusters
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.min_clusters = min_clusters
        self.target_min_clusters = target_min_clusters
        self.target_max_clusters = target_max_clusters
        self.assign_all_points = assign_all_points

        self._umap_reducer = None
        self._clusterer = None

    def _init_umap(self):
        """Lazily initialize UMAP reducer."""
        if self._umap_reducer is None:
            import umap

            self._umap_reducer = umap.UMAP(
                n_components=3,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric="cosine",
                random_state=42,
            )

    def fit_transform(
        self,
        embeddings: list[list[float]],
        existing_coords: list[list[float]] | None = None,
        node_sizes: list[float] | None = None,
        resolve_overlap: bool = True,
    ) -> ClusterResult:
        """Cluster embeddings and generate 3D coordinates for visualization.

        Clustering is performed in normalized embedding space for semantic quality.
        UMAP is used only for coordinate generation.

        Args:
            embeddings: List of embedding vectors
            existing_coords: Optional existing 3D coordinates to preserve positions
            node_sizes: Optional node sizes for collision resolution (e.g., based on star count)
            resolve_overlap: Whether to run collision resolution algorithm

        Returns:
            ClusterResult with labels, coordinates, and cluster info
        """
        if not embeddings:
            return ClusterResult(
                labels=[],
                n_clusters=0,
                coords_3d=[],
                cluster_centers={},
            )

        embeddings_array = np.array(embeddings, dtype=np.float32)
        normalized_embeddings = normalize_embeddings(embeddings_array)
        n_samples = len(embeddings)

        logger.info(f"Clustering {n_samples} embeddings")

        # Adjust parameters for small datasets
        effective_n_neighbors = min(self.n_neighbors, n_samples - 1)
        effective_min_cluster_size = min(max(2, self.min_cluster_size), n_samples)
        effective_min_samples = min(
            max(1, self.min_samples), effective_min_cluster_size
        )

        # UMAP dimensionality reduction
        self._init_umap()
        self._umap_reducer.n_neighbors = effective_n_neighbors

        if existing_coords and len(existing_coords) == n_samples:
            # Use existing coordinates if available
            coords_3d = np.array(existing_coords)
            logger.info("Using existing 3D coordinates")
        else:
            coords_3d = self._umap_reducer.fit_transform(normalized_embeddings)
            logger.info("Generated new 3D coordinates via UMAP")

        # Clustering strategy in embedding space
        if n_samples < 3:
            # Too few samples for meaningful clustering
            labels = np.array([0] * n_samples)
            n_clusters = 1
            logger.info("Too few samples, assigning all to cluster 0")
        else:
            # Try HDBSCAN first in normalized embedding space
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_cluster_size,
                min_samples=effective_min_samples,
                metric="euclidean",
                cluster_selection_method=self.cluster_selection_method,
            )
            labels = clusterer.fit_predict(normalized_embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            logger.info(
                f"HDBSCAN found {n_clusters} clusters (noise points: {(labels == -1).sum()})"
            )

            # Check if we need fallback clustering
            expected_clusters = estimate_cluster_count(n_samples, self.min_clusters)
            if self.target_min_clusters is not None:
                expected_clusters = max(expected_clusters, self.target_min_clusters)
            if self.target_max_clusters is not None:
                expected_clusters = min(expected_clusters, self.target_max_clusters)

            if n_clusters < self.min_clusters and n_samples >= 5:
                logger.warning(
                    f"HDBSCAN produced only {n_clusters} clusters, "
                    f"falling back to hierarchical clustering with {expected_clusters} clusters"
                )
                labels = fallback_hierarchical_clustering(
                    normalized_embeddings, expected_clusters
                )
                n_clusters = len(set(labels))
            elif self.assign_all_points and (labels == -1).any():
                # Assign noise points to nearest clusters
                labels = assign_noise_to_nearest_cluster(
                    coords_3d, labels, embeddings_array
                )

        labels = np.array(labels, dtype=int)
        labels = relabel_clusters(labels)

        # Soft merge highly similar clusters to reduce fragmentation.
        labels = merge_similar_clusters(
            labels=labels,
            embeddings=normalized_embeddings,
            similarity_threshold=0.9,
            target_max_clusters=self.target_max_clusters,
            min_similarity_for_forced_merge=0.5,
        )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Guardrail: keep final cluster count within the configured lower bound.
        # merge_similar_clusters() is a soft optimization and may reduce too much.
        if (
            self.target_min_clusters is not None
            and n_clusters < self.target_min_clusters
            and n_samples >= self.target_min_clusters
        ):
            min_target = self.target_min_clusters
            if self.target_max_clusters is not None:
                min_target = min(min_target, self.target_max_clusters)
            min_target = min(max(2, min_target), n_samples)
            logger.warning(
                "Cluster merge reduced clusters below target_min_clusters; "
                f"rebuilding with hierarchical fallback to {min_target} clusters"
            )
            labels = fallback_hierarchical_clustering(normalized_embeddings, min_target)
            labels = relabel_clusters(np.array(labels, dtype=int))
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Fallback for pathological cases: no clusters after processing
        if n_clusters == 0 and n_samples >= 2:
            fallback_target = self.target_min_clusters or self.min_clusters or 2
            fallback_target = min(max(2, fallback_target), n_samples)
            logger.warning(
                "No valid clusters after processing; applying hierarchical fallback "
                f"with {fallback_target} clusters"
            )
            labels = fallback_hierarchical_clustering(
                normalized_embeddings, fallback_target
            )
            labels = relabel_clusters(np.array(labels, dtype=int))
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        logger.info(f"Final clustering: {n_clusters} clusters, all points assigned")

        # Scale UMAP output to give nodes more room before collision resolution.
        # UMAP with min_dist=0.1 packs points tightly; scaling by 2x gives
        # better baseline spacing while preserving relative structure.
        if n_samples > 1:
            coords_3d = coords_3d * 2.0

        # Resolve node collisions to prevent overlap
        if resolve_overlap and n_samples > 1:
            logger.info("Resolving node collisions...")
            coords_3d = resolve_collisions(
                coords=coords_3d,
                node_sizes=node_sizes,
                min_distance=0.35,
                iterations=100,
                repulsion_strength=0.5,
            )
            logger.info("Collision resolution complete")

        # Compute cluster centers (after collision resolution)
        cluster_centers = {}
        unique_labels = set(labels.tolist())
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            mask = labels == cluster_id
            center = coords_3d[mask].mean(axis=0)
            cluster_centers[int(cluster_id)] = center.tolist()

        return ClusterResult(
            labels=labels.tolist(),
            n_clusters=n_clusters,
            coords_3d=coords_3d.tolist(),
            cluster_centers=cluster_centers,
        )


TOPIC_SYNONYMS = {
    "agent-memory": "agent-memory",
    "agent_memory": "agent-memory",
    "long-term-memory": "agent-memory",
    "longterm-memory": "agent-memory",
    "memory-augmented": "agent-memory",
    "mem0": "agent-memory",
    "rag-memory": "agent-memory",
    "llm-training": "llm-training",
    "distributed-training": "distributed-training",
    "distributed-training-framework": "distributed-training",
    "distributed-systems": "distributed-training",
    "deepspeed": "distributed-training",
    "fsdp": "distributed-training",
    "megatron": "distributed-training",
}


def normalize_topic_token(token: str) -> str:
    """Normalize one topic/token to improve semantic consistency."""
    normalized = token.strip().lower().replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return TOPIC_SYNONYMS.get(normalized, normalized)


def normalize_topic_lists(topic_lists: list[list[str]]) -> list[list[str]]:
    """Normalize topic lists with stable order and uniqueness per repository."""
    normalized_lists: list[list[str]] = []
    for topics in topic_lists:
        seen: set[str] = set()
        normalized_topics: list[str] = []
        for topic in topics or []:
            if not topic:
                continue
            normalized = normalize_topic_token(topic)
            if normalized not in seen:
                seen.add(normalized)
                normalized_topics.append(normalized)
        normalized_lists.append(normalized_topics)
    return normalized_lists


def sanitize_cluster_name(name: str | None) -> str:
    """Normalize cluster display names to avoid noisy duplicates."""
    if not name:
        return "未分类"

    cleaned = " ".join(name.strip().split())
    cleaned = cleaned.replace("工具集", "工具")
    cleaned = cleaned.replace("集合", "集")
    cleaned = re.sub(r"\s+[&+/,，、]\s+", " / ", cleaned)
    cleaned = cleaned.strip("-_/ ")
    return cleaned or "未分类"


def deduplicate_cluster_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Globally de-duplicate cluster names while keeping meaning stable."""
    if not entries:
        return entries

    for entry in entries:
        entry["name"] = sanitize_cluster_name(entry.get("name"))

    used_names: set[str] = set()
    for entry in sorted(
        entries, key=lambda item: item.get("repo_count", 0), reverse=True
    ):
        base_name = entry["name"]
        if base_name not in used_names:
            used_names.add(base_name)
            continue

        keywords = [k for k in (entry.get("keywords") or []) if k]
        suffix_source = keywords[0] if keywords else f"{entry.get('repo_count', 0)}"
        suffix = suffix_source.replace(" ", "")[:12] if suffix_source else "variant"

        candidate = f"{base_name} · {suffix}"
        seq = 2
        while candidate in used_names:
            candidate = f"{base_name} · {suffix}{seq}"
            seq += 1

        entry["name"] = candidate
        used_names.add(candidate)

    return entries


def build_cluster_naming_inputs(
    cluster_repos: list[Any],
) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """Build normalized and stable inputs for cluster naming."""
    if not cluster_repos:
        return [], [], [], []

    sorted_repos = sorted(
        cluster_repos,
        key=lambda repo: (
            -(repo.stargazers_count or 0),
            repo.full_name or "",
        ),
    )

    repo_names = [repo.full_name for repo in sorted_repos]
    descriptions = [repo.description or "" for repo in sorted_repos]
    topics = normalize_topic_lists([repo.topics or [] for repo in sorted_repos])
    languages = [repo.language or "" for repo in sorted_repos]
    return repo_names, descriptions, topics, languages


def generate_cluster_name(
    repo_names: list[str],
    descriptions: list[str],
    topics: list[list[str]],
) -> tuple[str, str, list[str]]:
    """Generate cluster name, description, and keywords using heuristics.

    This is a simple rule-based approach. For better results,
    use LLM-based summarization.

    Args:
        repo_names: Names of repos in cluster
        descriptions: Descriptions of repos
        topics: Topics/tags of repos

    Returns:
        Tuple of (name, description, keywords)
    """
    normalized_topics = normalize_topic_lists(topics)

    all_topics = []
    for topic_list in normalized_topics:
        if topic_list:
            all_topics.extend(topic_list)

    topic_counts = Counter(all_topics)
    top_topics = [t for t, _ in topic_counts.most_common(5)]

    # Extract common words from descriptions
    if descriptions:
        words = []
        for desc in descriptions:
            if desc:
                words.extend(desc.lower().split())

        # Filter common words
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "for",
            "to",
            "and",
            "or",
            "of",
            "in",
            "on",
            "with",
        }
        words = [w for w in words if w not in stop_words and len(w) > 2]
        word_counts = Counter(words)
        common_words = [w for w, _ in word_counts.most_common(3)]
    else:
        common_words = []

    # Generate name
    if top_topics:
        name = " / ".join(top_topics[:2]).title()
    elif common_words:
        name = " ".join(common_words[:2]).title()
    else:
        name = f"Cluster ({len(repo_names)} repos)"

    name = sanitize_cluster_name(name)

    # Generate description
    description = f"A cluster of {len(repo_names)} repositories"
    if top_topics:
        description += f" related to {', '.join(top_topics[:3])}"

    # Keywords
    keywords: list[str] = []
    for keyword in top_topics[:5] + common_words[:3]:
        if keyword and keyword not in keywords:
            keywords.append(keyword)

    return name, description, keywords


async def generate_cluster_name_llm(
    repo_names: list[str],
    descriptions: list[str],
    topics: list[list[str]],
    languages: list[str] | None = None,
    existing_cluster_names: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Generate cluster name, description, and keywords using LLM.

    This provides better results than the heuristic approach by using
    LLM to understand the semantic relationship between repos.

    Args:
        repo_names: Names of repos in cluster
        descriptions: Descriptions of repos
        topics: Topics/tags of repos
        languages: Primary languages of repos (optional)
        existing_cluster_names: Names already assigned to other clusters,
            used to ensure uniqueness.

    Returns:
        Tuple of (name, description, keywords)
    """
    from nebula.core.llm import get_llm_service

    llm_service = get_llm_service()

    return await llm_service.generate_cluster_info(
        repo_names=repo_names,
        descriptions=descriptions,
        topics=topics,
        languages=languages or [],
        existing_cluster_names=existing_cluster_names,
    )


class IncrementalAssignResult(BaseModel):
    """Result of incremental assignment for new repos."""

    new_coords: list[list[float]]  # 3D coordinates for each new repo
    new_labels: list[int]  # Cluster label for each new repo


def assign_new_repos_incrementally(
    existing_embeddings: np.ndarray,
    existing_coords: np.ndarray,
    existing_labels: np.ndarray,
    new_embeddings: np.ndarray,
    new_node_sizes: list[float] | None = None,
    k_neighbors: int = 5,
    noise_scale: float = 0.15,
) -> IncrementalAssignResult:
    """Assign new repos to existing clusters and interpolate their 3D coordinates.

    Instead of re-running UMAP and re-clustering everything, this function:
    1. Finds K nearest neighbors of each new repo in embedding space
    2. Interpolates 3D coordinates from those neighbors (weighted by similarity)
    3. Assigns the new repo to the cluster of its nearest neighbor

    This preserves the existing graph layout and only adds new nodes.

    Args:
        existing_embeddings: NxD normalized embeddings for existing repos
        existing_coords: Nx3 coordinates of existing repos
        existing_labels: N cluster labels for existing repos
        new_embeddings: MxD normalized embeddings for new repos
        new_node_sizes: Optional sizes for new nodes (for collision resolution)
        k_neighbors: Number of neighbors for interpolation
        noise_scale: Scale of random offset to avoid exact overlap with neighbors

    Returns:
        IncrementalAssignResult with coordinates and labels for new repos
    """
    if len(new_embeddings) == 0:
        return IncrementalAssignResult(new_coords=[], new_labels=[])

    n_existing = len(existing_embeddings)
    n_new = len(new_embeddings)

    logger.info(
        f"Incrementally assigning {n_new} new repos using {n_existing} existing repos"
    )

    # Effective k: can't use more neighbors than existing points
    effective_k = min(k_neighbors, n_existing)

    # Find nearest existing neighbors for each new embedding
    nn = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    nn.fit(existing_embeddings)
    distances, indices = nn.kneighbors(new_embeddings)

    new_coords = np.zeros((n_new, 3), dtype=np.float64)
    new_labels = np.zeros(n_new, dtype=int)

    rng = np.random.RandomState(42)

    for i in range(n_new):
        neighbor_indices = indices[i]
        neighbor_distances = distances[i]

        # Convert cosine distances to similarity weights (cosine dist = 1 - similarity)
        similarities = np.clip(1.0 - neighbor_distances, 0.01, 1.0)
        weights = similarities / similarities.sum()

        # Weighted average of neighbor coordinates
        neighbor_coords = existing_coords[neighbor_indices]
        interpolated = np.average(neighbor_coords, axis=0, weights=weights)

        # Add small random offset to avoid exact overlap
        offset = rng.randn(3) * noise_scale
        new_coords[i] = interpolated + offset

        # Assign cluster label from nearest neighbor (highest weight)
        new_labels[i] = int(existing_labels[neighbor_indices[0]])

    # Run collision resolution between new nodes and all existing+new nodes
    all_coords = np.vstack([existing_coords, new_coords])
    all_sizes = list(
        [1.0] * n_existing + (new_node_sizes if new_node_sizes else [1.0] * n_new)
    )

    resolved = resolve_collisions(
        coords=all_coords,
        node_sizes=all_sizes,
        min_distance=0.35,
        iterations=60,
        repulsion_strength=0.5,
    )

    # Extract only the new coordinates (existing ones stay put mostly,
    # but we only return new ones to minimize disruption)
    new_coords_resolved = resolved[n_existing:]

    logger.info(f"Incremental assignment complete: {n_new} repos assigned")

    return IncrementalAssignResult(
        new_coords=new_coords_resolved.tolist(),
        new_labels=new_labels.tolist(),
    )


# Global service instance
_clustering_service: ClusteringService | None = None


def get_clustering_service() -> ClusteringService:
    """Get global clustering service instance."""
    global _clustering_service
    if _clustering_service is None:
        # Stable defaults for mixed user datasets.
        # - Clustering runs in embedding space (not UMAP coordinates)
        # - target_max_clusters is treated as a soft cap via post-merge
        _clustering_service = ClusteringService(
            n_neighbors=40,
            min_cluster_size=20,
            min_samples=8,
            cluster_selection_method="eom",
            min_clusters=3,
            target_min_clusters=None,
            target_max_clusters=8,
        )
    return _clustering_service

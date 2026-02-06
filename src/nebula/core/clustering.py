"""Clustering service using UMAP + HDBSCAN.

This module provides semantic clustering of repositories based on their embeddings.
Includes collision resolution to prevent node overlap in 3D visualization.

Key features:
- Adaptive clustering with automatic parameter tuning
- Noise point assignment to ensure all nodes have a cluster
- Fallback to hierarchical clustering when HDBSCAN produces too few clusters
"""

import math

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from nebula.utils import get_logger

logger = get_logger(__name__)


def resolve_collisions(
    coords: np.ndarray,
    node_sizes: list[float] | None = None,
    min_distance: float = 0.15,
    iterations: int = 50,
    repulsion_strength: float = 0.1,
) -> np.ndarray:
    """Resolve node collisions in 3D space using force-directed repulsion.

    This algorithm iteratively pushes overlapping nodes apart while trying
    to preserve the overall cluster structure from UMAP.

    Args:
        coords: Nx3 array of 3D coordinates
        node_sizes: Optional list of node sizes (used for distance calculation)
        min_distance: Minimum distance between node centers
        iterations: Number of iterations for collision resolution
        repulsion_strength: How strongly nodes repel each other

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

    for iteration in range(iterations):
        moved = False
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
                if dist < required_dist and dist > 0.001:
                    # Calculate repulsion direction
                    direction = diff / dist
                    overlap = required_dist - dist

                    # Apply displacement (split evenly between both nodes)
                    displacement = direction * overlap * repulsion_strength
                    displacements[i] += displacement
                    displacements[j] -= displacement
                    moved = True
                elif dist <= 0.001:
                    # Nodes are at same position, add random displacement
                    random_dir = np.random.randn(3)
                    random_dir /= np.linalg.norm(random_dir) + 0.001
                    displacements[i] += random_dir * min_distance * repulsion_strength
                    displacements[j] -= random_dir * min_distance * repulsion_strength
                    moved = True

        # Apply displacements
        result += displacements

        # Early exit if no collisions were found
        if not moved:
            logger.debug(f"Collision resolution converged at iteration {iteration}")
            break

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


def fallback_hierarchical_clustering(
    coords: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Perform hierarchical clustering as fallback when HDBSCAN fails.

    Args:
        coords: Nx3 array of 3D coordinates
        n_clusters: Target number of clusters

    Returns:
        Array of cluster labels (0 to n_clusters-1)
    """
    n_samples = len(coords)

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
    labels = clustering.fit_predict(coords)

    return labels


class ClusterResult(BaseModel):
    """Result of clustering operation."""

    labels: list[int]  # Cluster label for each point (-1 = noise)
    n_clusters: int
    coords_3d: list[list[float]]  # 3D coordinates for visualization
    cluster_centers: dict[int, list[float]]  # Cluster ID -> center coordinates


class ClusteringService:
    """Service for clustering embeddings using UMAP + HDBSCAN.

    This service reduces high-dimensional embeddings to 3D for visualization
    and groups similar items into clusters.

    Key improvements over basic HDBSCAN:
    - Uses 'leaf' cluster selection for more fine-grained clusters
    - Assigns noise points to nearest clusters (no orphaned nodes)
    - Falls back to hierarchical clustering if too few clusters are found

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
            min_cluster_size: HDBSCAN minimum cluster size (lowered for more clusters)
            min_samples: HDBSCAN min samples (lowered for more sensitivity)
            cluster_selection_method: HDBSCAN method - 'leaf' for fine-grained, 'eom' for larger
            min_clusters: Minimum expected number of clusters (triggers fallback if not met)
            target_min_clusters: If set, ensure final cluster count is at least this value
            target_max_clusters: If set, ensure final cluster count is at most this value
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
        """Reduce dimensions and cluster embeddings.

        This method uses an adaptive clustering strategy:
        1. First tries HDBSCAN with 'leaf' mode for fine-grained clusters
        2. Assigns noise points to nearest clusters if assign_all_points=True
        3. Falls back to hierarchical clustering if too few clusters are found

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

        embeddings_array = np.array(embeddings)
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
            coords_3d = self._umap_reducer.fit_transform(embeddings_array)
            logger.info("Generated new 3D coordinates via UMAP")

        # Clustering strategy
        if n_samples < 3:
            # Too few samples for meaningful clustering
            labels = np.array([0] * n_samples)
            n_clusters = 1
            logger.info("Too few samples, assigning all to cluster 0")
        else:
            # Try HDBSCAN first with optimized parameters
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_cluster_size,
                min_samples=effective_min_samples,
                metric="euclidean",
                cluster_selection_method=self.cluster_selection_method,
            )
            labels = clusterer.fit_predict(coords_3d)
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
                labels = fallback_hierarchical_clustering(coords_3d, expected_clusters)
                n_clusters = len(set(labels))
            elif self.assign_all_points and (labels == -1).any():
                # Assign noise points to nearest clusters
                labels = assign_noise_to_nearest_cluster(
                    coords_3d, labels, embeddings_array
                )

        # Ensure labels is a numpy array for further processing
        labels = np.array(labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Enforce a target cluster count range (optional).
        # This provides stable UX: user can ask for "no more than N clusters"
        # without manually tuning HDBSCAN/UMAP parameters.
        if (
            self.target_max_clusters is not None
            and n_samples >= max(2, self.target_max_clusters)
            and n_clusters > self.target_max_clusters
        ):
            logger.info(
                f"Cluster count {n_clusters} exceeds target_max_clusters={self.target_max_clusters}; "
                "merging via hierarchical clustering"
            )
            labels = fallback_hierarchical_clustering(
                coords_3d, self.target_max_clusters
            )
            labels = np.array(labels)
            n_clusters = len(set(labels))

        if (
            self.target_min_clusters is not None
            and n_samples >= max(2, self.target_min_clusters)
            and n_clusters < self.target_min_clusters
        ):
            logger.info(
                f"Cluster count {n_clusters} is below target_min_clusters={self.target_min_clusters}; "
                "splitting via hierarchical clustering"
            )
            labels = fallback_hierarchical_clustering(
                coords_3d, self.target_min_clusters
            )
            labels = np.array(labels)
            n_clusters = len(set(labels))

        logger.info(f"Final clustering: {n_clusters} clusters, all points assigned")

        # Resolve node collisions to prevent overlap
        if resolve_overlap and n_samples > 1:
            logger.info("Resolving node collisions...")
            coords_3d = resolve_collisions(
                coords=coords_3d,
                node_sizes=node_sizes,
                min_distance=0.15,
                iterations=50,
                repulsion_strength=0.1,
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
    from collections import Counter

    # Flatten and count topics
    all_topics = []
    for topic_list in topics:
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
        name = " & ".join(top_topics[:2]).title()
    elif common_words:
        name = " ".join(common_words[:2]).title()
    else:
        name = f"Cluster ({len(repo_names)} repos)"

    # Generate description
    description = f"A cluster of {len(repo_names)} repositories"
    if top_topics:
        description += f" related to {', '.join(top_topics[:3])}"

    # Keywords
    keywords = list(set(top_topics[:5] + common_words[:3]))

    return name, description, keywords


async def generate_cluster_name_llm(
    repo_names: list[str],
    descriptions: list[str],
    topics: list[list[str]],
    languages: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Generate cluster name, description, and keywords using LLM.

    This provides better results than the heuristic approach by using
    LLM to understand the semantic relationship between repos.

    Args:
        repo_names: Names of repos in cluster
        descriptions: Descriptions of repos
        topics: Topics/tags of repos
        languages: Primary languages of repos (optional)

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
    )


# Global service instance
_clustering_service: ClusteringService | None = None


def get_clustering_service() -> ClusteringService:
    """Get global clustering service instance."""
    global _clustering_service
    if _clustering_service is None:
        # Fewer, more stable clusters by default (aim for 4-8 clusters).
        # - Larger min_cluster_size/min_samples => less fragmentation
        # - Larger n_neighbors => more global structure in UMAP
        # - Target range provides consistent UX without manual tuning
        _clustering_service = ClusteringService(
            n_neighbors=40,
            min_cluster_size=20,
            min_samples=8,
            cluster_selection_method="eom",
            min_clusters=4,
            target_min_clusters=4,
            target_max_clusters=8,
        )
    return _clustering_service

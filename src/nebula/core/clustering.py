"""Clustering service using UMAP + HDBSCAN.

This module provides semantic clustering of repositories based on their embeddings.
Includes collision resolution to prevent node overlap in 3D visualization.
"""

import numpy as np
from pydantic import BaseModel

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

    Usage:
        service = ClusteringService()
        result = service.fit_transform(embeddings)
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
    ):
        """Initialize clustering service.

        Args:
            n_neighbors: UMAP parameter for local neighborhood size
            min_dist: UMAP parameter for minimum distance between points
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN min samples (defaults to min_cluster_size)
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size

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

    def _init_hdbscan(self):
        """Lazily initialize HDBSCAN clusterer."""
        if self._clusterer is None:
            import hdbscan

            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )

    def fit_transform(
        self,
        embeddings: list[list[float]],
        existing_coords: list[list[float]] | None = None,
        node_sizes: list[float] | None = None,
        resolve_overlap: bool = True,
    ) -> ClusterResult:
        """Reduce dimensions and cluster embeddings.

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
        effective_min_cluster_size = min(self.min_cluster_size, max(2, n_samples // 5))

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

        # HDBSCAN clustering on 3D coordinates
        if n_samples < 5:
            # Too few samples for meaningful clustering
            labels = [0] * n_samples
            n_clusters = 1
        else:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_cluster_size,
                min_samples=min(self.min_samples, effective_min_cluster_size),
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(coords_3d).tolist()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        logger.info(f"Found {n_clusters} clusters")

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
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            mask = np.array(labels) == cluster_id
            center = coords_3d[mask].mean(axis=0)
            cluster_centers[cluster_id] = center.tolist()

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            coords_3d=coords_3d.tolist(),
            cluster_centers=cluster_centers,
        )

    def transform_new(
        self,
        new_embeddings: list[list[float]],
        reference_embeddings: list[list[float]],
        reference_coords: list[list[float]],
    ) -> list[list[float]]:
        """Transform new embeddings to 3D space using existing reference.

        Args:
            new_embeddings: New embedding vectors to transform
            reference_embeddings: Existing embeddings for fitting
            reference_coords: Existing 3D coordinates

        Returns:
            3D coordinates for new embeddings
        """
        if not new_embeddings:
            return []

        self._init_umap()

        # Fit on reference data
        ref_array = np.array(reference_embeddings)
        self._umap_reducer.fit(ref_array)

        # Transform new data
        new_array = np.array(new_embeddings)
        new_coords = self._umap_reducer.transform(new_array)

        return new_coords.tolist()


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
        _clustering_service = ClusteringService()
    return _clustering_service

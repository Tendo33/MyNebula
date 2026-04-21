import type { ClusterInfo, GraphData, GraphEdge, GraphNode, TimelineData } from '../types';
import { asRepoSearchCandidate, buildRepoSearchText, matchesRepoSearch, normalizeSearchQuery } from '../utils/search';

export interface GraphFiltersInput {
  selectedClusters: Set<number>;
  selectedStarLists: Set<number>;
  searchQuery: string;
  timeRange: [number, number] | null;
  minStars: number;
  languages: Set<string>;
}

export interface GraphFilterIndexes {
  nodeSearchText: Map<number, string>;
  edgeIndexesByNodeId: Map<number, number[]>;
  totalNodeCount: number;
}

export const buildGraphFilterIndexes = (rawData: GraphData | null): GraphFilterIndexes => {
  const nodeSearchText = new Map<number, string>();
  const edgeIndexesByNodeId = new Map<number, number[]>();
  if (!rawData) {
    return { nodeSearchText, edgeIndexesByNodeId, totalNodeCount: 0 };
  }

  rawData.nodes.forEach((node) => {
    nodeSearchText.set(node.id, buildRepoSearchText(asRepoSearchCandidate(node)));
  });

  rawData.edges.forEach((edge, edgeIndex) => {
    const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
    const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
    const sourceIndexes = edgeIndexesByNodeId.get(sourceId) ?? [];
    sourceIndexes.push(edgeIndex);
    edgeIndexesByNodeId.set(sourceId, sourceIndexes);

    if (targetId !== sourceId) {
      const targetIndexes = edgeIndexesByNodeId.get(targetId) ?? [];
      targetIndexes.push(edgeIndex);
      edgeIndexesByNodeId.set(targetId, targetIndexes);
    }
  });

  return { nodeSearchText, edgeIndexesByNodeId, totalNodeCount: rawData.nodes.length };
};

export const filterVisibleNodes = ({
  rawData,
  timelineData,
  filters,
  indexes,
}: {
  rawData: GraphData | null;
  timelineData: TimelineData | null;
  filters: GraphFiltersInput;
  indexes: GraphFilterIndexes;
}): GraphNode[] => {
  if (!rawData) {
    return [];
  }

  const normalizedSearchQuery = normalizeSearchQuery(filters.searchQuery);

  return rawData.nodes.filter((node) => {
    if (
      filters.selectedClusters.size > 0 &&
      (node.cluster_id == null || !filters.selectedClusters.has(node.cluster_id))
    ) {
      return false;
    }

    if (
      filters.selectedStarLists.size > 0 &&
      (node.star_list_id == null || !filters.selectedStarLists.has(node.star_list_id))
    ) {
      return false;
    }

    if (
      normalizedSearchQuery &&
      !matchesRepoSearch(
        asRepoSearchCandidate(node),
        normalizedSearchQuery,
        indexes.nodeSearchText.get(node.id)
      )
    ) {
      return false;
    }

    if (filters.minStars > 0 && node.stargazers_count < filters.minStars) {
      return false;
    }

    if (filters.languages.size > 0 && (!node.language || !filters.languages.has(node.language))) {
      return false;
    }

    if (filters.timeRange && timelineData && timelineData.points.length > 0) {
      const [startIdx, endIdx] = filters.timeRange;
      const startDate = timelineData.points[startIdx]?.date;
      const endDate = timelineData.points[endIdx]?.date;
      if (startDate && endDate && node.starred_at) {
        const nodeDate = node.starred_at.substring(0, 7);
        if (nodeDate < startDate || nodeDate > endDate) {
          return false;
        }
      }
    }

    return true;
  });
};

export const createVisibleNodeIds = (nodes: GraphNode[]): Set<number> =>
  new Set(nodes.map((node) => node.id));

export const filterVisibleEdges = (
  edges: GraphEdge[],
  visibleNodeIds: Set<number>,
  indexes?: GraphFilterIndexes
): GraphEdge[] => {
  if (!indexes || visibleNodeIds.size >= indexes.totalNodeCount / 2) {
    return edges.filter((edge) => {
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });
  }

  const matchedEdgeIndexes = new Set<number>();
  visibleNodeIds.forEach((nodeId) => {
    const connectedEdgeIndexes = indexes.edgeIndexesByNodeId.get(nodeId) ?? [];
    connectedEdgeIndexes.forEach((edgeIndex) => {
      const edge = edges[edgeIndex];
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      if (visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId)) {
        matchedEdgeIndexes.add(edgeIndex);
      }
    });
  });

  return Array.from(matchedEdgeIndexes)
    .sort((left, right) => left - right)
    .map((edgeIndex) => edges[edgeIndex]);
};

export const filterVisibleClusters = (
  clusters: ClusterInfo[],
  visibleNodes: GraphNode[]
): ClusterInfo[] => {
  const visibleClusterIds = new Set(
    visibleNodes
      .map((node) => node.cluster_id)
      .filter((clusterId): clusterId is number => clusterId != null)
  );
  return clusters.filter((cluster) => visibleClusterIds.has(cluster.id));
};

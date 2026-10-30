import type { ClusterInfo, GraphEdge, GraphNode, TimelinePoint } from './types';
import { matchesRepoSearch, normalizeSearchQuery } from './search';

export interface Filters {
  clusters: Set<number>;
  lists: Set<number>;
  query: string;
  /** Inclusive indexes into the timeline points. */
  timeRange: [number, number] | null;
  languages: Set<string>;
}

export const emptyFilters = (): Filters => ({
  clusters: new Set(),
  lists: new Set(),
  query: '',
  timeRange: null,
  languages: new Set(),
});

export const hasActiveFilters = (filters: Filters): boolean =>
  filters.clusters.size > 0 ||
  filters.lists.size > 0 ||
  filters.query.trim() !== '' ||
  filters.timeRange !== null ||
  filters.languages.size > 0;

export const filterNodes = (
  nodes: GraphNode[],
  timeline: TimelinePoint[],
  filters: Filters,
  searchText: Map<number, string>,
): GraphNode[] => {
  const query = normalizeSearchQuery(filters.query);
  const start = filters.timeRange ? timeline[filters.timeRange[0]]?.date : undefined;
  const end = filters.timeRange ? timeline[filters.timeRange[1]]?.date : undefined;

  return nodes.filter((node) => {
    if (filters.clusters.size && (node.cluster_id == null || !filters.clusters.has(node.cluster_id))) {
      return false;
    }
    if (filters.lists.size && (node.star_list_id == null || !filters.lists.has(node.star_list_id))) {
      return false;
    }
    if (query && !matchesRepoSearch(node, query, searchText.get(node.id))) return false;
    if (filters.languages.size && (!node.language || !filters.languages.has(node.language))) {
      return false;
    }
    if (start && end && node.starred_at) {
      const month = node.starred_at.slice(0, 7);
      if (month < start || month > end) return false;
    }
    return true;
  });
};

/** Undirected node -> neighbour ids. */
export const buildAdjacency = (edges: GraphEdge[]): Map<number, Set<number>> => {
  const index = new Map<number, Set<number>>();
  for (const [source, target] of edges) {
    if (!index.has(source)) index.set(source, new Set());
    if (!index.has(target)) index.set(target, new Set());
    index.get(source)!.add(target);
    index.get(target)!.add(source);
  }
  return index;
};

export const visibleClusters = (clusters: ClusterInfo[], nodes: GraphNode[]): ClusterInfo[] => {
  const ids = new Set(nodes.map((node) => node.cluster_id).filter((id) => id != null));
  return clusters.filter((cluster) => ids.has(cluster.id));
};

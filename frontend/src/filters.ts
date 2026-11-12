import type { ClusterInfo, GraphEdge, GraphNode, TimelinePoint } from './types';
import { matchesRepoSearch, normalizeSearchQuery } from './search';

export interface Filters {
  clusters: Set<number>;
  lists: Set<string>;
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
    if (filters.lists.size && !node.star_list_ids.some((id) => filters.lists.has(id))) {
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

/**
 * Nodes for one rail's counts. That rail's own selection is ignored, and so is
 * the search box: the rail is not rebuilt while the query changes.
 */
export const facetNodes = (
  nodes: GraphNode[],
  timeline: TimelinePoint[],
  filters: Filters,
  except: 'clusters' | 'lists' | 'languages',
): GraphNode[] => {
  const base = { ...filters, query: '' };
  if (except === 'clusters') return filterNodes(nodes, timeline, { ...base, clusters: new Set() }, new Map());
  if (except === 'lists') return filterNodes(nodes, timeline, { ...base, lists: new Set() }, new Map());
  return filterNodes(nodes, timeline, { ...base, languages: new Set() }, new Map());
};

/** Per-month counts for the timeline. The selected window and the search box are ignored. */
export const timelineFacets = (
  nodes: GraphNode[],
  timeline: TimelinePoint[],
  filters: Filters,
): { count: number; topLanguages: string[] }[] => {
  const scoped = filterNodes(nodes, timeline, { ...filters, query: '', timeRange: null }, new Map());
  const counts = timeline.map(() => 0);
  const languages = timeline.map(() => new Map<string, number>());
  const index = new Map(timeline.map((point, i) => [point.date, i]));
  for (const node of scoped) {
    const month = node.starred_at?.slice(0, 7);
    if (!month) continue;
    const at = index.get(month);
    if (at == null) continue;
    counts[at] += 1;
    if (!node.language) continue;
    languages[at].set(node.language, (languages[at].get(node.language) ?? 0) + 1);
  }
  return counts.map((count, i) => ({
    count,
    topLanguages: [...languages[i].entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 3)
      .map(([language]) => language),
  }));
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

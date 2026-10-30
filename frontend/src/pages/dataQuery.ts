import { matchesRepoSearch } from '../search';
import type { GraphNode } from '../types';

export type SortField = 'name' | 'language' | 'stargazers_count' | 'starred_at' | 'cluster' | 'summary' | 'last_commit_time';
export interface SortConfig {
  field: SortField;
  direction: 'asc' | 'desc';
}

export interface DataQuery {
  query: string;
  clusters: Set<number>;
  month: string | null;
  topic: string | null;
}

/** Same filters the old `/api/v2/data/repos` endpoint applied, now in memory. */
export const filterRepos = (nodes: GraphNode[], q: DataQuery): GraphNode[] => {
  const topic = q.topic?.trim().toLowerCase();
  return nodes.filter(
    (node) =>
      (!q.clusters.size || (node.cluster_id != null && q.clusters.has(node.cluster_id))) &&
      matchesRepoSearch(node, q.query) &&
      (!q.month || node.starred_at?.startsWith(q.month)) &&
      (!topic || (node.topics ?? []).some((value) => value.trim().toLowerCase() === topic)),
  );
};

const SORT_KEYS: Record<SortField, (node: GraphNode) => string | number | null | undefined> = {
  name: (n) => n.name.toLowerCase(),
  language: (n) => n.language,
  stargazers_count: (n) => n.stargazers_count,
  starred_at: (n) => n.starred_at,
  cluster: (n) => n.cluster_id,
  summary: (n) => n.ai_summary,
  last_commit_time: (n) => n.last_commit_time,
};

/** Stable sort with missing values last, ties broken by id like the old API. */
export const sortRepos = (nodes: GraphNode[], sort: SortConfig): GraphNode[] => {
  const key = SORT_KEYS[sort.field];
  const dir = sort.direction === 'asc' ? 1 : -1;
  return [...nodes].sort((a, b) => {
    const va = key(a);
    const vb = key(b);
    if (va == null || vb == null) return va == null && vb == null ? dir * (a.id - b.id) : va == null ? 1 : -1;
    if (va < vb) return -dir;
    if (va > vb) return dir;
    return dir * (a.id - b.id);
  });
};

export const nextSort = (prev: SortConfig, field: SortField): SortConfig => ({
  field,
  direction: prev.field === field && prev.direction === 'asc' ? 'desc' : 'asc',
});


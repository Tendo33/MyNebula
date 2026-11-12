import { matchesRepoSearch } from '../search';
import type { GraphNode } from '../types';

export type SortField = 'name' | 'language' | 'stargazers_count' | 'starred_at' | 'cluster' | 'summary' | 'last_commit_time';
export interface SortConfig {
  field: SortField;
  direction: 'asc' | 'desc';
}

export const DEFAULT_SORT: SortConfig = { field: 'starred_at', direction: 'desc' };
export const DATA_PAGE_SIZES = [25, 50, 100] as const;

const SORT_FIELDS = new Set<string>([
  'name',
  'language',
  'stargazers_count',
  'starred_at',
  'cluster',
  'summary',
  'last_commit_time',
]);

/** Default sort, page, and page size stay out of the link. */
export const readSort = (params: URLSearchParams): SortConfig => {
  const raw = params.get('sort');
  const field = raw && SORT_FIELDS.has(raw) ? (raw as SortField) : DEFAULT_SORT.field;
  const dir = params.get('dir');
  const direction = dir === 'asc' || dir === 'desc' ? dir : DEFAULT_SORT.direction;
  return { field, direction };
};

export const writeSort = (params: URLSearchParams, sort: SortConfig): void => {
  if (sort.field === DEFAULT_SORT.field && sort.direction === DEFAULT_SORT.direction) return;
  params.set('sort', sort.field);
  params.set('dir', sort.direction);
};

export const readPage = (params: URLSearchParams): number => {
  const page = Number.parseInt(params.get('page') ?? '', 10);
  return Number.isFinite(page) && page > 0 ? page : 1;
};

export const readPageSize = (params: URLSearchParams): number => {
  const size = Number.parseInt(params.get('size') ?? '', 10);
  for (const allowed of DATA_PAGE_SIZES) if (allowed === size) return allowed;
  return DATA_PAGE_SIZES[0];
};

/** Page that contains `id` in an already filtered and sorted id list. Null when it is absent. */
export const pageContaining = (ids: readonly number[], id: number, pageSize: number): number | null => {
  if (pageSize < 1) return null;
  const index = ids.indexOf(id);
  return index < 0 ? null : Math.floor(index / pageSize) + 1;
};

const MONTH = /^\d{4}-\d{2}$/;

/** `YYYY-MM` from a starred-at timestamp, or null when the value has no month. */
export const starMonth = (value: string | null | undefined): string | null =>
  value && MONTH.test(value.slice(0, 7)) ? value.slice(0, 7) : null;

export const readMonth = (params: URLSearchParams): string | null => {
  const month = params.get('month');
  return month && MONTH.test(month) ? month : null;
};

/** One language stays `language=`; several use `languages=`, matching the graph. */
export const readLanguages = (params: URLSearchParams): string[] =>
  (params.get('languages') ?? params.get('language') ?? '')
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

export const writeLanguages = (params: URLSearchParams, languages: Iterable<string>): void => {
  const list = [...new Set(languages)].sort();
  if (list.length === 1) params.set('language', list[0]);
  else if (list.length > 1) params.set('languages', list.join(','));
};

export interface DataQuery {
  query: string;
  clusters: Set<number>;
  languages: Set<string>;
  month: string | null;
  topic: string | null;
}

/** Same filters the old `/api/v2/data/repos` endpoint applied, now in memory. */
export const filterRepos = (nodes: GraphNode[], q: DataQuery): GraphNode[] => {
  const topic = q.topic?.trim().toLowerCase();
  return nodes.filter(
    (node) =>
      (!q.clusters.size || (node.cluster_id != null && q.clusters.has(node.cluster_id))) &&
      (!q.languages.size || (node.language != null && q.languages.has(node.language))) &&
      matchesRepoSearch(node, q.query) &&
      (!q.month || node.starred_at?.startsWith(q.month)) &&
      (!topic || (node.topics ?? []).some((value) => value.trim().toLowerCase() === topic)),
  );
};

/** Cluster chip counts. The cluster selection itself is not applied, so each chip shows what it would add. */
export const clusterFacetCounts = (nodes: GraphNode[], q: DataQuery): Map<number, number> => {
  const counts = new Map<number, number>();
  for (const node of filterRepos(nodes, { ...q, clusters: new Set() })) {
    if (node.cluster_id == null) continue;
    counts.set(node.cluster_id, (counts.get(node.cluster_id) ?? 0) + 1);
  }
  return counts;
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


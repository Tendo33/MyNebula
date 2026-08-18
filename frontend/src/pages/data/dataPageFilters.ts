export type SortField =
  | 'name'
  | 'language'
  | 'stargazers_count'
  | 'starred_at'
  | 'cluster'
  | 'summary'
  | 'last_commit_time';

export type SortDirection = 'asc' | 'desc';

export interface SortConfig {
  field: SortField;
  direction: SortDirection;
}

export const PAGE_SIZES = [25, 50, 100];
export const DEFAULT_PAGE_SIZE = 25;

/**
 * Read cluster selection from the URL.
 *
 * Two params are supported for backwards compatibility: `clusters` (a
 * comma-separated list) takes precedence, and a single legacy `cluster` is
 * accepted as a one-element selection. Non-numeric entries are dropped rather
 * than producing NaN filters.
 */
export const parseClusterParams = (params: URLSearchParams): number[] => {
  const clusterIdsParam = params.get('clusters');
  if (clusterIdsParam) {
    return clusterIdsParam
      .split(',')
      .map((value) => Number.parseInt(value, 10))
      .filter((value) => Number.isFinite(value));
  }

  const clusterIdParam = params.get('cluster');
  if (!clusterIdParam) {
    return [];
  }

  const parsedClusterId = Number.parseInt(clusterIdParam, 10);
  return Number.isFinite(parsedClusterId) ? [parsedClusterId] : [];
};

/**
 * Write query and cluster selection back to the URL.
 *
 * Existing unrelated params (month, topic) are preserved; a single cluster is
 * written as `cluster` and several as `clusters`, matching what
 * `parseClusterParams` reads.
 */
export const buildFilterParams = (
  searchParams: URLSearchParams,
  query: string,
  clusterIds: Set<number> | number[]
): URLSearchParams => {
  const nextParams = new URLSearchParams(searchParams);
  nextParams.delete('q');
  nextParams.delete('cluster');
  nextParams.delete('clusters');

  const trimmedQuery = query.trim();
  if (trimmedQuery) {
    nextParams.set('q', trimmedQuery);
  }

  const normalizedClusterIds = Array.from(clusterIds).sort((left, right) => left - right);
  if (normalizedClusterIds.length === 1) {
    nextParams.set('cluster', String(normalizedClusterIds[0]));
  } else if (normalizedClusterIds.length > 1) {
    nextParams.set('clusters', normalizedClusterIds.join(','));
  }

  return nextParams;
};

export const formatDataDate = (dateStr: string | null | undefined): string => {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

/** Next sort state for a header click: same field toggles, new field starts ascending. */
export const nextSortConfig = (previous: SortConfig, field: SortField): SortConfig => ({
  field,
  direction: previous.field === field && previous.direction === 'asc' ? 'desc' : 'asc',
});

/** Toggle one cluster in the selection without mutating the input set. */
export const toggleClusterSelection = (
  current: Set<number>,
  clusterId: number
): Set<number> => {
  const next = new Set(current);
  if (next.has(clusterId)) {
    next.delete(clusterId);
  } else {
    next.add(clusterId);
  }
  return next;
};

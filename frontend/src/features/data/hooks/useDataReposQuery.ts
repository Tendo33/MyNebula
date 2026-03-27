import { useQuery } from '@tanstack/react-query';

import { getDataReposV2 } from '../../../api/v2/data';
import { getGraphDataV2 } from '../../../api/v2/graph';
import type { DataRepoItem } from '../../../api/v2/data';
import type { ClusterInfo } from '../../../types';

interface UseDataReposQueryParams {
  clusterIds?: number[];
  searchQuery?: string;
  month?: string | null;
  topic?: string | null;
  sortField?: 'name' | 'language' | 'stargazers_count' | 'starred_at' | 'cluster' | 'summary' | 'last_commit_time';
  sortDirection?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}

export const useDataReposQuery = ({
  clusterIds,
  searchQuery,
  month,
  topic,
  sortField = 'starred_at',
  sortDirection = 'desc',
  limit = 25,
  offset = 0,
}: UseDataReposQueryParams) => {
  const normalizedClusterIds = (clusterIds ?? []).slice().sort((left, right) => left - right);
  const singleClusterId = normalizedClusterIds.length === 1 ? normalizedClusterIds[0] : undefined;
  const clusterIdsKey = normalizedClusterIds.length > 1 ? normalizedClusterIds.join(',') : '';

  const reposQuery = useQuery({
    queryKey: [
      'v2-data-repos',
      singleClusterId ?? null,
      clusterIdsKey,
      searchQuery ?? '',
      month ?? null,
      topic ?? null,
      sortField,
      sortDirection,
      limit,
      offset,
    ],
    queryFn: () =>
      getDataReposV2({
        cluster_id: singleClusterId,
        cluster_ids: normalizedClusterIds.length > 1 ? clusterIdsKey : undefined,
        q: searchQuery?.trim() || undefined,
        month: month || undefined,
        topic: topic || undefined,
        sort_field: sortField,
        sort_direction: sortDirection,
        limit,
        offset,
      }),
    staleTime: 15_000,
  });

  const graphQuery = useQuery({
    queryKey: ['v2-data-graph'],
    queryFn: () => getGraphDataV2({ version: 'active', include_edges: false }),
    staleTime: 15_000,
  });

  return {
    repos: reposQuery.data?.items ?? ([] as DataRepoItem[]),
    count: reposQuery.data?.count ?? 0,
    clusters: graphQuery.data?.clusters ?? ([] as ClusterInfo[]),
    totalNodes: graphQuery.data?.total_nodes ?? 0,
    loading: reposQuery.isLoading || graphQuery.isLoading,
    isFetching: reposQuery.isFetching || graphQuery.isFetching,
    error: reposQuery.error ?? graphQuery.error ?? null,
    retry: async () => {
      await Promise.all([reposQuery.refetch(), graphQuery.refetch()]);
    },
  };
};

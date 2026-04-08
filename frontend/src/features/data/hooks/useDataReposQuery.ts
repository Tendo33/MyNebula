import { useQuery } from '@tanstack/react-query';

import { getDataReposV2 } from '../../../api/v2/data';
import { normalizeSearchQuery } from '../../../utils/search';
import type { DataClusterInfo, DataRepoItem } from '../../../api/v2/data';

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
  const normalizedSearchQuery = normalizeSearchQuery(searchQuery);

  const reposQuery = useQuery({
    queryKey: [
      'v2-data-repos',
      singleClusterId ?? null,
      clusterIdsKey,
      normalizedSearchQuery,
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
        q: normalizedSearchQuery || undefined,
        month: month || undefined,
        topic: topic || undefined,
        sort_field: sortField,
        sort_direction: sortDirection,
        limit,
        offset,
      }),
    staleTime: 15_000,
  });

  return {
    repos: reposQuery.data?.items ?? ([] as DataRepoItem[]),
    count: reposQuery.data?.count ?? 0,
    clusters: reposQuery.data?.clusters ?? ([] as DataClusterInfo[]),
    totalNodes: reposQuery.data?.total_repos ?? 0,
    loading: reposQuery.isLoading,
    isFetching: reposQuery.isFetching,
    error: reposQuery.error ?? null,
    retry: async () => {
      await reposQuery.refetch();
    },
  };
};

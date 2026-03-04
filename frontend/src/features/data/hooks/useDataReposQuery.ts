import { useQuery } from '@tanstack/react-query';

import { getDataReposV2 } from '../../../api/v2/data';
import { getGraphDataV2 } from '../../../api/v2/graph';

interface UseDataReposQueryParams {
  clusterId?: number;
  searchQuery?: string;
}

export const useDataReposQuery = ({ clusterId, searchQuery }: UseDataReposQueryParams) => {
  const reposQuery = useQuery({
    queryKey: ['v2-data-repos', clusterId ?? null, searchQuery ?? ''],
    queryFn: () =>
      getDataReposV2({
        cluster_id: clusterId,
        q: searchQuery?.trim() || undefined,
        limit: 2000,
        offset: 0,
      }),
    staleTime: 15_000,
  });

  const graphQuery = useQuery({
    queryKey: ['v2-data-graph'],
    queryFn: () => getGraphDataV2({ version: 'active', include_edges: false }),
    staleTime: 15_000,
  });

  return {
    repos: reposQuery.data?.items ?? [],
    count: reposQuery.data?.count ?? 0,
    clusters: graphQuery.data?.clusters ?? [],
    totalNodes: graphQuery.data?.total_nodes ?? 0,
    loading: reposQuery.isLoading || graphQuery.isLoading,
    error: reposQuery.error ?? graphQuery.error ?? null,
  };
};

import { useQuery } from '@tanstack/react-query';

import { getGraphDataV2 } from '../../../api/v2/graph';

export const GRAPH_DATA_QUERY_KEY = 'graph-data';

export const useGraphDataQuery = (refreshNonce: number) =>
  useQuery({
    queryKey: [GRAPH_DATA_QUERY_KEY, refreshNonce],
    queryFn: () => getGraphDataV2({ version: 'active', include_edges: false }),
    retry: 1,
    staleTime: 10_000,
  });

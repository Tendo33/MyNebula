import { useQuery } from '@tanstack/react-query';

import { getGraphDataV2 } from '../../../api/v2/graph';
import { queryKeys } from '../../../lib/queryKeys';

export const GRAPH_DATA_QUERY_KEY = queryKeys.graphData()[0];

export const useGraphDataQuery = (refreshNonce: number, enabled = true) =>
  useQuery({
    queryKey: [...queryKeys.graphData(), refreshNonce],
    queryFn: () => getGraphDataV2({ version: 'active', include_edges: false }),
    enabled,
    retry: 1,
    staleTime: 10_000,
  });

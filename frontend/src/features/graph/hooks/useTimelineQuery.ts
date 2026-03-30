import { useQuery } from '@tanstack/react-query';

import { getTimelineDataV2 } from '../../../api/v2/graph';
import { queryKeys } from '../../../lib/queryKeys';

export const TIMELINE_QUERY_KEY = queryKeys.timeline()[0];

export const useTimelineQuery = (refreshNonce: number) =>
  useQuery({
    queryKey: [...queryKeys.timeline(), refreshNonce],
    queryFn: () => getTimelineDataV2('active'),
    retry: 1,
    staleTime: 10_000,
  });

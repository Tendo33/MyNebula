import { useQuery } from '@tanstack/react-query';

import { getTimelineDataV2 } from '../../../api/v2/graph';

export const TIMELINE_QUERY_KEY = 'timeline-data';

export const useTimelineQuery = (refreshNonce: number) =>
  useQuery({
    queryKey: [TIMELINE_QUERY_KEY, refreshNonce],
    queryFn: () => getTimelineDataV2('active'),
    retry: 1,
    staleTime: 10_000,
  });

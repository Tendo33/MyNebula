import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useInfiniteQuery } from '@tanstack/react-query';

import { getGraphEdgesPageV2 } from '../../../api/v2/graph';

export const GRAPH_EDGES_QUERY_KEY = 'graph-edges';

const MAX_AUTO_LOAD_PAGES = 20;

interface UseGraphEdgesInfiniteQueryParams {
  version: string;
  refreshNonce: number;
  enabled: boolean;
  limit?: number;
  maxAutoPages?: number;
}

export const useGraphEdgesInfiniteQuery = ({
  version,
  refreshNonce,
  enabled,
  limit = 1200,
  maxAutoPages = MAX_AUTO_LOAD_PAGES,
}: UseGraphEdgesInfiniteQueryParams) => {
  const pagesLoadedRef = useRef(0);
  const [autoLoadHalted, setAutoLoadHalted] = useState(false);
  const [edgesError, setEdgesError] = useState<string | null>(null);
  const seenNextCursorsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    seenNextCursorsRef.current = new Set();
    pagesLoadedRef.current = 0;
    setAutoLoadHalted(false);
    setEdgesError(null);
  }, [version, refreshNonce]);

  const query = useInfiniteQuery({
    queryKey: [GRAPH_EDGES_QUERY_KEY, version, refreshNonce],
    initialPageParam: 0,
    enabled,
    retry: 2,
    queryFn: ({ pageParam }) =>
      getGraphEdgesPageV2({
        version,
        cursor: pageParam,
        limit,
      }),
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
  const {
    data,
    error,
    hasNextPage,
    isError,
    isFetchingNextPage,
    status,
    fetchNextPage,
    refetch,
  } = query;

  const attemptLoadNextPage = useCallback(async () => {
    if (!hasNextPage || isFetchingNextPage || autoLoadHalted) {
      return;
    }
    if (pagesLoadedRef.current >= maxAutoPages) {
      setAutoLoadHalted(true);
      return;
    }
    const nextCursor = data?.pages.at(-1)?.next_cursor ?? null;
    if (nextCursor == null) {
      return;
    }
    if (seenNextCursorsRef.current.has(nextCursor)) {
      setAutoLoadHalted(true);
      setEdgesError(`Detected duplicated edge cursor ${nextCursor}, auto-loading halted`);
      return;
    }
    seenNextCursorsRef.current.add(nextCursor);
    pagesLoadedRef.current += 1;
    await fetchNextPage();
  }, [
    autoLoadHalted,
    data?.pages,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    maxAutoPages,
  ]);

  useEffect(() => {
    if (status === 'success' && hasNextPage && !isFetchingNextPage) {
      void attemptLoadNextPage();
    }
  }, [
    attemptLoadNextPage,
    hasNextPage,
    isFetchingNextPage,
    status,
  ]);

  useEffect(() => {
    if (error) {
      setEdgesError(error instanceof Error ? error.message : 'Failed to load graph edges');
    }
  }, [error]);

  const retryEdgeLoading = useCallback(async () => {
    setAutoLoadHalted(false);
    setEdgesError(null);
    seenNextCursorsRef.current.clear();
    if (isError) {
      await refetch();
      return;
    }
    await attemptLoadNextPage();
  }, [attemptLoadNextPage, isError, refetch]);

  const stagedEdges = useMemo(
    () => data?.pages.flatMap((page) => page.edges) ?? [],
    [data]
  );

  return {
    ...query,
    stagedEdges,
    edgesError,
    autoLoadHalted,
    retryEdgeLoading,
  };
};

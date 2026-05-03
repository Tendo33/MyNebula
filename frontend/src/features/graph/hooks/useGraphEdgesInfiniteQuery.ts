import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useInfiniteQuery } from '@tanstack/react-query';

import { getGraphEdgesPageV2 } from '../../../api/v2/graph';

export const GRAPH_EDGES_QUERY_KEY = 'graph-edges';

const DEFAULT_EDGE_PAGE_SIZE = 400;
const MAX_AUTO_LOAD_PAGES = 4;

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
  limit = DEFAULT_EDGE_PAGE_SIZE,
  maxAutoPages = MAX_AUTO_LOAD_PAGES,
}: UseGraphEdgesInfiniteQueryParams) => {
  const pagesLoadedRef = useRef(0);
  const autoLoadEnabledRef = useRef(true);
  const [autoLoadHalted, setAutoLoadHalted] = useState(false);
  const [edgesError, setEdgesError] = useState<string | null>(null);
  const seenNextCursorsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    if (!enabled) {
      autoLoadEnabledRef.current = true;
      seenNextCursorsRef.current = new Set();
      pagesLoadedRef.current = 0;
      setAutoLoadHalted(false);
      setEdgesError(null);
      return;
    }

    autoLoadEnabledRef.current = true;
    seenNextCursorsRef.current = new Set();
    pagesLoadedRef.current = 0;
    setAutoLoadHalted(false);
    setEdgesError(null);
  }, [enabled, version, refreshNonce]);

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
    if (!hasNextPage || isFetchingNextPage || autoLoadHalted || !autoLoadEnabledRef.current) {
      return;
    }
    if (pagesLoadedRef.current >= maxAutoPages) {
      autoLoadEnabledRef.current = false;
      setAutoLoadHalted(true);
      return;
    }
    const nextCursor = data?.pages.at(-1)?.next_cursor ?? null;
    if (nextCursor == null) {
      return;
    }
    if (seenNextCursorsRef.current.has(nextCursor)) {
      autoLoadEnabledRef.current = false;
      setAutoLoadHalted(true);
      setEdgesError(`Detected duplicated edge cursor ${nextCursor}, auto-loading halted`);
      return;
    }
    seenNextCursorsRef.current.add(nextCursor);
    try {
      await fetchNextPage();
      pagesLoadedRef.current += 1;
    } catch (error) {
      seenNextCursorsRef.current.delete(nextCursor);
      throw error;
    }
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
    autoLoadEnabledRef.current = true;
    setAutoLoadHalted(false);
    setEdgesError(null);
    seenNextCursorsRef.current.clear();
    if (isError) {
      await refetch();
      return;
    }
    await attemptLoadNextPage();
  }, [attemptLoadNextPage, isError, refetch]);

  const loadMoreEdges = useCallback(async () => {
    setEdgesError(null);
    const nextCursor = data?.pages.at(-1)?.next_cursor ?? null;
    if (nextCursor == null || isFetchingNextPage || !hasNextPage) {
      return;
    }
    if (seenNextCursorsRef.current.has(nextCursor)) {
      setAutoLoadHalted(true);
      setEdgesError(`Detected duplicated edge cursor ${nextCursor}, manual loading halted`);
      return;
    }
    seenNextCursorsRef.current.add(nextCursor);
    try {
      await fetchNextPage();
    } catch (error) {
      seenNextCursorsRef.current.delete(nextCursor);
      throw error;
    }
  }, [data?.pages, fetchNextPage, hasNextPage, isFetchingNextPage]);

  const stagedEdges = useMemo(
    () => data?.pages.flatMap((page) => page.edges) ?? [],
    [data]
  );
  const loadedPages = data?.pages.length ?? 0;
  const nextCursor = data?.pages.at(-1)?.next_cursor ?? null;
  const manualLoadBlocked = nextCursor !== null && seenNextCursorsRef.current.has(nextCursor);

  return {
    ...query,
    stagedEdges,
    edgesError,
    autoLoadHalted,
    retryEdgeLoading,
    loadMoreEdges,
    canLoadMoreEdges:
      Boolean(hasNextPage) && !isFetchingNextPage && !manualLoadBlocked,
    loadedPages,
    pageSize: limit,
  };
};

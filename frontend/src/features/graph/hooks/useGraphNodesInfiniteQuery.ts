import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useInfiniteQuery } from '@tanstack/react-query';

import { getGraphNodesPageV2 } from '../../../api/v2/graph';

export const GRAPH_NODES_QUERY_KEY = 'graph-nodes';

const DEFAULT_NODE_PAGE_SIZE = 400;
const MAX_AUTO_LOAD_PAGES = 4;

interface UseGraphNodesInfiniteQueryParams {
  version: string;
  refreshNonce: number;
  enabled: boolean;
  limit?: number;
  maxAutoPages?: number;
}

export const useGraphNodesInfiniteQuery = ({
  version,
  refreshNonce,
  enabled,
  limit = DEFAULT_NODE_PAGE_SIZE,
  maxAutoPages = MAX_AUTO_LOAD_PAGES,
}: UseGraphNodesInfiniteQueryParams) => {
  const pagesLoadedRef = useRef(0);
  const autoLoadEnabledRef = useRef(true);
  const [autoLoadHalted, setAutoLoadHalted] = useState(false);
  const [nodesError, setNodesError] = useState<string | null>(null);
  const seenNextCursorsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    if (!enabled) {
      autoLoadEnabledRef.current = true;
      seenNextCursorsRef.current = new Set();
      pagesLoadedRef.current = 0;
      setAutoLoadHalted(false);
      setNodesError(null);
      return;
    }

    autoLoadEnabledRef.current = true;
    seenNextCursorsRef.current = new Set();
    pagesLoadedRef.current = 0;
    setAutoLoadHalted(false);
    setNodesError(null);
  }, [enabled, version, refreshNonce]);

  const query = useInfiniteQuery({
    queryKey: [GRAPH_NODES_QUERY_KEY, version, refreshNonce],
    initialPageParam: 0,
    enabled,
    retry: 2,
    queryFn: ({ pageParam }) =>
      getGraphNodesPageV2({
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
      setNodesError(`Detected duplicated node cursor ${nextCursor}, auto-loading halted`);
      return;
    }
    seenNextCursorsRef.current.add(nextCursor);
    const pagesBefore = data?.pages.length ?? 0;
    try {
      const outcome = await fetchNextPage();
      // React Query resolves `fetchNextPage()` with a result object even when
      // the page failed — it only rejects if `throwOnError` is set. Checking
      // whether a page actually landed is therefore the only reliable signal.
      // Counting a failed page would consume auto-load budget, and leaving its
      // cursor in `seenNextCursors` would make a later manual retry of the same
      // cursor trip the duplicate guard and halt loading for good.
      const pagesAfter = outcome.data?.pages.length ?? pagesBefore;
      if (pagesAfter > pagesBefore) {
        pagesLoadedRef.current += 1;
      } else {
        seenNextCursorsRef.current.delete(nextCursor);
      }
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
      setNodesError(error instanceof Error ? error.message : 'Failed to load graph nodes');
    }
  }, [error]);

  const retryNodeLoading = useCallback(async () => {
    autoLoadEnabledRef.current = true;
    setAutoLoadHalted(false);
    setNodesError(null);
    seenNextCursorsRef.current.clear();
    if (isError) {
      await refetch();
      return;
    }
    await attemptLoadNextPage();
  }, [attemptLoadNextPage, isError, refetch]);

  const loadMoreNodes = useCallback(async () => {
    setNodesError(null);
    const nextCursor = data?.pages.at(-1)?.next_cursor ?? null;
    if (nextCursor == null || isFetchingNextPage || !hasNextPage) {
      return;
    }
    if (seenNextCursorsRef.current.has(nextCursor)) {
      setAutoLoadHalted(true);
      setNodesError(`Detected duplicated node cursor ${nextCursor}, manual loading halted`);
      return;
    }
    seenNextCursorsRef.current.add(nextCursor);
    const pagesBefore = data?.pages.length ?? 0;
    try {
      const outcome = await fetchNextPage();
      // Same reasoning as the auto-load path: a resolved promise does not mean
      // the page landed. Releasing the cursor on failure keeps a second manual
      // attempt possible instead of tripping the duplicate guard.
      if ((outcome.data?.pages.length ?? pagesBefore) <= pagesBefore) {
        seenNextCursorsRef.current.delete(nextCursor);
      }
    } catch (error) {
      seenNextCursorsRef.current.delete(nextCursor);
      throw error;
    }
  }, [data?.pages, fetchNextPage, hasNextPage, isFetchingNextPage]);

  const stagedNodes = useMemo(
    () => data?.pages.flatMap((page) => page.nodes) ?? [],
    [data]
  );
  const loadedPages = data?.pages.length ?? 0;
  const nextCursor = data?.pages.at(-1)?.next_cursor ?? null;
  const manualLoadBlocked = nextCursor !== null && seenNextCursorsRef.current.has(nextCursor);

  return {
    ...query,
    stagedNodes,
    nodesError,
    autoLoadHalted,
    retryNodeLoading,
    loadMoreNodes,
    canLoadMoreNodes:
      Boolean(hasNextPage) && !isFetchingNextPage && !manualLoadBlocked,
    loadedPages,
    pageSize: limit,
  };
};

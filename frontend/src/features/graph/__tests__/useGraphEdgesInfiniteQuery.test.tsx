import { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../api/v2/graph', () => ({
  getGraphEdgesPageV2: vi.fn(),
}));

import { getGraphEdgesPageV2 } from '../../../api/v2/graph';
import { useGraphEdgesInfiniteQuery } from '../hooks/useGraphEdgesInfiniteQuery';

const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

describe('useGraphEdgesInfiniteQuery', () => {
  beforeEach(() => {
    vi.mocked(getGraphEdgesPageV2).mockReset();
  });

  it('loads paged edges and merges them', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 10,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: null,
        version: 'snapshot-a',
      });

    const { result } = renderHook(
      () =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 1000,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.stagedEdges.length).toBe(2);
    });
    expect(result.current.autoLoadHalted).toBe(false);
  });

  it('halts auto-load when duplicated cursor is detected', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 5,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: 5,
        version: 'snapshot-a',
      });

    const { result } = renderHook(
      () =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 1000,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(true);
    });
    expect(result.current.edgesError).toContain('duplicated edge cursor');
  });

  it('supports manually loading more edges after auto-load is halted', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 5,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: 10,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 3, target: 4, weight: 0.6 }],
        next_cursor: 15,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 4, target: 5, weight: 0.5 }],
        next_cursor: null,
        version: 'snapshot-a',
      });

    const { result } = renderHook(
      () =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 200,
          maxAutoPages: 1,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(true);
    });
    expect(result.current.stagedEdges).toHaveLength(2);
    expect(result.current.canLoadMoreEdges).toBe(true);

    await act(async () => {
      await result.current.loadMoreEdges();
    });

    await waitFor(() => {
      expect(result.current.stagedEdges).toHaveLength(3);
    });
    expect(result.current.autoLoadHalted).toBe(true);
  });

  it('blocks repeated manual loading when the next cursor was already seen', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 5,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: 10,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 3, target: 4, weight: 0.6 }],
        next_cursor: 10,
        version: 'snapshot-a',
      });

    const { result } = renderHook(
      () =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 200,
          maxAutoPages: 1,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(true);
    });
    expect(result.current.stagedEdges).toHaveLength(2);

    await act(async () => {
      await result.current.loadMoreEdges();
    });

    await waitFor(() => {
      expect(result.current.stagedEdges).toHaveLength(3);
    });

    await act(async () => {
      await result.current.loadMoreEdges();
    });

    expect(result.current.stagedEdges).toHaveLength(3);
    expect(result.current.autoLoadHalted).toBe(true);
    expect(result.current.canLoadMoreEdges).toBe(false);
    expect(result.current.edgesError).toContain('manual loading halted');
    expect(vi.mocked(getGraphEdgesPageV2)).toHaveBeenCalledTimes(3);
  });

  it('does not consume auto-load budget when a page fetch fails transiently', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 5,
        version: 'snapshot-a',
      })
      .mockRejectedValueOnce(new Error('temporary edge failure'))
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: 10,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 3, target: 4, weight: 0.6 }],
        next_cursor: null,
        version: 'snapshot-a',
      });

    const { result } = renderHook(
      () =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 200,
          maxAutoPages: 2,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.stagedEdges).toHaveLength(1);
    });
    expect(result.current.stagedEdges).toHaveLength(1);

    await act(async () => {
      await result.current.retryEdgeLoading();
    });

    await waitFor(() => {
      expect(result.current.stagedEdges).toHaveLength(3);
    });
    expect(result.current.autoLoadHalted).toBe(false);
  });

  it('resets halted auto-load state when the graph view is disabled and re-enabled', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 5,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: 10,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 3, target: 4, weight: 0.6 }],
        next_cursor: null,
        version: 'snapshot-a',
      });

    const { result, rerender } = renderHook(
      ({ enabled }) =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled,
          limit: 200,
          maxAutoPages: 1,
        }),
      {
        initialProps: { enabled: true },
        wrapper: createWrapper(),
      }
    );

    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(true);
    });

    rerender({ enabled: false });

    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(false);
    });
    expect(result.current.edgesError).toBeNull();

    rerender({ enabled: true });

    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(false);
    });
    expect(result.current.edgesError).toBeNull();
  });
});

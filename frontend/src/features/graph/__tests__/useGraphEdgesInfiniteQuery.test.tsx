import { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
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
});

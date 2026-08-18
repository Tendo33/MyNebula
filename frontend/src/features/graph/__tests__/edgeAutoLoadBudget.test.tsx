import { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../api/v2/graph', () => ({
  getGraphEdgesPageV2: vi.fn(),
}));

import { getGraphEdgesPageV2 } from '../../../api/v2/graph';
import { useGraphEdgesInfiniteQuery } from '../hooks/useGraphEdgesInfiniteQuery';

/**
 * Regression guard for the auto-load budget accounting.
 *
 * `fetchNextPage()` resolves with a result object even when the page failed —
 * it only rejects when `throwOnError` is set. Before the fix the hook counted
 * every settled call as a loaded page, so a failed page both consumed
 * auto-load budget and left its cursor in `seenNextCursors`, which made a
 * later manual attempt at the same cursor trip the duplicate guard and halt
 * edge loading permanently.
 */
const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

const page = (source: number, target: number, next_cursor: number | null) => ({
  edges: [{ source, target, weight: 0.5 }],
  next_cursor,
  version: 'snapshot-a',
});

describe('useGraphEdgesInfiniteQuery auto-load budget', () => {
  beforeEach(() => {
    vi.mocked(getGraphEdgesPageV2).mockReset();
  });

  it('releases the cursor of a page that never landed, so manual retry still works', async () => {
    // Page 1 succeeds; every subsequent attempt fails, exhausting the hook's
    // internal retries and leaving the auto-load in an error state.
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce(page(1, 2, 5))
      .mockRejectedValue(new Error('edge page unavailable'));

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

    // Give the failing auto-load, including its retries, room to settle.
    await waitFor(
      () => {
        expect(result.current.edgesError).not.toBeNull();
      },
      { timeout: 8000 }
    );

    // The provider recovers. A manual retry targets cursor 5 again — the same
    // cursor the failed auto-load claimed.
    vi.mocked(getGraphEdgesPageV2).mockReset();
    vi.mocked(getGraphEdgesPageV2).mockResolvedValue(page(9, 10, null));

    await act(async () => {
      await result.current.loadMoreEdges();
    });

    // Before the fix, cursor 5 was still recorded as seen, so this manual
    // attempt tripped the duplicate guard and edge loading was stuck for good.
    await waitFor(
      () => {
        expect(result.current.stagedEdges).toHaveLength(2);
      },
      { timeout: 5000 }
    );
    expect(result.current.edgesError ?? '').not.toMatch(/duplicated edge cursor/);
  }, 20000);

  it('still counts a page that did land', async () => {
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce(page(1, 2, 5))
      .mockResolvedValueOnce(page(2, 3, 10))
      .mockResolvedValueOnce(page(3, 4, 20))
      .mockResolvedValue(page(4, 5, null));

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

    // Budget of 2 auto-loaded pages on top of the initial page.
    await waitFor(
      () => {
        expect(result.current.stagedEdges).toHaveLength(3);
      },
      { timeout: 5000 }
    );

    await waitFor(
      () => {
        expect(result.current.autoLoadHalted).toBe(true);
      },
      { timeout: 5000 }
    );
    // Halting on budget is not an error state; manual loading stays available.
    expect(result.current.edgesError).toBeNull();
  }, 15000);

  it('halts when the server repeats a cursor', async () => {
    // A server that keeps returning the same cursor would loop forever.
    vi.mocked(getGraphEdgesPageV2).mockResolvedValue(page(1, 2, 5));

    const { result } = renderHook(
      () =>
        useGraphEdgesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 200,
          maxAutoPages: 10,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(
      () => {
        expect(result.current.autoLoadHalted).toBe(true);
      },
      { timeout: 5000 }
    );
    expect(result.current.edgesError).toMatch(/duplicated edge cursor/);
  }, 15000);

  it('resets budget and cursor history when the snapshot version changes', async () => {
    vi.mocked(getGraphEdgesPageV2).mockResolvedValue(page(1, 2, null));

    const { result, rerender } = renderHook(
      ({ version }: { version: string }) =>
        useGraphEdgesInfiniteQuery({
          version,
          refreshNonce: 0,
          enabled: true,
          limit: 200,
          maxAutoPages: 2,
        }),
      { wrapper: createWrapper(), initialProps: { version: 'snapshot-a' } }
    );

    await waitFor(() => {
      expect(result.current.stagedEdges).toHaveLength(1);
    });

    await act(async () => {
      rerender({ version: 'snapshot-b' });
    });

    // A new snapshot starts from a clean budget, otherwise the second snapshot
    // would inherit the first one's exhausted state.
    await waitFor(() => {
      expect(result.current.autoLoadHalted).toBe(false);
    });
    expect(result.current.edgesError).toBeNull();
  }, 15000);
});

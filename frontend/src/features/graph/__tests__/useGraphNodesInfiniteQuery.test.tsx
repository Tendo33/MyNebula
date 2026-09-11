import { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../api/v2/graph', () => ({
  getGraphNodesPageV2: vi.fn(),
}));

import { getGraphNodesPageV2 } from '../../../api/v2/graph';
import { useGraphNodesInfiniteQuery } from '../hooks/useGraphNodesInfiniteQuery';

const node = {
  id: 1,
  github_id: 1,
  full_name: 'octo/nebula',
  name: 'nebula',
  html_url: 'https://github.com/octo/nebula',
  owner: 'octo',
  x: 0,
  y: 0,
  z: 0,
  size: 1,
  cluster_id: null,
  color: '#000000',
  star_list_id: null,
  stargazers_count: 1,
};

const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

describe('useGraphNodesInfiniteQuery', () => {
  beforeEach(() => {
    vi.mocked(getGraphNodesPageV2).mockReset();
  });

  it('loads paged nodes and merges them', async () => {
    vi.mocked(getGraphNodesPageV2)
      .mockResolvedValueOnce({
        nodes: [node],
        next_cursor: 10,
        version: 'snapshot-a',
      })
      .mockResolvedValueOnce({
        nodes: [{ ...node, id: 2, github_id: 2, name: 'other', full_name: 'octo/other' }],
        next_cursor: null,
        version: 'snapshot-a',
      });

    const { result } = renderHook(
      () =>
        useGraphNodesInfiniteQuery({
          version: 'snapshot-a',
          refreshNonce: 0,
          enabled: true,
          limit: 400,
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(result.current.stagedNodes.length).toBe(2);
    });
    expect(result.current.autoLoadHalted).toBe(false);
  });
});

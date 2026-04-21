import { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../api/v2/data', () => ({
  getDataReposV2: vi.fn(),
}));

import { getDataReposV2 } from '../../../api/v2/data';
import { useDataReposQuery } from './useDataReposQuery';

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

describe('useDataReposQuery', () => {
  beforeEach(() => {
    vi.mocked(getDataReposV2).mockReset();
    vi.mocked(getDataReposV2).mockResolvedValue({
      items: [],
      clusters: [],
      count: 0,
      total_repos: 0,
      limit: 25,
      offset: 0,
    });
  });

  it('normalizes shared stars search syntax before issuing data queries', async () => {
    renderHook(
      () =>
        useDataReposQuery({
          searchQuery: '  Stars:>42  ',
        }),
      { wrapper: createWrapper() }
    );

    await waitFor(() => {
      expect(getDataReposV2).toHaveBeenCalledWith(
        expect.objectContaining({
          q: 'stars:>42',
        })
      );
    });
  });
});

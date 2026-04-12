import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { queryClient } from '../lib/queryClient';

const graphApiMocks = vi.hoisted(() => ({
  getGraphDataV2: vi.fn(),
  getTimelineDataV2: vi.fn(),
  getGraphEdgesPageV2: vi.fn(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key }),
}));

vi.mock('../api/v2/graph', () => ({
  getGraphDataV2: graphApiMocks.getGraphDataV2,
  getTimelineDataV2: graphApiMocks.getTimelineDataV2,
  getGraphEdgesPageV2: graphApiMocks.getGraphEdgesPageV2,
}));

vi.mock('../hooks/useCommandPalette', () => ({
  __esModule: true,
  default: vi.fn(() => ({
    isOpen: false,
    close: vi.fn(),
    open: vi.fn(),
    toggle: vi.fn(),
  })),
}));

vi.mock('../components/ui/CommandPalette', () => ({
  __esModule: true,
  default: ({
    onSelectSearch,
  }: {
    onSelectSearch?: (value: string, facet?: 'search' | 'language' | 'tag') => void;
  }) => (
    <button type="button" onClick={() => onSelectSearch?.('graph', 'tag')}>
      trigger-tag-search
    </button>
  ),
}));

vi.mock('../pages/Dashboard', () => ({
  __esModule: true,
  default: () => <div>dashboard</div>,
}));

vi.mock('../pages/GraphPage', () => ({
  __esModule: true,
  default: () => <div>graph</div>,
}));

vi.mock('../pages/DataPage', () => ({
  __esModule: true,
  default: () => <div>data</div>,
}));

vi.mock('../pages/Settings', () => ({
  __esModule: true,
  default: () => <div>settings</div>,
}));

vi.mock('../contexts/AdminAuthContext', () => ({
  AdminAuthProvider: ({ children }: { children: ReactNode }) => children,
}));

import App from '../App';
import useCommandPalette from '../hooks/useCommandPalette';

const mockedUseCommandPalette = vi.mocked(useCommandPalette);

describe('App graph gating', () => {
  beforeEach(() => {
    queryClient.clear();
    graphApiMocks.getGraphDataV2.mockReset();
    graphApiMocks.getTimelineDataV2.mockReset();
    graphApiMocks.getGraphEdgesPageV2.mockReset();
    graphApiMocks.getGraphDataV2.mockResolvedValue({
      nodes: [],
      edges: [],
      clusters: [],
      star_lists: [],
      total_nodes: 0,
      total_edges: 0,
      total_clusters: 0,
      total_star_lists: 0,
      version: 'active',
    });
    graphApiMocks.getTimelineDataV2.mockResolvedValue({
      points: [],
      total_stars: 0,
      date_range: ['2026-01-01', '2026-01-01'],
      version: 'active',
    });
    graphApiMocks.getGraphEdgesPageV2.mockResolvedValue({
      edges: [],
      next_cursor: null,
      version: 'active',
    });
    mockedUseCommandPalette.mockReturnValue({
      isOpen: false,
      close: vi.fn(),
      open: vi.fn(),
      toggle: vi.fn(),
    });
    window.history.pushState({}, '', '/');
  });

  it('does not fetch graph payloads on non-graph routes when palette is closed', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(graphApiMocks.getGraphDataV2).not.toHaveBeenCalled();
      expect(graphApiMocks.getTimelineDataV2).not.toHaveBeenCalled();
      expect(graphApiMocks.getGraphEdgesPageV2).not.toHaveBeenCalled();
    });
  });

  it('does not fetch graph payloads on non-graph routes when palette is open', async () => {
    mockedUseCommandPalette.mockReturnValue({
      isOpen: true,
      close: vi.fn(),
      open: vi.fn(),
      toggle: vi.fn(),
    });

    render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(graphApiMocks.getGraphDataV2).not.toHaveBeenCalled();
      expect(graphApiMocks.getTimelineDataV2).not.toHaveBeenCalled();
      expect(graphApiMocks.getGraphEdgesPageV2).not.toHaveBeenCalled();
    });
  });

  it('normalizes tag palette navigation onto the shared q search param', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    );

    fireEvent.click(screen.getByRole('button', { name: 'trigger-tag-search' }));

    await waitFor(() => {
      expect(window.location.pathname).toBe('/graph');
      expect(window.location.search).toBe('?q=graph');
    });
  });
});

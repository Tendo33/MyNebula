import { render, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import type { GraphNode } from '../../types';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const routerFuture = { v7_startTransition: true, v7_relativeSplatPath: true } as const;
const sampleNode: GraphNode = {
  id: 1,
  github_id: 1,
  name: 'nebula',
  full_name: 'octo/nebula',
  description: 'desc',
  language: 'TypeScript',
  html_url: 'https://github.com/octo/nebula',
  owner: 'octo',
  owner_avatar_url: '',
  x: 0,
  y: 0,
  z: 0,
  cluster_id: 2,
  color: '#000',
  size: 1,
  star_list_id: null,
  stargazers_count: 42,
};

const otherNode: GraphNode = {
  ...sampleNode,
  id: 196,
  github_id: 196,
  name: 'supernova',
  full_name: 'octo/supernova',
  cluster_id: 3,
};

const graphState = {
  filteredData: {
    total_nodes: 1,
    nodes: [{ id: 1 }],
  },
  rawData: {
    nodes: [
      sampleNode,
      otherNode,
    ],
    edges: [],
    clusters: [
      { id: 2, name: 'Core', keywords: [], color: '#123456', repo_count: 1 },
      { id: 3, name: 'Explore', keywords: [], color: '#654321', repo_count: 1 },
    ],
    star_lists: [],
    total_nodes: 2,
    total_edges: 0,
    total_clusters: 2,
    total_star_lists: 0,
  },
  loadData: vi.fn(),
  retryEdgeLoading: vi.fn(),
  loadMoreEdges: vi.fn(),
  edgesLoading: false,
  canLoadMoreEdges: false,
  autoLoadHalted: false,
  loadedEdgePages: 0,
  edgePageSize: 400,
  error: null,
  selectedNode: null as GraphNode | null,
  setSelectedNode: vi.fn(),
  filters: {
    selectedClusters: new Set<number>(),
    selectedStarLists: new Set<number>(),
    searchQuery: '',
    timeRange: null,
    minStars: 0,
    languages: new Set<string>(),
  },
  setSelectedClusters: vi.fn(),
  setSearchQuery: vi.fn(),
  setSelectedLanguages: vi.fn(),
  clearFilters: vi.fn(),
};

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { count?: number },
      maybeOptions?: { count?: number }
    ) => {
      const fallback = typeof fallbackOrOptions === 'string' ? fallbackOrOptions : undefined;
      const options = typeof fallbackOrOptions === 'object' ? fallbackOrOptions : maybeOptions;
      if (key === 'dashboard.subtitle' && options?.count !== undefined) {
        return `${options.count} repos`;
      }
      return fallback ?? key;
    },
  }),
}));

vi.mock('../../contexts/GraphContext', () => ({
  useGraph: () => graphState,
}));

vi.mock('../../components/layout/Sidebar', () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock('../../components/graph/Graph2D', () => ({
  __esModule: true,
  default: () => <div data-testid="graph-2d" />,
}));

vi.mock('../../components/graph/Timeline', () => ({
  __esModule: true,
  default: () => <div data-testid="timeline" />,
}));

vi.mock('../../components/graph/ClusterPanel', () => ({
  __esModule: true,
  default: () => <div data-testid="cluster-panel" />,
}));

vi.mock('../../components/graph/StarListPanel', () => ({
  __esModule: true,
  default: () => <div data-testid="star-list-panel" />,
}));

vi.mock('../../components/graph/RepoDetailsPanel', () => ({
  RepoDetailsPanel: ({ node }: { node: { id: number } }) => <div data-testid="repo-details">{node.id}</div>,
}));

vi.mock('../../components/layout/LanguageSwitch', () => ({
  LanguageSwitch: () => <div data-testid="lang-switch" />,
}));

vi.mock('../../components/ui/SearchInput', () => ({
  SearchInput: ({ value }: { value: string }) => <div data-testid="search-input">{value}</div>,
}));

import GraphPage from '../GraphPage';

const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
};

describe('GraphPage URL state', () => {
  beforeEach(() => {
    graphState.selectedNode = null;
    graphState.filters.selectedClusters = new Set<number>();
    graphState.filters.languages = new Set<string>();
    graphState.filters.searchQuery = '';
    graphState.setSelectedNode.mockReset();
    graphState.setSelectedClusters.mockReset();
    graphState.setSelectedLanguages.mockReset();
    graphState.setSearchQuery.mockReset();
  });

  it('restores node, cluster, language and query filters from the URL', async () => {
    render(
      <MemoryRouter
        initialEntries={['/graph?node=1&cluster=2&language=TypeScript&q=nebula']}
        future={routerFuture}
      >
        <Routes>
          <Route path="/graph" element={<GraphPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(graphState.setSelectedNode).toHaveBeenCalledWith(
        expect.objectContaining({ id: 1 })
      );
      expect(graphState.setSelectedClusters).toHaveBeenCalledWith([2]);
      expect(graphState.setSelectedLanguages).toHaveBeenCalledWith(['TypeScript']);
      expect(graphState.setSearchQuery).toHaveBeenCalledWith('nebula');
    });
  });

  it('clears the selected node when the URL no longer contains a node param', async () => {
    graphState.selectedNode = graphState.rawData.nodes[0];

    render(
      <MemoryRouter initialEntries={['/graph']} future={routerFuture}>
        <Routes>
          <Route path="/graph" element={<GraphPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(graphState.setSelectedNode).toHaveBeenCalledWith(null);
    });
  });

  it('prefers the incoming node query over a stale selected node in store state', async () => {
    graphState.selectedNode = sampleNode;

    const { getByTestId } = render(
      <MemoryRouter initialEntries={['/graph?node=196']} future={routerFuture}>
        <LocationProbe />
        <Routes>
          <Route path="/graph" element={<GraphPage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(graphState.setSelectedNode).toHaveBeenCalledWith(
        expect.objectContaining({ id: 196 })
      );
    });

    await waitFor(() => {
      expect(getByTestId('location')).toHaveTextContent('/graph?node=196');
    });
  });

  it('does not show loading copy when edge loading is paused', async () => {
    graphState.autoLoadHalted = true;
    graphState.canLoadMoreEdges = true;
    graphState.loadedEdgePages = 4;
    graphState.edgePageSize = 400;
    graphState.edgesLoading = false;

    const { queryByText, getAllByText } = render(
      <MemoryRouter initialEntries={['/graph']} future={routerFuture}>
        <Routes>
          <Route path="/graph" element={<GraphPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(queryByText('Loading edges...')).not.toBeInTheDocument();
    expect(getAllByText('Load more edges').length).toBeGreaterThan(0);

    graphState.autoLoadHalted = false;
    graphState.canLoadMoreEdges = false;
    graphState.loadedEdgePages = 0;
  });

  it('disables paused edge loading CTA when more edges are blocked', async () => {
    graphState.autoLoadHalted = true;
    graphState.canLoadMoreEdges = false;

    const { getAllByRole } = render(
      <MemoryRouter initialEntries={['/graph']} future={routerFuture}>
        <Routes>
          <Route path="/graph" element={<GraphPage />} />
        </Routes>
      </MemoryRouter>
    );

    const pausedButtons = getAllByRole('button', { name: 'Load more edges' });
    expect(pausedButtons).toHaveLength(2);
    expect(pausedButtons.every((button) => (button as HTMLButtonElement).disabled)).toBe(true);

    graphState.autoLoadHalted = false;
  });
});

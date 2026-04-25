import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

const routerFuture = { v7_startTransition: true, v7_relativeSplatPath: true } as const;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

vi.mock('../../components/layout/Sidebar', () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock('../../components/layout/LanguageSwitch', () => ({
  LanguageSwitch: () => <div data-testid="lang-switch" />,
}));

vi.mock('../../components/ui/SearchInput', () => ({
  SearchInput: ({
    value,
    onSearch,
  }: {
    value: string;
    onSearch?: (value: string) => void;
  }) => (
    <input
      aria-label="search"
      value={value}
      onChange={(event) => onSearch?.(event.target.value)}
    />
  ),
}));

vi.mock('../../features/data/hooks/useDataReposQuery', () => ({
  useDataReposQuery: () => ({
    repos: [],
    clusters: [
      { id: 1, name: 'Cluster 1', color: '#111111', repo_count: 1, keywords: [] },
      { id: 2, name: 'Cluster 2', color: '#222222', repo_count: 1, keywords: [] },
    ],
    totalNodes: 2,
    count: 0,
    loading: false,
    error: null,
    retry: vi.fn(),
  }),
}));

import DataPage from '../DataPage';

const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
};

describe('DataPage URL state', () => {
  it('restores query and cluster filters from the URL and keeps them shareable', async () => {
    render(
      <MemoryRouter initialEntries={['/data?q=nebula&cluster=1']} future={routerFuture}>
        <LocationProbe />
        <Routes>
          <Route path="/data" element={<DataPage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByLabelText('search')).toHaveValue('nebula');

    fireEvent.click(screen.getByRole('button', { name: /Cluster 1/ }));

    await waitFor(() => {
      expect(screen.getByTestId('location')).toHaveTextContent('/data?q=nebula');
    });
  });
});

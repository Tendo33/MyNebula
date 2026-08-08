import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock('../../components/layout/Sidebar', () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock('../../components/layout/LanguageSwitch', () => ({
  LanguageSwitch: () => <div data-testid="lang-switch" />,
}));

vi.mock('../../features/dashboard/hooks/useDashboardQuery', () => ({
  useDashboardQuery: () => ({
    loading: false,
    stats: {
      totalRepos: 10,
      totalTopics: 6,
      totalClusters: 3,
      totalEdges: 9,
      topLanguages: [],
      topLanguage: 'TypeScript (6)',
      topClusters: [{ id: 1, name: 'Cluster 1', color: '#000000', repo_count: 4, keywords: ['graph'] }],
      topTopics: [{ topic: 'graph', count: 4 }],
      recentActivity: 4,
    },
    activityData: [],
    maxActivity: 1,
    error: null,
    retry: vi.fn(),
  }),
}));

import Dashboard from '../Dashboard';

describe('Dashboard v2 smoke', () => {
  it('renders v2 dashboard summary cards', () => {
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>
    );

    expect(screen.getByText('dashboard.total_repos')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
  });
});

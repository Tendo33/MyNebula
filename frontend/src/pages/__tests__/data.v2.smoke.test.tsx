import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => {
      if (key === 'common.repositories' && options?.count) {
        return `${options.count} repositories`;
      }
      return key;
    },
  }),
}));

vi.mock('../../components/layout/Sidebar', () => ({
  Sidebar: () => <div data-testid="sidebar" />,
}));

vi.mock('../../components/layout/LanguageSwitch', () => ({
  LanguageSwitch: () => <div data-testid="lang-switch" />,
}));

vi.mock('../../features/data/hooks/useDataReposQuery', () => ({
  useDataReposQuery: () => ({
    repos: [
      {
        id: 1,
        full_name: 'octo/repo',
        name: 'repo',
        owner: 'octo',
        owner_avatar_url: null,
        description: 'desc',
        ai_summary: 'summary',
        topics: ['ai'],
        language: 'TypeScript',
        stargazers_count: 42,
        html_url: 'https://github.com/octo/repo',
        cluster_id: 1,
        star_list_id: 1,
        starred_at: '2026-02-01T00:00:00Z',
        last_commit_time: '2026-02-01T00:00:00Z',
      },
    ],
    clusters: [{ id: 1, name: 'Cluster 1', color: '#000000', repo_count: 1, keywords: [] }],
    totalNodes: 1,
    count: 1,
    loading: false,
    error: null,
  }),
}));

import DataPage from '../DataPage';

describe('Data page v2 smoke', () => {
  it('renders repository rows from v2 data query', () => {
    render(
      <MemoryRouter>
        <DataPage />
      </MemoryRouter>
    );

    expect(screen.getByText('octo/repo')).toBeInTheDocument();
  });
});

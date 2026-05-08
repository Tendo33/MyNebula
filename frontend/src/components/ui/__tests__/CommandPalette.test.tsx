import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import React from 'react';

const setSelectedNode = vi.fn();
const setSearchQuery = vi.fn();
const searchRepos = vi.fn();
const tMock = (key: string, fallback?: string) => fallback ?? key;
const rawData = {
  nodes: [
    {
      id: 1,
      name: 'nebula',
      full_name: 'octo/nebula',
      description: 'desc',
      ai_summary: 'summary',
      owner: 'octo',
      owner_avatar_url: null,
      language: 'TypeScript',
      ai_tags: ['graph'],
      topics: ['graph'],
      stargazers_count: 42,
    },
  ],
  clusters: [],
};

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: tMock }),
}));

vi.mock('../../../contexts/GraphContext', () => ({
  useGraph: () => ({
    rawData,
    setSelectedNode,
    setSearchQuery,
  }),
}));

vi.mock('../../../api/repos', () => ({
  searchRepos: (...args: unknown[]) => searchRepos(...args),
}));

vi.mock('react-dom', async () => {
  const actual = await vi.importActual<typeof import('react-dom')>('react-dom');
  return {
    ...actual,
    createPortal: (node: React.ReactNode) => node,
  };
});

import CommandPalette from '../CommandPalette';

describe('CommandPalette', () => {
  beforeEach(() => {
    setSelectedNode.mockReset();
    setSearchQuery.mockReset();
    searchRepos.mockReset();
  });

  it('emits language selection to the graph route callback', () => {
    const onSelectSearch = vi.fn();

    render(
      <CommandPalette
        isOpen={true}
        onClose={vi.fn()}
        onSelectSearch={onSelectSearch}
      />
    );

    const input = screen.getByLabelText('Search repos, clusters, languages, tags...');
    fireEvent.change(input, { target: { value: 'Type' } });
    fireEvent.click(screen.getByText('TypeScript'));

    expect(setSearchQuery).toHaveBeenCalledWith('TypeScript');
    expect(onSelectSearch).toHaveBeenCalledWith('TypeScript', 'language');
  });

  it('falls back to remote semantic repo search when local repo results are empty', async () => {
    searchRepos.mockResolvedValue([
      {
        repo: {
          id: 2,
          github_repo_id: 200,
          full_name: 'remote/galaxy',
          name: 'galaxy',
          owner: 'remote',
          description: 'semantic result',
          language: 'Python',
          html_url: 'https://github.com/remote/galaxy',
          stargazers_count: 120,
          ai_summary: 'remote summary',
          topics: ['semantic'],
          forks_count: 5,
          watchers_count: 5,
          open_issues_count: 0,
          cluster_id: null,
          coord_x: null,
          coord_y: null,
          coord_z: null,
          starred_at: null,
          repo_updated_at: null,
          is_embedded: true,
          is_summarized: true,
        },
        score: 0.91,
      },
    ]);

    const onSelectNode = vi.fn();

    render(
      <CommandPalette
        isOpen={true}
        onClose={vi.fn()}
        onSelectNode={onSelectNode}
      />
    );

    fireEvent.change(screen.getByLabelText('Search repos, clusters, languages, tags...'), {
      target: { value: 'galaxy clusters' },
    });

    await waitFor(() => {
      expect(searchRepos).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'galaxy clusters',
        }),
        expect.objectContaining({
          signal: expect.any(AbortSignal),
        }),
      );
    });

    expect(await screen.findByText('remote/galaxy')).toBeInTheDocument();
    fireEvent.click(screen.getByText('remote/galaxy'));

    expect(setSelectedNode).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 2,
        full_name: 'remote/galaxy',
      })
    );
    expect(onSelectNode).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 2,
        full_name: 'remote/galaxy',
      })
    );
  });
});

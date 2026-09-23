import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { RepoDetailsPanel } from '../RepoDetailsPanel';
import { GraphNode } from '../../../types';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

const mocks = vi.hoisted(() => ({
  getRelatedRepos: vi.fn(),
  setSelectedNode: vi.fn(),
  rawData: null as null | { nodes: GraphNode[]; request_id: string },
}));

vi.mock('../../../contexts/GraphContext', () => ({
  useGraph: () => ({
    rawData: mocks.rawData,
    settings: { relatedMinSemantic: 0.65 },
    setSelectedNode: mocks.setSelectedNode,
  }),
}));

vi.mock('../../../api/repos', () => ({
  getRelatedRepos: mocks.getRelatedRepos,
}));

const node: GraphNode = {
  id: 1,
  github_id: 123,
  full_name: 'HKUDS/RAG-Anything',
  name: 'RAG-Anything',
  description: 'All-in-One RAG Framework',
  language: 'Python',
  html_url: 'https://github.com/HKUDS/RAG-Anything',
  owner: 'HKUDS',
  owner_avatar_url: 'https://example.com/avatar.png',
  x: 0,
  y: 0,
  z: 0,
  cluster_id: null,
  color: '#2D59C8',
  size: 1,
  star_list_id: null,
  stargazers_count: 15436,
};

describe('RepoDetailsPanel', () => {
  beforeEach(() => {
    mocks.rawData = null;
    mocks.getRelatedRepos.mockReset();
    mocks.setSelectedNode.mockReset();
  });

  it('renders the GitHub action with the same readable style as other external links', () => {
    render(<RepoDetailsPanel node={node} onClose={vi.fn()} />);

    const githubLink = screen.getByRole('link', { name: /github/i });

    expect(githubLink).toHaveAttribute('href', node.html_url);
    expect(githubLink.className).toContain('border-border');
    expect(githubLink.className).not.toContain('bg-primary');
  });

  it('keeps API results that are outside the currently loaded graph page', async () => {
    mocks.rawData = { nodes: [node], request_id: 'snapshot-1' };
    mocks.getRelatedRepos.mockResolvedValue([
      {
        repo: {
          id: 99,
          github_repo_id: 999,
          full_name: 'octo/related-repo',
          owner: 'octo',
          name: 'related-repo',
          description: 'A related repository outside the current graph page',
          language: 'TypeScript',
          html_url: 'https://github.com/octo/related-repo',
          stargazers_count: 120,
          topics: ['rag'],
          cluster_id: 3,
          coord_x: 1,
          coord_y: 2,
          coord_z: 3,
        },
        score: 0.82,
        reasons: ['semantic:very-high'],
        components: {
          semantic: 0.9,
          tag_overlap: 0.4,
          same_star_list: 0,
          same_language: 0,
        },
      },
    ]);

    render(<RepoDetailsPanel node={node} onClose={vi.fn()} />);

    const related = await screen.findByRole('button', { name: /related-repo/i });
    fireEvent.click(related);

    await waitFor(() => {
      expect(mocks.setSelectedNode).toHaveBeenCalledWith(
        expect.objectContaining({ id: 99, full_name: 'octo/related-repo' })
      );
    });
  });
});

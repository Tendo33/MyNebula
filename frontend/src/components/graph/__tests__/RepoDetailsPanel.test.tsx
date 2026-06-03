import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { RepoDetailsPanel } from '../RepoDetailsPanel';
import { GraphNode } from '../../../types';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

vi.mock('../../../contexts/GraphContext', () => ({
  useGraph: () => ({
    rawData: null,
    settings: { relatedMinSemantic: 0.65 },
    setSelectedNode: vi.fn(),
  }),
}));

vi.mock('../../../api/repos', () => ({
  getRelatedRepos: vi.fn(),
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
  it('renders the GitHub action with the same readable style as other external links', () => {
    render(<RepoDetailsPanel node={node} onClose={vi.fn()} />);

    const githubLink = screen.getByRole('link', { name: /github/i });

    expect(githubLink).toHaveAttribute('href', node.html_url);
    expect(githubLink).toHaveClass('bg-bg-hover');
    expect(githubLink).toHaveClass('text-text-main');
    expect(githubLink).toHaveClass('border-border-light');
    expect(githubLink).not.toHaveClass('bg-text-main');
    expect(githubLink).not.toHaveClass('text-bg-main');
  });
});

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const setSelectedNode = vi.fn();
const setSearchQuery = vi.fn();

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key }),
}));

vi.mock('../../../contexts/GraphContext', () => ({
  useGraph: () => ({
    rawData: {
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
    },
    setSelectedNode,
    setSearchQuery,
  }),
}));

import CommandPalette from '../CommandPalette';

describe('CommandPalette', () => {
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
});

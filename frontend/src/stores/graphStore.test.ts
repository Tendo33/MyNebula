import { afterEach, describe, expect, it, vi } from 'vitest';
import { useGraphStore } from './graphStore';

describe('useGraphStore', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('toggles cluster selection', () => {
    useGraphStore.setState({
      ...useGraphStore.getState(),
      filters: {
        ...useGraphStore.getState().filters,
        selectedClusters: new Set<number>(),
      },
    });

    useGraphStore.getState().toggleCluster(1);
    expect(useGraphStore.getState().filters.selectedClusters.has(1)).toBe(true);

    useGraphStore.getState().toggleCluster(1);
    expect(useGraphStore.getState().filters.selectedClusters.has(1)).toBe(false);
  });

  it('updates selected node', () => {
    useGraphStore.getState().setSelectedNode({
      id: 1,
      github_id: 1,
      full_name: 'owner/repo',
      name: 'repo',
      html_url: 'https://github.com/owner/repo',
      owner: 'owner',
      x: 0,
      y: 0,
      z: 0,
      cluster_id: null,
      color: '#000000',
      size: 1,
      star_list_id: null,
      stargazers_count: 0,
    });

    expect(useGraphStore.getState().selectedNode?.id).toBe(1);
  });

  it('keeps updating settings when localStorage writes fail', () => {
    const setItemSpy = vi
      .spyOn(window.localStorage.__proto__, 'setItem')
      .mockImplementation(() => {
        throw new Error('storage unavailable');
      });

    expect(() =>
      useGraphStore.getState().updateSettings({
        maxClusters: 12,
      })
    ).not.toThrow();

    expect(useGraphStore.getState().settings.maxClusters).toBe(12);
    expect(setItemSpy).toHaveBeenCalled();
  });
});

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./v2/graph', () => ({
  getGraphDataV2: vi.fn(async () => ({
    nodes: [],
    edges: [],
    clusters: [],
    star_lists: [],
    total_nodes: 0,
    total_edges: 0,
    total_clusters: 0,
    total_star_lists: 0,
  })),
  getTimelineDataV2: vi.fn(async () => ({
    points: [],
    total_stars: 0,
    date_range: ['', ''],
  })),
  getGraphEdgesPageV2: vi
    .fn(),
  rebuildGraphV2: vi.fn(),
}));

import { getGraphData, getGraphEdges } from './graph';
import { getGraphDataV2, getGraphEdgesPageV2 } from './v2/graph';

describe('api/graph wrappers', () => {
  beforeEach(() => {
    vi.mocked(getGraphEdgesPageV2).mockReset();
    vi.mocked(getGraphEdgesPageV2)
      .mockResolvedValueOnce({
        edges: [{ source: 1, target: 2, weight: 0.8 }],
        next_cursor: 10,
        version: 'v1',
      })
      .mockResolvedValueOnce({
        edges: [{ source: 2, target: 3, weight: 0.7 }],
        next_cursor: null,
        version: 'v1',
      });
  });

  it('requests v2 graph without inline edges by default', async () => {
    await getGraphData();
    expect(getGraphDataV2).toHaveBeenCalledWith({ version: 'active', include_edges: false });
  });

  it('loads graph edges through paged v2 API', async () => {
    const edges = await getGraphEdges({ max_nodes: 10 });
    expect(edges).toHaveLength(2);
  });
});

import { describe, expect, it } from 'vitest';

import {
  GRAPH_2D_COLORS,
  calculateNodeRadius,
  computeConvexHull,
  getNodeId,
} from '../graph2dUtils';

describe('graph2dUtils', () => {
  it('extracts numeric ids from node references', () => {
    expect(getNodeId(7)).toBe(7);
    expect(getNodeId({ id: 9 })).toBe(9);
  });

  it('keeps node radius within expected bounds', () => {
    expect(calculateNodeRadius(0)).toBeGreaterThanOrEqual(5);
    expect(calculateNodeRadius(10_000)).toBeLessThanOrEqual(30);
    expect(calculateNodeRadius(100)).toBeGreaterThan(calculateNodeRadius(1));
  });

  it('computes a convex hull around a point cloud', () => {
    const hull = computeConvexHull([
      { x: 0, y: 0 },
      { x: 2, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 2 },
      { x: 2, y: 2 },
    ]);

    expect(hull).toHaveLength(4);
    expect(hull).toEqual(
      expect.arrayContaining([
        { x: 0, y: 0 },
        { x: 2, y: 0 },
        { x: 0, y: 2 },
        { x: 2, y: 2 },
      ])
    );
  });

  it('exports the shared graph color palette', () => {
    expect(GRAPH_2D_COLORS.NODE_DEFAULT).toBe('#6B7280');
    expect(GRAPH_2D_COLORS.LINK_ACTIVE).toContain('139, 92, 246');
  });
});

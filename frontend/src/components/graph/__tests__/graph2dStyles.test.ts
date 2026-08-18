import { describe, expect, it } from 'vitest';

import { GRAPH_2D_COLORS as COLORS } from '../graph2dUtils';
import {
  GHOST_LINK_COLOR,
  GHOST_NODE_COLOR,
  resolveLinkColor,
  resolveLinkWidth,
  resolveNodeColor,
} from '../graph2dStyles';
import type { ProcessedLink, ProcessedNode } from '../graph2dTypes';

const node = (id: number, cluster_id: number | null = 1, color = '#abcdef') =>
  ({ id, cluster_id, color }) as Pick<ProcessedNode, 'id' | 'cluster_id' | 'color'>;

const link = (source: number, target: number) =>
  ({ source, target }) as Pick<ProcessedLink, 'source' | 'target'>;

const baseNodeContext = {
  selectedNodeId: undefined,
  activeHoverNode: null,
  hoverNeighbors: new Set<number>(),
  visibleNodeIds: new Set([1, 2, 3]),
};

describe('resolveNodeColor', () => {
  it('gives the selected node priority over every other rule', () => {
    // Selected must win even when filtered out and not hovered.
    expect(
      resolveNodeColor(node(9), {
        ...baseNodeContext,
        selectedNodeId: 9,
        visibleNodeIds: new Set<number>(),
      })
    ).toBe(COLORS.NODE_SELECTED);
  });

  it('colours the hovered node', () => {
    expect(
      resolveNodeColor(node(2), { ...baseNodeContext, activeHoverNode: node(2) })
    ).toBe(COLORS.NODE_HOVER);
  });

  it('ghosts a node filtered out of the current view', () => {
    expect(
      resolveNodeColor(node(9), { ...baseNodeContext, visibleNodeIds: new Set([1]) })
    ).toBe(GHOST_NODE_COLOR);
  });

  it('uses the cluster colour when nothing is hovered', () => {
    expect(resolveNodeColor(node(1, 1, '#123456'), baseNodeContext)).toBe('#123456');
  });

  it('falls back to the default colour when a node has none', () => {
    expect(resolveNodeColor(node(1, 1, ''), baseNodeContext)).toBe(COLORS.NODE_DEFAULT);
  });

  it('highlights neighbours of the hovered node', () => {
    expect(
      resolveNodeColor(node(2), {
        ...baseNodeContext,
        activeHoverNode: node(1),
        hoverNeighbors: new Set([2]),
      })
    ).toBe(COLORS.NODE_NEIGHBOR);
  });

  it('keeps same-cluster nodes at their cluster colour while hovering', () => {
    expect(
      resolveNodeColor(node(2, 7, '#abcabc'), {
        ...baseNodeContext,
        activeHoverNode: node(1, 7),
      })
    ).toBe('#abcabc');
  });

  it('dims unrelated nodes while hovering', () => {
    expect(
      resolveNodeColor(node(2, 8), { ...baseNodeContext, activeHoverNode: node(1, 7) })
    ).toBe(COLORS.NODE_DIM);
  });

  it('does not treat a null cluster as matching the hovered null cluster', () => {
    // `cluster_id == null` must not group every unclustered node together.
    expect(
      resolveNodeColor(node(2, null), { ...baseNodeContext, activeHoverNode: node(1, null) })
    ).toBe(COLORS.NODE_DIM);
  });
});

const baseLinkContext = {
  showTrajectories: true,
  activeHoverNodeId: undefined,
  visibleNodeIds: new Set([1, 2, 3]),
  selectedNodeId: undefined,
};

describe('resolveLinkColor', () => {
  it('hides links entirely when trajectories are off', () => {
    expect(resolveLinkColor(link(1, 2), { ...baseLinkContext, showTrajectories: false })).toBe(
      'rgba(0,0,0,0)'
    );
  });

  it('ghosts a link with a filtered-out endpoint', () => {
    expect(
      resolveLinkColor(link(1, 9), { ...baseLinkContext, visibleNodeIds: new Set([1]) })
    ).toBe(GHOST_LINK_COLOR);
  });

  it('treats the selected node as visible even when filtered out', () => {
    expect(
      resolveLinkColor(link(1, 9), {
        ...baseLinkContext,
        visibleNodeIds: new Set([1]),
        selectedNodeId: 9,
      })
    ).toBe(COLORS.LINK_DEFAULT);
  });

  it('highlights links touching the hovered node', () => {
    expect(resolveLinkColor(link(1, 2), { ...baseLinkContext, activeHoverNodeId: 2 })).toBe(
      COLORS.LINK_ACTIVE
    );
  });

  it('dims links not touching the hovered node', () => {
    expect(resolveLinkColor(link(1, 2), { ...baseLinkContext, activeHoverNodeId: 3 })).toBe(
      COLORS.LINK_DIM
    );
  });

  it('resolves object endpoints, which force-graph substitutes after simulation', () => {
    const objectLink = {
      source: { id: 1 },
      target: { id: 2 },
    } as unknown as Pick<ProcessedLink, 'source' | 'target'>;

    expect(resolveLinkColor(objectLink, { ...baseLinkContext, activeHoverNodeId: 1 })).toBe(
      COLORS.LINK_ACTIVE
    );
  });
});

describe('resolveLinkWidth', () => {
  it('is zero when trajectories are off', () => {
    expect(resolveLinkWidth(link(1, 2), { ...baseLinkContext, showTrajectories: false })).toBe(0);
  });

  it('is hairline for a link with a filtered-out endpoint', () => {
    expect(
      resolveLinkWidth(link(1, 9), { ...baseLinkContext, visibleNodeIds: new Set([1]) })
    ).toBe(0.2);
  });

  it('is the default width with no hover', () => {
    expect(resolveLinkWidth(link(1, 2), baseLinkContext)).toBe(1);
  });

  it('thickens links touching the hovered node', () => {
    expect(resolveLinkWidth(link(1, 2), { ...baseLinkContext, activeHoverNodeId: 1 })).toBe(2);
  });

  it('thins links away from the hovered node', () => {
    expect(resolveLinkWidth(link(1, 2), { ...baseLinkContext, activeHoverNodeId: 3 })).toBe(0.5);
  });
});

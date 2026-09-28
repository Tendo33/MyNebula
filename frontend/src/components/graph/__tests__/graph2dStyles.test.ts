import { describe, expect, it } from 'vitest';

import { GRAPH_2D_COLORS as COLORS } from '../graph2dUtils';
import {
  GHOST_LINK_COLOR,
  GHOST_NODE_COLOR,
  resolveLinkColor,
  resolveLinkWidth,
  resolveNodeColor,
  resolveNodeDimmed,
  shouldShowNodeLabel,
} from '../graph2dStyles';
import type { ProcessedLink, ProcessedNode } from '../graph2dTypes';

const node = (id: number, cluster_id: number | null = 1, color = '#abcdef') =>
  ({ id, cluster_id, color }) as Pick<ProcessedNode, 'id' | 'cluster_id' | 'color'>;

const link = (source: number, target: number) =>
  ({ source, target }) as Pick<ProcessedLink, 'source' | 'target'>;

const baseNodeContext = {
  selectedNodeId: undefined,
  selectedClusterId: undefined,
  activeHoverNode: null,
  selectedNeighbors: new Set<number>(),
  visibleNodeIds: new Set([1, 2, 3]),
};

describe('resolveNodeColor', () => {
  it('keeps a selected node in its cluster hue even when filtered out', () => {
    expect(
      resolveNodeColor(node(9, 4, '#abcdef'), {
        ...baseNodeContext,
        selectedNodeId: 9,
        selectedClusterId: 4,
        visibleNodeIds: new Set<number>(),
      })
    ).toBe('#abcdef');
  });

  it('uses the selection colour when the selected node has no cluster hue', () => {
    expect(
      resolveNodeColor(node(9, null, ''), {
        ...baseNodeContext,
        selectedNodeId: 9,
        visibleNodeIds: new Set<number>(),
      })
    ).toBe(COLORS.NODE_SELECTED);
  });

  it('colours a hovered node with its cluster hue', () => {
    expect(
      resolveNodeColor(node(2, 7, '#445566'), { ...baseNodeContext, activeHoverNode: node(2, 7) })
    ).toBe('#445566');
  });

  it('uses the hover colour when the hovered node has no cluster hue', () => {
    expect(
      resolveNodeColor(node(2, null, ''), { ...baseNodeContext, activeHoverNode: node(2, null) })
    ).toBe(COLORS.NODE_HOVER);
  });

  it('ghosts a node filtered out of the current view', () => {
    expect(
      resolveNodeColor(node(9), { ...baseNodeContext, visibleNodeIds: new Set([1]) })
    ).toBe(GHOST_NODE_COLOR);
  });

  it('stays neutral when nothing is hovered or selected', () => {
    expect(resolveNodeColor(node(1, 1, '#123456'), baseNodeContext)).toBe(COLORS.NODE_DEFAULT);
  });

  it('falls back to the default colour when a node has none', () => {
    expect(resolveNodeColor(node(1, 1, ''), baseNodeContext)).toBe(COLORS.NODE_DEFAULT);
  });

  it('keeps an immediate neighbor of the selection readable outside the cluster', () => {
    expect(
      resolveNodeColor(node(2, 8, '#ff00aa'), {
        ...baseNodeContext,
        selectedNodeId: 1,
        selectedClusterId: 7,
        selectedNeighbors: new Set([2]),
      })
    ).toBe(COLORS.NODE_DEFAULT);
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

  it('gives the selected cluster its hue and dims every other cluster', () => {
    const context = {
      ...baseNodeContext,
      selectedNodeId: 1,
      selectedClusterId: 7,
    };

    expect(resolveNodeColor(node(2, 7, '#abcabc'), context)).toBe('#abcabc');
    expect(resolveNodeColor(node(3, 8, '#111111'), context)).toBe(COLORS.NODE_DIM);
    expect(resolveNodeDimmed(node(2, 7), context)).toBe(false);
    expect(resolveNodeDimmed(node(3, 8), context)).toBe(true);
  });
});

describe('shouldShowNodeLabel', () => {
  it('labels only the selection, its immediate neighbors, and the hovered node', () => {
    const context = {
      ...baseNodeContext,
      selectedNodeId: 1,
      selectedClusterId: 7,
      activeHoverNode: node(4, 8),
      selectedNeighbors: new Set([2]),
    };

    expect(shouldShowNodeLabel(1, context, true)).toBe(true);
    expect(shouldShowNodeLabel(2, context, true)).toBe(true);
    expect(shouldShowNodeLabel(4, context, true)).toBe(true);
    expect(shouldShowNodeLabel(3, context, true)).toBe(false);
    expect(shouldShowNodeLabel(2, context, false)).toBe(false);
    expect(shouldShowNodeLabel(1, context, false)).toBe(true);
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

  it('does not ghost a link to a filtered-out selected node, but fades it outside the focus cluster', () => {
    expect(
      resolveLinkColor(link(1, 9), {
        ...baseLinkContext,
        visibleNodeIds: new Set([1]),
        selectedNodeId: 9,
        clusterByNodeId: new Map<number, number | null>([
          [1, 3],
          [9, 4],
        ]),
      })
    ).toBe(COLORS.LINK_DIM);
  });

  it('keeps edges inside the focused cluster and fades edges that leave it', () => {
    const clusterByNodeId = new Map<number, number | null>([
      [1, 7],
      [2, 7],
      [3, 8],
    ]);
    const context = {
      ...baseLinkContext,
      selectedNodeId: 1,
      selectedClusterId: 7,
      clusterByNodeId,
    };

    expect(resolveLinkColor(link(2, 1), context)).toBe(COLORS.LINK_DEFAULT);
    expect(resolveLinkColor(link(1, 3), context)).toBe(COLORS.LINK_DIM);
  });

  it('highlights the edge from the selection to an immediate neighbor', () => {
    expect(
      resolveLinkColor(link(1, 2), {
        ...baseLinkContext,
        selectedNodeId: 1,
        selectedClusterId: 7,
        selectedNeighbors: new Set([2]),
        clusterByNodeId: new Map<number, number | null>([
          [1, 7],
          [2, 9],
        ]),
      })
    ).toBe(COLORS.LINK_ACTIVE);
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

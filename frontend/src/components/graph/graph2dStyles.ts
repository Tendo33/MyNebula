import { GRAPH_2D_COLORS as COLORS, getNodeId } from './graph2dUtils';
import type { ProcessedLink, ProcessedNode } from './graph2dTypes';

/**
 * Pure colour and width resolution for the graph canvas.
 *
 * Extracted from `Graph2D` so the visual rules are directly testable: inside
 * the component they closed over hover/selection state and could only be
 * exercised by rendering a canvas.
 */

export interface NodeColorContext {
  selectedNodeId: number | undefined;
  activeHoverNode: Pick<ProcessedNode, 'id' | 'cluster_id'> | null;
  hoverNeighbors: Set<number>;
  visibleNodeIds: Set<number>;
}

export const GHOST_NODE_COLOR = 'rgba(200, 200, 200, 0.4)';
export const GHOST_LINK_COLOR = 'rgba(200, 200, 200, 0.05)';

export const resolveNodeColor = (
  node: Pick<ProcessedNode, 'id' | 'cluster_id' | 'color'>,
  { selectedNodeId, activeHoverNode, hoverNeighbors, visibleNodeIds }: NodeColorContext
): string => {
  // Selected node ALWAYS visible
  if (selectedNodeId !== undefined && selectedNodeId === node.id) {
    return COLORS.NODE_SELECTED;
  }

  // Hovered node
  if (activeHoverNode && node.id === activeHoverNode.id) {
    return COLORS.NODE_HOVER;
  }

  // Ghost mode for filtered out nodes.
  // Dim gray if it doesn't have an avatar. If it has an avatar, the
  // transparency is handled by globalAlpha in the node painter.
  if (!visibleNodeIds.has(node.id)) {
    return GHOST_NODE_COLOR;
  }

  // No hover - use cluster color
  if (!activeHoverNode) {
    return node.color || COLORS.NODE_DEFAULT;
  }

  // Neighbor of hovered node
  if (hoverNeighbors.has(node.id)) {
    return COLORS.NODE_NEIGHBOR;
  }

  // Same cluster as hovered
  if (activeHoverNode.cluster_id != null && node.cluster_id === activeHoverNode.cluster_id) {
    return node.color || COLORS.NODE_DEFAULT;
  }

  // Dim other nodes
  return COLORS.NODE_DIM;
};

export interface LinkStyleContext {
  showTrajectories: boolean;
  activeHoverNodeId: number | undefined;
  visibleNodeIds: Set<number>;
  selectedNodeId: number | undefined;
}

const linkEndpointVisibility = (
  link: Pick<ProcessedLink, 'source' | 'target'>,
  { visibleNodeIds, selectedNodeId }: Pick<LinkStyleContext, 'visibleNodeIds' | 'selectedNodeId'>
) => {
  const sourceId = getNodeId(link.source);
  const targetId = getNodeId(link.target);
  return {
    sourceId,
    targetId,
    bothVisible:
      (visibleNodeIds.has(sourceId) || selectedNodeId === sourceId) &&
      (visibleNodeIds.has(targetId) || selectedNodeId === targetId),
  };
};

export const resolveLinkColor = (
  link: Pick<ProcessedLink, 'source' | 'target'>,
  context: LinkStyleContext
): string => {
  if (!context.showTrajectories) return 'rgba(0,0,0,0)';

  const { sourceId, targetId, bothVisible } = linkEndpointVisibility(link, context);
  if (!bothVisible) return GHOST_LINK_COLOR;

  if (context.activeHoverNodeId === undefined) return COLORS.LINK_DEFAULT;

  // Highlight links connected to hovered node
  if (sourceId === context.activeHoverNodeId || targetId === context.activeHoverNodeId) {
    return COLORS.LINK_ACTIVE;
  }

  return COLORS.LINK_DIM;
};

export const resolveLinkWidth = (
  link: Pick<ProcessedLink, 'source' | 'target'>,
  context: LinkStyleContext
): number => {
  if (!context.showTrajectories) return 0;

  const { sourceId, targetId, bothVisible } = linkEndpointVisibility(link, context);
  if (!bothVisible) return 0.2;

  if (context.activeHoverNodeId === undefined) return 1;

  if (sourceId === context.activeHoverNodeId || targetId === context.activeHoverNodeId) {
    return 2;
  }

  return 0.5;
};

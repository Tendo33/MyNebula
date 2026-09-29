import { GRAPH_2D_COLORS as COLORS, getNodeId } from './graph2dUtils';
import type { ProcessedLink, ProcessedNode } from './graph2dTypes';

/**
 * Pure colour and width resolution for the graph canvas.
 *
 * Cluster hue is reserved for the hovered or selected cluster. Everything else
 * stays neutral, and nodes outside that cluster are dimmed by the painter.
 */

export interface NodeColorContext {
  selectedNodeId: number | undefined;
  selectedClusterId: number | null | undefined;
  activeHoverNode: Pick<ProcessedNode, 'id' | 'cluster_id'> | null;
  selectedNeighbors: Set<number>;
  visibleNodeIds: Set<number>;
}

export const GHOST_NODE_COLOR = 'rgba(200, 200, 200, 0.4)';
export const GHOST_LINK_COLOR = 'rgba(200, 200, 200, 0.05)';

const EMPTY_NEIGHBORS: ReadonlySet<number> = new Set();
const EMPTY_CLUSTERS: ReadonlyMap<number, number | null> = new Map();

/** Keep a sparse, weight-ranked backbone so every connected node has a visible route. */
export const buildOverviewLinks = (links: ProcessedLink[]): ReadonlySet<ProcessedLink> => {
  const covered = new Set<number>();
  const overview = new Set<ProcessedLink>();
  for (const link of [...links].sort((left, right) => right.weight - left.weight)) {
    const sourceId = getNodeId(link.source);
    const targetId = getNodeId(link.target);
    if (covered.has(sourceId) && covered.has(targetId)) continue;
    overview.add(link);
    covered.add(sourceId);
    covered.add(targetId);
  }
  return overview;
};

export const collectFocusClusterIds = (
  selectedClusterId: number | null | undefined,
  hoverClusterId: number | null | undefined,
): Set<number> => {
  const ids = new Set<number>();
  if (selectedClusterId != null) ids.add(selectedClusterId);
  if (hoverClusterId != null) ids.add(hoverClusterId);
  return ids;
};

export const nodeIsInFocusCluster = (
  clusterId: number | null | undefined,
  selectedClusterId: number | null | undefined,
  hoverClusterId: number | null | undefined,
): boolean =>
  clusterId != null &&
  (clusterId === selectedClusterId || clusterId === hoverClusterId);

const focusIsActive = (context: NodeColorContext): boolean =>
  context.selectedNodeId !== undefined || context.activeHoverNode != null;

export const resolveNodeColor = (
  node: Pick<ProcessedNode, 'id' | 'cluster_id' | 'color'>,
  context: NodeColorContext
): string => {
  const isSelected = context.selectedNodeId !== undefined && context.selectedNodeId === node.id;
  const isHovered = context.activeHoverNode != null && context.activeHoverNode.id === node.id;
  const filteredOut = !context.visibleNodeIds.has(node.id);

  if (filteredOut && !isSelected && !isHovered) {
    return GHOST_NODE_COLOR;
  }

  if (isSelected) {
    if (node.cluster_id != null && node.color) return node.color;
    return COLORS.NODE_SELECTED;
  }

  if (isHovered) {
    if (node.cluster_id != null && node.color) return node.color;
    return COLORS.NODE_HOVER;
  }

  if (!focusIsActive(context)) {
    return COLORS.NODE_DEFAULT;
  }

  if (
    nodeIsInFocusCluster(
      node.cluster_id,
      context.selectedClusterId,
      context.activeHoverNode?.cluster_id,
    )
  ) {
    return node.color || COLORS.NODE_DEFAULT;
  }

  if (context.selectedNeighbors.has(node.id)) {
    return COLORS.NODE_DEFAULT;
  }

  return COLORS.NODE_DIM;
};

/** Visible nodes outside the hovered/selected cluster (and their labeled neighbors) recede. */
export const resolveNodeDimmed = (
  node: Pick<ProcessedNode, 'id' | 'cluster_id'>,
  context: NodeColorContext
): boolean => {
  if (!context.visibleNodeIds.has(node.id)) return false;
  if (!focusIsActive(context)) return false;
  if (context.selectedNodeId === node.id) return false;
  if (context.activeHoverNode != null && context.activeHoverNode.id === node.id) return false;
  if (context.selectedNeighbors.has(node.id)) return false;
  return !nodeIsInFocusCluster(
    node.cluster_id,
    context.selectedClusterId,
    context.activeHoverNode?.cluster_id,
  );
};

export const resolveShowClusterHue = (
  node: Pick<ProcessedNode, 'id' | 'cluster_id'>,
  context: NodeColorContext
): boolean =>
  context.visibleNodeIds.has(node.id) &&
  nodeIsInFocusCluster(
    node.cluster_id,
    context.selectedClusterId,
    context.activeHoverNode?.cluster_id,
  );

/** Labels: the selected node, its immediate neighbors, and the hovered node. */
export const shouldShowNodeLabel = (
  nodeId: number,
  context: NodeColorContext,
  isVisible: boolean
): boolean => {
  if (nodeId === context.selectedNodeId) return true;
  if (context.activeHoverNode != null && nodeId === context.activeHoverNode.id) return true;
  return isVisible && context.selectedNeighbors.has(nodeId);
};

export interface LinkStyleContext {
  showTrajectories: boolean;
  activeHoverNodeId: number | undefined;
  visibleNodeIds: Set<number>;
  selectedNodeId: number | undefined;
  selectedClusterId?: number | null;
  hoverClusterId?: number | null;
  selectedNeighbors?: ReadonlySet<number>;
  clusterByNodeId?: ReadonlyMap<number, number | null>;
  overviewLinks?: ReadonlySet<ProcessedLink>;
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

const endpointEmphasized = (nodeId: number, context: LinkStyleContext): boolean => {
  const focusActive =
    context.selectedNodeId !== undefined || context.activeHoverNodeId !== undefined;
  if (!focusActive) return true;
  if (nodeId === context.selectedNodeId || nodeId === context.activeHoverNodeId) return true;
  if ((context.selectedNeighbors ?? EMPTY_NEIGHBORS).has(nodeId)) return true;
  const clusterId = (context.clusterByNodeId ?? EMPTY_CLUSTERS).get(nodeId);
  return nodeIsInFocusCluster(clusterId, context.selectedClusterId, context.hoverClusterId);
};

const classifyLink = (
  sourceId: number,
  targetId: number,
  context: LinkStyleContext
) => {
  const neighbors = context.selectedNeighbors ?? EMPTY_NEIGHBORS;
  const touchesHover =
    context.activeHoverNodeId !== undefined &&
    (sourceId === context.activeHoverNodeId || targetId === context.activeHoverNodeId);
  const touchesSelectedNeighbor =
    context.selectedNodeId !== undefined &&
    (sourceId === context.selectedNodeId || targetId === context.selectedNodeId) &&
    (neighbors.has(sourceId) || neighbors.has(targetId));
  return {
    sourceOn: endpointEmphasized(sourceId, context),
    targetOn: endpointEmphasized(targetId, context),
    touchesHover,
    touchesSelectedNeighbor,
  };
};

export const resolveLinkColor = (
  link: Pick<ProcessedLink, 'source' | 'target'>,
  context: LinkStyleContext
): string => {
  if (!context.showTrajectories) return 'rgba(0,0,0,0)';

  const { sourceId, targetId, bothVisible } = linkEndpointVisibility(link, context);
  if (!bothVisible) return GHOST_LINK_COLOR;
  if (
    context.selectedNodeId === undefined &&
    context.activeHoverNodeId === undefined &&
    context.overviewLinks &&
    !context.overviewLinks.has(link as ProcessedLink)
  ) return 'rgba(0,0,0,0)';

  const { sourceOn, targetOn, touchesHover, touchesSelectedNeighbor } = classifyLink(
    sourceId,
    targetId,
    context
  );
  if (touchesHover || touchesSelectedNeighbor) return COLORS.LINK_ACTIVE;
  if (sourceOn && targetOn) return COLORS.LINK_DEFAULT;
  return COLORS.LINK_DIM;
};

export const resolveLinkWidth = (
  link: Pick<ProcessedLink, 'source' | 'target'>,
  context: LinkStyleContext
): number => {
  if (!context.showTrajectories) return 0;

  const { sourceId, targetId, bothVisible } = linkEndpointVisibility(link, context);
  if (!bothVisible) return 0.2;
  if (
    context.selectedNodeId === undefined &&
    context.activeHoverNodeId === undefined &&
    context.overviewLinks &&
    !context.overviewLinks.has(link as ProcessedLink)
  ) return 0;

  const { sourceOn, targetOn, touchesHover, touchesSelectedNeighbor } = classifyLink(
    sourceId,
    targetId,
    context
  );
  if (touchesHover || touchesSelectedNeighbor) return 2;
  if (sourceOn && targetOn) return 1;
  return 0.5;
};

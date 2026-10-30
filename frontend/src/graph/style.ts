/**
 * Pure colour, width, and visibility rules for the graph canvas.
 *
 * Selecting a repository lights it and the repositories it links to; hovering
 * with nothing selected previews the hovered cluster. Everything else stays
 * neutral and dims.
 */

export const COLORS = {
  NODE_DEFAULT: '#8f8f8f',
  NODE_HOVER: '#171717',
  NODE_SELECTED: '#0070f3',
  NODE_DIM: 'rgba(143, 143, 143, 0.3)',
  LINK_DEFAULT: 'rgba(143, 143, 143, 0.35)',
  LINK_ACTIVE: 'rgba(0, 112, 243, 0.55)',
  LINK_DIM: 'rgba(143, 143, 143, 0.1)',
  LABEL_BG: 'rgba(255, 255, 255, 0.92)',
  LABEL_TEXT: '#171717',
  GHOST_NODE: 'rgba(200, 200, 200, 0.4)',
  GHOST_LINK: 'rgba(200, 200, 200, 0.05)',
  TRANSPARENT: 'rgba(0,0,0,0)',
} as const;

/** Nodes outside the current focus. Filtered ghosts stay at 0.25. */
export const UNFOCUSED_NODE_ALPHA = 0.18;

const NODE_BASE_SIZE = 5;
const NODE_MAX_SIZE = 30;

export const nodeRadius = (stars: number): number =>
  Math.min(Math.max(Math.log10(Math.max(stars, 1) + 1) * NODE_BASE_SIZE, NODE_BASE_SIZE), NODE_MAX_SIZE);

export interface StyleNode {
  id: number;
  cluster_id: number | null;
  color: string;
}

export interface StyleLink {
  sourceId: number;
  targetId: number;
  weight: number;
}

export interface FocusContext {
  selectedId: number | undefined;
  hover: Pick<StyleNode, 'id' | 'cluster_id'> | null;
  neighbors: ReadonlySet<number>;
  visible: ReadonlySet<number>;
}

/** Only a hover with nothing selected focuses a whole cluster. */
export const focusClusterId = (ctx: FocusContext): number | null =>
  ctx.selectedId === undefined ? (ctx.hover?.cluster_id ?? null) : null;

const inFocusCluster = (clusterId: number | null | undefined, ctx: FocusContext): boolean =>
  clusterId != null && clusterId === focusClusterId(ctx);

const focusActive = (ctx: FocusContext): boolean => ctx.selectedId !== undefined || ctx.hover != null;

export const nodeColor = (node: StyleNode, ctx: FocusContext): string => {
  const selected = ctx.selectedId === node.id;
  const hovered = ctx.hover?.id === node.id;
  if (!ctx.visible.has(node.id) && !selected && !hovered) return COLORS.GHOST_NODE;
  if (selected) return node.cluster_id != null && node.color ? node.color : COLORS.NODE_SELECTED;
  if (hovered) return node.cluster_id != null && node.color ? node.color : COLORS.NODE_HOVER;
  if (!focusActive(ctx)) return COLORS.NODE_DEFAULT;
  if (inFocusCluster(node.cluster_id, ctx)) return node.color || COLORS.NODE_DEFAULT;
  if (ctx.neighbors.has(node.id)) return node.color || COLORS.NODE_DEFAULT;
  return COLORS.NODE_DIM;
};

/** Visible nodes outside the focused cluster or the selection's links recede. */
export const nodeDimmed = (node: StyleNode, ctx: FocusContext): boolean => {
  if (!ctx.visible.has(node.id) || !focusActive(ctx)) return false;
  if (ctx.selectedId === node.id || ctx.hover?.id === node.id || ctx.neighbors.has(node.id)) return false;
  return !inFocusCluster(node.cluster_id, ctx);
};

export const showClusterHue = (node: StyleNode, ctx: FocusContext): boolean =>
  ctx.visible.has(node.id) && (inFocusCluster(node.cluster_id, ctx) || ctx.neighbors.has(node.id));

/** Labels: the selected node, its immediate neighbours, and the hovered node. */
export const showLabel = (id: number, ctx: FocusContext): boolean =>
  id === ctx.selectedId || id === ctx.hover?.id || (ctx.visible.has(id) && ctx.neighbors.has(id));

/** Sparse, weight-ranked backbone so every connected node has a visible route. */
export const buildOverviewLinks = <L extends StyleLink>(links: L[]): Set<L> => {
  const covered = new Set<number>();
  const overview = new Set<L>();
  for (const link of [...links].sort((a, b) => b.weight - a.weight)) {
    if (covered.has(link.sourceId) && covered.has(link.targetId)) continue;
    overview.add(link);
    covered.add(link.sourceId);
    covered.add(link.targetId);
  }
  return overview;
};

export interface LinkContext extends FocusContext {
  showTrajectories: boolean;
  clusterOf: ReadonlyMap<number, number | null>;
}

type LinkClass = 'hidden' | 'ghost' | 'active' | 'default' | 'dim';

/** `inOverview`: whether the link belongs to the unfocused backbone. */
export const classifyLink = (link: StyleLink, ctx: LinkContext, inOverview: boolean): LinkClass => {
  if (!ctx.showTrajectories) return 'hidden';
  const { sourceId: source, targetId: target } = link;
  const endpointVisible = (id: number) => ctx.visible.has(id) || ctx.selectedId === id;
  if (!endpointVisible(source) || !endpointVisible(target)) return 'ghost';
  const focus = focusActive(ctx);
  if (!focus && !inOverview) return 'hidden';

  const hoverId = ctx.hover?.id;
  const touches = (id: number | undefined) => id !== undefined && (source === id || target === id);
  if (touches(hoverId)) return 'active';
  if (touches(ctx.selectedId) && (ctx.neighbors.has(source) || ctx.neighbors.has(target))) {
    return 'active';
  }
  const emphasized = (id: number) =>
    !focus ||
    id === ctx.selectedId ||
    id === hoverId ||
    ctx.neighbors.has(id) ||
    inFocusCluster(ctx.clusterOf.get(id), ctx);
  return emphasized(source) && emphasized(target) ? 'default' : 'dim';
};

export const LINK_STYLE: Record<LinkClass, { color: string; width: number }> = {
  hidden: { color: COLORS.TRANSPARENT, width: 0 },
  ghost: { color: COLORS.GHOST_LINK, width: 0.2 },
  active: { color: COLORS.LINK_ACTIVE, width: 2 },
  default: { color: COLORS.LINK_DEFAULT, width: 1 },
  dim: { color: COLORS.LINK_DIM, width: 0.5 },
};

export const convexHull = (input: { x: number; y: number }[]): { x: number; y: number }[] => {
  if (input.length < 3) return input;
  const points = [...input].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: typeof points = [];
  for (const p of points) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: typeof points = [];
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const p = points[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
};

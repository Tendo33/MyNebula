import type { ClusterInfo } from '../../types';
import {
  GRAPH_2D_COLORS as COLORS,
  calculateNodeRadius,
  computeConvexHull,
} from './graph2dUtils';
import type { HullCache, ImageCache, ProcessedNode } from './graph2dTypes';

const CANVAS_FONT_FAMILY = '"Geist Variable", Geist, sans-serif';

/** Nodes outside the hovered or selected cluster. Filtered ghosts stay at 0.25. */
export const UNFOCUSED_NODE_ALPHA = 0.18;

/**
 * Pure canvas painting for the graph.
 *
 * Every dependency is an explicit parameter, so these can be exercised against
 * a fake 2D context. Inside `Graph2D` they were `useCallback`s closing over
 * component state and were only reachable by rendering a real canvas — which is
 * exactly where graph visual regressions originate.
 */

export interface PaintNodeOptions {
  node: ProcessedNode;
  ctx: CanvasRenderingContext2D;
  globalScale: number;
  color: string;
  isVisible: boolean;
  isSelected: boolean;
  isHovered: boolean;
  isDimmed: boolean;
  showLabel: boolean;
  showClusterHue: boolean;
  /** Owner avatars. Off keeps the same layout and draws ink circles. */
  hqRendering?: boolean;
  imageCache: ImageCache;
  onAvatarLoaded: () => void;
}

export const paintNodeOnCanvas = ({
  node,
  ctx,
  globalScale,
  color,
  isVisible,
  isSelected,
  isHovered,
  isDimmed,
  showLabel,
  showClusterHue,
  hqRendering = true,
  imageCache,
  onAvatarLoaded,
}: PaintNodeOptions): void => {
  const { x, y, name, stargazers_count, owner_avatar_url } = node;
  if (x === undefined || y === undefined) return;

  const radius = calculateNodeRadius(stargazers_count);

  ctx.save();

  if (!isVisible && !isSelected && !isHovered) {
    ctx.globalAlpha = 0.25;
  } else if (isDimmed && !isSelected && !isHovered) {
    ctx.globalAlpha = UNFOCUSED_NODE_ALPHA;
  }

  // Avatars are the high-quality pass. The layout stays the same without them.
  let avatarDrawn = false;
  if (hqRendering && owner_avatar_url) {
    const cached = imageCache.get(owner_avatar_url);

    if (cached === undefined) {
      // Start loading image
      imageCache.set(owner_avatar_url, 'loading');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.decoding = 'async';
      img.onload = () => {
        imageCache.set(owner_avatar_url, img);
        onAvatarLoaded();
      };
      img.onerror = () => {
        imageCache.set(owner_avatar_url, 'error');
      };
      img.src = owner_avatar_url;
    } else if (cached instanceof HTMLImageElement) {
      // Draw cached image in circular clip
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.clip();
      ctx.drawImage(cached, x - radius, y - radius, radius * 2, radius * 2);
      ctx.restore();
      avatarDrawn = true;
    }
  }

  if (!avatarDrawn) {
    ctx.beginPath();
    ctx.arc(
      x,
      y,
      !isVisible && !isSelected && !isHovered ? radius * 0.8 : radius,
      0,
      2 * Math.PI
    );
    ctx.fillStyle = color;
    ctx.fill();
  }

  if (showClusterHue && !isDimmed && !isSelected && !isHovered) {
    ctx.beginPath();
    ctx.arc(x, y, radius + 1.5 / globalScale, 0, 2 * Math.PI);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25 / globalScale;
    ctx.stroke();
  }

  if (isHovered || isSelected) {
    ctx.beginPath();
    ctx.arc(x, y, radius + 1.5 / globalScale, 0, 2 * Math.PI);
    ctx.strokeStyle = isSelected ? COLORS.NODE_SELECTED : COLORS.NODE_HOVER;
    ctx.lineWidth = 1.5 / globalScale;
    ctx.stroke();
  }

  const fontSize = Math.max(10 / globalScale, 8);

  if (showLabel) {
    const label = name;
    ctx.font = `500 ${fontSize}px ${CANVAS_FONT_FAMILY}`;
    const textWidth = ctx.measureText(label).width;
    const textHeight = fontSize;
    const padding = 3 / globalScale;
    const labelY = y + radius + fontSize + 2 / globalScale;

    // Background
    ctx.fillStyle = COLORS.LABEL_BG;
    ctx.fillRect(
      x - textWidth / 2 - padding,
      labelY - textHeight + 2 / globalScale,
      textWidth + padding * 2,
      textHeight + padding
    );

    // Text
    ctx.fillStyle = COLORS.LABEL_TEXT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x, labelY - textHeight / 2 + padding);
  }

  ctx.restore();
};

export const paintNodePointerArea = (
  node: ProcessedNode,
  color: string,
  ctx: CanvasRenderingContext2D
): void => {
  const { x, y, stargazers_count } = node;
  if (x === undefined || y === undefined) return;

  const radius = calculateNodeRadius(stargazers_count);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius + 5, 0, 2 * Math.PI); // Slightly larger for easier clicking
  ctx.fill();
};

export interface DrawClusterHullsOptions {
  nodes: ProcessedNode[];
  ctx: CanvasRenderingContext2D;
  globalScale: number;
  visibleNodeIds: Set<number>;
  clusterGroups: Map<number, ClusterInfo>;
  hullCache: HullCache;
  focusedClusterIds: ReadonlySet<number>;
}

/** Group visible, positioned nodes by cluster. Exported for direct testing. */
export const groupNodesByCluster = (
  nodes: ProcessedNode[],
  visibleNodeIds: Set<number>
): Map<number, ProcessedNode[]> => {
  const nodesByCluster = new Map<number, ProcessedNode[]>();
  nodes.forEach((node) => {
    // Only draw hull for visible nodes
    if (!visibleNodeIds.has(node.id)) return;
    if (node.cluster_id != null && node.x !== undefined && node.y !== undefined) {
      const group = nodesByCluster.get(node.cluster_id) || [];
      group.push(node);
      nodesByCluster.set(node.cluster_id, group);
    }
  });
  return nodesByCluster;
};

/** Position signature used to decide whether a cached hull is still valid. */
export const hullSignature = (points: { x: number; y: number }[]): string =>
  points.map((point) => `${Math.round(point.x)}:${Math.round(point.y)}`).join('|');

export const drawClusterHulls = ({
  nodes,
  ctx,
  globalScale,
  visibleNodeIds,
  clusterGroups,
  hullCache,
  focusedClusterIds,
}: DrawClusterHullsOptions): void => {
  const nodesByCluster = groupNodesByCluster(nodes, visibleNodeIds);

  nodesByCluster.forEach((clusterNodes, clusterId) => {
    if (!focusedClusterIds.has(clusterId)) return;
    if (clusterNodes.length < 3) return;

    const cluster = clusterGroups.get(clusterId);
    if (!cluster) return;

    const points = clusterNodes.map((n) => ({ x: n.x!, y: n.y! }));

    const signature = hullSignature(points);
    const cached = hullCache.get(clusterId);
    const hull = cached?.signature === signature ? cached.hull : computeConvexHull([...points]);
    if (!cached || cached.signature !== signature) {
      hullCache.set(clusterId, { signature, hull });
    }
    if (hull.length < 3) return;

    // Draw filled hull with cluster color
    ctx.beginPath();
    ctx.moveTo(hull[0].x, hull[0].y);
    for (let i = 1; i < hull.length; i++) {
      ctx.lineTo(hull[i].x, hull[i].y);
    }
    ctx.closePath();

    // Parse cluster color and add transparency
    const baseColor = cluster.color || '#808080';
    ctx.fillStyle = baseColor + '10'; // Very transparent
    ctx.fill();

    ctx.strokeStyle = baseColor + '30'; // Slightly more visible border
    ctx.lineWidth = 1 / globalScale;
    ctx.stroke();

    // Draw cluster label at center
    const centerX = clusterNodes.reduce((sum, n) => sum + n.x!, 0) / clusterNodes.length;
    const centerY = clusterNodes.reduce((sum, n) => sum + n.y!, 0) / clusterNodes.length;

    if (globalScale > 0.5 && cluster.name) {
      const fontSize = Math.max(14 / globalScale, 10);
      ctx.font = `600 ${fontSize}px ${CANVAS_FONT_FAMILY}`;
      ctx.fillStyle = baseColor + '60';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(cluster.name, centerX, centerY);
    }
  });
};

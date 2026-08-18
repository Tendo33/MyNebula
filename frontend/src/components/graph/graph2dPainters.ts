import type { ClusterInfo } from '../../types';
import {
  GRAPH_2D_COLORS as COLORS,
  calculateNodeRadius,
  computeConvexHull,
} from './graph2dUtils';
import type { HullCache, ImageCache, ProcessedNode } from './graph2dTypes';

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
  hqRendering: boolean;
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
  hqRendering,
  imageCache,
  onAvatarLoaded,
}: PaintNodeOptions): void => {
  const { x, y, name, stargazers_count, owner_avatar_url } = node;
  if (x === undefined || y === undefined) return;

  const radius = calculateNodeRadius(stargazers_count);

  ctx.save();

  // Apply transparency for filtered out ghost nodes
  if (!isVisible && !isSelected && !isHovered) {
    ctx.globalAlpha = 0.25;
  }

  // Try to draw avatar image
  let avatarDrawn = false;
  if (owner_avatar_url) {
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

  // Fallback: Draw solid color circle if no avatar
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
  } else if (color === COLORS.NODE_DIM && (isVisible || isSelected)) {
    // Keep avatar nodes visually dimmed when not in focus.
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.68)';
    ctx.fill();
    ctx.restore();
  }

  // Draw border for selected/hovered nodes
  if (isHovered || isSelected) {
    ctx.strokeStyle = isSelected ? COLORS.NODE_SELECTED : COLORS.NODE_HOVER;
    ctx.lineWidth = 2 / globalScale;
    ctx.stroke();

    // Draw outer glow if HQ rendering is enabled
    if (hqRendering) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 4 / globalScale, 0, 2 * Math.PI);
      ctx.strokeStyle = isSelected ? 'rgba(59, 130, 246, 0.3)' : 'rgba(139, 92, 246, 0.3)';
      ctx.lineWidth = 3 / globalScale;
      ctx.stroke();
    }
  }

  // Draw label
  const fontSize = Math.max(10 / globalScale, 8);
  // Hide label for ghost nodes unless hovered
  const showLabel = isHovered || isSelected || (isVisible && (globalScale > 2 || radius > 15));

  if (showLabel) {
    const label = name;
    ctx.font = `${fontSize}px Inter, system-ui, sans-serif`;
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
}: DrawClusterHullsOptions): void => {
  const nodesByCluster = groupNodesByCluster(nodes, visibleNodeIds);

  // Draw hull for each cluster with enough nodes
  nodesByCluster.forEach((clusterNodes, clusterId) => {
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
      ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.fillStyle = baseColor + '60';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(cluster.name, centerX, centerY);
    }
  });
};

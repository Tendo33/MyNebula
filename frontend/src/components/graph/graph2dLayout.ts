import type { ClusterInfo, GraphData, GraphEdge, GraphNode } from '../../types';
import { GRAPH_2D_COLORS as COLORS } from './graph2dUtils';
import {
  POSITION_SCALE,
  type ClusterCenter,
  type ClusterLayoutData,
  type ProcessedLink,
  type ProcessedNode,
} from './graph2dTypes';

/**
 * Pure derivation of force-graph input from snapshot payloads.
 *
 * These run off `rawData`, not `filteredData`, so the layout stays stable while
 * filters change.
 */

export const toProcessedNodes = (nodes: GraphNode[] | undefined): ProcessedNode[] => {
  if (!nodes?.length) return [];
  return nodes.map((n) => ({
    id: n.id,
    name: n.name,
    full_name: n.full_name,
    description: n.description,
    language: n.language,
    cluster_id: n.cluster_id,
    color: n.color || COLORS.NODE_DEFAULT,
    size: n.size,
    stargazers_count: n.stargazers_count,
    // Owner info for avatar display
    owner: n.owner,
    owner_avatar_url: n.owner_avatar_url,
    // AI-generated content
    ai_summary: n.ai_summary,
    ai_tags: n.ai_tags,
    // Use pre-computed positions if available (from clustering).
    // Non-finite coordinates would propagate NaN into the canvas transform.
    x: Number.isFinite(n.x) ? n.x * POSITION_SCALE : undefined,
    y: Number.isFinite(n.y) ? n.y * POSITION_SCALE : undefined,
  }));
};

export const toProcessedLinks = (edges: GraphEdge[] | undefined): ProcessedLink[] => {
  if (!edges) return [];
  return edges.map((e) => ({
    source: typeof e.source === 'object' ? e.source.id : e.source,
    target: typeof e.target === 'object' ? e.target.id : e.target,
    weight: e.weight,
  }));
};

/** Cluster centroids and membership, used by the custom d3 cluster force. */
export const buildClusterLayoutData = (nodes: ProcessedNode[]): ClusterLayoutData => {
  const centers = new Map<number, ClusterCenter>();
  const clusterNodes = new Map<number, ProcessedNode[]>();

  nodes.forEach((node) => {
    if (node.cluster_id == null || node.x === undefined || node.y === undefined) {
      return;
    }

    const clusterMembers = clusterNodes.get(node.cluster_id) ?? [];
    clusterMembers.push(node);
    clusterNodes.set(node.cluster_id, clusterMembers);

    const currentCenter = centers.get(node.cluster_id) ?? { x: 0, y: 0, count: 0 };
    currentCenter.x += node.x;
    currentCenter.y += node.y;
    currentCenter.count += 1;
    centers.set(node.cluster_id, currentCenter);
  });

  centers.forEach((center) => {
    center.x /= center.count;
    center.y /= center.count;
  });

  return { centers, clusterNodes };
};

export const buildClusterGroups = (rawData: GraphData | null): Map<number, ClusterInfo> => {
  const groups = new Map<number, ClusterInfo>();
  if (!rawData) return groups;
  rawData.clusters.forEach((cluster) => {
    groups.set(cluster.id, cluster);
  });
  return groups;
};

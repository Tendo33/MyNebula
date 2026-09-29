import { useEffect, useMemo, useRef } from 'react';
import { forceCollide, forceX, forceY } from 'd3-force';
import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d';

import { calculateNodeRadius } from '../graph2dUtils';
import {
  CENTER_PULL_STRENGTH,
  MIN_CLUSTER_DISTANCE,
  type ClusterLayoutData,
  type ProcessedNode,
  type RegisteredForce,
} from '../graph2dTypes';

/**
 * Custom d3 force that keeps cluster members compact while preventing cluster
 * centres from collapsing into each other.
 *
 * Exported separately from the hook so the repulsion and cohesion maths can be
 * exercised without a live force-graph instance.
 */
export const createClusterForce =
  (clusterLayoutData: ClusterLayoutData, forceScale = 1) =>
  (alpha: number): void => {
    if (alpha < 0.015) {
      return;
    }

    const centersArray = Array.from(clusterLayoutData.centers.entries());

    // Apply repulsion between cluster centers
    for (let i = 0; i < centersArray.length; i++) {
      for (let j = i + 1; j < centersArray.length; j++) {
        const [clusterId1, center1] = centersArray[i];
        const [clusterId2, center2] = centersArray[j];

        const dx = center2.x - center1.x;
        const dy = center2.y - center1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        // If clusters are too close, push them apart
        if (dist < MIN_CLUSTER_DISTANCE) {
          const force = ((MIN_CLUSTER_DISTANCE - dist) / dist) * alpha * 2 * forceScale;
          const fx = dx * force;
          const fy = dy * force;
          const cluster1Nodes = clusterLayoutData.clusterNodes.get(clusterId1) ?? [];
          const cluster2Nodes = clusterLayoutData.clusterNodes.get(clusterId2) ?? [];

          cluster1Nodes.forEach((node) => {
            node.vx = (node.vx || 0) - fx;
            node.vy = (node.vy || 0) - fy;
          });
          cluster2Nodes.forEach((node) => {
            node.vx = (node.vx || 0) + fx;
            node.vy = (node.vy || 0) + fy;
          });
        }
      }
    }

    clusterLayoutData.clusterNodes.forEach((clusterNodes, clusterId) => {
      const center = clusterLayoutData.centers.get(clusterId);
      if (!center) {
        return;
      }

      clusterNodes.forEach((node) => {
        if (node.x === undefined || node.y === undefined) {
          return;
        }

        const k = alpha * 0.2 * forceScale;
        node.vx = (node.vx || 0) + (center.x - node.x) * k;
        node.vy = (node.vy || 0) + (center.y - node.y) * k;
      });
    });
  };

/** Configure link, charge, centring, collision, and cluster forces. */
export const useGraphForces = ({
  graphRef,
  nodes,
  targetNodes,
  clusterLayoutData,
  layoutKey,
  forceScale,
  hasProjectedPositions,
  enabled,
}: {
  graphRef: React.MutableRefObject<ForceGraphMethods | undefined>;
  nodes: ProcessedNode[];
  targetNodes: ProcessedNode[];
  clusterLayoutData: ClusterLayoutData;
  layoutKey: string;
  forceScale: number;
  hasProjectedPositions: boolean;
  enabled: boolean;
}) => {
  const reheatedLayoutRef = useRef<{ key: string; nodes: ProcessedNode[] } | null>(null);
  const targets = useMemo(
    () => hasProjectedPositions
      ? new Map(targetNodes.map((node) => [node.id, { x: node.x!, y: node.y! }]))
      : null,
    [hasProjectedPositions, targetNodes]
  );

  useEffect(() => {
    if (!enabled || !graphRef.current || forceScale <= 0) return;

    const fg = graphRef.current;

    // Configure link force - increased distance for cross-cluster links
    fg.d3Force('link')
      ?.distance((link: LinkObject<NodeObject, LinkObject<NodeObject>>) => {
        if (targets) {
          const source = targets.get(Number(typeof link.source === 'object' ? link.source.id : link.source));
          const target = targets.get(Number(typeof link.target === 'object' ? link.target.id : link.target));
          if (source && target) return Math.hypot(target.x - source.x, target.y - source.y);
        }
        const sourceCluster = typeof link.source === 'object' ? link.source.cluster_id : null;
        const targetCluster = typeof link.target === 'object' ? link.target.cluster_id : null;
        // Same cluster: moderate distance, Different cluster: larger separation
        return sourceCluster === targetCluster ? 38 : 75;
      })
      .strength((link: LinkObject<NodeObject, LinkObject<NodeObject>>) => {
        const sourceCluster = typeof link.source === 'object' ? link.source.cluster_id : null;
        const targetCluster = typeof link.target === 'object' ? link.target.cluster_id : null;
        // Stronger links within same cluster, very weak for cross-cluster
        return targets ? 0.025 : (sourceCluster === targetCluster ? 0.7 : 0.05) * forceScale;
      });

    // Configure charge force (repulsion) - increased for more spacing
    fg.d3Force('charge')?.strength(targets ? 0 : -150 * forceScale).distanceMax(250);

    // Gentle pull toward the origin to avoid disconnected groups drifting far apart.
    fg.d3Force(
      'x',
      forceX<ProcessedNode>((node) => targets?.get(node.id)?.x ?? 0)
        .strength(targets ? 0.08 : CENTER_PULL_STRENGTH * forceScale) as unknown as RegisteredForce
    );
    fg.d3Force(
      'y',
      forceY<ProcessedNode>((node) => targets?.get(node.id)?.y ?? 0)
        .strength(targets ? 0.08 : CENTER_PULL_STRENGTH * forceScale) as unknown as RegisteredForce
    );

    // Add collision force to prevent overlap
    fg.d3Force(
      'collide',
      forceCollide<ProcessedNode>()
        .radius((node: ProcessedNode) => targets
          ? calculateNodeRadius(node.stargazers_count) + 1
          : calculateNodeRadius(node.stargazers_count) * 2.4 + 10)
        .strength(targets ? 0.45 : Math.max(0.2, 0.9 * forceScale))
        .iterations(targets ? 2 : forceScale < 1 ? 1 : 2) as unknown as RegisteredForce
    );

    // Register custom force
    fg.d3Force('cluster', targets ? null : createClusterForce(clusterLayoutData, forceScale));
    if (reheatedLayoutRef.current?.key !== layoutKey || reheatedLayoutRef.current.nodes !== nodes) {
      reheatedLayoutRef.current = { key: layoutKey, nodes };
      fg.d3ReheatSimulation();
    }
  }, [clusterLayoutData, enabled, forceScale, graphRef, layoutKey, nodes, targets]);
};

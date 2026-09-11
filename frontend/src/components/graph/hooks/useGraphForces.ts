import { useEffect } from 'react';
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
  (clusterLayoutData: ClusterLayoutData) =>
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
          const force = ((MIN_CLUSTER_DISTANCE - dist) / dist) * alpha * 2;
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

        const k = alpha * 0.2;
        node.vx = (node.vx || 0) + (center.x - node.x) * k;
        node.vy = (node.vy || 0) + (center.y - node.y) * k;
      });
    });
  };

/** Configure link, charge, centring, collision, and cluster forces. */
export const useGraphForces = ({
  graphRef,
  clusterLayoutData,
}: {
  graphRef: React.MutableRefObject<ForceGraphMethods | undefined>;
  clusterLayoutData: ClusterLayoutData;
}) => {
  useEffect(() => {
    if (!graphRef.current) return;

    const fg = graphRef.current;

    // Configure link force - increased distance for cross-cluster links
    fg.d3Force('link')
      ?.distance((link: LinkObject<NodeObject, LinkObject<NodeObject>>) => {
        const sourceCluster = typeof link.source === 'object' ? link.source.cluster_id : null;
        const targetCluster = typeof link.target === 'object' ? link.target.cluster_id : null;
        // Same cluster: moderate distance, Different cluster: larger separation
        return sourceCluster === targetCluster ? 38 : 75;
      })
      .strength((link: LinkObject<NodeObject, LinkObject<NodeObject>>) => {
        const sourceCluster = typeof link.source === 'object' ? link.source.cluster_id : null;
        const targetCluster = typeof link.target === 'object' ? link.target.cluster_id : null;
        // Stronger links within same cluster, very weak for cross-cluster
        return sourceCluster === targetCluster ? 0.7 : 0.05;
      });

    // Configure charge force (repulsion) - increased for more spacing
    fg.d3Force('charge')?.strength(-150).distanceMax(250);

    // Gentle pull toward the origin to avoid disconnected groups drifting far apart
    fg.d3Force('x', forceX(0).strength(CENTER_PULL_STRENGTH) as unknown as RegisteredForce);
    fg.d3Force('y', forceY(0).strength(CENTER_PULL_STRENGTH) as unknown as RegisteredForce);

    // Add collision force to prevent overlap
    fg.d3Force(
      'collide',
      forceCollide<ProcessedNode>()
        .radius((node: ProcessedNode) => calculateNodeRadius(node.stargazers_count) * 2.4 + 10)
        .strength(0.9)
        .iterations(2) as unknown as RegisteredForce
    );

    // Register custom force
    fg.d3Force('cluster', createClusterForce(clusterLayoutData));
    fg.d3ReheatSimulation();
  }, [graphRef, clusterLayoutData]);
};
